import argparse
import gc
import gzip
import importlib
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import sphn
import submitit
import torch
import torchaudio.functional as F
import whisper_timestamped as whisper

transcribe = importlib.import_module("whisper_timestamped.transcribe")
old_get_vad_segments = transcribe.get_vad_segments
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


# From https://github.com/facebookresearch/flashy/blob/3881ba496437cbbc34aea21c6ea42453d298e006/flashy/utils.py#L41
@contextmanager
def write_and_rename(path: Path, mode: str = "wb", suffix: str = ".tmp", pid=False):
    """
    Write to a temporary file with the given suffix, then rename it
    to the right filename. As renaming a file is usually much faster
    than writing it, this removes (or highly limits as far as I understand NFS)
    the likelihood of leaving a half-written checkpoint behind, if killed
    at the wrong time.
    """
    tmp_path = str(path) + suffix
    if pid:
        tmp_path += f".{os.getpid()}"
    with open(tmp_path, mode) as f:
        yield f
    os.rename(tmp_path, path)


def load_audio_paths(egs_path: Path) -> list[Path]:
    """Load audio paths from a JSONL egs file.

    Args:
        egs_path (Path): Path to JSONL file, might be gzipped.
    Returns:
        list of paths
    """
    open_fn = gzip.open if str(egs_path).lower().endswith(".gz") else open
    with open_fn(egs_path, "rb") as fp:  # type: ignore
        lines = fp.readlines()
    paths: list[Path] = []
    for line in lines:
        d = json.loads(line)
        audio_path = Path(d["path"])
        if not audio_path.is_absolute():
            audio_path = egs_path.resolve().parent / audio_path
        paths.append(audio_path)
    return paths


def init_logging(verbose: bool = False):
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


def process_one(
    in_file: Path,
    out_file: Path,
    language: str,
    w_model,
    params: "Params",
    channel: int = 0,
    seek_time: float | None = None,
    duration: float | None = None,
):
    logger.debug("Loading audio %s", in_file)
    gc.collect()
    torch.cuda.empty_cache()
    x, sr = sphn.read(in_file, start_sec=seek_time, duration_sec=duration)
    x = torch.from_numpy(x).cuda()
    dur = x.shape[-1] / sr
    if dur > 3600 * 4:
        raise RuntimeError("File is too long for now.")
    vocals = x[channel][None]
    vocals = F.resample(vocals, sr, SAMPLE_RATE)
    sr = SAMPLE_RATE

    def new_get_vad_segments(*args, **kwargs):
        segs = old_get_vad_segments(*args, **kwargs)
        outs = []
        last_end = 0
        d = int(SAMPLE_RATE * params.keep_silence_in_segments)
        logger.debug("Reintroducing %d samples at the boundaries of the segments.", d)
        for seg in segs:
            outs.append(
                {"start": max(last_end, seg["start"] - d), "end": seg["end"] + d}
            )
            last_end = outs[-1]["end"]
        return outs

    if params.keep_silence_in_segments:
        transcribe.get_vad_segments = new_get_vad_segments  # type: ignore

    vocals = vocals.cpu()
    vocals = vocals.numpy()[0]
    chunks = []
    this_duration = vocals.shape[-1] / sr
    logger.debug("Transcribing block in %s, of duration %.1f", language, this_duration)
    pipe_output = whisper.transcribe(
        w_model,
        vocals,
        language=language,
        vad="auditok" if this_duration > 10 else None,
        best_of=5,
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    for segment in pipe_output["segments"]:
        if "words" not in segment:
            logger.error("No words in %s: %r", in_file, segment)
            continue
        for word in segment["words"]:
            try:
                chunks.append(
                    {"text": word["text"], "timestamp": (word["start"], word["end"])}
                )
            except KeyError:
                logger.error("Missing key in %s: %r", in_file, word)
                raise
    outputs = {
        "alignments": [
            [chunk["text"], chunk["timestamp"], "SPEAKER_MAIN"] for chunk in chunks
        ]
    }
    logger.debug("Whisper applied.")
    with write_and_rename(out_file, "w", pid=True) as f:
        json.dump(outputs, f, ensure_ascii=False)
    logger.debug("Wrote file %s", out_file)


def run(params: "Params", shard: int = 0):
    init_logging(params.verbose)
    # local_rank = dora.distrib.get_distrib_spec().local_rank
    # shard += local_rank
    local_rank = 0
    logger.info("Hello, world, this is shard %d / %d.", shard, params.shards)
    params.shard = shard
    torch.cuda.set_device(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["OMP_NUM_THREADS"] = "2"

    logger.info("Loading all models.")
    device = torch.device(f"cuda:{local_rank}")
    w_model = whisper.load_model(params.whisper_model, device=device)

    logger.info("Loading egs %s.", params.egs)
    paths = load_audio_paths(params.egs)
    kept_paths = paths[shard :: params.shards]
    logger.info("Processing % 8d files out of % 8d.", len(kept_paths), len(paths))
    del paths

    for idx, path in enumerate(kept_paths):
        if (idx + 1) % 100 == 0:
            logger.info("Processing % 8d / % 8d files.", idx + 1, len(kept_paths))
        out_file = path.with_suffix(".json")
        err_file = path.with_suffix(".json.err")
        if out_file.exists():
            continue
        if err_file.exists() and not params.rerun_errors:
            continue
        try:
            if path.stat().st_size < 1000:
                logger.warning("Small file detected: %s", path)
                continue
            logger.debug("Processing file %s, out file is %s", path, out_file)
            process_one(
                path,
                out_file,
                channel=0,
                language=params.lang,
                w_model=w_model,
                params=params,
            )
        except Exception as err:
            if "cuda" in repr(err).lower():
                raise
            logger.exception("Error processing %s", path)
            err_file.touch()
            continue


@dataclass
class Params:
    egs: Path
    verbose: bool
    lang: str
    whisper_model: str
    keep_silence_in_segments: float
    rerun_errors: bool
    shards: int
    shard: int = 0


def parse_silence_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("Silence duration must be finite and non-negative.")
    return seconds


def main():
    parser = argparse.ArgumentParser(
        description="Annotate with transcripts and diarization."
    )
    parser.add_argument("egs", type=Path, help="Path to egs jsonl.gz file")
    parser.add_argument(
        "--log_folder", type=Path, default=Path.home() / "tmp" / "mass_annotate"
    )
    parser.add_argument(
        "-S", "--shards", type=int, default=1, help="Number of shards to schedule."
    )
    parser.add_argument("--lang", default="en", help="Force the language.")
    parser.add_argument("--partition", default="", help="Which partition to use.")
    parser.add_argument(
        "--whisper_model",
        default="medium",
        help="Which whisper to use, use medium for stereo!",
    )
    parser.add_argument(
        "--rerun_errors",
        action="store_true",
        help="Ignore previous errors and rerun failed files.",
    )
    parser.add_argument(
        "--keep_silence_in_segments",
        type=parse_silence_seconds,
        default=1.0,
        help="Seconds of silence to retain at segment boundaries (default: 1; "
        "0 disables retention). This can mitigate words being misplaced.",
    )
    parser.add_argument(
        "-l", "--local", action="store_true", help="Run locally to debug."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")

    args = parser.parse_args()

    if args.whisper_model == "large-v3":
        logger.warning(
            "You probably want to use medium whisper for stereo with VAD detection."
        )
    init_logging(args.verbose)
    kwargs = dict(args.__dict__)
    kwargs.pop("local")
    kwargs.pop("partition")
    kwargs.pop("log_folder")
    params = Params(**kwargs)

    if args.local:
        params.shards = 1
        run(params)
    else:
        executor = submitit.SlurmExecutor(folder=args.log_folder)
        executor.update_parameters(
            cpus_per_task=6,
            ntasks_per_node=1,
            gpus_per_node=1,
            time=60 * 24 * 10,
            signal_delay_s=30,
            partition=args.partition,
            stderr_to_stdout=True,
            array_parallelism=1000,
            exclude="",
            job_name="annotate",
        )
        jobs = []
        with executor.batch():
            for shard in range(args.shards):
                jobs.append(executor.submit(run, params, shard))
        print("Job id:", jobs[0].job_id)
        while True:
            done = 0
            for job in jobs:
                if job.done():
                    done += 1
            print(f"{done:04d} / {len(jobs):04d} jobs done.", end="\r")
            if done == len(jobs):
                break
            time.sleep(10.0)


if __name__ == "__main__":
    main()
