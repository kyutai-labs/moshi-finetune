"""No ASR models/GPU: real tokenizer code with an explicit frozen codec fixture."""

import argparse
import ast
import gzip
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import finetune.data.interleaver as interleaving

SOURCE = Path(interleaving.__file__).parents[2]


class TextTokenizer:
    def encode(self, text, **kwargs):
        return [[12] for _ in text] if isinstance(text, list) else [12]

    def bos_id(self):
        return 1

    def eos_id(self):
        return 2


class FrozenCodec:
    frame_rate = 12.5

    def __init__(self, frames):
        self.frames = frames

    def encode(self, audio):
        return torch.arange(1, 17).view(2, 8, 1).expand(2, 8, self.frames)


def tokenizer(frames):
    interleaver = interleaving.Interleaver(
        TextTokenizer(), 12.5, 3, 0, -1, device="cpu"
    )
    return interleaving.InterleavedTokenizer(FrozenCodec(frames), interleaver, 100)


@pytest.mark.parametrize("frames", [7, 25, 300, 1250])
def test_valid_text_frames_match_audio_exactly(tmp_path, monkeypatch, frames):
    path = tmp_path / "audio.wav"
    path.with_suffix(".json").write_text(json.dumps({"alignments": []}))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    sample = tokenizer(frames)(np.zeros((2, 24000), dtype=np.float32), 0, str(path))
    assert sample.codes.shape == (1, 17, 1250)
    assert torch.count_nonzero(sample.codes[:, 0] != -1) == frames
    assert torch.equal(sample.codes[0, 1:, 0], torch.arange(1, 17))
    assert (sample.codes[..., frames:] == -1).all()


def test_negative_audio_delay_cannot_extend_short_text_stream():
    interleaver = tokenizer(7).interleaver
    interleaver.audio_delay = -1
    stream = interleaver.prepare_item([], 7 / 12.5, num_audio_frames=7)
    assert stream.shape == (1, 1, 7)
    assert (stream == -1).all()


@pytest.mark.parametrize("conditions", [{}, {"description": "voice"}])
def test_sidecar_conditions_fail_before_cuda(tmp_path, monkeypatch, conditions):
    path = tmp_path / "audio.wav"
    path.with_suffix(".json").write_text(
        json.dumps({"alignments": [], "text_conditions": conditions})
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda _: pytest.fail("CUDA called"))
    with pytest.raises(ValueError, match="text_conditions.*not supported"):
        tokenizer(25)(np.zeros((2, 24000)), 0, str(path))


def annotation_functions():
    # Avoid importing Whisper/SLURM: compile the actual standalone functions and
    # CLI, with only the dispatch endpoints replaced. No GPU/cluster work runs.
    tree = ast.parse((SOURCE / "annotate.py").read_text())
    keep = {"load_audio_paths", "parse_silence_seconds", "Params", "main"}
    body = [n for n in tree.body if getattr(n, "name", None) in keep]
    called = []
    namespace = {
        "argparse": argparse,
        "gzip": gzip,
        "json": json,
        "math": math,
        "Path": Path,
        "dataclass": dataclass,
        "logger": SimpleNamespace(warning=lambda _: None),
        "init_logging": lambda _: None,
        "run": lambda params: called.append(params),
        "submitit": SimpleNamespace(
            SlurmExecutor=lambda **_: pytest.fail("SLURM used")
        ),
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "annotate.py", "exec"),
        namespace,
    )
    return namespace, called


@pytest.mark.parametrize("compressed", [False, True])
def test_manifest_relative_paths_ignore_cwd_shadow(tmp_path, monkeypatch, compressed):
    data = tmp_path / "data"
    data.mkdir()
    expected = data / "audio.wav"
    expected.write_bytes(b"manifest audio")
    (tmp_path / "audio.wav").write_bytes(b"wrong cwd audio")
    absolute = tmp_path / "absolute.wav"
    absolute.write_bytes(b"absolute audio")
    manifest = data / ("samples.jsonl.gz" if compressed else "samples.jsonl")
    contents = "\n".join(json.dumps({"path": p}) for p in ("audio.wav", str(absolute)))
    if compressed:
        manifest.write_bytes(gzip.compress(contents.encode()))
    else:
        manifest.write_text(contents)
    monkeypatch.chdir(tmp_path)
    namespace, _ = annotation_functions()
    paths = namespace["load_audio_paths"](manifest)
    assert paths[0].samefile(expected)
    assert paths[1] == absolute


@pytest.mark.parametrize(
    ("argument", "expected"), [(None, 1.0), ("0", 0.0), ("0.25", 0.25)]
)
def test_cli_silence_is_seconds_and_local_dispatch(monkeypatch, argument, expected):
    argv = ["annotate.py", "data.jsonl", "--local"]
    if argument is not None:
        argv += ["--keep_silence_in_segments", argument]
    monkeypatch.setattr(sys, "argv", argv)
    namespace, called = annotation_functions()
    namespace["main"]()
    assert len(called) == 1
    assert called[0].keep_silence_in_segments == expected
    assert type(called[0].keep_silence_in_segments) is float


@pytest.mark.parametrize("argument", ["False", "True", "-1", "nan", "inf"])
def test_cli_rejects_invalid_silence_duration(monkeypatch, argument):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate.py",
            "data.jsonl",
            "--local",
            "--keep_silence_in_segments",
            argument,
        ],
    )
    namespace, _ = annotation_functions()
    with pytest.raises(SystemExit) as error:
        namespace["main"]()
    assert error.value.code == 2
