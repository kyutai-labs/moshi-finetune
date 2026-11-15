import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from finetune.args import TrainArgs, WandbArgs
from finetune.utils import TrainState

logger = logging.getLogger("metrics_logger")

GB = 1024**3


def get_train_logs(
    state: TrainState,
    loss: float,
    num_real_tokens: int,
    lr: float,
    peak_allocated_mem: float,
    allocated_mem: float,
    train_args: TrainArgs,
) -> dict[str, float | int]:
    metrics = {
        "lr": lr,
        "step": state.step,
        "loss": loss,
        "prob_real_tokens": num_real_tokens / state.this_step_tokens,
        "percent_done": 100 * state.step / train_args.max_steps,
        "peak_allocated_mem": peak_allocated_mem / GB,
        "allocated_mem": allocated_mem / GB,
        "wps": state.wps,
        "avg_wps": state.avg_wps,
        "eta_in_seconds": state.eta,
    }

    return metrics


def get_eval_logs(
    step: int,
    train_loss: float,
    perplexity: float | None = None,
    eval_loss: float | None = None,
    text_eval_loss: float | None = None,
    audio_eval_loss: float | None = None,
    text_perplexity: float | None = None,
    audio_perplexity: float | None = None,
    eval_examples: int | None = None,
    eval_batches: int | None = None,
) -> dict[str, float | int]:
    eval_dict = {"step": step, "train_loss": train_loss}

    if perplexity is not None:
        eval_dict["perplexity"] = perplexity

    if eval_loss is not None:
        eval_dict["eval_loss"] = eval_loss

    if text_eval_loss is not None:
        eval_dict["text_eval_loss"] = text_eval_loss

    if audio_eval_loss is not None:
        eval_dict["audio_eval_loss"] = audio_eval_loss

    for key, value in (
        ("text_perplexity", text_perplexity),
        ("audio_perplexity", audio_perplexity),
        ("eval_examples", eval_examples),
        ("eval_batches", eval_batches),
    ):
        if value is not None:
            eval_dict[key] = value
    return eval_dict


def train_log_msg(state: TrainState, logs: dict[str, float | int], loss: float) -> str:
    metrics: dict[str, float | int | datetime] = dict(logs)  # shallow copy
    metrics.pop("eta_in_seconds")

    metrics["eta"] = datetime.now() + timedelta(seconds=state.eta)
    metrics["step"] = state.step
    metrics["loss"] = loss

    parts = []
    for key, fmt, new_name in [
        ("step", "06", None),
        ("percent_done", "03.1f", "done (%)"),
        ("loss", ".3f", None),
        ("lr", ".1e", None),
        ("peak_allocated_mem", ".1f", "peak_alloc_mem (GB)"),
        ("allocated_mem", ".1f", "alloc_mem (GB)"),
        ("wps", ".1f", "words_per_second"),
        ("avg_wps", ".1f", "avg_words_per_second"),
        ("eta", "%Y-%m-%d %H:%M:%S", "ETA"),
    ]:
        name = key if new_name is None else new_name
        try:
            parts.append(f"{name}: {metrics[key]:>{fmt}}")
        except KeyError:
            logger.error(f"{key} not found in {sorted(metrics.keys())}")
            raise

    return " - ".join(parts)


def eval_log_msg(logs: dict[str, float | int]) -> str:
    parts = []
    for key, fmt, new_name in [
        ("step", "06", None),
        ("perplexity", ".3f", "eval_perplexity"),
        ("eval_loss", ".3f", None),
        ("train_loss", ".3f", None),
        ("text_eval_loss", ".3f", None),
        ("audio_eval_loss", ".3f", None),
        ("text_perplexity", ".3f", None),
        ("audio_perplexity", ".3f", None),
        ("eval_examples", "d", None),
        ("eval_batches", "d", None),
    ]:
        name = key if new_name is None else new_name
        if key in logs:
            parts.append(f"{name}: {logs[key]:>{fmt}}")

    return " - ".join(parts)


class MetricsLogger:
    def __init__(
        self,
        dst_dir: Path,
        tag: str,
        is_master: bool,
        wandb_args: WandbArgs,
        config: dict[str, Any] | None = None,
    ):
        self.dst_dir = dst_dir
        self.tag = tag
        self.is_master = is_master
        self.jsonl_path = dst_dir / f"metrics.{tag}.jsonl"
        self.tb_dir = dst_dir / "tb"
        self.summary_writer: SummaryWriter | None = None

        if not self.is_master:
            return

        filename_suffix = f".{tag}"
        self.tb_dir.mkdir(exist_ok=True)
        self.summary_writer = SummaryWriter(
            log_dir=str(self.tb_dir),
            max_queue=1000,
            filename_suffix=filename_suffix,
        )
        self.is_wandb = wandb_args.project is not None

        if self.is_wandb:
            import wandb

            if wandb_args.key is not None:
                wandb.login(key=wandb_args.key)
            if wandb_args.offline:
                os.environ["WANDB_MODE"] = "offline"
            if wandb.run is None:
                logger.info("initializing wandb")
                wandb.init(
                    config=config,
                    dir=dst_dir,
                    project=wandb_args.project,
                    job_type="training",
                    name=wandb_args.run_name or dst_dir.name,
                    resume=False,
                )

            self.wandb_log = wandb.log

    def log(self, metrics: dict[str, float | int], step: int):
        if not self.is_master:
            return

        metrics_to_ignore = {"step"}
        assert self.summary_writer is not None
        for key, value in metrics.items():
            if key in metrics_to_ignore:
                continue
            assert isinstance(value, (int, float)), (key, value)
            self.summary_writer.add_scalar(
                tag=f"{self.tag}.{key}", scalar_value=value, global_step=step
            )

        if self.is_wandb:
            # grouping in wandb is done with /
            self.wandb_log(
                {
                    f"{self.tag}/{key}": value
                    for key, value in metrics.items()
                    if key not in metrics_to_ignore
                },
                step=step,
            )

        metrics_: dict[str, Any] = dict(metrics)  # shallow copy
        if "step" in metrics_:
            assert step == metrics_["step"]
        else:
            metrics_["step"] = step
        metrics_["at"] = datetime.utcnow().isoformat()
        with self.jsonl_path.open("a") as fp:
            fp.write(f"{json.dumps(metrics_)}\n")

    def close(self):
        if not self.is_master:
            return

        if self.summary_writer is not None:
            self.summary_writer.close()
            self.summary_writer = None

        if self.is_wandb:
            import wandb

            # to be sure we are not hanging while finishing
            wandb.finish()

    def __del__(self):
        if self.summary_writer is not None:
            raise RuntimeError(
                "MetricsLogger not closed properly! You should "
                "make sure the close() method is called!"
            )
