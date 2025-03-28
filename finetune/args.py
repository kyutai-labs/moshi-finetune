import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from simple_parsing.helpers import Serializable

from .data.args import DataArgs


@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class OptimArgs(Serializable):
    lr: float = 1e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05


@dataclass
class WandbArgs(Serializable):
    project: str | None = None  # Fill this argument to use wandb.
    offline: bool = False
    key: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.project is not None:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "`wandb` not installed. Either make sure `wandb` is installed or set `wandb:project` to None."
                )

            if len(self.project) == 0:
                raise ValueError("`wandb.project` must not be an empty string.")


@dataclass
class ModelPaths(Serializable):
    hf_repo_id: str = "kyutai/moshiko-pytorch-bf16"
    mimi_path: str = "tokenizer-e351c8d8-checkpoint125.safetensors"
    moshi_path: str = "model.safetensors"
    tokenizer_path: str = "tokenizer_spm_32k_3.model"

    def __post_init__(self) -> None:
        if self.hf_repo_id is not None:
            assert self.mimi_path is not None
            assert self.moshi_path is not None
            assert self.tokenizer_path is not None
        else:
            assert Path(self.mimi_path).exists()
            assert Path(self.moshi_path).exists()
            assert Path(self.tokenizer_path).exists()


@dataclass
class TrainArgs(Serializable):
    data: DataArgs

    run_dir: str  # Path to the directory where everything will be saved. It needs to be empty.
    # Name of the wandb run, if None it will be set to the name of the run_dir.
    moshi_paths: ModelPaths = field(default_factory=ModelPaths)
    first_codebook_weight_multiplier: float = 1.0
    text_padding_weight: float = 0.5

    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    # Number of steps to accumulate gradients before doing an optimizer step.
    num_microbatches: int = 1

    duration_sec: float = 10
    batch_size: int = 1
    max_norm: float = 1.0  # Gradient clipping.
    max_steps: int = 100  # Number of training steps.
    log_freq: int = 1  # Number of steps between each logging.

    # Number of steps between each checkpoint saving. If inferior to 1, only the last checkpoint will be saved.
    ckpt_freq: int = 0
    save_adapters: bool = True
    # If False, no checkpoints will be saved. This is useful for development.
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 3
    eval_freq: int = 0
    do_eval: bool = False

    # Efficiency
    # Determines whether gradient checkpointing should be utilized or not
    # during the training process. Gradient checkpointing can be beneficial in
    # reducing memory usage at the cost of slightly longer training times.
    gradient_checkpointing: bool = True

    world_size: int | None = field(init=False, default=None)

    # logging
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA
    lora: LoraArgs | None = field(default_factory=LoraArgs)
    full_finetuning: bool = False

    param_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        if not self.save_adapters:
            logging.warning(
                "You have disabled `save_adapters` and are thus merging the "
                "trained LoRA checkpoint into the base model upon checkpointing. "
                "This might lead to OOM errors - make sure you have enough CPU "
                "and GPU memory."
            )
