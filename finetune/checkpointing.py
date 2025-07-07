import json
import logging
import shutil
from pathlib import Path

import safetensors.torch
import torch
from moshi.models.lm import LMModel
from moshi.modules.lora import LoRALinear
from torch.distributed import barrier
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .distributed import get_rank, get_world_size
from .utils import TrainState

logger = logging.getLogger("checkpointing")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


class Checkpointer:
    """A class to save PyTorch model and optimizer states"""

    def __init__(
        self,
        model: FullyShardedDataParallel | LMModel,
        state: TrainState,
        run_dir: Path | str,
        config: dict,
        optimizer: torch.optim.Optimizer | None = None,
        num_ckpt_keep: int | None = None,
        full_finetuning: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.rank = get_rank()
        self.num_ckpt_keep = num_ckpt_keep
        self.full_finetuning = full_finetuning
        self.config = config

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def dst_dir(self) -> Path:
        return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "consolidated"

    @staticmethod
    def consolidated_path(ckpt_dir: Path, save_only_lora: bool = False) -> Path:
        suffix = "safetensors"
        prefix = "lora" if save_only_lora else "consolidated"

        return ckpt_dir / f"{prefix}.{suffix}"

    @staticmethod
    def _tmp(ckpt_dir: Path) -> Path:
        return ckpt_dir.with_name(f"tmp.{ckpt_dir.name}")

    def delete_old_ckpts(self) -> list[Path]:
        all_saved_ckpts = [d for d in self.ckpt_dir.iterdir() if d.is_dir()]

        # Sort directories by creation time (oldest to newest)
        all_saved_ckpts.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        ckpts_to_delete = all_saved_ckpts[self.num_ckpt_keep :]

        for ckpt_to_delete in ckpts_to_delete:
            try:
                shutil.rmtree(ckpt_to_delete)
                main_logger_info(f"Deleted ckpt: {ckpt_to_delete}")
            except OSError as e:
                main_logger_info(f"Error deleting directory {ckpt_to_delete}: {e}")

        return ckpts_to_delete

    def write_params_info(self, tmp_dst: Path):
        params_path = tmp_dst / "config.json"
        with open(params_path, "w") as f:
            f.write(json.dumps(self.config, indent=4))

    @staticmethod
    def get_non_lora_states(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in state_dict.items()
            if not any(l_key in k for l_key in ["lora", "frozen"])
        }

    @torch.no_grad()
    def retrieve_save_states(
        self, save_only_lora: bool, save_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        assert not (save_only_lora and self.full_finetuning), (
            "Cannot save LoRA checkpoint as LoRA training is not enabled."
        )

        # remove all potential hooks
        for module in self.model.modules():
            if isinstance(module, LoRALinear) and hasattr(module, "_merge_lora_handle"):
                module._merge_lora_handle.remove()  # type: ignore

        offload_to_cpu = get_world_size() > 1
        if save_only_lora:

            def is_trainable_fsdp(module: torch.nn.Module | FullyShardedDataParallel):
                is_fsdp = (
                    isinstance(module, FullyShardedDataParallel)
                    or get_world_size() == 1
                )
                all_params_have_grads = is_fsdp and all(
                    p.requires_grad for p in module.parameters()
                )

                # need to make sure only lowest fsdp wrap is used
                is_leaf_node = is_fsdp and (
                    get_world_size() == 1 or len(list(module.module.children())) == 0
                )  # type: ignore

                return is_fsdp and all_params_have_grads and is_leaf_node

            # extract all modules with only trainable weights
            modules = {
                k: m for k, m in self.model.named_modules() if is_trainable_fsdp(m)
            }

            states = {}
            for key, module in modules.items():
                assert (
                    isinstance(module, FullyShardedDataParallel)
                    or get_world_size() == 1
                ), (
                    "`module` should be an instance of `FullyShardedDataParallel` if `world_size > 1`"
                )
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                if get_world_size() > 1:
                    with module.summon_full_params(
                        module, writeback=True, offload_to_cpu=offload_to_cpu
                    ):
                        states.update(
                            {
                                f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                                for k, v in module.state_dict().items()
                            }
                        )
                else:
                    states.update(
                        {
                            f"{parent_prefix}.{k}": v.clone().to(dtype=save_dtype)
                            for k, v in module.state_dict().items()
                        }
                    )
        else:
            # merge weights if we don't just save LoRA
            def merge_lora(
                m: torch.nn.Module,
                destination: dict[str, torch.Tensor],
                prefix: str,
                *args,
            ):
                weight = m.merge_weight()  # type: ignore
                destination[prefix + "weight"] = weight

            for module in self.model.modules():
                if isinstance(module, LoRALinear):
                    module._merge_lora_handle = module._register_state_dict_hook(
                        merge_lora
                    )

            # make sure you have enough CPU RAM available to save the full model
            assert (
                isinstance(self.model, FullyShardedDataParallel)
                or get_world_size() == 1
            ), (
                "`self.model` should be an instance of `FullyShardedDataParallel` if `world_size > 1`"
            )
            if get_world_size() > 1:
                with self.model.summon_full_params(
                    self.model, writeback=True, offload_to_cpu=offload_to_cpu
                ):
                    states = self.get_non_lora_states(self.model.state_dict())
                    states = {k: v.to(dtype=save_dtype) for k, v in states.items()}
            else:
                states = self.get_non_lora_states(self.model.state_dict())
                states = {k: v.clone().to(dtype=save_dtype) for k, v in states.items()}

        states = dict(sorted(states.items()))
        return states

    @torch.no_grad()
    def save_checkpoint(
        self,
        save_only_lora: bool,
        dtype: torch.dtype = torch.float16,
    ):
        if self.full_finetuning:
            assert not save_only_lora, "Cannot save LoRA checkpoint in full finetuning"

        tmp_dst = self._tmp(self.dst_dir)
        main_logger_info(
            f"Dumping checkpoint in {self.dst_dir} using tmp name: {tmp_dst.name}"
        )

        assert not self.dst_dir.exists(), f"dst exists {self.dst_dir}"
        tmp_dst.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            states: dict[str, torch.Tensor] = self.retrieve_save_states(
                save_only_lora, dtype
            )

        barrier()

        if self.rank == 0:
            # save checkpoint in tmp path
            safetensors.torch.save_file(
                states,
                self.consolidated_path(
                    tmp_dst, save_only_lora=save_only_lora
                ),  # always use safetensors for checkpointing
            )
            self.write_params_info(tmp_dst)
            assert not self.dst_dir.exists(), f"should not happen! {self.dst_dir}"
            tmp_dst.rename(self.dst_dir)

            logger.info(
                f"Done dumping checkpoint in {self.dst_dir} for step: {self.state.step}"
            )

            # delete last n checkpoints
            if self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )

        main_logger_info("Done!")
