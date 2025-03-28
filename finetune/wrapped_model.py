import functools
import logging
import math
from pathlib import Path
from typing import Callable, Union

import safetensors
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi.models.lm import LMModel
from moshi.models.loaders import _is_safetensors
from moshi.modules.transformer import StreamingTransformerLayer
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .args import TrainArgs
from .distributed import get_rank, get_world_size

logger = logging.getLogger(__name__)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def get_fsdp_policy(is_lora: bool) -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes its own FSDP group so that only a single
      Transformer block is sharded at a time
    - If LoRA is enabled, we additionally create separate FSDP sub-groups for
      every trainable and non-trainable parameter group since this is a
      requirement for mixed requires_grad=True/False training. See:
      https://pytorch.org/docs/stable/fsdp.html
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(StreamingTransformerLayer,),
    )

    if not is_lora:
        return transformer_block_wrap_policy

    def fsdp_lora_policy_fn(module):
        return all(p.requires_grad for p in module.parameters())

    # For LoRA training, trainable and non-trainable parameters need to be put into
    # different FSDP groups
    fsdp_lora_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=fsdp_lora_policy_fn
    )

    policies = [fsdp_lora_policy, transformer_block_wrap_policy]

    return functools.partial(torch_wrap._or_policy, policies=policies)


def log_train_params(model: Union[torch.nn.Module, FullyShardedDataParallel]):
    world_size = get_world_size()

    num_params = world_size * sum(p.numel() for p in model.parameters())
    num_train_params = world_size * sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned "
        f"({num_train_params / num_params * 100:.2f}%)."
    )


def initialize_lora_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
    Initialize LoRA layers with Kaiming uniform and zeros.
    See original paper for more info: https://arxiv.org/abs/2106.09685 and
    original github repo:
    https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L122
    """
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]

                if m_name.split(".")[-1] == "lora_A":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "lora_B":
                    torch.nn.init.zeros_(param)
                else:
                    raise ValueError("Only Lora layers should be randomly initialized.")


def get_fsdp_model(args: TrainArgs) -> FullyShardedDataParallel | LMModel:
    """
    Initializes and returns a FullyShardedDataParallel (FSDP) LMModel or a non sharded LMModel if one GPU available.
    Args:
        args (TrainArgs): A configuration object containing training arguments
            and settings. Key attributes include:
            - param_dtype: The data type for model parameters (e.g., "bfloat16", "float32").
            - gradient_checkpointing: Whether to enable gradient checkpointing.
            - lora: Configuration for LoRA fine-tuning, including enabling, rank, and scaling.
            - moshi_paths: Paths to model weights, including HF repository ID and local paths.
            - full_finetuning: Whether to enable full model fine-tuning or only LoRA fine-tuning.
    Notes:
        - The function uses meta-device initialization for memory efficiency.
        - Then parameters are initialized on the first GPU (rank=0) only.
    """

    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32

    with torch.device("meta"):
        model = loaders.get_moshi_lm(
            filename=None,
            device="meta",
            dtype=param_dtype,
            lm_kwargs_overrides={
                "gradient_checkpointing": args.gradient_checkpointing,
                "lora": args.lora.enable,
                "lora_rank": args.lora.rank,
                "lora_scaling": args.lora.scaling,
            },
        )

    if get_rank() == 0:
        if args.moshi_paths.hf_repo_id is not None:
            moshi_weight = hf_hub_download(
                args.moshi_paths.hf_repo_id, args.moshi_paths.moshi_path
            )

        else:
            moshi_weight = Path(args.moshi_paths.moshi_path)

        assert _is_safetensors(moshi_weight), "Model is not safetensors"
        model_state_dict = safetensors.torch.load_file(moshi_weight)

        logger.info(f"Converting model to dtype {param_dtype} ...")

        for k, v in model_state_dict.items():
            model_state_dict[k] = v.to(param_dtype)

        model.load_state_dict(model_state_dict, assign=True)

        if args.lora.enable and not args.full_finetuning:
            logger.info("Initializing lora layers ...")
            # initialize LoRA layers
            initialize_lora_parameters(model, param_dtype)

        assert not any(p.is_meta for p in model.parameters()), (
            "All parameters should be initialized by now"
        )
        assert all(p.dtype == param_dtype for p in model.parameters()), (
            f"All parameters should be on {param_dtype}"
        )

        logger.info("Finished initialization!")
        param_init_fn = None
    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(p.is_meta for p in model.parameters()), (
            "All parameters should be on meta"
        )

    torch.distributed.barrier()

    # only finetune LoRA parameters and freeze before wrapping
    if args.lora.enable and not args.full_finetuning:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    if get_world_size() == 1:
        return model.cuda()

    auto_wrap_policy = get_fsdp_policy(args.lora.enable)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    wrapped_model = FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=param_init_fn,
        use_orig_params=True,
    )

    main_logger_info("Model sharded!")

    log_train_params(wrapped_model)

    return wrapped_model
