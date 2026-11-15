import torch
from torch.distributed.fsdp import FullyShardedDataParallel


def clip_grad_norm_(model: torch.nn.Module, max_norm: float) -> torch.Tensor:
    """Use the global norm across FSDP shards; all ranks must call this."""
    if isinstance(model, FullyShardedDataParallel):
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
