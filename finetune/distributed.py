import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Callable, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("distributed")

BACKEND = "nccl"


@lru_cache()
def get_rank() -> int:
    return dist.get_rank()


@lru_cache()
def get_world_size() -> int:
    return dist.get_world_size()


def visible_devices() -> list[str]:
    """Return CUDA visibility identifiers, which may be UUIDs rather than ints."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return [str(i) for i in range(torch.cuda.device_count())]
    return [device.strip() for device in value.split(",") if device.strip()]


def set_device():
    count = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    logger.info(f"torch.cuda.device_count: {count}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    logger.info(f"local rank: {local_rank}")
    if not torch.cuda.is_available() or count == 0:
        raise RuntimeError("No CUDA device is available.")
    # CUDA already applies the visibility mapping. A one-GPU-per-task launcher
    # exposes logical device 0 even when LOCAL_RANK is nonzero.
    device = 0 if count == 1 else local_rank
    if not 0 <= device < count:
        raise ValueError(f"LOCAL_RANK {local_rank} is outside {count} visible CUDA devices.")
    torch.cuda.set_device(device)


def prepare_run_dir(
    run_dir: Path,
    overwrite: bool,
    *,
    check_existing: bool = True,
    save_args: Callable[[Path], None] | None = None,
) -> None:
    """Initialize on rank 0 and propagate filesystem errors to every rank."""
    distributed = dist.is_initialized()
    rank = get_rank() if distributed else 0
    status: list[str | None] = [None]
    if rank == 0:
        try:
            if check_existing and run_dir.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Run dir {run_dir} already exists; choose another "
                        "or enable overwrite_run_dir."
                    )
                shutil.rmtree(run_dir)
            run_dir.mkdir(exist_ok=True, parents=True)
            args_path = run_dir / "args.yaml"
            if save_args is not None and not args_path.exists():
                save_args(args_path)
        except Exception as error:
            status[0] = f"Cannot initialize run dir {run_dir}: {type(error).__name__}: {error}"
    if distributed:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ
