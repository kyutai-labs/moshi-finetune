import logging
from typing import Iterator

import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from finetune.args import TrainArgs

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import loss_sum_and_weight
from .utils import TrainState

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    eval_data_loader: Iterator[Batch],
    state: TrainState,
    args: TrainArgs,
):
    """Evaluate a fresh iterator, with matched FSDP forward counts on all ranks.

    Evaluation uses the common prefix of rank-local batches and includes partial
    final batches. A rank with no batches makes the whole evaluation fail clearly.
    """
    device = next(model.parameters()).device
    distributed = dist.is_initialized()
    world_size = get_world_size() if distributed else 1
    # text NLL sum/weight, audio NLL sum/weight, examples, batches
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    try:
        iterator = iter(eval_data_loader)
        for _ in range(max(1, 40 // world_size)):
            error = None
            try:
                batch = next(iterator, None)
            except Exception as exception:
                batch = None
                error = exception
            available = torch.tensor(-1 if error is not None else int(batch is not None), device=device)
            if distributed:
                dist.all_reduce(available, op=dist.ReduceOp.MIN)
            if available.item() < 0:
                raise RuntimeError("Evaluation data loading failed on a rank.") from error
            if available.item() == 0:
                if batch is not None:
                    logger.warning("Stopping evaluation at the shortest rank; a local batch was not evaluated.")
                break
            codes = batch.codes
            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            with torch.no_grad():
                output = model(codes=codes, condition_tensors=condition_tensors)
                text_sum, text_weight = loss_sum_and_weight(
                    output.text_logits, codes[:, : model.audio_offset], output.text_mask,
                    mode="text", text_padding_weight=args.text_padding_weight,
                    text_padding_ids={model.text_padding_token_id, model.end_of_text_padding_id},
                )
                audio_sum, audio_weight = loss_sum_and_weight(
                    output.logits,
                    codes[:, model.audio_offset : model.audio_offset + model.dep_q], output.mask,
                    mode="audio", first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                )
                totals[:4] += torch.stack((text_sum, text_weight, audio_sum, audio_weight)).double()
                totals[4] += codes.shape[0]
                totals[5] += 1
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[5].item() == 0:
            raise ValueError("Evaluation has no batches shared by all ranks; provide data on every rank.")
        if (totals[1] + totals[3]).item() == 0:
            raise ValueError("Evaluation has no positively weighted targets.")
        text_mean = totals[0] / totals[1] if totals[1].item() > 0 else None
        audio_mean = totals[2] / totals[3] if totals[3].item() > 0 else None
        state.this_text_loss = text_mean.item() if text_mean is not None else None
        state.this_audio_loss = audio_mean.item() if audio_mean is not None else None
        state.this_eval_loss = (state.this_text_loss or 0) + (state.this_audio_loss or 0)
        state.this_text_perplexity = text_mean.exp().item() if text_mean is not None else None
        state.this_audio_perplexity = audio_mean.exp().item() if audio_mean is not None else None
        state.this_eval_perplexity = ((totals[0] + totals[2]) / (totals[1] + totals[3])).exp().item()
        state.this_eval_examples = int(totals[4].item())
        state.this_eval_batches = int(totals[5].item())
        logger.info("Eval finished: %s examples in %s rank-batches", state.this_eval_examples, state.this_eval_batches)
    finally:
        model.train(was_training)
