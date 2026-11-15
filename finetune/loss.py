import torch
from torch.nn import functional as F


def loss_sum_and_weight(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight

    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)
    # Exclude inactive positions before CE: delayed logits may contain NaNs.
    # An empty selection still has a differentiable zero sum.
    active = weights > 0
    losses = F.cross_entropy(logits[active], target[active], reduction="none")
    return (losses * weights[active]).sum(), weights[active].sum()


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
):
    total, weight = loss_sum_and_weight(
        logits, target, target_mask, mode,
        first_codebook_weight_multiplier, text_padding_weight, text_padding_ids,
    )
    denominator = torch.where(weight > 0, weight, torch.ones_like(weight))
    return total / denominator
