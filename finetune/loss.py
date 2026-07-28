import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
    frame_weights: torch.Tensor | None = None,
    first_codebook_frame_weights: torch.Tensor | None = None,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if frame_weights is not None:
        if frame_weights.ndim not in {2, 3}:
            raise ValueError(
                "frame weights must have shape [batch, time] or [batch, 1, time]"
            )
        if frame_weights.ndim == 2:
            frame_weights = frame_weights[:, None, :]
        if (
            frame_weights.shape[0] != weights.shape[0]
            or frame_weights.shape[1] != 1
            or frame_weights.shape[-1] != weights.shape[-1]
        ):
            raise ValueError(
                f"frame weights {tuple(frame_weights.shape)} do not match target mask "
                f"{tuple(weights.shape)}"
            )
        if not torch.all(torch.isfinite(frame_weights)) or torch.any(frame_weights < 0):
            raise ValueError("frame weights must be finite and non-negative")
        weights = weights * frame_weights
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
        if first_codebook_frame_weights is not None:
            if (
                first_codebook_frame_weights.ndim == 3
                and first_codebook_frame_weights.shape[1] == 1
            ):
                first_codebook_frame_weights = first_codebook_frame_weights[:, 0]
            if first_codebook_frame_weights.shape != weights[:, 0].shape:
                raise ValueError(
                    "first-codebook frame weights do not match the audio time axis"
                )
            if not torch.all(torch.isfinite(first_codebook_frame_weights)) or torch.any(
                first_codebook_frame_weights < 0
            ):
                raise ValueError(
                    "first-codebook frame weights must be finite and non-negative"
                )
            weights[:, 0] *= first_codebook_frame_weights
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight

    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)
    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
    weight_sum = torch.sum(weights)
    if weight_sum <= 0:
        raise ValueError(
            "effective loss weights must contain at least one positive value"
        )
    mb_loss = torch.sum(mb_loss) / weight_sum

    return mb_loss
