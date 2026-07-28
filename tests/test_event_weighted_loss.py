from types import SimpleNamespace

import pytest
import torch

from finetune.data.interleaver import Batch, InterleavedTokenizer, Sample
from finetune.loss import compute_loss_with_mask


def _audio_example():
    logits = torch.tensor(
        [
            [
                [[3.0, 0.0], [0.0, 3.0]],
                [[3.0, 0.0], [0.0, 3.0]],
            ]
        ]
    )
    target = torch.zeros((1, 2, 2), dtype=torch.long)
    mask = torch.ones_like(target, dtype=torch.bool)
    return logits, target, mask


def test_unit_frame_weights_preserve_the_existing_loss():
    logits, target, mask = _audio_example()

    baseline = compute_loss_with_mask(logits, target, mask, mode="audio")
    weighted = compute_loss_with_mask(
        logits,
        target,
        mask,
        mode="audio",
        frame_weights=torch.ones((1, 2)),
        first_codebook_frame_weights=torch.ones((1, 2)),
    )

    assert weighted == pytest.approx(baseline)


def test_frame_and_first_codebook_weights_select_requested_targets():
    logits, target, mask = _audio_example()

    first_frame = compute_loss_with_mask(
        logits,
        target,
        mask,
        mode="audio",
        frame_weights=torch.tensor([[1.0, 0.0]]),
    )
    first_codebook = compute_loss_with_mask(
        logits,
        target,
        mask,
        mode="audio",
        first_codebook_frame_weights=torch.tensor([[1.0, 0.0]]),
    )

    assert first_frame < first_codebook


@pytest.mark.parametrize(
    "frame_weights",
    [
        torch.ones((1, 2, 2)),
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[1.0, -1.0]]),
        torch.zeros((1, 2)),
    ],
)
def test_invalid_frame_weights_fail_explicitly(frame_weights):
    logits, target, mask = _audio_example()

    with pytest.raises(ValueError, match="weights"):
        compute_loss_with_mask(
            logits,
            target,
            mask,
            mode="audio",
            frame_weights=frame_weights,
        )


def test_sidecar_regions_are_clipped_to_the_current_chunk_and_later_regions_win():
    tokenizer = InterleavedTokenizer.__new__(InterleavedTokenizer)
    tokenizer.duration_sec = 2.0
    tokenizer.num_audio_frames = 20
    tokenizer.mimi = SimpleNamespace(frame_rate=10.0)

    weights = tokenizer._frame_weights(
        [
            {"start_s": 2.5, "end_s": 3.5, "text_weight": 2.0},
            {"start_s": 3.0, "end_s": 4.5, "text_weight": 3.0},
        ],
        start_sec=2.0,
        path="fixture.wav",
        device=torch.device("cpu"),
    )

    assert torch.all(weights["text"][:, :5] == 1.0)
    assert torch.all(weights["text"][:, 5:10] == 2.0)
    assert torch.all(weights["text"][:, 10:] == 3.0)
    assert torch.all(weights["audio"] == 1.0)


def test_batch_collates_event_weights_with_codes():
    first = Sample(
        codes=torch.zeros((1, 3, 2)),
        text_frame_weights=torch.ones((1, 2)),
        audio_frame_weights=torch.ones((1, 2)),
        first_codebook_frame_weights=torch.ones((1, 2)),
    )
    second = Sample(
        codes=torch.ones((1, 3, 2)),
        text_frame_weights=torch.full((1, 2), 2.0),
        audio_frame_weights=torch.full((1, 2), 3.0),
        first_codebook_frame_weights=torch.full((1, 2), 4.0),
    )

    batch = Batch.collate([first, second])

    assert batch.codes.shape == (2, 3, 2)
    assert batch.text_frame_weights.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert batch.audio_frame_weights.tolist() == [[1.0, 1.0], [3.0, 3.0]]
    assert batch.first_codebook_frame_weights.tolist() == [
        [1.0, 1.0],
        [4.0, 4.0],
    ]
