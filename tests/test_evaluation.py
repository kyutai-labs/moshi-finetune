"""Analytic targets, real evaluation/loader/loss, and real two-rank FSDP/Gloo."""

import ast
import datetime
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import finetune.eval as evaluation
from finetune.data import data_loader
from finetune.data.interleaver import Batch, Sample
from finetune.loss import compute_loss_with_mask
from finetune.utils import TrainState

SOURCE = Path(evaluation.__file__).parents[1]


class PredictableModel(torch.nn.Module):
    audio_offset = 1
    dep_q = 1
    text_padding_token_id = 3
    end_of_text_padding_id = 4

    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([math.log(0.8), math.log(0.2)]))
        self.forwarded = []

    def forward(self, codes, condition_tensors=None):
        self.forwarded.extend(codes[:, 2, 0].tolist())
        shape = (*codes[:, :1].shape, 2)
        return SimpleNamespace(
            text_logits=self.bias.reshape(1, 1, 1, 2).expand(shape),
            logits=(self.bias * 0).reshape(1, 1, 1, 2).expand(shape),
            text_mask=torch.ones_like(codes[:, :1], dtype=torch.bool),
            mask=torch.ones_like(codes[:, 1:2], dtype=torch.bool),
        )


def batch(index=0, size=1, text_target=0):
    codes = torch.zeros(size, 3, 1, dtype=torch.long)
    codes[:, 0] = text_target
    codes[:, 2] = index  # A non-target stream used only to track consumption.
    return Batch(codes)


def arguments():
    return SimpleNamespace(
        text_padding_weight=1.0,
        first_codebook_weight_multiplier=1.0,
        do_eval=True,
        data=None,
        batch_size=1,
    )


@pytest.fixture(autouse=True)
def cpu_single_rank_fixture(monkeypatch):
    # Also permits these same assertions to expose the old CPU-incompatible
    # implementation. Spawned Gloo workers import fresh modules without patches.
    original_tensor = torch.tensor

    def tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda value: value)
    monkeypatch.setattr(evaluation, "get_world_size", lambda: 1)
    monkeypatch.setattr(evaluation, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        dist, "all_gather", lambda outputs, value: outputs[0].copy_(value)
    )


@pytest.mark.parametrize("count", [1, 39, 40, 41, 50, 80])
def test_batch_limit_has_no_extra_consumption_or_denominator(count):
    consumed = []

    def batches():
        for index in range(count):
            consumed.append(index)
            yield batch(index)

    model, state = PredictableModel(), TrainState(1)
    evaluation.evaluate(model, batches(), state, arguments())
    processed = min(count, 40)
    assert consumed == list(range(processed))
    assert model.forwarded == list(range(processed))
    assert state.this_eval_examples == processed
    assert state.this_eval_batches == processed
    assert state.this_text_loss == pytest.approx(-math.log(0.8), abs=1e-6)
    assert state.this_audio_loss == pytest.approx(math.log(2), abs=1e-6)
    assert state.this_text_perplexity == pytest.approx(1.25, abs=1e-6)
    assert state.this_audio_perplexity == pytest.approx(2, abs=1e-6)
    assert state.this_eval_perplexity == pytest.approx(math.sqrt(2.5), abs=1e-6)


def test_training_constructs_a_fresh_iterator_each_evaluation():
    tree = ast.parse((SOURCE / "train.py").read_text())
    train = next(n for n in tree.body if getattr(n, "name", None) == "_train")
    branches = [n for n in train.body if isinstance(n, ast.If)]
    setup = next(n for n in branches if ast.unparse(n.test) == "args.do_eval")
    statements = [n for n in ast.walk(train) if isinstance(n, ast.Expr)]
    call = next(n for n in statements if ast.unparse(n).startswith("evaluate("))
    constructions = []

    def build(**kwargs):
        constructions.append(kwargs)
        return iter([batch()])

    namespace = {
        "partial": partial,
        "args": arguments(),
        "build_data_loader": build,
        "interleaved_tokenizer": None,
        "get_rank": lambda: 0,
        "get_world_size": lambda: 1,
        "model": PredictableModel(),
        "state": TrainState(1),
        "evaluate": evaluation.evaluate,
    }
    exec(
        compile(ast.Module(body=[setup], type_ignores=[]), "train.py", "exec"),
        namespace,
    )
    for _ in range(3):
        exec(
            compile(ast.Module(body=[call], type_ignores=[]), "train.py", "exec"),
            namespace,
        )
        assert math.isfinite(namespace["state"].this_eval_loss)
    assert len(constructions) == 3
    assert all(row["is_eval"] and row["seed"] is None for row in constructions)


def test_weighted_tokens_aggregate_variable_batch_sizes():
    model, state = PredictableModel(), TrainState(1)
    evaluation.evaluate(
        model, iter([batch(size=1), batch(size=3, text_target=1)]), state, arguments()
    )
    text_nll = (-math.log(0.8) - 3 * math.log(0.2)) / 4
    assert state.this_eval_examples == 4
    assert state.this_text_loss == pytest.approx(text_nll, abs=1e-6)
    assert state.this_eval_loss == pytest.approx(text_nll + math.log(2), abs=1e-6)
    assert state.this_eval_perplexity == pytest.approx(
        math.exp((text_nll + math.log(2)) / 2), abs=1e-6
    )


@pytest.mark.parametrize("training", [False, True])
def test_empty_evaluation_fails_and_restores_mode(training):
    model = PredictableModel().train(training)
    with pytest.raises(ValueError, match="no batches"):
        evaluation.evaluate(model, iter([]), TrainState(1), arguments())
    assert model.training is training


@pytest.mark.parametrize("count", [0, 1, 15, 16, 17])
def test_validation_loader_retains_partial_batches(monkeypatch, count):
    monkeypatch.setattr(
        data_loader,
        "build_dataset",
        lambda **_: iter(
            [Sample(torch.zeros(1, 3, 1, dtype=torch.long)) for _ in range(count)]
        ),
    )
    loader = data_loader.build_data_loader(
        None,
        SimpleNamespace(train_data="train", eval_data="eval", shuffle=False),
        16,
        None,
        0,
        1,
        True,
    )
    sizes = [value.codes.shape[0] for value in loader]
    assert sum(sizes) == count
    assert len(sizes) == math.ceil(count / 16)


@pytest.mark.parametrize("mask_empty", [False, True])
def test_zero_weight_loss_is_finite_with_zero_gradient(mask_empty):
    logits = torch.full(
        (1, 1, 3, 2), float("nan") if mask_empty else 0.0, requires_grad=True
    )
    target = torch.zeros(1, 1, 3, dtype=torch.long)
    mask = torch.full_like(target, not mask_empty, dtype=torch.bool)
    loss = compute_loss_with_mask(
        logits, target, mask, "text", text_padding_weight=0, text_padding_ids={0}
    )
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 0


def test_fractional_weight_is_not_clamped_to_one():
    logits = torch.tensor([[[[math.log(0.8), math.log(0.2)]]]], requires_grad=True)
    target = torch.zeros(1, 1, 1, dtype=torch.long)
    loss = compute_loss_with_mask(
        logits,
        target,
        torch.ones_like(target, dtype=torch.bool),
        "text",
        text_padding_weight=0.25,
        text_padding_ids={0},
    )
    assert loss.item() == pytest.approx(-math.log(0.8), abs=1e-6)


def test_metrics_include_modalities_and_actual_coverage():
    path = SOURCE / "finetune/monitoring/metrics_logger.py"
    tree = ast.parse(path.read_text())
    body = [
        n
        for n in tree.body
        if getattr(n, "name", None) in {"get_eval_logs", "eval_log_msg"}
    ]
    namespace = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    logs = namespace["get_eval_logs"](
        1,
        2,
        text_perplexity=1.25,
        audio_perplexity=2.0,
        eval_examples=17,
        eval_batches=2,
    )
    assert logs["eval_examples"] == 17
    assert "text_perplexity" in namespace["eval_log_msg"](logs)


def fsdp_worker(rank, rendezvous, case):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model = FSDP(
            PredictableModel(), device_id=torch.device("cpu"), use_orig_params=True
        )
        state = TrainState(1)
        if case == "uneven":
            loader = iter([batch(i) for i in range(2 if rank == 0 else 1)])
            evaluation.evaluate(model, loader, state, arguments())
            assert len(model.forwarded) == 1
            assert state.this_eval_batches == 2
            assert state.this_eval_examples == 2
        elif case == "tail":
            evaluation.evaluate(
                model,
                iter([batch(size=16 if rank == 0 else 15, text_target=rank)]),
                state,
                arguments(),
            )
            expected = (-16 * math.log(0.8) - 15 * math.log(0.2)) / 31
            assert state.this_eval_examples == 31
            assert state.this_text_loss == pytest.approx(expected, abs=1e-6)
        elif case == "empty_rank":
            with pytest.raises(ValueError, match="no batches"):
                evaluation.evaluate(
                    model, iter([batch()] if rank == 0 else []), state, arguments()
                )
        elif case == "data_error":

            def broken():
                if rank == 1:
                    raise ValueError("local decode failure")
                yield batch()

            with pytest.raises(RuntimeError, match="data loading failed"):
                evaluation.evaluate(model, broken(), state, arguments())
        assert model.training
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["uneven", "tail", "empty_rank", "data_error"])
def test_real_fsdp_evaluation_has_matched_collectives(tmp_path, case):
    mp.spawn(
        fsdp_worker, args=(str(tmp_path / "rendezvous"), case), nprocs=2, join=True
    )
