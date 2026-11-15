"""Real two-process CPU/Gloo FSDP, plus explicit CUDA-placement fixtures."""

import ast
import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from moshi.models.lm import LMModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

import finetune.distributed as distributed
from finetune.wrapped_model import get_fsdp_policy

SOURCE = Path(distributed.__file__).parents[1]


def train_statements():
    tree = ast.parse((SOURCE / "train.py").read_text())
    train = next(n for n in tree.body if getattr(n, "name", None) == "_train")

    def calls(node, name):
        if not isinstance(node, ast.Expr):
            return False
        if not isinstance(node.value, ast.Call):
            return False
        return ast.unparse(node.value.func).split(".")[-1] == name

    def normalizes_gradients(node):
        if not isinstance(node, ast.If):
            return False
        attributes = [getattr(t, "attr", None) for t in ast.walk(node.test)]
        if "num_microbatches" not in attributes:
            return False
        return "p.grad.div_" in ast.unparse(node)

    micro = []
    for node in ast.walk(train):
        if isinstance(node, ast.Assign):
            if any(ast.unparse(t) == "mb_loss" for t in node.targets):
                micro.append(node)
        if calls(node, "backward"):
            micro.append(node)
        if isinstance(node, ast.AugAssign):
            if ast.unparse(node.target) == "loss":
                micro.append(node)
    micro.sort(key=lambda n: n.lineno)
    normalize = [n for n in ast.walk(train) if normalizes_gradients(n)]
    clipping = next(n for n in ast.walk(train) if calls(n, "clip_grad_norm_"))

    def compile_nodes(nodes):
        return compile(ast.Module(body=nodes, type_ignores=[]), "train.py", "exec")

    return compile_nodes(micro), compile_nodes(normalize), compile_nodes([clipping])


def initialize(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=30),
    )


class Scalars(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0]))
        self.b = torch.nn.Parameter(torch.tensor([2.0]))

    def forward(self, inputs):
        return 3 * self.a.sum() + 4 * self.b.sum() + inputs.sum() * 0


def clip_worker(rank, rendezvous, destination):
    initialize(rank, rendezvous)
    try:
        model = FSDP(
            Scalars(),
            device_id=torch.device("cpu"),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
        )
        model(torch.ones(1)).backward()
        _, _, clip = train_statements()
        try:
            from finetune.training import clip_grad_norm_
        except ModuleNotFoundError:
            clip_grad_norm_ = None
        exec(
            clip,
            {
                "torch": torch,
                "model": model,
                "args": SimpleNamespace(max_norm=1),
                "clip_grad_norm_": clip_grad_norm_,
            },
        )
        squared = torch.tensor(
            sum(
                float(p.grad.square().sum())
                for p in model.parameters()
                if p.grad is not None
            )
        )
        dist.all_reduce(squared)
        norm = squared.sqrt().item()
        Path(destination, f"clip-{rank}.json").write_text(json.dumps({"norm": norm}))
        assert abs(norm - 1) < 1e-6
    finally:
        dist.destroy_process_group()


def accumulation_worker(rank, rendezvous, destination):
    initialize(rank, rendezvous)
    try:
        torch.manual_seed(812)
        raw = LMModel(
            n_q=4,
            dep_q=2,
            card=8,
            text_card=16,
            dim=16,
            num_heads=2,
            num_layers=1,
            hidden_scale=2,
            delays=[0, 0, 1, 0, 1],
            depformer_dim=16,
            depformer_num_heads=2,
            depformer_num_layers=1,
            device="cpu",
            dtype=torch.float32,
        )
        model = FSDP(
            raw,
            device_id=torch.device("cpu"),
            use_orig_params=True,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=get_fsdp_policy(False),
        )
        batches = [torch.randint(0, 8, (2, 5, 6)) for _ in range(2)]

        def losses(codes):
            out = model(codes)
            return (
                torch.nn.functional.cross_entropy(
                    out.text_logits[out.text_mask], codes[:, :1][out.text_mask]
                ),
                torch.nn.functional.cross_entropy(
                    out.logits[out.mask], codes[:, 1:3][out.mask]
                ),
            )

        reference_losses = []
        for codes in batches:
            text, audio = losses(codes)
            reference_losses.append((text + audio).item())
            (text + audio).backward()
        reference = {
            n: p.grad.clone() / 2
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        empty = sum(p.numel() == 0 and p.grad is None for p in model.parameters())
        assert empty > 0
        model.zero_grad(set_to_none=True)
        micro, normalize, _ = train_statements()
        namespace = {
            "torch": torch,
            "model": model,
            "args": SimpleNamespace(num_microbatches=2),
            "loss": torch.tensor(0.0),
        }
        for codes in batches:
            namespace["text_loss"], namespace["audio_loss"] = losses(codes)
            exec(micro, namespace)
        exec(normalize, namespace)
        error = max(
            float((p.grad - reference[n]).abs().max())
            for n, p in model.named_parameters()
            if p.grad is not None and p.grad.numel()
        )
        Path(destination, f"accumulate-{rank}.json").write_text(
            json.dumps({"empty_parameters": empty, "max_error": error})
        )
        assert error < 2e-6
        assert abs(namespace["loss"].item() - sum(reference_losses) / 2) < 1e-6
    finally:
        dist.destroy_process_group()


def directory_worker(rank, rendezvous, destination):
    initialize(rank, rendezvous)
    try:
        run = Path(destination, "run")
        writes = []

        def save(path):
            writes.append(rank)
            path.write_text(f"writer: {rank}")

        distributed.prepare_run_dir(run, True, save_args=save)
        assert not (run / "old").exists()
        assert (run / "args.yaml").read_text() == "writer: 0"
        assert writes == ([0] if rank == 0 else [])
        with pytest.raises(RuntimeError, match="already exists"):
            distributed.prepare_run_dir(run, False)

        def fail(_):
            raise ValueError("write failure witness")

        with pytest.raises(RuntimeError, match="write failure witness"):
            distributed.prepare_run_dir(
                Path(destination, "failed"), False, save_args=fail
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("worker", [clip_worker, accumulation_worker, directory_worker])
def test_two_process_training_contracts(tmp_path, worker):
    if worker is directory_worker:
        (tmp_path / "run").mkdir()
        (tmp_path / "run/old").touch()
    mp.spawn(
        worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True
    )


@pytest.mark.parametrize(
    "visibility",
    [
        None,
        "0,1",
        "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6,GPU-other",
        "MIG-GPU-example/1/0",
    ],
)
@pytest.mark.parametrize(("count", "local_rank", "expected"), [(1, 3, 0), (2, 1, 1)])
def test_device_selection_uses_cuda_logical_ids(
    monkeypatch, visibility, count, local_rank, expected
):
    if visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    distributed.set_device()
    assert selected == [expected]


@pytest.mark.parametrize("local_rank", [-1, 2])
def test_invalid_local_rank_fails_clearly(monkeypatch, local_rank):
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="LOCAL_RANK"):
        distributed.set_device()
