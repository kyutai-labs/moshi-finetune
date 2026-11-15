"""Real small LM and loader; CPU, no pretrained weights, single-rank fixture."""

import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F
from moshi.models import loaders

import finetune.checkpointing as checkpointing
from finetune.wrapped_model import initialize_lora_parameters


def config():
    return {
        "n_q": 4,
        "dep_q": 2,
        "card": 8,
        "text_card": 16,
        "dim": 16,
        "num_heads": 2,
        "hidden_scale": 2,
        "num_layers": 1,
        "delays": [0, 0, 1, 0, 1],
        "depformer_dim": 16,
        "depformer_num_heads": 2,
        "depformer_num_layers": 1,
    }


def logits(model, codes):
    out = model(codes)
    return torch.cat(
        (out.text_logits[out.text_mask].flatten(), out.logits[out.mask].flatten())
    ).detach()


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(checkpointing, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpointing, "get_world_size", lambda: 1)
    monkeypatch.setattr(checkpointing, "barrier", lambda: None)


@pytest.mark.parametrize("ft_embed", [False, True])
@pytest.mark.parametrize("updates", [0, 1])
@pytest.mark.parametrize("adapters_only", [False, True])
def test_export_preserves_model_function(tmp_path, ft_embed, updates, adapters_only):
    torch.manual_seed(20261115)
    cfg = config()
    training_config = dict(cfg, lora=True, lora_rank=2, lora_scaling=2.0)
    base = loaders.get_moshi_lm(None, lm_kwargs=cfg, dtype=torch.float32, device="cpu")
    base_path = tmp_path / "base.safetensors"
    safetensors.torch.save_file(base.state_dict(), base_path)
    model = loaders.get_moshi_lm(
        None, lm_kwargs=training_config, dtype=torch.float32, device="meta"
    )
    model.load_state_dict(base.state_dict(), strict=False, assign=True)
    initialize_lora_parameters(model, torch.float32)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora" in name or (ft_embed and "emb" in name)
    codes = torch.randint(0, 8, (2, 5, 8))
    codes[:, 0] = torch.randint(0, 16, (2, 8))
    if updates:
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.1
        )
        out = model(codes)
        loss = F.cross_entropy(
            out.text_logits[out.text_mask], codes[:, :1][out.text_mask]
        )
        loss += F.cross_entropy(out.logits[out.mask], codes[:, 1:3][out.mask])
        loss.backward()
        optimizer.step()
    model.eval()
    expected = logits(model, codes)
    original_config = dict(training_config)
    cp = checkpointing.Checkpointer(
        model, SimpleNamespace(step=1), tmp_path / "run", training_config
    )
    cp.save_checkpoint(adapters_only, dtype=torch.float32)
    exported = json.loads((cp.dst_dir / "config.json").read_text())
    assert training_config == original_config
    if adapters_only:
        assert exported["lora"] is True
        weights = cp.dst_dir / "lora.safetensors"
        actual = loaders.get_moshi_lm(
            base_path,
            lm_kwargs=exported,
            lora_weights=weights,
            dtype=torch.float32,
            device="cpu",
        )
    else:
        assert exported["lora"] is False
        assert not {"lora_rank", "lora_scaling", "lora_name"} & exported.keys()
        actual = loaders.get_moshi_lm(
            cp.dst_dir / "consolidated.safetensors",
            lm_kwargs=exported,
            dtype=torch.float32,
            device="cpu",
        )
    assert torch.allclose(logits(actual, codes), expected, atol=2e-6, rtol=0)


def test_full_finetuning_checkpoint_still_loads(tmp_path):
    torch.manual_seed(7)
    cfg = config()
    model = loaders.get_moshi_lm(None, lm_kwargs=cfg, dtype=torch.float32, device="cpu")
    cp = checkpointing.Checkpointer(
        model, SimpleNamespace(step=1), tmp_path, cfg, full_finetuning=True
    )
    cp.save_checkpoint(False, dtype=torch.float32)
    saved = json.loads((cp.dst_dir / "config.json").read_text())
    reloaded = loaders.get_moshi_lm(
        cp.dst_dir / "consolidated.safetensors",
        lm_kwargs=saved,
        dtype=torch.float32,
        device="cpu",
    )
    for key, value in model.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[key])


def test_merged_metadata_removes_stale_adapter_reference(tmp_path):
    cfg = dict(
        config(),
        lora=True,
        lora_rank=2,
        lora_scaling=2,
        lora_name="old-adapter.safetensors",
    )
    model = loaders.get_moshi_lm(
        None, lm_kwargs=config(), dtype=torch.float32, device="cpu"
    )
    cp = checkpointing.Checkpointer(model, SimpleNamespace(step=1), tmp_path, cfg)
    cp.write_params_info(tmp_path)
    exported = json.loads((tmp_path / "config.json").read_text())
    assert exported["lora"] is False
    assert "lora_name" not in exported
    assert cfg["lora_name"] == "old-adapter.safetensors"
