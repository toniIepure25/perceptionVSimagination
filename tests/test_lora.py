"""Tests for LoRA adapter module."""

import pytest
import sys
import os
import importlib.util
import torch
import numpy as np

# Direct module import to avoid torchvision dependency in models/__init__.py
_lora_path = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'fmri2img', 'models', 'lora_adapter.py'
)
_spec = importlib.util.spec_from_file_location("lora_adapter", _lora_path)
_lora_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lora_mod)

LoRALinear = _lora_mod.LoRALinear
LoRAAdapter = _lora_mod.LoRAAdapter
MultiRankLoRA = _lora_mod.MultiRankLoRA
LoRAAdaptedModel = _lora_mod.LoRAAdaptedModel
save_lora_adapter = _lora_mod.save_lora_adapter
load_lora_adapter = _lora_mod.load_lora_adapter


class TestLoRALinear:
    """Test low-rank linear layer."""

    def test_output_shape(self):
        """Output shape should match expected dimensions."""
        # already imported at module level
        lora = LoRALinear(in_features=768, out_features=768, rank=8)
        x = torch.randn(4, 768)
        out = lora(x)
        assert out.shape == (4, 768)

    def test_starts_near_zero(self):
        """LoRA should produce near-zero output at init (B initialized to 0)."""
        # already imported at module level
        lora = LoRALinear(768, 768, rank=4)
        x = torch.randn(4, 768)
        out = lora(x)
        assert out.abs().max().item() < 1e-5

    def test_param_count(self):
        """LoRA params should be rank × (in + out)."""
        # already imported at module level
        rank = 8
        in_f, out_f = 768, 512
        lora = LoRALinear(in_f, out_f, rank=rank)
        n_params = sum(p.numel() for p in lora.parameters())
        assert n_params == rank * in_f + rank * out_f


class TestLoRAAdapter:
    """Test LoRA adapter (standalone, residual)."""

    def test_residual_identity_at_init(self):
        """At init, adapter should be near-identity (output ≈ input)."""
        adapter = LoRAAdapter(embed_dim=768, rank=4, normalize=False)
        x = torch.randn(8, 768)
        out = adapter(x)
        # Should be close to x since LoRA starts at zero
        assert torch.allclose(x, out, atol=1e-4)

    def test_l2_normalize(self):
        """With normalize=True, output should be unit-norm."""
        adapter = LoRAAdapter(embed_dim=768, rank=4, normalize=True)
        x = torch.randn(4, 768)
        out = adapter(x)
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_gradient_flow(self):
        """Gradients should flow through LoRA parameters."""
        adapter = LoRAAdapter(embed_dim=768, rank=4)
        x = torch.randn(4, 768)
        out = adapter(x)
        loss = out.sum()
        loss.backward()
        for p in adapter.parameters():
            if p.requires_grad:
                assert p.grad is not None


class TestMultiRankLoRA:
    """Test multi-rank LoRA with automatic rank selection."""

    def test_output_shape(self):
        """Output shape should match."""
        multi = MultiRankLoRA(embed_dim=768)
        x = torch.randn(4, 768)
        out = multi(x)
        assert out.shape == (4, 768)

    def test_rank_weights_sum_to_one(self):
        """Softmax rank weights should sum to 1."""
        multi = MultiRankLoRA(embed_dim=768, ranks=[2, 4, 8])
        weights = torch.softmax(multi.weight_logits, dim=0)
        assert abs(weights.sum().item() - 1.0) < 1e-6

    def test_near_identity_at_init(self):
        """Multi-rank should also start near identity for same-dim."""
        multi = MultiRankLoRA(embed_dim=768, normalize=False)
        x = torch.randn(4, 768)
        out = multi(x)
        assert torch.allclose(x, out, atol=1e-3)


class TestLoRAAdaptedModel:
    """Test frozen encoder + LoRA wrapper."""

    def test_base_model_frozen(self):
        """Base model parameters should not require grad."""
        base = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 768),
        )
        adapter = LoRAAdapter(embed_dim=768, rank=4)
        adapted = LoRAAdaptedModel(base, adapter)
        for p in adapted.base_model.parameters():
            assert not p.requires_grad

    def test_lora_params_trainable(self):
        """LoRA parameters should require grad."""
        base = torch.nn.Linear(768, 768)
        adapter = LoRAAdapter(embed_dim=768, rank=4)
        adapted = LoRAAdaptedModel(base, adapter)
        trainable = [p for p in adapted.parameters() if p.requires_grad]
        assert len(trainable) > 0


class TestLoRASaveLoad:
    """Test save/load roundtrip."""

    def test_roundtrip(self, tmp_path):
        """Save and load should produce identical outputs."""
        adapter = LoRAAdapter(embed_dim=768, rank=8)
        # Modify weights so they're non-zero
        with torch.no_grad():
            for p in adapter.parameters():
                p.fill_(0.42)

        save_path = tmp_path / "lora.pt"
        save_lora_adapter(adapter, str(save_path))

        loaded, meta = load_lora_adapter(str(save_path))
        x = torch.randn(4, 768)

        out_orig = adapter(x)
        out_loaded = loaded(x)
        assert torch.allclose(out_orig, out_loaded, atol=1e-6)
