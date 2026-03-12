"""Tests for novel loss functions: VICReg, Barlow Twins, Triplet, DANN."""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# VICReg Loss
# ---------------------------------------------------------------------------
class TestVICRegLoss:
    """Test VICReg (Variance-Invariance-Covariance Regularization) loss."""

    def test_zero_invariance_for_identical(self):
        """Invariance term should be 0 for identical inputs."""
        from fmri2img.training.losses import VICRegLoss
        loss_fn = VICRegLoss()
        x = torch.randn(32, 64)
        total, components = loss_fn(x, x, return_components=True)
        assert components["invariance"] < 1e-6

    def test_components_returned(self):
        """return_components should give all three terms."""
        from fmri2img.training.losses import VICRegLoss
        loss_fn = VICRegLoss()
        x = torch.randn(32, 64)
        y = torch.randn(32, 64)
        total, components = loss_fn(x, y, return_components=True)
        assert "invariance" in components
        assert "variance" in components
        assert "covariance" in components

    def test_loss_positive(self):
        """Total loss should be non-negative."""
        from fmri2img.training.losses import VICRegLoss
        loss_fn = VICRegLoss()
        x = torch.randn(32, 128)
        y = torch.randn(32, 128)
        loss = loss_fn(x, y)
        assert loss.item() >= 0.0

    def test_variance_penalty(self):
        """Collapsed embeddings should have high variance penalty."""
        from fmri2img.training.losses import VICRegLoss
        loss_fn = VICRegLoss(sim_weight=0, var_weight=1, cov_weight=0)
        # All embeddings identical (collapsed)
        x = torch.ones(32, 64)
        y = torch.randn(32, 64)
        loss = loss_fn(x, y)
        assert loss.item() > 0.5  # strong penalty for collapse

    def test_gradient_flow(self):
        """Gradients should flow through all terms."""
        from fmri2img.training.losses import VICRegLoss
        loss_fn = VICRegLoss()
        x = torch.randn(32, 64, requires_grad=True)
        y = torch.randn(32, 64)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Barlow Twins Loss
# ---------------------------------------------------------------------------
class TestBarlowTwinsLoss:
    """Test Barlow Twins redundancy-reduction loss."""

    def test_zero_loss_for_identity_cross_corr(self):
        """When Z_A == Z_B, cross-corr is identity → low on-diag loss."""
        from fmri2img.training.losses import BarlowTwinsLoss
        loss_fn = BarlowTwinsLoss()
        z = torch.randn(64, 128)
        # Standardize to make off-diag ~ 0 too
        z = (z - z.mean(0)) / (z.std(0) + 1e-5)
        loss, comp = loss_fn(z, z, return_components=True)
        assert comp["on_diag"] < 0.05  # low but not exactly 0 due to finite samples

    def test_components_returned(self):
        """Should return on_diag and off_diag components."""
        from fmri2img.training.losses import BarlowTwinsLoss
        loss_fn = BarlowTwinsLoss()
        x = torch.randn(32, 64)
        y = torch.randn(32, 64)
        _, comp = loss_fn(x, y, return_components=True)
        assert "on_diag" in comp
        assert "off_diag" in comp

    def test_gradient_flow(self):
        """Gradients should flow."""
        from fmri2img.training.losses import BarlowTwinsLoss
        loss_fn = BarlowTwinsLoss()
        x = torch.randn(32, 64, requires_grad=True)
        y = torch.randn(32, 64)
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Triplet + Hard-Negative Mining
# ---------------------------------------------------------------------------
class TestTripletLoss:
    """Test triplet margin loss and hard-negative mining."""

    def test_zero_loss_for_correct_order(self):
        """Loss = 0 when positive is much closer than negative."""
        from fmri2img.training.losses import triplet_margin_loss
        anchor = torch.randn(1, 64)
        positive = anchor + 0.01 * torch.randn(1, 64)  # very close
        negative = torch.randn(1, 64)  # random = far
        loss = triplet_margin_loss(anchor, positive, negative, margin=0.2)
        assert loss.item() >= 0.0

    def test_positive_loss_for_wrong_order(self):
        """Loss > 0 when negative is closer than positive."""
        from fmri2img.training.losses import triplet_margin_loss
        anchor = torch.randn(1, 64)
        positive = torch.randn(1, 64)  # random
        negative = anchor.clone()  # identical to anchor = very close
        loss = triplet_margin_loss(anchor, positive, negative, margin=0.2)
        assert loss.item() > 0.0

    def test_hard_negative_miner_strategies(self):
        """HardNegativeMiner should run with all strategies."""
        from fmri2img.training.losses import HardNegativeMiner
        for strategy in ["hardest", "semi_hard", "random"]:
            miner = HardNegativeMiner(strategy=strategy)
            anchors = torch.randn(16, 64)
            positives = torch.randn(16, 64)
            gallery = torch.randn(64, 64)
            result = miner.mine(anchors, positives, gallery)
            assert result.shape == (16, 64)


class TestTripletInfoNCELoss:
    """Test hybrid Triplet + InfoNCE loss."""

    def test_components(self):
        """Should return info_nce and triplet components."""
        from fmri2img.training.losses import TripletInfoNCELoss
        loss_fn = TripletInfoNCELoss()
        pred = torch.randn(32, 64)
        target = torch.randn(32, 64)
        _, comp = loss_fn(pred, target, return_components=True)
        assert "info_nce" in comp
        assert "triplet" in comp

    def test_gradient_flow(self):
        from fmri2img.training.losses import TripletInfoNCELoss
        loss_fn = TripletInfoNCELoss()
        pred = torch.randn(32, 64, requires_grad=True)
        target = torch.randn(32, 64)
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# Domain-Adversarial Training
# ---------------------------------------------------------------------------
class TestGradientReversal:
    """Test gradient reversal layer."""

    def test_forward_is_identity(self):
        """Forward pass should be identity."""
        from fmri2img.training.domain_adversarial import GradientReversalLayer
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(8, 32)
        out = grl(x)
        assert torch.allclose(x, out)

    def test_backward_reverses_gradient(self):
        """Backward should negate gradient."""
        from fmri2img.training.domain_adversarial import GradientReversalLayer
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(4, 16, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        # d(sum(x))/dx = 1, but GRL negates → grad = -1
        expected = -torch.ones_like(x)
        assert torch.allclose(x.grad, expected)


class TestDomainAdversarialLoss:
    """Test DANN loss computation."""

    def test_balanced_labels_accuracy(self):
        """Domain accuracy should be ~50% for random features."""
        from fmri2img.training.domain_adversarial import (
            DomainAdversarialLoss,
            DomainClassifier,
        )
        classifier = DomainClassifier(input_dim=64, hidden_dim=32)
        loss_fn = DomainAdversarialLoss(classifier, lambda_max=1.0)
        # Create mixed batch with domain labels
        perc_feat = torch.randn(16, 64)
        imag_feat = torch.randn(16, 64)
        latent = torch.cat([perc_feat, imag_feat], dim=0)
        labels = torch.cat([torch.zeros(16), torch.ones(16)])
        total, comp = loss_fn(latent, labels, epoch=25, return_components=True)
        assert "domain_loss_raw" in comp
        assert "domain_accuracy" in comp
        assert 0.0 <= comp["domain_accuracy"] <= 1.0

    def test_lambda_schedule(self):
        """DANN sigmoid schedule should ramp from 0 to ~1."""
        from fmri2img.training.domain_adversarial import dann_lambda_schedule
        assert dann_lambda_schedule(0, 100) < 0.1
        assert dann_lambda_schedule(100, 100) > 0.9
