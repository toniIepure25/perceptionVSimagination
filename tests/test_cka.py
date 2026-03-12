"""Tests for CKA analysis module."""

import pytest
import numpy as np


class TestLinearCKA:
    """Test linear CKA implementation."""

    def test_identical_representations(self):
        """CKA(X, X) should equal 1.0."""
        from fmri2img.analysis.cka import linear_cka
        X = np.random.randn(50, 32)
        assert abs(linear_cka(X, X) - 1.0) < 1e-6

    def test_orthogonal_invariance(self):
        """CKA should be invariant to orthogonal transforms."""
        from fmri2img.analysis.cka import linear_cka
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        Q, _ = np.linalg.qr(rng.randn(10, 10))  # random orthogonal
        Y = X @ Q
        assert abs(linear_cka(X, Y) - 1.0) < 1e-5

    def test_isotropic_scaling_invariance(self):
        """CKA(X, αX) == 1.0 for any α ≠ 0."""
        from fmri2img.analysis.cka import linear_cka
        X = np.random.randn(50, 20)
        assert abs(linear_cka(X, 3.7 * X) - 1.0) < 1e-6

    def test_range_zero_to_one(self):
        """CKA should be in [0, 1]."""
        from fmri2img.analysis.cka import linear_cka
        rng = np.random.RandomState(0)
        X = rng.randn(40, 15)
        Y = rng.randn(40, 20)
        val = linear_cka(X, Y)
        assert 0.0 <= val <= 1.0 + 1e-6

    def test_different_feature_dims(self):
        """CKA works for X ∈ R^{n×p} and Y ∈ R^{n×q} with p ≠ q."""
        from fmri2img.analysis.cka import linear_cka
        rng = np.random.RandomState(1)
        X = rng.randn(60, 10)
        Y = rng.randn(60, 25)
        val = linear_cka(X, Y)
        assert 0.0 <= val <= 1.0 + 1e-6


class TestRBFCKA:
    """Test RBF kernel CKA."""

    def test_identical_representations(self):
        """RBF CKA(X, X) should be close to 1.0."""
        from fmri2img.analysis.cka import rbf_cka
        X = np.random.randn(50, 8)
        assert rbf_cka(X, X) > 0.99

    def test_range(self):
        """RBF CKA should be in [0, 1]."""
        from fmri2img.analysis.cka import rbf_cka
        rng = np.random.RandomState(2)
        X = rng.randn(40, 8)
        Y = rng.randn(40, 12)
        val = rbf_cka(X, Y)
        assert -0.01 <= val <= 1.01


class TestDebiasedCKA:
    """Test unbiased HSIC estimator."""

    def test_debiased_close_for_large_n(self):
        """Debiased and biased should converge for large n."""
        from fmri2img.analysis.cka import linear_cka
        rng = np.random.RandomState(3)
        X = rng.randn(500, 10)
        Y = rng.randn(500, 10)
        biased = linear_cka(X, Y, debiased=False)
        debiased = linear_cka(X, Y, debiased=True)
        assert abs(biased - debiased) < 0.05


class TestCKAMatrix:
    """Test CKA matrix computation."""

    def test_square_matrix_with_unit_diagonal(self):
        """CKA matrix diagonal should be 1.0."""
        from fmri2img.analysis.cka import compute_cka_matrix
        rng = np.random.RandomState(4)
        reps = {
            "A": rng.randn(30, 10),
            "B": rng.randn(30, 15),
            "C": rng.randn(30, 8),
        }
        result = compute_cka_matrix(reps)
        np.testing.assert_allclose(np.diag(result.matrix), 1.0, atol=1e-5)

    def test_symmetric(self):
        """CKA matrix should be symmetric."""
        from fmri2img.analysis.cka import compute_cka_matrix
        rng = np.random.RandomState(5)
        reps = {
            "A": rng.randn(30, 10),
            "B": rng.randn(30, 15),
        }
        result = compute_cka_matrix(reps)
        np.testing.assert_allclose(result.matrix, result.matrix.T, atol=1e-6)


class TestCKASignificance:
    """Test permutation-based significance testing."""

    def test_significant_for_identical(self):
        """Identical representations should have p ≈ 0."""
        from fmri2img.analysis.cka import compute_cka_significance
        X = np.random.randn(40, 10)
        reps = {"A": X, "B": X.copy()}
        result = compute_cka_significance(reps, n_permutations=100)
        assert result.matrix[0, 1] > 0.99
        assert result.pvalue_matrix[0, 1] < 0.05
