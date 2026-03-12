"""Tests for BH FDR correction and related statistical utilities."""

import pytest
import numpy as np


class TestBenjaminiHochberg:
    """Test Benjamini-Hochberg FDR correction."""

    def test_all_significant(self):
        """Very small p-values should all be significant."""
        from fmri2img.stats.inference import benjamini_hochberg_correction
        p_vals = [0.001, 0.002, 0.003, 0.004, 0.005]
        results = benjamini_hochberg_correction(p_vals, alpha=0.05)
        significant = [sig for _, sig in results]
        assert all(significant), "All tiny p-values should be significant"

    def test_none_significant(self):
        """Large p-values should not be significant."""
        from fmri2img.stats.inference import benjamini_hochberg_correction
        p_vals = [0.80, 0.85, 0.90, 0.95]
        results = benjamini_hochberg_correction(p_vals, alpha=0.05)
        significant = [sig for _, sig in results]
        assert not any(significant), "All large p-values should be non-significant"

    def test_monotonicity(self):
        """Adjusted p-values should be monotonically non-decreasing when sorted by raw p."""
        from fmri2img.stats.inference import benjamini_hochberg_correction
        rng = np.random.RandomState(42)
        p_vals = rng.uniform(0, 1, 50).tolist()
        results = benjamini_hochberg_correction(p_vals, alpha=0.05)
        adjusted = [adj for adj, _ in results]
        sorted_indices = np.argsort(p_vals)
        sorted_adj = [adjusted[i] for i in sorted_indices]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1] - 1e-10

    def test_dict_input(self):
        """Should accept dict input and return dict output."""
        from fmri2img.stats.inference import benjamini_hochberg_correction
        p_dict = {"test_a": 0.01, "test_b": 0.06, "test_c": 0.50}
        results = benjamini_hochberg_correction(p_dict, alpha=0.05)
        assert isinstance(results, dict)
        assert set(results.keys()) == {"test_a", "test_b", "test_c"}
        for name, (adj_p, is_sig) in results.items():
            assert isinstance(adj_p, float)
            assert isinstance(is_sig, bool)

    def test_adjusted_geq_raw(self):
        """Adjusted p-values should be >= raw p-values."""
        from fmri2img.stats.inference import benjamini_hochberg_correction
        p_vals = [0.01, 0.03, 0.05, 0.10]
        results = benjamini_hochberg_correction(p_vals)
        for raw, (adj, _) in zip(p_vals, results):
            assert adj >= raw - 1e-10


class TestBonferroni:
    """Test simple Bonferroni correction."""

    def test_basic(self):
        """Bonferroni multiplies p by m."""
        from fmri2img.stats.inference import bonferroni_correction
        p_vals = [0.01, 0.02, 0.03]
        results = bonferroni_correction(p_vals, alpha=0.05)
        adjusted = [adj for adj, _ in results]
        assert abs(adjusted[0] - 0.03) < 1e-10
        assert abs(adjusted[1] - 0.06) < 1e-10

    def test_capped_at_one(self):
        """Adjusted p-values should not exceed 1.0."""
        from fmri2img.stats.inference import bonferroni_correction
        p_vals = [0.5, 0.9]
        results = bonferroni_correction(p_vals)
        adjusted = [adj for adj, _ in results]
        assert all(a <= 1.0 for a in adjusted)


class TestBHVsBonferroni:
    """Compare BH and Bonferroni corrections."""

    def test_bh_less_conservative(self):
        """BH should reject at least as many (often more) than Bonferroni."""
        from fmri2img.stats.inference import (
            benjamini_hochberg_correction,
            bonferroni_correction,
        )
        rng = np.random.RandomState(99)
        p_vals = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0.1, 1, 40)])
        p_list = p_vals.tolist()
        results_bh = benjamini_hochberg_correction(p_list, alpha=0.05)
        results_bf = bonferroni_correction(p_list, alpha=0.05)
        n_bh = sum(sig for _, sig in results_bh)
        n_bf = sum(sig for _, sig in results_bf)
        assert n_bh >= n_bf, "BH should not be more conservative than Bonferroni"


class TestFDRThreshold:
    """Test FDR threshold computation."""

    def test_returns_float(self):
        """fdr_threshold should return a float."""
        from fmri2img.stats.inference import fdr_threshold
        p_vals = [0.001, 0.01, 0.05, 0.10, 0.50]
        result = fdr_threshold(p_vals, alpha=0.05)
        assert isinstance(result, float)

    def test_threshold_leq_alpha(self):
        """FDR threshold should be <= alpha."""
        from fmri2img.stats.inference import fdr_threshold
        p_vals = [0.001, 0.01, 0.05, 0.10, 0.50]
        result = fdr_threshold(p_vals, alpha=0.05)
        assert result <= 0.05 + 1e-10


class TestMultiTestSummary:
    """Test multi-correction summary utility."""

    def test_returns_all_corrections(self):
        """Should include all requested corrections."""
        from fmri2img.stats.inference import multi_test_summary
        p_vals = {"test1": 0.01, "test2": 0.04, "test3": 0.50}
        summary = multi_test_summary(p_vals)
        for key in p_vals:
            assert key in summary
            assert "bh_adj" in summary[key]
            assert "bonf_adj" in summary[key]
            assert "holm_adj" in summary[key]
