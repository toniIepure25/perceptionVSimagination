"""
Direction 5: The Individual Imagery Fingerprint
=================================================

Cross-subject analysis of imagery representations. Tests whether subjects
have consistent, individualized "imagery styles" detectable from decoder outputs.

Compares:
- Per-subject imagery degradation profiles (which CLIP dimensions degrade most)
- Adapter weight similarity across subjects
- Second-order RSA: do similar perceivers also imagine similarly?

References:
    Charest et al. (2014). "Unique semantic space in the brain of each
    beholder predicts neural responses" (individual representational spaces)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy import stats as scipy_stats

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)


def compute_degradation_profile(bundle: EmbeddingBundle) -> np.ndarray:
    """
    Per-dimension degradation from perception to imagery.

    For each CLIP dimension d, measures the drop in correlation between
    decoder prediction and ground truth when switching from perception to imagery.
    Returns a (D,) array where larger values indicate more degradation.
    """
    D = bundle.embed_dim

    perc_corr = np.zeros(D)
    imag_corr = np.zeros(D)

    for d in range(D):
        p = bundle.perception[:, d]
        pt = bundle.perception_targets[:, d]
        if np.std(p) > 1e-8 and np.std(pt) > 1e-8:
            perc_corr[d] = np.corrcoef(p, pt)[0, 1]

        ip = bundle.imagery[:, d]
        it = bundle.imagery_targets[:, d]
        if np.std(ip) > 1e-8 and np.std(it) > 1e-8:
            imag_corr[d] = np.corrcoef(ip, it)[0, 1]

    degradation = perc_corr - imag_corr
    return degradation


def compare_degradation_profiles(
    profiles: Dict[str, np.ndarray],
) -> Dict[str, Dict]:
    """
    Compare degradation profiles across subjects using correlation.

    High correlation means subjects degrade the same dimensions → universal
    imagery structure. Low correlation → individual imagery styles.
    """
    subjects = list(profiles.keys())
    n_subj = len(subjects)
    corr_matrix = np.zeros((n_subj, n_subj))

    for i in range(n_subj):
        for j in range(n_subj):
            r, _ = scipy_stats.pearsonr(profiles[subjects[i]], profiles[subjects[j]])
            corr_matrix[i, j] = r

    triu_idx = np.triu_indices(n_subj, k=1)
    pairwise_corrs = corr_matrix[triu_idx]

    return {
        "correlation_matrix": corr_matrix.tolist(),
        "subjects": subjects,
        "mean_pairwise_correlation": float(np.mean(pairwise_corrs)) if len(pairwise_corrs) > 0 else float("nan"),
        "std_pairwise_correlation": float(np.std(pairwise_corrs)) if len(pairwise_corrs) > 0 else float("nan"),
        "pairwise_correlations": {
            f"{subjects[i]}_vs_{subjects[j]}": float(corr_matrix[i, j])
            for i, j in zip(*triu_idx)
        },
    }


def compare_adapter_weights(
    adapter_checkpoints: Dict[str, str],
) -> Dict:
    """
    Load adapter checkpoints for multiple subjects and compare weight similarity.

    Similar adapter weights across subjects → universal imagery transform.
    Different weights → subject-specific imagery strategies.
    """
    import torch

    weights = {}
    for subject, path in adapter_checkpoints.items():
        try:
            ckpt = torch.load(path, map_location="cpu")
            state = ckpt.get("adapter_state_dict", ckpt)
            flat = torch.cat([p.flatten() for p in state.values()]).numpy()
            weights[subject] = flat
        except Exception as e:
            logger.warning(f"Could not load adapter for {subject}: {e}")

    if len(weights) < 2:
        return {"available": False, "reason": "Need at least 2 subjects"}

    subjects = list(weights.keys())
    n_subj = len(subjects)
    cosine_matrix = np.zeros((n_subj, n_subj))

    for i in range(n_subj):
        for j in range(n_subj):
            wi = weights[subjects[i]]
            wj = weights[subjects[j]]
            min_len = min(len(wi), len(wj))
            wi, wj = wi[:min_len], wj[:min_len]
            cos = np.dot(wi, wj) / (np.linalg.norm(wi) * np.linalg.norm(wj) + 1e-8)
            cosine_matrix[i, j] = cos

    triu_idx = np.triu_indices(n_subj, k=1)
    pairwise = cosine_matrix[triu_idx]

    return {
        "available": True,
        "cosine_similarity_matrix": cosine_matrix.tolist(),
        "subjects": subjects,
        "mean_cosine": float(np.mean(pairwise)),
        "std_cosine": float(np.std(pairwise)),
    }


def second_order_rsa(
    bundles: Dict[str, EmbeddingBundle],
    n_stimuli: Optional[int] = None,
) -> Dict:
    """
    Second-order RSA: compare RDMs across subjects.

    For each subject, computes a perception RDM and an imagery RDM on matched
    stimuli, then checks whether subjects with similar perception RDMs also
    have similar imagery RDMs.
    """
    from scipy.spatial.distance import pdist

    subjects = list(bundles.keys())
    n_subj = len(subjects)

    perc_rdms_flat = {}
    imag_rdms_flat = {}

    for subj, bundle in bundles.items():
        n = n_stimuli or min(bundle.perception.shape[0], bundle.imagery.shape[0], 100)
        perc_rdms_flat[subj] = pdist(bundle.perception[:n], metric="correlation")
        imag_rdms_flat[subj] = pdist(bundle.imagery[:n], metric="correlation")

    # Build second-order similarity matrices
    perc_sim = np.zeros((n_subj, n_subj))
    imag_sim = np.zeros((n_subj, n_subj))

    for i in range(n_subj):
        for j in range(n_subj):
            min_len = min(len(perc_rdms_flat[subjects[i]]), len(perc_rdms_flat[subjects[j]]))
            r_p, _ = scipy_stats.spearmanr(
                perc_rdms_flat[subjects[i]][:min_len],
                perc_rdms_flat[subjects[j]][:min_len],
            )
            r_i, _ = scipy_stats.spearmanr(
                imag_rdms_flat[subjects[i]][:min_len],
                imag_rdms_flat[subjects[j]][:min_len],
            )
            perc_sim[i, j] = r_p
            imag_sim[i, j] = r_i

    # Correlation between perception similarity and imagery similarity
    triu = np.triu_indices(n_subj, k=1)
    perc_flat = perc_sim[triu]
    imag_flat = imag_sim[triu]

    if len(perc_flat) >= 3:
        consistency_r, consistency_p = scipy_stats.pearsonr(perc_flat, imag_flat)
    else:
        consistency_r, consistency_p = float("nan"), float("nan")

    return {
        "perception_similarity_matrix": perc_sim.tolist(),
        "imagery_similarity_matrix": imag_sim.tolist(),
        "subjects": subjects,
        "perception_imagery_consistency_r": float(consistency_r),
        "perception_imagery_consistency_p": float(consistency_p),
    }


def analyze_cross_subject(
    bundles: Dict[str, EmbeddingBundle],
    adapter_checkpoints: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Full cross-subject analysis.
    """
    logger.info("Running cross-subject imagery analysis...")
    logger.info(f"  Subjects: {list(bundles.keys())}")

    # Per-subject degradation profiles
    profiles = {}
    per_subject_stats = {}
    for subj, bundle in bundles.items():
        profiles[subj] = compute_degradation_profile(bundle)
        per_subject_stats[subj] = {
            "perception_cosine_mean": float(np.mean(bundle.perception_cosines)),
            "imagery_cosine_mean": float(np.mean(bundle.imagery_cosines)),
            "transfer_ratio": float(
                np.mean(bundle.imagery_cosines) / max(np.mean(bundle.perception_cosines), 1e-8)
            ),
            "n_perception": bundle.perception.shape[0],
            "n_imagery": bundle.imagery.shape[0],
        }
        logger.info(f"  {subj}: P={per_subject_stats[subj]['perception_cosine_mean']:.3f}, "
                     f"I={per_subject_stats[subj]['imagery_cosine_mean']:.3f}, "
                     f"ratio={per_subject_stats[subj]['transfer_ratio']:.3f}")

    # Compare degradation profiles
    profile_comparison = compare_degradation_profiles(profiles)

    # Adapter weight comparison
    adapter_comparison = {"available": False, "reason": "No checkpoints provided"}
    if adapter_checkpoints:
        adapter_comparison = compare_adapter_weights(adapter_checkpoints)

    # Second-order RSA
    so_rsa = second_order_rsa(bundles)

    results = {
        "per_subject_stats": per_subject_stats,
        "degradation_profile_comparison": profile_comparison,
        "adapter_weight_comparison": adapter_comparison,
        "second_order_rsa": so_rsa,
        "n_subjects": len(bundles),
    }

    logger.info(f"  Mean pairwise degradation correlation: "
                f"{profile_comparison['mean_pairwise_correlation']:.3f}")
    logger.info(f"  Perception-imagery consistency: "
                f"r={so_rsa['perception_imagery_consistency_r']:.3f}")

    return results
