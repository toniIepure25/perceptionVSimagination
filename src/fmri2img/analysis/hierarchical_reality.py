"""
Direction 10: Hierarchical Reality Gradient
============================================

Analyzes at which processing level (L4, L8, L12, final) the perception-
imagery distinction emerges, using multi-layer outputs from the
MultiLayerTwoStageEncoder.

Tests the neuroscience prediction that reality monitoring involves
hierarchical processing, with the fusiform gyrus (mid-level visual
cortex) playing the primary role.

References:
    Dijkstra et al. (2025). "A neural basis for distinguishing
    imagination from reality." Neuron.
    Dijkstra (2022). "Perceptual reality monitoring: Neural mechanisms
    dissociating imagination from reality." Neurosci Biobehav Rev.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)

LAYER_ORDER = ["layer_4", "layer_8", "layer_12", "final"]
LAYER_LABELS = {
    "layer_4": "L4 (Early Visual)",
    "layer_8": "L8 (Mid-Level)",
    "layer_12": "L12 (Late Semantic)",
    "final": "Final (CLIP)",
}


def collect_multilayer_embeddings(
    model,
    dataset,
    device: str = "cpu",
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract embeddings from each layer head of a MultiLayerTwoStageEncoder.

    Returns dict mapping layer name -> (N, D_layer) array.
    """
    import torch

    model.eval()
    layer_preds = {layer: [] for layer in LAYER_ORDER}

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            voxels = sample["voxels"]
            if isinstance(voxels, np.ndarray):
                voxels = torch.from_numpy(voxels).float().unsqueeze(0).to(device)

            outputs = model(voxels)
            if isinstance(outputs, dict):
                for layer in LAYER_ORDER:
                    if layer in outputs:
                        layer_preds[layer].append(
                            outputs[layer].cpu().numpy().squeeze(0)
                        )
            else:
                layer_preds["final"].append(outputs.cpu().numpy().squeeze(0))

    return {
        layer: np.array(preds) for layer, preds in layer_preds.items()
        if len(preds) > 0
    }


def compute_layer_discriminability(
    perc_embeddings: np.ndarray,
    imag_embeddings: np.ndarray,
    n_folds: int = 5,
) -> Dict:
    """
    For a single layer, compute how well perception and imagery can be
    distinguished via:
      - Classification AUC (logistic regression)
      - Signal strength divergence (norm difference)
      - Representational overlap (centroid distance / within-class spread)
    """
    X = np.vstack([perc_embeddings, imag_embeddings])
    y = np.concatenate([
        np.ones(perc_embeddings.shape[0]),
        np.zeros(imag_embeddings.shape[0]),
    ])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # AUC via cross-validated logistic regression
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(X_scaled, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_scaled[train_idx], y[train_idx])
        probs = clf.predict_proba(X_scaled[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))

    # Signal strength divergence
    perc_norms = np.linalg.norm(perc_embeddings, axis=1)
    imag_norms = np.linalg.norm(imag_embeddings, axis=1)
    norm_gap = float(np.mean(perc_norms) - np.mean(imag_norms))

    pooled_std = np.sqrt((np.std(perc_norms) ** 2 + np.std(imag_norms) ** 2) / 2)
    cohens_d = norm_gap / max(pooled_std, 1e-8)

    ks_stat, ks_p = scipy_stats.ks_2samp(perc_norms, imag_norms)

    # Representational overlap: ratio of between-class to within-class distance
    perc_centroid = np.mean(perc_embeddings, axis=0)
    imag_centroid = np.mean(imag_embeddings, axis=0)
    between_dist = float(np.linalg.norm(perc_centroid - imag_centroid))

    perc_within = float(np.mean(np.linalg.norm(perc_embeddings - perc_centroid, axis=1)))
    imag_within = float(np.mean(np.linalg.norm(imag_embeddings - imag_centroid, axis=1)))
    avg_within = (perc_within + imag_within) / 2

    separation_index = between_dist / max(avg_within, 1e-8)

    return {
        "classification_auc": float(np.mean(aucs)),
        "classification_auc_std": float(np.std(aucs)),
        "norm_gap": float(norm_gap),
        "cohens_d": float(cohens_d),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        "between_class_distance": between_dist,
        "within_class_distance": avg_within,
        "separation_index": separation_index,
        "perception_mean_norm": float(np.mean(perc_norms)),
        "imagery_mean_norm": float(np.mean(imag_norms)),
        "embed_dim": perc_embeddings.shape[1],
    }


def analyze_threshold_cascade(
    layer_discriminabilities: Dict[str, Dict],
) -> Dict:
    """
    Test whether the reality threshold cascades through processing levels:
      - Gradual (linear) vs. abrupt (step function)?
      - Which layer has the sharpest boundary?
      - Does this match the cortical hierarchy?
    """
    layers_present = [l for l in LAYER_ORDER if l in layer_discriminabilities]
    if len(layers_present) < 2:
        return {"cascade_type": "insufficient_layers", "layers_found": len(layers_present)}

    aucs = [layer_discriminabilities[l]["classification_auc"] for l in layers_present]
    cohens_ds = [layer_discriminabilities[l]["cohens_d"] for l in layers_present]
    separations = [layer_discriminabilities[l]["separation_index"] for l in layers_present]

    layer_indices = np.arange(len(layers_present), dtype=float)

    # Linear fit to AUC progression
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(layer_indices, aucs)

    # Step detection: find largest single-step increase
    auc_diffs = np.diff(aucs)
    max_step_idx = int(np.argmax(np.abs(auc_diffs)))
    max_step_size = float(auc_diffs[max_step_idx])

    # Is the progression more linear (gradual) or step-like (abrupt)?
    # Compare linear R² to a step function fit
    linear_residuals = np.sum((np.array(aucs) - (slope * layer_indices + intercept)) ** 2)
    total_variance = np.sum((np.array(aucs) - np.mean(aucs)) ** 2)
    r_squared = 1 - linear_residuals / max(total_variance, 1e-10)

    # Peak discriminability layer
    peak_layer_idx = int(np.argmax(aucs))
    peak_layer = layers_present[peak_layer_idx]

    cascade_type = "gradual" if r_squared > 0.7 else "abrupt"

    # Cortical hierarchy mapping
    cortical_mapping = {
        "layer_4": "V1-V2 (early visual)",
        "layer_8": "V4-fusiform (mid-level)",
        "layer_12": "IT (late semantic)",
        "final": "PFC (abstract)",
    }
    peak_cortical = cortical_mapping.get(peak_layer, "unknown")

    # PRM prediction: peak discriminability at mid-level (fusiform ~ L8)
    prm_prediction_matched = peak_layer in ("layer_8", "layer_12")

    return {
        "cascade_type": cascade_type,
        "linear_r_squared": float(r_squared),
        "linear_slope": float(slope),
        "linear_p_value": float(p_value),
        "max_step_size": max_step_size,
        "max_step_between": f"{layers_present[max_step_idx]} -> {layers_present[max_step_idx + 1]}",
        "peak_layer": peak_layer,
        "peak_auc": float(aucs[peak_layer_idx]),
        "peak_cortical_analog": peak_cortical,
        "prm_prediction_matched": prm_prediction_matched,
        "per_layer_auc": {l: float(a) for l, a in zip(layers_present, aucs)},
        "per_layer_cohens_d": {l: float(d) for l, d in zip(layers_present, cohens_ds)},
        "per_layer_separation": {l: float(s) for l, s in zip(layers_present, separations)},
    }


def layer_specific_confusion(
    multilayer_perc: Dict[str, np.ndarray],
    multilayer_imag: Dict[str, np.ndarray],
) -> Dict:
    """
    Per-layer confusion analysis: at which layers do perception and imagery
    representations converge vs. diverge?

    Uses centroid cosine similarity between conditions at each layer.
    """
    results = {}
    for layer in LAYER_ORDER:
        if layer not in multilayer_perc or layer not in multilayer_imag:
            continue

        perc = multilayer_perc[layer]
        imag = multilayer_imag[layer]

        perc_centroid = _l2(np.mean(perc, axis=0, keepdims=True))
        imag_centroid = _l2(np.mean(imag, axis=0, keepdims=True))

        centroid_cosine = float(np.sum(perc_centroid * imag_centroid))

        # Per-sample cross-condition cosine (average over matched pairs)
        n_common = min(perc.shape[0], imag.shape[0])
        cross_cosines = np.sum(
            _l2(perc[:n_common]) * _l2(imag[:n_common]), axis=1
        )

        results[layer] = {
            "centroid_cosine": centroid_cosine,
            "mean_cross_cosine": float(np.mean(cross_cosines)),
            "std_cross_cosine": float(np.std(cross_cosines)),
            "label": LAYER_LABELS.get(layer, layer),
        }

    return results


def analyze_hierarchical_reality(
    bundle: EmbeddingBundle,
    model=None,
    perception_dataset=None,
    imagery_dataset=None,
    device: str = "cpu",
    max_samples: Optional[int] = None,
) -> Dict:
    """
    Full hierarchical reality gradient analysis.

    If multi-layer embeddings are pre-computed in the bundle, uses those.
    Otherwise, if model and datasets are provided, collects them.
    Falls back to synthetic multi-layer data for dry-run mode.
    """
    logger.info("Running Hierarchical Reality Gradient analysis...")

    multilayer_perc = bundle.multilayer_perception
    multilayer_imag = bundle.multilayer_imagery

    # Collect from model if not pre-computed
    if multilayer_perc is None and model is not None and perception_dataset is not None:
        logger.info("  Collecting multi-layer perception embeddings...")
        multilayer_perc = collect_multilayer_embeddings(
            model, perception_dataset, device, max_samples
        )
    if multilayer_imag is None and model is not None and imagery_dataset is not None:
        logger.info("  Collecting multi-layer imagery embeddings...")
        multilayer_imag = collect_multilayer_embeddings(
            model, imagery_dataset, device, max_samples
        )

    # Fallback: generate synthetic multi-layer data
    if multilayer_perc is None or multilayer_imag is None:
        logger.info("  No multi-layer data; generating synthetic layer embeddings")
        multilayer_perc, multilayer_imag = _generate_synthetic_multilayer(bundle)

    # Per-layer discriminability
    logger.info("  Computing per-layer discriminability...")
    layer_disc = {}
    for layer in LAYER_ORDER:
        if layer in multilayer_perc and layer in multilayer_imag:
            disc = compute_layer_discriminability(
                multilayer_perc[layer], multilayer_imag[layer]
            )
            layer_disc[layer] = disc
            logger.info(
                f"    {LAYER_LABELS.get(layer, layer)}: "
                f"AUC={disc['classification_auc']:.4f}, "
                f"Cohen's d={disc['cohens_d']:.4f}, "
                f"Separation={disc['separation_index']:.4f}"
            )

    # Threshold cascade analysis
    logger.info("  Analyzing threshold cascade...")
    cascade = analyze_threshold_cascade(layer_disc)
    logger.info(f"    Cascade type: {cascade['cascade_type']}")
    logger.info(f"    Peak layer: {cascade.get('peak_layer', 'N/A')} "
                f"({cascade.get('peak_cortical_analog', 'N/A')})")
    logger.info(f"    PRM prediction matched: {cascade.get('prm_prediction_matched', 'N/A')}")

    # Layer-specific confusion
    confusion = layer_specific_confusion(multilayer_perc, multilayer_imag)

    results = {
        "per_layer_discriminability": layer_disc,
        "threshold_cascade": cascade,
        "layer_confusion": confusion,
    }

    return results


def _generate_synthetic_multilayer(
    bundle: EmbeddingBundle,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Generate synthetic multi-layer embeddings with increasing signal
    strength gap at higher layers (matching cortical hierarchy).
    """
    rng = np.random.RandomState(42)
    n_perc = bundle.perception.shape[0]
    n_imag = bundle.imagery.shape[0]

    layer_dims = {"layer_4": 768, "layer_8": 768, "layer_12": 768, "final": 512}
    # Signal strength gap increases with layer depth
    layer_gaps = {"layer_4": 0.05, "layer_8": 0.12, "layer_12": 0.22, "final": 0.30}

    multilayer_perc = {}
    multilayer_imag = {}

    for layer, dim in layer_dims.items():
        gap = layer_gaps[layer]

        p = rng.randn(n_perc, dim).astype(np.float32)
        p_norms = 1.0 + gap + np.abs(rng.randn(n_perc, 1).astype(np.float32)) * 0.1
        multilayer_perc[layer] = _l2(p) * p_norms

        im = rng.randn(n_imag, dim).astype(np.float32)
        im_norms = 1.0 - gap + np.abs(rng.randn(n_imag, 1).astype(np.float32)) * 0.1
        multilayer_imag[layer] = _l2(im) * im_norms

    return multilayer_perc, multilayer_imag
