"""
Direction 7: Computational Reality Monitor (CRM)
=================================================

Implements Dijkstra's Perceptual Reality Monitoring (PRM) theory
computationally: trains classifiers to distinguish perception from
imagery embeddings using signal-strength features vs. content features.

Tests whether signal strength alone (L2 norm, decoding confidence,
entropy) is sufficient to distinguish conditions, as the fusiform
gyrus reality-monitoring mechanism predicts.

References:
    Dijkstra et al. (2025). "A neural basis for distinguishing
    imagination from reality." Neuron.
    Dijkstra & Fleming (2023). "Subjective signal strength distinguishes
    reality from imagination." Nature Communications.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)


def compute_signal_strength_features(
    embeddings: np.ndarray,
    targets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract signal-strength proxy features from decoded embeddings.

    Features (per trial):
        0: L2 norm (overall activation magnitude)
        1: Cosine similarity to target (decoding confidence)
        2: Embedding entropy (information spread across dimensions)
        3: Max absolute activation (peak signal)
        4: Kurtosis (peakedness of activation distribution)

    These proxy the neural signal strength that the fusiform gyrus
    monitors for reality judgments (Dijkstra et al., 2025).
    """
    n = embeddings.shape[0]
    features = np.zeros((n, 5), dtype=np.float32)

    features[:, 0] = np.linalg.norm(embeddings, axis=1)

    if targets is not None:
        e_norm = _l2(embeddings)
        t_norm = _l2(targets)
        features[:, 1] = np.sum(e_norm * t_norm, axis=1)
    else:
        features[:, 1] = features[:, 0]  # fallback: use norm

    # Entropy of squared (softmax-like) activation distribution
    abs_emb = np.abs(embeddings) + 1e-10
    probs = abs_emb / abs_emb.sum(axis=1, keepdims=True)
    features[:, 2] = -np.sum(probs * np.log(probs), axis=1)

    features[:, 3] = np.max(np.abs(embeddings), axis=1)

    for i in range(n):
        features[i, 4] = scipy_stats.kurtosis(embeddings[i])

    return features


def _build_classification_data(
    bundle: EmbeddingBundle,
    feature_mode: str = "signal_strength",
    n_pca_content: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and label vector y for reality classification.

    feature_mode:
        "signal_strength" — only signal-strength proxy features
        "content" — PCA-reduced embedding content
        "combined" — both feature sets concatenated
    """
    perc_ss = compute_signal_strength_features(bundle.perception, bundle.perception_targets)
    imag_ss = compute_signal_strength_features(bundle.imagery, bundle.imagery_targets)

    if feature_mode == "signal_strength":
        X = np.vstack([perc_ss, imag_ss])

    elif feature_mode == "content":
        all_emb = np.vstack([bundle.perception, bundle.imagery])
        n_components = min(n_pca_content, all_emb.shape[0], all_emb.shape[1])
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(all_emb)

    elif feature_mode == "combined":
        all_emb = np.vstack([bundle.perception, bundle.imagery])
        n_components = min(n_pca_content, all_emb.shape[0], all_emb.shape[1])
        pca = PCA(n_components=n_components)
        content = pca.fit_transform(all_emb)
        ss = np.vstack([perc_ss, imag_ss])
        X = np.hstack([ss, content])

    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    y = np.concatenate([
        np.ones(bundle.perception.shape[0]),
        np.zeros(bundle.imagery.shape[0]),
    ])
    return X, y


def train_reality_classifier(
    bundle: EmbeddingBundle,
    feature_mode: str = "signal_strength",
    n_folds: int = 5,
    n_pca_content: int = 50,
) -> Dict:
    """
    Train a logistic regression classifier to distinguish perception
    from imagery, using cross-validation.

    Returns AUC, accuracy, and feature importances.
    """
    X, y = _build_classification_data(bundle, feature_mode, n_pca_content)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs, accuracies = [], []
    all_probs = np.zeros(len(y))
    coefs = []

    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[test_idx])[:, 1]
        all_probs[test_idx] = probs

        auc = roc_auc_score(y[test_idx], probs)
        acc = clf.score(X[test_idx], y[test_idx])
        aucs.append(auc)
        accuracies.append(acc)
        coefs.append(clf.coef_[0])

    mean_coefs = np.mean(coefs, axis=0)

    fpr, tpr, thresholds = roc_curve(y, all_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = float(thresholds[optimal_idx])

    # Feature importance labels
    if feature_mode == "signal_strength":
        feature_names = ["l2_norm", "cosine_confidence", "entropy", "max_activation", "kurtosis"]
    elif feature_mode == "content":
        feature_names = [f"pc_{i}" for i in range(X.shape[1])]
    else:
        ss_names = ["l2_norm", "cosine_confidence", "entropy", "max_activation", "kurtosis"]
        pc_names = [f"pc_{i}" for i in range(X.shape[1] - 5)]
        feature_names = ss_names + pc_names

    importance = {name: float(abs(c)) for name, c in zip(feature_names, mean_coefs)}

    return {
        "feature_mode": feature_mode,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "optimal_threshold": optimal_threshold,
        "feature_importance": importance,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "n_folds": n_folds,
    }


def compute_reality_threshold(bundle: EmbeddingBundle) -> Dict:
    """
    Estimate the optimal signal-strength threshold separating perception
    from imagery, analogous to Dijkstra's fusiform reality threshold.

    Uses L2 norm as the primary signal strength measure, then finds
    the threshold via ROC analysis.
    """
    perc_norms = bundle.perception_norms
    imag_norms = bundle.imagery_norms

    all_norms = np.concatenate([perc_norms, imag_norms])
    labels = np.concatenate([np.ones(len(perc_norms)), np.zeros(len(imag_norms))])

    fpr, tpr, thresholds = roc_curve(labels, all_norms)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = float(thresholds[optimal_idx])
    auc = roc_auc_score(labels, all_norms)

    # Classification at threshold
    perc_above = float(np.mean(perc_norms >= threshold))
    imag_above = float(np.mean(imag_norms >= threshold))

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        (np.std(perc_norms) ** 2 + np.std(imag_norms) ** 2) / 2
    )
    cohens_d = float(
        (np.mean(perc_norms) - np.mean(imag_norms)) / max(pooled_std, 1e-8)
    )

    ks_stat, ks_p = scipy_stats.ks_2samp(perc_norms, imag_norms)

    return {
        "optimal_threshold": threshold,
        "auc_norm_only": float(auc),
        "perception_mean_norm": float(np.mean(perc_norms)),
        "perception_std_norm": float(np.std(perc_norms)),
        "imagery_mean_norm": float(np.mean(imag_norms)),
        "imagery_std_norm": float(np.std(imag_norms)),
        "cohens_d": cohens_d,
        "perception_above_threshold": perc_above,
        "imagery_above_threshold": imag_above,
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
    }


def analyze_misclassifications(
    bundle: EmbeddingBundle,
    classifier_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Analyze which trials are misclassified by the reality monitor.

    Tests PRM prediction: misclassified imagery trials should have
    the highest signal strength (vivid imagery that "fools" the
    reality monitor), and misclassified perception trials should
    have the lowest signal strength (weak perception).
    """
    n_perc = bundle.perception.shape[0]
    perc_probs = classifier_probs[:n_perc]
    imag_probs = classifier_probs[n_perc:]

    # Misclassified imagery: predicted as perception (prob > threshold)
    imag_misclassified = imag_probs > threshold
    # Misclassified perception: predicted as imagery (prob < threshold)
    perc_misclassified = perc_probs < threshold

    imag_norms = bundle.imagery_norms
    perc_norms = bundle.perception_norms

    results = {
        "n_imagery_misclassified": int(np.sum(imag_misclassified)),
        "n_perception_misclassified": int(np.sum(perc_misclassified)),
        "imagery_misclass_rate": float(np.mean(imag_misclassified)),
        "perception_misclass_rate": float(np.mean(perc_misclassified)),
    }

    # PRM prediction: misclassified imagery has higher norms
    if np.sum(imag_misclassified) > 0 and np.sum(~imag_misclassified) > 0:
        mc_norms = imag_norms[imag_misclassified]
        correct_norms = imag_norms[~imag_misclassified]
        stat, p = scipy_stats.mannwhitneyu(
            mc_norms, correct_norms, alternative="greater"
        )
        results["imagery_misclass_higher_norm_p"] = float(p)
        results["imagery_misclass_mean_norm"] = float(np.mean(mc_norms))
        results["imagery_correct_mean_norm"] = float(np.mean(correct_norms))
    else:
        results["imagery_misclass_higher_norm_p"] = float("nan")

    # PRM prediction: misclassified perception has lower norms
    if np.sum(perc_misclassified) > 0 and np.sum(~perc_misclassified) > 0:
        mc_norms = perc_norms[perc_misclassified]
        correct_norms = perc_norms[~perc_misclassified]
        stat, p = scipy_stats.mannwhitneyu(
            mc_norms, correct_norms, alternative="less"
        )
        results["perception_misclass_lower_norm_p"] = float(p)
        results["perception_misclass_mean_norm"] = float(np.mean(mc_norms))
        results["perception_correct_mean_norm"] = float(np.mean(correct_norms))
    else:
        results["perception_misclass_lower_norm_p"] = float("nan")

    # Category breakdown of misclassifications
    category_misclass = defaultdict(lambda: {"total": 0, "misclassified": 0})
    for i, meta in enumerate(bundle.imagery_meta):
        stype = meta.get("stimulus_type", "unknown")
        category_misclass[stype]["total"] += 1
        if imag_misclassified[i]:
            category_misclass[stype]["misclassified"] += 1

    results["category_misclassification"] = {
        k: {
            "total": v["total"],
            "misclassified": v["misclassified"],
            "rate": v["misclassified"] / max(v["total"], 1),
        }
        for k, v in category_misclass.items()
    }

    return results


def analyze_reality_monitor(
    bundle: EmbeddingBundle,
    n_folds: int = 5,
    n_pca_content: int = 50,
) -> Dict:
    """
    Full Computational Reality Monitor analysis.

    Trains classifiers in three modes (signal_strength, content, combined),
    estimates the reality threshold, and analyzes misclassification patterns.
    """
    logger.info("Running Computational Reality Monitor (CRM) analysis...")

    # Train classifiers in all three modes
    classifiers = {}
    for mode in ["signal_strength", "content", "combined"]:
        logger.info(f"  Training {mode} classifier...")
        classifiers[mode] = train_reality_classifier(
            bundle, feature_mode=mode, n_folds=n_folds, n_pca_content=n_pca_content
        )
        logger.info(f"    AUC = {classifiers[mode]['mean_auc']:.4f} "
                     f"± {classifiers[mode]['std_auc']:.4f}")

    # Reality threshold from signal strength (norm)
    logger.info("  Estimating reality threshold...")
    threshold_results = compute_reality_threshold(bundle)
    logger.info(f"    Optimal threshold: {threshold_results['optimal_threshold']:.4f}")
    logger.info(f"    AUC (norm only): {threshold_results['auc_norm_only']:.4f}")
    logger.info(f"    Cohen's d: {threshold_results['cohens_d']:.4f}")

    # Misclassification analysis using the combined classifier's probabilities
    X_combined, y = _build_classification_data(bundle, "combined", n_pca_content)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_scaled, y)
    all_probs = clf.predict_proba(X_scaled)[:, 1]

    misclass = analyze_misclassifications(bundle, all_probs)
    logger.info(f"  Imagery misclassification rate: {misclass['imagery_misclass_rate']:.4f}")
    logger.info(f"  Perception misclassification rate: {misclass['perception_misclass_rate']:.4f}")

    # Summary: does signal strength alone rival content-based classification?
    ss_auc = classifiers["signal_strength"]["mean_auc"]
    content_auc = classifiers["content"]["mean_auc"]
    combined_auc = classifiers["combined"]["mean_auc"]

    results = {
        "classifiers": classifiers,
        "reality_threshold": threshold_results,
        "misclassification_analysis": misclass,
        "summary": {
            "signal_strength_auc": float(ss_auc),
            "content_auc": float(content_auc),
            "combined_auc": float(combined_auc),
            "signal_strength_sufficient": bool(ss_auc > 0.5 * (content_auc + 0.5)),
            "prm_theory_supported": bool(
                threshold_results["cohens_d"] > 0.2
                and threshold_results["ks_p_value"] < 0.05
            ),
        },
    }

    logger.info(f"  PRM theory supported: {results['summary']['prm_theory_supported']}")
    return results
