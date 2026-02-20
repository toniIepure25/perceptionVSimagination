"""
Direction 9: Adversarial Reality Probing
=========================================

Trains a miniature GAN-like framework where a Generator learns the minimal
transform to make imagery embeddings indistinguishable from perception,
while a Discriminator tries to tell them apart.

This quantifies the exact "distance to reality" for each imagery trial
and produces a learned reality boundary — the computational analog of
the fusiform gyrus reality monitor.

References:
    Dijkstra et al. (2025). "A neural basis for distinguishing
    imagination from reality." Neuron.
    Goodfellow et al. (2014). "Generative Adversarial Nets." NeurIPS.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .core import EmbeddingBundle, _l2

logger = logging.getLogger(__name__)


class RealityDiscriminator(nn.Module):
    """
    Small MLP that classifies embeddings as perception (1) or imagery (0).

    Computational analog of the fusiform gyrus reality monitor: it learns
    to detect whether a decoded embedding came from perception or imagery
    based on representational features.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RealityGenerator(nn.Module):
    """
    Small residual MLP that transforms imagery embeddings toward the
    perception manifold with near-identity initialization.

    The perturbation needed to fool the discriminator quantifies how
    "far from reality" each imagery trial is.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Near-identity: residual starts near zero
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = self.fc1(h)
        h = self.gelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return residual + h


def train_adversarial_reality(
    bundle: EmbeddingBundle,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr_d: float = 1e-3,
    lr_g: float = 5e-4,
    hidden_dim: int = 256,
    device: str = "cpu",
    seed: int = 42,
) -> Dict:
    """
    Alternating adversarial training:
      1. Train discriminator to distinguish perception from imagery
      2. Train generator to fool discriminator (make imagery look like perception)

    Returns training dynamics, final discriminator accuracy, and per-trial
    perturbation distances.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    embed_dim = bundle.embed_dim
    D = RealityDiscriminator(embed_dim, hidden_dim).to(device)
    G = RealityGenerator(embed_dim, hidden_dim).to(device)

    opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    perc_t = torch.from_numpy(bundle.perception).float().to(device)
    imag_t = torch.from_numpy(bundle.imagery).float().to(device)

    n_perc = perc_t.shape[0]
    n_imag = imag_t.shape[0]

    history = {
        "d_loss": [], "g_loss": [],
        "d_acc_real": [], "d_acc_fake": [], "d_acc_total": [],
    }

    for epoch in range(n_epochs):
        # Shuffle
        perm_p = torch.randperm(n_perc, device=device)
        perm_i = torch.randperm(n_imag, device=device)

        n_batches = max(1, min(n_perc, n_imag) // batch_size)
        epoch_d_loss, epoch_g_loss = 0.0, 0.0
        epoch_d_correct_real, epoch_d_correct_fake, epoch_d_total = 0, 0, 0

        for b in range(n_batches):
            start = b * batch_size

            # Sample real (perception) and fake (imagery)
            p_idx = perm_p[start:start + batch_size]
            i_idx = perm_i[start % n_imag:(start % n_imag) + batch_size]
            if len(i_idx) == 0:
                continue

            real = perc_t[p_idx]
            fake_raw = imag_t[i_idx]
            bs = min(len(real), len(fake_raw))
            real, fake_raw = real[:bs], fake_raw[:bs]

            # --- Train Discriminator ---
            D.train()
            G.eval()
            opt_d.zero_grad()

            fake = G(fake_raw).detach()
            d_real = D(real)
            d_fake = D(fake)

            loss_d = (
                criterion(d_real, torch.ones_like(d_real)) +
                criterion(d_fake, torch.zeros_like(d_fake))
            ) / 2
            loss_d.backward()
            opt_d.step()

            epoch_d_loss += loss_d.item()
            epoch_d_correct_real += (d_real > 0).sum().item()
            epoch_d_correct_fake += (d_fake < 0).sum().item()
            epoch_d_total += bs * 2

            # --- Train Generator ---
            D.eval()
            G.train()
            opt_g.zero_grad()

            fake = G(fake_raw)
            d_fake = D(fake)
            loss_g = criterion(d_fake, torch.ones_like(d_fake))
            loss_g.backward()
            opt_g.step()

            epoch_g_loss += loss_g.item()

        n_b = max(n_batches, 1)
        d_acc_real = epoch_d_correct_real / max(epoch_d_total // 2, 1)
        d_acc_fake = epoch_d_correct_fake / max(epoch_d_total // 2, 1)
        d_acc_total = (epoch_d_correct_real + epoch_d_correct_fake) / max(epoch_d_total, 1)

        history["d_loss"].append(epoch_d_loss / n_b)
        history["g_loss"].append(epoch_g_loss / n_b)
        history["d_acc_real"].append(d_acc_real)
        history["d_acc_fake"].append(d_acc_fake)
        history["d_acc_total"].append(d_acc_total)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch + 1}/{n_epochs} — D_loss: {history['d_loss'][-1]:.4f}, "
                f"G_loss: {history['g_loss'][-1]:.4f}, D_acc: {d_acc_total:.4f}"
            )

    return D, G, history


def compute_perturbation_distance(
    bundle: EmbeddingBundle,
    generator: nn.Module,
    device: str = "cpu",
) -> Dict:
    """
    For each imagery trial, compute the distance of the generator's
    transformation — how much the embedding needed to be perturbed
    to cross the "reality threshold."

    Small distance = imagery was already close to perception.
    Large distance = imagery representation was far from reality.
    """
    generator.eval()
    imag_t = torch.from_numpy(bundle.imagery).float().to(device)

    with torch.no_grad():
        transformed = generator(imag_t).cpu().numpy()

    original = bundle.imagery

    # Per-trial perturbation metrics
    l2_dist = np.linalg.norm(transformed - original, axis=1)
    cosine_shift = 1.0 - np.sum(
        _l2(transformed) * _l2(original), axis=1
    )
    norm_change = np.linalg.norm(transformed, axis=1) - np.linalg.norm(original, axis=1)

    return {
        "l2_distance": l2_dist,
        "cosine_shift": cosine_shift,
        "norm_change": norm_change,
        "mean_l2": float(np.mean(l2_dist)),
        "std_l2": float(np.std(l2_dist)),
        "mean_cosine_shift": float(np.mean(cosine_shift)),
        "mean_norm_change": float(np.mean(norm_change)),
    }


def analyze_discriminator_features(
    discriminator: nn.Module,
    bundle: EmbeddingBundle,
    device: str = "cpu",
    top_k: int = 20,
) -> Dict:
    """
    Extract which embedding dimensions the discriminator relies on most
    via gradient-based feature importance.
    """
    discriminator.eval()

    all_emb = np.vstack([bundle.perception, bundle.imagery])
    emb_t = torch.from_numpy(all_emb).float().to(device).requires_grad_(True)

    logits = discriminator(emb_t)
    logits.sum().backward()

    grad = emb_t.grad.cpu().numpy()
    importance = np.mean(np.abs(grad), axis=0)

    top_dims = np.argsort(importance)[::-1][:top_k]

    return {
        "dimension_importance": importance.tolist(),
        "top_dimensions": top_dims.tolist(),
        "top_importance_values": importance[top_dims].tolist(),
        "importance_gini": float(_gini(importance)),
    }


def _gini(x: np.ndarray) -> float:
    """Gini coefficient measuring how concentrated feature importance is."""
    x = np.abs(x)
    if x.sum() == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n)


def analyze_adversarial_reality(
    bundle: EmbeddingBundle,
    n_epochs: int = 100,
    batch_size: int = 64,
    hidden_dim: int = 256,
    device: str = "cpu",
) -> Dict:
    """
    Full adversarial reality probing analysis.

    Trains discriminator + generator, then analyzes perturbation distances,
    feature importance, and training dynamics.
    """
    logger.info("Running Adversarial Reality Probing analysis...")

    D, G, history = train_adversarial_reality(
        bundle, n_epochs=n_epochs, batch_size=batch_size,
        hidden_dim=hidden_dim, device=device,
    )

    # Perturbation analysis
    logger.info("  Computing perturbation distances...")
    perturbation = compute_perturbation_distance(bundle, G, device)
    logger.info(f"    Mean L2 perturbation: {perturbation['mean_l2']:.4f}")
    logger.info(f"    Mean cosine shift: {perturbation['mean_cosine_shift']:.4f}")
    logger.info(f"    Mean norm change: {perturbation['mean_norm_change']:.4f}")

    # Feature importance
    logger.info("  Analyzing discriminator features...")
    feat_importance = analyze_discriminator_features(D, bundle, device)
    logger.info(f"    Importance Gini: {feat_importance['importance_gini']:.4f}")

    # Convergence analysis
    initial_d_acc = history["d_acc_total"][0] if history["d_acc_total"] else 0.5
    final_d_acc = history["d_acc_total"][-1] if history["d_acc_total"] else 0.5
    min_d_acc = min(history["d_acc_total"]) if history["d_acc_total"] else 0.5

    convergence = {
        "initial_d_accuracy": float(initial_d_acc),
        "final_d_accuracy": float(final_d_acc),
        "min_d_accuracy": float(min_d_acc),
        "generator_succeeded": bool(final_d_acc < 0.6),
        "n_epochs_to_equilibrium": int(
            next((i for i, a in enumerate(history["d_acc_total"]) if a < 0.55), n_epochs)
        ),
    }
    logger.info(f"  Convergence: D_acc {initial_d_acc:.4f} -> {final_d_acc:.4f}")
    logger.info(f"  Generator succeeded: {convergence['generator_succeeded']}")

    results = {
        "training_dynamics": {
            "d_loss": history["d_loss"],
            "g_loss": history["g_loss"],
            "d_acc_total": history["d_acc_total"],
            "d_acc_real": history["d_acc_real"],
            "d_acc_fake": history["d_acc_fake"],
        },
        "perturbation": {
            "mean_l2": perturbation["mean_l2"],
            "std_l2": perturbation["std_l2"],
            "mean_cosine_shift": perturbation["mean_cosine_shift"],
            "mean_norm_change": perturbation["mean_norm_change"],
            "per_trial_l2": perturbation["l2_distance"].tolist(),
        },
        "feature_importance": {
            "top_dimensions": feat_importance["top_dimensions"],
            "top_importance_values": feat_importance["top_importance_values"],
            "importance_gini": feat_importance["importance_gini"],
        },
        "convergence": convergence,
    }

    return results
