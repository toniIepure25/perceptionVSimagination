"""
External Model Loader — FMRI2images Bridge
===========================================

Loads checkpoints from the separate FMRI2images project and wraps them in a
compatible interface for cross-project comparison in perception-vs-imagery
analyses.

FMRI2images Architecture:
    - Encoder: 4-layer residual MLP [8192, 8192, 4096, 2048]
    - Decoder: Shared backbone → vMF heads (mu 328960-d, kappa 1-d)
    - Input: 15,724 raw voxels (nsdgeneral ROI)
    - CLIP: ViT-bigG/14 (LAION-2B), 1280-d × 257 tokens
    - Training: vMF-NCE + SoftCLIP + MixCo + EMA, bf16, queue 1024

This module:
    1. Reconstructs the encoder architecture from checkpoint weights
    2. Loads state_dict (optionally from EMA shadow)
    3. Provides a forward() that maps raw voxels → latent (2048-d)
    4. Optionally applies the decoder mu-head to get 328960-d bigG tokens
    5. Extracts CLS-equivalent (mean-pool tokens) for cross-model comparison

Usage:
    loader = ExternalModelLoader(
        checkpoint_path="/path/to/checkpoint_best.pt",
        device="cuda",
    )
    # Get 2048-d latent
    latent = loader.encode(raw_voxels)         # (B, 2048)
    # Get bigG token predictions
    tokens = loader.decode_tokens(raw_voxels)  # (B, 257, 1280)
    # Get mean-pooled 1280-d vector (for cosine comparison)
    cls = loader.predict_cls(raw_voxels)        # (B, 1280)

References:
    - FMRI2images project: /home/jovyan/work/FMRI2images/
    - Best checkpoint: experimental_results/N1v27a_bigg_tokens/subj01/checkpoint_best.pt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture reconstruction from checkpoint shapes
# ---------------------------------------------------------------------------


@dataclass
class ExternalModelConfig:
    """Configuration inferred from an FMRI2images checkpoint."""

    input_dim: int = 15724
    encoder_dims: Tuple[int, ...] = (8192, 8192, 4096, 2048)
    decoder_backbone_dim: int = 2048
    token_output_dim: int = 328960  # 257 tokens × 1280
    n_tokens: int = 257
    token_dim: int = 1280
    use_ema: bool = True
    subject: str = "subj01"


class _ResidualBlock(nn.Module):
    """Residual block matching FMRI2images architecture.
    
    Internal structure (from checkpoint keys):
        net.0: LayerNorm(dim)
        net.1: Linear(dim, dim)
        net.2: GELU()          (no params)
        net.3: Dropout()       (no params)
        net.4: Linear(dim, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _build_encoder(config: ExternalModelConfig) -> nn.Sequential:
    """
    Reconstruct the FMRI2images encoder from config.

    Architecture (inferred from checkpoint weight keys, sorted by index):
        0: Linear(input_dim, dims[0])      # input projection
        1: LayerNorm(dims[0])
        2: GELU()                          # no params
        3: Dropout()                       # no params
        4: ResidualBlock(dims[0])          # .net = [LN, Linear, GELU, Drop, Linear]
        ---
        5: Linear(dims[0], dims[1])        # transition (may be same dim)
        6: LayerNorm(dims[1])
        7: GELU()
        8: Dropout()
        9: ResidualBlock(dims[1])
        ---
        10: Linear(dims[1], dims[2])
        11: LayerNorm(dims[2])
        12: GELU()
        13: Dropout()
        14: ResidualBlock(dims[2])
        ---
        15: Linear(dims[2], dims[3])
        16: LayerNorm(dims[3])
        17: GELU()
        18: Dropout()
        19: ResidualBlock(dims[3])
    """
    layers: list[nn.Module] = []
    dims = config.encoder_dims

    # First block: input projection
    prev_dim = config.input_dim
    for dim in dims:
        layers.append(nn.Linear(prev_dim, dim))     # Linear
        layers.append(nn.LayerNorm(dim))             # LayerNorm
        layers.append(nn.GELU())                     # GELU
        layers.append(nn.Dropout(0.0))               # Dropout
        layers.append(_ResidualBlock(dim))            # ResBlock
        prev_dim = dim

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Checkpoint introspection
# ---------------------------------------------------------------------------


def infer_config_from_checkpoint(state_dict: Dict[str, torch.Tensor]) -> ExternalModelConfig:
    """
    Infer ExternalModelConfig from the shapes in a state_dict.

    Looks for:
        encoder.encoder.0.weight → (first_hidden, input_dim)
        decoder.mu_head.weight   → (token_output_dim, last_hidden)
    """
    config = ExternalModelConfig()

    # Input dimension from first linear layer
    for key in sorted(state_dict.keys()):
        if "encoder.encoder.0.weight" in key:
            shape = state_dict[key].shape
            config = ExternalModelConfig(
                input_dim=shape[1],
                encoder_dims=config.encoder_dims,
            )
            break

    # Token output dimension from mu_head
    for key in sorted(state_dict.keys()):
        if "mu_head.weight" in key:
            shape = state_dict[key].shape
            n_out = shape[0]
            # Try to infer n_tokens × token_dim
            if n_out % 257 == 0:
                token_dim = n_out // 257
                config = ExternalModelConfig(
                    input_dim=config.input_dim,
                    encoder_dims=config.encoder_dims,
                    token_output_dim=n_out,
                    n_tokens=257,
                    token_dim=token_dim,
                )
            break

    return config


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------


class ExternalModelLoader:
    """
    Load and wrap an FMRI2images checkpoint for cross-project comparison.

    This class handles:
        1. Loading a checkpoint (with optional EMA shadow weights)
        2. Reconstructing the encoder architecture from weight shapes
        3. Providing encode / decode / predict_cls methods
        4. Device and dtype management

    The loader does NOT require the FMRI2images codebase to be importable.
    It reconstructs the architecture purely from checkpoint weight shapes.

    Args:
        checkpoint_path: Path to checkpoint_best.pt (or any .pt checkpoint)
        device: Target device (default: "cpu")
        dtype: Target dtype (default: torch.float32)
        use_ema: If True and EMA shadow exists, use those weights (default: True)
        config_override: Optional config to override auto-detection

    Example:
        >>> loader = ExternalModelLoader("checkpoint_best.pt", device="cuda")
        >>> voxels = torch.randn(4, 15724).cuda()
        >>> cls_pred = loader.predict_cls(voxels)  # (4, 1280)
        >>> print(cls_pred.shape)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_ema: bool = True,
        config_override: Optional[ExternalModelConfig] = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Expected at: experimental_results/N1v27a_bigg_tokens/subj01/checkpoint_best.pt"
            )

        logger.info("Loading FMRI2images checkpoint from %s", self.checkpoint_path)
        ckpt = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )

        # Extract metadata
        self.epoch = ckpt.get("epoch", -1)
        self.global_step = ckpt.get("global_step", -1)
        self.subject = ckpt.get("subject", "unknown")
        self.val_loss = ckpt.get("val_loss", float("nan"))
        self._raw_config = ckpt.get("config", {})
        self._model_config = ckpt.get("model_config", {})

        # Choose state_dict: EMA shadow or vanilla
        if use_ema and "ema_shadow" in ckpt and ckpt["ema_shadow"]:
            logger.info("Using EMA shadow weights (recommended for inference)")
            raw_sd = ckpt["ema_shadow"]
        elif "model_state_dict" in ckpt:
            raw_sd = ckpt["model_state_dict"]
        else:
            raise KeyError(
                "Checkpoint has neither 'ema_shadow' nor 'model_state_dict'"
            )

        # Infer config from weights
        self.config = config_override or infer_config_from_checkpoint(raw_sd)

        # Separate encoder and decoder weights
        encoder_sd = {}
        decoder_sd = {}
        for k, v in raw_sd.items():
            if k.startswith("encoder."):
                encoder_sd[k] = v
            elif k.startswith("decoder."):
                decoder_sd[k] = v

        # Build and load encoder
        self.encoder = _build_encoder(self.config)
        self._load_encoder_weights(encoder_sd)
        self.encoder.to(self.device, self.dtype).eval()

        # Build and load decoder (mu_head only — we skip kappa for inference)
        self.mu_head: Optional[nn.Module] = None
        self._load_decoder_weights(decoder_sd)

        logger.info(
            "FMRI2images model loaded: epoch=%d, step=%d, val_loss=%.4f, "
            "input=%d, latent=%d, tokens=%d×%d",
            self.epoch,
            self.global_step,
            self.val_loss,
            self.config.input_dim,
            self.config.encoder_dims[-1],
            self.config.n_tokens,
            self.config.token_dim,
        )

    # -----------------------------------------------------------------------
    # Weight loading helpers
    # -----------------------------------------------------------------------

    def _load_encoder_weights(self, encoder_sd: Dict[str, torch.Tensor]) -> None:
        """Load encoder weights with flexible key matching."""
        # Strip 'encoder.' prefix (may be double: encoder.encoder.X → encoder.X → X)
        stripped = {}
        for k, v in encoder_sd.items():
            clean = k
            # Remove leading 'encoder.' prefixes until we get bare layer names
            while clean.startswith("encoder."):
                clean = clean[len("encoder."):]
            stripped[clean] = v

        # Try direct load first
        missing, unexpected = self.encoder.load_state_dict(stripped, strict=False)
        if missing:
            logger.warning(
                "Encoder missing keys (may indicate architecture mismatch): %s",
                missing[:5],
            )
        if unexpected:
            logger.debug("Encoder unexpected keys (ignored): %s", unexpected[:5])

    def _load_decoder_weights(self, decoder_sd: Dict[str, torch.Tensor]) -> None:
        """Load decoder mu_head weights if available."""
        # Find mu_head keys, stripping all prefix layers
        mu_weight = None
        mu_bias = None
        for k, v in decoder_sd.items():
            if "mu_head" in k and "weight" in k:
                mu_weight = v
            elif "mu_head" in k and "bias" in k:
                mu_bias = v

        if mu_weight is None:
            logger.info("No mu_head weights found — decode_tokens unavailable")
            return

        # Build a single linear layer for mu_head
        w = mu_weight
        b = mu_bias
        self.mu_head = nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        self.mu_head.weight = nn.Parameter(w)
        if b is not None:
            self.mu_head.bias = nn.Parameter(b)
        self.mu_head.to(self.device, self.dtype).eval()
        logger.info(
            "Loaded mu_head: %d → %d", w.shape[1], w.shape[0]
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Encode raw voxels to latent representation.

        Args:
            voxels: Raw fMRI voxels (B, input_dim) where input_dim=15724

        Returns:
            Latent representation (B, 2048)
        """
        x = voxels.to(self.device, self.dtype)
        return self.encoder(x)

    @torch.no_grad()
    def decode_tokens(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Map raw voxels → predicted bigG token embeddings.

        Args:
            voxels: Raw fMRI voxels (B, input_dim)

        Returns:
            Token predictions (B, n_tokens, token_dim) — e.g. (B, 257, 1280)

        Raises:
            RuntimeError: If mu_head was not loaded from checkpoint
        """
        if self.mu_head is None:
            raise RuntimeError(
                "mu_head not loaded — cannot decode tokens. "
                "Check that checkpoint contains decoder.mu_head weights."
            )
        latent = self.encode(voxels)
        flat_tokens = self.mu_head(latent)  # (B, 328960)
        return flat_tokens.view(-1, self.config.n_tokens, self.config.token_dim)

    @torch.no_grad()
    def predict_cls(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Predict a CLS-equivalent embedding by mean-pooling over tokens.

        Since FMRI2images predicts 257 patch+CLS tokens, we mean-pool them
        to get a single 1280-d vector suitable for cosine similarity comparison.

        Args:
            voxels: Raw fMRI voxels (B, input_dim)

        Returns:
            CLS-equivalent embedding (B, token_dim) — e.g. (B, 1280)
        """
        tokens = self.decode_tokens(voxels)  # (B, 257, 1280)
        cls = tokens.mean(dim=1)  # (B, 1280)
        return F.normalize(cls, dim=-1)

    @torch.no_grad()
    def predict_cls_token(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Extract only the first (CLS) token prediction.

        Args:
            voxels: Raw fMRI voxels (B, input_dim)

        Returns:
            CLS token (B, token_dim) — e.g. (B, 1280)
        """
        tokens = self.decode_tokens(voxels)  # (B, 257, 1280)
        return F.normalize(tokens[:, 0], dim=-1)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for logging / STATUS.md."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        mu_params = (
            sum(p.numel() for p in self.mu_head.parameters())
            if self.mu_head
            else 0
        )
        return {
            "checkpoint": str(self.checkpoint_path),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "subject": self.subject,
            "val_loss": self.val_loss,
            "input_dim": self.config.input_dim,
            "encoder_dims": self.config.encoder_dims,
            "latent_dim": self.config.encoder_dims[-1],
            "n_tokens": self.config.n_tokens,
            "token_dim": self.config.token_dim,
            "encoder_params": enc_params,
            "decoder_params": mu_params,
            "total_params": enc_params + mu_params,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ExternalModelLoader("
            f"input={s['input_dim']}, "
            f"encoder={s['encoder_dims']}, "
            f"tokens={s['n_tokens']}×{s['token_dim']}, "
            f"params={s['total_params']:,}, "
            f"epoch={s['epoch']})"
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def load_fmri2images_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
    use_ema: bool = True,
) -> ExternalModelLoader:
    """
    Convenience function to load an FMRI2images checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Target device
        use_ema: Use EMA shadow if available

    Returns:
        ExternalModelLoader ready for inference
    """
    return ExternalModelLoader(
        checkpoint_path=checkpoint_path,
        device=device,
        use_ema=use_ema,
    )


def compare_predictions(
    our_pred: torch.Tensor,
    external_pred: torch.Tensor,
    metric: str = "cosine",
) -> Dict[str, float]:
    """
    Compare predictions from our model and FMRI2images.

    Since the two models use different CLIP backbones (ViT-L/14 768-d vs
    ViT-bigG/14 1280-d), direct embedding comparison requires a shared
    evaluation space. Options:
        - "cosine": Requires same-dim inputs (use after a projection head)
        - "rank_correlation": Compare rank orders of retrieval results
        - "retrieval_overlap": Fraction of shared top-K retrievals

    Args:
        our_pred: Our model predictions (B, D₁)
        external_pred: FMRI2images predictions (B, D₂)
        metric: Comparison metric

    Returns:
        Dictionary of comparison scores
    """
    results: Dict[str, float] = {}

    if metric == "cosine" and our_pred.shape[-1] == external_pred.shape[-1]:
        cos = F.cosine_similarity(our_pred, external_pred, dim=-1)
        results["cosine_mean"] = cos.mean().item()
        results["cosine_std"] = cos.std().item()
        results["cosine_median"] = cos.median().item()

    elif metric == "rank_correlation":
        # Spearman correlation of pairwise distance matrices
        from scipy.stats import spearmanr

        d1 = torch.cdist(our_pred.float(), our_pred.float()).cpu().numpy().ravel()
        d2 = torch.cdist(
            external_pred.float(), external_pred.float()
        ).cpu().numpy().ravel()
        rho, pval = spearmanr(d1, d2)
        results["spearman_rho"] = float(rho)
        results["spearman_pval"] = float(pval)

    elif metric == "retrieval_overlap":
        # For each query, find top-K in each model's similarity matrix
        k = min(10, our_pred.shape[0] - 1)
        sim1 = our_pred @ our_pred.T
        sim2 = external_pred @ external_pred.T
        sim1.fill_diagonal_(-float("inf"))
        sim2.fill_diagonal_(-float("inf"))
        topk1 = sim1.topk(k, dim=-1).indices
        topk2 = sim2.topk(k, dim=-1).indices
        overlap = 0.0
        for i in range(our_pred.shape[0]):
            s1 = set(topk1[i].tolist())
            s2 = set(topk2[i].tolist())
            overlap += len(s1 & s2) / k
        results["retrieval_overlap@10"] = overlap / our_pred.shape[0]

    return results
