"""
LoRA (Low-Rank Adaptation) for Perception→Imagery Transfer
==========================================================

Implements parameter-efficient domain adaptation using low-rank
decomposition (Hu et al., 2022). Instead of fine-tuning all encoder
parameters for imagery, LoRA injects small trainable rank-r matrices
into the frozen perception encoder:

    h = W·x + (α/r) · B(A(x))

where W is frozen, A ∈ R^{d×r}, B ∈ R^{r×d}, and only A, B are
trained. This yields extreme parameter efficiency:
    - MLP adapter: 768 × 1536 + 1536 × 768 ≈ 2.4M params
    - LoRA (r=4): 768 × 4 × 2 = 6,144 params (400× fewer)

Unique to this project:
    - MultiRankLoRA: parallel branches at ranks [2, 4, 8, 16] with
      learnable softmax-weighted combination — automatic rank selection
      within a single training run
    - LoRAAdaptedModel: wraps any perception encoder, injects LoRA
      into specified layers while freezing the rest

References:
    Hu et al. (2022). "LoRA: Low-Rank Adaptation of Large Language
        Models." ICLR. https://arxiv.org/abs/2106.09685
    Aghajanyan et al. (2021). "Intrinsic Dimensionality Explains the
        Effectiveness of Language Model Fine-Tuning." ACL.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core LoRA modules
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer: W·x + (α/r) · B(A(x)).

    Decomposes the weight update ΔW into low-rank factors A, B:
        ΔW = (α/r) · B @ A

    Initialization:
    - A: Kaiming uniform (random features)
    - B: zeros (starts as identity — no change at init)
    This ensures the adapted model starts from the original behavior.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of low-rank decomposition (default: 4)
        alpha: Scaling factor (default: 1.0). Higher α = stronger adaptation.
        dropout: Dropout on input before LoRA branch (default: 0.0)
        merge_weights: If True, merge LoRA into base weights for inference
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merge_weights = merge_weights
        self.merged = False

        # Low-rank factors
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Initialize: A random, B zero → starts as identity
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA delta: (α/r) · B(A(dropout(x))).

        This is the *residual only* — add to the base layer's output:
            output = base_linear(x) + lora(x)

        Args:
            x: Input tensor, shape (..., in_features)

        Returns:
            LoRA delta, shape (..., out_features)
        """
        return self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


class LoRAAdapter(nn.Module):
    """
    Standalone LoRA adapter for embedding transformation.

    Applies LoRA as a residual to the input embedding:
        output = x + (α/r) · B(A(dropout(x)))

    Optionally L2-normalizes the output (matching the convention
    of LinearAdapter and MLPAdapter in adapters.py).

    Args:
        embed_dim: Embedding dimension (default: 768 for ViT-L/14)
        rank: Low-rank dimension (default: 4)
        alpha: Scaling factor (default: 1.0)
        dropout: Input dropout (default: 0.0)
        normalize: L2-normalize output (default: True)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.normalize = normalize
        self.lora = LoRALinear(
            embed_dim, embed_dim, rank=rank, alpha=alpha, dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        condition_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply LoRA adaptation.

        Args:
            x: Input embeddings, shape (B, embed_dim)
            condition_idx: Unused (included for API compatibility with
                LinearAdapter/MLPAdapter)

        Returns:
            Adapted embeddings, shape (B, embed_dim)
        """
        out = x + self.lora(x)
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiRankLoRA(nn.Module):
    """
    Multi-rank LoRA with learnable rank selection.

    Runs parallel LoRA branches at multiple ranks and combines them
    with softmax-weighted mixing. This performs automatic rank selection
    during training — the model learns which rank is optimal.

    Architecture:
        x → LoRA_r2(x) ──┐
        x → LoRA_r4(x) ──┤ softmax(w) → weighted sum → output
        x → LoRA_r8(x) ──┤
        x → LoRA_r16(x) ─┘

    The mixing weights are softmax-normalized logits (learnable), ensuring
    they sum to 1. After training, the dominant rank can be read off from
    get_effective_ranks(), and the model can be distilled to a single-rank
    LoRA for deployment.

    This is novel: existing LoRA implementations require manual rank
    selection. MultiRankLoRA automates this via differentiable architecture
    search within the LoRA paradigm.

    Args:
        embed_dim: Embedding dimension (default: 768)
        ranks: List of ranks to try (default: [2, 4, 8, 16])
        alpha: LoRA scaling factor
        dropout: Input dropout
        normalize: L2-normalize output
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ranks: Optional[List[int]] = None,
        alpha: float = 1.0,
        dropout: float = 0.0,
        normalize: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ranks = ranks or [2, 4, 8, 16]
        self.normalize = normalize

        # Parallel LoRA branches
        self.branches = nn.ModuleList(
            [
                LoRALinear(embed_dim, embed_dim, rank=r, alpha=alpha, dropout=dropout)
                for r in self.ranks
            ]
        )

        # Learnable mixing weights (softmax-normalized)
        self.weight_logits = nn.Parameter(torch.zeros(len(self.ranks)))

    def forward(
        self,
        x: torch.Tensor,
        condition_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-rank LoRA with learned weighting.

        Args:
            x: Input embeddings, shape (B, embed_dim)
            condition_idx: Unused (API compatibility)

        Returns:
            Adapted embeddings, shape (B, embed_dim)
        """
        weights = F.softmax(self.weight_logits, dim=0)  # (n_ranks,)

        # Weighted sum of LoRA deltas
        delta = torch.zeros_like(x)
        for w, branch in zip(weights, self.branches):
            delta = delta + w * branch(x)

        out = x + delta
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def get_effective_ranks(self) -> Dict[int, float]:
        """
        Get the learned weight for each rank.

        Returns:
            Dict mapping rank → weight (sums to 1.0)
        """
        weights = F.softmax(self.weight_logits, dim=0).detach().cpu().numpy()
        return {r: float(w) for r, w in zip(self.ranks, weights)}

    def get_dominant_rank(self) -> int:
        """Return the rank with highest learned weight."""
        ranks_weights = self.get_effective_ranks()
        return max(ranks_weights, key=ranks_weights.get)

    def num_trainable_params(self) -> int:
        """Count trainable parameters across all branches + weights."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRAAdaptedModel(nn.Module):
    """
    Wraps a perception encoder with LoRA injection for imagery adaptation.

    Freezes the base model entirely and adds LoRA modules to specified
    layers. During forward pass, the base model runs as usual, and LoRA
    deltas are added to the output.

    This is the simplest integration: LoRA is applied only to the final
    output embedding (not injected into internal layers). For deeper
    injection, subclass and override _inject_lora().

    Args:
        base_model: Pre-trained perception encoder (frozen)
        adapter: LoRA adapter module (LoRAAdapter or MultiRankLoRA)

    Example:
        >>> encoder = load_two_stage_encoder("checkpoint.pt")
        >>> lora = LoRAAdapter(embed_dim=768, rank=4)
        >>> model = LoRAAdaptedModel(encoder, lora)
        >>> print(f"Trainable: {model.num_trainable_params()}")  # ~6K
        >>> output = model(fmri_input)
    """

    def __init__(self, base_model: nn.Module, adapter: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def forward(
        self,
        x: torch.Tensor,
        condition_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: base encoder (frozen) → LoRA adaptation.

        Args:
            x: fMRI input, shape (B, input_dim)
            condition_idx: Optional condition index (API compatibility)

        Returns:
            Adapted embeddings, shape (B, embed_dim)
        """
        with torch.no_grad():
            base_out = self.base_model(x)
        return self.adapter(base_out, condition_idx)

    def get_base_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get unadapted base model embedding (for comparison)."""
        with torch.no_grad():
            return self.base_model(x)

    def num_trainable_params(self) -> int:
        """Count trainable parameters (LoRA only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode: bool = True):
        """Override: keep base model in eval, only train adapter."""
        super().train(mode)
        self.base_model.eval()
        return self


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_lora_adapter(
    adapter: nn.Module,
    path: str,
    meta: Optional[Dict] = None,
    full_model: Optional[nn.Module] = None,
) -> None:
    """
    Save LoRA adapter checkpoint.

    Args:
        adapter: LoRAAdapter or MultiRankLoRA module
        path: Save path
        meta: Optional metadata dict (embed_dim, rank, etc.)
        full_model: If provided, also save full model state for convenience
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "adapter_state_dict": adapter.state_dict(),
        "adapter_type": type(adapter).__name__,
        "meta": meta or {},
    }

    # Auto-populate meta
    if isinstance(adapter, LoRAAdapter):
        state["meta"].setdefault("embed_dim", adapter.embed_dim)
        state["meta"].setdefault("rank", adapter.rank)
        state["meta"].setdefault("adapter_class", "LoRAAdapter")
    elif isinstance(adapter, MultiRankLoRA):
        state["meta"].setdefault("embed_dim", adapter.embed_dim)
        state["meta"].setdefault("ranks", adapter.ranks)
        state["meta"].setdefault("effective_ranks", adapter.get_effective_ranks())
        state["meta"].setdefault("dominant_rank", adapter.get_dominant_rank())
        state["meta"].setdefault("adapter_class", "MultiRankLoRA")

    if full_model is not None:
        state["full_model_state_dict"] = full_model.state_dict()

    state["meta"].setdefault("trainable_params", adapter.num_trainable_params())

    torch.save(state, path)
    logger.info(
        f"Saved {type(adapter).__name__} to {path} "
        f"({adapter.num_trainable_params()} params)"
    )


def load_lora_adapter(
    path: str,
    embed_dim: int = 768,
    map_location: str = "cpu",
) -> Tuple[nn.Module, Dict]:
    """
    Load LoRA adapter checkpoint.

    Auto-detects adapter type (LoRAAdapter or MultiRankLoRA) from
    saved metadata.

    Args:
        path: Checkpoint path
        embed_dim: Embedding dimension (fallback if not in meta)
        map_location: Device mapping

    Returns:
        (adapter_module, metadata_dict)
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    meta = ckpt.get("meta", {})
    adapter_class = meta.get("adapter_class", ckpt.get("adapter_type", "LoRAAdapter"))

    if adapter_class == "MultiRankLoRA":
        ranks = meta.get("ranks", [2, 4, 8, 16])
        dim = meta.get("embed_dim", embed_dim)
        adapter = MultiRankLoRA(embed_dim=dim, ranks=ranks)
    else:
        rank = meta.get("rank", 4)
        dim = meta.get("embed_dim", embed_dim)
        adapter = LoRAAdapter(embed_dim=dim, rank=rank)

    adapter.load_state_dict(ckpt["adapter_state_dict"])
    logger.info(f"Loaded {adapter_class} from {path} ({adapter.num_trainable_params()} params)")
    return adapter, meta
