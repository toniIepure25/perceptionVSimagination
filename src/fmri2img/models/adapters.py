"""
Imagery Adaptation Modules
===========================

Lightweight adapter modules for efficient cross-domain transfer from perception
to imagery. These adapters are trained on top of frozen perception encoders to
improve imagery reconstruction without full retraining.

Scientific Design:
- Adapters keep base model frozen (preserves perception capabilities)
- Minimal parameters (<<1% of base model) for efficient training
- Identity initialization allows gradient-based adaptation
- Optional condition embeddings for multi-domain learning

References:
- Houlsby et al. (2019): "Parameter-Efficient Transfer Learning for NLP"
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
- Perez et al. (2018): "FiLM: Visual Reasoning with a General Conditioning Layer"
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConditionEmbedding(nn.Module):
    """
    Learnable condition embeddings for multi-domain adaptation.
    
    Supports two conditioning modes:
    - 'add': Additive conditioning (condition vector added to input)
    - 'film': FiLM-style conditioning (affine transformation: γ * x + β)
    
    Args:
        embed_dim: Embedding dimensionality
        n_conditions: Number of conditions (e.g., 2 for perception/imagery)
        mode: 'add' or 'film'
    """
    
    def __init__(
        self,
        embed_dim: int,
        n_conditions: int = 2,
        mode: Literal['add', 'film'] = 'add'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_conditions = n_conditions
        self.mode = mode
        
        if mode == 'add':
            # Simple additive embedding
            self.embeddings = nn.Embedding(n_conditions, embed_dim)
            # Initialize near zero for smooth adaptation
            nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)
        
        elif mode == 'film':
            # FiLM-style: learn scale (γ) and shift (β)
            self.gamma = nn.Embedding(n_conditions, embed_dim)
            self.beta = nn.Embedding(n_conditions, embed_dim)
            # Initialize gamma near 1, beta near 0 (identity transform)
            nn.init.normal_(self.gamma.weight, mean=1.0, std=0.01)
            nn.init.normal_(self.beta.weight, mean=0.0, std=0.01)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'add' or 'film'")
    
    def forward(self, x: torch.Tensor, condition_idx: torch.Tensor) -> torch.Tensor:
        """
        Apply condition embedding to input.
        
        Args:
            x: Input tensor (B, embed_dim)
            condition_idx: Condition indices (B,) in [0, n_conditions)
        
        Returns:
            Conditioned tensor (B, embed_dim)
        """
        if self.mode == 'add':
            # x' = x + condition_embedding
            cond_embed = self.embeddings(condition_idx)  # (B, embed_dim)
            return x + cond_embed
        
        elif self.mode == 'film':
            # x' = γ * x + β (FiLM)
            gamma = self.gamma(condition_idx)  # (B, embed_dim)
            beta = self.beta(condition_idx)    # (B, embed_dim)
            return gamma * x + beta


class LinearAdapter(nn.Module):
    """
    Simple linear adapter: y = W x + b
    
    Initialized as identity (W=I, b=0) to preserve base model behavior
    at initialization. Gradient descent can then adapt W and b for imagery.
    
    Args:
        embed_dim: Embedding dimensionality (typically 512 for CLIP)
        use_condition: Whether to use condition embeddings
        condition_mode: 'add' or 'film'
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        use_condition: bool = False,
        condition_mode: Literal['add', 'film'] = 'add',
        normalize: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_condition = use_condition
        self.normalize = normalize
        
        # Linear transformation
        self.linear = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Initialize as identity
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        # Optional condition embedding
        if use_condition:
            self.condition_embed = ConditionEmbedding(
                embed_dim=embed_dim,
                n_conditions=2,  # perception=0, imagery=1
                mode=condition_mode
            )
        else:
            self.condition_embed = None
    
    def forward(
        self,
        x: torch.Tensor,
        condition_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional conditioning.
        
        Args:
            x: Input embeddings (B, embed_dim)
            condition_idx: Condition indices (B,) or None
        
        Returns:
            Adapted embeddings (B, embed_dim), L2-normalized
        """
        # Apply condition embedding (if enabled)
        if self.use_condition and condition_idx is not None:
            if self.condition_embed is None:
                raise ValueError("Condition embedding not initialized")
            x = self.condition_embed(x, condition_idx)
        
        # Linear transformation
        x = self.linear(x)
        
        # L2 normalize (if enabled)
        if getattr(self, 'normalize', True):
            x = torch.nn.functional.normalize(x, dim=-1)
        
        return x


class MLPAdapter(nn.Module):
    """
    MLP adapter with residual connection: y = x + MLP(LayerNorm(x))
    
    Architecture:
        x → LayerNorm → Linear → GELU → Linear → (+x) → L2-normalize
    
    Residual connection ensures identity at initialization, allowing
    the adapter to learn refinements without disrupting base model.
    
    Args:
        embed_dim: Embedding dimensionality
        hidden_scale: Hidden dimension = embed_dim * hidden_scale
        dropout: Dropout probability
        use_condition: Whether to use condition embeddings
        condition_mode: 'add' or 'film'
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        hidden_scale: float = 2.0,
        dropout: float = 0.1,
        use_condition: bool = False,
        condition_mode: Literal['add', 'film'] = 'add',
        normalize: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_condition = use_condition
        self.normalize = normalize
        
        hidden_dim = int(embed_dim * hidden_scale)
        
        # Pre-normalization (stabilizes training)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Initialize small weights for near-identity at start
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Optional condition embedding
        if use_condition:
            self.condition_embed = ConditionEmbedding(
                embed_dim=embed_dim,
                n_conditions=2,
                mode=condition_mode
            )
        else:
            self.condition_embed = None
    
    def forward(
        self,
        x: torch.Tensor,
        condition_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input embeddings (B, embed_dim)
            condition_idx: Condition indices (B,) or None
        
        Returns:
            Adapted embeddings (B, embed_dim), L2-normalized
        """
        # Apply condition embedding (if enabled)
        if self.use_condition and condition_idx is not None:
            if self.condition_embed is None:
                raise ValueError("Condition embedding not initialized")
            x = self.condition_embed(x, condition_idx)
        
        # Residual connection: y = x + MLP(norm(x))
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        x = residual + x
        
        # L2 normalize (if enabled)
        if getattr(self, 'normalize', True):
            x = torch.nn.functional.normalize(x, dim=-1)
        
        return x


class AdaptedModel(nn.Module):
    """
    Wrapper combining frozen base encoder + trainable adapter.
    
    Forward pass:
        voxels -> base_model(frozen) -> base_embed -> adapter(trainable) -> adapted_embed
    
    Only adapter parameters are trainable, preserving base model weights.
    
    Args:
        base_model: Frozen perception encoder (ridge/mlp/two_stage)
        adapter: Trainable adapter module
    """
    
    def __init__(self, base_model: nn.Module, adapter: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Ensure adapter is trainable
        for param in self.adapter.parameters():
            param.requires_grad = True
        
        logger.info(f"Created AdaptedModel:")
        logger.info(f"  Base model parameters (frozen): {sum(p.numel() for p in self.base_model.parameters()):,}")
        logger.info(f"  Adapter parameters (trainable): {sum(p.numel() for p in self.adapter.parameters()):,}")
    
    def forward(
        self,
        x: torch.Tensor,
        condition_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through base model and adapter.
        
        Args:
            x: Input voxels (B, input_dim)
            condition_idx: Condition indices for adapter (B,) or None
        
        Returns:
            Adapted CLIP embeddings (B, 512)
        """
        # Base model forward (no gradients)
        with torch.no_grad():
            base_embed = self.base_model(x)
        
        # Adapter forward (with gradients)
        adapted_embed = self.adapter(base_embed, condition_idx=condition_idx)
        
        return adapted_embed
    
    def get_base_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get base model embedding (for analysis)."""
        with torch.no_grad():
            return self.base_model(x)
    
    def num_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiTargetAdapter(nn.Module):
    """
    Adapts multiple outputs from a MultiTargetDecoder independently.
    Applies the chosen adapter type to each output (clip, tokens, sd_latent).
    """
    def __init__(
        self,
        adapter_type: str = 'mlp',
        clip_dim: int = 512,
        token_dim: int = 1024,
        sd_latent_dim: int = 4 * 64 * 64,
        use_condition: bool = False,
        condition_mode: Literal['add', 'film'] = 'add',
        **kwargs
    ):
        super().__init__()
        # CLIP gets L2 normalized
        self.clip_adapter = create_adapter(
            adapter_type, embed_dim=clip_dim, use_condition=use_condition,
            condition_mode=condition_mode, normalize=True, **kwargs
        )
        # Tokens get L2 normalized
        self.token_adapter = create_adapter(
            adapter_type, embed_dim=token_dim, use_condition=use_condition,
            condition_mode=condition_mode, normalize=True, **kwargs
        )
        # SD latents do NOT get L2 normalized
        self.sd_latent_adapter = create_adapter(
            adapter_type, embed_dim=sd_latent_dim, use_condition=use_condition,
            condition_mode=condition_mode, normalize=False, **kwargs
        )
        
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        condition_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        out = {}
        
        if 'clip' in x_dict:
            out['clip'] = self.clip_adapter(x_dict['clip'], condition_idx)
            
        if 'tokens' in x_dict:
            b, n, d = x_dict['tokens'].shape
            t = x_dict['tokens'].view(b * n, d)
            c_idx = condition_idx.repeat_interleave(n) if condition_idx is not None else None
            t_adapted = self.token_adapter(t, c_idx)
            out['tokens'] = t_adapted.view(b, n, d)
            
        if 'sd_latent' in x_dict:
            b, c, h, w = x_dict['sd_latent'].shape
            flat = x_dict['sd_latent'].view(b, c * h * w)
            flat_adapted = self.sd_latent_adapter(flat, condition_idx)
            out['sd_latent'] = flat_adapted.view(b, c, h, w)
            
        return out


class MultiTargetAdapter(nn.Module):
    """
    Adapts multiple outputs from a MultiTargetDecoder independently.
    Applies the chosen adapter type to each output (clip, tokens, sd_latent).
    """
    def __init__(
        self,
        adapter_type: str = 'mlp',
        clip_dim: int = 512,
        token_dim: int = 1024,
        sd_latent_dim: int = 4 * 64 * 64,
        use_condition: bool = False,
        condition_mode: Literal['add', 'film'] = 'add',
        **kwargs
    ):
        super().__init__()
        # CLIP gets L2 normalized
        self.clip_adapter = create_adapter(
            adapter_type, embed_dim=clip_dim, use_condition=use_condition,
            condition_mode=condition_mode, normalize=True, **kwargs
        )
        # Tokens get L2 normalized
        self.token_adapter = create_adapter(
            adapter_type, embed_dim=token_dim, use_condition=use_condition,
            condition_mode=condition_mode, normalize=True, **kwargs
        )
        # SD latents do NOT get L2 normalized
        self.sd_latent_adapter = create_adapter(
            adapter_type, embed_dim=sd_latent_dim, use_condition=use_condition,
            condition_mode=condition_mode, normalize=False, **kwargs
        )
        
    def forward(
        self, 
        x_dict: Dict[str, torch.Tensor], 
        condition_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        out = {}
        
        if 'clip' in x_dict:
            out['clip'] = self.clip_adapter(x_dict['clip'], condition_idx)
            
        if 'tokens' in x_dict:
            b, n, d = x_dict['tokens'].shape
            t = x_dict['tokens'].view(b * n, d)
            c_idx = condition_idx.repeat_interleave(n) if condition_idx is not None else None
            t_adapted = self.token_adapter(t, c_idx)
            out['tokens'] = t_adapted.view(b, n, d)
            
        if 'sd_latent' in x_dict:
            b, c, h, w = x_dict['sd_latent'].shape
            flat = x_dict['sd_latent'].view(b, c * h * w)
            flat_adapted = self.sd_latent_adapter(flat, condition_idx)
            out['sd_latent'] = flat_adapted.view(b, c, h, w)
            
        return out


def create_adapter(
    adapter_type: str,
    embed_dim: int = 512,
    use_condition: bool = False,
    condition_mode: str = 'add',
    normalize: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create adapters.
    
    Args:
        adapter_type: 'linear', 'mlp', or 'multi_target'
        embed_dim: Embedding dimensionality
        use_condition: Whether to use condition embeddings
        condition_mode: 'add' or 'film'
        normalize: Whether to apply L2 normalization
        **kwargs: Additional adapter-specific arguments
    
    Returns:
        Adapter module
    """
    if adapter_type == 'multi_target':
        return MultiTargetAdapter(
            adapter_type=kwargs.get('base_adapter_type', 'mlp'),
            clip_dim=kwargs.get('clip_dim', 512),
            token_dim=kwargs.get('token_dim', 1024),
            sd_latent_dim=kwargs.get('sd_latent_dim', 4 * 64 * 64),
            use_condition=use_condition,
            condition_mode=condition_mode,
            **kwargs
        )
        
    if adapter_type == 'linear':
        return LinearAdapter(
            embed_dim=embed_dim,
            use_condition=use_condition,
            condition_mode=condition_mode,
            normalize=normalize
        )
    
    elif adapter_type == 'mlp':
        return MLPAdapter(
            embed_dim=embed_dim,
            hidden_scale=kwargs.get('hidden_scale', 2.0),
            dropout=kwargs.get('dropout', 0.1),
            use_condition=use_condition,
            condition_mode=condition_mode,
            normalize=normalize
        )
    
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Use 'linear', 'mlp', or 'multi_target'")


def save_adapter(
    adapter: nn.Module,
    path: str,
    meta: Dict[str, Any],
    full_model: Optional[nn.Module] = None
) -> None:
    """
    Save adapter checkpoint with metadata.
    
    Args:
        adapter: Trained adapter module
        path: Output checkpoint path
        meta: Metadata (adapter_type, training info, etc.)
        full_model: Optional full AdaptedModel (for saving both parts)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'adapter_state_dict': adapter.state_dict(),
        'meta': meta
    }
    
    if full_model is not None and isinstance(full_model, AdaptedModel):
        checkpoint['full_model_state_dict'] = full_model.state_dict()
    
    torch.save(checkpoint, path)
    logger.info(f"Saved adapter checkpoint to {path}")


def load_adapter(
    path: str,
    adapter_type: str = None,
    embed_dim: int = 512,
    map_location: str = 'cpu'
) -> tuple[nn.Module, Dict[str, Any]]:
    """
    Load adapter checkpoint.
    
    Args:
        path: Checkpoint path
        adapter_type: 'linear' or 'mlp' (auto-detected if None)
        embed_dim: Embedding dimensionality
        map_location: Device for loading
    
    Returns:
        (adapter, metadata)
    """
    checkpoint = torch.load(path, map_location=map_location)
    meta = checkpoint['meta']
    
    # Auto-detect adapter type
    if adapter_type is None:
        adapter_type = meta.get('adapter_type', 'mlp')
    
    # Create adapter
    adapter = create_adapter(
        adapter_type=adapter_type,
        embed_dim=embed_dim,
        use_condition=meta.get('use_condition', False),
        condition_mode=meta.get('condition_mode', 'add'),
        hidden_scale=meta.get('hidden_scale', 2.0),
        dropout=meta.get('dropout', 0.1),
        base_adapter_type=meta.get('base_adapter_type', 'mlp'),
        clip_dim=meta.get('clip_dim', 512),
        token_dim=meta.get('token_dim', 1024),
        sd_latent_dim=meta.get('sd_latent_dim', 4 * 64 * 64)
    )
    
    # Load weights
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    
    logger.info(f"Loaded {adapter_type} adapter from {path}")
    logger.info(f"  Parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    
    return adapter, meta
