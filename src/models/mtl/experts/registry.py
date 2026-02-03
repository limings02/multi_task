"""
Expert Registry: Factory for building heterogeneous experts.

Supported expert types:
  - mlp: Multi-layer perceptron with configurable dims, activation, dropout, batch normalization
  - crossnet_v2: DCN-v2 style CrossNet with optional low-rank factorization and projection
  - identity: Pass-through expert (for debugging/ablation)

Usage:
    spec = {"type": "mlp", "dims": [256, 128], "activation": "relu", "dropout": 0.1}
    expert = build_expert(spec, in_dim=64, out_dim=128)

Author: Heterogeneous Expert Extension for PLE-Lite
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn

from src.models.backbones.layers import MLP

logger = logging.getLogger(__name__)


# =============================================================================
# Expert Registry: type_name -> builder_function
# =============================================================================
EXPERT_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_expert(name: str):
    """Decorator to register an expert builder function."""
    def decorator(fn: Callable[..., nn.Module]):
        EXPERT_REGISTRY[name.lower()] = fn
        return fn
    return decorator


# =============================================================================
# CrossNet-v2: DCN-v2 style cross network with optional low-rank factorization
# =============================================================================
class CrossNetV2(nn.Module):
    """
    DCN-v2 style Cross Network.
    
    Core formula per layer:
        x_{l+1} = x_0 * (W_l @ x_l + b_l) + x_l
    
    Where * is element-wise multiplication (Hadamard product).
    
    Optional low-rank factorization (DCN-Mix style):
        Instead of W_l (d x d), use: U_l @ V_l^T (d x r x d) for reduced parameters.
        
    Args:
        in_dim: Input dimension (= x_0 dimension)
        num_layers: Number of cross layers
        low_rank: If > 0, use low-rank factorization with this rank (r). If 0 or None, use full rank.
        use_bias: Whether to include bias in cross layers
    """
    
    def __init__(
        self,
        in_dim: int,
        num_layers: int = 3,
        low_rank: Optional[int] = None,
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.low_rank = low_rank if low_rank and low_rank > 0 else None
        self.use_bias = use_bias
        
        if self.low_rank is not None:
            # Low-rank factorization: W = U @ V^T
            # U: [d, r], V: [d, r] => W: [d, d]
            self.U = nn.ParameterList([
                nn.Parameter(torch.empty(in_dim, self.low_rank))
                for _ in range(num_layers)
            ])
            self.V = nn.ParameterList([
                nn.Parameter(torch.empty(in_dim, self.low_rank))
                for _ in range(num_layers)
            ])
            # Initialize with Xavier uniform
            for u, v in zip(self.U, self.V):
                nn.init.xavier_uniform_(u)
                nn.init.xavier_uniform_(v)
        else:
            # Full-rank weight matrices
            self.W = nn.ParameterList([
                nn.Parameter(torch.empty(in_dim, in_dim))
                for _ in range(num_layers)
            ])
            for w in self.W:
                nn.init.xavier_uniform_(w)
        
        if use_bias:
            self.bias = nn.ParameterList([
                nn.Parameter(torch.zeros(in_dim))
                for _ in range(num_layers)
            ])
        else:
            self.bias = None
        
        self.output_dim = in_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through cross layers.
        
        Args:
            x: Input tensor [B, D]
            
        Returns:
            Output tensor [B, D] after num_layers cross operations
        """
        x0 = x  # Store original input
        xl = x
        
        for i in range(self.num_layers):
            if self.low_rank is not None:
                # Low-rank: x_{l+1} = x_0 * (U @ (V^T @ x_l) + b) + x_l
                # First: V^T @ x_l -> [B, r]
                vt_xl = torch.matmul(xl, self.V[i])  # [B, r]
                # Then: U @ (V^T @ x_l) -> [B, d]
                wx = torch.matmul(vt_xl, self.U[i].t())  # [B, d]
            else:
                # Full-rank: W @ x_l
                wx = torch.matmul(xl, self.W[i])  # [B, d]
            
            if self.bias is not None:
                wx = wx + self.bias[i]
            
            # Element-wise multiplication with x0, then residual connection
            xl = x0 * wx + xl
        
        return xl


class CrossNetV2Expert(nn.Module):
    """
    CrossNet-v2 Expert wrapper with optional output projection.
    
    Combines CrossNet-v2 with an optional linear projection to target output dimension.
    
    Args:
        in_dim: Input dimension
        out_dim: Target output dimension (must match for gate mixing)
        num_layers: Number of cross layers
        low_rank: Low-rank dimension (None for full-rank)
        use_bias: Whether to use bias in cross layers
        proj_enabled: If True, project output to out_dim; if False and in_dim != out_dim, raise error
        proj_activation: Activation for projection layer (None for linear)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 3,
        low_rank: Optional[int] = None,
        use_bias: bool = True,
        proj_enabled: bool = True,
        proj_activation: Optional[str] = None,
    ):
        super().__init__()
        self.cross = CrossNetV2(
            in_dim=in_dim,
            num_layers=num_layers,
            low_rank=low_rank,
            use_bias=use_bias,
        )
        
        cross_out_dim = self.cross.output_dim  # = in_dim for CrossNet
        
        # Output projection (if needed)
        if cross_out_dim != out_dim or proj_enabled:
            layers: List[nn.Module] = [nn.Linear(cross_out_dim, out_dim)]
            if proj_activation:
                layers.append(_make_activation(proj_activation))
            self.proj = nn.Sequential(*layers)
        else:
            self.proj = None
        
        self.output_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cross(x)
        if self.proj is not None:
            h = self.proj(h)
        return h


# =============================================================================
# MLP Expert: Standard multi-layer perceptron
# =============================================================================
class MLPExpert(nn.Module):
    """
    MLP Expert with guaranteed output dimension alignment.
    
    If the last hidden dim does not match out_dim, an additional linear projection is added.
    
    Args:
        in_dim: Input dimension
        out_dim: Required output dimension (must match for gate mixing)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function name
        dropout: Dropout probability
        use_bn: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        hidden_dims = hidden_dims or []
        
        # Build MLP layers
        if hidden_dims:
            self.mlp = MLP(
                input_dim=in_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout=dropout,
                use_bn=use_bn,
            )
            mlp_out_dim = self.mlp.output_dim
        else:
            self.mlp = None
            mlp_out_dim = in_dim
        
        # Output projection (if needed to match out_dim)
        if mlp_out_dim != out_dim:
            self.out_proj = nn.Linear(mlp_out_dim, out_dim)
        else:
            self.out_proj = None
        
        self.output_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mlp is not None:
            x = self.mlp(x)
        if self.out_proj is not None:
            x = self.out_proj(x)
        return x


# =============================================================================
# Identity Expert: Pass-through (for debugging/ablation)
# =============================================================================
class IdentityExpert(nn.Module):
    """
    Identity expert that optionally projects input to match output dimension.
    
    If in_dim == out_dim, acts as pure identity.
    If in_dim != out_dim, adds a linear projection (with optional activation).
    
    Useful for:
    - Ablation studies (removing expert capacity)
    - Debugging pipeline issues
    - Baseline comparisons
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        proj_activation: Optional[str] = None,
    ):
        super().__init__()
        if in_dim != out_dim:
            layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
            if proj_activation:
                layers.append(_make_activation(proj_activation))
            self.proj = nn.Sequential(*layers)
        else:
            self.proj = None
        
        self.output_dim = out_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            return self.proj(x)
        return x


# =============================================================================
# Utility functions
# =============================================================================
def _make_activation(name: str) -> nn.Module:
    """Create activation module from name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "selu":
        return nn.SELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1)
    if name in ("none", "linear", "identity"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


# =============================================================================
# Register expert builders
# =============================================================================
@register_expert("mlp")
def build_mlp_expert(spec: Dict[str, Any], in_dim: int, out_dim: int) -> nn.Module:
    """
    Build MLP expert from spec.
    
    Spec fields:
        - dims: List[int] - hidden layer dimensions (required)
        - activation: str - activation function (default: "relu")
        - dropout: float - dropout probability (default: 0.0)
        - use_bn: bool - use batch normalization (default: False)
    """
    return MLPExpert(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_dims=list(spec.get("dims", [])),
        activation=str(spec.get("activation", "relu")),
        dropout=float(spec.get("dropout", 0.0)),
        use_bn=bool(spec.get("use_bn", False)),
    )


@register_expert("crossnet_v2")
def build_crossnet_v2_expert(spec: Dict[str, Any], in_dim: int, out_dim: int) -> nn.Module:
    """
    Build CrossNet-v2 expert from spec.
    
    Spec fields:
        - num_layers: int - number of cross layers (default: 3)
        - low_rank: int | None - low-rank dimension (default: None, full-rank)
        - use_bias: bool - use bias in cross layers (default: True)
        - proj.enabled: bool - whether to project output (default: True if in_dim != out_dim)
        - proj.out_dim: int - projection output dim (must equal out_dim if specified)
        - proj.activation: str | None - projection activation (default: None)
    """
    proj_cfg = spec.get("proj", {}) or {}
    proj_enabled = bool(proj_cfg.get("enabled", True))
    proj_out_dim = proj_cfg.get("out_dim")
    
    # Validate projection output dimension
    if proj_out_dim is not None and int(proj_out_dim) != out_dim:
        logger.warning(
            f"CrossNet-v2 proj.out_dim ({proj_out_dim}) != required out_dim ({out_dim}), "
            f"using out_dim={out_dim}"
        )
    
    return CrossNetV2Expert(
        in_dim=in_dim,
        out_dim=out_dim,
        num_layers=int(spec.get("num_layers", 3)),
        low_rank=spec.get("low_rank"),
        use_bias=bool(spec.get("use_bias", True)),
        proj_enabled=proj_enabled,
        proj_activation=proj_cfg.get("activation"),
    )


@register_expert("identity")
def build_identity_expert(spec: Dict[str, Any], in_dim: int, out_dim: int) -> nn.Module:
    """
    Build identity expert from spec.
    
    Spec fields:
        - proj_activation: str | None - activation for dimension projection (default: None)
    """
    return IdentityExpert(
        in_dim=in_dim,
        out_dim=out_dim,
        proj_activation=spec.get("proj_activation"),
    )


# =============================================================================
# Main factory function
# =============================================================================
def build_expert(
    spec: Dict[str, Any],
    in_dim: int,
    out_dim: int,
) -> nn.Module:
    """
    Build an expert module from specification.
    
    Args:
        spec: Expert specification dict. Must contain "type" field.
              Additional fields depend on expert type.
        in_dim: Input dimension (from backbone/composer output)
        out_dim: Required output dimension (must match expert_out_dim for gate mixing)
        
    Returns:
        nn.Module: Expert module with guaranteed output_dim attribute
        
    Raises:
        ValueError: If expert type is not registered or spec is invalid
        
    Example:
        >>> spec = {"type": "mlp", "dims": [256, 128], "activation": "relu"}
        >>> expert = build_expert(spec, in_dim=64, out_dim=128)
        >>> assert expert.output_dim == 128
    """
    if not isinstance(spec, dict):
        raise ValueError(f"Expert spec must be a dict, got {type(spec)}")
    
    expert_type = str(spec.get("type", "")).lower()
    if not expert_type:
        raise ValueError("Expert spec must contain 'type' field")
    
    if expert_type not in EXPERT_REGISTRY:
        available = list(EXPERT_REGISTRY.keys())
        raise ValueError(f"Unknown expert type '{expert_type}'. Available: {available}")
    
    builder = EXPERT_REGISTRY[expert_type]
    expert = builder(spec, in_dim, out_dim)
    
    # Validate output dimension
    if not hasattr(expert, "output_dim"):
        raise RuntimeError(f"Expert {expert_type} does not have output_dim attribute")
    if expert.output_dim != out_dim:
        raise RuntimeError(
            f"Expert {expert_type} output_dim ({expert.output_dim}) != required out_dim ({out_dim})"
        )
    
    return expert


def build_expert_list(
    specs: List[Dict[str, Any]],
    in_dim: int,
    out_dim: int,
) -> nn.ModuleList:
    """
    Build a list of experts from specifications.
    
    Args:
        specs: List of expert specifications
        in_dim: Input dimension
        out_dim: Required output dimension (same for all experts)
        
    Returns:
        nn.ModuleList: List of expert modules
        
    Raises:
        ValueError: If any spec is invalid or specs list is empty
    """
    if not specs:
        raise ValueError("Expert specs list cannot be empty")
    
    experts = nn.ModuleList()
    for i, spec in enumerate(specs):
        try:
            expert = build_expert(spec, in_dim, out_dim)
            # Store expert name for logging
            name = spec.get("name", f"expert_{i}")
            expert._expert_name = name
            experts.append(expert)
        except Exception as e:
            raise ValueError(f"Failed to build expert[{i}]: {e}") from e
    
    return experts


__all__ = [
    "EXPERT_REGISTRY",
    "build_expert",
    "build_expert_list",
    "CrossNetV2",
    "CrossNetV2Expert",
    "MLPExpert",
    "IdentityExpert",
]
