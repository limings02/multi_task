from __future__ import annotations

from typing import List

import torch
from torch import nn

from src.models.backbones.layers import MLP


class TaskHead(nn.Module):
    """
    Small tower head for a single task. Produces raw logits (no activation).

    When `out=1`, forward returns shape [B] for convenient loss consumption.
    """

    def __init__(
        self,
        in_dim: int,
        mlp_dims: List[int],
        out: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        self.out = int(out)
        self.mlp_hidden_dims = list(mlp_dims)
        self.mlp = MLP(
            input_dim=in_dim,
            hidden_dims=mlp_dims,
            activation=activation,
            dropout=dropout,
            use_bn=use_bn,
        )
        self.out_proj = nn.Linear(self.mlp.output_dim, self.out)
        # Tighter init to keep initial logits near zero; biases zero.
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.01)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.mlp(h)
        logit = self.out_proj(x)
        if self.out == 1:
            # squeeze last dim so downstream losses expect [B]
            logit = logit.squeeze(-1)
        return logit


__all__ = ["TaskHead"]
