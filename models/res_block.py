import math
import torch
import torch.nn as nn
from models.layer_norm import LayerNorm


class ResidualConnection(nn.Module):
    """
    Residual Connection module that applies residual connection to the input tensor.

    Args:
        dropout (float): The dropout probability.
    """

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Forward pass of the ResidualConnection module.

        Args:
            x (torch.Tensor): The input tensor.
            sublayer (nn.Module): The sublayer module to be applied.

        Returns:
            torch.Tensor: The output tensor after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))
