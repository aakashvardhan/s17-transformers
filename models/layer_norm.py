import math

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Args:
        eps (float): A small value added to the denominator for numerical stability. Default is 10**-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        alpha (nn.Parameter): Learnable parameter for scaling.
        bias (nn.Parameter): Learnable parameter for bias.

    """

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer normalization module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # eps is added to the denominator to avoid division by zero
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
