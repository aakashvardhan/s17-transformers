import math

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    A feed-forward block module in a transformer model.

    Args:
        d_model (int): The input and output dimension of the block.
        d_ff (int): The dimension of the intermediate layer.
        dropout (float): The dropout probability.

    Attributes:
        linear1 (nn.Linear): The first linear layer.
        dropout (nn.Dropout): The dropout layer.
        linear2 (nn.Linear): The second linear layer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # Squeeze and expand
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feed-forward block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
    