import math
import torch
import torch.nn as nn

from models.layer_norm import LayerNorm
from models.attention_block import MultiHeadAttentionBlock
from models.feed_forward import FeedForwardBlock
from models.res_block import ResidualConnection


class EncoderBlock(nn.Module):
    """
    A block in the encoder layer of a transformer model.

    Args:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block.
        feed_forward_block (FeedForwardBlock): The feed-forward block.
        dropout (float): The dropout rate.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block.
        feed_forward_block (FeedForwardBlock): The feed-forward block.
        residual_connections (nn.ModuleList): List of residual connections.

    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(2)]
        )

    def forward(self, x, src_mask):
        """
        Forward pass of the encoder block.

        Args:
            x (torch.Tensor): The input tensor.
            src_mask (torch.Tensor): The source mask tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    The Encoder class represents a stack of encoder layers in a transformer model.

    Args:
        layers (nn.ModuleList): A list of encoder layers.

    Attributes:
        layers (nn.ModuleList): A list of encoder layers.
        norm (LayerNorm): Layer normalization module.

    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
            mask (torch.Tensor): Mask tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, hidden_size).

        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
