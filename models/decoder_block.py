import math
import torch
import torch.nn as nn

from models.layer_norm import LayerNorm
from models.attention_block import MultiHeadAttentionBlock
from models.feed_forward import FeedForwardBlock
from models.res_block import ResidualConnection


class DecoderBlock(nn.Module):
    """
    Decoder block in a transformer model.

    Args:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block for the decoder.
        cross_attention_block (MultiHeadAttentionBlock): The cross-attention block for the decoder.
        feed_forward_block (FeedForwardBlock): The feed-forward block for the decoder.
        dropout (float): The dropout rate.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): The self-attention block for the decoder.
        cross_attention_block (MultiHeadAttentionBlock): The cross-attention block for the decoder.
        feed_forward_block (FeedForwardBlock): The feed-forward block for the decoder.
        residual_connections (nn.ModuleList): List of residual connections.

    """

    def __init__(
        self,
        self_attention_block: MultiHeadAttentionBlock,
        cross_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        dropout: float,
    ):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the decoder block.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_output (torch.Tensor): The output from the encoder.
            src_mask (torch.Tensor): The mask for the source sequence.
            tgt_mask (torch.Tensor): The mask for the target sequence.

        Returns:
            torch.Tensor: The output tensor.

        """
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask)
        )  # decoder input is passed to self attention block
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask
            ),
        )
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    The Decoder module of the transformer model.

    Args:
        layers (nn.ModuleList): List of decoder layers.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        norm (LayerNorm): Layer normalization module.

    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the decoder module.

        Args:
            x (torch.Tensor): Input tensor.
            encoder_output (torch.Tensor): Output tensor from the encoder module.
            src_mask (torch.Tensor): Mask for the source sequence.
            tgt_mask (torch.Tensor): Mask for the target sequence.

        Returns:
            torch.Tensor: Output tensor after passing through the decoder module.

        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
