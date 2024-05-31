import math
import torch
import torch.nn as nn

from models.layer_norm import LayerNorm
from models.attention_block import MultiHeadAttentionBlock
from models.feed_forward import FeedForwardBlock
from models.res_block import ResidualConnection

from models.encoder_block import EncoderBlock
from models.encoder_block import Encoder

from models.decoder_block import DecoderBlock
from models.decoder_block import Decoder

from models.pos_embed import PositionalEncoding
from models.input_embed import InputEmbeddings

import lightning as L

class ProjectionLayer(nn.Module):
    """
    A projection layer that maps the input tensor to the vocabulary size.

    Args:
        d_model (int): The dimensionality of the input tensor.
        vocab_size (int): The size of the vocabulary.

    Attributes:
        proj (nn.Linear): Linear transformation layer.

    Methods:
        forward(x): Performs forward pass through the projection layer.

    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        """
        Performs forward pass through the projection layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, vocab_size).

        """
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """
    A Transformer model that consists of an encoder, a decoder, input embeddings, positional encodings, and a projection layer.

    Args:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embed (InputEmbeddings): The input embeddings for the source sequence.
        tgt_embed (InputEmbeddings): The input embeddings for the target sequence.
        src_pos (PositionalEncoding): The positional encoding for the source sequence.
        tgt_pos (PositionalEncoding): The positional encoding for the target sequence.
        projection_layer (ProjectionLayer): The projection layer.

    Attributes:
        encoder (Encoder): The encoder module.
        decoder (Decoder): The decoder module.
        src_embed (InputEmbeddings): The input embeddings for the source sequence.
        tgt_embed (InputEmbeddings): The input embeddings for the target sequence.
        src_pos (PositionalEncoding): The positional encoding for the source sequence.
        tgt_pos (PositionalEncoding): The positional encoding for the target sequence.
        projection_layer (ProjectionLayer): The projection layer.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encode the source sequence.

        Args:
            src (Tensor): The source sequence tensor.
            src_mask (Tensor): The source mask tensor.

        Returns:
            Tensor: The encoded source sequence.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        """
        Decode the target sequence.

        Args:
            encoder_output (Tensor): The output of the encoder.
            src_mask (Tensor): The source mask tensor.
            tgt (Tensor): The target sequence tensor.
            tgt_mask (Tensor): The target mask tensor.

        Returns:
            Tensor: The decoded target sequence.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, decoder_output):
        """
        Project the decoder output.

        Args:
            decoder_output (Tensor): The output of the decoder.

        Returns:
            Tensor: The projected output.
        """
        return self.projection_layer(decoder_output)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    """
    Build a Transformer model.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum length of source sequences.
        tgt_seq_len (int): Maximum length of target sequences.
        d_model (int, optional): Dimension of the model. Defaults to 512.
        N (int, optional): Number of encoder and decoder layers. Defaults to 6.
        h (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        d_ff (int, optional): Dimension of the feed-forward network. Defaults to 2048.

    Returns:
        Transformer: The constructed Transformer model.
    """
    # Validate input parameters
    if not all(
        isinstance(x, int) and x > 0
        for x in [
            src_vocab_size,
            tgt_vocab_size,
            src_seq_len,
            tgt_seq_len,
            d_model,
            N,
            h,
            d_ff,
        ]
    ):
        raise ValueError("All size parameters must be positive integers.")
    if not (0 <= dropout <= 1):
        raise ValueError("Dropout must be between 0 and 1.")

    # Create Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create Positional Encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create Encoder and Decoder layers
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    return Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )