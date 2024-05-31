import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional embedding module for adding positional information to the input tensor.

    Args:
        d_model (int): The dimensionality of the model.
        seq_len (int): The length of the sequence.
        dropout (float): The dropout probability.

    Attributes:
        d_model (int): The dimensionality of the model.
        seq_len (int): The length of the sequence.
        dropout (nn.Dropout): Dropout layer for regularization.
        pe (torch.Tensor): Positional embeddings of shape (1, seq_len, d_model).

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        Initialize the PositionalEmbedding module.

        Args:
            d_model (int): The dimensionality of the model.
            seq_len (int): The length of the sequence.
            dropout (float): The dropout probability.

        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # This creates [0, 1, 2, 3, ..., seq_len] and unsqueeze to make it a column vector
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # log and exp are used to prevent overflow and numerical instability
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer(
            "pe", pe
        )  # pe is not a parameter, but should be part of the state dict
        # register_buffer is used to make sure that pe is saved and loaded correctly when saving and loading the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the PositionalEmbedding module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with positional embeddings added.

        """

        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # broadcasting (1, seq_len, d_model) to (batch_size, seq_len, d_model)
        return self.dropout(x)
