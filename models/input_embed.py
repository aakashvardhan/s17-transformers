import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initialize the InputEmbedding module.

        Args:
            d_model (int): The dimensionality of the embedding.
            vocab_size (int): The size of the vocabulary.

        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # randomly initialized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the InputEmbedding module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The embedded input tensor of shape (batch_size, sequence_length, d_model).

        """
        return self.embedding(x) * math.sqrt(self.d_model)
