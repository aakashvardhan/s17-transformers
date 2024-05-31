import math
import torch
import torch.nn as nn


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block module.

    Args:
        d_model (int): The input and output dimension of the block.
        h (int): The number of attention heads.
        dropout (float): The dropout probability.

    Attributes:
        d_model (int): The input and output dimension of the block.
        h (int): The number of attention heads.
        d_k (int): The dimension of each attention head.
        w_q (nn.Linear): Linear layer for query projection.
        w_k (nn.Linear): Linear layer for key projection.
        w_v (nn.Linear): Linear layer for value projection.
        w_o (nn.Linear): Linear layer for output projection.
        dropout (nn.Dropout): Dropout layer for attention scores.

    Methods:
        attention(query, key, value, mask=None, dropout=None):
            Computes the attention scores and weighted sum of values.
        forward(q, k, v, mask):
            Performs the forward pass of the attention block.

    """

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0  # d_model must be divisible by h

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Computes the attention scores and weighted sum of values.

        Args:
            query (torch.Tensor): The query tensor.
            key (torch.Tensor): The key tensor.
            value (torch.Tensor): The value tensor.
            mask (torch.Tensor, optional): The mask tensor for masking certain positions.
            dropout (nn.Dropout, optional): The dropout layer for attention scores.

        Returns:
            torch.Tensor: The weighted sum of values.
            torch.Tensor: The attention scores.

        """
        # mask can be encoder or decoder mask
        d_k = query.size(-1)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            # where ever mask is 0, replace with -1e9
        attention_scores = attention_scores.softmax(dim=-1)
        # (batch_size, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch_size, h, seq_len) -> (batch_size, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Performs the forward pass of the attention block.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.
            mask (torch.Tensor): The mask tensor for masking certain positions.

        Returns:
            torch.Tensor: The output tensor.

        """
        query = self.w_q(
            q
        )  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key = self.w_k(
            k
        )  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.w_v(
            v
        )  # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # Concatenate all heads
        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.h * self.d_k)
        # Memory is contiguous in memory for better performance
        return self.w_o(x)
