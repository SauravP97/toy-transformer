"""Transformer Head architecture."""

import torch.nn as nn
import torch

from transformer_head import Head


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, embedding_dimension, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, embedding_dimension, block_size) for _ in range(num_heads)]
        )
        # The linear layer takes embedding_dimension as the input feature count.
        # But the Head returns output of shape: (B, T, head_size)
        # When concatenated by the last dimension the list of heads output is
        # concatenated to: (B, T, head_size * num_heads)
        # It's important for the output shape be able to be fed into the Linear layer
        # And for that: head_size * num_heads = embedding_dimension
        #
        # If the above condition is respected then we can say:
        # head_size * num_heads = embedding_dimension = X
        # Giving:
        # (B, T, X) @ (X, X) -> (B, T, X)
        # Hence, output of the Multi Head attention block: (B, T, X)
        # output: (B, T, embedding_dimension)
        self.proj = nn.Linear(embedding_dimension, embedding_dimension)

    def forward(self, x):
        # The attention Heads can run in parallel enabling each Head to learn a
        # unique behavior about the Training data.
        # The output of the Heads are concatenated by the last dimension.
        # Each Head outputs the shape -> (B, T, head_size)
        # The Multi Head block will return -> (B, T, head_size * num_heads)
        out = torch.cat(
            [head(x) for head in self.heads], dim=-1
        )  # (B, T, head_size * num_heads)
        out = self.proj(out)  # (B, T, embedding_dimension)
        return out
