"""Attention Block architecture"""

import torch.nn as nn

from transformer_multi_head import MultiHeadAttention
from feed_forward_layer import FeedForward


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, embedding_dimension, n_head, block_size):
        super().__init__()
        head_size = embedding_dimension // n_head
        self.multi_head_attention = MultiHeadAttention(
            n_head, head_size, embedding_dimension, block_size
        )
        self.feed_forward_layer = FeedForward(embedding_dimension)
        self.ln1 = nn.LayerNorm(embedding_dimension)
        self.ln2 = nn.LayerNorm(embedding_dimension)

    def forward(self, x):
        # Residual Connections: https://en.wikipedia.org/wiki/Residual_neural_network
        x = x + self.multi_head_attention(self.ln1(x))  # (B, T, embedding_dimension)
        x = x + self.feed_forward_layer(self.ln2(x))  # (B, T, embedding_dimension)
        return x
