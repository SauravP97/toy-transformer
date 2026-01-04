"""Transformer Head architecture."""

import torch.nn as nn
import torch.nn.functional as F
import torch


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size, embedding_dimension, block_size):
        super().__init__()
        self.key = nn.Linear(embedding_dimension, head_size, bias=False)
        self.query = nn.Linear(embedding_dimension, head_size, bias=False)
        self.value = nn.Linear(embedding_dimension, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B,T,head_size
        q = self.query(x)  # B,T,head_size

        # perform the weighted aggregation of the values
        v = self.value(x)  # B,T,head_size

        # compute attention scores ("affinities")
        # (B,T,head_size) @ (B,head_size, T)  -> (B,T,T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # allow tokens to only talk to the tokens at previous positions.
        # avoid tokens to talk to the future tokens!
        # This is a decoder only style attention.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B,T,T
        wei = F.softmax(wei, dim=-1)  # B,T,T

        out = wei @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out
