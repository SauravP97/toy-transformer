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
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.current_pos = 0

    def forward(self, x, kv_cache_enabled=False):
        B, T, C = x.shape
        k_current = self.key(x)  # B,T,head_size
        q = self.query(x)  # B,T,head_size

        # perform the weighted aggregation of the values
        v_current = self.value(x)  # B,T,head_size

        if kv_cache_enabled:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_current, v_current
            else:
                self.cache_k = torch.cat([self.cache_k, k_current], dim=1)
                self.cache_v = torch.cat([self.cache_v, v_current], dim=1)
            k, v = self.cache_k, self.cache_v
        else:
            k, v = k_current, v_current

        # compute attention scores ("affinities")
        # (B,T,head_size) @ (B,head_size, T)  -> (B,T,T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # allow tokens to only talk to the tokens at previous positions.
        # avoid tokens to talk to the future tokens!
        # This is a decoder only style attention.
        if kv_cache_enabled:
            wei = wei.masked_fill(
                self.tril[self.current_pos:self.current_pos + q.shape[-2], :k.shape[-2]] == 0,
                float("-inf")
            )  # B,T1,T
            self.current_pos += q.shape[-2]
        else:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # B,T,T
        
        wei = F.softmax(wei, dim=-1)  # B,T,T

        out = wei @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

    def reset_kv_cache(self):
        self.cache_k,self.cache_v  = None, None
        self.current_pos = 0