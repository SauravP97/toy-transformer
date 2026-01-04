"""Transformer architecture."""

import torch.nn as nn
import torch
import torch.nn.functional as F

from attention_block import Block


class Transformer(nn.Module):
    """Transformer module"""

    def __init__(self, vocab_size, embedding_dimension, block_size, n_head, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimension)
        self.positional_embedding_table = nn.Embedding(block_size, embedding_dimension)
        self.blocks = nn.Sequential(
            *[Block(embedding_dimension, n_head) for _ in range(n_layer)]
        )
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        self.linear = nn.Linear(embedding_dimension, vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)  # B, T, embedding_dimension
        positional_embedding = self.positional_embedding_table(
            torch.arange(T, device=self.device)
        )  # T, embedding_dimension
        x = token_embedding + positional_embedding  # B, T, embedding_dimension

        x = self.blocks(x)  # B, T, embedding_dimension
        x = self.layer_norm(x)  # B, T, embedding_dimension
        logits = self.linear(x)  # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=1000):
        for _ in range(max_new_tokens):
            idx_trimmed = idx[:, -self.block_size :]  # B, T
            logits, loss = self.forward(idx_trimmed)  # B, T, embedding_dimension
            next_predicted_logit = logits[:, -1, :]  # B, embedding_dimension
            probs = F.softmax(next_predicted_logit, dim=-1)  # B, 1
            next_predicted_character = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_predicted_character), dim=1)  # B,T+1

        return idx
