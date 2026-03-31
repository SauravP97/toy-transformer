"""Transformer architecture."""

import torch.nn as nn
import torch
import torch.nn.functional as F

from attention_block import Block


class Transformer(nn.Module):
    """Transformer module"""

    def __init__(
        self, vocab_size, embedding_dimension, block_size, n_head, n_layer, device
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimension)
        self.positional_embedding_table = nn.Embedding(block_size, embedding_dimension)
        self.blocks = nn.ModuleList(
            [Block(embedding_dimension, n_head, block_size) for _ in range(n_layer)]
        )
        self.layer_norm = nn.LayerNorm(embedding_dimension)
        self.linear = nn.Linear(embedding_dimension, vocab_size)
        self.device = device
        self.block_size = block_size
        self.current_pos = 0

    def forward(self, idx, targets=None, kv_cache_enabled=False):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)  # B, T, embedding_dimension
        
        if kv_cache_enabled:
            positional_embedding = self.positional_embedding_table(
                torch.arange(self.current_pos, self.current_pos + T, device=self.device)
            )
            self.current_pos += T
        else:
            positional_embedding = self.positional_embedding_table(
                torch.arange(T, device=self.device)
            )  # T, embedding_dimension
        
        x = token_embedding + positional_embedding  # B, T, embedding_dimension

        for block in self.blocks:
            x = block(x, kv_cache_enabled=kv_cache_enabled)

        # x = self.blocks(x, kv_cache_enabled=kv_cache_enabled)  # B, T, embedding_dimension
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

    def reset_kv_cache(self):
        for block in self.blocks:
            block.reset_kv_cache()
        self.current_pos = 0

    def generate(self, idx, max_new_tokens=1000):
        for _ in range(max_new_tokens):
            idx_trimmed = idx[:, -self.block_size :]  # B, T
            logits, loss = self.forward(idx_trimmed)  # B, T, embedding_dimension
            next_predicted_logit = logits[:, -1, :]  # B, embedding_dimension
            probs = F.softmax(next_predicted_logit, dim=-1)  # B, 1
            next_predicted_character = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_predicted_character), dim=1)  # B,T+1

        return idx

    def generate_with_kv_cache_enabled(self, idx, max_new_tokens=1000):
        def generate_block(generated_idx, tokens_to_generate):
            self.reset_kv_cache()
            tokens = generated_idx.shape[-1]
            generated_idx_trimmed = generated_idx[:, tokens-1:tokens]
            logits, loss = self.forward(generated_idx_trimmed, kv_cache_enabled=True)

            for _ in range(tokens_to_generate-1):
                next_predicted_logit = logits[:, -1, :]  # B, embedding_dimension
                probs = F.softmax(next_predicted_logit, dim=-1)  # B, 1
                next_predicted_character = torch.multinomial(probs, num_samples=1)
                generated_idx_trimmed = torch.cat(
                    (generated_idx_trimmed, next_predicted_character), 
                    dim=1
                )  # B,T+1
                logits, loss = self.forward(next_predicted_character, kv_cache_enabled=True)
            
            return generated_idx_trimmed

        if max_new_tokens < self.block_size:
            # One partial block to execute
            return torch.cat((idx, generate_block(idx, max_new_tokens)), dim=1)

        num_blocks = max_new_tokens // self.block_size

        if max_new_tokens % self.block_size == 0:
            # Equally divided blocks
            for _ in range(num_blocks):
                idx = torch.cat((idx, generate_block(idx, self.block_size)), dim=1)
        else:
            # Last block is partial
            for _ in range(num_blocks-1):
                idx = torch.cat((idx, generate_block(idx, self.block_size)), dim=1)
            
            idx = torch.cat((idx, generate_block(idx, max_new_tokens%self.block_size)), dim=1)
   
        return idx
