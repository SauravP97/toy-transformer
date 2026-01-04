"""Feed Forward layer."""

import torch.nn as nn


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, embedding_dimension):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, 4 * embedding_dimension),
            nn.ReLU(),
            nn.Linear(4 * embedding_dimension, embedding_dimension),
        )

    def forward(self, x):
        return self.feed_forward(x)  # (B, T, embedding_dimension)
