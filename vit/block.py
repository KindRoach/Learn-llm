import torch.nn as nn

from .ffnn import FFNN
from .self_att import MultiHeadSelfAttention


class Block(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffnn = FFNN(dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Residual connection
        x = x + self.ffnn(self.norm2(x))  # Residual connection
        return x
