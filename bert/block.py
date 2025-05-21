"""
Modified from: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

from torch import nn

from .ffnn import FFNN
from .self_att import CausalSelfAttention


class Block(nn.Module):
    def __init__(self, d, H, bias=False, dropout=0.2, ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        bias: whether to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, H, bias, dropout)
        self.ln_2 = nn.LayerNorm(d)
        self.ffnn = FFNN(d, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffnn(self.ln_2(x))
        return x
