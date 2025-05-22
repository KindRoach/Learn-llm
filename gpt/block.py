"""
Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
from typing import Optional

from torch import nn

from .ffnn import FFNN
from .kvcache import KVCache
from .self_att import CausalSelfAttention


class Block(nn.Module):
    def __init__(self, layer_id, d, H, T, bias=False, dropout=0.2, ):
        """
        Arguments:
        layer_di: id of layer to get KVCache
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        bias: whether to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(layer_id, d, H, T, bias, dropout)
        self.ln_2 = nn.LayerNorm(d)
        self.ffnn = FFNN(d, bias, dropout)

    def forward(self, x, kv_cache: Optional[KVCache] = None):
        x = x + self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + self.ffnn(self.ln_2(x))
        return x
