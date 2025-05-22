"""
Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .kvcache import KVCache


class CausalSelfAttention(nn.Module):

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
        assert d % H == 0

        self.layer_id = layer_id

        # key, query, value projections for all heads, but in a batch
        # output is 3X the dimension because it includes key, query and value
        self.c_attn = nn.Linear(d, 3 * d, bias=bias)

        # projection of concatenated attention head outputs
        self.c_proj = nn.Linear(d, d, bias=bias)

        # dropout modules
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.H = H
        self.d = d

        # causal mask to ensure that attention is only applied to
        # the left in the input sequence
        self.register_buffer(
            "mask", torch.tril(torch.ones((T, T), dtype=torch.bool)).view(1, 1, T, T)
        )

    def forward(self, x, kv_cache: Optional[KVCache] = None):
        B, T, _ = x.size()  # batch size, sequence length, embedding dimensionality

        # compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v = self.c_attn(x).split(self.d, dim=2)  # [B, T, d]

        # reshape tensor into sequences of smaller token vectors for each head
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2)  # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache.get(self.layer_id)
            kv_cache.update(self.layer_id, k, v)
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # compute the attention matrix, perform masking, and apply dropout
        T_k = k.size(2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T_k] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # compute output vectors for each token
        y = att @ v  # [B, H, T, d // H]

        # concatenate outputs from each attention head and linearly project
        y = y.transpose(1, 2).contiguous().view(B, T, self.d)
        y = self.resid_dropout(self.c_proj(y))
        return y
