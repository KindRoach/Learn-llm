"""
Modified from: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import math

import torch.nn.functional as F
from torch import nn

from .flash_attn.triton_impl import flash_attention_triton


class CausalSelfAttention(nn.Module):

    def __init__(self, d, H, bias=False, dropout=0.2, flash_attn=False):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        bias: whether to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        assert d % H == 0

        self.flash_attn = flash_attn

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

    def forward(self, x):
        B, T, _ = x.size()  # batch size, sequence length, embedding dimensionality

        # compute query, key, and value vectors for all heads in batch
        # split the output into separate query, key, and value tensors
        q, k, v = self.c_attn(x).split(self.d, dim=2)  # [B, T, d]

        # reshape tensor into sequences of smaller token vectors for each head
        # [B, H, T, d // H]
        q = q.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        k = k.view(B, T, self.H, self.d // self.H).transpose(1, 2)
        v = v.view(B, T, self.H, self.d // self.H).transpose(1, 2)

        if self.flash_attn:
            y = flash_attention_triton(q, k, v)
            y = y.view(B, T, -1)
            return y
        else:
            # compute the attention matrix, and apply dropout
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # [B, H, T, T]
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # compute output vectors for each token
            y = att @ v  # [B, H, T, d // H]

            # concatenate outputs from each attention head and linearly project
            y = y.transpose(1, 2).contiguous().view(B, T, self.d)
            y = self.resid_dropout(self.c_proj(y))
            return y
