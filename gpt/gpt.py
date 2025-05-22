"""
Source: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
from typing import Optional

import torch
from torch import nn

from .block import Block
from .kvcache import KVCache


class GPT(nn.Module):

    def __init__(self, d, H, T, V, layers, bias=False, dropout=0.2, ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        V: size of the token vocabulary
        layers: number of decoder-only blocks
        bias: whether to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.wte = nn.Embedding(V, d)  # token embeddings
        self.wpe = nn.Embedding(T, d)  # position embeddings
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(i, d, H, T, bias, dropout) for i in range(layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V, bias=bias)

    def forward(self, idx, kv_cache: Optional[KVCache] = None):
        # idx is a [B, T] matrix of token indices
        # targets is a [B, T] matrix of target (next) token indices
        device = idx.device
        _, T = idx.size()  # [B, T]

        if kv_cache is not None:
            cache_len = kv_cache.cache_lens[0]  # assume same length across all layers
            pos = torch.arange(cache_len, cache_len + T, dtype=torch.long, device=device)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=device)

        # generate token and position embeddings
        tok_emb = self.wte(idx)  # [B, T, d]
        pos_emb = self.wpe(pos)  # [T, d]
        x = self.drop(tok_emb + pos_emb)

        # pass through all decoder-only blocks
        for block in self.blocks:
            x = block(x, kv_cache)
        x = self.ln_f(x)  # final layer norm

        # logits of predict tokens
        logits = self.head(x[:, -1])
        return logits
