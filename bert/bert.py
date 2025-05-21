"""
Modified from: https://github.com/karpathy/nanoGPT/blob/master/model.py
"""

import torch
from torch import nn

from .block import Block


class BERT(nn.Module):

    def __init__(self, d, H, T, V, C, layers, bias=False, dropout=0.2, ):
        """
        Arguments:
        d: size of embedding dimension
        H: number of attention heads
        T: maximum length of input sequences (in tokens)
        V: size of the token vocabulary
        C: size of the class num
        layers: number of decoder-only blocks
        bias: whether to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.wte = nn.Embedding(V, d)  # token embeddings
        self.wpe = nn.Embedding(T, d)  # position embeddings
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d, H, bias, dropout) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, C, bias=bias)

    def forward(self, idx):
        # idx is a [B, T] matrix of token indices
        # targets is a [B, T] matrix of target (next) token indices
        device = idx.device
        _, T = idx.size()  # [B, T]
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # generate token and position embeddings
        tok_emb = self.wte(idx)  # [B, T, d]
        pos_emb = self.wpe(pos)  # [T, d]
        x = self.drop(tok_emb + pos_emb)

        # pass through all encoder-only blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)  # final layer norm

        # logits of predict class
        logits = self.head(x[:, 0])
        return logits
