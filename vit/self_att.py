import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_emb, num_heads, dropout=0.2):
        super().__init__()
        assert d_emb % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_emb // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_emb, d_emb * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_emb, d_emb)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, L, d_k)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)  # (B, h, L, d_k)

        # merge multi-head output
        out = out.transpose(1, 2).contiguous().view(B, L, C)  # (B, L, C)
        out = self.proj_drop(self.proj(out))
        return out
