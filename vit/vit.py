import torch.nn as nn

from .block import TransformerBlock
from .emb import Embedding


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12, dropout=0.2):
        super().__init__()
        self.embedding = Embedding(img_size, patch_size, in_channels, embed_dim, dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        logits = self.head(x[:, 0])
        return logits
