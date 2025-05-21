import torch

from vit import ViT

batch_size = 4
img_size = 224
patch_size = 16
in_channels = 3
num_classes = 1000
embed_dim = 768
depth = 12
num_heads = 12
dropout = 0.1

# Create the model
model = ViT(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    num_classes=num_classes,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    dropout=dropout
)

dummy_input = torch.randn(batch_size, in_channels, img_size, img_size)

# Perform a forward pass
output = model(dummy_input)

# Print the output shape
print("Inference logits shape:", output.shape)
