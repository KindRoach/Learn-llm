import torch

from bert import BERT

batch_size = 4
seq_length = 1024
embedding_dim = 768
num_heads = 12
max_seq_length = 8196
vocab_size = 100000
num_classes = 100
num_layers = 28

model = BERT(
    d=embedding_dim,
    H=num_heads,
    T=max_seq_length,
    V=vocab_size,
    C=num_classes,
    layers=num_layers,
)

input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

with torch.no_grad():
    model.eval()
    logits = model(input_ids)

print(f"Inference logits shape: {logits.shape}")
