import torch
from tqdm import tqdm

from gpt import GPT

batch_size = 4
seq_length = 1024
embedding_dim = 768
num_heads = 12
max_seq_length = 8196
vocab_size = 100000
num_layers = 28

model = GPT(
    d=embedding_dim,
    H=num_heads,
    T=max_seq_length,
    V=vocab_size,
    layers=num_layers,
).cuda()
model.eval()

input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).cuda()

N = 1000
with torch.no_grad():
    logits = model(input_ids)
    for _ in tqdm(range(N), desc="native"):
        logits = model(input_ids)

    model.compile()
    logits = model(input_ids)
    for _ in tqdm(range(N), desc="compile"):
        logits = model(input_ids)

print(f"Inference logits shape: {logits.shape}")
