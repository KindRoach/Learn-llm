import torch
from tqdm import tqdm

from gpt import GPT

batch_size = 4
embedding_dim = 768
num_heads = 12
max_seq_length = 8196
vocab_size = 100000
num_layers = 28
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = GPT(
    d=embedding_dim,
    H=num_heads,
    T=max_seq_length,
    V=vocab_size,
    layers=num_layers,
).to(device)

model.eval()
model.compile()

seq_length = 1024
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

# tiger JIT compile
logits = model(input_ids)

with torch.no_grad():
    # Prefill
    with tqdm(total=seq_length, desc="Prefill", unit="token") as pbar:
        logits = model(input_ids)
        pbar.update(seq_length)

    # Decode
    for _ in tqdm(range(max_seq_length - seq_length), desc="Decode", unit="token"):
        new_token_id = logits.argmax(dim=-1).view(batch_size, 1)
        input_ids = torch.cat((input_ids, new_token_id), dim=-1)
        logits = model(input_ids)
