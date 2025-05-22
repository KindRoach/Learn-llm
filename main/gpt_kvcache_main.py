import torch
from tqdm import tqdm

from gpt import GPT, KVCache

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

kv_cache = KVCache(
    num_layers=num_layers,
    B=batch_size,
    H=num_heads,
    max_T=max_seq_length,
    d_head=embedding_dim // num_heads,
).to(device)

seq_length = 1024
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

with torch.no_grad():
    # Prefill
    with tqdm(total=seq_length, desc="Prefill", unit="token") as pbar:
        logits = model(input_ids, kv_cache=kv_cache)
        pbar.update(seq_length)

    # Decode
    for _ in tqdm(range(max_seq_length - seq_length), desc="Decode", unit="token"):
        new_token_id = logits.argmax(dim=-1).view(batch_size, 1)
        logits = model(new_token_id, kv_cache=kv_cache)
