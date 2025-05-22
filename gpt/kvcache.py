import torch


class KVCache:
    def __init__(self, num_layers: int, B: int, H: int, max_T: int, d_head: int):
        self.max_T = max_T
        self.B = B
        self.H = H
        self.d_head = d_head
        self.k = [torch.zeros(B, H, max_T, d_head) for _ in range(num_layers)]
        self.v = [torch.zeros(B, H, max_T, d_head) for _ in range(num_layers)]
        self.cache_lens = [0 for _ in range(num_layers)]

    def reset(self):
        self.cache_lens = [0 for _ in range(len(self.k))]

    def to(self, device: torch.device):
        self.k = [key.to(device) for key in self.k]
        self.v = [value.to(device) for value in self.v]

    def update(self, layer: int, new_k: torch.Tensor, new_v: torch.Tensor):
        B, H, T_new, D = new_k.size()
        T_cache = self.cache_lens[layer]
        T_total = T_cache + T_new
        max_T = self.max_T

        if T_total <= max_T:
            self.k[layer][:, :, T_cache:T_total, :] = new_k
            self.v[layer][:, :, T_cache:T_total, :] = new_v
            self.cache_lens[layer] = T_total
        else:
            # shift to drop old cache and add new cache
            overflow = T_total - max_T
            self.k[layer] = torch.roll(self.k[layer], shifts=-overflow, dims=2)
            self.v[layer] = torch.roll(self.v[layer], shifts=-overflow, dims=2)
            self.k[layer][:, :, -T_new:, :] = new_k
            self.v[layer][:, :, -T_new:, :] = new_v
            self.cache_lens[layer] = max_T

    def get(self, layer: int):
        L = self.cache_lens[layer]
        return self.k[layer][:, :, :L, :], self.v[layer][:, :, :L, :]
