import math

import torch
import torch.nn.functional as F


def flash_decode_torch_q1(q, k, v, l_chunk_size=128, kv_chunk_size=32):
    B, H, T, D = q.shape
    _, _, L, _ = k.shape
    assert T == 1

    scale = 1.0 / (D ** 0.5)
    device = q.device

    l_chunk_num = int(math.ceil(L / l_chunk_size))
    per_l_lse = torch.zeros((B, H, l_chunk_num), device=device)  # [B, H, l_chunk_num]
    per_l_output = torch.zeros((B, H, l_chunk_num, D), device=device)  # [B, H, l_chunk_num, D]

    for ls in range(0, L, l_chunk_size):
        le = min(ls + l_chunk_size, L)

        # for each l chunk, perform flash attention
        max_score = torch.full((B, H, 1), float("-inf"), device=device)  # [B, H, 1]
        exp_sum = torch.zeros((B, H, 1), device=device)  # [B, H, 1]
        out_chunk = torch.zeros((B, H, 1, D), device=device)  # [B, H, 1, D]

        for ks in range(ls, le, kv_chunk_size):
            ke = min(ks + kv_chunk_size, le)
            k_chunk = k[:, :, ks:ke]  # [B, H, Ck, D]
            v_chunk = v[:, :, ks:ke]  # [B, H, Ck, D]

            attn_scores = torch.matmul(q, k_chunk.transpose(-1, -2)) * scale  # [B, H, 1, Ck]

            block_max = attn_scores.max(dim=-1).values  # [B, H, 1]
            max_score_new = torch.maximum(max_score, block_max)  # [B, H, 1]
            exp_scores = torch.exp(attn_scores - max_score_new.unsqueeze(-1))  # [B, H, 1, Ck]

            exp_max_diff = torch.exp(max_score - max_score_new)  # [B, H, 1]
            exp_sum = exp_max_diff * exp_sum + exp_scores.sum(dim=-1)  # [B, H, 1]
            out_chunk = exp_max_diff.unsqueeze(-1) * out_chunk + torch.matmul(exp_scores, v_chunk)  # [B, H, 1, D]

            max_score = max_score_new

        out_chunk = out_chunk / exp_sum.unsqueeze(-1)

        # record output and lse for each l chunk
        li = ls // l_chunk_size
        per_l_lse[:, :, li: li + 1] = torch.log(exp_sum) + max_score  # [B, H, 1]
        per_l_output[:, :, li: li + 1] = out_chunk  # [B, H, 1, D]

    # reduce refer to https://github.com/Dao-AILab/flash-attention/issues/1248
    lse_final = torch.logsumexp(per_l_lse, dim=2, keepdim=True)  # [B, H, 1]
    exp_final = torch.exp(per_l_lse - lse_final).unsqueeze(2)  # [B, H, 1, l_chunk_num]
    output_final = torch.matmul(exp_final, per_l_output)  # [B, H, 1, D]

    return output_final


def flash_decode_torch(q, k, v, q_chunk_size=32, l_chunk_size=128, kv_chunk_size=32):
    B, H, T, D = q.shape
    _, _, L, _ = k.shape
    scale = 1.0 / (D ** 0.5)
    device = q.device

    output = torch.zeros_like(q)

    for qs in range(0, T, q_chunk_size):
        qe = min(qs + q_chunk_size, T)
        q_chunk = q[:, :, qs:qe]  # [B, H, Cq, D]

        l_chunk_num = int(math.ceil(L / l_chunk_size))
        per_l_lse = torch.zeros((B, H, qe - qs, l_chunk_num,), device=device)  # [B, H, Cq, l_chunk_num]
        per_l_output = torch.zeros((B, H, qe - qs, l_chunk_num, D), device=device)  # [B, H, Cq, l_chunk_num, D]

        for ls in range(0, L, l_chunk_size):
            le = min(ls + l_chunk_size, L)

            max_score = torch.full((B, H, qe - qs), float("-inf"), device=device)  # [B, H, Cq]
            exp_sum = torch.zeros((B, H, qe - qs), device=device)  # [B, H, Cq]
            out_chunk = torch.zeros((B, H, qe - qs, D), device=device)  # [B, H, Cq, D]

            for ks in range(ls, le, kv_chunk_size):
                ke = min(ks + kv_chunk_size, le)
                k_chunk = k[:, :, ks:ke]  # [B, H, Ck, D]
                v_chunk = v[:, :, ks:ke]  # [B, H, Ck, D]

                attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scale  # [B, H, Cq, Ck]

                block_max = attn_scores.max(dim=-1).values  # [B, H, Cq]
                max_score_new = torch.maximum(max_score, block_max)  # [B, H, Cq]
                exp_scores = torch.exp(attn_scores - max_score_new.unsqueeze(-1))  # [B, H, Cq, Ck]

                exp_max_diff = torch.exp(max_score - max_score_new)  # [B, H, Cq]
                exp_sum = exp_max_diff * exp_sum + exp_scores.sum(dim=-1)  # [B, H, Cq]
                out_chunk = exp_max_diff.unsqueeze(-1) * out_chunk + torch.matmul(exp_scores, v_chunk)  # [B, H, Cq, D]

                max_score = max_score_new

            out_chunk = out_chunk / exp_sum.unsqueeze(-1)

            li = ls // l_chunk_size
            per_l_lse[:, :, :, li: li + 1] = (torch.log(exp_sum) + max_score).unsqueeze(3)  # [B, H, Cq, 1]
            per_l_output[:, :, :, li: li + 1] = out_chunk.unsqueeze(3)  # [B, H, Cq, 1, D]

        lse_final = torch.logsumexp(per_l_lse, dim=3, keepdim=True)  # [B, H, 1, Cq]
        exp_final = torch.exp(per_l_lse - lse_final).unsqueeze(3)  # [B, H, l_chunk_num, Cq]
        output_final = torch.matmul(exp_final, per_l_output).squeeze(3)  # [B, H, Cq, D]
        output[:, :, qs:qe] = output_final

    return output


def reference_decode(q, k, v) -> torch.Tensor:
    """
    Naive scaled dot-product attention.
    q: [B, H, T, D]
    k, v: [B, H, L, D]
    """
    scale = 1.0 / (q.shape[-1] ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, L]
    attn_probs = F.softmax(attn_scores, dim=-1)
    out = torch.matmul(attn_probs, v)  # [B, H, T, D]
    return out


def run_test(B=2, T=16, L=1024, H=4, D=64, atol=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Testing FlashDecoding on device: {device}")

    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)

    if T == 1:
        out_flash = flash_decode_torch_q1(q, k, v)
    else:
        out_flash = flash_decode_torch(q, k, v)

    out_ref = reference_decode(q, k, v)

    max_diff = (out_flash - out_ref).abs().max().item()
    print(f"Max difference vs reference: {max_diff:.6f}")
    if max_diff > atol:
        print("❌ Test FAILED: difference too large.")
    else:
        print("✅ Test PASSED: output is close to reference.")


if __name__ == "__main__":
    run_test()
