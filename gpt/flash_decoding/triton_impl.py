import math

import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def triton_flash_decode_parallel_stage(
        Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
        B, H, L, L_chunk_num,
        D: tl.constexpr,
        BLOCK_SIZE_L: tl.constexpr,
        BLOCK_SIZE_KV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    l_block_id = tl.program_id(2)

    q_offset = (batch_id * H + head_id) * D + tl.arange(0, D)
    q = tl.load(Q_ptr + q_offset)

    kv_offset = (batch_id * H + head_id) * L * D + l_block_id * BLOCK_SIZE_L * D

    k_block_ptr = tl.make_block_ptr(
        K_ptr + kv_offset,
        (BLOCK_SIZE_L, D),
        (D, 1),
        (0, 0),
        (BLOCK_SIZE_KV, D),
        (0, 1)
    )

    v_block_ptr = tl.make_block_ptr(
        V_ptr + kv_offset,
        (BLOCK_SIZE_L, D),
        (D, 1),
        (0, 0),
        (BLOCK_SIZE_KV, D),
        (0, 1)
    )

    scale = 1.0 / (D ** 0.5)

    max_score = tl.full((1,), float("-inf"), dtype=tl.float32)
    exp_sum = tl.zeros((1,), dtype=tl.float32)
    out_chunk = tl.zeros((1, D), dtype=tl.float32)

    for _ in range(0, BLOCK_SIZE_L, BLOCK_SIZE_KV):
        k_chunk = tl.load(k_block_ptr)
        v_chunk = tl.load(v_block_ptr)
        att = tl.sum(q * k_chunk, axis=1) * scale

        chunk_max = tl.max(att)
        max_score_new = tl.maximum(max_score, chunk_max)
        exp_scores = tl.exp(att - max_score_new)

        exp_max_diff = tl.exp(max_score - max_score_new)
        exp_sum = exp_max_diff * exp_sum + tl.sum(exp_scores)
        out_chunk = exp_max_diff[:, None] * out_chunk + tl.sum(exp_scores * v_chunk.T, axis=1)

        max_score = max_score_new
        k_block_ptr = k_block_ptr.advance((BLOCK_SIZE_KV, 0))
        v_block_ptr = v_block_ptr.advance((BLOCK_SIZE_KV, 0))

    out_chunk = out_chunk / exp_sum[:, None]
    lse = tl.log(exp_sum) + max_score

    lse_offset = (batch_id * H + head_id) * L_chunk_num + l_block_id + tl.arange(0, 1)
    tl.store(LSE_ptr + lse_offset, lse)

    out_offset = ((batch_id * H + head_id) * L_chunk_num + l_block_id) * D + tl.arange(0, D)
    tl.store(O_ptr + out_offset, out_chunk.reshape(D, ))


def flash_decode_triton_q1(q, k, v, l_chunk_size=128, kv_chunk_size=32):
    B, H, T, D = q.shape
    _, _, L, _ = k.shape

    assert T == 1

    l_chunk_num = int(math.ceil(L / l_chunk_size))
    per_l_lse = torch.zeros((B, H, l_chunk_num), device=q.device)
    per_l_output = torch.zeros((B, H, l_chunk_num, D), device=q.device)

    grid = (B, H, l_chunk_num)
    triton_flash_decode_parallel_stage[grid](
        q, k, v, per_l_output, per_l_lse,
        B, H, L, l_chunk_num, D,
        BLOCK_SIZE_L=l_chunk_size,
        BLOCK_SIZE_KV=kv_chunk_size
    )

    # reduce refer to https://github.com/Dao-AILab/flash-attention/issues/1248
    lse_final = torch.logsumexp(per_l_lse, dim=2, keepdim=True)  # [B, H, 1]
    exp_final = torch.exp(per_l_lse - lse_final).unsqueeze(2)  # [B, H, 1, l_chunk_num]
    output = torch.matmul(exp_final, per_l_output)  # [B, H, 1, D]

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


def run_test(B=2, L=1024, H=4, D=64, atol=1e-4, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Testing FlashDecoding on device: {device}")

    torch.manual_seed(42)
    q = torch.randn(B, H, 1, D, device=device)
    k = torch.randn(B, H, L, D, device=device)
    v = torch.randn(B, H, L, D, device=device)

    out_flash = flash_decode_triton_q1(q, k, v)
    out_ref = reference_decode(q, k, v)

    max_diff = (out_flash - out_ref).abs().max().item()
    print(f"Max difference vs reference: {max_diff:.6f}")
    if max_diff > atol:
        print("❌ Test FAILED: difference too large.")
    else:
        print("✅ Test PASSED: output is close to reference.")


if __name__ == "__main__":
    run_test()
