#!/usr/bin/env python3
"""Numerics for the k3-w1 Triton kernels, against an explicit dequant.

Runs standalone -- no vLLM engine, no model, no cluster. What it checks:

  1. the bit/scale unpack matches the format c/backend_cuda_k3.cu reads
  2. the grouped GEMM matches a dequantize-then-matmul reference
  3. SITU matches vllm's SituAndMul.forward_native formula
  4. a whole MoE forward matches a dense reference
  5. round-tripping a synthetic store through K3W1Store recovers the weights

Usage:  python3 test_numerics.py
"""

import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from vllm_k3_w1.kernels import (  # noqa: E402
    GROUP,
    dequant_reference,
    situ_and_mul,
    w1_gemv,
    w1_grouped_gemm,
)

DEV = "cuda"
AMP = 1.69
FAIL = []


def check(name, got, want, rtol=2e-2, atol=2e-2):
    ok = torch.allclose(got.float(), want.float(), rtol=rtol, atol=atol)
    err = (got.float() - want.float()).abs().max().item()
    denom = want.float().abs().max().item() or 1.0
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} max|err|={err:.3e} "
          f"rel={err / denom:.3e}")
    if not ok:
        FAIL.append(name)
    return ok


def check_norm(name, got, want, tol=1e-2):
    """Relative Frobenius error, for outputs that go through bf16 stages.

    Elementwise allclose is the wrong test once a sum over top_k experts can
    cancel: an output near zero built from terms of magnitude 1e5 keeps an
    absolute error of ~1e3 no matter how correct the kernel is. The norm ratio
    is the accuracy statement that actually means something here, and the
    second number below is the check that would still catch a real bug -- the
    worst elementwise relative error among entries that did not cancel.
    """
    g, w = got.float(), want.float()
    rel = (g - w).norm().item() / (w.norm().item() or 1.0)
    big = w.abs() > 0.01 * w.abs().max()
    worst = (((g - w).abs() / w.abs().clamp_min(1e-9))[big].max().item()
             if big.any() else 0.0)
    ok = rel < tol and worst < 5e-2
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} ||err||/||ref||={rel:.3e} "
          f"worst-elem-rel={worst:.3e}")
    if not ok:
        FAIL.append(name)
    return ok


def rand_store(E, N, K, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    packed = torch.randint(0, 256, (E, N, K // 8), dtype=torch.uint8,
                           device=DEV, generator=g)
    # Keep exponents near the bias so the reference stays in fp32 range.
    scale = torch.randint(120, 135, (E, N, K // GROUP), dtype=torch.uint8,
                          device=DEV, generator=g)
    return packed, scale


def test_unpack():
    print("[1] bit/scale unpack")
    # One expert, one row, one group: bit j of byte b is weight 8b+j, LSB first.
    packed = torch.tensor([[[0b00000001, 0b10000000, 0, 255]]],
                          dtype=torch.uint8, device=DEV)
    scale = torch.tensor([[[127]]], dtype=torch.uint8, device=DEV)
    W = dequant_reference(packed, scale, 1.0)[0, 0]
    want = torch.full((32,), -1.0, device=DEV)
    want[0] = 1.0        # byte0 bit0
    want[15] = 1.0       # byte1 bit7
    want[24:] = 1.0      # byte3 all ones
    check("LSB-first bit order, +/-1", W, want)
    # UE8M0: exponent e means 2^(e-127).
    s2 = torch.tensor([[[130]]], dtype=torch.uint8, device=DEV)
    W2 = dequant_reference(packed, s2, 1.0)[0, 0]
    check("UE8M0 scale = 2^(e-127)", W2, want * 8.0)
    check("amplitude folds in", dequant_reference(packed, scale, AMP)[0, 0],
          want * AMP)


def _align(topk_ids, block_m, num_experts):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    return moe_align_block_size(topk_ids, block_m, num_experts, None)


def test_gemm(E=4, N=128, K=256, T=8, top_k=2):
    print(f"[2] grouped GEMM  E={E} N={N} K={K} T={T} top_k={top_k}")
    torch.manual_seed(0)
    packed, scale = rand_store(E, N, K)
    x = torch.randn(T, K, dtype=torch.bfloat16, device=DEV)
    topk_ids = torch.randint(0, E, (T, top_k), dtype=torch.int32, device=DEV)
    topk_w = torch.rand(T, top_k, dtype=torch.float32, device=DEV)

    sorted_ids, expert_ids, npad = _align(topk_ids, 64, E)
    out = torch.zeros(T * top_k, N, dtype=torch.bfloat16, device=DEV)
    w1_grouped_gemm(x, packed, scale, sorted_ids, expert_ids, npad, topk_w,
                    out, top_k, AMP, mul_routed_weight=False, block_m=64)

    W = dequant_reference(packed, scale, AMP)
    ref = torch.zeros_like(out, dtype=torch.float32)
    for t in range(T):
        for j in range(top_k):
            ref[t * top_k + j] = x[t].float() @ W[topk_ids[t, j]].T
    check("grouped GEMM vs dequant-and-matmul", out, ref)

    # ...and with the routing weight applied on the output.
    out2 = torch.zeros_like(out)
    w1_grouped_gemm(x, packed, scale, sorted_ids, expert_ids, npad, topk_w,
                    out2, top_k, AMP, mul_routed_weight=True, block_m=64)
    check("routed weight applied", out2, ref * topk_w.reshape(-1, 1))


def test_situ():
    print("[3] SITU activation")
    torch.manual_seed(0)
    x = torch.randn(64, 256, dtype=torch.bfloat16, device=DEV) * 3
    beta, lin = 4.0, 25.0
    d = x.shape[-1] // 2
    g, u = x[..., :d].float(), x[..., d:].float()
    ref = (beta * torch.tanh(g / beta) * torch.sigmoid(g)) * (
        lin * torch.tanh(u / lin))
    check("situ vs SituAndMul.forward_native", situ_and_mul(x, beta, lin), ref)


def test_moe_forward(E=8, H=256, I=128, T=4, top_k=3):
    print(f"[4] full MoE forward  E={E} H={H} I={I} T={T} top_k={top_k}")
    torch.manual_seed(0)
    beta, lin = 4.0, 25.0
    w13p, w13s = rand_store(E, 2 * I, H, seed=1)
    w2p, w2s = rand_store(E, H, I, seed=2)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV)
    topk_ids = torch.randint(0, E, (T, top_k), dtype=torch.int32, device=DEV)
    topk_w = torch.rand(T, top_k, dtype=torch.float32, device=DEV)

    sorted_ids, expert_ids, npad = _align(topk_ids, 16, E)

    def grouped(a, p, s, out, tk, mul):
        w1_grouped_gemm(a, p, s, sorted_ids, expert_ids, npad, topk_w, out, tk,
                        AMP, mul, block_m=16)

    def gemv(a, p, s, out, tk, mul):
        w1_gemv(a, p, s, topk_ids, topk_w, None, out, tk, AMP, mul)

    outs = {}
    for name, gemm in (("grouped", grouped), ("gemv", gemv)):
        inter = torch.zeros(T * top_k, 2 * I, dtype=torch.bfloat16, device=DEV)
        gemm(x, w13p, w13s, inter, top_k, False)
        h = situ_and_mul(inter, beta, lin)
        down = torch.zeros(T * top_k, H, dtype=torch.bfloat16, device=DEV)
        gemm(h, w2p, w2s, down, 1, True)
        outs[name] = down.view(T, top_k, H).sum(1)

    W13 = dequant_reference(w13p, w13s, AMP)
    W2 = dequant_reference(w2p, w2s, AMP)
    ref = torch.zeros(T, H, dtype=torch.float32, device=DEV)
    for t in range(T):
        for j in range(top_k):
            e = topk_ids[t, j]
            gu = x[t].float() @ W13[e].T
            g, u = gu[:I], gu[I:]
            act = (beta * torch.tanh(g / beta) * torch.sigmoid(g)) * (
                lin * torch.tanh(u / lin))
            ref[t] += topk_w[t, j] * (act @ W2[e].T)
    for name, got in outs.items():
        check_norm(f"MoE forward ({name}) vs dense reference", got, ref)
    check_norm("gemv matches grouped", outs["gemv"], outs["grouped"], tol=5e-3)


def test_store_roundtrip():
    print("[5] flat-store round trip")
    import os
    import tempfile

    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vllm_k3_w1.loader import K3W1Store

    H, I, E, L = 512, 256, 2, 2      # slot = 2*(I*H/8 + I*H/32) + H*I/8 + H*I/32
    st_bytes = 2 * (I * (H // 8) + I * (H // 32)) + H * (I // 8) + H * (I // 32)
    if st_bytes % 4096:
        print(f"  SKIP  synthetic slot {st_bytes} not 4096-aligned")
        return
    rng = np.random.default_rng(0)
    blobs = [rng.integers(0, 256, st_bytes, dtype=np.uint8) for _ in range(L * E)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "experts.w2")
        with open(p, "wb") as f:
            for b in blobs:
                f.write(b.tobytes())
        store = K3W1Store(d, H, I, E)
        ok = store.slot == st_bytes and store.n_slots == L * E
        print(f"  {'PASS' if ok else 'FAIL'}  slot/geometry           "
              f"slot={store.slot} slots={store.n_slots}")
        if not ok:
            FAIL.append("store geometry")
        s = store.read_slot(1, 1)                      # last slot
        want = blobs[1 * E + 1]
        got = np.concatenate([s[k].reshape(-1) for k in
                              ("w1p", "w1s", "w2p", "w2s", "w3p", "w3s")])
        ok = np.array_equal(got, want)
        print(f"  {'PASS' if ok else 'FAIL'}  slot layout w1p|w1s|w2p|w2s|w3p|w3s")
        if not ok:
            FAIL.append("slot layout")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability()}\n")
    test_unpack()
    test_gemm()
    test_situ()
    test_moe_forward()
    test_store_roundtrip()
    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
