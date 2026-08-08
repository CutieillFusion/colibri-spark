#!/usr/bin/env python3
"""Numerics for the dense int4-g64 / int8-row kernels.

The quantizers here have to reproduce c/kimi_k3.c exactly -- same group size,
same scale definition, same clamp -- or a vLLM run and an engine run diverge
for reasons that look like a kernel bug. So the first test checks the packing
against a hand-computed case rather than only round-tripping our own code.

Usage:  python3 test_dense.py
"""

import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from vllm_k3_w1.dense_kernels import (  # noqa: E402
    I4_GROUP,
    dequant_reference,
    k3_dense_matmul,
    quantize_i4g,
    quantize_i8,
)

DEV = "cuda"
FAIL = []


def check(name, got, want, tol=2e-2):
    g, w = got.float(), want.float()
    rel = (g - w).norm().item() / (w.norm().item() or 1.0)
    ok = rel < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} ||err||/||ref||={rel:.3e}")
    if not ok:
        FAIL.append(name)
    return ok


def eq(name, cond, note=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name:46s} {note}")
    if not cond:
        FAIL.append(name)


def test_i4_packing():
    print("[1] int4-g64 packing matches kimi_k3.c:1316-1328")
    # One group of 64 with a known max, so the scale is exact.
    w = torch.zeros(1, 64, device=DEV)
    w[0, 0] = 7.0        # max -> s = 7/7 = 1.0
    w[0, 1] = -7.0
    w[0, 2] = 3.4        # round(3.4) = 3
    w[0, 3] = -0.4       # round(-0.4) = 0
    q, s = quantize_i4g(w)
    eq("scale = max|w| / 7", abs(s[0, 0].item() - 1.0) < 1e-6,
       f"s={s[0,0].item():.4f}")
    eq("packed shape [O, I/2]", tuple(q.shape) == (1, 32), str(tuple(q.shape)))
    # byte 0 holds elements 0 (low nibble) and 1 (high), offset by 8
    b0 = int(q[0, 0].item())
    eq("byte0 = (v0+8) | ((v1+8)<<4)", (b0 & 0xF) == 7 + 8 and (b0 >> 4) == -7 + 8,
       f"lo={b0 & 0xF} hi={b0 >> 4}")
    d = dequant_reference(q, s, 4)
    eq("dequant recovers 7, -7, 3, 0",
       [d[0, i].item() for i in range(4)] == [7.0, -7.0, 3.0, 0.0],
       str([d[0, i].item() for i in range(4)]))

    # The clamp is asymmetric on purpose: dividing by 7 means -8 is only
    # reachable by rounding, never by the max element.
    w2 = torch.full((1, 64), -1.0, device=DEV)
    w2[0, 0] = 1.0
    q2, s2 = quantize_i4g(w2)
    d2 = dequant_reference(q2, s2, 4)
    eq("no overflow past -8", d2.min().item() >= -8.0, f"min={d2.min().item()}")


def test_i8_packing():
    print("[2] int8-per-row packing matches kimi_k3.c:1306-1314")
    w = torch.randn(8, 256, device=DEV)
    q, s = quantize_i8(w)
    eq("scale = max|row| / 127",
       torch.allclose(s, w.abs().amax(1) / 127.0, rtol=1e-5),
       "")
    eq("one scale per row", tuple(s.shape) == (8,), str(tuple(s.shape)))
    check("int8 round trip", dequant_reference(q, s, 8), w, tol=1e-2)


def test_gemv_and_gemm():
    print("[3] kernels vs dequantize-and-matmul")
    torch.manual_seed(0)
    N, K = 512, 1024
    w = torch.randn(N, K, device=DEV) * 0.05
    for bits, quant in ((4, quantize_i4g), (8, quantize_i8)):
        q, s = quant(w) if bits == 8 else quant(w, I4_GROUP)
        W = dequant_reference(q, s, bits)
        for M in (1, 4, 64):
            x = torch.randn(M, K, dtype=torch.bfloat16, device=DEV)
            ref = x.float() @ W.T
            got = k3_dense_matmul(x, q, s, bits)
            path = "gemv" if M <= 8 else "gemm"
            check(f"int{bits} M={M:<3d} ({path})", got, ref, tol=3e-2)

        # bias, and a non-2D input, since vLLM passes [tokens, hidden]
        x = torch.randn(2, 3, K, dtype=torch.bfloat16, device=DEV)
        b = torch.randn(N, dtype=torch.bfloat16, device=DEV)
        ref = x.float().reshape(-1, K) @ W.T + b.float()
        got = k3_dense_matmul(x, q, s, bits, bias=b)
        eq(f"int{bits} keeps leading dims", tuple(got.shape) == (2, 3, N),
           str(tuple(got.shape)))
        check(f"int{bits} with bias", got.reshape(-1, N), ref, tol=3e-2)


def test_bits_policy():
    print("[4] bit-width policy mirrors the engine's split")
    from vllm_k3_w1.dense_method import bits_for_prefix

    cases = [
        ("model.layers.3.self_attn.q_a_proj", 8, "MLA low-rank"),
        ("model.layers.3.self_attn.kv_b_proj", 8, "MLA"),
        ("model.layers.3.self_attn.fused_qkv_a_proj", 8, "MLA fused"),
        ("model.layers.7.self_attn.q_proj", 4, "KDA"),
        ("model.layers.7.self_attn.o_proj", 4, "KDA out"),
        ("model.layers.7.mlp.shared_experts.gate_up_proj", 4, "shared"),
        ("model.layers.7.mlp.routed_expert_down_proj", 4, "latent"),
        ("lm_head", 8, "head"),
    ]
    for prefix, want, note in cases:
        got = bits_for_prefix(prefix, 4, 8, 8)
        eq(f"{note:12s} -> int{want}", got == want, prefix)


def test_memory_saving():
    print("[5] resident bytes")
    N, K = 4096, 7168
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
    bf16 = w.numel() * 2
    q4, s4 = quantize_i4g(w)
    q8, s8 = quantize_i8(w)
    b4 = q4.numel() + s4.numel() * 4
    b8 = q8.numel() + s8.numel() * 4
    print(f"        bf16     {bf16 / 2**20:8.1f} MiB   16.00 bits/w")
    print(f"        int4-g64 {b4 / 2**20:8.1f} MiB   {b4 * 8 / w.numel():5.2f} bits/w "
          f"({bf16 / b4:.2f}x)")
    print(f"        int8-row {b8 / 2**20:8.1f} MiB   {b8 * 8 / w.numel():5.2f} bits/w "
          f"({bf16 / b8:.2f}x)")
    eq("int4-g64 is 4.5 bits/weight", abs(b4 * 8 / w.numel() - 4.5) < 0.01)


def test_speed():
    print("[6] decode-shape throughput")
    N, K = 4096, 7168
    w = torch.randn(N, K, device=DEV) * 0.05
    x = torch.randn(1, K, dtype=torch.bfloat16, device=DEV)
    for bits, quant in ((4, quantize_i4g), (8, quantize_i8)):
        q, s = quant(w) if bits == 8 else quant(w, I4_GROUP)
        wb = q.numel() + s.numel() * 4
        k3_dense_matmul(x, q, s, bits)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            k3_dense_matmul(x, q, s, bits)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 50 * 1e3
        print(f"        int{bits} [{N}x{K}] M=1: {ms:.4f} ms  "
              f"{wb / (ms * 1e-3) / 1e9:.1f} GB/s")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability()}\n")
    test_i4_packing()
    test_i8_packing()
    test_gemv_and_gemm()
    test_bits_policy()
    test_memory_saving()
    test_speed()
    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
