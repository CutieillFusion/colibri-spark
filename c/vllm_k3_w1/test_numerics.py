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

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from vllm_k3_w1.kernels import (  # noqa: E402
    GROUP,
    dequant_reference,
    situ_and_mul,
    w1_gemv,
    w1_grouped_gemm,
)
from vllm_k3_w1.loader import SCALE_NIBBLE_BASE  # noqa: E402

DEV = "cuda"
AMP = 1.69
FAIL = []


def check(name, got, want, rtol=2e-2, atol=2e-2):
    ok = torch.allclose(got.float(), want.float(), rtol=rtol, atol=atol)
    err = (got.float() - want.float()).abs().max().item()
    denom = want.float().abs().max().item() or 1.0
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name:44s} max|err|={err:.3e} "
        f"rel={err / denom:.3e}"
    )
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
    worst = (
        ((g - w).abs() / w.abs().clamp_min(1e-9))[big].max().item()
        if big.any()
        else 0.0
    )
    ok = rel < tol and worst < 5e-2
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name:44s} ||err||/||ref||={rel:.3e} "
        f"worst-elem-rel={worst:.3e}"
    )
    if not ok:
        FAIL.append(name)
    return ok


def rand_store(E, N, K, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    packed = torch.randint(
        0, 256, (E, N, K // 8), dtype=torch.uint8, device=DEV, generator=g
    )
    # Keep exponents near the bias so the reference stays in fp32 range.
    # More than 99.99% of the real stores are 119..122. Keep the randomized
    # matmul test representative; boundary values are covered separately by
    # test_scale_packing.
    exponent = torch.randint(
        119,
        123,
        (E, N, K // GROUP),
        dtype=torch.uint8,
        device=DEV,
        generator=g,
    )
    delta = exponent - SCALE_NIBBLE_BASE
    scale = delta[..., 0::2] | (delta[..., 1::2] << 4)
    return packed, scale


def packed_scale(*exponents):
    values = list(exponents)
    if len(values) % 2:
        values.append(SCALE_NIBBLE_BASE)
    delta = [value - SCALE_NIBBLE_BASE for value in values]
    packed = [delta[i] | (delta[i + 1] << 4) for i in range(0, len(delta), 2)]
    return torch.tensor([[packed]], dtype=torch.uint8, device=DEV)


def test_unpack():
    print("[1] bit/scale unpack")
    # One expert, one row, one group: bit j of byte b is weight 8b+j, LSB first.
    packed = torch.tensor(
        [[[0b00000001, 0b10000000, 0, 255]]], dtype=torch.uint8, device=DEV
    )
    scale = packed_scale(121)
    W = dequant_reference(packed, scale, 1.0)[0, 0]
    want = torch.full((32,), -(2.0**-6), device=DEV)
    want[0] = 2.0**-6  # byte0 bit0
    want[15] = 2.0**-6  # byte1 bit7
    want[24:] = 2.0**-6  # byte3 all ones
    check("LSB-first bit order, +/-1", W, want)
    # UE8M0: exponent e means 2^(e-127).
    s2 = packed_scale(124)
    W2 = dequant_reference(packed, s2, 1.0)[0, 0]
    check("UE8M0 scale = 2^(e-127)", W2, want * 8.0)
    check("amplitude folds in", dequant_reference(packed, scale, AMP)[0, 0], want * AMP)


def test_scale_packing():
    print("[1b] lossless scale packing")
    import numpy as np

    from vllm_k3_w1.loader import pack_scale_nibbles

    raw = np.array([[109, 124, 120, 121]], dtype=np.uint8)
    got = pack_scale_nibbles(raw)
    want = np.array([[0xF0, 0xCB]], dtype=np.uint8)
    ok = np.array_equal(got, want)
    print(f"  {'PASS' if ok else 'FAIL'}  two exponents per byte, low nibble first")
    if not ok:
        FAIL.append("scale nibble order")
    try:
        pack_scale_nibbles(np.array([[108, 120]], dtype=np.uint8))
    except ValueError:
        print("  PASS  out-of-range exponent fails closed")
    else:
        print("  FAIL  out-of-range exponent fails closed")
        FAIL.append("scale range validation")


def _align(topk_ids, block_m, num_experts, expert_map=None):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(topk_ids, block_m, num_experts, expert_map)


def test_gemm(E=4, N=128, K=256, T=8, top_k=2):
    print(f"[2] grouped GEMM  E={E} N={N} K={K} T={T} top_k={top_k}")
    torch.manual_seed(0)
    packed, scale = rand_store(E, N, K)
    x = torch.randn(T, K, dtype=torch.bfloat16, device=DEV)
    topk_ids = torch.randint(0, E, (T, top_k), dtype=torch.int32, device=DEV)
    topk_w = torch.rand(T, top_k, dtype=torch.float32, device=DEV)

    sorted_ids, expert_ids, npad = _align(topk_ids, 64, E)
    out = torch.zeros(T * top_k, N, dtype=torch.bfloat16, device=DEV)
    w1_grouped_gemm(
        x,
        packed,
        scale,
        sorted_ids,
        expert_ids,
        npad,
        topk_w,
        out,
        top_k,
        AMP,
        mul_routed_weight=False,
        block_m=64,
    )

    W = dequant_reference(packed, scale, AMP)
    ref = torch.zeros_like(out, dtype=torch.float32)
    for t in range(T):
        for j in range(top_k):
            ref[t * top_k + j] = x[t].float() @ W[topk_ids[t, j]].T
    check("grouped GEMM vs dequant-and-matmul", out, ref)

    # ...and with the routing weight applied on the output.
    out2 = torch.zeros_like(out)
    w1_grouped_gemm(
        x,
        packed,
        scale,
        sorted_ids,
        expert_ids,
        npad,
        topk_w,
        out2,
        top_k,
        AMP,
        mul_routed_weight=True,
        block_m=64,
    )
    check("routed weight applied", out2, ref * topk_w.reshape(-1, 1))


def test_situ():
    print("[3] SITU activation")
    torch.manual_seed(0)
    x = torch.randn(64, 256, dtype=torch.bfloat16, device=DEV) * 3
    beta, lin = 4.0, 25.0
    d = x.shape[-1] // 2
    g, u = x[..., :d].float(), x[..., d:].float()
    ref = (beta * torch.tanh(g / beta) * torch.sigmoid(g)) * (lin * torch.tanh(u / lin))
    check("situ vs SituAndMul.forward_native", situ_and_mul(x, beta, lin), ref)


def test_moe_forward(E=8, H=256, intermediate=128, T=4, top_k=3):
    print(f"[4] full MoE forward  E={E} H={H} I={intermediate} T={T} top_k={top_k}")
    torch.manual_seed(0)
    beta, lin = 4.0, 25.0
    w13p, w13s = rand_store(E, 2 * intermediate, H, seed=1)
    w2p, w2s = rand_store(E, H, intermediate, seed=2)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV)
    topk_ids = torch.randint(0, E, (T, top_k), dtype=torch.int32, device=DEV)
    topk_w = torch.rand(T, top_k, dtype=torch.float32, device=DEV)

    sorted_ids, expert_ids, npad = _align(topk_ids, 16, E)

    def grouped(a, p, s, out, tk, mul):
        w1_grouped_gemm(
            a, p, s, sorted_ids, expert_ids, npad, topk_w, out, tk, AMP, mul, block_m=16
        )

    def gemv(a, p, s, out, tk, mul):
        w1_gemv(a, p, s, topk_ids, topk_w, None, out, tk, AMP, mul)

    outs = {}
    for name, gemm in (("grouped", grouped), ("gemv", gemv)):
        inter = torch.zeros(
            T * top_k, 2 * intermediate, dtype=torch.bfloat16, device=DEV
        )
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
            gu = (x[t].float() @ W13[e].T).to(torch.bfloat16).float()
            g, u = gu[:intermediate], gu[intermediate:]
            act = (beta * torch.tanh(g / beta) * torch.sigmoid(g)) * (
                lin * torch.tanh(u / lin)
            )
            act = act.to(torch.bfloat16).float()
            routed = topk_w[t, j] * (act @ W2[e].T)
            ref[t] += routed.to(torch.bfloat16).float()
    for name, got in outs.items():
        check_norm(f"MoE forward ({name}) vs dense reference", got, ref)
    check_norm("gemv matches grouped", outs["gemv"], outs["grouped"], tol=5e-3)

    resident, scratch = E // 2, 2
    w13p_compact = torch.empty_like(w13p[: resident + scratch])
    w13s_compact = torch.empty_like(w13s[: resident + scratch])
    w2p_compact = torch.empty_like(w2p[: resident + scratch])
    w2s_compact = torch.empty_like(w2s[: resident + scratch])
    w13p_compact[:resident].copy_(w13p[:resident])
    w13s_compact[:resident].copy_(w13s[:resident])
    w2p_compact[:resident].copy_(w2p[:resident])
    w2s_compact[:resident].copy_(w2s[:resident])

    def partial_pass(experts, resident_pass=False, pairs_out=None):
        expert_map = torch.full((E,), -1, dtype=torch.int32, device=DEV)
        if resident_pass:
            expert_map[:resident] = torch.arange(
                resident, dtype=torch.int32, device=DEV
            )
        else:
            for offset, expert in enumerate(experts):
                slot = resident + offset
                w13p_compact[slot].copy_(w13p[expert])
                w13s_compact[slot].copy_(w13s[expert])
                w2p_compact[slot].copy_(w2p[expert])
                w2s_compact[slot].copy_(w2s[expert])
                expert_map[expert] = slot
        ids, experts_by_block, padded = _align(topk_ids, 16, E, expert_map)
        inter = torch.zeros(
            T * top_k, 2 * intermediate, dtype=torch.bfloat16, device=DEV
        )
        w1_grouped_gemm(
            x,
            w13p_compact,
            w13s_compact,
            ids,
            experts_by_block,
            padded,
            topk_w,
            inter,
            top_k,
            AMP,
            False,
            block_m=16,
        )
        activated = situ_and_mul(inter, beta, lin)
        down = (
            torch.zeros(T * top_k, H, dtype=torch.bfloat16, device=DEV)
            if pairs_out is None
            else pairs_out.view(T * top_k, H)
        )
        w1_grouped_gemm(
            activated,
            w2p_compact,
            w2s_compact,
            ids,
            experts_by_block,
            padded,
            topk_w,
            down,
            1,
            AMP,
            True,
            block_m=16,
        )
        return down.view(T, top_k, H)

    streamed_pairs = partial_pass([], resident_pass=True)
    tail = sorted(
        {int(expert) for expert in topk_ids.flatten().tolist() if expert >= resident}
    )
    for start in range(0, len(tail), scratch):
        streamed_pairs.add_(partial_pass(tail[start : start + scratch]))
    streamed = streamed_pairs.sum(1)
    check_norm(
        "expert-wise streamed passes match resident output",
        streamed,
        outs["grouped"],
        tol=5e-3,
    )

    direct_pairs = torch.zeros_like(streamed_pairs)
    partial_pass([], resident_pass=True, pairs_out=direct_pairs)
    for start in range(0, len(tail), scratch):
        partial_pass(tail[start : start + scratch], pairs_out=direct_pairs)
    check(
        "disjoint streamed passes share one pair buffer",
        direct_pairs,
        streamed_pairs,
        rtol=0,
        atol=0,
    )

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm_k3_w1.method import KimiK3OneBitMoEMethod

    method = object.__new__(KimiK3OneBitMoEMethod)
    method.moe = SimpleNamespace(
        w13_num_shards=2,
        activation_situ_beta=beta,
        activation_situ_linear_beta=lin,
    )
    layer = SimpleNamespace(
        activation=MoEActivation.SITU,
        expert_map=torch.arange(E, dtype=torch.int32, device=DEV),
        global_num_experts=E,
        k3_amplitude=AMP,
        k3_num_experts=E,
        k3_resident=E,
        k3_stream_slots=0,
        w13_qweight=w13p,
        w13_scales=w13s,
        w2_qweight=w2p,
        w2_scales=w2s,
    )
    previous_chunk = os.environ.get("K3_MOE_TOKEN_CHUNK")
    try:
        os.environ["K3_MOE_TOKEN_CHUNK"] = "0"
        whole = method._forward_routed(layer, x, topk_w, topk_ids)
        os.environ["K3_MOE_TOKEN_CHUNK"] = "2"
        chunked = method._forward_routed(layer, x, topk_w, topk_ids)
    finally:
        if previous_chunk is None:
            os.environ.pop("K3_MOE_TOKEN_CHUNK", None)
        else:
            os.environ["K3_MOE_TOKEN_CHUNK"] = previous_chunk
    check("full-resident token chunking is bit exact", chunked, whole, rtol=0, atol=0)


def test_store_roundtrip():
    print("[5] flat-store round trip")
    import os
    import tempfile

    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vllm_k3_w1.loader import K3W1Store

    H, intermediate, E, L = 512, 256, 2, 2
    st_bytes = (
        2 * (intermediate * (H // 8) + intermediate * (H // 32))
        + H * (intermediate // 8)
        + H * (intermediate // 32)
    )
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
        store = K3W1Store(d, H, intermediate, E)
        ok = store.slot == st_bytes and store.n_slots == L * E
        print(
            f"  {'PASS' if ok else 'FAIL'}  slot/geometry           "
            f"slot={store.slot} slots={store.n_slots}"
        )
        if not ok:
            FAIL.append("store geometry")
        s = store.read_slot(1, 1)  # last slot
        want = blobs[1 * E + 1]
        got = np.concatenate(
            [s[k].reshape(-1) for k in ("w1p", "w1s", "w2p", "w2s", "w3p", "w3s")]
        )
        ok = np.array_equal(got, want)
        print(f"  {'PASS' if ok else 'FAIL'}  slot layout w1p|w1s|w2p|w2s|w3p|w3s")
        if not ok:
            FAIL.append("slot layout")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    print(
        f"device: {torch.cuda.get_device_name(0)} "
        f"cap={torch.cuda.get_device_capability()}\n"
    )
    test_unpack()
    test_scale_packing()
    test_gemm()
    test_situ()
    test_moe_forward()
    test_store_roundtrip()
    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
