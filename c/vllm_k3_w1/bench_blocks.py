#!/usr/bin/env python3
"""Sweep block shapes for the k3-w1 grouped GEMM at K3's real dimensions.

The first working configuration ran at 29 GB/s of expert bytes against ~235
GB/s of achievable DRAM on a GB10, so the kernel is ~8x off the bound that
actually matters -- at B=1 this path is pure weight streaming, nothing else.
This finds where the block shape is costing that.

BLOCK_M must equal the block_size passed to moe_align_block_size (the kernel
indexes expert_ids at BLOCK_M granularity) and must be >= 16 for tl.dot.
"""

import itertools
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)

from vllm_k3_w1.kernels import w1_grouped_gemm  # noqa: E402

H, I, TOPK, AMP = 3584, 3072, 16, 1.69
DEV = "cuda"


def bench(packed, scale, x, topk_ids, topk_w, top_k, bm, bn, bk, iters=20):
    E = packed.shape[0]
    N = packed.shape[1]
    T = x.shape[0]
    try:
        sids, eids, npad = moe_align_block_size(topk_ids, bm, E, None)
        out = torch.zeros(T * top_k, N, dtype=x.dtype, device=DEV)

        def run():
            w1_grouped_gemm(x, packed, scale, sids, eids, npad, topk_w, out,
                            top_k, AMP, mul_routed_weight=False,
                            block_m=bm, block_n=bn, block_k=bk)

        run()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1e3
        wbytes = packed.numel() + scale.numel()
        return ms, wbytes / (ms * 1e-3) / 1e9
    except Exception as e:                       # bad shape for this Triton
        return None, str(e).split("\n")[0][:60]


def main():
    E = TOPK
    g = torch.Generator(device=DEV).manual_seed(0)
    # gate_up shape: the bigger of the two GEMMs, and the one our CUDA kernel
    # was tuned against (225 GB/s there), so it is the fair comparison.
    packed = torch.randint(0, 256, (E, 2 * I, H // 8), dtype=torch.uint8,
                           device=DEV, generator=g)
    scale = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=DEV)
    wbytes = packed.numel() + scale.numel()
    print(f"gate_up: E={E} N={2 * I} K={H}  {wbytes / 2**20:.1f} MiB of weights")
    print(f"{'T':>3} {'BM':>4} {'BN':>5} {'BK':>5} {'ms':>9} {'GB/s':>8}")

    best = {}
    for T in (1, 8):
        x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV) * 0.05
        topk_ids = torch.arange(TOPK, device=DEV, dtype=torch.int32).repeat(T, 1) % E
        topk_w = torch.full((T, TOPK), 1.0 / TOPK, dtype=torch.float32, device=DEV)
        for bm, bn, bk in itertools.product((16, 32, 64), (32, 64, 128, 256),
                                            (128, 256, 512)):
            ms, gb = bench(packed, scale, x, topk_ids, topk_w, TOPK, bm, bn, bk)
            if ms is None:
                continue
            print(f"{T:>3} {bm:>4} {bn:>5} {bk:>5} {ms:>9.3f} {gb:>8.1f}")
            if T not in best or gb > best[T][0]:
                best[T] = (gb, bm, bn, bk, ms)

    print()
    for T, (gb, bm, bn, bk, ms) in sorted(best.items()):
        print(f"BEST T={T}: BM={bm} BN={bn} BK={bk} -> {ms:.3f} ms, {gb:.1f} GB/s")


if __name__ == "__main__":
    main()
