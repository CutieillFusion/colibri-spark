#!/usr/bin/env python3
"""GEMV path vs grouped-GEMM path at K3's real gate_up shape.

Correctness first (against the grouped path, which test_numerics already
checked against an explicit dequant), then bandwidth. Reference points:
c/backend_cuda_k3.cu reaches 225 GB/s on this exact shape, and achievable
DRAM on a GB10 is ~235 GB/s.
"""

import itertools
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)

from vllm_k3_w1.kernels import w1_gemv, w1_grouped_gemm  # noqa: E402

H, I, TOPK, AMP = 3584, 3072, 16, 1.69
DEV = "cuda"


def main():
    E = TOPK
    g = torch.Generator(device=DEV).manual_seed(0)
    packed = torch.randint(0, 256, (E, 2 * I, H // 8), dtype=torch.uint8,
                           device=DEV, generator=g)
    scale = torch.randint(124, 131, (E, 2 * I, H // 32), dtype=torch.uint8,
                          device=DEV, generator=g)
    wbytes = packed.numel() + scale.numel()
    N = 2 * I
    print(f"gate_up: E={E} N={N} K={H}  {wbytes / 2**20:.1f} MiB "
          f"({wbytes / E / 2**20:.2f} MiB/expert)")

    T = 1
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV) * 0.05
    topk_ids = torch.arange(TOPK, device=DEV, dtype=torch.int32).repeat(T, 1) % E
    topk_w = torch.rand(T, TOPK, dtype=torch.float32, device=DEV)

    # Reference: the grouped path at its best-known config.
    sids, eids, npad = moe_align_block_size(topk_ids, 16, E, None)
    ref = torch.zeros(T * TOPK, N, dtype=x.dtype, device=DEV)
    w1_grouped_gemm(x, packed, scale, sids, eids, npad, topk_w, ref, TOPK,
                    AMP, mul_routed_weight=True, block_m=16, block_n=32,
                    block_k=128)
    torch.cuda.synchronize()

    got = torch.zeros_like(ref)
    w1_gemv(x, packed, scale, topk_ids, topk_w, None, got, TOPK, AMP,
            mul_routed_weight=True)
    torch.cuda.synchronize()
    rel = (got.float() - ref.float()).norm().item() / ref.float().norm().item()
    print(f"\ngemv vs grouped: ||err||/||ref|| = {rel:.3e} "
          f"{'OK' if rel < 1e-2 else 'MISMATCH'}")
    if rel >= 1e-2:
        sys.exit("gemv path disagrees with the validated grouped path")

    def timeit(fn, iters=30):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3

    ms = timeit(lambda: w1_grouped_gemm(x, packed, scale, sids, eids, npad,
                                        topk_w, ref, TOPK, AMP, True, 16, 32, 128))
    print(f"\ngrouped BM=16 BN=32 BK=128: {ms:.3f} ms  "
          f"{wbytes / (ms * 1e-3) / 1e9:.1f} GB/s")

    print(f"\n{'BN':>5} {'BK':>5} {'warps':>6} {'ms':>9} {'GB/s':>8}")
    best = None
    for bn, bk, w in itertools.product((32, 64, 128, 256), (128, 256, 512),
                                       (2, 4, 8)):
        try:
            out = torch.zeros_like(ref)

            def run(bn=bn, bk=bk, w=w, out=out):
                w1_gemv(x, packed, scale, topk_ids, topk_w, None, out, TOPK,
                        AMP, True, block_n=bn, block_k=bk, num_warps=w)

            ms = timeit(run)
        except Exception as e:
            print(f"{bn:>5} {bk:>5} {w:>6}   {str(e).split(chr(10))[0][:44]}")
            continue
        gb = wbytes / (ms * 1e-3) / 1e9
        print(f"{bn:>5} {bk:>5} {w:>6} {ms:>9.3f} {gb:>8.1f}")
        if best is None or gb > best[0]:
            best = (gb, bn, bk, w, ms)

    if best:
        gb, bn, bk, w, ms = best
        print(f"\nBEST gemv: BN={bn} BK={bk} warps={w} -> {ms:.3f} ms, {gb:.1f} GB/s")
        print(f"  92 layers: {ms * 92:.1f} ms  ->  {1000 / (ms * 92):.2f} tok/s "
              f"(MoE gate_up only, all {E} experts local)")


if __name__ == "__main__":
    main()
