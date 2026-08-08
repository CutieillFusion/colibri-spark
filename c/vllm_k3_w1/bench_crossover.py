#!/usr/bin/env python3
"""Where does the grouped GEMM overtake the GEMV path?

The two scale differently in weight traffic, which is all that matters here:

  gemv     T * top_k slot reads  -- one per (token, slot) pair, no reuse
  grouped  ~distinct-experts-touched slot reads, but pads each expert's token
           list to BLOCK_M and feeds tl.dot, so at T=1 it does 16 rows of
           arithmetic for 1 useful row

So gemv should win only while T*top_k is close to the number of distinct
experts -- i.e. essentially T=1 -- and lose linearly after that. This measures
the crossover rather than assuming it, because the threshold it sets is what
the method dispatches on.
"""

import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)

from vllm_k3_w1.kernels import situ_and_mul, w1_gemv, w1_grouped_gemm  # noqa: E402

H, I, TOPK, AMP = 3584, 3072, 16, 1.69
DEV = "cuda"


def timeit(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


GLOBAL_E, RANKS = 896, 4


def main():
    # One rank's view: it owns GLOBAL_E/RANKS experts, and of the top_k
    # selected per token, only the ~top_k/RANKS that map locally do any work.
    # Modelling this as "E=4 resident, selected 4x each" would be wrong in the
    # direction that flatters the grouped path -- it would let one slot read
    # serve four selections. The selections are distinct experts.
    E = GLOBAL_E // RANKS
    g = torch.Generator(device=DEV).manual_seed(0)
    w13p = torch.randint(0, 256, (E, 2 * I, H // 8), dtype=torch.uint8,
                         device=DEV, generator=g)
    w13s = torch.randint(124, 131, (E, 2 * I, H // 32), dtype=torch.uint8,
                         device=DEV, generator=g)
    w2p = torch.randint(0, 256, (E, H, I // 8), dtype=torch.uint8,
                        device=DEV, generator=g)
    w2s = torch.randint(124, 131, (E, H, I // 32), dtype=torch.uint8,
                        device=DEV, generator=g)

    # expert_map: global id -> local slot, or -1 if another rank owns it.
    # Rank 0 owns e % RANKS == 0, matching the engine's e%world split.
    expert_map = torch.full((GLOBAL_E,), -1, dtype=torch.int32, device=DEV)
    local = torch.arange(0, GLOBAL_E, RANKS, device=DEV)
    expert_map[local] = torch.arange(E, dtype=torch.int32, device=DEV)

    resident = (w13p.numel() + w2p.numel() + w13s.numel() + w2s.numel())
    print(f"one EP rank of {RANKS}: {E} resident experts "
          f"({resident / 2**30:.2f} GiB), H={H} I={I} top_k={TOPK} of {GLOBAL_E}")
    print(f"{'T':>5} {'local sel':>10} {'gemv ms':>10} {'grouped ms':>12} "
          f"{'winner':>9} {'92-layer tok/s':>16}")

    for T in (1, 2, 4, 8, 16, 64):
        x = torch.randn(T, H, dtype=torch.bfloat16, device=DEV) * 0.05
        # top_k distinct global experts per token, uniformly drawn.
        topk_ids = torch.stack([
            torch.randperm(GLOBAL_E, device=DEV, generator=g)[:TOPK]
            for _ in range(T)
        ]).to(torch.int32)
        n_local = int((expert_map[topk_ids.long()] >= 0).sum().item())
        topk_w = torch.full((T, TOPK), 1.0 / TOPK, dtype=torch.float32,
                            device=DEV)
        sids, eids, npad = moe_align_block_size(topk_ids, 16, GLOBAL_E,
                                                expert_map)

        def layer(mode):
            inter = torch.zeros(T * TOPK, 2 * I, dtype=x.dtype, device=DEV)
            down = torch.zeros(T * TOPK, H, dtype=x.dtype, device=DEV)
            if mode == "gemv":
                w1_gemv(x, w13p, w13s, topk_ids, topk_w, expert_map, inter,
                        TOPK, AMP, False)
                h = situ_and_mul(inter, 4.0, 25.0)
                w1_gemv(h, w2p, w2s, topk_ids, topk_w, expert_map, down, 1,
                        AMP, True)
            else:
                w1_grouped_gemm(x, w13p, w13s, sids, eids, npad, topk_w, inter,
                                TOPK, AMP, False, block_m=16)
                h = situ_and_mul(inter, 4.0, 25.0)
                w1_grouped_gemm(h, w2p, w2s, sids, eids, npad, topk_w, down, 1,
                                AMP, True, block_m=16)
            return down.view(T, TOPK, H).sum(1)

        # Agreement check before timing, so a fast-but-wrong config cannot win.
        a, b = layer("gemv").float(), layer("grouped").float()
        rel = (a - b).norm().item() / (b.norm().item() or 1.0)
        assert rel < 1e-2, f"T={T}: paths disagree, rel={rel:.3e}"

        t_gemv = timeit(lambda: layer("gemv"))
        t_grp = timeit(lambda: layer("grouped"))
        best = min(t_gemv, t_grp)
        win = "gemv" if t_gemv < t_grp else "grouped"
        print(f"{T:>5} {n_local:>10} {t_gemv:>10.3f} {t_grp:>12.3f} {win:>9} "
              f"{1000 / (best * 92) * T:>16.2f}")


if __name__ == "__main__":
    main()
