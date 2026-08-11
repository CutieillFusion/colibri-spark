#!/usr/bin/env python3
"""Stable real-shape prefill or decode workload for Nsight Compute."""

import argparse
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: E402
    moe_align_block_size,
)
from vllm_k3_w1.kernels import w1_gemv, w1_grouped_gemm  # noqa: E402

HIDDEN = 3584
INTERMEDIATE = 3072
TOPK = 16
GLOBAL_EXPERTS = 896
LOCAL_EXPERTS = 448
AMPLITUDE = 1.69


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prefill", "decode"))
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    tokens = 512 if args.mode == "prefill" else 1
    generator = torch.Generator(device="cuda").manual_seed(0)
    packed = torch.randint(
        0,
        256,
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 8),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    scales = torch.randint(
        0,
        256,
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 64),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    expert_map = torch.full((GLOBAL_EXPERTS,), -1, dtype=torch.int32, device="cuda")
    owned = torch.arange(0, GLOBAL_EXPERTS, 2, device="cuda")
    expert_map[owned] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device="cuda")
    topk_ids = torch.stack(
        [
            torch.randperm(GLOBAL_EXPERTS, device="cuda", generator=generator)[:TOPK]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    topk_weights = torch.full(
        (tokens, TOPK), 1 / TOPK, dtype=torch.float32, device="cuda"
    )
    inputs = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    output = torch.zeros(
        tokens * TOPK,
        2 * INTERMEDIATE,
        dtype=torch.bfloat16,
        device="cuda",
    )

    if args.mode == "prefill":
        sorted_ids, expert_ids, padded = moe_align_block_size(
            topk_ids, 16, GLOBAL_EXPERTS, expert_map
        )

        def run():
            w1_grouped_gemm(
                inputs,
                packed,
                scales,
                sorted_ids,
                expert_ids,
                padded,
                topk_weights,
                output,
                TOPK,
                AMPLITUDE,
                False,
                block_m=16,
            )

    else:

        def run():
            w1_gemv(
                inputs,
                packed,
                scales,
                topk_ids,
                topk_weights,
                expert_map,
                output,
                TOPK,
                AMPLITUDE,
                False,
            )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
