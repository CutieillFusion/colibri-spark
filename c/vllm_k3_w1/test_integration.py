#!/usr/bin/env python3
"""Registration + real-shape checks for the k3_w1 method.

Separate from test_numerics.py because this one needs vLLM importable, not
just torch+triton. Checks that the method registers through vLLM's public
plugin API, that it dispatches to the right thing per layer type, and that
the kernels accept K3's actual dimensions (latent 3584, inter 3072) rather
than only the small synthetic ones.
"""

import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

H, INTER, TOPK = 3584, 3072, 16
AMP = 1.69
FAIL = []


def ok(name, cond, note=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name:46s} {note}")
    if not cond:
        FAIL.append(name)


def test_registration():
    print("[1] plugin registration")
    import vllm_k3_w1  # noqa: F401  -- import registers the method
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        get_quantization_config,
    )

    ok("k3_w1 in QUANTIZATION_METHODS", "k3_w1" in QUANTIZATION_METHODS)
    cls = get_quantization_config("k3_w1")
    ok(
        "get_quantization_config resolves",
        cls.__name__ == "KimiK3OneBitConfig",
        cls.__name__,
    )

    cfg = cls.from_config({})
    ok("default amplitude 1.69", abs(cfg.amplitude - 1.69) < 1e-9, f"a={cfg.amplitude}")
    cfg2 = cls.from_config({"amplitude": 2.0})
    ok("amplitude is configurable", abs(cfg2.amplitude - 2.0) < 1e-9)

    try:
        cls.from_config({"group_size": 64})
        ok("rejects non-32 group size", False)
    except ValueError:
        ok("rejects non-32 group size", True)

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    lin = torch.nn.Linear(8, 8)
    from vllm.model_executor.layers.linear import LinearBase

    ok(
        "non-Linear non-MoE -> None",
        cfg.get_quant_method(lin, "x") is None
        or isinstance(cfg.get_quant_method(lin, "x"), UnquantizedLinearMethod),
        "(plain nn.Linear is not LinearBase)",
    )
    ok("LinearBase is a real class", isinstance(LinearBase, type))


def test_stream_partitions():
    print("[2] streamed expert passes")
    from vllm_k3_w1.method import _local_stream_misses, _stream_missing

    layer = SimpleNamespace(
        k3_resident=2,
        k3_num_experts=6,
        k3_stream_slots=2,
        expert_map=torch.tensor([0, 1, 2, 3, 4, 5, -1]),
    )
    topk_ids = torch.tensor([[0, 2, 6], [1, 3, 6], [0, 4, 5]], dtype=torch.int32)
    misses = _local_stream_misses(layer, topk_ids)
    ok("prefill finds every distinct tail expert", misses.tolist() == [2, 3, 4, 5])

    try:
        _stream_missing(layer, torch.tensor([[2, 3, 4]], dtype=torch.int32))
        ok("single-pass scratch overflow fails closed", False)
    except RuntimeError:
        ok("single-pass scratch overflow fails closed", True)

    class FakeStore:
        def read_slot(self, ordinal, slot):
            del ordinal
            return {
                "w1p": torch.full((2, 2), slot, dtype=torch.uint8).numpy(),
                "w3p": torch.full((2, 2), slot, dtype=torch.uint8).numpy(),
                "w1s": torch.full((2, 2), 119 + slot, dtype=torch.uint8).numpy(),
                "w3s": torch.full((2, 2), 119 + slot, dtype=torch.uint8).numpy(),
                "w2p": torch.full((3, 2), slot, dtype=torch.uint8).numpy(),
                "w2s": torch.full((3, 2), 119 + slot, dtype=torch.uint8).numpy(),
            }

    stream_layer = SimpleNamespace(
        k3_resident=2,
        k3_num_experts=6,
        k3_stream_slots=2,
        expert_map=torch.tensor([0, 1, 4, 5, -1, -1]),
        k3_store=FakeStore(),
        k3_moe_ordinal=0,
        w13_qweight=torch.zeros((4, 4, 2), dtype=torch.uint8),
        w13_scales=torch.zeros((4, 4, 1), dtype=torch.uint8),
        w2_qweight=torch.zeros((4, 3, 2), dtype=torch.uint8),
        w2_scales=torch.zeros((4, 3, 1), dtype=torch.uint8),
    )
    remapped = _stream_missing(stream_layer, torch.tensor([[2, 3]], dtype=torch.int32))
    copied = (
        torch.all(stream_layer.w13_qweight[2] == 4)
        and torch.all(stream_layer.w13_qweight[3] == 5)
        and remapped.tolist() == [0, 1, 2, 3, -1, -1]
    )
    ok("all missing experts are loaded and remapped", bool(copied))


def test_real_shapes():
    print(f"[3] K3 shapes  H={H} I={INTER} top_k={TOPK}")
    from vllm_k3_w1.kernels import situ_and_mul, w1_gemv

    # Two residency cases that bracket the deployment:
    #   E=16 -- every routed expert local (single node, or TP-only)
    #   E=4  -- 896 experts over 4 EP ranks, so ~16/4 land on each rank
    for E in (16, 4):
        _shapes_for(E, situ_and_mul, w1_gemv)


def _shapes_for(E, situ_and_mul, w1_gemv):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    w13p = torch.randint(
        0, 256, (E, 2 * INTER, H // 8), dtype=torch.uint8, device=dev, generator=g
    )
    packed_121 = (121 - 109) | ((121 - 109) << 4)
    w13s = torch.full(
        (E, 2 * INTER, H // 64), packed_121, dtype=torch.uint8, device=dev
    )
    w2p = torch.randint(
        0, 256, (E, H, INTER // 8), dtype=torch.uint8, device=dev, generator=g
    )
    w2s = torch.full(
        (E, H, INTER // 64), packed_121, dtype=torch.uint8, device=dev
    )
    weight_bytes = w13p.numel() + w2p.numel() + w13s.numel() + w2s.numel()
    print(
        f"  weights: {weight_bytes / 2**20:.1f} MiB for {E} experts "
        f"({weight_bytes / E / 2**20:.2f} MiB/expert)"
    )

    for tokens in (1, 8):
        # topk_ids spans the local experts; with E<TOPK the same expert is
        # selected more than once, which is what EP sharding looks like from
        # one rank's side.
        x = torch.randn(tokens, H, dtype=torch.bfloat16, device=dev) * 0.05
        topk_ids = (
            torch.arange(TOPK, device=dev, dtype=torch.int32).repeat(tokens, 1) % E
        )
        topk_w = torch.full((tokens, TOPK), 1.0 / TOPK, dtype=torch.float32, device=dev)

        def run(tokens=tokens, x=x, topk_ids=topk_ids, topk_w=topk_w):
            inter = torch.zeros(tokens * TOPK, 2 * INTER, dtype=x.dtype, device=dev)
            w1_gemv(
                x,
                w13p,
                w13s,
                topk_ids,
                topk_w,
                None,
                inter,
                TOPK,
                AMP,
                mul_routed_weight=False,
            )
            h = situ_and_mul(inter, 4.0, 25.0)
            down = torch.zeros(tokens * TOPK, H, dtype=x.dtype, device=dev)
            w1_gemv(
                h,
                w2p,
                w2s,
                topk_ids,
                topk_w,
                None,
                down,
                1,
                AMP,
                mul_routed_weight=True,
            )
            return down.view(tokens, TOPK, H).sum(1)

        out = run()
        finite = torch.isfinite(out).all().item()
        ok(
            f"E={E} T={tokens} forward runs, finite output",
            finite,
            f"|out|max={out.abs().max().item():.3f}",
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 100
        # Bytes actually touched: one slot per routed selection, not per
        # resident expert, since a program only reads the expert it routed to.
        touched = TOPK * (
            w13p[0].numel() + w2p[0].numel() + w13s[0].numel() + w2s[0].numel()
        )
        print(
            f"        T={tokens}: {ms:.3f} ms/layer  "
            f"({touched / (ms * 1e-3) / 1e9:.1f} GB/s), "
            f"92 layers = {ms * 92:.1f} ms -> {1000 / (ms * 92):.2f} tok/s "
            f"(MoE only)"
        )


def test_rank_order():
    print("[4] deterministic Ray rank order")
    from vllm_k3_w1.patch_rank_order import _order_by_ip

    items = [(3, "node-c", "10.0.0.3"), (1, "node-a", "10.0.0.1")]
    ordered = _order_by_ip(items, ["10.0.0.1", "10.0.0.3"])
    ok("rank bundles follow configured IP order", ordered == [items[1], items[0]])
    try:
        _order_by_ip(items, ["10.0.0.1", "10.0.0.2"])
    except RuntimeError:
        ok("rank placement mismatch fails closed", True)
    else:
        ok("rank placement mismatch fails closed", False)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    print(
        f"device: {torch.cuda.get_device_name(0)} "
        f"cap={torch.cuda.get_device_capability()}\n"
    )
    test_registration()
    test_stream_partitions()
    test_real_shapes()
    test_rank_order()
    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
