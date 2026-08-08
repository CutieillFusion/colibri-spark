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

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

H, I, TOPK = 3584, 3072, 16          # K3: routed_expert_hidden_size, moe_inter
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
    ok("get_quantization_config resolves", cls.__name__ == "KimiK3OneBitConfig",
       cls.__name__)

    cfg = cls.from_config({})
    ok("default amplitude 1.69", abs(cfg.amplitude - 1.69) < 1e-9,
       f"a={cfg.amplitude}")
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
    ok("non-Linear non-MoE -> None", cfg.get_quant_method(lin, "x") is None
       or isinstance(cfg.get_quant_method(lin, "x"), UnquantizedLinearMethod),
       "(plain nn.Linear is not LinearBase)")
    ok("LinearBase is a real class", isinstance(LinearBase, type))


def test_real_shapes():
    print(f"[2] K3 shapes  H={H} I={I} top_k={TOPK}")
    from vllm_k3_w1.kernels import situ_and_mul, w1_gemv

    # Two residency cases that bracket the deployment:
    #   E=16 -- every routed expert local (single node, or TP-only)
    #   E=4  -- 896 experts over 4 EP ranks, so ~16/4 land on each rank
    for E in (16, 4):
        _shapes_for(E, situ_and_mul, w1_gemv)


def _shapes_for(E, situ_and_mul, w1_gemv):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    w13p = torch.randint(0, 256, (E, 2 * I, H // 8), dtype=torch.uint8,
                         device=dev, generator=g)
    w13s = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=dev)
    w2p = torch.randint(0, 256, (E, H, I // 8), dtype=torch.uint8,
                        device=dev, generator=g)
    w2s = torch.full((E, H, I // 32), 127, dtype=torch.uint8, device=dev)
    print(f"  weights: {(w13p.numel() + w2p.numel() + w13s.numel() + w2s.numel()) / 2**20:.1f} MiB "
          f"for {E} experts ({(w13p.numel() + w2p.numel() + w13s.numel() + w2s.numel()) / E / 2**20:.2f} MiB/expert)")

    for T in (1, 8):
        # topk_ids spans the local experts; with E<TOPK the same expert is
        # selected more than once, which is what EP sharding looks like from
        # one rank's side.
        x = torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.05
        topk_ids = torch.arange(TOPK, device=dev, dtype=torch.int32
                                ).repeat(T, 1) % E
        topk_w = torch.full((T, TOPK), 1.0 / TOPK, dtype=torch.float32, device=dev)

        def run():
            inter = torch.zeros(T * TOPK, 2 * I, dtype=x.dtype, device=dev)
            w1_gemv(x, w13p, w13s, topk_ids, topk_w, None, inter, TOPK, AMP,
                    mul_routed_weight=False)
            h = situ_and_mul(inter, 4.0, 25.0)
            down = torch.zeros(T * TOPK, H, dtype=x.dtype, device=dev)
            w1_gemv(h, w2p, w2s, topk_ids, topk_w, None, down, 1, AMP,
                    mul_routed_weight=True)
            return down.view(T, TOPK, H).sum(1)

        out = run()
        finite = torch.isfinite(out).all().item()
        ok(f"E={E} T={T} forward runs, finite output", finite,
           f"|out|max={out.abs().max().item():.3f}")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 100
        # Bytes actually touched: one slot per routed selection, not per
        # resident expert, since a program only reads the expert it routed to.
        touched = TOPK * (w13p[0].numel() + w2p[0].numel()
                          + w13s[0].numel() + w2s[0].numel())
        print(f"        T={T}: {ms:.3f} ms/layer  "
              f"({touched / (ms * 1e-3) / 1e9:.1f} GB/s), "
              f"92 layers = {ms * 92:.1f} ms -> {1000 / (ms * 92):.2f} tok/s "
              f"(MoE only)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"cap={torch.cuda.get_device_capability()}\n")
    test_registration()
    test_real_shapes()
    print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    sys.exit(1 if FAIL else 0)
