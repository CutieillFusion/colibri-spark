#!/usr/bin/env python3
"""Build a real (truncated) Kimi-K3 through vLLM with --quantization k3_w1.

Everything so far has tested the kernels and the registration in isolation.
This exercises the actual path: config -> registry -> model construction ->
get_quant_method for every layer -> create_weights -> the dummy loader ->
process_weights_after_loading. It is the cheapest thing that can catch a
plumbing mistake -- a parameter shape vLLM disagrees with, a layer our method
is never asked about, a quantization override that does not take.

Truncated to 5 layers (0 dense, 1-3 KDA, 4 MLA, matching the real
first_k_dense_replace=1 and full_attn_layers stride of 4) and 32 experts, so
it fits one node at TP=1. Architecture dims are left alone so the code path
is the real one.

Usage:  python3 test_model_build.py [num_layers] [num_experts]
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, __file__.rsplit("/", 2)[0])

LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
EXPERTS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
SRC = "/nas/models/moonshotai/Kimi-K3"


def make_tiny(dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(SRC):
        p = os.path.join(SRC, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") and f != "config.json":
            shutil.copy2(p, dst)
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    t = cfg.get("text_config", cfg)
    t["num_hidden_layers"] = LAYERS
    t["num_experts"] = EXPERTS
    t["num_experts_per_token"] = min(4, EXPERTS)
    t["n_group"] = 1
    t["num_expert_group"] = 1
    t["topk_group"] = 1
    lac = t.get("linear_attn_config", {})
    lac["full_attn_layers"] = [i for i in lac.get("full_attn_layers", [])
                               if i < LAYERS]
    t["linear_attn_config"] = lac
    # The checkpoint declares compressed-tensors/MXFP4; --quantization must
    # win, so drop it and let the CLI flag be the only source of truth.
    t.pop("quantization_config", None)
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    print(f"tiny K3: {LAYERS} layers, {EXPERTS} experts, "
          f"full_attn={lac['full_attn_layers']}")
    return dst


def main():
    import vllm_k3_w1  # noqa: F401  registers k3_w1
    from vllm import LLM

    d = make_tiny(os.path.join(tempfile.gettempdir(), "k3-tiny"))
    llm = LLM(
        model=d,
        quantization="k3_w1",
        load_format="dummy",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=512,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
        # B=1 serving: activation memory scales with the prefill chunk, not
        # with anything we need, so cap it.
        max_num_batched_tokens=int(os.environ.get("K3_MNBT", "8192")),
        max_num_seqs=int(os.environ.get("K3_MNS", "256")),
        **({"limit_mm_per_prompt": {"image": 0, "video": 0}}
           if os.environ.get("K3_NOMM") else {}),
    )

    # Walk the built model and report which method landed on what.
    import collections

    import torch

    from vllm_k3_w1.dense_method import KimiK3DenseLinearMethod
    from vllm_k3_w1.method import KimiK3OneBitMoEMethod

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    counts = collections.Counter()
    bad = []
    miss = []
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None:
            continue
        if isinstance(qm, KimiK3OneBitMoEMethod):
            counts["moe:1bit"] += 1
            for p in ("w13_qweight", "w13_scales", "w2_qweight", "w2_scales"):
                if not hasattr(mod, p):
                    bad.append(f"{name} missing {p}")
        elif isinstance(qm, KimiK3DenseLinearMethod):
            counts[f"dense:int{getattr(mod, 'k3_bits', '?')}"] += 1
            if not hasattr(mod, "k3_qweight"):
                bad.append(f"{name} not quantized")
        else:
            vis = "vision" in name or "mm_projector" in name
            counts[f"other:{type(qm).__name__}{' (vision)' if vis else ''}"] += 1
            if not vis and "Linear" in type(qm).__name__:
                miss.append((name, tuple(getattr(mod, "weight",
                                                 torch.empty(0)).shape)))

    print("\nquant methods actually bound:")
    for k, v in sorted(counts.items()):
        print(f"  {k:38s} {v}")
    if miss:
        # Anything here is a language-model matrix still sitting in bf16, i.e.
        # 4x its quantized size. Worth knowing about by name.
        nb = 0
        for _, sh in miss:
            n = 2
            for dim in sh:
                n *= dim
            nb += n
        print(f"\n  UNQUANTIZED language-model linears ({len(miss)}, "
              f"{nb / 1e6:.0f} MB in this 5-layer model):")
        seen = set()
        for n, sh in miss:
            key = __import__("re").sub(r"\.\d+\.", ".N.", n)
            if key in seen:
                continue
            seen.add(key)
            print(f"    {key[-56:]:56s} {sh}")
    if bad:
        print("\nPROBLEMS:")
        for b in bad[:10]:
            print(f"  {b}")

    out = llm.generate(["The capital of France is"],
                       __import__("vllm").SamplingParams(max_tokens=8,
                                                         temperature=0))
    txt = out[0].outputs[0].text
    print(f"\ngenerated (random weights, so gibberish is expected and fine):")
    print(f"  {txt!r}")
    from vllm_k3_w1 import patch_latent
    lq, lt = patch_latent.verify(model)
    print(f"\nlatent projections quantized: {lq}/{lt}")
    if lt and lq != lt:
        bad.append(f"latent projections still bf16: {lt - lq}/{lt}")
    print(f"\n{'BUILD OK' if not bad else 'BUILD HAD PROBLEMS'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
