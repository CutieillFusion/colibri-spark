#!/usr/bin/env python3
"""Load REAL weights: dense from the checkpoint, experts from the 1-bit store.

test_model_build.py used load_format=dummy, so it proved the plumbing but not
that any actual bytes arrive in the right place. This one loads for real and
then checks the expert parameters against the store file read independently --
if the loader put a slot in the wrong place, or transposed w1/w3, or picked
the wrong layer ordinal, this catches it.

Still a truncated model, so the generated text is not meaningful. Truncation
changes the function being computed; only the full 93 layers can be judged on
output quality.

Env:  K3_W1_DIR (required), K3_W1_SHARD (optional)
Usage: python3 test_real_load.py [num_layers] [num_experts]
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, __file__.rsplit("/", 2)[0])

LAYERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
EXPERTS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SRC = os.environ.get("K3_SRC", "/nas/models/moonshotai/Kimi-K3")
FAIL = []


def eq(name, cond, note=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name:44s} {note}")
    if not cond:
        FAIL.append(name)


def make_tiny(dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(SRC):
        p = os.path.join(SRC, f)
        if os.path.isfile(p) and f != "config.json":
            if f.endswith(".safetensors"):
                # symlink the shards so the loader sees the real data without
                # copying 1.5 TB
                lk = os.path.join(dst, f)
                if not os.path.exists(lk):
                    os.symlink(p, lk)
            else:
                shutil.copy2(p, dst)
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    t = cfg.get("text_config", cfg)
    t["num_hidden_layers"] = LAYERS
    t["num_experts"] = EXPERTS
    t["num_experts_per_token"] = min(4, EXPERTS)
    t["n_group"] = t["num_expert_group"] = t["topk_group"] = 1
    lac = t.get("linear_attn_config", {})
    lac["full_attn_layers"] = [i for i in lac.get("full_attn_layers", [])
                               if i < LAYERS]
    t["linear_attn_config"] = lac
    t.pop("quantization_config", None)
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=1)
    return dst


def main():
    if not os.environ.get("K3_W1_DIR"):
        sys.exit("set K3_W1_DIR to the directory holding experts.w2")
    import torch

    import vllm_k3_w1  # noqa: F401
    from vllm import LLM, SamplingParams
    from vllm_k3_w1.loader import K3W1Store

    d = make_tiny(os.path.join(tempfile.gettempdir(), "k3-real"))
    print(f"tiny K3: {LAYERS} layers, {EXPERTS} experts, real weights\n")

    llm = LLM(model=d, quantization="k3_w1", trust_remote_code=True,
              tensor_parallel_size=1, max_model_len=256,
              gpu_memory_utilization=0.75, enforce_eager=True,
              max_num_batched_tokens=1024, max_num_seqs=1,
              limit_mm_per_prompt={"image": 0, "video": 0})
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Independent read of the store, compared against what landed in the layer.
    sh = os.environ.get("K3_W1_SHARD", "")
    sworld = int(sh.split("/")[1]) if sh else 1
    store = K3W1Store(os.environ["K3_W1_DIR"], 3584, 3072,
                      896 // sworld if sworld > 1 else 896)
    print(f"\n{store}")

    checked = 0
    for name, mod in model.named_modules():
        if not hasattr(mod, "w13_qweight"):
            continue
        import re
        li = int(re.search(r"layers\.(\d+)\.", name).group(1))
        ordinal = li - 1                      # first_k_dense_replace = 1
        for slot in (0, min(3, EXPERTS - 1)):
            s = store.read_slot(ordinal, slot)
            I = mod.w13_qweight.shape[1] // 2
            got_w1 = mod.w13_qweight[slot][:I].cpu().numpy()
            got_w3 = mod.w13_qweight[slot][I:].cpu().numpy()
            got_w2 = mod.w2_qweight[slot].cpu().numpy()
            got_s1 = mod.w13_scales[slot][:I].cpu().numpy()
            ok = ((got_w1 == s["w1p"]).all() and (got_w3 == s["w3p"]).all()
                  and (got_w2 == s["w2p"]).all() and (got_s1 == s["w1s"]).all())
            eq(f"layer {li} expert {slot} bytes match store", ok)
            checked += 1
    eq("checked at least one slot", checked > 0, f"{checked} slots")

    # Not all-zero, not all-same: a store read that silently returned nothing
    # would still "match" if both sides were empty.
    for name, mod in model.named_modules():
        if hasattr(mod, "w13_qweight"):
            u = int(torch.unique(mod.w13_qweight[0][:64]).numel())
            eq("expert bytes look like real data", u > 32, f"{u} distinct values")
            break

    out = llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=12, temperature=0))
    print(f"\ngenerated (truncated model -> not meaningful):\n  "
          f"{out[0].outputs[0].text!r}")
    print(f"\n{'REAL LOAD OK' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
