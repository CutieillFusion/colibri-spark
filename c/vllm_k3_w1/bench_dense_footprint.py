#!/usr/bin/env python3
"""Measure the real per-node dense footprint, against the 14.0 GB prediction.

Builds every non-expert tensor at its actual checkpoint shape, applies the
TP=4 split the vLLM model would apply, quantizes with the same functions
KimiK3DenseLinearMethod uses, and measures MemAvailable rather than RSS
(bench_unified.py showed RSS underreports unified pages by ~30x).

Also measures PEAK, because vLLM loads every weight before running any
process_weights_after_loading hook -- so the bf16 set and the quantized set
are briefly alive together. That peak, not the steady state, is what decides
whether the expert store can already be resident when dense loads.

Reads shapes from the safetensors headers only; no weight data is read, so
this runs in seconds rather than reading 1.5 TB.
"""

import glob
import json
import re
import struct
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from vllm_k3_w1.dense_kernels import quantize_i4g, quantize_i8  # noqa: E402
from vllm_k3_w1.dense_method import bits_for_prefix  # noqa: E402

CKPT = "/nas/models/moonshotai/Kimi-K3"
TP = 4
SKIP_VISION = True


def mem_avail():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 1e9
    return 0.0


# How each tensor family shards at TP=4. "col" splits dim 0 (output),
# "row" splits dim 1 (input), "rep" replicates. The latent projections are
# ReplicatedLinear by design (models/kimi_k3/nvidia/model.py:558,575) and the
# router is replicated on every rank.
def shard_of(name):
    # Verified against the model rather than guessed:
    #   kda.py:344  in_proj_qkvgfab is a MergedColumnParallelLinear, so the
    #               checkpoint's separate q/k/v/g_proj/f_a/beta all shard --
    #               g_proj alone is 16.4 GB, so getting this wrong costs 12 GB
    #   kda.py:356  f_b_proj  ColumnParallel      kda.py:370  conv1d Column
    #   kda.py:458  o_proj    RowParallel
    #   mla.py:187  fused_qkv_a_proj MergedColumn    :196 q_b_proj Column
    #   mla.py:213  kv_a_proj_with_mqa REPLICATED    :221 kv_b_proj Column
    #   mla.py:251  o_proj    RowParallel
    #   model.py:558,575  routed_expert_{down,up}_proj REPLICATED by design
    if re.search(r"\.(o_proj|down_proj|out_proj)\.weight$", name):
        return "row"
    if "kv_a_proj_with_mqa" in name:
        return "rep"                       # low-rank latent needed in full
    if "routed_expert_" in name or "block_sparse_moe.gate" in name:
        return "rep"
    if re.search(r"self_attn\..*_proj\.weight$", name):
        return "col"                       # every other attention projection
    if re.search(r"\.(gate_proj|up_proj|gate_up_proj|in_proj|conv1d)\.weight$",
                 name):
        return "col"
    if "lm_head" in name or "embed" in name:
        return "col"
    return "rep"


def main():
    hdrs = {}
    for p in sorted(glob.glob(CKPT + "/*.safetensors")):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdrs.update(json.loads(f.read(n)))
    hdrs.pop("__metadata__", None)

    plan = []
    for name, m in hdrs.items():
        if ".experts." in name and ("weight_packed" in name
                                    or "weight_scale" in name):
            continue
        if SKIP_VISION and ("vision" in name or "mm_projector" in name):
            continue
        shape = m["shape"]
        if len(shape) != 2 or min(shape) < 2:
            continue                      # norms/biases: negligible, skip
        O, I = shape
        s = shard_of(name)
        if s == "col":
            O = O // TP
        elif s == "row":
            I = I // TP
        # VERIFIED by building the model (test_model_build.py): vLLM
        # constructs the latent projections and the router gate with
        # quant_config=None (model.py:562,579), so get_quant_method is never
        # consulted for them and they stay bf16 no matter what we register.
        # Counting them as quantized understated the footprint by ~7.7 GB.
        # The router gate keeps quant_config=None and stays bf16, which is
        # what we want anyway -- it is the routing decision. The latent
        # projections are reclaimed by patch_latent (verified 8/8 in
        # test_model_build.py), so they quantize like anything else.
        if re.search(r"block_sparse_moe\.gate", name):
            bits = 16
        else:
            bits = bits_for_prefix(name.replace(".weight", ""), 4, 8, 8)
        plan.append((name, O, I, bits))

    bf16_b = sum(O * I * 2 for _, O, I, _ in plan)
    print(f"{len(plan)} matrices, {bf16_b / 1e9:.1f} GB bf16 per node at TP={TP}"
          f"{' (vision skipped)' if SKIP_VISION else ''}")

    # What is replicated, and is that right? A wrong "rep" is worth 4x, so
    # print the biggest ones rather than trusting the regex.
    import collections
    rep = collections.Counter()
    for name, m in hdrs.items():
        if ".experts." in name and ("weight_packed" in name
                                    or "weight_scale" in name):
            continue
        if len(m["shape"]) != 2 or min(m["shape"]) < 2:
            continue
        if shard_of(name) == "rep":
            rep[re.sub(r"\.\d+\.", ".N.", name)] += (
                m["data_offsets"][1] - m["data_offsets"][0])
    print(f"\n  replicated (x{TP} cost vs sharded):")
    for k, v in rep.most_common(8):
        print(f"    {k[-58:]:58s} {v / 1e9:6.2f} GB")
    print(f"    {'TOTAL replicated':58s} {sum(rep.values()) / 1e9:6.2f} GB\n")

    base = mem_avail()
    print(f"baseline MemAvailable {base:.1f} GB\n")

    # Phase 1: all bf16 resident, as vLLM has it just after load_weights.
    bufs = [torch.empty(O, I, dtype=torch.bfloat16, device="cuda")
            for _, O, I, _ in plan]
    torch.cuda.synchronize()
    after_bf16 = mem_avail()
    print(f"[1] bf16 loaded          {base - after_bf16:7.1f} GB consumed")

    # Phase 2: quantize module by module, freeing each bf16 as vLLM's
    # per-module process_weights_after_loading would.
    peak = after_bf16
    q = []
    for i, (_, O, I, bits) in enumerate(plan):
        w = bufs[i]
        if bits >= 16:
            q.append((w.clone(), torch.empty(0, device=w.device)))  # stays bf16
        else:
            b = bits if not (bits == 4 and I % 64) else 8
            q.append(quantize_i4g(w, 64) if b == 4 else quantize_i8(w))
        bufs[i] = None
        del w
        if i % 200 == 0:
            peak = min(peak, mem_avail())
    torch.cuda.synchronize()
    del bufs
    # Without this the caching allocator keeps every freed bf16 block and the
    # measurement reads ~4x the real steady state.
    torch.cuda.empty_cache()
    after_q = mem_avail()
    qb = sum(a.numel() * a.element_size() + s.numel() * s.element_size()
             for a, s in q)
    print(f"[2] quantized, bf16 freed{base - after_q:7.1f} GB consumed "
          f"(tensor bytes {qb / 1e9:.1f} GB)")
    print(f"    transient peak       {base - peak:7.1f} GB")

    exp = 224 * 92 * 5160960 / 1e9
    dense = base - after_q
    print(f"\nper-node budget")
    print(f"  dense (measured)       {dense:7.1f} GB")
    print(f"  experts 1-bit, 224x92  {exp:7.1f} GB")
    print(f"  {'-' * 32}")
    print(f"  weights total          {dense + exp:7.1f} GB")
    print(f"  MemAvailable (idle)    {base:7.1f} GB")
    print(f"  headroom               {base - dense - exp:7.1f} GB "
          f"<- needs ~2.5 framework + 0.45/seq KDA state + KV")


if __name__ == "__main__":
    main()
