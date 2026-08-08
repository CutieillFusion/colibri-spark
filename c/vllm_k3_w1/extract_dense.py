#!/usr/bin/env python3
"""Write a dense-only K3 checkpoint: everything except the MXFP4 experts.

The full checkpoint is 1.5 TB, of which 1446 GB is routed-expert MXFP4 that we
replace with the 1-bit store and therefore never read. Pointing vLLM at it
anyway costs 3.4 hours per node over NFS. This extracts the 114 GB that is
actually needed, seeking to each tensor's byte range rather than streaming
whole shards, so the read is 114 GB and not 1.5 TB.

Output is plain safetensors plus the config/tokenizer files, so vLLM loads it
with no special casing -- the expert parameters are filled separately by
patch_loader from experts.w2.

Usage:
    python3 extract_dense.py <src> <dst> [--layers N] [--experts N]

--layers/--experts rewrite the config for a truncated model; the tensor set is
filtered to match so nothing dangles.
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time

EXPERT_RE = re.compile(r"\.experts\.\d+\.(w1|w2|w3)\.weight_(packed|scale)$")
LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
SHARD_MAX = 40 << 30          # 40 GB per output shard


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--experts", type=int, default=0)
    # Text-only serving: with limit_mm_per_prompt image=0 the vision tower is
    # not built at all (model load drops 7.78 -> 6.95 GiB), so its weights
    # would just sit unused.
    ap.add_argument("--keep-vision", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    shards = sorted(f for f in os.listdir(a.src) if f.endswith(".safetensors"))
    plan = []                                  # (src, offset, nbytes, name, meta)
    total = 0
    for sf in shards:
        p = os.path.join(a.src, sf)
        hdr, base = read_header(p)
        for name, m in hdr.items():
            if name == "__metadata__":
                continue
            if EXPERT_RE.search(name):
                continue
            if not a.keep_vision and ("vision" in name or "mm_projector" in name):
                continue
            if a.layers:
                lm = LAYER_RE.search(name)
                if lm and int(lm.group(1)) >= a.layers:
                    continue
            lo, hi = m["data_offsets"]
            plan.append((p, base + lo, hi - lo, name, m))
            total += hi - lo
    print(f"{len(plan)} tensors, {total / 1e9:.1f} GB (from {len(shards)} shards)")

    # Group into output shards, writing each as a standalone safetensors file.
    groups, cur, cur_b = [], [], 0
    for item in plan:
        if cur and cur_b + item[2] > SHARD_MAX:
            groups.append(cur)
            cur, cur_b = [], 0
        cur.append(item)
        cur_b += item[2]
    if cur:
        groups.append(cur)

    index = {"metadata": {"total_size": total}, "weight_map": {}}
    t0 = time.time()
    done = 0
    for gi, g in enumerate(groups):
        out_name = f"model-{gi + 1:05d}-of-{len(groups):05d}.safetensors"
        out = os.path.join(a.dst, out_name)
        hdr, off = {}, 0
        for _, _, nb, name, m in g:
            hdr[name] = {"dtype": m["dtype"], "shape": m["shape"],
                         "data_offsets": [off, off + nb]}
            index["weight_map"][name] = out_name
            off += nb
        blob = json.dumps(hdr).encode()
        pad = (-len(blob)) % 8
        blob += b" " * pad
        with open(out, "wb") as fo:
            fo.write(struct.pack("<Q", len(blob)))
            fo.write(blob)
            src_open, fh = None, None
            for sp, so, nb, _, _ in g:
                if sp != src_open:
                    if fh:
                        fh.close()
                    fh = open(sp, "rb", buffering=0)
                    src_open = sp
                fh.seek(so)
                left = nb
                while left:
                    chunk = fh.read(min(left, 1 << 24))
                    if not chunk:
                        sys.exit(f"short read in {sp}")
                    fo.write(chunk)
                    left -= len(chunk)
                done += nb
            if fh:
                fh.close()
        el = time.time() - t0
        print(f"  {out_name}  {off / 1e9:6.1f} GB   "
              f"{done / 1e9:6.1f}/{total / 1e9:.1f} GB  "
              f"{done / el / 1e6:.0f} MB/s  eta {(total - done) / (done / el) / 60:.0f}m")

    json.dump(index, open(os.path.join(a.dst, "model.safetensors.index.json"), "w"))

    for f in os.listdir(a.src):
        p = os.path.join(a.src, f)
        if os.path.isfile(p) and not f.endswith(".safetensors") \
                and not f.endswith(".index.json"):
            shutil.copy2(p, a.dst)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    t = cfg.get("text_config", cfg)
    if a.layers:
        t["num_hidden_layers"] = a.layers
        lac = t.get("linear_attn_config", {})
        lac["full_attn_layers"] = [i for i in lac.get("full_attn_layers", [])
                                   if i < a.layers]
        t["linear_attn_config"] = lac
    if a.experts:
        t["num_experts"] = a.experts
        t["num_experts_per_token"] = min(t.get("num_experts_per_token", 8),
                                         a.experts)
    # The 1-bit store replaces the checkpoint's MXFP4, so the declaration must
    # go -- otherwise compressed-tensors competes with --quantization k3_w1.
    t.pop("quantization_config", None)
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(a.dst, "config.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {a.dst}")


if __name__ == "__main__":
    main()
