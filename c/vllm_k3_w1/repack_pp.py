#!/usr/bin/env python3
"""Repack the 1-bit expert store from TP=4 to TP=2 x PP=2.

Why: at TP=4 a flat 4-rank communicator spans both RoCE islands, so every one
of the ~186 all-reduces per token is forced onto the 1 GbE bridge at 5.5 ms
each -- about 1 s/token before any compute. With TP=2 inside an island the
same all-reduce costs 0.020 ms over RoCE, and only a couple of activation
handoffs per token cross the bridge. It also halves everything replicated
per rank (latent projections, router, norms), which is where the memory
margin has to come from.

Layout. The TP=4 store holds e%4==r for all 92 MoE layers, addressed
    slot = moe_ordinal * 224 + e//4                       (kimi_k3.c:1455)
The TP=2 x PP=2 store holds one PP stage's layers and e%2==t:
    slot = local_moe_ordinal * 448 + j,   global e = 2j + t

Since e = 2j + t gives e%4 == t for even j and t+2 for odd j, with e//4 == j//2
either way, the target is a straight interleave of two source shards: even
target slots from shard t, odd from shard t+2. Both sources contribute the
same layer range, so a stage's half of each source is all that has to move.

Usage (per node):
    repack_pp.py slice  <src_store> <out.bin> --first N --count N
    repack_pp.py merge  <even.bin> <odd.bin> <out_dir> --layers N
"""

import argparse
import os
import sys
import time

SLOT = 5160960                    # 4.92 MiB, checked against the store below
SRC_EXPERTS = 224                 # experts per layer in a TP=4 shard
CHUNK = 64                        # slots per read, keeps I/O sequential


def store_path(p):
    return os.path.join(p, "experts.w2") if os.path.isdir(p) else p


def check(path, experts_per_layer):
    n = os.path.getsize(path)
    if n % SLOT:
        sys.exit(f"{path}: {n} is not a whole number of {SLOT}-byte slots")
    slots = n // SLOT
    if slots % experts_per_layer:
        sys.exit(f"{path}: {slots} slots is not a multiple of {experts_per_layer}")
    return slots, slots // experts_per_layer


def cmd_slice(a):
    """Copy a contiguous run of MoE layers out of a store, verbatim."""
    src = store_path(a.src)
    slots, layers = check(src, SRC_EXPERTS)
    if a.first + a.count > layers:
        sys.exit(f"{src} has {layers} layers; asked for [{a.first},"
                 f"{a.first + a.count})")
    total = a.count * SRC_EXPERTS * SLOT
    print(f"slice {src} layers [{a.first},{a.first + a.count}) "
          f"-> {a.out}  {total / 1e9:.1f} GB", flush=True)
    t0 = time.time()
    done = 0
    with open(src, "rb", buffering=0) as fi, open(a.out, "wb", buffering=0) as fo:
        fi.seek(a.first * SRC_EXPERTS * SLOT)
        left = total
        buf = bytearray(CHUNK * SLOT)
        while left:
            n = min(left, len(buf))
            got = fi.readinto(memoryview(buf)[:n])
            if got != n:
                sys.exit(f"short read at {done}")
            fo.write(memoryview(buf)[:n])
            left -= n
            done += n
            if done % (16 << 30) < len(buf):
                el = time.time() - t0
                print(f"  {done / 1e9:6.1f}/{total / 1e9:.1f} GB "
                      f"{done / el / 1e6:.0f} MB/s", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_merge(a):
    """Interleave two same-layer-range slices into one TP=2 store.

    `even` supplies target slots with j even (source shard t), `odd` supplies
    j odd (source shard t+2). Reading both sequentially and writing one
    output keeps every stream linear.
    """
    for p in (a.even, a.odd):
        n = os.path.getsize(p)
        want = a.layers * SRC_EXPERTS * SLOT
        if n != want:
            sys.exit(f"{p}: {n} bytes, expected {want} for {a.layers} layers")
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "experts.w2")
    total = a.layers * 2 * SRC_EXPERTS * SLOT
    print(f"merge -> {out}  {total / 1e9:.1f} GB "
          f"({a.layers} layers x {2 * SRC_EXPERTS} experts)", flush=True)
    t0 = time.time()
    done = 0
    with open(a.even, "rb", buffering=0) as fe, \
         open(a.odd, "rb", buffering=0) as fo_, \
         open(out, "wb", buffering=0) as fw:
        be, bo = bytearray(SLOT), bytearray(SLOT)
        for L in range(a.layers):
            for i in range(SRC_EXPERTS):
                if fe.readinto(be) != SLOT or fo_.readinto(bo) != SLOT:
                    sys.exit(f"short read at layer {L} expert {i}")
                fw.write(be)          # target j = 2i   (even -> shard t)
                fw.write(bo)          # target j = 2i+1 (odd  -> shard t+2)
                done += 2 * SLOT
            if L % 8 == 0 and L:
                el = time.time() - t0
                print(f"  layer {L}/{a.layers}  {done / 1e9:6.1f} GB  "
                      f"{done / el / 1e6:.0f} MB/s", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice")
    s.add_argument("src"); s.add_argument("out")
    s.add_argument("--first", type=int, required=True)
    s.add_argument("--count", type=int, required=True)
    s.set_defaults(fn=cmd_slice)
    m = sub.add_parser("merge")
    m.add_argument("even"); m.add_argument("odd"); m.add_argument("out_dir")
    m.add_argument("--layers", type=int, required=True)
    m.set_defaults(fn=cmd_merge)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
