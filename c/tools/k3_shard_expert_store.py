#!/usr/bin/env python3
"""Write shard r/N of the packed expert store to a local file.

Recovered into the repo after a reboot cleared /tmp and took the only copy with
it. The full provisioning chain for the four-Spark deployment is:

  1. stage the snapshot   rsync /nas/models/moonshotai/Kimi-K3 -> <node>:<dir>
  2. build the store      /nas/vllm-moet/tools/pack_experts_2bit.py \
                            --src <snapshot> --out <store> --bits 1
                          (--bits 1 keeps only the e2m1 sign bit; the 2-bit
                           codebook is sign-symmetric so the sign is all a
                           1-bit quantiser can keep. 8.86 -> 4.92 MiB/expert)
  3. shard it             this script, once per rank
  4. point the engine at it with K3_W1_DIR and K3_W2_SHARD=r/N

Keep the working set OFF /tmp: /usr/lib/tmpfiles.d/tmp.conf carries
"D /tmp 1777 root root 30d", and the D empties it at every boot.

Slots for experts with e%N==r, in (moe_layer, e/N) order — the layout the
engine's K3_W2_SHARD=r/N expects. Written locally rather than piped over ssh:
streaming through a python+crypto pipe managed only ~110 MB/s, while writing
to NVMe and then rsyncing the finished file is several times faster.
"""
import json
import os
import sys

d, r, N, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
man = json.load(open(os.path.join(d, "manifest.json")))
slot, E, L = man["slot_bytes"], man["n_experts"], len(man["layers"])
print(f"[shard] {r}/{N}: {L} layers x {E//N} experts x {slot} B "
      f"= {L*(E//N)*slot/1e9:.1f} GB", file=sys.stderr, flush=True)
with open(os.path.join(d, "experts.w2"), "rb", buffering=0) as f, \
     open(out, "wb", buffering=1 << 25) as o:
    for li in range(L):
        for e in range(r, E, N):
            f.seek((li * E + e) * slot)
            got = 0
            while got < slot:
                b = f.read(slot - got)
                if not b:
                    raise SystemExit(f"short read L{li} e{e}")
                o.write(b)
                got += len(b)
        if li % 10 == 0:
            print(f"[shard] layer {li+1}/{L}", file=sys.stderr, flush=True)
print("[shard] PACKED", file=sys.stderr, flush=True)
