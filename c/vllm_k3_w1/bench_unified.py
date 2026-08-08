#!/usr/bin/env python3
"""Is a CUDA allocation on GB10 charged once or twice against system RAM?

This decides whether the vLLM port has ~3 GB of headroom or ~-11 GB. Our own
engine prints, at startup:

    [K3/CUDA] device 0 is INTEGRATED (#653): placement duplicates weights in
              shared RAM and shrinks the expert cache by the same amount

and needed a purpose-built reclaim pass (unpin + MADV_DONTNEED + PROT_NONE) to
recover 14.47 GB. But that duplication is a property of *how the engine loads*
-- host buffer, then mirror to device, keeping both -- not necessarily of the
hardware. If torch's device allocations come out of the same physical pool and
the host copy is dropped, there is nothing to reclaim.

Measures MemAvailable, which counts what the system can actually hand out,
rather than RSS, which misreports shared/unified pages.
"""

import gc
import sys

import torch

GB = 1 << 30


def mem():
    """(MemFree, MemAvailable) in GB, from the kernel rather than RSS."""
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            d[k] = int(v.split()[0]) * 1024
    return d["MemFree"] / 1e9, d["MemAvailable"] / 1e9


def rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / 1e9
    return 0.0


def report(tag, base_avail, base_rss):
    f, a = mem()
    print(f"  {tag:34s} avail {a:7.1f} GB (d {base_avail - a:+6.1f})  "
          f"rss {rss():6.1f} (d {rss() - base_rss:+6.1f})  "
          f"torch {torch.cuda.memory_allocated() / 1e9:6.1f}")
    return a


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    torch.cuda.init()
    gc.collect(); torch.cuda.empty_cache()
    _, base_a = mem(); base_r = rss()
    print(f"device {torch.cuda.get_device_name(0)}  "
          f"cap {torch.cuda.get_device_capability()}")
    print(f"baseline: avail {base_a:.1f} GB, rss {base_r:.1f} GB\n")

    print(f"[A] direct device allocation of {N} GB")
    xs = [torch.empty(GB, dtype=torch.uint8, device="cuda") for _ in range(N)]
    for x in xs:
        x.fill_(1)                      # touch, so nothing is lazily unbacked
    torch.cuda.synchronize()
    a1 = report(f"{N} GB on cuda", base_a, base_r)
    print(f"      -> charged {base_a - a1:.1f} GB for {N} GB requested "
          f"= {(base_a - a1) / N:.2f}x")
    del xs
    gc.collect(); torch.cuda.empty_cache()
    report("after free + empty_cache", base_a, base_r)

    print(f"\n[B] host -> device copy, host copy dropped ({N} GB)")
    keep = []
    for _ in range(N):
        h = torch.empty(GB, dtype=torch.uint8)     # CPU
        h.fill_(2)
        keep.append(h.to("cuda", non_blocking=False))
        del h
    torch.cuda.synchronize()
    gc.collect()
    a2 = report(f"{N} GB device, host dropped", base_a, base_r)
    print(f"      -> charged {base_a - a2:.1f} GB = {(base_a - a2) / N:.2f}x")

    print(f"\n[C] the engine's pattern: keep BOTH copies alive ({N} GB)")
    hosts = [torch.empty(GB, dtype=torch.uint8) for _ in range(N)]
    for h in hosts:
        h.fill_(3)
    a3 = report(f"{N} GB device + {N} GB host", base_a, base_r)
    print(f"      -> charged {base_a - a3:.1f} GB for {N} GB of weights "
          f"= {(base_a - a3) / N:.2f}x")
    del hosts, keep
    gc.collect(); torch.cuda.empty_cache()
    report("after free", base_a, base_r)

    print("\nverdict: [A]/[B] near 1.0x means unified allocation is charged")
    print("         once and the port needs no reclaim pass. Near 2.0x means")
    print("         it does, and ~14 GB has to come from somewhere.")


if __name__ == "__main__":
    main()
