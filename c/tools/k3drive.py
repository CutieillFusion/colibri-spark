#!/usr/bin/env python3
"""Drive one kimi_k3 SERVE rank through cold + warm1 + warm2, sequentially.

Requests are submitted one at a time: the next SUBMIT is written only after
this rank's DONE for the previous one has been read, so nothing is prequeued.
DATA records are length-framed and may contain newlines, so the reader honours
the frame instead of scanning for a line.
"""
import os, subprocess, sys, time

TAG  = sys.argv[1]
RANK = int(sys.argv[2])
BIN  = "/home/norquistdylan/colibri-spark/c/kimi_k3"
LOG  = f"/tmp/k3-{TAG}-r{RANK}.log"

_BASE = b"Explain how a B-tree index works, then write the insert function in C."
# CTXPAD=N prepends filler to reach roughly N tokens of context, for measuring
# how decode throughput degrades with sequence length. The 69 KDA layers hold a
# fixed-size recurrent state; only the 24 MLA layers cache KV, so memory grows
# at 27 KB/token and the real limit is MLA attention cost, not RAM.
_PAD = int(os.environ.get("CTXPAD", "0"))
if _PAD > 0:
    _filler = (b"The quick brown fox jumps over the lazy dog near the river bank. "
               b"Databases index data to make lookups fast and predictable. ")
    # ~0.25 tok/byte for English; overshoot slightly then let the engine report np
    MSG = _filler * max(1, int(_PAD * 4 / len(_filler))) + _BASE
else:
    MSG = _BASE
WIRE = b"K3CHAT1\nM user %d\n%sG 0\n" % (len(MSG), MSG)

env = dict(os.environ)
env.update({
    "SERVE": "1", "SNAP": "/home/norquistdylan/k3data/Kimi-K3",
    "K3_WORLD": "4", "K3_GROUP_SIZE": "2", "K3_NET_RD2": "1",
    "K3_NET_SPLIT": os.environ.get("SPLIT", "1"),
    "K3_NET_TTFB": os.environ.get("TTFB", "0"),
    "K3_NET_BUSYPOLL": os.environ.get("BPOLL", "50"),
    "K3_MASTER_PORT": os.environ.get("PORT", "29901"),
    "K3_RANK": str(RANK),
    "K3_GPUS": "0", "K3_EXPERT_GPU": "1", "K3_EXPERT_GB": os.environ.get("EGB", "107"), "K3_GPU_GB": "30",
    "K3_MAXT": os.environ.get("MAXT", "512"), "K3_CHUNK": os.environ.get("CHUNK", "32"), "K3_TP_ATTN": "1",
    "K3_PF_SHAPE": os.environ.get("PFSHAPE", "tile"),
    "K3_KVNEON": os.environ.get("KVNEON", "1"),
    "K3_MATT_BATCH": os.environ.get("MATTB", "0"),
    "K3_AR_PERSIST": os.environ.get("ARP", "1"),
    "K3_CTL_PERSIST": os.environ.get("CTLP", "0"),
    "K3_DENSE_ILP": os.environ.get("ILP", "1"),
    "K3_DENSE_MULTI": os.environ.get("MULTI", "0"),
    "K3_FDOT": os.environ.get("FDOT", "0"),
    "K3_BFDOT": os.environ.get("BFDOT", "0"),
    "K3_DENSE_MANAGED": os.environ.get("MGD", "0"),
    "K3_FREE_HOST": os.environ.get("FREEH", "1"),
    "K3_PREFILL_GPU": os.environ.get("PFGPU", "1"),
    **({} if os.environ.get("MBITS","")=="" else {"K3_MLA_BITS": os.environ["MBITS"]}),
    **({} if os.environ.get("HBITS","")=="" else {"K3_HEAD_BITS": os.environ["HBITS"]}),
    **({} if os.environ.get("BITS","")=="" else {"K3_BITS": os.environ["BITS"]}),
    "K3_ROUTER_I8": os.environ.get("RI8", "0"),
    "K3_ROUTER_F16": os.environ.get("RF16", "0"),
    "K3_ROUTER2": os.environ.get("R2", "0"),
    "K3_HUGE": os.environ.get("HUGE", "1"),
    "K3_PIN": os.environ.get("PIN", "1"),
    "OMP_NUM_THREADS": os.environ.get("OMPT", "8"),
    "K3_DENSE_GPU": "1", "K3_DENSE_EXACT": "1",
    "K3_KDA_OVERLAP": os.environ.get("OVERLAP", "1"),
    "K3_CTL_GPU": os.environ.get("CTLGPU", "1"),
    "K3_CTL_GPU2": os.environ.get("CTLGPU2", "0"),
    **({} if os.environ.get("DEVGB")=="auto" else
       {"K3_DENSE_DEV_GB": os.environ.get("DEVGB", "18")}),  # "auto" = let the engine size it
    "K3_DENSE_I4W": os.environ.get("I4W", "4"),
    "K3_DENSE_RTHRESH": os.environ.get("RTHR", "4096"),
    "K3_DENSE_CENSUS": os.environ.get("CENSUS", "0"),
    "K3_EXPERT_BATCH": os.environ.get("EBATCH", "1"),
    "K3_FUSE_QKVG": os.environ.get("FUSE", "0"),   # 4-wide fold; only pays with mirrors on
    "K3_SITU_GPU": os.environ.get("SITU", "1"),
    "K3_CTL_FB_GPU": os.environ.get("FBGPU", "1"),
    **({"GOMP_SPINCOUNT": os.environ["SPIN"]} if os.environ.get("SPIN") else {}),
    **({"OMP_WAIT_POLICY": os.environ["WAITP"]} if os.environ.get("WAITP") else {}),
    "K3_CTL_THREADS": os.environ.get("CTLT", "0"),
    "K3_CTL_DYN": os.environ.get("CTLD", "0"),
})
UP    = {1: "10.10.12.1", 2: "192.168.0.159", 3: "10.10.34.1"}
CROSS = {2: "192.168.0.159", 3: "192.168.0.230"}
W1    = {0: (os.environ.get("W1R0") or "/home/norquistdylan/k3data/K3-w1-r0"), 1: "/home/norquistdylan/k3data/K3-w1-r1", 2: "/home/norquistdylan/k3data/K3-w1-r2", 3: "/home/norquistdylan/k3data/K3-w1-r3"}
if RANK in UP:    env["K3_UP_HOST"] = UP[RANK]
if RANK in CROSS: env["K3_CROSS_HOST"] = CROSS[RANK]
env["K3_W1_DIR"] = W1[RANK]
env["K3_W2_SHARD"] = f"{RANK}/4" if (RANK or W1[0].endswith("-r0")) else "0/1"

log = open(LOG, "wb")
cmd = [BIN]
if os.environ.get("NCU") and str(RANK) in os.environ.get("NCU_RANKS", "0,1,2,3").split(","):
    cmd = ["/usr/local/cuda/bin/ncu", "--replay-mode", "kernel", "--target-processes", "all",
           "--kernel-name", os.environ.get("NCU_KERNEL", "regex:k3_dense_i4g_exactW"),
           "--launch-skip", os.environ.get("NCU_SKIP", "20000"),
           "--launch-count", os.environ.get("NCU_COUNT", "12"),
           *(["--metrics", os.environ["NCU_METRICS"]] if os.environ.get("NCU_METRICS")
             else ["--section", "SpeedOfLight", "--section", "Occupancy", "--section", "LaunchStats"]),
           "--csv", "--log-file", f"/tmp/ncu-r{RANK}.csv"] + cmd
elif os.environ.get("NSYS"):
    cmd = ["nsys", "profile", "-o", f"/tmp/k3prof-r{RANK}", "--force-overwrite", "true",
           "--trace=cuda,osrt", "--delay", os.environ.get("NSYS_DELAY", "240"),
           "--duration", os.environ.get("NSYS_DUR", "45"), "--sample=none"] + cmd
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=log, env=env, bufsize=0)
buf = bytearray()

def readline():
    while True:
        i = buf.find(b"\n")
        if i >= 0:
            line = bytes(buf[:i + 1]); del buf[:i + 1]
            log.write(line); log.flush()
            return line
        chunk = p.stdout.read(1)
        if not chunk: raise SystemExit(f"rank{RANK}: child closed stdout")
        buf.extend(chunk)

def readexact(n):
    while len(buf) < n:
        chunk = p.stdout.read(n - len(buf))
        if not chunk: raise SystemExit(f"rank{RANK}: EOF in DATA")
        buf.extend(chunk)
    out = bytes(buf[:n]); del buf[:n]
    log.write(out); log.flush()
    return out

def pump_until_done(rid):
    while True:
        line = readline()
        f = line.split()
        if len(f) >= 3 and f[0] == b"DATA":
            readexact(int(f[2]) + 1)
        elif f and f[0] == b"DONE" and f[1] == rid.encode():
            return line.decode(errors="replace").strip()
        elif f and f[0] == b"ERROR":
            raise SystemExit(f"rank{RANK}: {line!r}")

# wait for READY
t0 = time.time()
while True:
    if b"READY" in readline(): break
print(f"rank{RANK}: ready in {time.time()-t0:.1f}s", flush=True)

def submit(rid):
    hdr = b"SUBMIT %s 0 %d 100 0 1\n" % (rid.encode(), len(WIRE))
    p.stdin.write(hdr + WIRE + b"\n"); p.stdin.flush()
    t = time.time()
    done = pump_until_done(rid)
    print(f"rank{RANK}: {done}  ({time.time()-t:.1f}s wall)", flush=True)

def setknob(spec):
    k, v = spec.split("=", 1)
    p.stdin.write(("SET %s %s\n" % (k, v)).encode()); p.stdin.flush()
    while True:
        line = readline()
        if line.startswith(b"SETOK") or line.startswith(b"ERROR"):
            print(f"rank{RANK}: {line.decode(errors='replace').strip()}", flush=True)
            return line.startswith(b"SETOK") and b"live=1" in line

# ARMS="K3_DENSE_ILP=0,K3_DENSE_ILP=1" runs every arm against ONE loaded engine.
# A restart costs 146 s of load plus ~960 s refilling the expert cache at
# ctx 1963 -- 81% of an arm's wall clock, all of it discarded. Knobs that SET
# reports live=0 for are NOT switchable this way (K3_EXPERT_GB above all: the
# cache is sized at init), and the driver says so rather than silently
# measuring the same configuration twice.
# ONLY VALID FOR BIT-EXACT KNOBS. cold+settle warm the expert cache for
# whichever arm runs first; a knob that perturbs numerics moves router decisions,
# so a later arm then measures a cache warmed for the wrong expert set. That is
# how BFDOT was mis-measured at -18% when it is really a wash. ILP, RTHRESH and
# AR_PERSIST change no arithmetic and are fine here. Anything else: restart per
# arm with the knob set in the environment.
_BITEXACT = {"K3_DENSE_ILP", "K3_DENSE_RTHRESH", "K3_AR_PERSIST", "K3_DENSE_I4W",
             "K3_MATT_BATCH", "K3_CHUNK", "K3_PF_SHAPE", "K3_DENSE_MULTI"}
ARMS   = os.environ.get("ARMS", "")
PASSES = int(os.environ.get("PASSES", "2"))
if ARMS and RANK == 0:
    _bad = {a.split("=")[0] for a in ARMS.split(",")} - _BITEXACT
    if _bad:
        print("rank0: WARNING %s perturbs numerics; interleaved arms share a cache "
              "warmed for whichever ran first -- restart per arm instead" % sorted(_bad),
              flush=True)
if ARMS:
    submit("cold")
    submit("settle")                     # the cache is still filling at high EGB
    for spec in ARMS.split(","):
        ok = setknob(spec)
        tag = spec.replace("=", "").replace("/", "").replace(".", "")
        if not ok and RANK == 0:
            print("rank0: WARNING %s is not live-settable; this arm repeats the previous one" % spec,
                  flush=True)
        for i in range(PASSES): submit("%sp%d" % (tag, i))
else:
    for rid in ("cold", "warm1", "warm2"): submit(rid)

p.stdin.close()
try: p.wait(timeout=int(os.environ.get('EXIT_WAIT','60')))
except subprocess.TimeoutExpired: p.kill()
log.close()
