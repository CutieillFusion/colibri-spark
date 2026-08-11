# Serving Kimi-K3 on vLLM across 4x DGX Spark (GB10)

Working notes. Written 2026-08-10 at the point where the model loads and the
KV cache allocates, and the remaining blocker is a gloo transport error.

Code lives in `vllm_k3_w1/` (in `colibri-spark` git, deployed to `~/k3w1` on
each Spark). It registers out-of-tree via `register_quantization_config`, so
**vLLM itself is unmodified** -- everything is a plugin plus a few targeted
monkey-patches of vLLM internals, each documented at its call site.

---

## Goal

Serve Kimi-K3 (2.8T params / 104B active, 93 layers: 69 KDA + 24 gated MLA,
896 experts, top-16) on four DGX Sparks at B=1, text-only, with an
OpenAI-compatible API. Then measure prefill and decode across context lengths
and profile with ncu.

**Not reached.** No served model, no timings, no profile. Everything up to
and including KV-cache allocation works.

---

## Status

| stage | state |
|---|---|
| vLLM `main` built for sm_121 with K3 kernels | works |
| 1-bit MoE Triton kernels | works, 110 GB/s, validated vs explicit dequant |
| int4-g64/g128 + int8 dense kernels | works, reproduce `kimi_k3.c` exactly |
| Expert loader from `experts.w2` | works, byte-identical to the store |
| TP=2 x PP=2 repack | done, all four stores exact |
| Model load | **works** -- 108.17 GiB/rank |
| KV cache allocation | **works** -- 2,633 tokens |
| Serving | **BLOCKED** -- gloo transport error |

---

## The blocking bug

After the KV cache allocates, the engine dies with:

```
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/pair...]
RuntimeError: Engine core initialization failed
```

Reproducible. **Not a timeout** -- persists with
`VLLM_DISTRIBUTED_TIMEOUT_SECONDS=3600`. Not memory: it happens after the
weights and KV are both allocated, with the load having completed cleanly.

Next step is the full traceback, which was never captured:

```bash
ssh spark1 'docker logs k3-head 2>&1 | grep -B10 -A25 "gloo/transport"'
```

Untested suspicion: gloo is pinned to `enP7s7` (the 1 GbE management LAN) via
`GLOO_SOCKET_IFNAME` because a 4-rank group spans both RoCE islands. A plain
4-rank gloo `all_gather_object` over that interface was verified working early
on, so a specific collective or a peer that stopped responding under load is
more likely than the interface being wrong.

---

## Hardware and why the topology is what it is

Four GB10 Sparks, 121 GiB unified LPDDR5X each (no separate VRAM -- "GPU
memory" is system RAM). Two 200 GbE RoCE islands, spark1+spark2 and
spark3+spark4, joined **only** by a 1 GbE LAN.

NCCL's `netName` is per-communicator and all-or-nothing, so a flat 4-rank
communicator spans both islands and is forced onto the 1 GbE -- including the
intra-island legs that have RoCE sitting idle. Measured, 14 KB all-reduce:

```
flat 4-rank across islands            5.507 ms
2-rank intra-island, forced Socket    2.861 ms
2-rank intra-island, net_name="IB"    0.020 ms     <- 143x
```

At ~186 all-reduces/token that is ~1.0 s/token of pure collective for TP=4.
Hence **TP=2 x PP=2**: TP stays inside an island on RoCE, and only activation
handoffs cross the bridge.

`patch_hier_ar.py` also implements a hierarchical all-reduce (island leg on
IB, cross leg on Socket) for the TP=4 case; `torch` exposes
`ProcessGroupNCCL.Options().config.net_name` in this build, so it needs no
vLLM or NCCL patch. Enable with `K3_HIER_AR=1`. Unused at TP=2.

---

## Memory: the arithmetic that drove everything

Per node at TP=2 x PP=2, measured:

```
dense weights, int4-g128 / int8         ~12 GB
experts, 1-bit, 46 layers x 448        106.4 GB
                                       --------
model loading took                     108.2 GiB
free after weights                       3.6 GB
```

The 1-bit expert store is what makes this possible at all. The checkpoint
ships MXFP4 experts at 17,547,264 B each = 362 GB/node; the 1-bit store is
5,160,960 B/expert = 106.4 GB, the only format that is fully RAM-resident.

Non-expert weights are 114.4 GB in the checkpoint (bf16, exempted by its own
`ignore` list) and must be quantized at load or they do not fit -- our C
engine does the same (`kimi_k3.c:23`).

### What finally made it fit

`--gpu-memory-utilization 0.918` **plus** `--kv-cache-memory` forced.

The critical insight: on unified memory `torch.cuda.mem_get_info` reports
**system-wide** free memory, so each rank's "available KV" is measured against
whatever its node looked like at the moment it profiled. Observed free values
drift 112.33-112.75 GiB between samples, and ranks that profile later see a
fuller system. That produced a 2.18 GiB spread between two ranks of the same
PP stage with identical weights, and made identical configs behave differently
across runs. Forcing an explicit KV size removes the computation entirely.

`--gpu-memory-utilization` does **not** reserve memory. It only sizes the KV
budget and drives a startup precheck. Lowering it does not create headroom;
proven by identical post-fill free memory (3.6 vs 3.7 GB) at 0.910 and 0.918.

---

## Layout of the plugin

| file | role |
|---|---|
| `method.py` | `KimiK3OneBitConfig` (registers `k3_w1`) + MoE method; dispatches per layer type |
| `kernels.py` | 1-bit expert Triton kernels: `w1_gemv` (decode) and `w1_grouped_gemm` (prefill) |
| `dense_method.py` / `dense_kernels.py` | int4-g64/g128 and int8 for everything non-expert |
| `embed_method.py` | int8 vocab embedding and lm_head |
| `loader.py` | `K3W1Store` -- reads the flat `experts.w2` |
| `patch_loader.py` | skips checkpoint MXFP4 experts, fills from the store, phase tracing |
| `patch_latent.py` | routes latent projections through the quant config |
| `patch_shm.py` | skips the same-node shm probe (deadlocks here) |
| `patch_hier_ar.py` | hierarchical TP all-reduce (TP=4 only) |
| `repack_pp.py` | converts the TP=4 store to TP=2 x PP=2 |
| `extract_dense.py` | writes a dense-only checkpoint (114 GB, not 1.5 TB) |
| `run_k3_node.sh` + `k3_{head,worker}_inner.sh` | per-node launcher |
| `bench_serve.py` | prefill/decode timings, ready to run once serving works |

### The 1-bit format

`W[o,i] = a * 2^(scale[o, i//32] - 127) * (bit ? +1 : -1)`

One sign bit per weight, LSB-first, UE8M0 group-32 scales inherited verbatim
from the MXFP4 checkpoint, plus a global amplitude `a = 1.69` that exists
nowhere in the file (our engine reads it from `K3_W1_A`). Slot layout is
`[w1p | w1s | w2p | w2s | w3p | w3s]`, addressed arithmetically -- see
`loader.py` and `c/kimi_k3.c:1455-1480`.

`w13 = [gate; up]` is our `[w1; w3]`: w1 takes tanh*sigmoid, w3 the linear
clip, matching vLLM's `SituAndMul` exactly (beta 4.0, linear_beta 25.0).

### Kernel choice

Two paths because one shape does not cover both regimes. Measured per full MoE
layer on one EP rank of four (224 resident experts, 16-of-896, H=3584 I=3072):

```
T          1      2      4      8     16     64
gemv    0.371  0.617  0.891  1.480  3.019  12.365 ms
group   0.461  0.738  1.083  1.910  3.560   9.711 ms
```

`w1_gemv` needs no block-sorting pass and wins at decode; the grouped `tl.dot`
path wins once tokens-per-expert exceeds ~1. Crossover between 16 and 64.

Block shape mattered more than the algorithm: the first correct config ran at
29 GB/s, and `BLOCK_M` 64->16 plus dropping `tl.dot` for decode took it to
110 GB/s. Our CUDA kernel does 225 GB/s on the same shape, so ~2x remains.

---

## Traps, all of which cost real time

1. **`/tmp` does not survive a reboot.** Lost the 26-minute vLLM build from
   all four nodes this way. Everything now lives on `/home` (`~/k3vllm`,
   `~/k3w1`, `~/k3data`). `K3_WORK` / `K3_PLUGIN` override the paths.
2. **A reboot invalidates the host key**, so *pushes* to a rebooted node fail
   with a bare rsync `255`. Pull from it instead.
3. **spark4 cannot resolve the other hostnames** -- fine as an rsync source,
   not as a destination by name.
4. **`*.egg-info` is root-owned** by the in-container `pip install -e` and
   breaks rsync. Always exclude it.
5. **Ray does not assign ranks in hostname order** (observed rank 2 on spark4,
   rank 3 on spark3). `patch_loader.apply_expert_map_shard` builds
   `expert_map` from the shard the local store actually holds. Without it
   every expert loads from the wrong slot, silently.
6. **`round_robin` expert placement silently downgrades to `linear`** unless
   `num_expert_group > 1`, and K3 has 1. Under linear, rank 0 wants globals
   0..223 while its store holds 0,4,8,... Also silent.
7. **`in_the_same_node_as` deadlocks** (`parallel_state.py`): it wraps a
   create-and-broadcast in `contextlib.suppress(OSError)`, so a failed shm
   create skips the broadcast while every other rank blocks in it.
8. **`custom_all_reduce._init_mnnvl_buffer` hangs** on one rank while others
   log "no MNNVL multicast" and move on. `--disable-custom-all-reduce`.
9. **`ParallelLMHead` extends `VocabParallelEmbedding`, not `LinearBase`**, so
   a quant config is never consulted for it. Same for the embedding, and for
   the latent projections (built with `quant_config=None`).
10. **The head node wedges during load** -- kernel answers ICMP, sshd cannot
    complete a handshake. `system.slice MemoryMin=2G` (see
    `/nas/k3_protect_ssh.sh`) makes nodes recover on their own afterwards.
    Container `--memory` caps are **inert**: CUDA allocations bypass cgroup
    accounting on unified memory.

---

## Running it

```bash
# one time per node, persists across reboots
sudo bash /nas/k3_protect_ssh.sh

# config that gets furthest
E="K3_QUANT_EMBED=0 K3_MLA_BITS=4 K3_HEAD_BITS=4 K3_QUANT_GATE=1 K3_GROUP=128"
ssh spark1 "K3_MAXLEN=2048 K3_UTIL=0.918 K3_KV_BYTES=402653184 K3_MNBT=512 $E \
  bash ~/k3w1/vllm_k3_w1/run_k3_node.sh"
sleep 30
for i in 2 3 4; do ssh spark$i "$E bash ~/k3w1/vllm_k3_w1/run_k3_node.sh"; done
```

Per-rank memory traces land in `~/k3w1/logs/fill-<host>.log`, written with a
flush per line so they survive a SIGKILL. **This instrumentation is what
produced every real finding here** -- vLLM's own per-rank output is forwarded
to the head, and the head is exactly what becomes unreachable.

Untested: `K3_RESIDENT_EXPERTS` / `K3_STREAM_SLOTS` stream a tail of experts
from NVMe instead of holding them resident. Written, never executed -- the
env was not forwarded into the container on the one run that tried it.

---

## What I got wrong

Recorded because the pattern is the lesson. Seven hypotheses for the memory
shortfall, six wrong:

1. container cgroup cap -- inert, CUDA bypasses cgroup accounting
2. per-unit `MemoryMin` -- protects resident pages, not sshd's forked children
3. `system.slice` reservation -- helps recovery, not survival
4. eager expert allocation -- deferral worked all along
5. lowering `--gpu-memory-utilization` -- does not reserve memory
6. int8 embed/lm_head -- no measurable change
7. forced `--kv-cache-memory` -- **this one worked**

Also wrong along the way: that TP=2 x PP=2 would save ~8 GB (it saves ~2.3 --
only *replicated* weights halve, sharded ones are 1/4 either way); that
`--swap-space` existed to reclaim (removed upstream); that driver separation
was possible (it is not, with 4 GPUs all needed).

The single highest-value thing done was the durable per-rank trace, and it was
built far too late -- after roughly ten failed launches rather than before the
third. Measure first.
