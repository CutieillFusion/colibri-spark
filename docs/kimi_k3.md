# Kimi K3 engine (`c/kimi_k3.c`)

A sibling engine for [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3)
(2.8T parameters, 104B active, 93 layers), following the one-engine-per-family
pattern (`colibri.c` = GLM-5.2, `olmoe.c`, `inkling.c`). It shares `st.h`,
`json.h`, `tok.h`, `quant.h` and touches nothing in the GLM engine.

```
make kimi_k3
./kimi_k3 <model_dir> "prompt" --ngen 64
```

`<model_dir>` is either the plain HF snapshot (config.json +
`model-*-of-000096.safetensors`) or a repacked container (below). Text-only:
the vision tower (shards 95–96) is never read.

## Architecture notes

K3 is not a DeepSeek-shaped model; four pieces are new relative to everything
else in this repo:

- **Hybrid attention, NoPE.** 69 KDA (Kimi Delta Attention) layers + 24 gated
  MLA layers (every 4th, plus the final layer). There is no positional
  encoding anywhere — position lives in KDA's convolution and decay. The MLA
  block has DeepSeek dims (q_lora 1536, kv_lora 512, nope 128 + 64 uncached
  extra dims, v 128, 96 heads) and both attention types multiply their output
  by a full-rank sigmoid gate `σ(W_g x)` before `o_proj`.
- **KDA** per head (dim 128, 96 heads): `q,k,v = SiLU(causal-conv4(W x))`,
  q/k L2-normalized (ε=1e-6 inside the sqrt), q scaled by 128^-1/2; per-channel
  decay `gk = -5·σ(exp(A_log[h])·(W_fb W_fa x + dt_bias))`; the delta-rule
  state update `S = (I − βkkᵀ)·Diag(e^gk)·S + βkvᵀ`; output
  `W_o[σ(W_g x) ⊙ RMSNorm_head(Sᵀq)]`. The checkpoint stores `A_log` as
  `[128]`: that is the per-head `[96]` parameter zero-padded (verified —
  entries 96..127 are exactly 0). Decode state: 96×128×128 f32 per layer.
- **AttnRes replaces the residual stream.** Each layer keeps a running
  `prefix_sum`; at layers 0, 12, 24, …, 84 it is snapshotted into a block
  list. Twice per layer (before attention, before the MLP) and once at the
  end, the hidden state is REPLACED by a softmax mix over
  `[snapshots…, prefix_sum]`, scored by `(v · (res_norm.w ⊙ res_proj.w)) /
  rms(v)` and mixing the raw (un-normalized) entries in fp32.
- **Stable LatentMoE.** Sigmoid router `[896, 7168]` + score-correction bias,
  top-16 selected on biased scores, weights are the raw scores renormalized.
  The token is projected 7168→3584, the 16 experts run in that latent space
  (GLU, inter 3072), the aggregate is RMSNorm-ed and projected back
  3584→7168; two fused full-width shared experts (inter 6144) are added.
  Activation is **SiTU-GLU**: `4·tanh(g/4)·σ(g) · 25·tanh(u/25)`.

### Weights

Only the routed experts (`experts.N.w{1,2,3}`) are quantized in the
checkpoint — **MXFP4** (compressed-tensors `mxfp4-pack-quantized`,
quantization-aware trained): e2m1 nibbles packed two per byte (low nibble =
even column, bit 3 = sign) with a ue8m0 power-of-two scale per 32 columns,
`w = v · 2^(scale-127)`. 17.55 MB per expert, 82,432 experts ≈ 1.45 TB.
`quant.h` gained `matmul_mxfp4` (scalar + AVX2) that computes on this layout
directly — expert bytes are **never converted**; QAT weights are exactly the
trained values and any re-encode only adds error.

Everything else is BF16 and is quantized into RAM **at load time** (int8
per-row / int4-g64), or read pre-quantized from a repacked container.

## Streaming design

Experts are streamed straight from the safetensors shards through a per-layer
LRU (`K3_EXPERT_GB`). Measured layout facts the engine exploits: the six
tensors of an expert are stored back-to-back (one expert = one `pread`), and
experts are *not* id-ordered inside a shard, so loads are issued in disk-offset
order. A token's misses are read **in parallel** (OMP over working-set slots)
and by default with **O_DIRECT** (`K3_DIRECT=0` for buffered): the resident
weights leave little page-cache headroom, and flat routing means cached reads
mostly cannot be reused anyway — measured on the 93-layer model this took
expert reads from ~1.8 to ~6.3 GB/s (drive ceiling 7.1) and decode from ~21
to ~9.4 s/token. The LRU slot floor is 1 (experts are consumed one at a time),
so `K3_EXPERT_GB` is honored even at tiny budgets. On top of that, `K3_PIPE`
(default on) runs the reads on loader threads so expert j's matmuls overlap
expert j+1's pread, `K3_IDOT` (default on) computes the expert matmuls with
per-32-group int8-quantized activations via integer dots (the e2m1 doubled
values are exact int8 — same trick as `dot_i4i8`), and `K3_TOPP` optionally
drops the low-weight tail of the top-16 (renormalized; quality-gate any
setting with a `K3_LOGITS` A/B first). Note that K3's router was trained with Quantile Balancing (deliberately
flat expert usage), so LRU hit rates are structurally lower than on models
with skewed routing — expect the expert tier to be bandwidth-bound.

## Repacked container (`tools/k3_repack.py`)

```
python3 tools/k3_repack.py <hf_src> <dst> --bits 8 [--mla-bits 8] [--head-bits 8]
```

One streaming read of the source (so a disk-to-disk copy can *be* the
conversion): experts pass through byte-identical (id-ordered, single-pread
layout), the big BF16 matrices are quantized with exactly the engine's
load-time algorithm and stored as `U8` + `<name>.qs` f32 scales, the small /
sensitive tensors (norms, router, conv taps, `dt_bias`, `A_log`, `f_a/f_b`,
`b_proj`, embeddings) pass through, vision is dropped. Output is spec-valid
safetensors (`model-XXXXX-of-000094.safetensors`) plus a regenerated index;
interrupted runs resume per shard. `--verify-full` re-reads every expert byte
and compares against the source.

The engine auto-detects container tensors (dtype U8 + `.qs` sidecar) and skips
load-time quantization. Setting `K3_BITS=4` **explicitly** on an int8 container
downcasts the int8 matrices to int4-g64 at load (~35 vs ~57 GiB resident on
the 93-layer model — fits next to a desktop session; the int8 grid is 16x
finer than int4, so the double-quant noise ~ direct int4). Unset `K3_BITS`
keeps the container's own bits. Startup: measured on two layers, init drops **30.4 s → 0.6 s**;
on the full model this removes a 10–15 minute quantization pass. Quantized
values are bit-identical to the load-time path (verified by hidden-state
trace), so the two formats are numerically interchangeable.

Sizes: source 1.56 TB → ≈1.50 TB (`--bits 8`) / ≈1.48 TB (`--bits 4`). The
experts (93 % of bytes) are already at 4.25 bits/weight and cannot shrink
losslessly; sub-4-bit expert re-encodes (fmt=5/6) would be double quantization
of QAT weights and are deliberately not offered here.

## Tokenizer

The HF repo ships only a raw tiktoken vocab (`tiktoken.model`).
`tools/k3_tokenizer.py` writes a `tokenizer.json` for it, and `tok.h` gained:

- a **kimi pre-tokenizer family** (sniffed via `\p{Han}` in the Split regex):
  the o200k case-aware rules plus a leading Han-run rule, Han excluded from
  the letter classes, and no `/` tail in the punctuation rule;
- a **rank-BPE mode**, used when `model.merges` is empty: merge the adjacent
  pair whose concatenation has the lowest vocab id — which is tiktoken's own
  algorithm, so encoding is exact by construction. (Merge-list recovery from
  ranks is provably lossy: the standard reconstruction mis-merges e.g.
  "newlines"; that is why the generator emits no merges.)

`tools/k3_tokenizer.py --ctest tests/test_tok_kimi` cross-checks the C
tokenizer against tiktoken; it matches on all 18 corpus cases (Chinese,
kanji-vs-kana, Korean, CRLF, contractions, emoji, mixed-script boundaries).

## Validation

`tools/k3_ref.py` is an independent numpy implementation of the full layer
stack (KDA recurrence, gated MLA, AttnRes, LatentMoE with MXFP4 dequant). With
injected inputs (`K3_X0`, bypassing the embedding) and f32 weights
(`K3_BITS=32`), the C engine matches it on real checkpoint weights across all
four layer types to **rel-L2 ≤ 2.2e-6** (float32 noise).

One property worth knowing: the AttnRes mix logits sit close together
(margins ~0.02), so the mix leverages *weight* noise strongly even though it
does not amplify *input* perturbations (a 1e-4 input perturbation stays 1e-4
through the stack). Load-time int8 produces a hidden-state drift growing to
~14 % over four layers on synthetic inputs — per-tensor quantization SNR is a
uniform ~1 %, so this is architectural sensitivity, not an outlier problem.
Judge quantization choices on real-text logits, not synthetic-vector norms.

## Environment variables

| var | default | meaning |
|---|---|---|
| `K3_BITS` | 4 | load-time bits for KDA/latent/shared/dense mats (4, 8, 32=f32) |
| `K3_MLA_BITS` | 8 | load-time bits for MLA projections |
| `K3_HEAD_BITS` | 8 | load-time bits for lm_head |
| `K3_EXPERT_GB` | 8 | routed-expert LRU budget |
| `K3_DIRECT` | 1 | O_DIRECT expert reads (0 = buffered + WILLNEED) |
| `K3_IDOT` | 1 | int8-activation expert matmuls (0 = exact-float kernel) |
| `K3_PIPE` | 1 | overlap expert loads with compute (loader threads) |
| `K3_LOAD_THREADS` | 4 | loader threads for `K3_PIPE` |
| `K3_TOPP` | 0 | keep routed experts to cumulative weight p (0 = off) |
| `K3_DENSE_GPU` | 1 | zero-copy CUDA dense GEMV during decode (0 = device-mirror path) |
| `K3_DENSE_EXACT` | 1 | stock-order bit-exact reduction (0 = legacy warp reduction) |
| `K3_DENSE_DEV_GB` | 0 | device-mirror budget for exact dense weights (18 on the four-Spark config) |
| `K3_KDA_OVERLAP` | 1 with CUDA | run strict-f32 KDA control projections on a CPU worker underneath the independent GPU q/k/v/g projections (0 = synchronous fused CPU path) |
| `K3_OMP_THREADS` | 10 | OpenMP workers used by `k3_launch.sh`; on GB10 this lets B=1 work occupy the 10 X925 cores while CUDA/network helpers can spill onto the 10 A725 cores |
| `K3_NET_RD2` | 1 at 4 nodes | two-phase recursive-doubling collective for the paired Spark topology (0 restores the hierarchical tree) |
| `K3_CHUNK` | 32 | prefill chunk size (1 = token-at-a-time; forced 1 under `K3_TRACE`) |
| `K3_THINK` | 1 | chat mode: open the structural think channel (0 = response-only) |
| `K3_DIRS` | — | extra shard directories (multi-drive split, no duplication) |
| `K3_MAXT` | prompt+ngen | context capacity |
| `K3_LAYERS` | all | truncate the stack (validation) |
| `K3_TRACE` | — | dump f32 hidden state after every layer (validation) |
| `K3_X0` | — | inject input rows, bypass embedding (validation) |
| `COLI_TEMP` | 0 | 0 = greedy, else softmax temperature |

### Four-Spark B=1 decode

For the 1-bit per-rank expert stores, `K3_EGB=80` holds the observed working
set for the benchmark prompt. With four nodes, RD2, the default 10 OpenMP
workers, and a warmed persistent server, a 100-token B=1 chat decode measured
**3.510 and 3.505 tok/s** on consecutive passes (99.5% expert hits), with a
2.072 tok/s cold-cache pass. Generated bytes matched the previous exact kernel
on every rank and pass. The gain comes from folding the exact GEMVs' 256
logical reduction lanes onto 128 physical CUDA threads: every accumulator and
reduction-tree edge is preserved, while real K3 shapes rise from roughly
85–103 to 139–155 GB/s for int4 and 130–144 to 153–167 GB/s for int8. This is a
workload working-set result, not a claim that 80 GB can hold all 224 experts per
layer; arbitrary prompts may still miss and stream from NVMe.

The int8 fold lands where its shapes live: `head` 0.799 -> 0.703, `mproj`
0.991 -> 0.834, `mout` 1.603 -> 1.519 seconds per 100 tokens.

### The routed-expert kernel was losing 2.2x to shared-memory bank conflicts

`warp_row_dot_w1` gives lane L the groups L, L+32, ... and reads the staged
activation at `shx[32L + 1024k + j]`. The stride of 32 floats between lanes puts
**every lane of the warp on bank j**: a 32-way conflict on each of the 32 loads
per group, 
paid on every group of every row of every expert.

It was invisible from the outside because nothing looked saturated. A probe
(`tests/bench_k3_w1` plus a throwaway kernel) settled it by holding the memory
pattern fixed and varying the work:

| variant | ms | GB/s (weight bytes) |
|---|---:|---:|
| production inner loop | 0.0515 | 66.8 |
| identical loads, arithmetic removed | 0.0515 | 66.8 |
| weights + scales only, no shared reads | 0.0102 | 336.1 |
| production loop, staged x padded to 33 | 0.0205 | 167.9 |

The first two being *equal* is the finding: the ALU was free, and shared memory
was the entire cost. Staging x with a 33-float stride per 32-value group sends
lane L to bank (L + j) % 32 -- all distinct -- for one extra float per group of
shared memory and no change to a single value or summation order.

Measured on the real expert shape: 0.089 -> 0.041 ms hot (57.8 -> 126.5 GB/s),
0.092 -> 0.057 ms cold-cycled, and `expert` 4.219 -> 2.937 s/100 tokens.

A second step takes the stride from 33 to **36**. 33 fixes the conflict for
scalar loads but is not 16-byte aligned, so it forbids vector loads. 36 floats
= 144 B is aligned AND still conflict-free for 128-bit accesses, because the
hardware splits a warp's float4 loads into four phases of eight lanes and eight
lanes x four banks covers exactly the 32 banks. The 32 scalar shared loads per
group become 8 float4 loads, in the same order:

| gate_up variant | ms | GB/s |
|---|---:|---:|
| stride 32 (original) | 0.0513 | 67.1 |
| stride 33, scalar | 0.0205 | 168.0 |
| **stride 36, float4** | **0.0153** | **224.5** |
| no activation reads at all (floor) | 0.0102 | 336.0 |

Real expert shape 0.041 -> 0.036 ms hot, 0.057 -> 0.055 cold, `expert`
2.937 -> 2.782 s/100 tokens. Note the cold gain is far smaller than the hot one:
cold is bound by the weight stream from DRAM, so end to end this is +0.8%
(3.494/3.488 -> 3.523/3.511), inside run-to-run noise even though the PROF2
term it targets moves cleanly.

Two things this ruled out along the way: zero-copy is not a factor (weights in
`cudaMalloc` memory measured 58.6 GB/s against 57.8 zero-copy, and cycling 48
distinct slots gave 56.3, so neither the mapped-host path nor cache residency
mattered), and the fixed per-call cost of two launches plus two `memcpyAsync`
plus one stream sync is only 11 us of the 89 -- batching whole layers cannot be
worth more than ~12%. Amortising the staged x over 2/4/8/16 rows per warp was
also tried and lost monotonically: x is 14 KB and served from L2, so restaging
it is nearly free while the larger row tile costs occupancy.

`warp_row_dot_w2` has the identical indexing and therefore the identical
conflict. It is untouched only because the 2-bit store is not what this
deployment runs, so the fix could not be validated end to end.

Further folding is not the way forward. Folds of 4 and 8, tiling 2 and 4 rows
per CTA to amortise the x re-read, and pairing lanes {2p, 2p+1} so one byte
load feeds both nibbles were each built and measured bit-exact, and each was
neutral or slower. The reason is that these GEMVs already run at 64–71% of this
part's **235 GB/s** achievable read bandwidth (measured; 273 GB/s theoretical —
note `cudaDevAttrMemoryClockRate` reports LPDDR5X's 8533 MT/s data rate, so
doubling it overstates peak by 2x). `lat_up` already reaches 94%.

### The lm_head was replicated; greedy decode only needs the argmax

`lm_head` is [vocab 163840, hidden 7168] and every rank computed ALL of it --
1.17 GB and, after the 4-wide fold and mirrors, still 5.1 ms/token done four
times over. Greedy sampling needs only the argmax, so each rank takes a
contiguous row slice and the ranks agree on a winner with **one collective per
token**, not per layer.

Exactness has two parts. Row o's dot does not depend on any other row, so a
slice produces the same values as the replicated computation. And `sample_tok`'s
greedy rule is `if(lo[i] > lo[b])` -- strict, so the FIRST index wins ties; a
rank's local scan uses the same strict compare, and scanning ranks in ascending
order preserves it globally because the slices are ascending and contiguous.
The reduction is a zero-filled sum used as a gather: each rank writes only its
own (value, index) slot, so the sum IS the gather, and 163840 is exact in f32.

`temp > 0` needs the whole vector for softmax and top-p, and `K3_LOGITS` and the
trace dump need it too, so those keep the full head; the caller sets
`head_greedy` from the request before `forward`.

The slice view is cached in the Model. Rebuilding it per token would reset its
registration and device mirror every time, the same trap the KDA `sh_ready`
views avoid.

Measured: `head` **0.508 -> 0.150 s/100 tokens**, collectives 18315 -> 18414 per
100 tokens (one more per token, as expected), mirror footprint 16.95 -> 16.06 GB
because only the slice needs mirroring. End to end **3.797/3.850 -> 3.914/3.906
tok/s**, output byte identical.

### The mirror budget defaulted to zero, and that cost 12%

`K3_DENSE_DEV_GB` gates whether dense weights get a `cudaMalloc` copy (2 MB
pages) instead of being read through the host-registered zero-copy mapping
(SMMU 4 KB pages). The per-tensor cap had already been raised from 64 MB to
2 GB, but **the budget itself defaulted to 0.0**, so no tensor was ever
mirrored unless a human set the variable. Measured, same session, all four
combinations:

| | I4W=2 | I4W=4 |
|---|---|---|
| no mirrors | 3.856 / 3.859 | 3.862 / 3.862 |
| 40 GB budget | 4.019 / 4.020 | **4.284 / 4.276** |

Two things worth reading off that table. The mirrors are worth ~11%. And the
4-wide fold is worth +6.5% **but only with mirrors on** -- at zero-copy it is
worth nothing (3.856 vs 3.862), which is exactly what the comment above the
cap predicted and why the two had to be tested together rather than in turn.

So the answer to "is the budget or the cap binding" is *neither*: the report
line says `0.00 GB skipped over the cap`, and the budget was never consulted
because it was zero. The default is now auto-sized from `cudaMemGetInfo`. A
fifth of free was still too tight -- it placed 11.45 GB, skipped 4.16 GB and
reached only 4.210/4.206 -- so it takes a third, capped at 24 GB, which places
the entire 16.06 GB eligible set and reaches **4.327/4.306**.

### Two harness bugs that made A/Bs compare a config against itself

Both were silent, and both invalidated measurements that had already been
reported:

1. `run_e2e.sh` forwarded only `PORT` to the four nodes. Any variable set on
   the driving shell -- `DEVGB`, `I4W` -- stayed there, so both arms of an A/B
   ran the driver defaults and came back identical. Three A/Bs died this way
   before the `dense mirrors: 0.00 GB placed` line exposed it while the shell
   said 40 GB.
2. The speed harness (`k3drive.py`) defaulted `DEVGB` to 0 while the quality
   harness (`k3gen.py`) defaulted it to 18. Speed and correctness had therefore
   never been measured on the same configuration.

The rule this leaves: an A/B is only valid if the run's own log echoes the
setting under test. `run_e2e.sh` now prints what it forwards, and the mirror
report line is checked on every mirror measurement.

### Teacher-forced logits do not exercise the decode path

`d1657cc` (rotating conv window, stack score buffer) changes free-running
output: from `3d86a11` onward the six-prompt gate goes 1/6 against its
reference, and bisection puts the change exactly there -- `5047fe4` (topk) is
6/6, and the commits between are documentation only. But the teacher-forced
logit capture over a 300-token prompt is **bit-identical** between the two
builds, same sha256 over all 40 MB.

Both facts are real. `--ngen 0` runs prefill only, and the rotation lives in
the S==1 decode path, so the strongest instrument in the harness is blind to
it by construction. It is not a state leak either: prompt 1 generated alone in
a fresh engine hashes identically to prompt 1 generated second, so the engine
is deterministic and position-independent -- the difference is per-token and
numerical.

The mechanism is FP contraction, not ordering. There is no `-ffast-math`, so
GCC may not reorder the sum, but its default **is** `-ffp-contract=fast`, and
`acc += cd[j]*wd[(hh+1+j)&(K-1)]` does not contract to FMA the same way the
old contiguous `acc += cd[j]*wd[j]` did. Same operands, same order, different
rounding, and top-16-of-896 routing turns that into a different token a few
hundred steps later.

What this changes going forward: free-running generation is the only gate that
covers decode, teacher-forced logits are a prefill-only instrument, and any
future exactness claim has to name which of the two it was checked with.

### A per-layer collective costs 1.18 ms, and that governs what is worth doing

The shared experts and the latent projections are loaded with `w_load`, not
`w_load_rows`, so unlike the attention projections they are **replicated on
every rank**. All four ranks log `shared=4.27` and `latent=1.59` s/100 -- the
signature of four machines computing the same answer. That is 58.6 ms/token of
4x redundant work on a 231 ms token, and sharding it looked like the single
biggest remaining lever.

It was implemented exactly: row-shard gate/up, run SiTU on the local slice
before gathering (so three quarters of the `situf_` calls disappear too), then
gather with an allreduce over disjoint slices, which is bit-exact because every
element is contributed by one rank and summed against zeros.

The compute did exactly what it should. `shared` fell to ~24 ms/token of real
work. But `netmoe` went from 0.947 to **10.991 s/100**, and end-to-end fell from
4.327/4.306 to 2.919/3.370.

| | compute saved | collective added |
|---|---|---|
| per layer | ~0.2 ms | **~1.18 ms** |
| per token (93 layers) | ~19 ms | ~110 ms |

So the replication is not an oversight -- it is the cheaper side of a trade the
fabric dictates. **A blocking collective on this topology costs about 1.18 ms,
so any change that adds one per layer has to save more than 1.18 ms per layer
to break even.** Sharding the shared experts offers 0.2. Nothing else in the
per-layer budget is anywhere near 1.18 ms either, which rules out the whole
family of "shard the replicated thing" optimizations below the lm_head level --
lm_head works precisely because it is once per token, not once per layer.

### Pinned staging for the activation vectors: no gain

nsys over a 30 s steady-state window shows 65,419 host-to-device and 65,419
device-to-host transfers averaging 28 KB, doing 208 ms of actual DMA between
them while the memcpy API calls consume 12 seconds. That is a 98% overhead
ratio and the classic pageable-memory signature -- the weights are registered
but the activations are `falloc`'d scratch that changes address every call.

Staging both directions through `cudaHostAlloc` buffers measured 4.304/4.314
against 4.327/4.306: a wash. On GB10 the GPU reads the same physical LPDDR5X
the CPU does, so pinning does not buy the DMA path it would on a discrete card,
and with the GPU already idle ~76% the copy time was hidden anyway. Reverted.
The lesson is that the 12 seconds of API time is not 12 seconds of critical
path, and `cuda_api_sum` alone cannot tell you which.

### The chip is power-capped, which changes what "optimization" means

`nvidia-smi -q -d PERFORMANCE` reports **SW Power Cap: Active** with 61.7 hours
accumulated, while HW thermal and HW power-brake slowdown are both zero and the
CPU governor is already `performance` at its 2.808 GHz ceiling. GB10 is a
superchip: the CPU and the GPU draw from **one power budget**. So moving work
from one to the other does not reduce the energy per token, it only relocates
it, and under a binding cap the total throughput barely moves.

That theory explains the whole result set, retrospectively:

| change | kind | outcome |
|---|---|---|
| topk O(K^2 E) -> O(KE) | eliminates work | 5.1x on the term |
| conv memmove -> rotating window | eliminates work | 5.5x on the term |
| MLA absorb -> GPU | relocates work | +0.4%, near noise |
| KDA control part 1 -> GPU | relocates work | rejected, 9.8 GB/s |
| KDA control part 2 -> GPU | relocates work | dead wash, see below |

Every accepted win removed computation. Every relocation underdelivered. The
rule to carry forward is that under a power cap you go faster by doing **less
total work**, not by moving work to a quieter unit.

### Control part 2 on the GPU: correct, exact, and worth nothing

Part 2 of `kda_control_b1` looked like the ideal migration candidate -- 3072
output rows against part 1's 152, and no transcendental, so it reproduces the
CPU order exactly. Implemented as `k3_ctl_fb` on its own stream (the control
runs on its own pthread and must not share `g_stream`), with `f_b` transposed
to `[hd][pn]` so a warp reads consecutive floats instead of striding 512 B.

Interleaved A/B, same session, alternating to cancel drift:

| | warm1 | warm2 |
|---|---|---|
| part 2 on GPU | 3.877 | 3.888 |
| part 2 on CPU | 3.877 | 3.883 |

A dead wash, so it was reverted. The likely mechanism is that the control
thread's kernel now queues behind the main thread's dense GEMVs, so the control
finishes no earlier than it did on the CPU -- and under the power cap above,
there was no throughput to win by relocating it anyway.

**A measurement trap this exposed.** The first attempt split the one
`omp parallel` region into two so part 2 could sit outside it. That alone cost
**-11%** (4.31 -> 3.85) through per-layer team construction, and because the
CPU fallback path was inside the same restructure, *both* arms of the A/B
regressed together and looked like a GPU problem. Deciding `gpu` before the
region and keeping the CPU branch inside it restored parity.

### Absolute numbers drift between sessions; only interleaved A/B is valid

Identical code at `fb898f3` measured **4.313/4.308** one day and
**3.873/3.869** the next, with no change to the binary, no other load, and
temperatures at 61-68 C. That is a ~10% session-to-session drift, larger than
almost every optimization measured in this log. Consequently a number from one
session may not be compared with a number from another. Every A/B from here on
alternates arms inside a single session, and reports both warm runs.

### The only CPU work that can currently move to the GPU

`mla_forward`'s absorb GEMVs -- `w_addrow` building `qabs`, and `w_rowdot`
producing the head output -- are pure MAC loops with **no transcendentals**, so
unlike everything else on the CPU they can move to the GPU bit-exactly. The CPU
order is reproducible with one thread per output element:

* `w_addrow` (fmt 4): `acc[i] += (scl[r][g] * coef) * (nibble - 8)`, outer
  accumulation over rows in increasing order.
* `w_rowdot` (fmt 4): per group `ga = sum x[i]*(nibble-8)` in pairs, then
  `a += ga * scl[g]`.

Two things make this work where the control projections failed. Parallelism:
batching all heads gives `mhn * kvl` = 12288 threads for `qabs`, against the
control's 152 rows which could only fill five warps. And round trips: per-head
calls would be 576 per token and eat the gain, so all heads go in one launch --
two per layer.

Measured `matt` 0.896 -> 0.849 s/100 tokens, end to end 4.299/4.285 ->
4.313/4.308, output byte-identical. The gain is much smaller than the
arithmetic suggests because 48 launches per token cost ~0.5 ms, which offsets
most of what the absorb saved.

**Why nothing else can follow it.** Every other CPU term contains a
transcendental -- `expf` in SiLU, SiTU, the softmaxes and the KDA decay gate --
and CUDA's `expf`/`tanhf` differ from glibc's in the last ulp. On this model
that is not a small difference: the fused SiTU kernel is correct to 1.46e-11 in
isolation and 5.96e-07 in-engine, and still moves **27% of expert selections**,
giving PCC 0.873 and 54.7% top-1 agreement. Top-16-of-896 routing amplifies a
ulp perturbation into a different computation over 92 layers. So under a
PCC >= 0.9999 bar, anything upstream of the router is effectively pinned to the
CPU by arithmetic, not by engineering.

### Two CPU terms were algorithmic, not numerical

Both found by reading the code behind PROF2 terms rather than by profiling --
neither has a kernel, a collective or a bandwidth story, so nsys could not point
at either.

**The depthwise conv window was memmoved every token.** `kda_forward` kept a
K=4 window per channel oldest-first and shifted it down by one each token:
K-1 float moves per channel, three tensors x pn channels x 69 layers. K is a
power of two, so a rotating write head indexes the same values in the same
order with no moves at all. `kconv` **1.103 -> 0.200 s/100 tokens (5.5x)**.

That is far more than the moves themselves cost. The shift was a
read-modify-write over the window (147 KB per layer, read AND written) and it
serialised the loop body, which blocked vectorisation of the tap dot that
follows. Removing it fixed all three.

**`matt` malloc'd its score buffer per head per token** -- 576 allocations per
token for `nt <= 512` floats. A stack buffer removes them.

Together with the top-16 selection fix these took decode from 3.913/3.897 to
**4.084/4.079 tok/s**, all byte-identical.

The rule that predicted which of these would work: remove computation WITHOUT
increasing the bytes touched. The KV-decode hoist broke it (traded 2-byte reads
for 4-byte reads) and lost; these three respect it and won.

### Routed-expert selection was O(K^2 * E)

`moe_forward` picked the top-16 of 896 experts by running K passes over all E
experts, and inside each pass walking the already-chosen list to test
membership, while recomputing `st[e] + rbias[e]` every time. That is
~129K inner operations per (layer, token) where ~14K suffices, and at 92 sparse
layers it cost **0.436 s/100 tokens**.

A `used[]` flag plus one precomputed biased score gives the identical sequence
-- same strict `>`, so the lowest index still wins ties, and the weight is still
the raw sigmoid score -- for about 9x less work. Bit-exact by construction: only
the membership test and a common subexpression changed, not a single
floating-point operation or its order.

`topk` **0.436 -> 0.085 s/100 tokens** (5.1x), `moe` 11.851 -> 11.547, and end
to end 3.913/3.897 -> **3.957/3.968 tok/s**. Output byte-identical.

Worth noting how this was missed for so long: it is pure CPU control flow with
no kernel, no collective and no memory-bandwidth story, so none of the
profiling that found the bank conflict or the SMMU mapping would ever have
pointed at it. It showed up only from reading the code of the last unexamined
PROF2 term.

### Both major GEMV paths are now at their memory walls

Probes that hold the memory pattern fixed and vary the work, run COLD (cycling
48 distinct weight banks so nothing stays in L2 -- the hot form of this probe
was misleading twice):

| 1-bit expert gate_up variant | hot | cold |
|---|---:|---:|
| stride 32 (original, bank-conflicted) | 66.8 GB/s | 67.2 |
| stride 33, scalar | 167.9 | 166.3 |
| **stride 36, float4 (current)** | 223.9 | **201.9** |
| weights + scales only, no activation reads (floor) | 336.2 | **191.9** |

Cold, the production kernel is at the floor: there is nothing left to win in it.
The hot column had suggested a 2x gap that does not exist once the working set
streams.

The dense path lands the same way. With mirrors and the 4-wide fold, KDA
q/k/v/g measures 246.7 GB/s against ~235 GB/s achievable, so the big shapes are
back at the DRAM wall; that is also why the 8-wide fold, which halves the loads
again and is bit-exact, measured slower (shared gate 377 -> 296 GB/s) -- 32
threads is one warp and 24 blocks/SM reaches only 768 threads.

Device mirrors do NOT generalise to the expert slots: with the fixed kernel they
measure 145.7 GB/s zero-copy against 148.3 device, i.e. no difference. The
SMMU page effect is specific to the dense weights.

### The 4-wide fold, and why it only pays once the mapping is fixed

Exactness forces a thread to own logical lanes that the stock tree combines
LAST. It combines lane bit 7 first and bits 1 then 0 last, so a thread may own
lanes 4p..4p+3 with four accumulators merged at the end as (V0+V2)+(V1+V3),
while each chain's own reduction over the 64 threads (n=32..1) replays lane bits
7..2 in the stock order. Those four lanes are four CONSECUTIVE elements, so the
pair costs one float4, two adjacent weight bytes and -- since gs is a multiple
of 4 -- a single group scale: four loads per four elements instead of twelve.

On zero-copy weights this is worth **nothing** (0.332 vs 0.333 ms), which is the
same answer the 2-wide pair fold and the shared-memory scale staging gave, and
for the same reason: that path is bound by the weight stream from DRAM, so
instruction count is invisible.

On device mirrors it is worth a lot:

| shape | 2-wide | 4-wide |
|---|---:|---:|
| KDA q/k/v/g 12288x7168 | 182.7 GB/s | **242.6** |
| KDA o_proj 7168x12288 | 173.5 | **238.9** |
| shared gate 6144x7168 | 255.1 | **403.1** |
| lat_up 7168x3584 | 238.0 | **374.1** |

Fixing the page mapping removes the DRAM wall, and the load count immediately
becomes the limit. End to end with `K3_DENSE_DEV_GB=16`: **3.780/3.778 against
3.605/3.609**, with `shared` 5.117 -> 4.312, `latent` 1.845 -> 1.596 and `moe`
12.882 -> 11.946 s/100 tokens.

The lesson generalises: three separate load-reduction ideas measured neutral
while the memory path was the bottleneck, and the one that survived only did so
after the bottleneck moved. Re-measure a rejected optimisation when you change
what limits it.

### Zero-copy is not free for the dense weights

The dense path reads its weights zero-copy so the bytes stay available to the
expert cache, and on an integrated part that looks free -- it is the same DRAM.
It is not. Host-registered pages reach the GPU through the SMMU at 4 KB
granularity; `cudaMalloc`'d memory uses large pages. Same kernel, same
`K3_DENSE_EXACT=1`, `rel=0.0e+00` either way:

| shape | zero-copy | device-resident |
|---|---:|---:|
| shared gate 6144x7168 int4 | 153.5 GB/s | **214.9** |
| MLA q_b 18432x1536 int8 | 145.1 | **239.2** |
| KDA q/k/v/g 12288x7168 int4 | 150.9 | 180.6 |
| KDA o_proj 7168x12288 int4 | 153.5 | 172.5 |
| lat_up 7168x3584 int4 | 201.2 | 212.2 |
| lm_head 163840x7168 int8 | 166.4 | 153.1 (2-wide) / **234.5** (4-wide) |

The engine could not exploit this because `w_matmul` offered only two
combinations: the fast exact kernel with zero-copy, or a device mirror with the
stock `quant_matmul` (59 GB/s). `K3_DENSE_DEV_GB` adds the missing one -- the
exact kernel reading a device mirror -- under a byte budget.

The per-tensor cap started at 64 MB because lm_head measured slower mirrored.
That turned out to be an artefact of the 2-wide fold: with the 4-wide kernel the
same tensor goes 166.7 GB/s zero-copy -> 234.5 mirrored, so the large-page
mapping pays even for a 1.17 GB stream that cannot stay cache-resident. The cap
is now 2 GB and exists only to stop one tensor eating the whole budget; at
18 GB the engine reports `16.95 GB placed, 0.00 skipped` and `head` falls from
0.708 to 0.508 s/100 tokens.

| budget | warm tok/s | `moe` s/100 |
|---:|---:|---:|
| 0 | 3.525 / 3.524 | 13.225 |
| 12 GB | 3.574 / 3.591 | 12.791 |
| 28 GB | 3.596 / 3.604 | 12.435 |
| 16 GB | 3.605 / 3.609 | — |
| **18 GB + 4-wide + 2 GB cap** | **3.790 / 3.847** | 11.720 |

Expert hit rate stayed 99.5% at every setting. The budget stops binding at
**15.36 GB**: the engine reports `15.36 GB placed, 0.00 GB skipped for budget,
1.54 GB skipped over the 64 MB cap`, so 16, 28 and 40 GB all mirror exactly the
same tensors and only 16 is worth reserving. The 1.54 GB refused by the cap is
lm_head and friends, which measured SLOWER mirrored -- the cap is doing its job.

Mirroring costs RAM the expert cache would otherwise hold, so it stays opt-in;
16 GB is the tuned value here.

### Decode is not GPU-bound

Measured with `nsys profile --trace=cuda,osrt` on rank 0 over a 26.5 s decode
slice (78,473 kernels), after the four-Spark B1 config reached 3.505 tok/s:

| | |
|---|---:|
| GPU busy in kernels | 6.01 s -- **22.7%** |
| GPU idle between kernels | 20.46 s -- **77.3%** |
| median inter-kernel gap | 21 us |
| gaps > 200 us | 7,554 of them, 17.98 s = **67.9% of the window** |

Kernel work is roughly 92 ms of a ~285 ms token. Further GEMV tuning therefore
has little left to give; the time is in the gaps, which are collectives and CPU
phases. Two things that look alarming in a raw kernel summary are not:
`quant_matmul` shows as 48% of GPU time, but splitting by duration puts 6.83 s
of its 7.61 s in calls >= 400 us, which are **prefill** (`coli_k3_dense`
declines S != 1); decode-sized calls total only 0.78 s. And `cudaMemcpyAsync`
dominates the CUDA-runtime table purely by count (108k calls), not by cost.

### The KDA control cannot move to the GPU: only 152 independent rows

The exact CPU order was solved. `kda_control_b1`'s dot is
`float v=0; for(i) v+=x[i]*w[i];`, which GCC at `-O3 -march=native` turns into a
vector `fmul` plus **scalar** `fadd`s -- products rounded before accumulation,
and no `fmla`. Linking the real CPU object against candidate CUDA orderings over
200 random 7168-length trials:

| GPU ordering | bit-exact |
|---|---|
| `__fmul_rn` then `__fadd_rn`, sequential | **200/200** |
| 4-wide products added back in lane order | **200/200** |
| `fmaf` (FFMA) | 11/200 |
| plain `v += x[i]*w[i]` (nvcc contracts it) | 11/200 |

That 11/200 is almost certainly why an earlier operation-ordered CUDA oracle
looked exact on sampled shapes and still changed full-model routing. Anyone
retrying this must use the explicit intrinsics.

It still does not work, for a reason that has nothing to do with arithmetic.
Exactness forces one thread per output row, and part 1 has only `hd + hn` = 152
rows. Five warps cannot keep enough loads in flight to stream 4.36 MB: measured
**0.443 ms/layer at 9.8 GB/s** cold, against 235 GB/s achievable. Unrolling to
eight loads deep changed nothing (0.436 ms, 10.0 GB/s) because the limit is the
bandwidth-delay product -- 5 warps x 8 loads x 128 B is ~5 KB in flight where
saturating 235 GB/s at DRAM latency needs ~117 KB. Ten CPU cores, with
out-of-order execution and hardware prefetch, simply do better on this shape.

End to end the full path measured **2.751/2.743 against 3.485/3.480**. Reverted.

Note the hot/cold gap once more: the same kernel timed 0.170 ms/layer reusing a
single buffer and 0.443 ms/layer cycling 69 of them, which is what the engine
does.

To make this path viable you would need materially more independent rows --
`f_a` is [128, hidden] and is NOT sharded, so every rank recomputes the same
`t1`. Sharding it would add a collective per KDA layer, which at ~0.82 ms is far
more than the whole control costs.

### Exposed collective time, measured per site

PROF2 reports `netkda`, `netmla` and `netmoe` -- the collective time actually on
the critical path, as opposed to the inclusive `net` timer. Warm four-Spark:

| site | s/100 tokens | ms/token |
|---|---:|---:|
| `netkda` (KDA attention all-reduce) | 3.315 | 33.2 |
| `netmla` (MLA all-reduce) | 1.045 | 10.5 |
| `netmoe` (latent wait, the rest hides under shared experts) | 0.838 | 8.4 |
| **exposed total** | **5.198** | **52.0** |

`net` itself is 8.291 s/100, so about 31 ms/token of collective genuinely does
hide. `kout` is 4.016 of which `netkda` is 3.315, i.e. ~83% collective and only
~7 ms/token of actual o_proj.

**This bounds what the network is worth.** At 283 ms/token, deleting every
exposed collective leaves 231 ms/token = **4.33 tok/s**. Uniform fabric alone
does not reach 5 tok/s; that also needs ~31 ms/token off compute.

### Why the control runs on a helper thread: pageable D2H is synchronous

`ctljoin` was 19.1 ms/token while only ~4 ms is genuine imbalance (control 31 ms
against 27 ms of q/k/v/g), so most of it looked like pthread creation and
scheduling latency. The apparently clean fix is to drop the helper thread
entirely: queue the four q/k/v/g GEMVs without syncing, run the control on the
main thread underneath them, and flush once.

It measured **2.797/2.808 against 3.536/3.528**. `ctljoin` did go to exactly
0.000 as designed, but `kproj` rose 4.686 -> 5.758 and `netkda` 3.315 -> 5.054.

The reason is that `cudaMemcpyAsync` device-to-host into **pageable** memory is
synchronous, and the engine's activation buffers are plain `malloc`, so the
deferred calls never deferred.

Redone properly -- deferred downloads landing in a PINNED staging pool with the
host-side copy-out moved to the flush -- the overlap does work, and it still
loses:

| term | helper thread | pinned defer + inline control |
|---|---:|---:|
| `kproj` | 4.521 | **3.685** |
| `ctljoin` | 1.790 | **0.000** |
| `khead` | 0.760 | 2.012 |
| `netkda` | 3.459 | 5.039 |

2.977/2.961 against 3.530/3.522. `kproj` fell by the predicted 8.4 ms/token and
`ctljoin` went to exactly zero, but running the control on the main thread
streams 5.93 MB immediately before `khead`, whose per-layer KDA state is 6.3 MB,
and it desynchronises the ranks -- `khead` and `netkda` together give back twice
what `kproj` saved.

Deferring the four GEMVs on their own, with the control left on its helper
thread, is **neutral**: 3.527/3.530, `kproj` unchanged. That is expected once
you look at the numbers -- the four GEMVs are issued back to back on one stream
and execute in that order regardless, and nsys puts `cudaStreamSynchronize` at
about 0.6 us, so removing three of them per layer saves nothing. The blocking
was always in the download waiting for a kernel that has to run anyway.

So the deferred-sync machinery only pays if there is independent CPU work to run
underneath, and the only candidate -- the control projections -- costs more
elsewhere than it saves. Both were reverted.

That is also the answer to why a helper thread exists at all: with pageable
destinations there is no other way to overlap CPU work with a dense GEMV, and
with pinned ones the only work available to overlap makes things worse.

### The f32 router path is faster and NOT bit-exact -- do not ship it

`router_w` is fmt 0 (raw f32, [n_experts 896, hidden 7168] = 25.7 MB per layer,
2.36 GB/token) and was the last dense tensor still falling through to the stock
path, because `coli_k3_dense` admitted only fmt 1 and 4. An exact f32 kernel is
easy -- the fmt-0 reference is `sum += xs[i]*w[i]` strided by 256 with the usual
256-lane tree and NO scale, and at f32 both operands vectorise, so the 4-wide
fold costs two float4 loads per four elements.

It measures well. Against the stock path at the router shape:
**447.5 GB/s mirrored against 298.4**, and `rel=0.0e+00` in the microbenchmark.
It must be gated on the tensor actually being mirrored, because zero-copy it is
only 150.2 GB/s -- the stock path uploads and this one would not.

End to end it reached **3.947/4.021 tok/s** against 3.896/3.896, with `router`
1.318 -> 1.185 and `moe` 11.844 -> 11.184 s/100 tokens.

**And it changes the output.** The generated bytes stop matching
`a3439dc6...a83ee2`, and expert hit rate moves 99.5% -> 99.7%, i.e. ROUTING
changed. The text stays coherent, so this is a last-ulp difference in the router
scores flipping borderline top-16 selections -- not garbage, but not the same
computation.

The microbenchmark says `rel=0.0e+00` because it compares one shape against
`coli_cuda_matmul`. In the engine the reference is not uniform: the 30 GB
`coli_cuda_matmul` budget is shared across every dense tensor it places, so once
it is exhausted some routers fall back to quant.h's **CPU** f32 matmul, whose
summation order differs from any GPU kernel. The accepted hash encodes that
mixed placement. Enabling the k3 path moves those layers onto the GPU and
changes their bits.

So this belongs with the precision change: a real speedup that needs a quality
gate rather than the reference hash. Reverted.

### ctljoin is not thread startup -- measured, not inferred

After the dense path got mirrors and the 4-wide fold, `ctljoin` grew from 1.79
to 2.21 s/100 tokens. That is expected and not a regression: `kproj` is
max(control, GPU) and the GPU side shrank, so the control is now unambiguously
the critical path there and GPU work in `kproj` is worth nothing.

The obvious reading -- that the gap is the per-layer `pthread_create` plus a
fresh OpenMP team, roughly 760 thread creations per token -- is **wrong**.
PROF2 now reports `ctlstart`, the spawn-to-work-start latency:

| term | s/100 tokens |
|---|---:|
| `ctlwork` | 3.179 |
| `ctljoin` | 2.210 |
| **`ctlstart`** | **0.177** |

1.8 ms/token, i.e. 26 us per layer, which is just `pthread_create`. Startup is
not the problem, so a persistent worker pool cannot fix this -- and that
measurement is what stopped it being built a second time.

What remains is `ctlwork` itself: 31.8 ms/token in-engine against 7.4 ms
standalone cold. That 4x is contention for LPDDR with the GPU, and two more
attempts confirm it is not a parallelism problem:

| setting | warm tok/s | `ctlwork` |
|---|---:|---:|
| **OMP=10, default spin** | **3.896 / 3.896** | **3.179** |
| OMP=12 | 3.673 / 3.677 | 3.763 |
| OMP=10, `GOMP_SPINCOUNT=0` | 3.740 / 3.733 | 3.433 |

More threads make the control *slower*. That was read as a bandwidth-bound
region, and the conclusion drawn -- that the only lever left was fewer bytes,
i.e. stored precision -- was **wrong**.

The right reading is that this region runs on a pthread created fresh for every
KDA layer, so its OpenMP team is CONSTRUCTED FROM SCRATCH 69 times per token,
and construction scales with the worker count while the work is bandwidth-bound.
So the correct move is FEWER workers, not more, and there is a clear optimum:

| control workers | warm tok/s | `ctljoin` |
|---|---:|---:|
| 4 | 3.600 / 3.656 | 1.181 |
| **6 (default)** | **4.279 / 4.279** | **1.006** |
| 8 | 4.198 / 4.196 | 1.541 |
| 10 (ambient) | 4.084 / 4.079 | 2.183 |

`ctljoin` **2.183 -> 1.006** and `kproj` 4.399 -> 3.272 s/100 tokens. Bit-exact:
each output row is still summed sequentially by a single thread, so the worker
count cannot change a value. Passive waiting (`GOMP_SPINCOUNT=0`) still hurts
everything else (`khead` 0.609 -> 1.218) and is unrelated.

The lesson: "more threads made it slower" has two very different explanations,
and picking the wrong one closed off a 5% win for several iterations.

### The KDA control's cost is contention, not thread churn

PROF2 now reports `ctlwork` (time inside `kda_control_b1`, wherever it runs) and
`ctljoin` (the exposed wait). Warm four-Spark decode:

| | s/100 tokens |
|---|---:|
| `ctlwork` | 3.085 |
| `ctljoin` | 1.868 |
| `kproj` (contains ctljoin plus the q/k/v/g GEMVs) | 4.600 |

The same work standalone, streaming 69 distinct weight sets so nothing stays
cached, is **0.74 s/100 tokens at 55 GB/s**. So the work is cheap and the engine
pays 4x for it. The obvious suspect is that the old form creates a pthread per
KDA layer and each builds its own OpenMP team -- roughly 760 thread creations
per token.

That suspect is wrong. Replacing it with one long-lived worker fed by a
condition variable (so its team is built once and reused, and it blocks rather
than spins between layers) measured **2.982/2.932 against 3.544/3.536**, and the
breakdown says why:

| term | pthread per layer | persistent worker |
|---|---:|---:|
| `ctlwork` | 3.085 | 3.355 |
| `ctljoin` | 1.868 | 1.987 |
| `kproj` | 4.600 | 4.675 |
| **`attn`** | **13.726** | **17.417** |

`kproj` is flat; `attn` grows by 37 ms/token. The persistent thread keeps a
second 10-thread OpenMP team resident, and that team contends with the main
thread's regions (`khead`, the conv sweeps) even though it is asleep. The
transient per-layer thread is cheaper precisely BECAUSE its team dies with it.
This confirms the earlier "hot team stole CPU" rejection on a warm measurement,
not just the cold one.

The remaining 4x on `ctlwork` is therefore contention for LPDDR with the GPU,
which is streaming its own weights throughout. Shrinking the 409 MB stream is
the only lever left on it, and that means changing the stored precision of
`f_a`/`f_b`/`b_proj` -- a numerics change, not an exact one.

### An OpenMP region costs ~58 us in the engine, not the ~2 us a microbenchmark shows

This is the single most misleading measurement in this codebase, and it has now
produced three wrong predictions in a row. A tight loop around an empty
`#pragma omp parallel` on this box reports **1.95 us**. In the engine the same
region costs roughly **58 us**, because between regions there is GPU work,
syscalls and collective waits, so the GOMP team exhausts its spin count, sleeps,
and has to be woken.

Measured by parallelising three provably bit-exact loops at once -- the
shared-expert SiTU activation (elementwise), `res_mix`'s scoring loop (split
over entries so each inner double reduction stays sequential) and `out += sd`
(elementwise). Isolated harnesses predicted 6.60 -> 2.49 ms/token for `res_mix`
and 1.60 -> 0.20 s/100 for SiTU, about 8.8 ms/token in total. The four-Spark
result went the other way: **3.269/3.268 against 3.485/3.480**, a ~6%
regression, i.e. ~18 ms/token lost across ~463 new regions per token.

So the `SERIAL BY DESIGN` note on the KDA conv loop generalises: do not
parallelise a region on the decode path unless it is worth *well over* 58 us of
work, and do not trust a hot-team microbenchmark to tell you whether it is.

The same class of error has now appeared three times, always because a harness
reproduced the arithmetic but not the machine state:

| harness said | engine said | what the harness got wrong |
|---|---|---|
| int4 group scales cost +21-57% | no change | probe used device memory; production is zero-copy |
| KDA control is 0.11 ms/layer | ~1.14 ms/layer exposed | one 5.93 MB working set stayed L2-hot |
| omp region is 1.95 us | ~58 us | tight loop kept the team spinning |

Reproduce production conditions -- registered host memory, a cold working set,
and realistic gaps between regions -- or measure end to end.

### The strict-f32 KDA control is memory-bound, and overlap is what saves it

`kda_control_b1` reads 5.93 MB of f32 weights per KDA layer -- `f_a` [128,7168],
`b_proj` [24,7168] and `f_b` [3072,128] -- which is **409 MB per token**, the
largest CPU memory stream in decode. Under `K3_KDA_OVERLAP` it runs on its own
pthread while the main thread blocks in `pthread_join`; the OSRT trace shows
4,225 joins totalling 10.4 s in that 26.5 s window.

That overlap is worth a great deal: **`K3_KDA_OVERLAP=0` measured 2.748/2.719
tok/s against 3.510/3.505**. Do not remove it to avoid the per-layer thread
churn -- the churn is real (~760 thread creations per token counting the
OpenMP teams) but it is far cheaper than exposing the control work.

Beware benchmarking this in isolation: a standalone harness that loops over one
5.93 MB working set reports 0.11 ms/layer (47-54 GB/s) because it is L2-hot.
In the engine the stream is cold every token AND contends with the GPU for the
same LPDDR, which is why exposing it costs ~79 ms/token rather than ~8.

Giving the region more workers does not help, and the schedule is not why:

| control threads | schedule | warm tok/s |
|---|---|---:|
| 10 (ambient `OMP_NUM_THREADS`) | static | **3.510 / 3.505** |
| 18 | static | 3.123 / 3.128 |
| 18 | dynamic | 3.141 / 3.146 |

GB10 is 10 Cortex-X925 + 10 Cortex-A725, so the obvious reading is that a
static split makes an efficiency core the critical path -- but `dynamic`
scheduling recovers almost none of it. The region is bandwidth-bound and
already contending with the GPU, so extra cores add contention rather than
throughput. Shrinking the 409 MB stream (or moving it to the GPU with an
order-exact f32 GEMV) is the only lever here; adding parallelism is not.

### Where the remaining decode time is

Decode is dominated by collective *latency*, not bandwidth or dense compute.
Measured with `tests/bench_k3_net` on the four ranks, a **1-float** all-reduce
costs 0.82 ms against 0.87 ms for the 28 KB attention payload — the payload is
almost free, the round trip is everything. `t_kout` includes the attention
all-reduce, so its 4.27 s/100 tokens is ~80% collective and only ~0.08 ms/layer
of actual o_proj.

The cause is the cross-pair link. RTTs, same cluster:

| link | RTT |
|---|---|
| `enp1s0f0np0` 200 GbE, within pair | 0.003–0.008 ms |
| `enP7s7` 1 GbE bridge, across pairs | 0.352 ms min / 1.057 ms avg |

**Each Spark has two 200 GbE ports and both are cabled to the same partner**
(verified by neighbour MAC: `10.20.12.2` is spark2's second port, `10.20.34.2`
is spark4's). The second cable in each pair is pure redundancy. Re-patching
spark1:port1 -> spark3:port1 and spark2:port1 -> spark4:port1 yields exactly the
topology RD2 already implements, at 200 GbE, with no code change — only new
`CROSS` addresses in `k3_launch.sh`. That is the single largest remaining win
and it needs no new hardware.

Two software attempts on this were measured and rejected: bounded spinning
before `poll()` in `k3_exchange_add` (to dodge the 231/433 us LPI-2/LPI-3 exit
latencies) was within run-to-run noise at 300 us and worse at 1000 us, because
back-to-back collectives never let the core idle in the first place. EEE is
`enabled - active` on `enP7s7` and is a plausible contributor to that 1 ms
average, but disabling it needs root and was not tested.

## Chunked prefill

Prefill processes the prompt in chunks of `K3_CHUNK` tokens, layer-major:
every dense matmul batches over the chunk (each weight matrix streams from
RAM once per chunk instead of once per token), the MoE loads each unique
expert of the chunk once (measured at C=32 on real text: 2.7x dedup —
neighbouring tokens share experts far more than QB-flat routing suggests —
9.6 instead of 25.8 GB/token), and the lm_head runs only on the chunk's last
token. Sequential state (KDA recurrence, MLA cache, AttnRes bookkeeping)
advances per token inside each layer in the original order, so chunked
results are **bit-identical** to token-at-a-time (verified: 125-position
teacher-forced logit streams at C=32 vs C=1 match exactly). Measured prefill:
~5.3 -> **2.0 s/token** at C=32. The KDA state-update sweeps are AVX2
(scalar fallback for head dims not divisible by 8).

## Chat, API, and Web

```sh
# standalone one-turn diagnostic
./kimi_k3 <model_dir> --chat "your question" [--system "..."] --ngen 300

# first-class multi-turn interfaces (config.json auto-detects Kimi K3)
COLI_MODEL=<model_dir> ./coli chat
COLI_MODEL=<model_dir> ./coli serve
COLI_MODEL=<model_dir> ./coli web
```

K3's chat format ("XTML", from the checkpoint's own `encoding_k3.py`) uses
only four special tokens — `<|open|>`, `<|close|>`, `<|sep|>`,
`<|end_of_msg|>` — around ordinary-text tag names and attributes:

```
<|open|>message role="user"<|sep|>TEXT<|close|>message<|sep|><|end_of_msg|>
<|open|>message role="assistant"<|sep|><|open|>think<|sep|>        <- generation prompt
```

The assistant's thinking is a *structural* channel and is preserved when an
OpenAI-compatible client sends prior `reasoning_content`; `enable_thinking=false`
opens `<response>` directly. The gateway does not flatten XTML into a string:
it sends length-framed messages to the C engine, which builds every structural
and ordinary-text segment at the tokenizer boundary required by K3's rank-BPE.
The multi-turn wire was compared against the official `encoding_k3.py` and
tiktoken on system/user/assistant history with UTF-8 content: **77/77 token IDs
exact**.

`coli chat` starts a private local server for Kimi and keeps the 2.8T model
loaded for the whole terminal session. `coli serve` exposes streaming and
non-streaming `/v1/chat/completions`; `coli web` uses that same API. Reasoning
is returned as `reasoning_content`, response text as `content`, and
`<|end_of_msg|>` remains the model-owned stop token. `STOP` and `CANCEL` are
honoured between generated tokens.

## Current limitations

- Decode is single-token (no speculative decoding — K3 has no MTP head).
- Tool declarations/calls and image content are not exposed through the shared
  gateway yet; unsupported requests fail explicitly.
- CPU only (no CUDA/Metal/Vulkan tier).
- The protocol, tokenizer, gateway, TUI, and Web client paths are locally
  testable without the 1.5 TB checkpoint. A release claim still requires one
  full-model multi-turn TUI/Web run on a host that owns the complete snapshot.
