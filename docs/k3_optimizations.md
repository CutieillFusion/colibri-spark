# Kimi-K3 four-Spark decode: optimization tracker

Baseline for this list: `d19dc26`, **3.903 / 3.898 tok/s** warm, byte-identical
to reference `a3439dc6...a83ee2`. Target 5 tok/s = 200 ms/token; currently
256 ms/token.

Budget, measured (s per 100 tokens; x10 for ms/token):

| term | s/100 | nature |
|---|---:|---|
| exposed collectives (`netkda` 3.32 + `netmla` 1.01 + `netmoe` 1.01) | 5.34 | 1 GbE round trip, hardware |
| `ctlwork` (22.0 ms exposed as `ctljoin`) | 3.18 | CPU, bandwidth-contended |
| `shared` | 4.31 | GPU; also the work hiding the latent collective |
| `kproj` | 4.42 | GPU + control wait |
| `kout` | 3.90 | 85% collective |
| `expert` | 2.64 | GPU, at its memory wall |
| `latent` | 1.59 | GPU |
| `router` | 1.32 | GPU, stock path |
| `mout` | 1.29 | 78% collective |
| `kconv` | 0.20 | CPU (was 1.10) |
| `matt` | 0.84 | CPU |
| `khead` | 0.61 | CPU |
| `topk` | 0.09 | CPU (was 0.44) |
| `resmix` | 0.47 | CPU, now measured |
| `head` | 0.15 | GPU (sharded) |

Two more facts that shape the list:
- **GPU is idle ~76%** of decode. Kernel work is ~92 ms of a 256 ms token.
- **`cudaMemcpyAsync` is 23% of a decode slice** (5.130 s of 22.17 s). Every
  GEMV round-trips through host memory because the CPU touches data between
  kernels. `cudaLaunchKernel` is only 1.3%, which is why CUDA graphs are not
  the answer.

## Rules

- Anything **bit-exact** is gated on the generation hash matching
  `a3439dc6...a83ee2` on all four ranks, plus the C test suite.
- Anything **not bit-exact** is gated on `tools/k3_quality.py`, which must have
  been calibrated and negative-controlled in the same session. Teacher-forced
  metrics alone are never sufficient.
- Test before AND after, end to end. Microbenchmarks on this engine have
  mispredicted the direction four separate times (see `docs/kimi_k3.md`).

## Bit-exact — DONE

- [x] Top-16 expert selection O(K^2*E) -> O(K*E) (`topk` 0.436 -> 0.085 s/100) — 3.913 -> 3.957
- [x] Rotating conv window instead of a per-token memmove (`kconv` 1.103 -> 0.200 s/100, 5.5x) — 3.982 -> 4.084
- [x] `matt` score buffer on the stack, not malloc'd per head per token
- [x] K3_DENSE_DEV_GB auto-sized: it defaulted to 0, so nothing was ever
      mirrored -- 3.86 -> 4.33 tok/s, output unchanged 6/6
- [x] 4-wide fold confirmed OPTIMAL: W=2 4.019, W=4 4.258, W=8 4.170,
      W=4 x 2rows 4.234 -- a peak, not a slope; +6.5% but only with mirrors on
- [ ] REVERTED: MLA absorb on GPU (fb898f3) -- +0.4% is inside noise
- [ ] REVERTED: KDA control part 2 on GPU -- dead wash, 3.877/3.888 vs 3.877/3.883
- [ ] REJECTED: shard the replicated shared experts -- compute saving 0.2 ms/layer
      vs 1.18 ms/layer for the added collective; 4.33 -> 2.92 tok/s
- [ ] REJECTED: pinned staging for activation vectors -- wash on unified memory
- [ ] REJECTED (retest): K3_FUSE_QKVG with mirrors on -- still a wash, 4.248 vs 4.258
- [ ] BLOCKED: dense kernel diagnosis needs ncu; ERR_NVGPUCTRPERM, needs root
      `options nvidia NVreg_RestrictProfilingToAdminUsers=0` + module reload
- [x] float4 shared loads for the w2 expert kernel: ALREADY DONE -- k3_w1_down_fast
      shares warp_row_dot_w1 with the gate/up kernel, so both got stride-36 float4
- [x] short-row dispatch: I<=4096 uses 4 rows/block, latent 1.596 -> 1.555, 6/6 exact
- [ ] REJECTED: mapped activations (cudaHostAllocMapped) -- 4.06 vs 4.30; x is
      re-read once per block, so it must stay L2-resident in device memory
- [x] split the RD2 cross exchange: bridge bytes halved, netkda -12.4%,
      netmla -19%, 6/6 exact. End to end +0.7% (noise) -- the collective terms
      are the honest result
- [ ] CORRECTED: there is NO expert work imbalance -- counts are balanced within
      2.6% and anti-correlate with time. Residual real skew is 2.2 ms/token and
      its cause is unidentified (store layout, I/O volume, host load all ruled out)
- [x] split the exposed collectives only: netmoe -15%, 4.362 -> 4.391, 6/6 exact
- [ ] REJECTED: SO_BUSY_POLL -- inapplicable, the exchange loop uses MSG_DONTWAIT
      so recv() never blocks and the option only affects blocking receives
- [x] BATCHED 1-BIT EXPERT KERNELS, default ON: 4.502 -> 4.648 (+3.0%), expert
      2.691 -> 2.121 s/100 (-21%), 6/6 byte-identical. ncu found the reason the
      old assumption was wrong: Waves Per SM 1.56, DRAM 19% of peak -- the expert
      kernels are TAIL-bound, not bandwidth-bound
- [x] K3_EXPERT_GB 80 -> 88: 4.810 -> 5.033 tok/s, PAST 5. Removes rank 0's
      cache misses, which were the straggler creating collective skew. 100 is
      faster (5.157) but OOM-kills on long runs -- lazy slots hide the ceiling
      from a short benchmark
- [x] PREFILL GEMM (k3_dense_i4g_exactW_S): S!=1 fell off the fast path entirely,
      so prefill ran the 15x-slower generic kernel. 275 -> 194 ms/prefill-token
      cold, request 310 -> 137 s, 6/6 byte-identical. Census: gemm=9984
      generic=4496 (the 4496 are int8 MLA/lm_head), zero declines
- [x] PROF2P: prefill phase instrumented at last (PROF2's baseline sat below the
      prefill loop, leaving 74% of a prompted request unmeasured)
- [x] BATCHED FUSED SiTU: was syncing once per token; shared 14.38 -> 7.34 s,
      prefill 6.91 -> 7.40 tok/s, 6/6 byte-identical
- [ ] NEXT: prefill kernels are decode-shaped -- 256-lane tree reduction with
      block-wide syncs per token. Warp-per-row + shuffle + register tiling
- [ ] OPEN: prefill still ~13x off its 10 ms/token floor -- expert cache thrashes
      at C=32 (~390 unique of 896 per layer vs 210 slots). Sweep K3_CHUNK now that
      small chunks are no longer punished by dense
- [ ] NEXT: release the host copy of mirrored tensors (~16 GB) to raise the
      cache ceiling further
- [ ] LARGELY EXPLAINED: ~80% of collective time is rank arrival skew.
      TTFB says 33 us/exchange moving bytes, RTT says ~27 us latency, leaving
      ~216 us of skew. Both links measure ~54 us RTT, so the 1 GbE bridge is NOT
      a latency floor. ~35 ms/token at stake; thread count does not move it
- [ ] PARTIAL: rank-0 0/4 shard cuts cross-rank spread 0.417 -> 0.265 but is a
      dead heat end to end; kept for symmetry and 319 GB of disk
- [x] BUG FIX: K3_EXPERT_BATCH=1 crashed the engine (2-bit kernels on a 1-bit
      store); guard now checks !w1_mode. Untestable here until a 2-bit store exists
- [ ] CLOSED BY ARITHMETIC: hiding the 7.3 ms/token of expert I/O -- per layer
      there is 0.643 ms to hide and 0.467 ms of cover, so >=0.176 ms must be
      exposed and exactly 0.176 is. Reordering only moves which term shows it
- [ ] REJECTED: MADV_HUGEPAGE for the expert cache. Verified applied (engine
      AnonHugePages 0 -> 40.78 GB, ~10M fewer PTEs) and throughput unchanged:
      4.409/4.395 vs 4.427/4.407. The mirror win was the ACCESS PATH
      (device-resident vs SMMU-translated), not page size
- [ ] REJECTED: speculative decoding. Batching is WORSE per token here --
      C=1 234.7 ms, C=4 360.4, C=8 304.4 -- so four together cost 1442 ms
      against 939 ms sequential. No acceptance rate rescues a 1.5x loss, and
      fixing the S==1 dense limit leaves 1190 vs 939
- [ ] REJECTED: big.LITTLE pinning (10x X925 @3.9GHz + 10x A725 @2.808GHz).
      main->big/control->little 4.253, +loaders->little 4.347, unpinned 4.347/4.355.
      khead worse in every pinned variant -- it is bandwidth-bound and two
      clusters give more aggregate path width than one faster one
- [x] OMP team 10 -> 8 workers: khead 10.4 -> 7.8 ms/token (-24%), 4.373 -> 4.389,
      6/6 exact. 10 OMP + 6 control + loaders + network did not fit 20 cores
- [ ] MEASURED: SiTU is the only serial CPU loop left, but it sits inside the
      allreduce_start/wait window -- speeding it up moves time to netmoe, not out
      of the token. khead and matt are already omp-parallel
- [ ] REJECTED: KDA control on the GPU (block-per-row tree). Removes 26 ms/token
      of CPU -- ctlwork 27.5->11.3, ctljoin 9.9->0.04, kproj 32.7->24.8 -- and
      costs a third of throughput: netkda +44.7 ms, netmla +19.4, khead +14.6.
      CPU time is free here; GPU/memory time is not
- [ ] HARNESS BUG: cmp_logits.py is prefill-only and certified the above as
      PCC 1.000000000 / bit-identical. Warning added; use run_gen.sh for C==1
- [ ] DEAD: two-pass router with safe pruning. The int8 error bound is 0.1818
      against a 16th-to-17th score gap of 0.0032 -- 57x too loose, and wider than
      the whole 1st-to-16th spread. Only int16 clears the bound, worth ~2.8% for
      ~200 lines and a correctness hazard. Declined
- [x] FUSED SiTU on the GPU, default ON: 4.358 -> 4.469 (+2.3%), shared 42.9 ->
      35.4 ms/token. PCC 0.999999999 vs a 0.999 bar, top-1 100%, free-running
      coherent. Passes because the sigmoid is computed in double and rounded
      once (CUDA expf misses glibc 30.3% of the time, the double form 0.062%)
- [ ] FAILS 0.999: int8 router. +6.2% (4.483 -> 4.760, router 13.0 -> 4.4 ms) but
      PCC 0.985126289, top-1 87.5%, max|dlogit| 4.27. Kept behind K3_ROUTER_I8=1.
      Perturbs a DECISION (top-16 of 896) not a value -- see the 0.0032 score gap
- [ ] FAILS 0.999: fp16 router. +3.1% (4.516 -> 4.664), top-1 100%, PCC 0.994432661.
      Better than int8 on every measure and still short. K3_ROUTER_F16=1
- [ ] WRONG, CORRECTED: "kproj is int8 (6.07 GB/token) not int4 -- the largest byte source
      in the model. The int8->int4-g64 downcast in w_load_rows exists but does not
      fire under K3_BITS=4 (formats byte-identical); worth ~17 ms / 7.6% if fixed
- [ ] LEAD: f32 group scales are 11.1% of every fmt=4 tensor; fp16 scales would
      halve that (~0.9 GB/token). Value perturbation, same class as fused SiTU
- [ ] REJECTED: K3_MLA_BITS=4 / K3_HEAD_BITS=4. mproj -25% as bytes predict, but
      matt +34% (CPU absorb unpacks nibbles) and head 5.7x worse; 4.563 -> 4.389
- [ ] STILL OPEN: kproj at ~105 GB/s is unexplained. Three hypotheses falsified;
      needs ncu. The "it is int8" resolution was a shape-collision error
- [ ] REJECTED: two-pass router (int8 search + exact f32 decision). Correct, it
      is SLOWER than f32 -- 4.412 vs 4.508, router 1.626 vs 1.298. Candidates
      average ~106 of 896 (not the 29 one sample showed) and the single-block
      selection kernel costs more than the bytes saved. K3_ROUTER2=1
- [ ] SUPERSEDED: fp16/bf16 router (halves rather than quarters bytes, keeps
      relative precision), or int8 first pass + exact re-score near the boundary
- [ ] OPEN (non-exact): router is f32, 2.36 GB/token (~15% of all traffic). int8/int4 would
      cut it 4-8x but it feeds top-16-of-896 selection -- needs the quality gate
- [ ] OPEN: d1657cc changes decode output via FP contraction; decide whether to
      pin -ffp-contract=off for the conv loop or re-baseline the reference
- [x] MLA absorb GEMVs moved to the GPU, BIT-EXACT (`matt` 0.896 -> 0.849) — the
      first successful CPU->GPU migration; see the note in docs/kimi_k3.md for
      why it is the only one currently possible
- [x] Control region sized to 6 workers, not the ambient 10 (`ctljoin` 2.183 -> 1.006) — 4.084 -> 4.279

- [x] Exact int8 GEMV folded onto 128 threads (`dee7fd0`) — 3.238 -> 3.296
- [x] 32-way shared-memory bank conflict in the 1-bit expert GEMV (`18fb60e`) — 3.296 -> 3.510
- [x] `float4` shared loads at stride 36 (`7fcaf73`) — 3.510 -> 3.523
- [x] Device mirrors for the exact dense kernel (`9cdbc3a`) — 3.525 -> 3.605
- [x] Right-size the mirror budget to 16 GB (`c193073`)
- [x] 4-wide consecutive fold, int4 (`34b5c61`) — 3.609 -> 3.780
- [x] 4-wide fold int8 + lm_head mirrored (`a99d7b3`) — 3.780 -> 3.847
- [x] Shard the lm_head for greedy decode (`908dba4`) — 3.797 -> 3.914

## Bit-exact — REJECTED (measured, do not retry without a new reason)

- [x] ~~Fold factors 4 and 8 on the {t,t+128} construction~~ — slower
- [x] ~~8-wide consecutive fold~~ — 32 threads is one warp, 50% occupancy
- [x] ~~Row tiling 2/4 per CTA~~ — x re-reads are L2-served
- [x] ~~Pair fold {2p, 2p+1}~~ — halves loads, DRAM-bound so invisible
- [x] ~~Shared-memory scale staging~~ — probe used device memory, production is zero-copy
- [x] ~~GPU port of the KDA control~~ — 152 rows -> 5 warps -> 9.8 GB/s
- [x] ~~Persistent control worker~~ — its OpenMP team contends; `attn` +37 ms
- [x] ~~Control inline under deferred GEMVs~~ — `khead` and `netkda` regress
- [x] ~~Deferred GEMV sync alone~~ — neutral; one stream, back-to-back anyway
- [x] ~~Parallelising SiTU / `res_mix` / `out+=sd`~~ — omp region costs ~58 us in-engine
- [x] ~~Parallelising SiTU ALONE~~ — retried on the theory that the earlier loss
      was the bundled region count (463 added regions). It is not: the loop is
      ~67 us of work per layer against a ~58 us region, so there is nothing to
      win, and contention with the concurrent latent all-reduce makes it worse.
      4.104/4.120 against 4.284/4.273, `shared` 4.262 -> 5.092.
- [x] ~~`khead` memset sizing + `__restrict__` on the ARM path~~ — the memsets
      cleared the full 512-float scratch when only hd=128 entries are used, and
      the AVX2 block is x86-only so ARM takes the scalar branch. Both fixed,
      no gain (khead 0.986 -> 1.016, end to end inside noise): GCC already
      vectorises that loop (16 NEON FMAs in the body) and the extra memset is
      negligible against the 424 MB/token of state the loop sweeps.
- [x] ~~Caching the block-snapshot RMS in `res_mix`~~ — genuinely redundant
      computation (snapshots are fixed, their RMS was recomputed ~36x each), but
      no gain: 0.474 -> 0.462 s/100, end to end inside noise. That loop is bound
      by reading the snapshots, not by the double arithmetic.
- [x] ~~More control threads (18) / dynamic schedule~~ — but FEWER (6) is a large win, see above
- [x] ~~`OMP_NUM_THREADS`=12, `GOMP_SPINCOUNT`=0~~ — both worse
- [x] ~~Bounded spin before `poll()` in the collective~~ — inside noise
- [x] ~~Device mirrors for expert slots~~ — 145.7 vs 148.3 GB/s, no effect
- [x] ~~CUDA graphs~~ — launches are 1.3% of wall; topology is data-dependent
- [x] ~~Hoisting the MLA KV decode out of the per-head loop~~ — bf16 `kv_dec`
      looked ~48x redundant (Lt does not depend on the head, and it is decoded
      twice per head). Decoding once into f32 scratch measured WORSE: `matt`
      0.836 -> 0.903, end to end 3.944/3.940 against 3.957/3.968. Those loops
      are bound by KV memory traffic, not decode instructions, so replacing
      2-byte reads with 4-byte reads costs more than the shifts saved. bf16 KV
      is a bandwidth optimisation and decoding on the fly is correct.

## Not bit-exact — REQUIRES `tools/k3_quality.py`

Ordered by risk-adjusted value: activation-only ulp differences first, routing
last. Each must be landed and gated SEPARATELY so a failure is attributable.

- [!] **1. SiTU fused into the shared-expert GPU epilogue** (~10 ms/token)
      KERNEL CORRECT, MODEL DIVERGES. Do not ship without a scored eval.
      Speed: **4.001/3.995 against 3.903/3.898 (+2.5%)**, `shared` 4.310 ->
      3.477 s/100 with `netmoe` 1.013 -> 1.302 (a faster shared expert exposes
      more of the collective it was hiding, so ~3 of the ~10 ms comes back).
      Correctness of the kernel is established twice over:
        * `tests/bench_k3_situ` isolates it against `coli_k3_dense` x2 plus the
          CPU `situf_` at the real 6144x7168 shape: worst |diff| **1.46e-11**
          (rel 7e-08) at both S=1 and S=8.
        * an in-engine verifier recomputing the reference per layer: worst
          **5.96e-07** on values of order 1-4, i.e. float32 ulp.
      And yet end to end the model moves a long way: **top-1 agreement 54.7%**,
      max |dlogit| **11.4**, and only **80.2%** of route-histogram bins
      unchanged. Reproduced on a clean capture, so it is not a capture artefact.
      **This is the finding, and it generalises.** K3 selects top-16 of 896
      experts on a sigmoid router. A ulp-level activation change flips
      borderline selections, a different expert runs, and the difference
      compounds over 92 layers. So for THIS model, "numerically correct" does
      not imply "close output" -- divergence metrics cannot distinguish a good
      non-exact change from a bad one, and neither can reading samples (the
      candidate's sqrt(2) proof is a different but equally valid proof).
      Consequence for the whole non-exact list below: accepting any of it
      requires a SCORED EVAL (benchmark accuracy on a held-out set), not a
      similarity gate. `tools/k3_quality.py` remains the right instrument for
      catching regressions and reward-hacking, but it cannot license a change
      that legitimately relocates the model.
      Code is kept: the kernel and `coli_k3_gate_up_situ` stay in
      `backend_cuda_k3.cu` with `tests/bench_k3_situ` exercising them, but they
      are NOT wired into `moe_forward`, so decode is untouched.

- [ ] **2. KDA conv + SiLU on the GPU** (~8 ms/token)
      Risk: reassess. Item 1 shows ulp-level changes relocate this model
      wholesale, so "LOW risk because it is only `expf`" was wrong. Needs the
      same scored eval.
- [ ] **3. Device-resident activations across a layer** (up to ~20-40 ms/token)
      Risk: MED. No arithmetic change by itself, but only becomes possible
      once 1, 2, 6 and 7 remove the CPU steps in the middle. This is the 23%
      `cudaMemcpyAsync` term.
- [ ] **4. Control stored at bf16 or int8** (~15-20 ms/token)
      Risk: MED-HIGH. `f_a`/`f_b` feed the KDA DECAY GATE, so error compounds
      through the recurrent state. The repo kept these unquantized on purpose.
- [ ] **5. f32 router on the exact dense path** (measured: 3.896 -> 4.021)
      Risk: HIGH despite being small — it perturbs ROUTING directly. Observed
      expert hit 99.5% -> 99.7%. Build it last.
- [ ] **6. MLA attention on the GPU** (~6 ms/token) — needs KV cache on device.
- [ ] **7. KDA recurrence on the GPU** (~4 ms/token) — state is 566 MB, fits.
- [ ] **8. `res_mix` on the GPU** (~4 ms/token) — fp64 reductions.
- [ ] **9. `topk` on the GPU** (~3 ms/token) — drives host expert dispatch.

## Still open, bit-exact

- [ ] **`float4` shared loads for the w2 (2-bit) expert kernel** — same bank
      conflict as w1 had. Cannot be validated end to end here because the 2-bit
      store is not what this deployment runs.

## Ceiling

Even a perfect fabric only reaches ~4.33 tok/s (256 - 53 = 203 ms/token), so
the collective work alone was never going to deliver 5. Reaching 5 needs the
non-exact list above AND the fabric.

---

## Context scaling (this round)

The earlier list was written from a short-prompt profile. A sweep across
138/513/1963/4863-token prompts showed every weight-bound term (kproj, shared,
latent, router, ctljoin) flat to within 2-3% over a 35x context increase, while
`matt` fit `0.83 + 0.00356*T` s per 100 decode tokens. At T=4863, matt alone was
18.1 s of a 63 s/100-token decode. Long context is an MLA-attention problem, not
a weights problem.

- [x] **Vectorise the bf16 KV loops** — the score and context loops walked the
      cache through `kv_dec()` (u16 load, widen, shift, type-pun), which gcc will
      not vectorise. One X925 core: 2.35 GMAC/s scalar, 19.96 GMAC/s with
      `vshll_n_u16(v,16)` reinterpreted as f32 — 8.5x. At ctx 1963 this took
      decode 3.06 -> 3.62 tok/s (+18.2%) and prefill 6.66 -> 8.04 (+20.7%).
      **matt was compute-bound on the conversion, not bandwidth-bound on the
      cache** — which is why every cache-traffic theory had failed to move it.

- [x] **Tile both operands in the prefill GEMM** — the prefill kernel had been
      rewritten twice on the theory that its shape was the cost. Counting bytes
      showed why that kept failing: both shapes were row-parallel, so every block
      re-read the whole activation block — 2.8 GB of x against 12.4 MB of
      weights, ~1.05 TB/s at the measured rate, i.e. the L2 roofline. Tiling to
      32 rows x 32 tokens cuts x traffic 32x. 7.57 -> 8.19 tok/s.

- [ ] ~~**Head-batched MLA attention**~~ — MEASURED SLOWER (-14% decode, +74%
      prefill matt), left off behind `K3_MATT_BATCH`. "24 heads each re-read the
      cache" counts LOGICAL reads; the heads run concurrently on the same L, so
      the hardware already served them from cache. Batching swapped which operand
      is re-read and quadrupled the OpenMP regions per token. Byte-identical
      output, so the construction was right and the premise was wrong.

### The pattern, seventh confirmation

Removing work wins; moving or batching it loses. Vectorising `kv_dec` removed
instructions and won 18%. Head-batching rearranged the same instructions and lost
14%. Every accepted win this project has taken removes work (device mirror
budget, fused SiTU, split cross-exchange, batched 1-bit expert kernels, tiled
prefill, NEON KV); every rejected one moved it.

### Quality

Budget loosened to e2e PCC >= 0.999. Teacher-forced against the exact path:

| change | e2e PCC | min/position | top-1 |
|---|---|---|---|
| NEON KV only | 0.999994 | 0.999908 | 100% |
| tile prefill only | 0.999902 | 0.998689 | 100% |
| both (shipped) | 0.999504 | 0.997518 | 100% |

Free-running output is coherent and semantically equivalent but no longer
byte-identical: the dense path feeds the router, and the router picks top-16 of
896, so any reordering can flip an expert.

### Next

`shared` (4.470 s/100 tok) is now the largest decode term, but its timer starts
right after an async `k3_net_allreduce_start` and `netmoe` is only 0.145 — the
shared-expert compute is *covering* the latent collective. Shrinking it gains
nothing unless the collective shrinks with it. Measure the collective in
isolation before optimising either.
