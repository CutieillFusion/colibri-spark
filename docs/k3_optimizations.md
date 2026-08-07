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

---

## The stated open list, resolved

Three of the four items were already implemented; checking beat assuming.

- [x] **lm_head sharded + argmax allreduce** — already in (`kimi_k3.c:2499`).
      Each rank takes a contiguous vocab range and only the argmax crosses the
      wire; the full head is kept when `temp>0`, `K3_LOGITS` or trace need the
      whole vector. `head` measures 0.203 s/100 tokens, which is what a
      quarter of a 1.17 GB int8 matrix costs and a quarter of what a replicated
      one would.

- [x] **4-wide consecutive fold for the exact int4 kernel** — already in and the
      default (`K3_DENSE_I4W=4`). W=8 is bit-exact and halves the loads again but
      measured SLOWER (shared gate 377 -> 296 GB/s): 32 threads is one warp, so
      24 blocks/SM reaches only 768 threads.

- [x] **Is `K3_DENSE_DEV_GB` or the 64 MB cap binding?** — neither. The cap
      stopped being 64 MB when the 4-wide fold made mirroring pay for lm_head
      too; it is 2 GB and guards only against one tensor eating the budget. With
      a budget set, 16.06 GB is placed, 0.00 skipped for budget, 0.00 for cap.
      **The real finding: the speed harness was passing `DEVGB=0`, which
      disables mirroring outright.** Every measurement this session ran that way.

      | | tok/s | mirrored |
      |---|---|---|
      | DEVGB=0 | 4.699 | 0.00 GB |
      | DEVGB=auto | 5.153 | 16.06 GB |
      | DEVGB=24 | 5.178 | 16.06 GB |

      auto and 24 resolve to the same budget, so their 0.5% spread is the noise
      floor. Mirroring is bit-exact (6/6 identical) — same kernel, same order,
      different placement.

- [ ] **float4 shared loads for the w2 (2-bit) expert kernel** — still not
      validatable end to end: this deployment runs the 1-bit store.

## Where decode time goes now (5.21 tok/s, standard prompt)

`shared` 3.53 and `arwork` 2.73 are two legs of one overlap: the shared-expert
compute exists to cover the latent allreduce. `netmoe` 0.27 is the exposed
remainder. Cutting either leg alone gains nothing — the other becomes the floor.
That pair is ~20% of decode and needs both halves attacked together, which
retires "optimise the largest term" as a strategy here.

- [x] **One persistent allreduce worker** — `allreduce_start` spawned a thread
      per call, 9108 per 100 decode tokens. arspawn 0.181 -> 0.011, and arwork
      fell 3.248 -> 2.733 because creating a thread also delayed the collective's
      *start*. 5.139 -> 5.211 tok/s, 6/6 byte-identical.

---

## Round 3: where the dense path actually loses

`ncu` on the shipped `exactW<4>` with every tensor mirrored shows Memory 21.85%,
Compute 21.85%, L1 26.68%, L2 14.99% at 115% achieved occupancy. High occupancy
with every pipe near a fifth of peak reads as latency-bound, so the first move
was more memory work in flight.

- [x] **ILP variant** (ushort weight load instead of two byte loads, reduction
      unrolled by 2) — 5.227 -> 5.278 tok/s, kout -5.9%, 6/6 byte-identical.
      ncu called it a wash at kernel level (320 vs 322 us); under replay
      instrumentation a 1% change is invisible, so the e2e A/B decides.

- [ ] ~~**Batched dense** (N tensors, one x, one sync)~~ — MEASURED 7.5% WORSE.

### The real constraint

Per rank the KDA projections are 4 x [3072, 7168] at 4032 B/row = 3.42 GB/token
over 69 layers. Against kproj = 3.06 s/100 tokens that is **112 GB/s, versus
234.5 GB/s for the same kernel on mirrored weights in isolation.** The dense path
gets 48% of what the kernel can do.

That gap is not round-trip overhead, and batching proved it: removing three of
four uploads and three of four syncs made kproj 38% *worse*. The tell was
`ctlwork` — the concurrent CPU control thread got 47% slower without doing
anything different. One long kernel holds the memory system against the CPU
where several shorter ones leave gaps.

So the isolated 234.5 GB/s was measured with nothing else running, and in
production the CPU is on the same LPDDR5X. **Half the dense bandwidth is not
available, and no amount of kernel work will recover it.** The lever is CPU-side
traffic, not GPU-side efficiency.

A first attempt to confirm this by halving `OMP_NUM_THREADS` left kproj unchanged
(3.073 vs 3.072) — but the concurrent consumer is the single KDA control thread,
which `OMP_NUM_THREADS` does not size. That test was aimed at the wrong thread.

### Threading

- [x] **Persistent allreduce worker** — +1.4%, 6/6 identical.
- [ ] ~~**Persistent KDA control worker**~~ — 54% SLOWER.

Refined rule: persistent workers pay for I/O-bound work (the allreduce thread
sleeps on sockets and holds no core) and cost for CPU-bound work (the control
thread competes with the OpenMP pool for the same ten cores).

### Measurement hygiene

The same config measured 5.211 and 5.275 tok/s in two separate script runs.
Cross-run drift exceeds the ~0.5% within-run spread, so only both-arms-in-one-
script comparisons are valid. Every A/B above runs its arms back to back.

---

## Round 4: the CPU and the GPU are fighting over one memory system

Round 3 inferred this from a failed batching experiment. Round 4 has direct
evidence, from a change that runs entirely on the CPU.

The KDA control path (`kda_control_b1`) stores fa/fb/bp unquantised and reduces
them with `for(i) v+=x[i]*w[i]`. gcc cannot vectorise that -- FP addition is not
associative and this build has no `-ffast-math`. One X925 core, 152 rows of
I=7168: **1.96 GMAC/s scalar, 19.86 with NEON, 10.1x.** That is the third
unvectorised CPU loop found this project (after matt's `kv_dec`), and the
pattern is now worth stating: *anything the engine keeps in fp32 or an odd width
and reduces with a plain loop is probably running at a tenth of its speed.*

With it on, at the standard prompt (s/100 tokens):

    ctlwork  2.72 -> 2.19    ctljoin 0.89 -> 0.44
    kproj    3.22 -> 2.85    attn   10.08 -> 9.55

**`kproj` fell 12% for a change that never touches the GPU.** That is the direct
confirmation: less CPU traffic on LPDDR5X means more bandwidth for the dense
kernel. It also explains the 112 vs 234.5 GB/s gap without any appeal to
launch overhead.

### Why it is still off

End to end it is a wash, and at ctx 1963 with EGB=88 it OOM-kills all four ranks
(confirmed in the kernel log).

fa/fb/bp produce the KDA gate and beta, so reordering their reduction changes the
recurrence, the residual stream, and therefore **which experts the router picks**.
A different expert set has a different cache footprint: RSS came out at exactly
72.47 GB with it on and 70.94 GB with it off, in both A/B orderings --
deterministic, not timing. At ctx 1963 the scalar arm already sits at 97.9 GB of
121, so a 1.5 GB shift is fatal.

Trading expert cache for it does not pay either. At ctx 1963:

| | tok/s | RSS | attn | moe | load |
|---|---|---|---|---|---|
| FDOT=0, EGB=88 | 3.916 | 97.9 | 13.010 | 11.454 | 0.858 |
| FDOT=1, EGB=80 | 3.955 | 94.0 | 12.324 | 12.114 | 1.511 |

+1.0%, inside noise. The attention side gains 0.686 s and the smaller expert
cache gives back 0.660 s. EGB=84 would split the difference but lands ~2 GB from
an OOM cliff that grows with context, and "a throughput number from a workload
that never reaches peak residency is not a safe configuration" is a rule this
project already paid for once.

So the win is real and blocked on memory headroom, not on being wrong. It
becomes available if expert residency is reduced some other way -- for example
releasing host copies of mirrored tensors, which is still open.

### Memory is fungible between the mirror and the expert cache

The open question was always phrased as "is the mirror budget binding?". At the
standard prompt it is not -- RSS is 71 GB of 121, nothing is skipped, and more
mirror is free. At ctx 1963 RSS is 98 GB and `load` shows 0.87 s/100 tokens of
real expert streaming, so there the two consumers genuinely compete. Measured at
ctx 1963, two warm passes each:

| | tok/s | mirrored | RSS | expert hit |
|---|---|---|---|---|
| DEVGB=auto, EGB=88 | 3.932 | 16.06 GB | 97.99 | 99.0% |
| DEVGB=8, EGB=96 | 3.907 | 8.00 GB | 103.02 | 99.8% |

-0.6%, inside noise. Moving 8 GB from one to the other changes nothing: what the
dense path loses in bandwidth the expert cache gives back in hit rate. **A GB is
a GB**, and the earlier DEVGB=0 vs auto result (4.699 -> 5.153, +9.6% for 16 GB
of mirror) prices what a GB is worth.

That reframes the largest remaining item. This GPU is integrated, so
`coli_k3_devmirror` does not move the weights to separate memory -- it
DUPLICATES them in the same LPDDR5X, which the startup log says outright
("placement duplicates weights in shared RAM and shrinks the expert cache by the
same amount"). 16.06 GB is currently held twice.

Recovering the host half is not a trade like the table above; it is 16 GB of
pure gain, and on the pricing above that is worth roughly what mirroring itself
is worth. It also unblocks the vectorised control dot, which needs only 1.5 GB.

- [ ] **Free the host copy of mirrored tensors.** Weights are `pread` into heap
      (st.h deliberately avoids mmap), so this is a real `free()`, not an
      madvise. The risk is precise and must be handled first: `w_matmul` falls
      back to the CPU path on `w_blob(w)` whenever `coli_k3_dense` declines a
      shape, and `w_rowdot`/`w_addrow` read `w->f`/`w->q8` directly. Freeing
      without proving no path can reach the host copy turns a decline into a
      use-after-free. The safe shape is to free only tensors whose GPU path is
      unconditional for every shape they are called with, and to make the
      fallback assert rather than read freed memory.

---

## Round 5: the de-duplication route is closed, and headroom is worse than reported

The plan from round 4 was to recover the 16.06 GB that mirroring duplicates.
`cudaMallocManaged` was the safe way to do it -- dereferenceable from both
sides, so the host copy can be freed and the pointers repointed, with no way for
a CPU fallback to reach a dangling pointer.

- [ ] ~~**Managed-memory mirror**~~ — 7.9% SLOWER *and* uses more memory.

    | | tok/s | shared | kout | RSS |
    |---|---|---|---|---|
    | cudaMalloc | 5.206 | 3.509 | 2.949 | 70.94 |
    | cudaMallocManaged | 4.796 | 4.440 | 3.212 | 86.21 |

Two things settled by that one run.

**Managed pages do not keep the mapping.** Every dense timer regresses. Whatever
`cudaMalloc` gets on this part, the unified allocator does not.

**RSS has been lying.** It went UP 15.3 GB, not down. `cudaMalloc`'d device
pages on this integrated GPU are physically resident but *not charged to the
process*; managed pages are. So the duplication was never visible in RSS, the
98 GB seen at ctx 1963 already excludes 16 GB that is really there, and real
occupancy is ~114 of 121 GB. That explains the OOM that killed the vectorised
control dot far better than "a 1.5 GB shift" did -- there was almost nothing
left. It also means **no memory experiment in rounds 3-4 can be validated by
watching RSS**, including the expert-cache-vs-mirror trade.

### What that leaves

Freeing the host copy is still the right idea and still worth ~16 GB, but it now
has to be done by the risky route (real `free()`, plus proving no path can reach
`w_blob(w)`), and its payoff can only be confirmed by throughput, not by RSS.

### Tooling

`deploy.sh` gated on `make ... | grep -iE " error"`. That missed a CUDA 13
signature change (`cudaMemAdvise` now takes a `cudaMemLocation`) which compiled
on the local nvcc and failed on spark1's, and an A/B then ran for minutes
against a stale binary -- second instance of the failure that script exists to
prevent. It now checks make's exit status. **The local and cluster toolchains
differ; spark1's build is the authoritative one.**

---

## Round 6: reclaiming the duplicated host copy

Round 5 closed the safe route (managed memory: -7.9%) and left only the risky
one -- a real reclaim, plus proving no path can reach `w_blob(w)`.

- [x] **Reclaim the host half of mirrored tensors** — 9.61 GB returned.

`free()` is not usable: `w_matmul` falls back to the CPU on `w_blob(w)` whenever
`coli_k3_dense` declines a shape, `w_rowdot`/`w_addrow` dereference `w->f`/
`w->q8` directly, and `w_rows()` hands out ALIASING views into a parent buffer.
A missed reader would not crash -- it would read whatever malloc placed there
next and emit plausible output.

So: **unpin, discard, protect.**

    coli_k3_unregister    pages are cudaHostRegistered; pinned pages survive madvise
    madvise(DONTNEED)     actually returns the physical pages
    mprotect(PROT_NONE)   a missed reader SIGSEGVs at a known address, never zeros

The pointer is never freed, so malloc cannot re-issue the range. Eligibility is
decided by **observation, not argument**: every host-side reader marks its
tensor, and the sweep runs at the start of the SECOND request, because mirrors
are created lazily and only a completed prefill+decode has exercised every shape.

    [K3/RECLAIM] 1248 ranges returned 9.61 GB; 0 skipped (read on host)

Nothing mirrored is ever read on the host in this configuration. Output is
byte-identical, 6/6, as it must be -- no arithmetic changed.

### The payoff

The reclaim is worth nothing by itself: the expert cache is sized by
`K3_EXPERT_GB` and does not grow into headroom on its own. It is worth what it
*unlocks*. EGB=100 previously OOM-killed the ranks; with 9.61 GB back it fits.
At ctx 1963, warm2:

| | tok/s | expert hit | RSS |
|---|---|---|---|
| EGB=88, no reclaim | 3.879 | 99.0% | 97.9 |
| **EGB=100 + reclaim** | **4.476** | 100.0% | 98.1 |
| EGB=96 + reclaim + FDOT=1 | 4.168 | 99.8% | 98.1 |

**+15.4%.** The last row is worth noting: even with headroom available, the
vectorised control dot still does not pay for the memory it displaces -- the
same verdict round 4 reached, now tested with the constraint relaxed.

At the standard 38-token prompt the same config is 5.094 vs 5.166, **-1.4%**:
the cache already hits 100% there, so a bigger one buys nothing and the extra
slots cost a little. The gain is entirely in the regime where the cache binds.

### Left opt-in

`K3_FREE_HOST=1`, default off. It passed 6/6 with 0 ranges skipped, and it fails
loudly by construction rather than silently -- but "loudly" still means a
SIGSEGV, and a shape that only some future prompt reaches on the CPU would take
the server down. Recommended for long-context serving together with EGB=100;
not defaulted.

### Registering the other three mirror sites

The first sweep returned 9.61 GB of 16.06 with **0 skipped and 0 unaligned** --
and those zeros were the tell. Nothing was being rejected, so the shortfall had
to be ranges the registry never saw. `coli_k3_devmirror` is called from four
places and only one was instrumented.

    before   1248 ranges   9.61 GB   0 skipped     0 unaligned
    after    1700 ranges  14.47 GB  120 skipped   38 unaligned

90% of the mirror now comes back. The remainder is the router's derived int8
copies, whose f32 original is deliberately left registered because the exact
pass reads ~29 of 896 rows on the host.

**The 120 skipped ranges matter more than the extra 4.86 GB.** They are the
first evidence the tripwire fires at all: with one site instrumented, every
registered tensor happened to be GPU-only, so "0 skipped" could equally have
meant the marking was broken. 120 mirrored tensors *are* read on the host and
are correctly excluded -- exactly the ranges that would have become
plausible-looking garbage under a naive `free()`. Still 6/6 byte-identical.

### Expert cache, at ctx 1963 (warm2)

| EGB | slots/layer | tok/s | vs baseline |
|---|---|---|---|
| 88 (no reclaim) | 185 | 3.879 | — |
| 100 + reclaim | 210 | 4.377 | +12.8% |
| **110 + reclaim** | **231** | **4.614** | **+19.0%** |

Still climbing at 110. EGB=100 OOM-killed the ranks before the reclaim existed.

---

## Round 7: iteration speed, and what it immediately caught

Measuring the harness itself: at ctx 1963 one A/B arm costs 1366 s, of which
**81% is warmup that gets discarded**.

| stage | s | share | |
|---|---|---|---|
| model load | 146 | 11% | repeated per arm |
| cold | 455 | 33% | fills expert cache — discarded |
| warm1 | 508 | 37% | *still* filling at EGB=110 — discarded |
| **warm2** | **257** | **19%** | the only number used |

- [x] **Live knobs + one engine per sweep.** The `SET` command existed but
      reported `live=0` for nearly everything, because the knobs were
      function-local `static int x = -1; if (x<0) x = getenv(...)` latches.
      Hoisted to file scope behind `coli_k3_set_knob`, and the driver gained
      `ARMS="KNOB=V,KNOB=V,..."` with `PASSES=N`, running every arm against one
      loaded engine.

      **53 s per measurement against 253 s — 4.8x** — and arms can be
      interleaved, which controls for drift instead of hoping it cancels.

      `K3_EXPERT_GB` is deliberately *not* live: the cache is sized at init. The
      driver prints a WARNING when a knob reports `live=0` rather than silently
      measuring the same configuration twice, which is the failure this whole
      mechanism could otherwise introduce.

### It found a false positive in its first run

Re-measuring `K3_DENSE_ILP` interleaved, four passes per arm:

    ILP=0   5.238 5.231 5.256 5.256   mean 5.245
    ILP=1   5.237 5.232 5.256 5.264   mean 5.247   +0.04%

The committed claim was **+1.0%**, from one pass per restart. Within-arm spread
here is 0.48%, and the run shows a clear upward drift across its own 8 passes
(5.23 early, 5.26 late) — so the original comparison was reading that drift.
**The ILP kernel is neutral, not a 1% win.** It stays because it is bit-exact
and costs nothing, not because it is faster.

That is the real argument for this change: two rounds ago the FDOT A/B flipped
sign between orderings and cost two extra full A/Bs to diagnose. Interleaved
passes would have shown it immediately.

### Re-measuring earlier claims with interleaved passes

The new harness exists to catch what single-pass-per-restart comparisons miss.
Run against the claims closest to the noise floor:

| claim | as committed | interleaved, 4+ passes/arm | verdict |
|---|---|---|---|
| `K3_DENSE_ILP` | +1.0% | **+0.04%** | neutral |
| `K3_AR_PERSIST` | +1.4% | **+0.56%** | real, a third the size |
| `K3_DENSE_RTHRESH` 4096 vs 0 | (untested since old code) | **+0.13%** | wash |

Two of three inflated. Both original numbers came from one measurement per
restart with a within-arm spread of ~0.5%, so they were reading drift as signal.
`AR_PERSIST` survives -- 3 of 4 of its passes beat every control pass -- but at
0.56%, not 1.4%. The mechanism behind it (arwork fell 3.248 -> 2.733 because a
per-call thread delayed the collective's *start*) was measured directly and is
not in doubt; only the end-to-end magnitude was overstated.

RTHRESH being a wash also retires the multi-row short-row kernel as a tuning
knob: 0, 4096 and 65536 are indistinguishable.

**Absolute numbers still drift between runs** (5.08-5.18 in one sweep, 5.25-5.29
in the next, same binary and config). Only interleaved within-run comparisons
mean anything.

## The stated open list, finally closed

- [x] **lm_head sharded + argmax allreduce** — was already implemented
      (`kimi_k3.c:2499`) before this list was written.
- [x] **4-wide consecutive fold** — already implemented and the default
      (`K3_DENSE_I4W=4`); W=8 measured slower.
- [x] **`K3_DENSE_DEV_GB` vs the 64 MB cap** — neither binds. The cap became
      2 GB when the 4-wide fold made mirroring pay for lm_head. The real finding
      was that the harness was passing `DEVGB=0`, disabling mirroring entirely
      and costing 9.6%.
- [x] **float4 shared loads for the w2 expert kernel** — implemented, and
      **unvalidated by construction**: this cluster runs the 1-bit store, so
      `k3_w2_*` never executes and no A/B can price it. The change is bit-exact
      (one byte codes four consecutive activations, so the four scalar reads
      were always one float4; `xs` is 128-byte aligned). Verified only that it
      compiles under CUDA 13 and leaves the live 1-bit path byte-identical, 6/6.
      The bank-conflict half of the problem is deliberately **not** addressed:
      lane L reads at stride 64 floats so all lanes hit one bank, and fixing it
      needs `K3_W1_SHSTRIDE`-style padding plus a new store loop — not something
      to land blind on a path that cannot be measured.

---

## Round 8: long context is one term, and that term is at the memory roofline

Profiling the best long-context config (EGB=110 + reclaim, 4.60 tok/s at
ctx 1963) against the 38-token profile, s/100 decode tokens:

| term | 38 tok | 1963 tok | delta |
|---|---|---|---|
| **matt** | 0.735 | **2.935** | **+2.20** |
| arwork | 2.733 | 3.012 | +0.28 |
| khead | 0.959 | 1.086 | +0.13 |
| shared | 3.525 | 3.543 | +0.02 |
| kproj | 3.179 | 3.173 | -0.01 |
| ctlwork | 2.716 | 2.674 | -0.04 |

Total decode goes 19.2 -> 21.7 s and **matt accounts for 2.20 of the 2.5**.
Long context is one term. Everything else is flat over a 52x context increase.

### It is not thread-starved

matt does 119 GMAC per 100 tokens and takes 2.935 s = **40 GMAC/s**, against
19.86 GMAC/s measured on one X925 core -- apparently 4x of headroom at 8
threads. Testing that directly, at ctx 1963:

    OMPT=8    4.602  4.589    mean 4.596
    OMPT=16   4.270  4.311    mean 4.291    -6.6%

More threads is *worse*. The box is 10x Cortex-X925 at 3.9 GHz plus 10x A725 at
2.808 GHz, so past ~10 threads `schedule(static)` starts handing equal shares to
cores that are 1.39x slower, and every parallel region waits on the straggler.

### It is at the memory roofline, jointly with the GPU

    matt L traffic   40 GMAC/s x 2 B (bf16)   =  80 GB/s
    dense GPU path   measured on kproj        = 112 GB/s
                                                -------
                                                192 GB/s   of 235 achievable, 82%

That closes the loop on three rounds of evidence. The GPU dense path runs at 112
of 234.5 GB/s not because of the kernel but because the CPU is on the same
LPDDR5X; the CPU's biggest consumer is matt; and matt in turn cannot go faster
because the pair is near the pool's limit. This is also why FDOT's CPU-side win
showed up as `kproj` falling 12% -- the bandwidth it stopped using went straight
to the GPU.

### What that leaves

Only **moving fewer bytes** helps now, and matt's bytes are the KV cache.
`kvq` is bf16; the loader comment already notes the latents are post-rmsnorm
(`kva_ln`) so their range is normalised for fp8. Halving that 80 GB/s would free
~40 GB/s for the GPU as well as speeding matt directly -- the only remaining
change with two-sided payoff.

- [ ] **fp8 KV cache.** Not bit-exact; needs the PCC >= 0.999 gate. Note the
      existing `kv_enc` already handles Inf/NaN explicitly and rounds to
      nearest-even, so an fp8 variant must do the same rather than truncate.

---

## Round 9: the KV width is already optimal, and bf16 is why

Round 8 named fp8/int8 KV as "the only remaining change with two-sided payoff":
matt is bandwidth-bound on the latent cache, so halving its bytes should speed
matt *and* hand bandwidth back to the GPU. The extrapolation put it at
4.94-5.01 tok/s at ctx 1963, i.e. exactly on target.

It is wrong, and the reason is instruction count rather than bytes.

| cache | bytes/el | matt (s/100 tok) | decode cost |
|---|---|---|---|
| int8 + per-token scale | 1 | **4.157** | `vmovl_s8` → `vmovl_s16` → `vcvtq_f32_s32` = 3 instr / 4 elems |
| **bf16 (shipped)** | 2 | **2.935** | `vshll_n_u16(v,16)` + free reinterpret = **1 instr / 4 elems** |
| fp32 (probe) | 4 | **7.272** | direct load = 0 instr, but 2x the bytes |

int8 halves the bytes and is 42% SLOWER. fp32 removes the decode entirely and is
148% slower. **bf16 sits at the minimum of both curves** — it is the only width
here whose decode is a single instruction, because bf16→f32 is exactly a shift.

That also kills fp8 without needing to build it: an E4M3 bit layout costs ~10
NEON ops per four elements (mask, shift, bias the exponent, re-assemble, then
patch zero/NaN), an order more than bf16 and worse than the int8 that already
lost.

So matt is not purely bandwidth-bound as round 8 concluded from the fp32 probe
alone. It sits where bytes and decode ops both bind, and the current format is
at that optimum. The probe was real — doubling bytes did cost 2.48x — but one
point in one direction does not locate a minimum, and int8 was the control that
should have been run before drawing the conclusion.

**The KV-width avenue is closed.** Long context stays at 4.60 tok/s, matt stays
the single term that grows with context, and nothing in the cache format is left
to give.

The int8 path is kept behind `-DK3_KV_INT8` (with the per-token scale, the
`ldot_`/`laxpy_` helpers and the head-batched path all converted) so the
measurement can be repeated rather than re-derived.

---

## Round 10: BFDOT is 1.9x in a microbenchmark and -25% in the engine

These cores advertise `bf16` and `svebf16`, so `BFDOT` is available: it consumes
bf16 operands straight into f32 accumulators, eight elements per instruction,
with no widening at all. Against the shipped widen+fma on one X925 core
(nt=2013, kvl=512, 24 heads):

    widen+fma   1.27 ms/pass   19.51 GMAC/s
    BFDOT       0.66 ms/pass   37.23 GMAC/s    1.9x

Score PCC 0.999998949 against the shipped path (both operands must be bf16, so
the absorbed query is rounded once per head).

In the engine, interleaved at ctx 1963: **4.610 -> 3.467 tok/s, -25%.**

`-mcpu=native` does not imply `+bf16` on gcc 13 even though `/proc/cpuinfo`
lists it — `vbfdotq_f32` fails to inline with "target specific option mismatch".
Only `-march=armv9-a+bf16` compiles it. The microbenchmark built its whole file
that way, so `bfdot_` **inlined**. The engine instead carries a function-level
`__attribute__((target("+bf16")))`, which keeps `-mcpu=native` for everything
else but makes the function **un-inlinable** — turning 116 M row-dots per 100
decode tokens into real calls across a target boundary.

**The 1.9x was real and unattainable as implemented.** A microbenchmark that
compiles differently from the code it stands in measures a different program.
This is the third time this project has been caught by measuring the wrong
thing (nsys wall-clock vs kernel instances; ncu replay swamping a 1% change;
now inlining), and the pattern is the same: the probe has to share the build of
the thing it predicts.

- [ ] **Global `-march=armv9-a+bf16`.** Would let BFDOT inline and is the only
      way to collect the 1.9%. Untested, and not obviously a win: it drops
      `-mcpu=native` scheduling for the X925 and the `+i8mm` SMMLA path
      `colibri.c` relies on. That is a whole-binary change to buy one loop, and
      it needs its own A/B rather than an assumption.

Left off (`K3_BFDOT=1`), gated on `HWCAP2_BF16` at startup because the target
attribute emits BFDOT regardless of the base arch and a host without it would
SIGILL rather than fall back.

### Global `-march=armv9-a+bf16`: the arch change is free, BFDOT still loses

Tested it. The build accepts it on spark1 (not on the local host — spark1's
toolchain is authoritative), the deployed binary contains BFDOT, and it inlines.

**The arch change costs nothing.** BFDOT=0 on the armv9 build measures 4.616
tok/s against 4.610 on the `-march=native` build — so dropping native tuning and
`+i8mm` is not the concern I flagged.

**BFDOT itself now speeds up what it touches** — matt 2.967 → 2.576, −13% —
and still loses 18% end to end. The profile says why:

| term | BFDOT=0 | BFDOT=1 | delta |
|---|---|---|---|
| **matt** | 2.967 | 2.576 | **−0.391** |
| arwork | 2.760 | 6.030 | +3.270 |
| netmoe | 0.270 | 3.535 | +3.265 |
| expert | 1.339 | 2.462 | +1.123 |
| load | 0.000 | 1.085 | +1.085 |

Expert hit rate goes 100.0% → 99.8%. Rounding the absorbed query to bf16 changes
attention scores, which changes the residual stream, which changes **which
experts the router picks** — the identical mechanism that blocked the vectorised
control dot in round 4. A different expert set misses a cache tuned to the old
one, streaming appears, and ranks then arrive at the collective at different
times, which is the +3.3 s in `arwork`/`netmoe`.

### A caveat this exposes on the EGB=110 result

The +19% from EGB=110 depends on a **100.0%** expert hit rate, and the cliff off
it is steep: a 0.2 percentage-point drop cost 1.085 s of streaming and 3.3 s of
collective skew. The harness runs the same prompt for cold, settle and every
measured pass, which is the most cache-favourable schedule possible. That number
is real for repeated-prompt serving and should not be read as a general +19%.

It also means **any** numerically-perturbing change is now penalised twice at
long context: once on its own merits, and again for moving expert selection off
a perfectly warmed cache. Round 4's FDOT and this both died that way.
