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
| `kconv` | 1.10 | CPU, `expf`-bound |
| `matt` | 0.84 | CPU |
| `khead` | 0.61 | CPU |
| `topk` | 0.44 | CPU |
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
- [x] ~~More control threads (18) / dynamic schedule~~ — bandwidth-bound
- [x] ~~`OMP_NUM_THREADS`=12, `GOMP_SPINCOUNT`=0~~ — both worse
- [x] ~~Bounded spin before `poll()` in the collective~~ — inside noise
- [x] ~~Device mirrors for expert slots~~ — 145.7 vs 148.3 GB/s, no effect
- [x] ~~CUDA graphs~~ — launches are 1.3% of wall; topology is data-dependent

## Not bit-exact — REQUIRES `tools/k3_quality.py`

Ordered by risk-adjusted value: activation-only ulp differences first, routing
last. Each must be landed and gated SEPARATELY so a failure is attributable.

- [ ] **1. SiTU fused into the shared-expert GPU epilogue** (~10 ms/token)
      Risk: LOW. `tanhf`/`expf` ulp only, in activations. The routed-expert
      kernel already fuses SiTU on the GPU, so the construction exists.
- [ ] **2. KDA conv + SiLU on the GPU** (~8 ms/token)
      Risk: LOW-MED. `expf` ulp. Small work per launch, so it must be fused
      with a neighbour rather than launched alone.
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
