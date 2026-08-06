/* Kimi K3 inference engine in pure C — sibling of colibri.c (GLM-5.2) / olmoe.c /
 * inkling.c, sharing st.h / json.h / tok.h / quant.h.
 *
 * Architecture (2.8T total / 104B active, 93 layers, hidden 7168):
 *   - Hybrid attention: 69 KDA (Kimi Delta Attention, linear/recurrent) +
 *     24 gated MLA layers (every 4th, plus the final layer). MLA is NoPE —
 *     no positional encoding anywhere in the model; position lives in KDA's
 *     decay/conv. Both attention types carry a full-rank sigmoid output gate.
 *   - AttnRes (Attention Residuals) REPLACE the plain residual stream: a
 *     running prefix_sum plus block snapshots (layers 0,12,24,...,84), mixed
 *     by softmax twice per layer (before attention, before MLP) and once at
 *     the end. Weights: per-layer {self_attention,mlp}_res_{norm,proj}.
 *   - Stable LatentMoE: router (sigmoid + e_score_correction_bias, top-16 of
 *     896, renormalized raw scores), shared latent down/up projections
 *     (7168<->3584), per-expert GLU in the 3584 latent with moe_inter 3072,
 *     RMSNorm on the aggregate, plus 2 fused shared experts (inter 6144) at
 *     full width. Activation is SiTU-GLU:
 *         b1*tanh(g/b1)*sigmoid(g) * b2*tanh(u/b2),  b1=4, b2=25.
 *   - Routed experts are NATIVE MXFP4 (QAT; e2m1 nibbles + ue8m0 scale per
 *     32, compressed-tensors "mxfp4-pack-quantized") and are streamed
 *     straight from the original HF shards — never re-encoded, never
 *     converted. Everything else is BF16 in the checkpoint and quantized at
 *     LOAD TIME into RAM (int8 per-row / int4-g64 / f32, see K3_*BITS).
 *
 * KDA per-head recurrence (head dim 128, 96 heads; fla fused_recurrent_kda):
 *     q,k,v = SiLU(ShortConv4(W{q,k,v} x));  q,k L2-normalized (eps 1e-6
 *     inside the sqrt), q *= 128^-0.5
 *     z = W_fb(W_fa x) + dt_bias            (per channel, dt_bias[12288])
 *     gk = gmin * sigmoid(exp(A_log[h]) * z),  gmin=-5;  alpha = exp(gk)
 *     S = (I - beta k k^T) Diag(alpha) S + beta k v^T,  beta = sigmoid(W_b x)
 *     o = S^T q;  out = W_o [ sigmoid(W_g x) * RMSNorm_head(o) ]
 *   (checkpoint A_log is [128] = per-head [96] zero-padded, first 96 used)
 *
 * Model dir = the HF snapshot (config.json + model-*-of-000096.safetensors).
 * tokenizer.json is synthesized once by tools/k3_tokenizer.py (the HF repo
 * ships only tiktoken.model); without it the engine still runs on raw ids.
 *
 * ENV:
 *   K3_BITS=4|8|32       load-time quant of KDA/latent/shared/dense (default 4)
 *   K3_MLA_BITS=8|4|32   MLA projections (default 8)
 *   K3_HEAD_BITS=8|4|32  lm_head (default 8)
 *   K3_EXPERT_GB=N       routed-expert LRU cache budget (default 8)
 *   K3_DIRECT=0|1        O_DIRECT expert reads (default 1; buffered fallback)
 *   K3_IDOT=0|1          int8-activation expert matmuls (default 1; 0 = float)
 *   K3_PIPE=0|1          overlap expert loads with compute (default 1)
 *   K3_LOAD_THREADS=N    loader threads for K3_PIPE (default 4)
 *   K3_TOPP=F            keep routed experts to cumulative weight F (0 = off)
 *   K3_DENSE_GPU=0|1     zero-copy dense decode GEMV (default 1)
 *   K3_DENSE_EXACT=0|1   stock-order exact reduction (default 1)
 *   K3_CHUNK=N           prefill chunk size (default 32; 1 = token-at-a-time)
 *   K3_THINK=0|1         chat mode: open the think channel (default 1)
 *   K3_LAYERS=N          truncate to first N layers (validation; skips head)
 *   K3_TRACE=path        dump f32 hidden state after every layer (validation)
 *   K3_LOGITS=path       dump f32 logits per PREFILL position (teacher-forced
 *                        bit-width comparisons; use with --ngen 0)
 *   K3_MAXT=N            KV/context capacity (default prompt+ngen)
 *   COLI_TEMP=F          0 = greedy (default), else softmax temperature
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)
#include <sys/resource.h>
#include <sys/select.h>
#include <unistd.h>
#endif
#include <pthread.h>
#include <sys/mman.h>
#include <stdatomic.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "st.h"
#include "tok.h"
#include "quant.h"
#include "k3_net.h"
#ifdef COLI_CUDA
#include "backend_cuda.h"
#include "backend_cuda_k3.h"
#endif

/* MLA latent-cache element. Only the 24 non-KDA layers cache anything (the 69
 * KDA layers carry a fixed 0.40 GB recurrent state), and MLA already compresses
 * to a 576-float latent -- but the cache is still scanned end to end for every
 * token, so this width sets BOTH the memory ceiling at long context and the
 * bandwidth of that scan:
 *
 *        ctx     fp32     bf16      fp8
 *       256K   13.5 GB   6.8 GB   3.4 GB
 *         1M   54.0 GB  27.0 GB  13.5 GB
 *
 * bf16 is a pure truncation of fp32 -- no scale, no clamp, no calibration --
 * so it cannot introduce the range bugs a real 8-bit format would, and the
 * conversion is one shift. The latents are post-rmsnorm (kva_ln), so their
 * range is already normalised if we later go to fp8. Build with -DK3_KV_FP32
 * to A/B against the original fp32 cache -- compile-time, so the inner loops
 * carry no branch either way. */
#ifdef K3_KV_FP32
typedef float kvq;
static inline kvq   kv_enc(float f){ return f; }
static inline float kv_dec(kvq h)  { return h; }
#else
typedef uint16_t kvq;
static inline kvq kv_enc(float f){
    union{float f;uint32_t u;}b; b.f=f;
    uint32_t u=b.u;
    /* Inf/NaN must survive. A blind (u+0x8000)>>16 carries out of the mantissa
     * into the exponent and sign, so 0x7fffffff -> -0, 0xffffffff -> +0 and
     * 0x7f800001 -> +Inf: an invalid value silently becomes a plausible finite
     * one, which converts a numerical failure into quietly wrong output. Keep
     * the top half verbatim and force the mantissa non-zero so a NaN stays a
     * NaN rather than decaying to Inf. */
    if((u&0x7F800000u)==0x7F800000u){
        uint16_t h=(uint16_t)(u>>16);
        if(u&0x007FFFFFu) h|=0x0040;         /* was NaN: keep it quiet-NaN */
        return h;
    }
    /* round-to-nearest-EVEN, the IEEE default: ties go to the even mantissa
     * instead of always up, so repeated store/load cycles do not drift. */
    uint32_t lsb=(u>>16)&1u;
    return (kvq)((u+0x7FFFu+lsb)>>16);
}
static inline float kv_dec(kvq h)  { union{uint32_t u;float f;}b; b.u=(uint32_t)h<<16; return b.f; }
#endif

/* ---------- bf16 KV-cache inner loops -------------------------------------
 * The MLA score and context loops walk the cache through kv_dec(): a u16 load,
 * a widen, a 16-bit shift and a type-pun before the multiply. gcc will not
 * vectorise that shape, and it is the term that grows with context --
 * matt = 0.83 + 0.00356*T seconds per 100 decode tokens, 54% of decode at
 * T=4863. Measured on one X925 core at nt=4863, kvl=512, 24 heads:
 *     scalar/union   25.45 ms/pass    2.35 GMAC/s
 *     NEON widen      2.99 ms/pass   19.96 GMAC/s   (8.5x)
 * So matt was compute-bound on the conversion, not bandwidth-bound on the
 * cache -- which is why it stayed flat under every cache-traffic change.
 *
 * bf16 -> f32 is exactly a shift left by 16, so vshll_n_u16(v,16) reinterpreted
 * as f32 converts four elements in one instruction. Two accumulators, the same
 * convention the int8/int4 weight kernels in quant.h already use.
 *
 * kvdot_ reorders the reduction (8 partial sums rather than one running total),
 * so it is not bit-exact -- pairwise accumulation is if anything more accurate,
 * and it gets the PCC gate. kvaxpy_ is elementwise: each acc[i] sees the same
 * adds in the same order, so that one IS bit-exact. */
static int g_kvneon = 1;                      /* K3_KVNEON=0 restores the scalar loops */
static int g_fdot   = 0;                      /* K3_FDOT=1: faster, but see the note below */
static int g_mattb  = 0;                      /* head-batched MLA: correct but slower, see below */
#if defined(__ARM_NEON) && !defined(K3_KV_FP32)
static inline float kvdot_(const float *q, const kvq *L, int n){
    float32x4_t a0=vdupq_n_f32(0), a1=vdupq_n_f32(0);
    int i=0;
    for(;i+7<n;i+=8){
        uint16x8_t v=vld1q_u16(L+i);
        a0=vfmaq_f32(a0, vld1q_f32(q+i),   vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v),16)));
        a1=vfmaq_f32(a1, vld1q_f32(q+i+4), vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v),16)));
    }
    float s=vaddvq_f32(a0)+vaddvq_f32(a1);
    for(;i<n;i++) s+=q[i]*kv_dec(L[i]);
    return s;
}
static inline void kvaxpy_(float *acc, const kvq *L, float s, int n){
    float32x4_t sv=vdupq_n_f32(s);
    int i=0;
    for(;i+7<n;i+=8){
        uint16x8_t v=vld1q_u16(L+i);
        vst1q_f32(acc+i,   vfmaq_f32(vld1q_f32(acc+i),   sv,
                  vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v),16))));
        vst1q_f32(acc+i+4, vfmaq_f32(vld1q_f32(acc+i+4), sv,
                  vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v),16))));
    }
    for(;i<n;i++) acc[i]+=s*kv_dec(L[i]);
}
#else
static inline float kvdot_(const float *q, const kvq *L, int n){
    float s=0; for(int i=0;i<n;i++) s+=q[i]*kv_dec(L[i]); return s;
}
static inline void kvaxpy_(float *acc, const kvq *L, float s, int n){
    for(int i=0;i<n;i++) acc[i]+=s*kv_dec(L[i]);
}
#endif

/* Plain fp32 dot. The KDA control path stores fa/fb/bp unquantised and reduces
 * them with `for(i) v+=x[i]*w[i]`, which gcc cannot vectorise: FP addition is
 * not associative and this build does not pass -ffast-math. Measured on one
 * X925 core over 152 rows of I=7168:
 *     scalar   0.557 ms/pass    1.96 GMAC/s
 *     NEON     0.055 ms/pass   19.86 GMAC/s   (10.1x)
 * Four accumulators, so the reduction order changes -- same category as kvdot_.
 *
 * OFF BY DEFAULT. It does speed up what it touches, consistently and in the
 * predicted direction (standard prompt, s/100 tokens):
 *
 *     ctlwork  2.72 -> 2.19   ctljoin 0.89 -> 0.44
 *     kproj    3.22 -> 2.85   attn   10.08 -> 9.55
 *
 * kproj falling 12% for a CPU-side change is independent confirmation that the
 * GPU dense path is contending with the CPU for LPDDR5X: less CPU traffic, more
 * GPU bandwidth. But end to end it is a wash (5.146 vs 5.163 mean over two
 * A/Bs with the arms in both orders), and at ctx 1963 it OOM-kills all four
 * ranks.
 *
 * The reason is not that it allocates anything. fa/fb/bp produce the KDA gate
 * and beta, so a different summation order changes the recurrence, the residual
 * stream, and therefore which experts the router picks. A different expert set
 * has a different cache footprint: RSS came out at exactly 72.47 GB with it on
 * and exactly 70.94 GB with it off, in both orderings -- deterministic, not
 * timing. At ctx 1963 the scalar arm already sits at 98.07 GB of 121, so a
 * 1.5 GB shift is fatal.
 *
 * So this is blocked on memory headroom rather than on being wrong: it needs
 * EGB retuned down before it can be turned on, and that trade has to be
 * measured (the 80 -> 88 expert-cache increase was itself worth +4.6%).
 * K3_FDOT=1 to re-run the A/B. */
#if defined(__ARM_NEON)
static inline float fdot_(const float *x, const float *w, int n){
    float32x4_t a0=vdupq_n_f32(0),a1=vdupq_n_f32(0),a2=vdupq_n_f32(0),a3=vdupq_n_f32(0);
    int i=0;
    for(;i+15<n;i+=16){
        a0=vfmaq_f32(a0,vld1q_f32(x+i),   vld1q_f32(w+i));
        a1=vfmaq_f32(a1,vld1q_f32(x+i+4), vld1q_f32(w+i+4));
        a2=vfmaq_f32(a2,vld1q_f32(x+i+8), vld1q_f32(w+i+8));
        a3=vfmaq_f32(a3,vld1q_f32(x+i+12),vld1q_f32(w+i+12));
    }
    float v=vaddvq_f32(a0)+vaddvq_f32(a1)+vaddvq_f32(a2)+vaddvq_f32(a3);
    for(;i<n;i++) v+=x[i]*w[i];
    return v;
}
#else
static inline float fdot_(const float *x, const float *w, int n){
    float v=0; for(int i=0;i<n;i++) v+=x[i]*w[i]; return v;
}
#endif


/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, vocab, first_dense, dense_inter;
    /* MLA */
    int n_heads, q_lora, kv_lora, qk_nope, qk_rope, qk_head, v_head;
    float attn_scale;
    /* KDA */
    int kda_heads, kda_hd, kda_proj, conv_k;
    float gate_lb;
    /* MoE */
    int n_experts, topk, moe_inter, latent, n_shared;
    float situ_b1, situ_b2;
    /* AttnRes */
    int res_bs;
    float eps;
    int8_t is_kda[128];
    int bos, eos[8], n_eos;
} Cfg;

/* ---------- RAM-resident weight, quantized at load ---------- */
typedef struct { int fmt; float *f; int8_t *q8; uint8_t *q4; float *s; int O, I, gs;
#ifdef COLI_CUDA
    /* Device mirror of this tensor. Allocated lazily by the first w_matmul on
     * the serial path (see the eligibility note there). Zeroed by the calloc
     * of the owning Layer, so a CPU-only build and an un-placed tensor look
     * identical. */
    ColiCudaTensor *cuda; int cuda_device; int8_t cuda_failed, cuda_placed;
    int8_t k3_reg, k3_dense_off;      /* zero-copy dense path: registered / declined */
    int8_t k3_htouched;               /* this tensor's HOST blob was dereferenced */
    int8_t fmt_reported;              /* census: shape logged once */
    int8_t gs_reported;
    int8_t pf_reported;
    /* Optional device copies for the exact dense kernel. Host-registered pages
     * reach the GPU through the SMMU at 4 KB granularity; cudaMalloc'd memory
     * uses large pages, which measured up to 1.65x on the SAME kernel. Null
     * when the budget declined -- the zero-copy pointers are still valid. */
    void *k3_dw, *k3_ds; int8_t k3_dev_tried;
#endif
} W;

typedef struct {                          /* KDA layer */
    W q, k, v, o, g;
    /* Head-sharded views, built ONCE. Building them per call re-uploaded the
     * slice to the GPU every time (tensor count 1021 -> 1466, budget pinned at
     * 40 GB, attn 13.3 -> 19.9 s) because each fresh W carries a null cuda
     * handle. The view must outlive the call for the placement to be reused. */
    W qs, ks, vs, gs, os, qkvg; int sh_ready, fuse;  /* qkvg = fused q|k|v|g */
    float *conv_q, *conv_k, *conv_v;      /* [proj*4] depthwise taps, oldest first */
    W fa, fb;                             /* decay low-rank, f32 [hd,hidden] [proj,hd] */
    W bp;                                 /* beta proj f32 [heads,hidden] */
    float *dt, *A, *onw;                  /* dt_bias[proj], exp(A_log)[heads], o_norm[hd] */
} Kda;

typedef struct {                          /* gated MLA layer */
    W qa, qb, kva, kvb, o, g;
    /* Head-sharded views, same scheme as KDA: q_b/g rows by head, o columns.
     * q_a and kv_a stay replicated -- they produce the shared latents every
     * head consumes, and they are small (5.5 and 2 MB vs 44 MB for o/g). */
    W qbs, gs_, os; int sh_ready;
    float *qa_ln, *kva_ln;
} Mla;

typedef struct {                          /* LatentMoE */
    float *router, *rbias, *lat_norm;     /* [E,hidden] f32, [E], [latent] */
    W router_w;
    int8_t *r_q8; float *r_s8;      /* int8 search copy for the two-pass router */
    void *r_dq; float *r_ds, *r_db; int r_reg;                           /* same f32 buffer, so the router goes
                                           * through w_matmul and lands on the GPU
                                           * instead of quant.h's CPU f32 path */
    W lat_down, lat_up, sh_gate, sh_up, sh_down;
} Moe;

typedef struct {
    int kda, sparse;
    Kda a; Mla m; Moe moe;
    W d_gate, d_up, d_down;               /* dense layer only */
    float *in_ln, *post_ln;
    float *attn_sw, *mlp_sw;              /* AttnRes score weights: norm.w * proj.w */
} Layer;

/* ---------- routed-expert streaming (native MXFP4 from the HF shards) ---- */
typedef struct { int fd[6]; int64_t off[6]; int contig; } ERef;  /* w1p w1s w2p w2s w3p w3s */
typedef struct { int eid; uint8_t *buf, *base; uint64_t used; } Slot;
                          /* base = 4K-aligned allocation (O_DIRECT target);
                           * buf = expert data view inside it (= base + off%4K) */
typedef struct { Slot *s; int n, cap; } LCache;

typedef struct {
    Cfg c;
    shards S;
    char pfx[40];                         /* "language_model." or "" */
    Layer *L;
    float *final_norm, *out_sw;
    W lm_head;
    /* Row slice of lm_head this rank owns when the head is sharded. Cached:
     * building the view per token would reset its registration and device
     * mirror every time. */
    W lm_head_shard; int lm_head_shard_ready;
    int head_greedy;                  /* caller's temp<=0, set before forward */
    int head_tok;                     /* set when the sharded path ran; else -1 */
    int has_head;
    Slot ws[64];                          /* working set: parallel loads land here,
                                           * then swap into the layer LRU */
    /* KDA state */
    float **kstate;                       /* [layer] -> [heads*hd*hd], S[k][v] */
    float **cwq, **cwk, **cwv;    /* conv windows [proj*conv_k], oldest first */
    int8_t cwh[128];              /* rotating write head into those windows */
    /* MLA cache */
    kvq **Lc, **Rc; int max_t;
    /* experts */
    ERef *eref;                           /* [n_layers][n_experts] (dense rows zeroed) */
    LCache *ecache;
    int64_t e_w1p, e_w1s, e_w2p, e_w2s, e_slot;
    int w2_fd, w2_dfd, w1_mode;           /* expert store: buffered + O_DIRECT, 1-bit? */
    uint32_t *route_hist;                 /* K3_ROUTE_STATS: [layer][expert] counts */
    float *hz_batch;                      /* [64][latent] batched-expert results */
    uint64_t clock, hits, miss, ebytes;
    double t_attn, t_moe, t_eload, t_head;
    /* fine-grained moe breakdown (K3_PROFILE=1): coarse t_moe hid that
     * routed experts were only ~1/5 of it, which sent one optimisation
     * pass at the wrong term. */
    double t_router, t_topk, t_latent, t_shared, t_expert, t_rnorm;
    double t_ekernel;                     /* GPU call only, inside t_expert */
    double t_kproj, t_kconv, t_khead, t_kout;  /* kda_forward breakdown */
    double t_ctl, t_ctljoin;                   /* control work vs join wait */
    double t_ctlstart;                         /* create -> work actually begins */
    double t_net_kda, t_net_mla, t_net_moe;     /* EXPOSED collective time by site */
    double t_resmix;                           /* AttnRes softmax mix (CPU) */
    double t_mproj, t_mcache, t_matt, t_mout;   /* mla_forward breakdown */
    uint64_t n_ekernel;
    FILE *trace;
} Model;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#if defined(__APPLE__)
    return r.ru_maxrss/(1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss/(1024.0*1024.0);
#endif
}
static float *falloc(int64_t n){ float *p=malloc((size_t)n*sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} return p; }
static float *fcalloc(int64_t n){ float *p=calloc((size_t)n,sizeof(float)); if(!p){fprintf(stderr,"OOM %lld floats\n",(long long)n);exit(1);} return p; }
static inline float sigmoidf_(float x){ return 1.f/(1.f+expf(-x)); }
static inline float siluf_(float x){ return x/(1.f+expf(-x)); }
static void softmax_(float *x, int n){ float m=x[0]; for(int i=1;i<n;i++) if(x[i]>m)m=x[i];
    float s=0; for(int i=0;i<n;i++){ x[i]=expf(x[i]-m); s+=x[i]; } for(int i=0;i<n;i++) x[i]/=s; }
static void rmsnorm_(float *out, const float *x, const float *w, int D, float eps){
    double ms=0; for(int i=0;i<D;i++) ms+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ms/D)+eps);
    for(int i=0;i<D;i++) out[i]=x[i]*r*w[i];
}

/* B=1 KDA control projections. f_a and beta are independent and share one
 * OpenMP team; f_b follows after the team's implicit barrier. Keeping the dot
 * loop text identical to quant.h's f32 matmul preserves its strict accumulation
 * order while removing two fork/join pairs per KDA layer. The whole job is also
 * independent of q/k/v/g and may run on a pthread underneath their GPU calls. */
typedef struct {
    const Kda *a; const float *x;
    float *t1, *graw, *braw;
    int hidden, hd, p0, pn, h0, hn;
} KdaCtrlJob;

/* Worker count for the control region ONLY.
 *
 * This runs on a pthread created fresh each KDA layer, so its OpenMP team is
 * built from scratch 69 times per token -- and team construction scales with
 * the worker count while the work itself is bandwidth-bound. That is why 18 and
 * 12 workers both measured SLOWER here than 10, which looks paradoxical until
 * you read it as creation cost rather than contention. Fewer, larger slices
 * therefore beat more workers, and there is a clear optimum:
 *
 *   workers  4      6      8      10 (ambient)
 *   tok/s    3.63   4.28   4.20   4.08
 *   ctljoin  1.181  1.006  1.541  2.183
 *
 * Six is the peak -- below that the region is genuinely too narrow and the
 * work itself dominates. Bit-exact either way: each output row is still summed
 * sequentially by a single thread. */
static int k3_ctl_threads(void){
    static int n = -1;
    if (n < 0) { const char *e = getenv("K3_CTL_THREADS"); n = e ? atoi(e) : 6; if (n < 1) n = 6; }
    return n;
}

static void kda_control_b1(KdaCtrlJob *j){
    int nt_ = k3_ctl_threads();
    #pragma omp parallel num_threads(nt_ > 0 ? nt_ : omp_get_max_threads())
    {
        #pragma omp for schedule(static)
        for(int o=0;o<j->hd+j->hn;o++){
            const float *w; const float *x; float *dst; int I;
            if(o<j->hd){
                w=j->a->fa.f+(int64_t)o*j->hidden; x=j->x;
                dst=j->t1+o; I=j->hidden;
            } else {
                int r=o-j->hd;
                w=j->a->bp.f+(int64_t)r*j->hidden; x=j->x;
                dst=j->braw+j->h0+r; I=j->hidden;
            }
            float v;
            if(g_fdot) v=fdot_(x,w,I);
            else { v=0; for(int i=0;i<I;i++) v+=x[i]*w[i]; }
            *dst=v;
        }
        #pragma omp for schedule(static)
        for(int r=0;r<j->pn;r++){
            const float *w=j->a->fb.f+(int64_t)r*j->hd;
            float v;
            if(g_fdot) v=fdot_(j->t1,w,j->hd);
            else { v=0; for(int i=0;i<j->hd;i++) v+=j->t1[i]*w[i]; }
            j->graw[j->p0+r]=v;
        }
    }
}

static double g_ctl_secs=0, g_ctl_start=0;  /* work only / spawn-to-start latency */
static double g_ctl_spawn=0;
static void *kda_control_worker(void *p){ double a=now_s(); g_ctl_start+=a-g_ctl_spawn;
                                          kda_control_b1((KdaCtrlJob*)p);
                                          g_ctl_secs+=now_s()-a; return NULL; }

/* Same per-call-thread shape the latent allreduce had, so the same fix was
 * tried: one worker parked on a condvar instead of a pthread_create per KDA
 * layer per token (6900 per 100 decode tokens).
 *
 * MEASURED MUCH WORSE, off by default. A/B at the standard prompt:
 *
 *                  per-call   persistent
 *     decode tok/s    5.275        2.406    -54%
 *     ctlstart        0.209        1.354
 *     ctljoin         0.911        3.084
 *     arwork          2.850        9.832
 *
 * The allreduce worker wins from exactly the property this one lacks. That
 * thread blocks on sockets: it sleeps, holds no core, and competes with nothing
 * -- so keeping it alive removes a create and lets the exchange start at once.
 * kda_control_b1 is CPU-bound and runs against the OpenMP pool on the same ten
 * cores. A thread created per call is placed fresh by the scheduler on whatever
 * is free; a resident one holds a core the pool wants, and under K3_PIN the
 * contention is not resolvable. It degraded the allreduce as well (arwork 3.4x)
 * because two resident threads plus the pool oversubscribe the cluster.
 *
 * The rule this refines: persistent workers pay for I/O-bound work and cost for
 * CPU-bound work. K3_CTL_PERSIST=1 to re-run the A/B. */
static pthread_mutex_t g_ctl_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_ctl_cv = PTHREAD_COND_INITIALIZER;
static KdaCtrlJob     *g_ctl_job = NULL;
static pthread_t       g_ctl_th;
static int g_ctl_req=0, g_ctl_done=1, g_ctl_quit=0, g_ctl_started=0, g_ctl_persist=-1;
/* default 0: see the A/B above */

static void *kda_control_persist(void *unused){
    (void)unused;
    for(;;){
        pthread_mutex_lock(&g_ctl_mu);
        while(!g_ctl_req && !g_ctl_quit) pthread_cond_wait(&g_ctl_cv,&g_ctl_mu);
        if(g_ctl_quit){ pthread_mutex_unlock(&g_ctl_mu); return NULL; }
        g_ctl_req=0;
        pthread_mutex_unlock(&g_ctl_mu);
        double a=now_s(); g_ctl_start+=a-g_ctl_spawn;
        kda_control_b1(g_ctl_job);
        g_ctl_secs+=now_s()-a;
        pthread_mutex_lock(&g_ctl_mu);
        g_ctl_done=1; pthread_cond_signal(&g_ctl_cv);
        pthread_mutex_unlock(&g_ctl_mu);
    }
}

/* Returns 1 when the job was handed off and ctl_join() must be called. */
static int kda_control_submit(KdaCtrlJob *j, pthread_t *fallback){
    if(g_ctl_persist<0){ const char *e=getenv("K3_CTL_PERSIST"); g_ctl_persist=e?atoi(e):0; }
    if(!g_ctl_persist) return pthread_create(fallback,NULL,kda_control_worker,j)==0 ? 2 : 0;
    if(!g_ctl_started){
        if(pthread_create(&g_ctl_th,NULL,kda_control_persist,NULL)!=0) return 0;
        g_ctl_started=1;
    }
    g_ctl_job=j;
    pthread_mutex_lock(&g_ctl_mu);
    g_ctl_done=0; g_ctl_req=1;
    pthread_cond_signal(&g_ctl_cv);
    pthread_mutex_unlock(&g_ctl_mu);
    return 1;
}
static void kda_control_join(int mode, pthread_t fallback){
    if(mode==2){ pthread_join(fallback,NULL); return; }
    pthread_mutex_lock(&g_ctl_mu);
    while(!g_ctl_done) pthread_cond_wait(&g_ctl_cv,&g_ctl_mu);
    pthread_mutex_unlock(&g_ctl_mu);
}


/* Tensor parallelism is a property of the multi-node split, NOT of the CUDA
 * backend -- model_init, kda_forward and mla_forward all use these
 * unconditionally, so they must live OUTSIDE the COLI_CUDA guard below or the
 * CPU-only build fails to compile.
 *
 * K3_TP_ATTN=0 disables tensor-parallel attention, leaving dense replicated.
 * TP attention adds a SECOND collective per layer (186/token instead of 93).
 * At 2 nodes that was worth it (+12%, collective 1.37 ms). Measured again at 4
 * nodes with the collective at 1.95 ms, TP ON still wins (0.87 vs 0.64 tok/s):
 * the extra 93 reduces cost ~0.18 s/token against a larger sharding saving. */
static int      g_k3_tp_attn = 1;
/* Rank's head range [*h0, *h0+*hn) out of H heads. MUST match the split the
 * kda/mla forwards compute, or a rank loads one slice and multiplies another;
 * both derive it from this single formula for that reason. */
static void k3_head_shard(int H, int *h0, int *hn){
    int wsz = g_k3_tp_attn ? k3_net_world() : 1, wrk = k3_net_rank();
    if(wsz<=1){ *h0=0; *hn=H; return; }
    int per=(H+wsz-1)/wsz, a=wrk*per, b=a+per;
    if(a>H) a=H; if(b>H) b=H;
    *h0=a; *hn=b-a;
}

/* Which matmuls actually reach the exact K3 kernel. nsys says the generic
 * quant_matmul is 64% of GPU time at 1.37 ms a call against 91 us for
 * k3_dense_i4g_exactW, so the split matters more than either kernel does. */
static _Atomic long g_dp_k3=0, g_dp_off=0, g_dp_fmt=0, g_dp_par=0;
/* Which path a PREFILL (S>1) dense call actually takes. The S>1 GEMM was added
 * because the generic kernel is 15x slower per launch, but prefill barely
 * moved, so count rather than assume. */
static _Atomic long g_pf_gemm=0, g_pf_f16=0, g_pf_generic=0, g_pf_cpu=0;
/* How many routed experts this rank actually ran. Per-rank expert time is
 * reproducibly ordered 0 > 1 > 3 > 2 across runs, which is a fixed bias rather
 * than per-token variance, so the question is whether the ranks run different
 * NUMBERS of experts or the same number at different cost. */
static _Atomic long g_exp_n=0, g_exp_layers=0;
static int g_dense_census = 0;    /* K3_DENSE_CENSUS=1 */
#ifdef COLI_CUDA
/* ---------- optional GPU placement for the RESIDENT (dense) tensors ----------
 *
 * Scope: attention (KDA + MLA), the LatentMoE up/down projections, the shared
 * experts and the head — i.e. everything w_load() already quantizes into RAM.
 * The ROUTED experts are deliberately untouched: they stream from the shards as
 * native MXFP4 and never become a W, so they keep the CPU path.
 *
 * fmt is the same encoding backend_cuda.h documents (0=f32, 1=int8, 4=grouped
 * int4) because both engines share quant.h — so coli_cuda_matmul is a drop-in
 * for the dispatch below, uploading on first use and reusing after.
 *
 * GB10 caveat (upstream #653): this GPU is *integrated* — device memory and
 * host RAM are one physical pool. Placing a tensor therefore DUPLICATES it
 * rather than moving it off-RAM, and every byte spent here is a byte not
 * available to the routed-expert cache, which is the actual bottleneck. Hence
 * placement is opt-in (K3_GPUS) and hard-capped (K3_GPU_GB).
 */
static int    g_k3_cuda = 0;
/* K3_EXPERT_GPU: routed-expert matmuls on the GPU (backend_cuda_k3.cu).
 * Cleared on any failure so a run degrades to the CPU path rather than dying
 * mid-token. Declared here because k3_cuda_report() below reports the split. */
static int      g_k3_expert_gpu = 0;
static int      g_k3_ready = 0;      /* backend_cuda_k3 initialised */
/* K3_DENSE_GPU: zero-copy dense GEMV. ON by default after the exact-order
 * kernel took a 300-token 4-node run from 1.09 -> 1.21 tok/s: shared experts
 * 43.9 -> 33.3 s, latent projections 16.8 -> 13.2 s, with bit-exact standalone
 * outputs and token-identical generation. K3_DENSE_GPU=0 keeps the old
 * device-mirror path for A/B. K3_DENSE_EXACT=0 selects the older warp reduction
 * (slightly different arithmetic and therefore not suitable as the default). */
static int      g_k3_dense_gpu = 1;
static uint64_t g_k3_exp_gpu = 0, g_k3_exp_cpu = 0;
static int      g_k3_expert_batch = 0;   /* K3_EXPERT_BATCH=1; see the note at its use */
static int    g_k3_cuda_devs[COLI_CUDA_MAX_DEVICES];
static int    g_k3_cuda_ndev = 0;
static int    g_k3_cuda_rr = 0;
static int64_t g_k3_gpu_left = 0;          /* bytes still placeable (K3_GPU_GB) */
static int64_t g_k3_gpu_used = 0;
static int    g_k3_cuda_nfail = 0;

static int64_t w_bytes(const W *w){
    int64_t O=w->O, I=w->I;
    if(w->fmt==0) return O*I*4;
    if(w->fmt==1) return O*I + O*4;
    { int64_t rb=(I+1)/2, ng=(I+w->gs-1)/w->gs; return O*rb + O*ng*4; }
}

/* The weight blob coli_cuda_* expects for this format. */
static const void *w_blob(const W *w){
    return w->fmt==0 ? (const void*)w->f
         : w->fmt==1 ? (const void*)w->q8 : (const void*)w->q4;   /* fmt 2 and 4 both live in q4 */
}

/* K3_GPUS="0" or "0,1" places resident tensors on those CUDA ordinals; unset
 * keeps the engine CPU-only. K3_GPU_GB caps the placement (default 8 GB).
 * Called BEFORE the shards are mapped so a bad device or a driver problem is
 * reported in a second rather than after mmap'ing 1.4 TB of checkpoint. */
static void k3_cuda_init_from_env(void){
    if(!getenv("K3_GPUS")) return;
    for(const char *p=getenv("K3_GPUS"); *p && g_k3_cuda_ndev<COLI_CUDA_MAX_DEVICES; ){
        g_k3_cuda_devs[g_k3_cuda_ndev++]=(int)strtol(p,(char**)&p,10);
        while(*p==','||*p==' ') p++;
    }
    if(g_k3_cuda_ndev<=0 || !coli_cuda_init(g_k3_cuda_devs,g_k3_cuda_ndev)){
        fprintf(stderr,"[K3/CUDA] coli_cuda_init failed — staying on CPU\n");
        g_k3_cuda_ndev=0; return;
    }
    g_k3_cuda=1;
    double gb = getenv("K3_GPU_GB")?atof(getenv("K3_GPU_GB")):8.0;
    g_k3_gpu_left=(int64_t)(gb*1e9);
    fprintf(stderr,"[K3/CUDA] %d device(s), budget %.1f GB for resident tensors "
        "(routed experts stay on the CPU streaming path)\n",g_k3_cuda_ndev,gb);
    for(int i=0;i<g_k3_cuda_ndev;i++)
        if(coli_cuda_device_integrated(g_k3_cuda_devs[i]))
            fprintf(stderr,"[K3/CUDA] device %d is INTEGRATED (#653): placement duplicates "
                "weights in shared RAM and shrinks the expert cache by the same amount\n",
                g_k3_cuda_devs[i]);
}

/* Placement summary — call after generation so the budget's effect is visible. */
static void k3_cuda_report(void){
    if(!g_k3_cuda) return;
    size_t n=0,b=0; coli_cuda_stats(-1,&n,&b);
    fprintf(stderr,"[K3/CUDA] %zu tensors resident on GPU, %.2f GB (budget used %.2f GB, "
        "%d tensor(s) fell back)\n",n,b/1e9,g_k3_gpu_used/1e9,g_k3_cuda_nfail);
    if(g_k3_exp_gpu||g_k3_exp_cpu)
        fprintf(stderr,"[K3/EXP] routed experts: %llu on GPU, %llu on CPU\n",
            (unsigned long long)g_k3_exp_gpu,(unsigned long long)g_k3_exp_cpu);
}
#endif


/* ---------- tensor parallelism: row-slice view of a dense W ----------------
 * Attention is head-structured, so sharding q/k/v/g by head is just a
 * contiguous ROW range of each [P, hidden] matrix. A W is only pointers plus
 * dims, so a sub-view needs no loader change, no extra memory and no new
 * format -- and each rank then READS only its slice, which is where the time
 * goes (attn is ~18 GB/token, the largest single term once experts shard).
 *
 * o_proj would need a COLUMN slice, which is strided and not expressible this
 * way; instead the per-head outputs are all-reduced back to full width and
 * o_proj is computed redundantly. That leaves ~20% of KDA traffic unsharded
 * but avoids restructuring the loader. */
/* Compact an already-loaded [O,I] tensor down to rows [R0,R0+NR).
 *
 * The quantizing path in w_load_rows reads only the rows it wants, but the
 * pre-quantized U8 containers and the raw-f32 path read whole tensors, and
 * with tensor-parallel attention on by default EVERY multi-node load of
 * q/k/v/g asks for a partial range -- so refusing them would make repacked
 * containers and K3_BITS=32 unusable on more than one node. Rows are
 * contiguous in every format we carry (fmt=0 f32, fmt=1 int8 + per-row scale,
 * fmt=4 int4 + per-group scale), so the slice is a memmove plus a shrink and
 * those configurations keep the same memory saving, just after a full read.
 *
 * realloc is only shrinking here; if it ever declines, keeping the original
 * (larger) buffer is still correct, so the result is dropped rather than
 * treated as fatal. */
static void w_keep_rows(W *w, int R0, int NR){
    if(NR==w->O) return;
    int64_t I=w->I;
    #define KEEP_(p,elems,off) do{ if(p){ \
        memmove((p),(char*)(p)+(size_t)(off),(size_t)(elems)); \
        void *t_=realloc((p),(size_t)(elems)); if(t_) (p)=t_; } }while(0)
    if(w->fmt==0){
        KEEP_(w->f,(int64_t)NR*I*sizeof(float),(int64_t)R0*I*sizeof(float));
    } else if(w->fmt==1){
        KEEP_(w->q8,(int64_t)NR*I,(int64_t)R0*I);
        KEEP_(w->s,(int64_t)NR*sizeof(float),(int64_t)R0*sizeof(float));
    } else {
        int64_t rb=((int64_t)I+1)/2, ng=((int64_t)I+w->gs-1)/w->gs;
        KEEP_(w->q4,(int64_t)NR*rb,(int64_t)R0*rb);
        KEEP_(w->s,(int64_t)NR*ng*sizeof(float),(int64_t)R0*ng*sizeof(float));
    }
    #undef KEEP_
    w->O=NR;
}

/* Release a tensor's host-side buffers. Only safe once nothing can read them
 * again: no w_rows view aliases them (w_rows is a pointer view, w_cols copies),
 * no CPU helper reads them (w_addrow/w_rowdot on kv_b_proj), and either the
 * tensor is device-resident or this rank never multiplies it. On GB10 the host
 * and device copies occupy ONE physical pool, so this is a real reclaim, not
 * bookkeeping. */
static void w_free_host(W *w){
    free(w->f); free(w->q8); free(w->q4); free(w->s);
    w->f=NULL; w->q8=NULL; w->q4=NULL; w->s=NULL;
}

static W w_rows(const W *w, int r0, int nrows){
    W v = *w;
    v.O = nrows;
#ifdef COLI_CUDA
    v.cuda = NULL; v.cuda_placed = 0; v.cuda_failed = 0; v.k3_reg = 0; v.k3_dense_off = 0; v.k3_htouched = 0; v.k3_dw=NULL; v.k3_ds=NULL; v.k3_dev_tried=0;
#endif
    if(w->fmt==0)      v.f  = w->f  + (int64_t)r0*w->I;
    else if(w->fmt==1){ v.q8 = w->q8 + (int64_t)r0*w->I; v.s = w->s + r0; }
    else {
        int64_t rb=((int64_t)w->I+1)/2, ng=((int64_t)w->I+w->gs-1)/w->gs;
        v.q4 = w->q4 + (int64_t)r0*rb; v.s = w->s + (int64_t)r0*ng;
    }
    return v;
}


/* Column slice of a dense W: y = W[:, i0:i0+n] @ x[i0:i0+n], i.e. the
 * row-parallel half of tensor parallelism. Unlike w_rows this must COPY --
 * a column range is strided across rows -- but the copy is contiguous WITHIN
 * each row, so it is a per-row memcpy and the result is a normal W the GPU
 * path can place and cache like any other.
 *
 * Costs half the original per rank (o_proj int4 is 44 MB/layer, so 22 MB), and
 * in exchange the all-reduce shrinks: partial outputs are [hidden] = 7168
 * floats instead of gathering [P] = 12288. Requires i0 aligned to the group
 * size, which holds since i0 = head*128 and gs = 64. */
static W w_cols(const W *w, int i0, int n){
    W v = *w;
    v.I = n;
#ifdef COLI_CUDA
    v.cuda=NULL; v.cuda_placed=0; v.cuda_failed=0; v.k3_reg=0; v.k3_dense_off=0; v.k3_htouched=0; v.k3_dw=NULL; v.k3_ds=NULL; v.k3_dev_tried=0;
#endif
    int64_t O=w->O;
    if(w->fmt==0){
        v.f=falloc(O*n);
        for(int64_t o=0;o<O;o++) memcpy(v.f+o*n, w->f+o*w->I+i0, (size_t)n*sizeof(float));
    } else if(w->fmt==1){
        v.q8=malloc((size_t)O*n);
        if(!v.q8){fprintf(stderr,"OOM w_cols\n");exit(1);}
        for(int64_t o=0;o<O;o++) memcpy(v.q8+o*n, w->q8+o*w->I+i0, (size_t)n);
        /* fmt=1 scales are per ROW, so a column slice does not change their
         * VALUES -- but they must still be COPIED, not aliased. `v = *w` above
         * carries w->s over by pointer, and the o_proj path frees the parent
         * (w_free_host) immediately after this call, which would leave every
         * fmt=1 view holding a dangling scale array. fmt=4 below already
         * copies; this branch silently did not. */
        v.s=falloc(O);
        memcpy(v.s, w->s, (size_t)O*sizeof(float));
    } else {
        int64_t rb=((int64_t)w->I+1)/2, ng=((int64_t)w->I+w->gs-1)/w->gs;
        int64_t rbl=n/2,               ngl=n/w->gs;
        v.q4=malloc((size_t)O*rbl); v.s=falloc(O*ngl);
        if(!v.q4){fprintf(stderr,"OOM w_cols\n");exit(1);}
        for(int64_t o=0;o<O;o++){
            memcpy(v.q4+o*rbl, w->q4+o*rb+i0/2, (size_t)rbl);
            memcpy(v.s+o*ngl,  w->s+o*ng+i0/w->gs, (size_t)ngl*sizeof(float));
        }
    }
    return v;
}


/* Fuse four same-shaped projections that share an input into ONE W.
 * q/k/v/g are each [P,hidden] applied to the same x, so concatenating their
 * rows makes a single [4*n, hidden] matmul. Attention stopped responding to
 * weight-traffic cuts (halving q/k/v/g AND o_proj left attn at 11.8 s), which
 * says it is bound by per-call GPU round trips -- ~6 matmuls/layer x 93, each
 * an upload/download/sync. This removes three of the four for the largest of
 * them. Copies the head slice only, so under TP it costs 4*pn rows, not 4*P. */
static W w_concat4(const W *a, const W *b, const W *c, const W *d, int r0, int n){
    const W *src[4]={a,b,c,d};
    W v=*a; v.O=4*n;
#ifdef COLI_CUDA
    v.cuda=NULL; v.cuda_placed=0; v.cuda_failed=0; v.k3_reg=0; v.k3_dense_off=0; v.k3_htouched=0; v.k3_dw=NULL; v.k3_ds=NULL; v.k3_dev_tried=0;
#endif
    if(a->fmt==1){
        v.q8=malloc((size_t)4*n*a->I); v.s=falloc((int64_t)4*n);
        if(!v.q8){fprintf(stderr,"OOM w_concat4\n");exit(1);}
        for(int j=0;j<4;j++){
            memcpy(v.q8+(int64_t)j*n*a->I, src[j]->q8+(int64_t)r0*a->I, (size_t)n*a->I);
            memcpy(v.s+(int64_t)j*n,       src[j]->s+r0,                (size_t)n*sizeof(float));
        }
    } else {
        int64_t rb=((int64_t)a->I+1)/2, ng=((int64_t)a->I+a->gs-1)/a->gs;
        v.q4=malloc((size_t)4*n*rb); v.s=falloc((int64_t)4*n*ng);
        if(!v.q4){fprintf(stderr,"OOM w_concat4\n");exit(1);}
        for(int j=0;j<4;j++){
            memcpy(v.q4+(int64_t)j*n*rb, src[j]->q4+(int64_t)r0*rb, (size_t)n*rb);
            memcpy(v.s+(int64_t)j*n*ng,  src[j]->s+(int64_t)r0*ng,  (size_t)n*ng*sizeof(float));
        }
    }
    return v;
}

/* ---------- W: load-time quantization + matvec ---------- */


#ifdef COLI_CUDA
/* ---- reclaiming the host half of a mirrored tensor ------------------------
 * This GPU is integrated: coli_k3_devmirror does not move bytes to separate
 * memory, it re-maps them at a coarser granularity, so a mirrored tensor is
 * resident TWICE in one LPDDR5X pool. 16.06 GB is held twice, and none of it
 * shows in RSS -- cudaMalloc'd device pages are not charged to the process --
 * so real occupancy is ~114 of 121 GB, not the 98 the profiler reports.
 *
 * Reclaiming the host half is worth ~16 GB but cannot be done with free():
 * w_matmul falls back to the CPU on w_blob(w) whenever coli_k3_dense declines
 * a shape, w_rowdot/w_addrow read w->f/w->q8 directly, and w_rows() hands out
 * ALIASING views into a parent's buffer. A missed reader would not crash, it
 * would return whatever malloc put there next -- plausible-looking garbage,
 * the one failure mode worth engineering against here.
 *
 * So: unpin (the pages are cudaHostRegistered, and pinned pages cannot be
 * discarded), madvise(MADV_DONTNEED) to actually return the physical pages,
 * then mprotect(PROT_NONE) so any reader we failed to account for takes an
 * immediate SIGSEGV at a known address instead of reading zeros. The pointer
 * is never free()d, so malloc cannot hand the range out again.
 *
 * Eligibility is decided by observation, not by argument: every host-side
 * reader marks its tensor, the sweep runs after a full request has exercised
 * both prefill and decode shapes, and any range that was read is skipped. */
typedef struct { void *p; size_t n; } K3Range;
static K3Range g_mir[8192]; static int g_mir_n=0;
static void   *g_htouch[8192]; static int g_htouch_n=0;
static pthread_mutex_t g_mir_mu=PTHREAD_MUTEX_INITIALIZER;

static void k3_mir_register(void *p, size_t n){
    if(!p||!n) return;
    pthread_mutex_lock(&g_mir_mu);
    if(g_mir_n<(int)(sizeof g_mir/sizeof *g_mir)){ g_mir[g_mir_n].p=p; g_mir[g_mir_n].n=n; g_mir_n++; }
    pthread_mutex_unlock(&g_mir_mu);
}
/* Called once per tensor from every path that dereferences a host blob. */
static void k3_host_touch(const void *p){
    if(!p) return;
    pthread_mutex_lock(&g_mir_mu);
    if(g_htouch_n<(int)(sizeof g_htouch/sizeof *g_htouch)) g_htouch[g_htouch_n++]=(void*)p;
    pthread_mutex_unlock(&g_mir_mu);
}
#define K3_HOST_TOUCH(w) do{ if(!(w)->k3_htouched){ ((W*)(w))->k3_htouched=1; \
                             k3_host_touch(w_blob(w)); } }while(0)

static void k3_host_reclaim_sweep(void){
    long pg=sysconf(_SC_PAGESIZE);
    size_t got=0; int n=0, skipped=0, tiny=0;
    pthread_mutex_lock(&g_mir_mu);
    for(int i=0;i<g_mir_n;i++){
        uintptr_t lo=(uintptr_t)g_mir[i].p, hi=lo+g_mir[i].n;
        int touched=0;
        for(int t=0;t<g_htouch_n;t++){
            uintptr_t q=(uintptr_t)g_htouch[t];
            if(q>=lo&&q<hi){ touched=1; break; }
        }
        if(touched){ skipped++; continue; }
        uintptr_t a=(lo+(uintptr_t)pg-1)&~(uintptr_t)(pg-1), b=hi&~(uintptr_t)(pg-1);
        if(b<=a){ tiny++; continue; }
        coli_k3_unregister(g_mir[i].p);                 /* pinned pages survive madvise */
        if(madvise((void*)a,b-a,MADV_DONTNEED)==0 && mprotect((void*)a,b-a,PROT_NONE)==0){
            got+=b-a; n++;
        }
    }
    pthread_mutex_unlock(&g_mir_mu);
    fprintf(stderr,"[K3/RECLAIM] %d ranges returned %.2f GB; %d skipped (read on host), "
                   "%d too small to page-align\n",n,got/1e9,skipped,tiny);
}
#endif

#ifdef COLI_CUDA
static int w_k3_ptrs_any(const W *w, const void **blob, const float **sc){
    if(!g_k3_dense_gpu || !g_k3_cuda || w->k3_dense_off) return 0;
    if(w->fmt!=1 && w->fmt!=2 && w->fmt!=4) return 0;
    W *mw=(W*)w;
    int64_t rb=(w->fmt==4)?((int64_t)w->I+1)/2
             :(w->fmt==2)?((int64_t)w->I*2):(int64_t)w->I;
    int64_t nsc=(w->fmt==4)?((int64_t)w->I+w->gs-1)/w->gs:1;
    if(!mw->k3_reg){
        if(coli_k3_register((void*)w_blob(w),(size_t)w->O*rb) &&
           (w->fmt==2 || coli_k3_register(w->s,(size_t)w->O*nsc*sizeof(float)))) mw->k3_reg=1;
        else { mw->k3_dense_off=1; return 0; }
    }
    if(!mw->k3_dev_tried){
        void *dw=coli_k3_devmirror(w_blob(w),(size_t)w->O*rb);
        if(dw){ if(w->fmt==2){ mw->k3_dw=dw; mw->k3_ds=NULL;
                               k3_mir_register((void*)w_blob(w),(size_t)w->O*rb); }
                else { void *ds=coli_k3_devmirror(w->s,(size_t)w->O*nsc*sizeof(float));
                       if(ds){ mw->k3_dw=dw; mw->k3_ds=ds;
                               k3_mir_register((void*)w_blob(w),(size_t)w->O*rb);
                               k3_mir_register((void*)w->s,(size_t)w->O*nsc*sizeof(float)); } } }
        mw->k3_dev_tried=1;
    }
    *blob = mw->k3_dw ? (const void*)mw->k3_dw : w_blob(w);
    *sc   = (w->fmt==2) ? NULL : (mw->k3_dw ? (const float*)mw->k3_ds : w->s);
    return 1;
}
#endif

#ifdef COLI_CUDA
/* One S==1 dense call with the register/mirror latch, for callers that need to
 * drive it per token. */
static int w_k3_dense_one(float *y, const float *x, const W *w){
    const void *bl; const float *sc;
    if(!w_k3_ptrs_any(w,&bl,&sc)) return 0;
    return coli_k3_dense(y,x,bl,sc,w->fmt,1,w->I,w->O,w->gs);
}

/* N tensors against one x in a single round trip. Every dense call uploads x,
 * launches, downloads y and synchronises the stream; the four KDA projections
 * share x, so per layer that is four uploads of the same vector and four
 * synchronisations for one kernel's worth of bytes. */
static int w_k3_dense_many(float *const *ys, const float *x, const W *const *ws, int n){
    const void *bl[8]; const float *sc[8];
    if(n<2||n>8) return 0;
    for(int j=0;j<n;j++){
        if(!w_k3_ptrs_any(ws[j],&bl[j],&sc[j])) return 0;
        if(ws[j]->fmt!=ws[0]->fmt||ws[j]->I!=ws[0]->I||
           ws[j]->O!=ws[0]->O||ws[j]->gs!=ws[0]->gs) return 0;
    }
    return coli_k3_dense_multi(ys,x,bl,sc,n,ws[0]->fmt,ws[0]->I,ws[0]->O,ws[0]->gs);
}
#endif

static void w_matmul(float *y, const float *x, const W *w, int S){
#ifdef COLI_CUDA
    /* Zero-copy dense GEMV first: no upload and no duplicate of the weights,
     * which is what frees RAM for the expert cache. Falls through to the
     * uploading path when it declines (prefill, or x too big for shared). */
    if(S==1 && g_dense_census) { if(w->k3_dense_off) atomic_fetch_add_explicit(&g_dp_off,1,memory_order_relaxed);
               else if(!(w->fmt==1||w->fmt==4)) { atomic_fetch_add_explicit(&g_dp_fmt,1,memory_order_relaxed);
                   if(!((W*)w)->fmt_reported){ ((W*)w)->fmt_reported=1;
                       fprintf(stderr,"[K3/DENSE] generic: I=%d O=%d fmt=%d bytes=%.1fMB\n",
                               w->I,w->O,w->fmt,(double)w->I*w->O*4/1e6); } }
               else if(omp_in_parallel()) atomic_fetch_add_explicit(&g_dp_par,1,memory_order_relaxed); }
    /* Prefill (S>1) for int4: one launch, weight row staged per block, C tokens
     * walked against it. Without this the generic quant_matmul runs -- 1370 us
     * a launch against 91 us -- and prefill gets none of the mirroring, the
     * 4-wide fold or the exact path. */
#ifdef COLI_CUDA
    if(g_k3_dense_gpu && g_k3_cuda && !w->k3_dense_off && !omp_in_parallel()
       && S>1 && w->fmt==4){
        const void *bl; const float *sc;
        if(w_k3_ptrs_any(w,&bl,&sc) &&
           coli_k3_dense_s(y,x,bl,sc,w->fmt,S,w->I,w->O,w->gs)){
            if(g_dense_census) atomic_fetch_add_explicit(&g_pf_gemm,1,memory_order_relaxed);
            return; }
        if(g_dense_census && !((W*)w)->pf_reported){ ((W*)w)->pf_reported=1;
            fprintf(stderr,"[K3/PREFILL] GEMM declined I=%d O=%d fmt=%d gs=%d S=%d\n",
                    w->I,w->O,w->fmt,w->gs,S); }
    }
    if(S>1 && g_dense_census && w->fmt!=4)
        atomic_fetch_add_explicit(&g_pf_generic,1,memory_order_relaxed);
#endif
    /* fp16 has no CPU path worth using and the generic CUDA path cannot take
     * it, so drive the S==1 kernel once per token for prefill. Without this the
     * teacher-forced logits capture -- which is prefill only -- would measure
     * the scalar fallback instead of the kernel that actually runs in decode. */
    if(g_k3_dense_gpu && g_k3_cuda && !w->k3_dense_off && !omp_in_parallel() && S>1
       && w->fmt==2){
        int ok=1;
        for(int t=0;t<S && ok;t++)
            ok = w_k3_dense_one(y+(int64_t)t*w->O, x+(int64_t)t*w->I, w);
        if(ok) return;
    }
    if(g_k3_dense_gpu && g_k3_cuda && !w->k3_dense_off && !omp_in_parallel() && S==1
       && (w->fmt==1||w->fmt==2||w->fmt==4)){
        W *mw=(W*)w;
        /* fmt 2 is fp16 with no scale array; everything else carries one. */
        int64_t rb=(w->fmt==4)?((int64_t)w->I+1)/2
                 :(w->fmt==2)?((int64_t)w->I*2):(int64_t)w->I;
        int64_t nsc=(w->fmt==4)?((int64_t)w->I+w->gs-1)/w->gs:1;
        if(!mw->k3_reg){
            if(coli_k3_register((void*)w_blob(w),(size_t)w->O*rb) &&
               (w->fmt==2 || coli_k3_register(w->s,(size_t)w->O*nsc*sizeof(float)))) mw->k3_reg=1;
            else mw->k3_dense_off=1;
        }
        if(mw->k3_reg){
            if(!mw->k3_dev_tried){
                void *dw=coli_k3_devmirror(w_blob(w),(size_t)w->O*rb);
                if(dw){
                    if(w->fmt==2){ mw->k3_dw=dw; mw->k3_ds=NULL;
                                   k3_mir_register((void*)w_blob(w),(size_t)w->O*rb); }
                    else {
                        void *ds=coli_k3_devmirror(w->s,(size_t)w->O*nsc*sizeof(float));
                        if(ds){ mw->k3_dw=dw; mw->k3_ds=ds;     /* both or neither */
                                k3_mir_register((void*)w_blob(w),(size_t)w->O*rb);
                                k3_mir_register((void*)w->s,(size_t)w->O*nsc*sizeof(float)); }
                    }
                }
                mw->k3_dev_tried=1;
            }
            const void *bl = mw->k3_dw ? mw->k3_dw : w_blob(w);
            const float *sc = (w->fmt==2) ? NULL
                            : mw->k3_dw ? (const float*)mw->k3_ds : w->s;
            if(coli_k3_dense(y,x,bl,sc,w->fmt,S,w->I,w->O,w->gs)){
                if(g_dense_census){
                    atomic_fetch_add_explicit(&g_dp_k3,1,memory_order_relaxed);
                    if(!((W*)w)->gs_reported){ ((W*)w)->gs_reported=1;
                        int64_t wb=(int64_t)w->O*((w->I+1)/2);
                        int64_t sb=(int64_t)w->O*((w->I+w->gs-1)/w->gs)*4;
                        fprintf(stderr,"[K3/SCALE] I=%d O=%d fmt=%d gs=%d  weights=%.1fMB "
                                "scales=%.1fMB = %.1f%% of the tensor\n",
                                w->I,w->O,w->fmt,w->gs,wb/1e6,sb/1e6,100.0*sb/(wb+sb)); } }
                return; }
            mw->k3_dense_off=1;   /* declined for a stable reason; stop retrying */
            fprintf(stderr,"[K3/DENSE] declined I=%d O=%d fmt=%d gs=%d -> generic\n",
                    w->I,w->O,w->fmt,w->gs);
        }
    }
    /* Serial path only: placement mutates w lazily and the CUDA driver must not
     * be entered from inside an OpenMP region (mirrors colibri.c's guard). */
    if(g_k3_cuda && !w->cuda_failed && !omp_in_parallel()){
        W *mw = (W*)w;                     /* lazy device mirror; see note above */
        if(!mw->cuda_placed){
            int64_t need = w_bytes(w);
            if(need > g_k3_gpu_left) mw->cuda_failed = 1;   /* budget spent: stay on CPU */
            else {
                mw->cuda_device = g_k3_cuda_devs[g_k3_cuda_rr++ % g_k3_cuda_ndev];
                mw->cuda_placed = 1;
                g_k3_gpu_left -= need; g_k3_gpu_used += need;
            }
        }
        if(mw->cuda_placed){
            if(w->fmt!=2) K3_HOST_TOUCH(w);        /* passes the host blob down */
            if(w->fmt!=2 &&
               coli_cuda_matmul(&mw->cuda,y,x,w_blob(w),w->s,w->fmt,S,w->I,w->O,
                                mw->cuda_device,w->gs)) return;
            mw->cuda_failed = 1;
            if(w->fmt!=2 && g_k3_cuda_nfail++ < 8)
                fprintf(stderr,"[K3/CUDA] tensor [%d,%d] fmt=%d on device %d disabled "
                    "after an error; falling back to CPU\n",w->O,w->I,w->fmt,mw->cuda_device);
        }
    }
#endif
    /* Everything below reads the HOST blob. This is the fallback the reclaim
     * sweep must never surprise: reaching it after the pages were returned
     * would read whatever came next, not crash. Marking here is what makes the
     * tensor ineligible. */
#ifdef COLI_CUDA
    K3_HOST_TOUCH(w);
#endif
    if(w->fmt==0)      matmul(y,x,w->f,S,w->I,w->O);
    else if(w->fmt==2){                 /* fp16 weights: expand and dot */
        const uint16_t *hw=(const uint16_t*)w->q4;
        for(int t=0;t<S;t++){
            const float *xt=x+(int64_t)t*w->I; float *yt=y+(int64_t)t*w->O;
            for(int o=0;o<w->O;o++){
                const uint16_t *hr=hw+(int64_t)o*w->I; float v=0;
                for(int i=0;i<w->I;i++){
                    uint32_t h=hr[i], sgn=(h&0x8000u)<<16, e=(h>>10)&0x1f, m=h&0x3ff, b;
                    if(!e) b=sgn;
                    else b=sgn|((e+112u)<<23)|(m<<13);
                    float wv; memcpy(&wv,&b,4); v+=xt[i]*wv; }
                yt[o]=v; } }
    }
    else if(w->fmt==1) matmul_q(y,x,w->q8,w->s,S,w->I,w->O);
    else if(w->fmt==4) matmul_i4_grouped(y,x,w->q4,w->s,S,w->I,w->O,w->gs);
    else { fprintf(stderr,"w_matmul: bad fmt %d\n",w->fmt); exit(1); }
}
/* acc[0..I) += coef * row r (MLA absorb builds q_abs from kv_b rows) */
static void w_addrow(const W *w, int r, float coef, float *acc){
    int I=w->I;
#ifdef COLI_CUDA
    K3_HOST_TOUCH(w);                 /* dereferences the host blob directly */
#endif
    if(w->fmt==0){ const float *p=w->f+(int64_t)r*I; for(int i=0;i<I;i++) acc[i]+=coef*p[i]; }
    else if(w->fmt==1){ const int8_t *p=w->q8+(int64_t)r*I; float s=w->s[r]*coef;
        for(int i=0;i<I;i++) acc[i]+=s*p[i]; }
    else { int rb=(I+1)/2, ng=(I+w->gs-1)/w->gs; const uint8_t *p=w->q4+(int64_t)r*rb;
        const float *scl=w->s+(int64_t)r*ng;
        for(int g=0;g*w->gs<I;g++){ float s=scl[g]*coef; int e=(g+1)*w->gs; if(e>I)e=I;
            for(int i=g*w->gs;i<e;i+=2){ uint8_t b=p[i>>1];
                acc[i]+=s*(float)((int)(b&0xF)-8);
                if(i+1<e) acc[i+1]+=s*(float)((int)(b>>4)-8); } } }
}
static float w_rowdot(const W *w, int r, const float *x){
    int I=w->I; float a=0;
#ifdef COLI_CUDA
    K3_HOST_TOUCH(w);                 /* dereferences the host blob directly */
#endif
    if(w->fmt==0){ const float *p=w->f+(int64_t)r*I; for(int i=0;i<I;i++) a+=x[i]*p[i]; return a; }
    if(w->fmt==1){ const int8_t *p=w->q8+(int64_t)r*I; for(int i=0;i<I;i++) a+=x[i]*p[i]; return a*w->s[r]; }
    { int rb=(I+1)/2, ng=(I+w->gs-1)/w->gs; const uint8_t *p=w->q4+(int64_t)r*rb;
      const float *scl=w->s+(int64_t)r*ng;
      for(int g=0;g*w->gs<I;g++){ float ga=0; int e=(g+1)*w->gs; if(e>I)e=I;
          for(int i=g*w->gs;i<e;i+=2){ uint8_t b=p[i>>1];
              ga+=x[i]*(float)((int)(b&0xF)-8);
              if(i+1<e) ga+=x[i+1]*(float)((int)(b>>4)-8); }
          a+=ga*scl[g]; } }
    return a;
}

#define QCHUNK 1024                      /* rows per load-quantize pass */
static int g_bits_env=0;                 /* K3_BITS explicitly set: enables the
                                          * int8-container -> int4 load downcast */
static int g_k3_direct=-1;               /* K3_DIRECT: O_DIRECT expert reads */
static int g_k3_idot=1;                  /* K3_IDOT: int8-activation expert matmuls */
static int g_k3_pipe=1;                  /* K3_PIPE: overlap loads with compute */
static float g_k3_topp=0.f;              /* K3_TOPP: routed-expert top-p pruning */
/* Load rows [R0,R0+NR) of the [O,I] tensor `name`; NR==O loads all of it.
 *
 * With tensor-parallel attention each rank multiplies ONLY its own head slice,
 * so loading the whole projection and taking a w_rows() view spends
 * (world-1)/world of the bytes on rows this rank never reads -- 10.25 GB per
 * rank on the KDA q/k/v/g alone at world=4. Worse, the view aliases the
 * parent's buffer, which is exactly what stops us releasing the host copy once
 * the tensor is device-resident (see the #653 note above): the wasted memory
 * cannot even be reclaimed later.
 *
 * Rows are contiguous in a row-major [O,I] tensor, so a row range is a
 * contiguous ELEMENT range and st_read_slice_f32 fetches it directly -- the
 * quantizer already streams in QCHUNK-row blocks, so this only shifts the read
 * offset and shrinks the allocation. O is still validated against the tensor's
 * true numel, so a wrong shard bound fails loudly instead of loading garbage.
 *
 * Slicing is supported on the load-time-quantizing path only. The U8
 * pre-quantized container and the raw-f32 path read whole tensors, so they
 * refuse a partial request rather than silently returning full rows. */
static void w_load_rows(Model *m, W *w, const char *name, int O, int I, int bits,
                        int R0, int NR){
    char nm[512]; snprintf(nm,sizeof(nm),"%s%s",m->pfx,name);
    st_tensor *t=st_find(&m->S,nm);
    if(!t) st_die_missing(&m->S,nm);
    if(NR<0||R0<0||(int64_t)R0+NR>O){
        fprintf(stderr,"%s: bad row slice [%d,%d) of %d rows\n",nm,R0,R0+NR,O); exit(1); }
    memset(w,0,sizeof(*w)); w->O=O; w->I=I;   /* O until the slice is applied */
    if(t->dtype==3){
        /* repacked container (tools/k3_repack.py): pre-quantized U8 + .qs f32
         * scales — no load-time quantization, K3_BITS is ignored for these
         * (except the explicit =4 downcast below) */
        char qn[560]; snprintf(qn,sizeof(qn),"%s.qs",nm);
        st_tensor *ts=st_find(&m->S,qn);
        if(!ts){ fprintf(stderr,"%s: quantized (U8) but no %s scale sidecar\n",nm,qn); exit(1); }
        if(t->nbytes==(int64_t)O*I && ts->numel==O){                  /* int8 per-row */
            if(g_bits_env && bits==4 && I%64==0){
                /* EXPLICIT K3_BITS=4 on an int8 container: downcast to int4-g64
                 * at load. Halves resident RAM (the 62 GB box cannot hold the
                 * 93-layer non-expert set at int8 next to a desktop session);
                 * the int8 grid is 16x finer than int4, so the double-quant
                 * noise ~ the direct-int4 noise. Unset K3_BITS keeps the
                 * container's own bits — the default is untouched. */
                int gs=64, rb=I/2, ng=I/gs;
                int8_t *q8=malloc((size_t)O*I); float *s8=falloc(O);
                if(!q8){fprintf(stderr,"OOM int8 tmp %s\n",nm);exit(1);}
                st_read_raw(&m->S,nm,q8,1); st_read_f32(&m->S,qn,s8,0);
                w->fmt=4; w->gs=gs;
                w->q4=malloc((int64_t)O*rb); w->s=falloc((int64_t)O*ng);
                if(!w->q4){fprintf(stderr,"OOM int4 %s\n",nm);exit(1);}
                for(int r=0;r<O;r++){
                    const int8_t *src=q8+(int64_t)r*I; float sc8=s8[r];
                    uint8_t *dst=w->q4+(int64_t)r*rb; float *scl=w->s+(int64_t)r*ng;
                    for(int g=0;g<ng;g++){ const int8_t *gp=src+g*gs;
                        int am=0; for(int i=0;i<gs;i++){ int a=gp[i]<0?-gp[i]:gp[i]; if(a>am)am=a; }
                        float s=am*sc8/7.f; if(s<1e-20f)s=1e-20f; scl[g]=s; float inv=sc8/s;
                        for(int i=0;i<gs;i+=2){
                            int v0=(int)lrintf(gp[i]*inv);   if(v0>7)v0=7; if(v0<-8)v0=-8;
                            int v1=(int)lrintf(gp[i+1]*inv); if(v1>7)v1=7; if(v1<-8)v1=-8;
                            dst[(g*gs+i)>>1]=(uint8_t)((v0+8)|((v1+8)<<4)); } } }
                free(q8); free(s8);
                w_keep_rows(w,R0,NR);
                return;
            }
            w->fmt=1; w->q8=malloc((size_t)O*I);
            if(!w->q8){fprintf(stderr,"OOM int8 %s\n",nm);exit(1);}
            st_read_raw(&m->S,nm,w->q8,1);
            w->s=falloc(O); st_read_f32(&m->S,qn,w->s,0);
        } else if(I%64==0 && t->nbytes==(int64_t)O*(I/2) && ts->numel==(int64_t)O*(I/64)){
            w->fmt=4; w->gs=64;                                       /* int4-g64 */
            w->q4=malloc((size_t)O*(I/2));
            if(!w->q4){fprintf(stderr,"OOM int4 %s\n",nm);exit(1);}
            st_read_raw(&m->S,nm,w->q4,1);
            w->s=falloc((int64_t)O*(I/64)); st_read_f32(&m->S,qn,w->s,0);
        } else {
            fprintf(stderr,"%s: U8 tensor is %lld bytes / %lld scales — matches neither int8 [%d,%d] nor int4-g64, refusing (untrusted container)\n",
                    nm,(long long)t->nbytes,(long long)ts->numel,O,I); exit(1);
        }
        w_keep_rows(w,R0,NR);
        return;
    }
    if(t->numel!=(int64_t)O*I){ fprintf(stderr,"%s: numel %lld != %dx%d\n",nm,(long long)t->numel,O,I); exit(1); }
    if(bits>=32){
        w->fmt=0; w->f=falloc((int64_t)O*I); st_read_f32(&m->S,nm,w->f,0);
        w_keep_rows(w,R0,NR); return; }
    int gs=64;
    if(bits<=4 && I%gs){ bits=8; }        /* int4-g64 wants I%64==0; fall back */
    float *scr=falloc((int64_t)QCHUNK*I);
    if(bits>4){ w->fmt=1; w->q8=malloc((int64_t)NR*I); w->s=falloc(NR);
        if(!w->q8){fprintf(stderr,"OOM int8 %s\n",nm);exit(1);}
        for(int r0=0;r0<NR;r0+=QCHUNK){ int n=NR-r0<QCHUNK?NR-r0:QCHUNK;
            st_read_slice_f32(&m->S,nm,(int64_t)(R0+r0)*I,(int64_t)n*I,scr,1);
            for(int r=0;r<n;r++){ const float *src=scr+(int64_t)r*I;
                float am=0; for(int i=0;i<I;i++){ float a=fabsf(src[i]); if(a>am)am=a; }
                float s=am/127.f; if(s<1e-20f)s=1e-20f; w->s[r0+r]=s; float inv=1.f/s;
                int8_t *dst=w->q8+(int64_t)(r0+r)*I;
                for(int i=0;i<I;i++){ int v=(int)lrintf(src[i]*inv); if(v>127)v=127; if(v<-127)v=-127; dst[i]=(int8_t)v; } } }
    } else { w->fmt=4; w->gs=gs; int rb=I/2, ng=I/gs;
        w->q4=malloc((int64_t)NR*rb); w->s=falloc((int64_t)NR*ng);
        if(!w->q4){fprintf(stderr,"OOM int4 %s\n",nm);exit(1);}
        for(int r0=0;r0<NR;r0+=QCHUNK){ int n=NR-r0<QCHUNK?NR-r0:QCHUNK;
            st_read_slice_f32(&m->S,nm,(int64_t)(R0+r0)*I,(int64_t)n*I,scr,1);
            for(int r=0;r<n;r++){ const float *src=scr+(int64_t)r*I;
                uint8_t *dst=w->q4+(int64_t)(r0+r)*rb; float *scl=w->s+(int64_t)(r0+r)*ng;
                for(int g=0;g<ng;g++){ const float *gp=src+g*gs;
                    float am=0; for(int i=0;i<gs;i++){ float a=fabsf(gp[i]); if(a>am)am=a; }
                    float s=am/7.f; if(s<1e-20f)s=1e-20f; scl[g]=s; float inv=1.f/s;
                    for(int i=0;i<gs;i+=2){
                        int v0=(int)lrintf(gp[i]*inv);   if(v0>7)v0=7; if(v0<-8)v0=-8;
                        int v1=(int)lrintf(gp[i+1]*inv); if(v1>7)v1=7; if(v1<-8)v1=-8;
                        dst[(g*gs+i)>>1]=(uint8_t)((v0+8)|((v1+8)<<4)); } } } }
    }
    free(scr);
    w->O=NR;   /* this path allocated and quantized only the slice, so no
                * w_keep_rows compaction is needed -- it never read the rest */
}

static void w_load(Model *m, W *w, const char *name, int O, int I, int bits){
    w_load_rows(m,w,name,O,I,bits,0,O);
}
static float *f32_load(Model *m, const char *name, int64_t want){
    char nm[512]; snprintf(nm,sizeof(nm),"%s%s",m->pfx,name);
    st_tensor *t=st_find(&m->S,nm);
    if(!t) st_die_missing(&m->S,nm);
    if(want>0 && t->numel!=want){ fprintf(stderr,"%s: numel %lld != %lld\n",nm,(long long)t->numel,(long long)want); exit(1); }
    float *p=falloc(t->numel); st_read_f32(&m->S,nm,p,0); return p;
}

/* ---------- config ---------- */
static double req_num(jval *r, const char *k){
    jval *v=json_get(r,k);
    if(!v||v->t!=J_NUM){ fprintf(stderr,"config.json: missing or non-numeric \"%s\"\n",k); exit(1); }
    return v->num;
}
static void load_cfg(Cfg *c, const char *snap){
    char path[2048]; snprintf(path,sizeof(path),"%s/config.json",snap);
    long n; char *buf;
    { FILE *f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
      fseek(f,0,SEEK_END); n=ftell(f); fseek(f,0,SEEK_SET);
      if(n<0||n>(64L<<20)){ fprintf(stderr,"%s: bad size\n",path); exit(1); }
      buf=malloc((size_t)n+1); if(!buf){fprintf(stderr,"OOM cfg\n");exit(1);}
      if(fread(buf,1,(size_t)n,f)!=(size_t)n){ fprintf(stderr,"%s: short read\n",path); exit(1); }
      buf[n]=0; fclose(f); }
    char *arena=NULL; jval *root=json_parse(buf,&arena);
    jval *tc=json_get(root,"text_config"); if(!tc||tc->t!=J_OBJ) tc=root;
    memset(c,0,sizeof(*c));
    c->hidden      =(int)req_num(tc,"hidden_size");
    c->n_layers    =(int)req_num(tc,"num_hidden_layers");
    c->vocab       =(int)req_num(tc,"vocab_size");
    c->first_dense =(int)req_num(tc,"first_k_dense_replace");
    c->dense_inter =(int)req_num(tc,"intermediate_size");
    c->n_heads     =(int)req_num(tc,"num_attention_heads");
    c->q_lora      =(int)req_num(tc,"q_lora_rank");
    c->kv_lora     =(int)req_num(tc,"kv_lora_rank");
    c->qk_nope     =(int)req_num(tc,"qk_nope_head_dim");
    c->qk_rope     =(int)req_num(tc,"qk_rope_head_dim");
    c->v_head      =(int)req_num(tc,"v_head_dim");
    c->n_experts   =(int)req_num(tc,"num_experts");
    c->topk        =(int)req_num(tc,"num_experts_per_token");
    c->moe_inter   =(int)req_num(tc,"moe_intermediate_size");
    c->latent      =(int)req_num(tc,"routed_expert_hidden_size");
    c->n_shared    =(int)req_num(tc,"num_shared_experts");
    c->res_bs      =(int)req_num(tc,"attn_res_block_size");
    c->situ_b1     =(float)req_num(tc,"activation_situ_beta");
    c->situ_b2     =(float)req_num(tc,"activation_situ_linear_beta");
    jval *ep=json_get(tc,"rms_norm_eps"); c->eps=ep?(float)ep->num:1e-5f;
    c->qk_head=c->qk_nope+c->qk_rope;
    c->attn_scale=1.f/sqrtf((float)c->qk_head);
    jval *la=json_get(tc,"linear_attn_config");
    if(!la||la->t!=J_OBJ){ fprintf(stderr,"config.json: missing linear_attn_config\n"); exit(1); }
    c->kda_heads=(int)req_num(la,"num_heads");
    c->kda_hd   =(int)req_num(la,"head_dim");
    c->conv_k   =(int)req_num(la,"short_conv_kernel_size");
    jval *lb=json_get(la,"gate_lower_bound"); c->gate_lb=lb?(float)lb->num:-5.f;
    c->kda_proj=c->kda_heads*c->kda_hd;
    if(c->hidden<1||c->hidden>65536||c->n_layers<1||c->n_layers>128||
       c->n_experts<1||c->n_experts>4096||c->topk<1||c->topk>64||c->topk>c->n_experts||
       c->vocab<1||c->vocab>(1<<22)||c->kda_proj<1||c->kda_proj>(1<<20)||
       c->conv_k<1||c->conv_k>8||c->latent<32||c->latent%32||c->moe_inter%32||
       c->res_bs<1||c->kda_hd>512||c->kv_lora>4096){
        fprintf(stderr,"config.json: dimension out of range\n"); exit(1); }
    jval *kl=json_get(la,"kda_layers");
    if(!kl||kl->t!=J_ARR){ fprintf(stderr,"config.json: missing kda_layers\n"); exit(1); }
    for(int i=0;i<kl->len;i++){ int v=(int)kl->kids[i]->num;      /* 1-indexed */
        if(v>=1&&v<=c->n_layers) c->is_kda[v-1]=1; }
    jval *b=json_get(root,"bos_token_id"); if(!b) b=json_get(tc,"bos_token_id");
    c->bos = b&&b->t==J_NUM ? (int)b->num : -1;
    jval *e=json_get(root,"eos_token_id"); if(!e) e=json_get(tc,"eos_token_id");
    if(e&&e->t==J_NUM) c->eos[c->n_eos++]=(int)e->num;
    else if(e&&e->t==J_ARR) for(int i=0;i<e->len&&c->n_eos<8;i++) c->eos[c->n_eos++]=(int)e->kids[i]->num;
    free(buf); (void)arena;
}

/* ---------- init ---------- */
static void expert_table_init(Model *m){
    Cfg *c=&m->c;
    /* K3_W2_DIR: read experts from a packed 2-bit store (tools/
     * pack_experts_2bit.py) instead of the HF MXFP4 shards. Same ue8m0 scales,
     * codes are 2-bit indices into {-4,-1,1,4}: 8.86 MiB per expert instead of
     * 16.74, i.e. 1.89x less to stream and to dequantize. The store is one
     * flat file with slots in (moe_layer, expert) order, so every expert is a
     * single contiguous pread rather than six ranges. */
    const char *w1dir = getenv("K3_W1_DIR");
    const char *w2dir = w1dir ? w1dir : getenv("K3_W2_DIR");
    m->w1_mode = w1dir ? 1 : 0;
    if(w2dir){
        int den = m->w1_mode ? 8 : 4;     /* 1 bit vs 2 bits per weight */
        m->e_w1p=(int64_t)c->moe_inter*(c->latent/den); m->e_w1s=(int64_t)c->moe_inter*(c->latent/32);
        m->e_w2p=(int64_t)c->latent*(c->moe_inter/den); m->e_w2s=(int64_t)c->latent*(c->moe_inter/32);
    } else {
        m->e_w1p=(int64_t)c->moe_inter*(c->latent/2); m->e_w1s=(int64_t)c->moe_inter*(c->latent/32);
        m->e_w2p=(int64_t)c->latent*(c->moe_inter/2); m->e_w2s=(int64_t)c->latent*(c->moe_inter/32);
    }
    m->e_slot=2*(m->e_w1p+m->e_w1s)+m->e_w2p+m->e_w2s;
    m->eref=calloc((size_t)c->n_layers*c->n_experts,sizeof(ERef));
    if(!m->eref){fprintf(stderr,"OOM expert table\n");exit(1);}
    if(w2dir){
        char p[1024]; snprintf(p,sizeof(p),"%s/experts.w2",w2dir);   /* same filename both widths */
        int fd=open(p,O_RDONLY);
        if(fd<0){ fprintf(stderr,"[K3/W2] cannot open %s: %s\n",p,strerror(errno)); exit(1); }
        m->w2_fd=fd;
        /* A second O_DIRECT descriptor. expert_read() normally gets one from
         * st_direct_fd(), which only knows shards registered in m->S — this
         * store is not, so without this it silently falls back to BUFFERED
         * reads. Measured cost of that fallback: 132 GB took 67 s (~2 GB/s)
         * instead of the ~20 s the device does at 6.7 GB/s, because a 766 GB
         * working set against 116 GB of RAM just thrashes the page cache.
         * Slot size and offsets are multiples of 4096, so the aligned path
         * needs no head/tail slack. */
        m->w2_dfd=open(p,O_RDONLY|O_DIRECT);
        if(m->w2_dfd<0){
            fprintf(stderr,"[K3/W2] O_DIRECT open failed (%s) — using buffered reads, "
                "expect ~3x slower expert loads\n",strerror(errno));
            m->w2_dfd=0;
        }
        if(m->e_slot%4096){ fprintf(stderr,"[K3/W2] slot %lld not 4096-aligned\n",
            (long long)m->e_slot); exit(1); }
        int64_t off[6]={0,m->e_w1p,m->e_w1p+m->e_w1s,
                        m->e_w1p+m->e_w1s+m->e_w2p,
                        m->e_w1p+m->e_w1s+m->e_w2p+m->e_w2s,
                        2*m->e_w1p+m->e_w1s+m->e_w2p+m->e_w2s};
        /* K3_W2_SHARD=r/N: this store holds only experts with e%N==r, packed
         * contiguously, so each node carries 1/N of the 766 GB. Matches the
         * e%world expert-parallel split in moe_forward, and is what makes a
         * RAM-resident expert set reachable at all. */
        int sr=0,sw=1;
        if(getenv("K3_W2_SHARD") && sscanf(getenv("K3_W2_SHARD"),"%d/%d",&sr,&sw)==2 && sw>0){
            if(c->n_experts%sw){ fprintf(stderr,"[K3/W2] %d experts not divisible by %d\n",
                c->n_experts,sw); exit(1); }
        } else { sr=0; sw=1; }
        int Eloc=c->n_experts/sw;
        int nmoe=0;
        for(int li=0;li<c->n_layers;li++){
            if(!m->L[li].sparse) continue;
            for(int e2=0;e2<c->n_experts;e2++){
                ERef *er=&m->eref[(int64_t)li*c->n_experts+e2];
                if(sw>1 && e2%sw!=sr){ for(int k=0;k<6;k++) er->fd[k]=-1; continue; }
                int64_t base=((int64_t)nmoe*Eloc+e2/sw)*m->e_slot;
                for(int k=0;k<6;k++){ er->fd[k]=fd; er->off[k]=base+off[k]; }
                er->contig=1;              /* one pread per expert */
            }
            nmoe++;
        }
        struct stat sb; int64_t want=(int64_t)nmoe*Eloc*m->e_slot;
        if(fstat(fd,&sb)==0 && sb.st_size<want){
            fprintf(stderr,"[K3/W2] %s is %lld bytes, need %lld for %d MoE layers x %d experts"
                " — refusing (incomplete pack)\n",p,(long long)sb.st_size,(long long)want,
                nmoe,Eloc); exit(1); }
        m->hz_batch=falloc((int64_t)64*c->latent);   /* K3_BATCH_MAX x latent */
        fprintf(stderr,"[K3/W2] packed 2-bit experts: %s, slot %.2f MiB, %d layers x %d"
            " (shard %d/%d, %d-bit) = %.1f GB\n",p,m->e_slot/1048576.0,nmoe,Eloc,sr,sw,
            m->w1_mode?1:2,want/1e9);
        return;
    }
    const char *mat[3]={"w1","w2","w3"};
    int64_t want[6]={m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s,m->e_w1p,m->e_w1s};
    int missing=0;
    for(int li=0;li<c->n_layers;li++){
        if(!m->L[li].sparse) continue;
        for(int e2=0;e2<c->n_experts;e2++){
            ERef *er=&m->eref[(int64_t)li*c->n_experts+e2];
            for(int k=0;k<6;k++){
                char nm[512];
                snprintf(nm,sizeof(nm),"%smodel.layers.%d.block_sparse_moe.experts.%d.%s.weight_%s",
                         m->pfx,li,e2,mat[k/2],(k&1)?"scale":"packed");
                st_tensor *t=st_find(&m->S,nm);
                if(!t){ missing++; er->fd[k]=-1; continue; }
                if(t->nbytes!=want[k]){ fprintf(stderr,"%s: %lld bytes, expected %lld — refusing (untrusted container)\n",
                        nm,(long long)t->nbytes,(long long)want[k]); exit(1); }
                er->fd[k]=t->fd; er->off[k]=t->off;
            }
            /* HF shards store the six tensors back-to-back (measured: 0 gaps
             * across a whole layer) — collapse the load to ONE pread when so */
            er->contig=1;
            for(int k=0;k<5;k++)
                if(er->fd[k]!=er->fd[k+1]||er->off[k]+want[k]!=er->off[k+1]) er->contig=0;
        }
    }
    if(missing) fprintf(stderr,"[K3] WARNING: %d expert tensors missing (incomplete download?) — touching one aborts\n",missing);
}

static void model_init(Model *m, const char *snap, int n_layers_env){
    memset(m,0,sizeof(*m));
    load_cfg(&m->c,snap);
    Cfg *c=&m->c;
    if(n_layers_env>0&&n_layers_env<c->n_layers) c->n_layers=n_layers_env;
#ifdef COLI_CUDA
    k3_cuda_init_from_env();                       /* before the shards: fail fast */
#endif
    st_init_multi(&m->S,snap,getenv("K3_DIRS"));   /* K3_DIRS: extra shard dirs (multi-drive split) */
    m->pfx[0]=0;   /* probe a layer-0 tensor: embed/head live in one of the LAST shards */
    if(!st_has(&m->S,"model.layers.0.input_layernorm.weight")&&
       st_has(&m->S,"language_model.model.layers.0.input_layernorm.weight"))
        snprintf(m->pfx,sizeof(m->pfx),"language_model.");
    if((c->n_layers+c->res_bs-1)/c->res_bs+1>16){ fprintf(stderr,"attn_res: too many blocks\n"); exit(1); }
    g_bits_env = getenv("K3_BITS")!=NULL;
    g_k3_direct = getenv("K3_DIRECT")?atoi(getenv("K3_DIRECT")):1;
    g_k3_idot  = getenv("K3_IDOT")?atoi(getenv("K3_IDOT")):1;
    g_k3_pipe  = getenv("K3_PIPE")?atoi(getenv("K3_PIPE")):1;
    g_k3_topp  = getenv("K3_TOPP")?(float)atof(getenv("K3_TOPP")):0.f;
    if(g_k3_topp>0.f)
        fprintf(stderr,"[K3] TOPP=%.2f: routed experts pruned to cumulative weight (quality lever — A/B with K3_LOGITS)\n",g_k3_topp);
    /* Resolved here, not with the other flags below, because the head-sliced
     * attention loads depend on it. */
    g_k3_tp_attn = getenv("K3_TP_ATTN")?atoi(getenv("K3_TP_ATTN")):1;
    int bits   = getenv("K3_BITS")?atoi(getenv("K3_BITS")):4;
    int mbits  = getenv("K3_MLA_BITS")?atoi(getenv("K3_MLA_BITS")):8;
    int hbits  = getenv("K3_HEAD_BITS")?atoi(getenv("K3_HEAD_BITS")):8;
    double t0=now_s();
    m->L=calloc(c->n_layers,sizeof(Layer));
    m->kstate=calloc(c->n_layers,sizeof(float*));
    memset(m->cwh,0,sizeof(m->cwh));
    m->cwq=calloc(c->n_layers,sizeof(float*));
    m->cwk=calloc(c->n_layers,sizeof(float*));
    m->cwv=calloc(c->n_layers,sizeof(float*));
    char nm[512];
    #define NM(...) (snprintf(nm,sizeof(nm),__VA_ARGS__),nm)
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        l->kda=c->is_kda[i];
        l->sparse=(i>=c->first_dense);
        l->in_ln  =f32_load(m,NM("model.layers.%d.input_layernorm.weight",i),c->hidden);
        l->post_ln=f32_load(m,NM("model.layers.%d.post_attention_layernorm.weight",i),c->hidden);
        /* AttnRes score weight = res_norm.weight * res_proj.weight (elementwise) */
        { float *rn=f32_load(m,NM("model.layers.%d.self_attention_res_norm.weight",i),c->hidden);
          float *rp=f32_load(m,NM("model.layers.%d.self_attention_res_proj.weight",i),c->hidden);
          l->attn_sw=falloc(c->hidden); for(int d=0;d<c->hidden;d++) l->attn_sw[d]=rn[d]*rp[d];
          free(rn); free(rp);
          rn=f32_load(m,NM("model.layers.%d.mlp_res_norm.weight",i),c->hidden);
          rp=f32_load(m,NM("model.layers.%d.mlp_res_proj.weight",i),c->hidden);
          l->mlp_sw=falloc(c->hidden); for(int d=0;d<c->hidden;d++) l->mlp_sw[d]=rn[d]*rp[d];
          free(rn); free(rp); }
        if(l->kda){
            Kda *a=&l->a; int P=c->kda_proj;
            /* q/k/v/g are row-parallel: this rank only ever multiplies its own
             * head rows, so load just those. At world=4 that is 3/4 of 13.67 GB
             * left unread, and it removes the w_rows aliasing that pinned the
             * host copy alive after device placement. */
            int kh0,khn; k3_head_shard(c->kda_heads,&kh0,&khn);
            int kr0=kh0*c->kda_hd, krn=khn*c->kda_hd;
            w_load_rows(m,&a->q,NM("model.layers.%d.self_attn.q_proj.weight",i),P,c->hidden,bits,kr0,krn);
            w_load_rows(m,&a->k,NM("model.layers.%d.self_attn.k_proj.weight",i),P,c->hidden,bits,kr0,krn);
            w_load_rows(m,&a->v,NM("model.layers.%d.self_attn.v_proj.weight",i),P,c->hidden,bits,kr0,krn);
            w_load_rows(m,&a->g,NM("model.layers.%d.self_attn.g_proj.weight",i),P,c->hidden,bits,kr0,krn);
            w_load(m,&a->o,NM("model.layers.%d.self_attn.o_proj.weight",i),c->hidden,P,bits);
            a->conv_q=f32_load(m,NM("model.layers.%d.self_attn.q_conv1d.weight",i),(int64_t)P*c->conv_k);
            a->conv_k=f32_load(m,NM("model.layers.%d.self_attn.k_conv1d.weight",i),(int64_t)P*c->conv_k);
            a->conv_v=f32_load(m,NM("model.layers.%d.self_attn.v_conv1d.weight",i),(int64_t)P*c->conv_k);
            /* These three projections are f32 in the checkpoint. Keep them as
             * W objects so each rank retains only its head rows; the CPU path
             * still calls the same f32 matmul and is arithmetic-identical. */
            w_load(m,&a->fa,NM("model.layers.%d.self_attn.f_a_proj.weight",i),c->kda_hd,c->hidden,32);
            w_load_rows(m,&a->fb,NM("model.layers.%d.self_attn.f_b_proj.weight",i),P,c->kda_hd,32,kr0,krn);
            w_load_rows(m,&a->bp,NM("model.layers.%d.self_attn.b_proj.weight",i),c->kda_heads,c->hidden,32,kh0,khn);
#ifdef COLI_CUDA
            /* Parallel GPU reduction changed KDA routing and generated text;
             * keep these strict-f32 control projections on the CPU. */
            a->fa.cuda_failed=a->fb.cuda_failed=a->bp.cuda_failed=1;
#endif
            a->dt=f32_load(m,NM("model.layers.%d.self_attn.dt_bias",i),P);
            a->onw=f32_load(m,NM("model.layers.%d.self_attn.o_norm.weight",i),c->kda_hd);
            { /* A_log in the checkpoint is [kda_hd] = per-head zero-padded */
              char an[512]; snprintf(an,sizeof(an),"%smodel.layers.%d.self_attn.A_log",m->pfx,i);
              st_tensor *t=st_find(&m->S,an); if(!t) st_die_missing(&m->S,an);
              if(t->numel<c->kda_heads){ fprintf(stderr,"%s: %lld < heads\n",an,(long long)t->numel); exit(1); }
              float *al=falloc(t->numel); st_read_f32(&m->S,an,al,0);
              a->A=falloc(c->kda_heads);
              for(int h=0;h<c->kda_heads;h++) a->A[h]=expf(al[h]);
              free(al); }
            m->kstate[i]=fcalloc((int64_t)c->kda_heads*c->kda_hd*c->kda_hd);
            m->cwq[i]=fcalloc((int64_t)P*c->conv_k);
            m->cwk[i]=fcalloc((int64_t)P*c->conv_k);
            m->cwv[i]=fcalloc((int64_t)P*c->conv_k);
        } else {
            Mla *a=&l->m;
            w_load(m,&a->qa,NM("model.layers.%d.self_attn.q_a_proj.weight",i),c->q_lora,c->hidden,mbits);
            int mh0_,mhn_; k3_head_shard(c->n_heads,&mh0_,&mhn_);   /* row-parallel, as KDA above */
            w_load_rows(m,&a->qb,NM("model.layers.%d.self_attn.q_b_proj.weight",i),c->n_heads*c->qk_head,c->q_lora,mbits,
                        mh0_*c->qk_head,mhn_*c->qk_head);
            w_load(m,&a->kva,NM("model.layers.%d.self_attn.kv_a_proj_with_mqa.weight",i),c->kv_lora+c->qk_rope,c->hidden,mbits);
            w_load(m,&a->kvb,NM("model.layers.%d.self_attn.kv_b_proj.weight",i),c->n_heads*(c->qk_nope+c->v_head),c->kv_lora,mbits);
            w_load(m,&a->o,NM("model.layers.%d.self_attn.o_proj.weight",i),c->hidden,c->n_heads*c->v_head,mbits);
            w_load_rows(m,&a->g,NM("model.layers.%d.self_attn.g_proj.weight",i),c->n_heads*c->v_head,c->hidden,mbits,
                        mh0_*c->v_head,mhn_*c->v_head);
            a->qa_ln =f32_load(m,NM("model.layers.%d.self_attn.q_a_layernorm.weight",i),c->q_lora);
            a->kva_ln=f32_load(m,NM("model.layers.%d.self_attn.kv_a_layernorm.weight",i),c->kv_lora);
        }
        if(l->sparse){
            Moe *o=&l->moe;
            o->router=f32_load(m,NM("model.layers.%d.block_sparse_moe.gate.weight",i),(int64_t)c->n_experts*c->hidden);
            /* 25.7 MB/layer x 92 = 2.36 GB/token. On the CPU f32 path that is
             * ~0.21 s/token at the ~11 GB/s those cores manage; as a W it uses
             * the same GPU dispatch as every other dense tensor. */
            o->router_w.O=c->n_experts; o->router_w.I=c->hidden;
            /* K3_ROUTER_I8: quantize the router to int8, per output row.
             *
             * It is the largest tensor still on the generic kernel and the
             * single biggest byte source left -- 25.7 MB/layer, 2.36 GB/token,
             * ~15% of everything the model reads. At int8 that is 0.59 GB, and
             * it also moves onto the exact int8 kernel and the device mirror
             * instead of coli_cuda_matmul.
             *
             * NOT exact, and the risky kind of inexact: this feeds top-16-of-896
             * selection, where the measured gap between the 16th and 17th biased
             * score is 0.0032. Gate on end-to-end PCC and the free-running
             * output, never on the hash. */
            { static int ri8=-1;
              if(ri8<0){ const char *e=getenv("K3_ROUTER_I8"); ri8=e?atoi(e):0; }
              int64_t RO=c->n_experts, RI=c->hidden;
              static int rf16=-1;
              if(rf16<0){ const char *e=getenv("K3_ROUTER_F16"); rf16=e?atoi(e):0; }
              static int r2=-1;
              if(r2<0){ const char *e=getenv("K3_ROUTER2"); r2=e?atoi(e):0; }
              if(r2 && !ri8 && !rf16){
                  /* Two-pass: int8 for the search, exact f32 for the decision.
                   * Keeps the f32 rows (needed for re-scoring) and adds a 0.59 GB
                   * int8 copy. Quantizing outright failed the PCC bar because the
                   * error is comparable to the 0.0032 top-16 boundary gap; this
                   * spends bytes only where the ranking is already settled. */
                  o->r_q8=malloc((size_t)RO*RI);
                  o->r_s8=falloc(RO);
                  if(!o->r_q8){fprintf(stderr,"OOM router search copy\n");exit(1);}
                  for(int64_t r=0;r<RO;r++){
                      const float *rw=o->router+r*RI;
                      float mx=0.f;
                      for(int64_t i=0;i<RI;i++){ float a=fabsf(rw[i]); if(a>mx) mx=a; }
                      float sc=mx/127.f; if(sc<1e-30f) sc=1e-30f;
                      o->r_s8[r]=sc; float inv=1.f/sc; int8_t *q=o->r_q8+r*RI;
                      for(int64_t i=0;i<RI;i++){
                          int v=(int)lrintf(rw[i]*inv);
                          if(v>127)v=127; if(v<-127)v=-127; q[i]=(int8_t)v; } }
                  o->router_w.fmt=0; o->router_w.f=o->router;
              } else if(rf16 && !ri8){
                  /* fp16 keeps relative precision everywhere (~0.05%) where int8
                   * per-row ties the error to the row max (~2%), so it perturbs
                   * the top-16 DECISION far less for half the bytes rather than
                   * a quarter. Stored in q4 as raw 16-bit halves. */
                  o->router_w.q4=malloc((size_t)RO*RI*2);
                  if(!o->router_w.q4){fprintf(stderr,"OOM router fp16\n");exit(1);}
                  uint16_t *h=(uint16_t*)o->router_w.q4;
                  for(int64_t i=0;i<RO*RI;i++){
                      float v=o->router[i];
                      uint32_t b; memcpy(&b,&v,4);
                      uint32_t sgn=(b>>16)&0x8000u; int32_t e2=(int32_t)((b>>23)&0xff)-127+15;
                      uint32_t man=b&0x7fffffu;
                      uint16_t out;
                      if(e2<=0) out=(uint16_t)sgn;                       /* flush tiny to zero */
                      else if(e2>=31) out=(uint16_t)(sgn|0x7bffu);       /* clamp to max half */
                      else {
                          uint32_t r=man&0x1fffu; man>>=13;
                          if(r>0x1000u || (r==0x1000u && (man&1))){ man++; if(man>0x3ffu){man=0;e2++;} }
                          out=(uint16_t)(sgn|((uint32_t)e2<<10)|man);
                      }
                      h[i]=out; }
                  o->router_w.fmt=2;
                  free(o->router); o->router=NULL;
              } else if(ri8){
                  o->router_w.q8=malloc((size_t)RO*RI);
                  o->router_w.s=falloc(RO);
                  if(!o->router_w.q8){fprintf(stderr,"OOM router int8\n");exit(1);}
                  for(int64_t r=0;r<RO;r++){
                      const float *rw=o->router+r*RI;
                      float mx=0.f;
                      for(int64_t i=0;i<RI;i++){ float a=fabsf(rw[i]); if(a>mx) mx=a; }
                      float sc=mx/127.f; if(sc<1e-30f) sc=1e-30f;
                      o->router_w.s[r]=sc;
                      float inv=1.f/sc; int8_t *q=o->router_w.q8+r*RI;
                      for(int64_t i=0;i<RI;i++){
                          int v=(int)lrintf(rw[i]*inv);
                          if(v>127)v=127; if(v<-127)v=-127; q[i]=(int8_t)v; } }
                  o->router_w.fmt=1;
                  free(o->router); o->router=NULL;      /* gives back 25.7 MB/layer */
              } else { o->router_w.fmt=0; o->router_w.f=o->router; } }
            o->rbias =f32_load(m,NM("model.layers.%d.block_sparse_moe.gate.e_score_correction_bias",i),c->n_experts);
            o->lat_norm=f32_load(m,NM("model.layers.%d.block_sparse_moe.routed_expert_norm.weight",i),c->latent);
            w_load(m,&o->lat_down,NM("model.layers.%d.block_sparse_moe.routed_expert_down_proj.weight",i),c->latent,c->hidden,bits);
            w_load(m,&o->lat_up,NM("model.layers.%d.block_sparse_moe.routed_expert_up_proj.weight",i),c->hidden,c->latent,bits);
            int shi=c->moe_inter*c->n_shared;
            w_load(m,&o->sh_gate,NM("model.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight",i),shi,c->hidden,bits);
            w_load(m,&o->sh_up,NM("model.layers.%d.block_sparse_moe.shared_experts.up_proj.weight",i),shi,c->hidden,bits);
            w_load(m,&o->sh_down,NM("model.layers.%d.block_sparse_moe.shared_experts.down_proj.weight",i),c->hidden,shi,bits);
        } else {
            w_load(m,&l->d_gate,NM("model.layers.%d.mlp.gate_proj.weight",i),c->dense_inter,c->hidden,bits);
            w_load(m,&l->d_up,NM("model.layers.%d.mlp.up_proj.weight",i),c->dense_inter,c->hidden,bits);
            w_load(m,&l->d_down,NM("model.layers.%d.mlp.down_proj.weight",i),c->hidden,c->dense_inter,bits);
        }
        if(i%8==0) fprintf(stderr,"[K3] loaded layer %d/%d (%.1fs, RSS %.1f GB)\n",i+1,c->n_layers,now_s()-t0,rss_gb());
    }
    snprintf(nm,sizeof(nm),"%smodel.norm.weight",m->pfx);
    m->has_head = st_has(&m->S,nm);
    if(m->has_head){
        m->final_norm=f32_load(m,"model.norm.weight",c->hidden);
        { float *rn=f32_load(m,"model.output_attn_res_norm.weight",c->hidden);
          float *rp=f32_load(m,"model.output_attn_res_proj.weight",c->hidden);
          m->out_sw=falloc(c->hidden); for(int d=0;d<c->hidden;d++) m->out_sw[d]=rn[d]*rp[d];
          free(rn); free(rp); }
        w_load(m,&m->lm_head,"lm_head.weight",c->vocab,c->hidden,hbits);
    } else fprintf(stderr,"[K3] final norm/lm_head not present — trace-only mode\n");
    if(getenv("K3_ROUTE_STATS")){
        m->route_hist=calloc((size_t)c->n_layers*c->n_experts,sizeof(uint32_t));
        if(!m->route_hist){fprintf(stderr,"OOM route hist\n");exit(1);}
    }
    expert_table_init(m);
#ifdef COLI_CUDA
    /* K3_EXPERT_GPU=1 moves the routed-expert matmuls onto the GPU. Needs a
     * device from K3_GPUS; measured on GB10 this is where ~84% of decode time
     * sits, of which only ~14% is I/O. */
    g_k3_expert_batch = getenv("K3_EXPERT_BATCH")?atoi(getenv("K3_EXPERT_BATCH")):1;
    g_k3_dense_gpu    = getenv("K3_DENSE_GPU")?atoi(getenv("K3_DENSE_GPU")):1;
    /* NOTE: g_k3_tp_attn is resolved EARLIER, before the weights load -- the
     * head-sliced loads below depend on it, and reading it here would be too
     * late (the projections would already be resident at full width). */
    { const char *e=getenv("K3_DENSE_CENSUS"); g_dense_census = e?atoi(e):0; }
    int want_expert_gpu=getenv("K3_EXPERT_GPU")&&atoi(getenv("K3_EXPERT_GPU"));
    /* The zero-copy dense kernels share backend_cuda_k3's stream/scratch
     * initialization but do not require routed experts to run on the GPU. */
    int k3_ready=(g_k3_cuda&&(want_expert_gpu||g_k3_dense_gpu))
                 ?coli_k3_init(g_k3_cuda_devs[0],c->latent,c->moe_inter):0;
    g_k3_ready=k3_ready;
    g_k3_expert_gpu=want_expert_gpu&&k3_ready;
    if(want_expert_gpu&&!g_k3_cuda)
        fprintf(stderr,"[K3/EXP] K3_EXPERT_GPU needs K3_GPUS — staying on CPU\n");
#endif
    /* expert LRU cache, per-layer slots from the global budget */
    double egb = getenv("K3_EXPERT_GB")?atof(getenv("K3_EXPERT_GB")):8.0;
    int nmoe=0; for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse) nmoe++;
    int cap=(int)((egb*1e9)/((double)m->e_slot*(nmoe?nmoe:1)));
    /* floor 1, NOT topk: experts are loaded and consumed one at a time inside
     * a token, so slots never need to hold a whole top-k set. A topk floor
     * would silently commit topk*nmoe slots (~26 GB on the 93-layer model)
     * regardless of K3_EXPERT_GB. */
    if(cap<1) cap=1;
    if(cap>c->n_experts) cap=c->n_experts;
    m->ecache=calloc(c->n_layers,sizeof(LCache));
    for(int i=0;i<c->n_layers;i++) if(m->L[i].sparse){
        m->ecache[i].cap=cap; m->ecache[i].s=calloc(cap,sizeof(Slot));
        for(int j2=0;j2<cap;j2++) m->ecache[i].s[j2].eid=-1;
    }
    fprintf(stderr,"[K3] init done in %.1fs | %d layers | expert cache %d/layer (%.1f MB/slot) | RSS %.1f GB\n",
            now_s()-t0,c->n_layers,cap,m->e_slot/1e6,rss_gb());
    #undef NM
}

/* ---------- AttnRes softmax mix over [bres rows..., prefix] ---------- */
static double g_resmix_secs=0;
static void res_mix(float *out, const float *prefix, const float *bres, int nb, int D,
                    const float *sw, float eps){
    double rm0=now_s();
    const float *v[16]; float sc[16];
    for(int e=0;e<nb;e++) v[e]=bres+(int64_t)e*D;
    v[nb]=prefix;
    for(int e=0;e<=nb;e++){
        double ms=0,dot=0;
        for(int d=0;d<D;d++){ double x=v[e][d]; ms+=x*x; dot+=x*(double)sw[d]; }
        sc[e]=(float)(dot/sqrt(ms/D+eps));
    }
    softmax_(sc,nb+1);
    for(int d=0;d<D;d++){ float a=0; for(int e=0;e<=nb;e++) a+=sc[e]*v[e][d]; out[d]=a; }
    g_resmix_secs+=now_s()-rm0;
}

/* ---------- KDA layer (chunk of C tokens; projections batched, recurrence
 * sequential per token, AVX2 on the state sweeps) ---------- */
static void kda_forward(Model *m, Layer *l, int li, const float *x, int C, float *out){
    Cfg *c=&m->c; Kda *a=&l->a;
    int P=c->kda_proj, H=c->kda_heads, hd=c->kda_hd, K=c->conv_k;
    float *q=falloc((int64_t)C*P), *k=falloc((int64_t)C*P), *v=falloc((int64_t)C*P);
    float *gp=falloc((int64_t)C*P), *on=falloc((int64_t)C*P);
    float *t1=falloc((int64_t)C*c->kda_hd), *graw=falloc((int64_t)C*P), *braw=falloc((int64_t)C*H);
    /* Head-sharded: rank r owns heads [h0,h1). Buffers stay full width so the
     * per-head loop and the conv/recurrent state keep their existing indexing;
     * only the slices this rank owns are filled, and `on` is reduced to full
     * width before o_proj. */
    int wsz=g_k3_tp_attn?k3_net_world():1, wrk=k3_net_rank();
    int h0=0, h1=H;
    if(wsz>1){ int per=(H+wsz-1)/wsz; h0=wrk*per; h1=h0+per; if(h1>H) h1=H; if(h0>H) h0=H; }
    int hn=h1-h0, pn=hn*hd, p0=h0*hd;
    if(wsz>1){ memset(q,0,(size_t)C*P*sizeof(float)); memset(k,0,(size_t)C*P*sizeof(float));
               memset(v,0,(size_t)C*P*sizeof(float)); memset(gp,0,(size_t)C*P*sizeof(float));
               memset(graw,0,(size_t)C*P*sizeof(float)); memset(braw,0,(size_t)C*H*sizeof(float));
               memset(on,0,(size_t)C*P*sizeof(float)); }
    KdaCtrlJob cj={a,x,t1,graw,braw,c->hidden,hd,p0,pn,h0,hn};
    pthread_t ctlth; int ctl_active=0;
#ifdef COLI_CUDA
    int ctl_wanted=1;
    { const char *e=getenv("K3_KDA_OVERLAP"); if(e) ctl_wanted=atoi(e); }
    int ctl_overlap=C==1 && g_k3_cuda && g_k3_dense_gpu && ctl_wanted;
    g_ctl_spawn=now_s();
    if(ctl_overlap) ctl_active=kda_control_submit(&cj,&ctlth);
#endif
    if(wsz>1 && hn>0){
        if(!a->sh_ready){
            /* offset 0: w_load_rows already loaded only [p0,p0+pn), so the
             * tensor IS the slice. The view is now the whole buffer. */
            a->qs=w_rows(&a->q,0,pn); a->ks=w_rows(&a->k,0,pn);
            a->vs=w_rows(&a->v,0,pn); a->gs=w_rows(&a->g,0,pn);
            a->os=w_cols(&a->o,p0,pn);          /* row-parallel o_proj */
            /* w_cols COPIED this rank's columns (a column slice is not
             * contiguous, so it cannot be sliced at load like the rows above).
             * The full o is dead from here on -- release it, 3.4 GB of KDA. */
            w_free_host(&a->o);
            /* K3_FUSE_QKVG=1: measured NEUTRAL (0.78 vs 0.77, inside run-to-run
             * noise) while costing ~6.4 GB RSS, so it is opt-in. It does what it
             * claims -- GPU tensor count 1021 -> 814, four launches per layer
             * become one -- which is itself the finding: attention responds
             * neither to halving its weight traffic nor to removing 207 round
             * trips per token, so the cost is somewhere neither hypothesis
             * reached and wants a CUDA profiler, not another guess. */
            a->fuse = getenv("K3_FUSE_QKVG") && atoi(getenv("K3_FUSE_QKVG"));
            if(a->fuse) a->qkvg=w_concat4(&a->q,&a->k,&a->v,&a->g,0,pn);
            a->sh_ready=1;
        }
        double kp0=now_s();
        float *dst[4]={q,k,v,gp};
        if(a->fuse){                       /* ONE call for all four, then scatter */
            float *tmp=falloc((int64_t)C*4*pn);
            w_matmul(tmp,x,&a->qkvg,C);
            for(int u2=0;u2<4;u2++)
                for(int t=0;t<C;t++)
                    memcpy(dst[u2]+(int64_t)t*P+p0,
                           tmp+(int64_t)t*4*pn+(int64_t)u2*pn,(size_t)pn*sizeof(float));
            free(tmp);
        } else {
            float *tmp=falloc((int64_t)C*pn*4);
            W *sw[4]={&a->qs,&a->ks,&a->vs,&a->gs};
            int batched=0;
#ifdef COLI_CUDA
            if(C==1){
                float *ys[4]={tmp,tmp+pn,tmp+2*(int64_t)pn,tmp+3*(int64_t)pn};
                const W *cw[4]={sw[0],sw[1],sw[2],sw[3]};
                batched=w_k3_dense_many(ys,x,cw,4);
            }
#endif
            if(!batched)
                for(int u2=0;u2<4;u2++) w_matmul(tmp+(int64_t)u2*C*pn,x,sw[u2],C);
            for(int u2=0;u2<4;u2++)
                for(int t=0;t<C;t++)
                    memcpy(dst[u2]+(int64_t)t*P+p0,
                           tmp+(int64_t)u2*C*pn+(int64_t)t*pn,(size_t)pn*sizeof(float));
            free(tmp);
        }
        m->t_kproj+=now_s()-kp0;
    } else if(hn>0){
        double kp0=now_s();
        w_matmul(q,x,&a->q,C); w_matmul(k,x,&a->k,C); w_matmul(v,x,&a->v,C);
        w_matmul(gp,x,&a->g,C);
        m->t_kproj+=now_s()-kp0;
    }
    double kf0=now_s();
    if(ctl_active) kda_control_join(ctl_active,ctlth);
    else if(C==1){ double a=now_s(); kda_control_b1(&cj); g_ctl_secs+=now_s()-a; }
    else {
        w_matmul(t1,x,&a->fa,C);
        if(wsz>1){
            for(int t=0;t<C;t++){
                w_matmul(graw+(int64_t)t*P+p0,t1+(int64_t)t*c->kda_hd,&a->fb,1);
                w_matmul(braw+(int64_t)t*H+h0,x+(int64_t)t*c->hidden,&a->bp,1);
            }
        } else {
            w_matmul(graw,t1,&a->fb,C);
            w_matmul(braw,x,&a->bp,C);
        }
    }
    m->t_ctljoin+=now_s()-kf0;
    m->t_kproj+=now_s()-kf0;
    float qscale=1.f/sqrtf((float)hd);
    for(int t=0;t<C;t++){
        float *qt=q+(int64_t)t*P, *kt=k+(int64_t)t*P, *tv=v+(int64_t)t*P;
        float *gpt=gp+(int64_t)t*P, *ont=on+(int64_t)t*P;
        const float *rgt=graw+(int64_t)t*P, *bt=braw+(int64_t)t*H;
        /* depthwise causal conv (window: oldest..newest) + SiLU, rolls forward */
        double kc0=now_s();
        float *wins[3]={m->cwq[li],m->cwk[li],m->cwv[li]};
        float *vecs[3]={qt,kt,tv}; float *taps[3]={a->conv_q,a->conv_k,a->conv_v};
        /* SERIAL BY DESIGN -- do not restore the `omp parallel for` here. This
         * is 4 taps over pn channels: ~30K ops per (layer,tensor), ~30us of
         * arithmetic. Three regions per layer x 69 KDA layers = 207 fork/joins
         * per token, and measured that cost 53.9s of a 475.8s 474-token run
         * (11% of total) for 2.5 MFLOP of real work -- ~550us of barrier per
         * region. Even at world=1 (pn=12288, ~120us serial) one fork/join
         * costs more than the whole loop. */
        /* The window used to be shifted down by one every token: K-1 float
         * moves per channel, 3 tensors x pn channels x 69 layers. K is a power
         * of two, so a rotating head indexes the same values in the same order
         * with no moves at all -- bit-exact, and it touches fewer bytes rather
         * than more (the rule the KV-decode hoist broke). */
        int hh = m->cwh[li];
        for(int w2=0;w2<3;w2++){
            float *win=wins[w2], *vec=vecs[w2]; const float *cw=taps[w2];
            for(int d=p0;d<p0+pn;d++){
                float *wd=win+(int64_t)d*K;
                wd[hh]=vec[d];                       /* newest overwrites oldest */
                float acc=0; const float *cd=cw+(int64_t)d*K;
                for(int j=0;j<K;j++) acc+=cd[j]*wd[(hh+1+j)&(K-1)];
                vec[d]=siluf_(acc);
            }
        }
        m->cwh[li]=(int8_t)((hh+1)&(K-1));
        m->t_kconv+=now_s()-kc0;
        double kh0=now_s();
        #pragma omp parallel for schedule(static)
        for(int h=h0;h<h1;h++){
            const float *qh=qt+(int64_t)h*hd, *kh=kt+(int64_t)h*hd, *vh=tv+(int64_t)h*hd;
            float qn[512], kn[512], alpha[512], kS[512], vt[512], oh[512];
            float sq=0,sk=0;
            for(int i=0;i<hd;i++){ sq+=qh[i]*qh[i]; sk+=kh[i]*kh[i]; }
            sq=1.f/sqrtf(sq+1e-6f); sk=1.f/sqrtf(sk+1e-6f);
            for(int i=0;i<hd;i++){ qn[i]=qh[i]*sq*qscale; kn[i]=kh[i]*sk; }
            for(int i=0;i<hd;i++){
                float z=rgt[(int64_t)h*hd+i]+a->dt[(int64_t)h*hd+i];
                alpha[i]=expf(c->gate_lb*sigmoidf_(a->A[h]*z));
            }
            float beta=sigmoidf_(bt[h]);
            float *S=m->kstate[li]+(int64_t)h*hd*hd;
            memset(kS,0,sizeof(kS));
#ifdef __AVX2__
            if(!(hd&7)){
                for(int kk=0;kk<hd;kk++){
                    float *row=S+(int64_t)kk*hd;
                    __m256 al8=_mm256_set1_ps(alpha[kk]), kv8=_mm256_set1_ps(kn[kk]);
                    for(int vv=0;vv<hd;vv+=8){
                        __m256 r=_mm256_mul_ps(_mm256_loadu_ps(row+vv),al8);
                        _mm256_storeu_ps(row+vv,r);
                        _mm256_storeu_ps(kS+vv,_mm256_fmadd_ps(kv8,r,_mm256_loadu_ps(kS+vv)));
                    }
                }
                for(int vv=0;vv<hd;vv++) vt[vv]=(vh[vv]-kS[vv])*beta;
                memset(oh,0,sizeof(oh));
                for(int kk=0;kk<hd;kk++){
                    float *row=S+(int64_t)kk*hd;
                    __m256 kv8=_mm256_set1_ps(kn[kk]), qq8=_mm256_set1_ps(qn[kk]);
                    for(int vv=0;vv<hd;vv+=8){
                        __m256 r=_mm256_fmadd_ps(kv8,_mm256_loadu_ps(vt+vv),_mm256_loadu_ps(row+vv));
                        _mm256_storeu_ps(row+vv,r);
                        _mm256_storeu_ps(oh+vv,_mm256_fmadd_ps(qq8,r,_mm256_loadu_ps(oh+vv)));
                    }
                }
            } else {
#endif
            for(int kk=0;kk<hd;kk++){
                float *row=S+(int64_t)kk*hd; float al=alpha[kk], kv=kn[kk];
                for(int vv=0;vv<hd;vv++){ row[vv]*=al; kS[vv]+=kv*row[vv]; }
            }
            for(int vv=0;vv<hd;vv++) vt[vv]=(vh[vv]-kS[vv])*beta;
            memset(oh,0,sizeof(oh));
            for(int kk=0;kk<hd;kk++){
                float *row=S+(int64_t)kk*hd; float kv=kn[kk], qq=qn[kk];
                for(int vv=0;vv<hd;vv++){ row[vv]+=kv*vt[vv]; oh[vv]+=qq*row[vv]; }
            }
#ifdef __AVX2__
            }
#endif
            /* per-head RMSNorm * sigmoid(full-rank gate) */
            double ms=0; for(int vv=0;vv<hd;vv++) ms+=(double)oh[vv]*oh[vv];
            float r=1.f/sqrtf((float)(ms/hd)+c->eps);
            float *dst=ont+(int64_t)h*hd;
            for(int vv=0;vv<hd;vv++) dst[vv]=oh[vv]*r*a->onw[vv]*sigmoidf_(gpt[(int64_t)h*hd+vv]);
        }
        m->t_khead+=now_s()-kh0;
    }
    double ko0=now_s();
    /* Gather the head slices back to full width before o_proj. Implemented as
     * a sum over zero-filled buffers rather than a true all-gather: each rank
     * wrote only its own heads and left the rest zero, so the reduction IS the
     * concatenation. Costs P floats (48 KB) instead of P/N, which at ~0.5 ms
     * per layer is cheaper than adding a second collective. o_proj itself
     * stays replicated -- it needs a strided COLUMN slice, which a row view
     * cannot express. */
    if(wsz>1 && hn>0){
        /* Row-parallel: each rank multiplies only its own head columns, so the
         * partial outputs sum to the full result. Reduces [C,hidden] rather
         * than gathering [C,P] -- 7168 floats instead of 12288, and the
         * separate gather disappears entirely. */
        float *cmp=falloc((int64_t)C*pn);
        for(int t=0;t<C;t++) memcpy(cmp+(int64_t)t*pn,on+(int64_t)t*P+p0,(size_t)pn*sizeof(float));
        w_matmul(out,cmp,&a->os,C);
        free(cmp);
        { double na=now_s(); k3_net_allreduce(out,(size_t)C*c->hidden); m->t_net_kda+=now_s()-na; }
    } else if(wsz>1){
        memset(out,0,(size_t)C*c->hidden*sizeof(float));
        { double na=now_s(); k3_net_allreduce(out,(size_t)C*c->hidden); m->t_net_kda+=now_s()-na; }
    } else w_matmul(out,on,&a->o,C);
    m->t_kout+=now_s()-ko0;
    free(q);free(k);free(v);free(gp);free(on);free(t1);free(graw);free(braw);
}

/* ---------- gated MLA layer (chunk of C tokens, NoPE, absorb; projections
 * batched, per-token causal attention — token t attends to 0..pos0+t) ------ */
static void mla_forward(Model *m, Layer *l, int li, const float *x, int pos0, int C, float *out){
    Cfg *c=&m->c; Mla *a=&l->m;
    int H=c->n_heads, qh=c->qk_head, vh=c->v_head, kvl=c->kv_lora, qr=c->qk_rope;
    float *qa=falloc((int64_t)C*c->q_lora), *qv=falloc((int64_t)C*H*qh);
    float *ckv=falloc((int64_t)C*(kvl+qr));
    float *gv=falloc((int64_t)C*H*vh), *ctx=falloc((int64_t)C*H*vh);
    double mt0=now_s();
    w_matmul(qa,x,&a->qa,C);
    for(int t=0;t<C;t++)
        rmsnorm_(qa+(int64_t)t*c->q_lora,qa+(int64_t)t*c->q_lora,a->qa_ln,c->q_lora,c->eps);
    int wsz=g_k3_tp_attn?k3_net_world():1, wrk=k3_net_rank();
    int mh0=0, mh1=H;
    if(wsz>1){ int per=(H+wsz-1)/wsz; mh0=wrk*per; mh1=mh0+per; if(mh1>H) mh1=H; if(mh0>H) mh0=H; }
    int mhn=mh1-mh0;
    if(wsz>1 && mhn>0 && !a->sh_ready){
        a->qbs=w_rows(&a->qb,0,mhn*qh);        /* already head-sliced at load */
        a->gs_=w_rows(&a->g, 0,mhn*vh);
        a->os =w_cols(&a->o, mh0*vh,mhn*vh);
        w_free_host(&a->o);                    /* copied by w_cols; see KDA note */
        a->sh_ready=1;
    }
    if(wsz>1 && mhn>0){
        memset(qv,0,(size_t)C*H*qh*sizeof(float));
        float *tq=falloc((int64_t)C*mhn*qh);
        w_matmul(tq,qa,&a->qbs,C);
        for(int t=0;t<C;t++) memcpy(qv+(int64_t)t*H*qh+mh0*qh,tq+(int64_t)t*mhn*qh,
                                    (size_t)mhn*qh*sizeof(float));
        free(tq);
    } else w_matmul(qv,qa,&a->qb,C);
    m->t_mproj+=now_s()-mt0; mt0=now_s();
    w_matmul(ckv,x,&a->kva,C);
    m->t_mproj+=now_s()-mt0; mt0=now_s();
    for(int t=0;t<C;t++){                            /* append the whole chunk to the
                                                      * cache first: token t's scores
                                                      * only read rows 0..pos0+t */
        kvq *Lrow=m->Lc[li]+(int64_t)(pos0+t)*kvl, *Rrow=m->Rc[li]+(int64_t)(pos0+t)*qr;
        const float *cv=ckv+(int64_t)t*(kvl+qr);
        float lt[4096];                              /* kvl <= 4096, enforced at cfg load */
        rmsnorm_(lt,cv,a->kva_ln,kvl,c->eps);
        for(int i=0;i<kvl;i++) Lrow[i]=kv_enc(lt[i]);
        for(int i=0;i<qr;i++)  Rrow[i]=kv_enc(cv[kvl+i]);  /* NoPE: cached raw, no rotation */
    }
    m->t_mcache+=now_s()-mt0; mt0=now_s();
    if(wsz>1 && mhn>0){
        memset(gv,0,(size_t)C*H*vh*sizeof(float));
        float *tg=falloc((int64_t)C*mhn*vh);
        w_matmul(tg,x,&a->gs_,C);
        for(int t=0;t<C;t++) memcpy(gv+(int64_t)t*H*vh+mh0*vh,tg+(int64_t)t*mhn*vh,
                                    (size_t)mhn*vh*sizeof(float));
        free(tg);
    } else w_matmul(gv,x,&a->g,C);
    m->t_mproj+=now_s()-mt0; mt0=now_s();
    int Hl=mh1-mh0;
    if(g_mattb && Hl>1){
        /* Head-batched MLA attention.
         *
         * Every head reads the SAME KV rows, but the per-head loop walks the
         * whole cache once PER HEAD -- 24 passes over nt*kvl per layer per
         * token on this rank. Vectorising kv_dec took matt from 7.73 to 2.86
         * s/100 tokens at ctx 1963 and thereby moved the binding constraint
         * from the conversion onto that access pattern.
         *
         * So put t in the outer position and the heads inside: each L[t] is
         * loaded once and consumed by all 24 heads out of L1, and DRAM sees
         * the cache once per layer rather than once per head. The absorbed
         * queries (Hl x kvl = 49 KB) become the re-read operand instead, but
         * they are L2-resident where the cache is not.
         *
         * Bit-exact against the per-head loop: each sc[h][t] is the same dot
         * product, and splitting kvl across threads gives every clat[h][i] a
         * private accumulator that sees the same t in the same order -- no
         * reduction, no races. Confirmed empirically: 6/6 prompts byte-identical.
         *
         * MEASURED SLOWER, so this is off by default (K3_MATT_BATCH=1 to try
         * it again). A/B at ctx 1963, one binary:
         *
         *                    per-head   batched
         *     decode tok/s      3.597     3.098    -14%
         *     matt (decode)     2.806     3.369
         *     matt (prefill)   34.618    60.207    +74%
         *
         * The premise was wrong. "24 heads each re-read the cache" counts
         * LOGICAL reads: the heads run concurrently on the same L, so the
         * hardware was already serving those from cache and DRAM never saw 24
         * passes. Batching does not remove traffic, it swaps which operand is
         * re-read -- per t we now touch all Hl*kvl of absorbed query (49 KB,
         * scattered across 24 arrays) instead of one contiguous 2 KB row -- and
         * it turns one OpenMP region per token into four. Prefill suffers most
         * because C=32 pays those barriers 32 times per chunk.
         *
         * Seventh instance of the pattern this engine keeps showing: on a
         * shared-memory box, REMOVING work wins and MOVING or BATCHING work
         * loses. The kv_dec vectorisation above removed instructions and won
         * 18%; this rearranges the same instructions and loses 14%. */
        int64_t ntmax=(int64_t)pos0+C;
        float *qab=falloc((int64_t)Hl*kvl), *cla=falloc((int64_t)Hl*kvl);
        float *scm=falloc((int64_t)Hl*ntmax);
        for(int tt=0;tt<C;tt++){
            int nt=pos0+tt+1;
            const float *qvt=qv+(int64_t)tt*H*qh, *gvt=gv+(int64_t)tt*H*vh;
            float *ctxt=ctx+(int64_t)tt*H*vh;
            #pragma omp parallel for schedule(static)
            for(int hh=0;hh<Hl;hh++){
                const float *qp=qvt+(int64_t)(mh0+hh)*qh;
                int rbase=(mh0+hh)*(c->qk_nope+vh);
                float *qa=qab+(int64_t)hh*kvl; memset(qa,0,kvl*sizeof(float));
                for(int d=0;d<c->qk_nope;d++) w_addrow(&a->kvb,rbase+d,qp[d],qa);
            }
            #pragma omp parallel for schedule(static)
            for(int t=0;t<nt;t++){
                const kvq *Lt=m->Lc[li]+(int64_t)t*kvl, *Rt=m->Rc[li]+(int64_t)t*qr;
                for(int hh=0;hh<Hl;hh++){
                    const float *qrp=qvt+(int64_t)(mh0+hh)*qh+c->qk_nope;
                    scm[(int64_t)hh*nt+t]=(kvdot_(qab+(int64_t)hh*kvl,Lt,kvl)
                                          +kvdot_(qrp,Rt,qr))*c->attn_scale;
                }
            }
            #pragma omp parallel for schedule(static)
            for(int hh=0;hh<Hl;hh++) softmax_(scm+(int64_t)hh*nt,nt);
            memset(cla,0,(size_t)Hl*kvl*sizeof(float));
            #pragma omp parallel
            {
                int nth=omp_get_num_threads(), tid=omp_get_thread_num();
                int per=(((kvl+nth-1)/nth)+7)&~7;      /* keep slices 8-aligned */
                int i0=tid*per, i1=i0+per>kvl?kvl:i0+per;
                if(i0<kvl) for(int t=0;t<nt;t++){
                    const kvq *Lt=m->Lc[li]+(int64_t)t*kvl;
                    for(int hh=0;hh<Hl;hh++)
                        kvaxpy_(cla+(int64_t)hh*kvl+i0,Lt+i0,scm[(int64_t)hh*nt+t],i1-i0);
                }
            }
            #pragma omp parallel for schedule(static)
            for(int hh=0;hh<Hl;hh++){
                int h=mh0+hh, rbase=h*(c->qk_nope+vh);
                float *cx=ctxt+(int64_t)h*vh, *cl=cla+(int64_t)hh*kvl;
                for(int d=0;d<vh;d++)
                    cx[d]=w_rowdot(&a->kvb,rbase+c->qk_nope+d,cl)*sigmoidf_(gvt[(int64_t)h*vh+d]);
            }
        }
        free(qab); free(cla); free(scm);
        goto matt_done;
    }
    for(int tt=0;tt<C;tt++){
        int nt=pos0+tt+1;
        const float *qvt=qv+(int64_t)tt*H*qh, *gvt=gv+(int64_t)tt*H*vh;
        float *ctxt=ctx+(int64_t)tt*H*vh;
        #pragma omp parallel for schedule(static)
        for(int h=mh0;h<mh1;h++){
            const float *qp=qvt+(int64_t)h*qh, *qrp=qp+c->qk_nope;
            int rbase=h*(c->qk_nope+vh);
            float qabs[4096]; memset(qabs,0,kvl*sizeof(float));
            for(int d=0;d<c->qk_nope;d++) w_addrow(&a->kvb,rbase+d,qp[d],qabs);
            float scbuf[512]; float *sc = (nt<=512) ? scbuf : falloc(nt);
            /* Branch once per head, never inside the reduction, so the scalar
             * arm measures what it measured before this change. */
            if(g_kvneon){
                for(int t=0;t<nt;t++){
                    const kvq *Lt=m->Lc[li]+(int64_t)t*kvl, *Rt=m->Rc[li]+(int64_t)t*qr;
                    sc[t]=(kvdot_(qabs,Lt,kvl)+kvdot_(qrp,Rt,qr))*c->attn_scale;
                }
            } else {
                for(int t=0;t<nt;t++){
                    const kvq *Lt=m->Lc[li]+(int64_t)t*kvl, *Rt=m->Rc[li]+(int64_t)t*qr;
                    float s2=0; for(int i=0;i<kvl;i++) s2+=qabs[i]*kv_dec(Lt[i]);
                    for(int i=0;i<qr;i++) s2+=qrp[i]*kv_dec(Rt[i]);
                    sc[t]=s2*c->attn_scale;
                }
            }
            softmax_(sc,nt);
            float clat[4096]; memset(clat,0,kvl*sizeof(float));
            if(g_kvneon){
                for(int t=0;t<nt;t++) kvaxpy_(clat,m->Lc[li]+(int64_t)t*kvl,sc[t],kvl);
            } else {
                for(int t=0;t<nt;t++){
                    const kvq *Lt=m->Lc[li]+(int64_t)t*kvl; float s2=sc[t];
                    for(int i=0;i<kvl;i++) clat[i]+=s2*kv_dec(Lt[i]);
                }
            }
            if(nt>512) free(sc);
            float *cx=ctxt+(int64_t)h*vh;
            for(int d=0;d<vh;d++)
                cx[d]=w_rowdot(&a->kvb,rbase+c->qk_nope+d,clat)*sigmoidf_(gvt[(int64_t)h*vh+d]);
        }
    }
matt_done:
    m->t_matt+=now_s()-mt0; mt0=now_s();
    if(wsz>1 && mhn>0){
        float *cc=falloc((int64_t)C*mhn*vh);
        for(int t=0;t<C;t++) memcpy(cc+(int64_t)t*mhn*vh,ctx+(int64_t)t*H*vh+mh0*vh,
                                    (size_t)mhn*vh*sizeof(float));
        w_matmul(out,cc,&a->os,C);
        free(cc);
        { double na=now_s(); k3_net_allreduce(out,(size_t)C*c->hidden); m->t_net_mla+=now_s()-na; }
    } else if(wsz>1){
        memset(out,0,(size_t)C*c->hidden*sizeof(float));
        { double na=now_s(); k3_net_allreduce(out,(size_t)C*c->hidden); m->t_net_mla+=now_s()-na; }
    } else w_matmul(out,ctx,&a->o,C);
    m->t_mout+=now_s()-mt0;
    free(qa);free(qv);free(ckv);free(gv);free(ctx);
}

/* ---------- routed experts: LRU + pread from the shards ----------
 * Loads are issued in PARALLEL (OMP over the token's misses, working-set
 * slots) and, when K3_DIRECT=1 (default) and st.h has an O_DIRECT twin fd,
 * bypass the page cache: measured on the box 7.1 GB/s direct vs 2.9 buffered
 * (and ~1.8 effective once the resident weights leave no cache headroom). */
static Slot *slot_find(Model *m, int li, int eid){
    LCache *lc=&m->ecache[li];
    for(int i=0;i<lc->n;i++) if(lc->s[i].eid==eid){ m->hits++; lc->s[i].used=++m->clock; return &lc->s[i]; }
    return NULL;
}
static void expert_read(Model *m, int li, int eid, Slot *s){
    if(!s->base){
        if(posix_memalign((void**)&s->base,4096,(size_t)m->e_slot+8192)){
            fprintf(stderr,"OOM expert slot\n"); exit(1); }
#ifdef COLI_CUDA
        /* Register at birth: slots are allocated once and thereafter only
         * SWAPPED between the LRU and the working set, so one mapping per
         * allocation covers every expert that will ever occupy it. */
        if(g_k3_expert_gpu && !coli_k3_register(s->base,(size_t)m->e_slot+8192)){
            fprintf(stderr,"[K3/EXP] slot registration failed — expert GPU path off\n");
            g_k3_expert_gpu=0;
        }
#endif
    }
    ERef *er=&m->eref[(int64_t)li*m->c.n_experts+eid];
    int64_t sizes[6]={m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s,m->e_w1p,m->e_w1s};
    if(er->fd[0]<0){ fprintf(stderr,"[K3] expert L%d E%d missing on disk\n",li,eid); exit(1); }
    if(er->contig){
        int dfd = g_k3_direct ? (m->w2_dfd ? m->w2_dfd : st_direct_fd(&m->S,er->fd[0])) : -1;
        if(dfd>=0){
            /* aligned window read; sub-4K head/tail slack handled explicitly.
             * The tail past the last aligned block (or past EOF) is fetched
             * with a tiny buffered pread — O_DIRECT wants aligned lengths. */
            int64_t a0=er->off[0]&~4095LL, pad=er->off[0]-a0;
            int64_t want=pad+m->e_slot;
            struct stat sb;
            int64_t dlen=(want+4095)&~4095LL;
            if(fstat(dfd,&sb)==0 && a0+dlen>sb.st_size) dlen=(sb.st_size-a0)&~4095LL;
            if(dlen>0) st_pread_full(dfd,s->base,dlen,a0,"pread expert direct");
            if(dlen<want)
                st_pread_full(er->fd[0],s->base+dlen,want-dlen,a0+dlen,"pread expert tail");
            s->buf=s->base+pad;
        } else {
            st_pread_full(er->fd[0],s->base,m->e_slot,er->off[0],"pread expert");
            s->buf=s->base;
        }
    } else {
        uint8_t *dst=s->base;
        for(int k=0;k<6;k++){
            if(er->fd[k]<0){ fprintf(stderr,"[K3] expert L%d E%d tensor %d missing on disk\n",li,eid,k); exit(1); }
            st_pread_full(er->fd[k],dst,sizes[k],er->off[k],"pread expert");
            dst+=sizes[k];
        }
        s->buf=s->base;
    }
    s->eid=eid;
}

static inline float situf_(float g, float u, float b1, float b2){
    return b1*tanhf(g/b1)*sigmoidf_(g) * b2*tanhf(u/b2);
}

/* u += wk * E(z) for one loaded expert slot (SiTU-GLU in the latent).
 * gate/up are [moe_inter] scratch, hz is [latent] scratch. */
static void expert_apply(Model *m, Slot *s, const float *z, float wk,
                         float *u, float *gate, float *up, float *hz){
    Cfg *c=&m->c;
    uint8_t *w1p=s->buf, *w1s=w1p+m->e_w1p, *w2p=w1s+m->e_w1s, *w2s=w2p+m->e_w2p,
            *w3p=w2s+m->e_w2s, *w3s=w3p+m->e_w1p;
#ifdef COLI_CUDA
    if(g_k3_expert_gpu){
        /* One call replaces all three matmul_mxfp4's; SiTU is fused into the
         * first kernel's epilogue so `up` never leaves the device. */
        double tk0=now_s();
        int ok = m->w1_mode ? coli_k3_expert_w1(w1p,w1s,w2p,w2s,w3p,w3s,hz,z,
                                c->latent,c->moe_inter,c->situ_b1,c->situ_b2)
               : m->w2_fd    ? coli_k3_expert_w2(w1p,w1s,w2p,w2s,w3p,w3s,hz,z,
                                c->latent,c->moe_inter,c->situ_b1,c->situ_b2)
                             : coli_k3_expert(w1p,w1s,w2p,w2s,w3p,w3s,hz,z,
                                c->latent,c->moe_inter,c->situ_b1,c->situ_b2);
        m->t_ekernel+=now_s()-tk0; m->n_ekernel++;
        if(ok){
            g_k3_exp_gpu++;
            for(int i=0;i<c->latent;i++) u[i]+=wk*hz[i];
            return;
        }
        fprintf(stderr,"[K3/EXP] expert kernel failed — falling back to CPU\n");
        g_k3_expert_gpu=0;
    }
    /* quant.h has no 2-bit kernel: decoding a w2 slot as e2m1 would read
     * 4-bit nibbles out of 2-bit codes and produce plausible-looking garbage
     * rather than an error. Refuse instead. */
    if(m->w2_fd){
        fprintf(stderr,"[K3/EXP] K3_W2_DIR needs the GPU expert kernel "
            "(K3_GPUS + K3_EXPERT_GPU=1); there is no 2-bit CPU path\n"); exit(1); }
    g_k3_exp_cpu++;
#endif
    void (*mm)(float*,const float*,const uint8_t*,const uint8_t*,int,int,int)
        = g_k3_idot ? matmul_mxfp4_i8 : matmul_mxfp4;
    mm(gate,z,w1p,w1s,1,c->latent,c->moe_inter);
    mm(up,z,w3p,w3s,1,c->latent,c->moe_inter);
    for(int i=0;i<c->moe_inter;i++) gate[i]=situf_(gate[i],up[i],c->situ_b1,c->situ_b2);
    mm(hz,gate,w2p,w2s,1,c->moe_inter,c->latent);
    for(int i=0;i<c->latent;i++) u[i]+=wk*hz[i];
}

/* ---------- async loader pool (K3_PIPE): expert preads overlap compute ----
 * A batch of jobs is submitted per token+layer; the compute loop below waits
 * per-expert on its ready flag, so expert j's math runs while j+1.. load.
 * One batch in flight at a time (the submitter consumes every job before the
 * next submit), so the flags need no generation counter. */
#define LP_MAX 64
typedef struct { int li, eid; Slot *s; } LJob;
static struct {
    pthread_t th[16]; int nth, started;
    pthread_mutex_t mx; pthread_cond_t cv;
    Model *m;
    LJob job[LP_MAX];
    _Atomic int ready[LP_MAX];
    _Atomic int next; int count;
} g_lp = { .mx=PTHREAD_MUTEX_INITIALIZER, .cv=PTHREAD_COND_INITIALIZER };

static void *lp_main(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&g_lp.mx);
        while(atomic_load_explicit(&g_lp.next,memory_order_relaxed)>=g_lp.count)
            pthread_cond_wait(&g_lp.cv,&g_lp.mx);
        int idx=atomic_fetch_add_explicit(&g_lp.next,1,memory_order_relaxed);
        pthread_mutex_unlock(&g_lp.mx);
        if(idx>=g_lp.count) continue;
        LJob *j=&g_lp.job[idx];
        expert_read(g_lp.m,j->li,j->eid,j->s);
        atomic_store_explicit(&g_lp.ready[idx],1,memory_order_release);
    }
    return NULL;
}
static void lp_start(void){
    if(g_lp.started) return;
    g_lp.nth = getenv("K3_LOAD_THREADS")?atoi(getenv("K3_LOAD_THREADS")):4;
    if(g_lp.nth<1) g_lp.nth=1;
    if(g_lp.nth>16) g_lp.nth=16;
    g_lp.count=0; atomic_store(&g_lp.next,0);
    for(int i=0;i<g_lp.nth;i++)
        if(pthread_create(&g_lp.th[i],NULL,lp_main,NULL)){
            fprintf(stderr,"[K3] K3_PIPE: pthread_create failed, falling back\n");
            g_k3_pipe=0; g_lp.nth=i; break;
        }
    g_lp.started=1;
}
static void lp_submit(Model *m, int n){
    pthread_mutex_lock(&g_lp.mx);
    g_lp.m=m;
    for(int q=0;q<n;q++) atomic_store_explicit(&g_lp.ready[q],0,memory_order_relaxed);
    atomic_store_explicit(&g_lp.next,0,memory_order_relaxed);
    g_lp.count=n;
    pthread_cond_broadcast(&g_lp.cv);
    pthread_mutex_unlock(&g_lp.mx);
}

/* the general expert pass for one layer over a CHUNK of positions: nu unique
 * experts, expert j applied to pcnt[j] positions (poslist/wlist rows starting
 * at pfirst[j]). Loads run in blocks of <=LP_MAX working-set slots, pipelined
 * with compute under K3_PIPE (expert j's matmuls overlap expert j+1's read),
 * else all-parallel up front. Z/U are [C, stride] position-major. */
static void experts_apply_union(Model *m, int li, int nu, const int *uids,
                                const int *pfirst, const int *pcnt,
                                const int *poslist, const float *wlist,
                                const float *Z, int stride, int C, float *U,
                                float *gate, float *up, float *hz){
    for(int base=0;base<nu;base+=LP_MAX){
        int nb=nu-base<LP_MAX?nu-base:LP_MAX;
        Slot *use[LP_MAX]; int missk[LP_MAX]; int qof[LP_MAX]; int nmiss=0;
        for(int j=0;j<nb;j++){
            use[j]=slot_find(m,li,uids[base+j]); qof[j]=-1;
            if(!use[j]){ m->miss++; use[j]=&m->ws[nmiss]; qof[j]=nmiss; missk[nmiss++]=j; }
        }
        if(nmiss){
            if(g_k3_pipe && !g_lp.started) lp_start();
            if(g_k3_pipe){
                for(int q=0;q<nmiss;q++){
                    g_lp.job[q].li=li; g_lp.job[q].eid=uids[base+missk[q]]; g_lp.job[q].s=&m->ws[q];
                }
                lp_submit(m,nmiss);
            } else {
                double t0=now_s();
                #pragma omp parallel for schedule(dynamic,1)
                for(int q=0;q<nmiss;q++) expert_read(m,li,uids[base+missk[q]],&m->ws[q]);
                m->t_eload+=now_s()-t0;
            }
            m->ebytes+=(uint64_t)nmiss*(uint64_t)m->e_slot;
        }
        /* Decode (C==1) applies every expert to the SAME z, so a whole layer
         * fits one launch. Prefill cannot: an expert serves several positions,
         * each with its own z, and the batched kernel shares one. */
        int batchable=0;
#ifdef COLI_CUDA
        /* ON by default since the 1-bit batch kernels landed. The old note said
         * batching lost 0.58 -> 0.51 tok/s because waiting for every expert to
         * load beat overlapping load with compute -- that was measured when the
         * expert tier still missed often. It no longer does (hit rate 99.6%),
         * and ncu says why batching now wins: the single-expert kernels run at
         * Waves Per SM = 1.56 with DRAM at 19% of peak, i.e. tail-bound, not
         * bandwidth-bound. One launch for a layer's ~4.6 experts takes that to
         * ~7 waves. Measured 4.502/4.538 -> 4.648/4.668, expert 2.691 -> 2.121
         * s/100, six-prompt output 6/6 byte-identical. */
        /* !w1_mode is required: coli_k3_expert_batch_w2 launches the 2-bit
         * kernels unconditionally, so on a 1-bit store it walks every slot with
         * 2-bit strides and runs off the end -- K3_EXPERT_BATCH=1 killed the
         * engine with an illegal memory access. The single-expert path at
         * expert_apply already dispatches on w1_mode; this guard did not. */
        batchable = (g_k3_expert_batch && C==1 && m->w2_fd
                     && g_k3_expert_gpu && nb<=64);
        for(int j=0;j<nb&&batchable;j++) if(pcnt[base+j]!=1) batchable=0;
#endif
        for(int j=0;j<nb;j++){
            if(g_k3_pipe && qof[j]>=0 &&
               !atomic_load_explicit(&g_lp.ready[qof[j]],memory_order_acquire)){
                double t0=now_s();      /* t_eload = UN-hidden I/O (wait) time */
                while(!atomic_load_explicit(&g_lp.ready[qof[j]],memory_order_acquire))
                    usleep(50);
                m->t_eload+=now_s()-t0;
            }
            if(batchable) continue;     /* just wait here; compute below */
            int f=pfirst[base+j];
            for(int p2=0;p2<pcnt[base+j];p2++){
                int t=poslist[f+p2];
                expert_apply(m,use[j],Z+(int64_t)t*stride,wlist[f+p2],
                             U+(int64_t)t*stride,gate,up,hz);
            }
        }
#ifdef COLI_CUDA
        if(batchable){
            const void *p1[64],*s1[64],*p2b[64],*s2b[64],*p3[64],*s3[64];
            for(int j=0;j<nb;j++){
                uint8_t *b=use[j]->buf;
                p1[j]=b;                                   s1[j]=b+m->e_w1p;
                p2b[j]=b+m->e_w1p+m->e_w1s;                s2b[j]=(uint8_t*)p2b[j]+m->e_w2p;
                p3[j]=(uint8_t*)s2b[j]+m->e_w2s;           s3[j]=(uint8_t*)p3[j]+m->e_w1p;
            }
            /* w1_mode picks the sign-bit kernels; the 2-bit entry point would
             * walk a 1-bit slot with 2-bit strides and run off the end. */
            int bok = m->w1_mode
              ? coli_k3_expert_batch_w1(p1,s1,p2b,s2b,p3,s3,nb,m->hz_batch,Z,
                                        m->c.latent,m->c.moe_inter,m->c.situ_b1,m->c.situ_b2)
              : coli_k3_expert_batch_w2(p1,s1,p2b,s2b,p3,s3,nb,m->hz_batch,Z,
                                        m->c.latent,m->c.moe_inter,m->c.situ_b1,m->c.situ_b2);
            if(bok){
                g_k3_exp_gpu+=nb;
                for(int j=0;j<nb;j++){
                    const float *h=m->hz_batch+(int64_t)j*m->c.latent;
                    float wk=wlist[pfirst[base+j]];
                    for(int i=0;i<m->c.latent;i++) U[i]+=wk*h[i];
                }
            } else {   /* batch refused: fall back to one-at-a-time, same math */
                for(int j=0;j<nb;j++){
                    int f=pfirst[base+j];
                    expert_apply(m,use[j],Z,wlist[f],U,gate,up,hz);
                }
            }
        }
#endif
        /* promotion: swap the freshly-read slots into the layer LRU */
        LCache *lc=&m->ecache[li];
        int promo = nmiss<lc->cap ? nmiss : lc->cap;
        for(int a=0;a<promo;a++){
            int q=nmiss-1-a; Slot *dst;
            if(lc->n<lc->cap) dst=&lc->s[lc->n++];
            else { int lru=0; for(int i=1;i<lc->n;i++) if(lc->s[i].used<lc->s[lru].used) lru=i; dst=&lc->s[lru]; }
            Slot tmp=*dst; *dst=m->ws[q]; m->ws[q]=tmp;
            dst->used=++m->clock;
        }
    }
}


#ifdef COLI_CUDA
/* Register + device-mirror a weight and hand back the pointers the K3 kernels
 * want. Same latch as w_matmul: one attempt, then remember the answer. */
static int w_k3_ptrs(const W *w, const void **blob, const float **sc){
    if(!g_k3_dense_gpu || !g_k3_cuda || w->k3_dense_off || w->fmt!=4) return 0;
    W *mw=(W*)w;
    int64_t rb=((int64_t)w->I+1)/2, nsc=((int64_t)w->I+w->gs-1)/w->gs;
    if(!mw->k3_reg){
        if(coli_k3_register((void*)w_blob(w),(size_t)w->O*rb) &&
           coli_k3_register(w->s,(size_t)w->O*nsc*sizeof(float))) mw->k3_reg=1;
        else { mw->k3_dense_off=1; return 0; }
    }
    if(!mw->k3_dev_tried){
        void *dw=coli_k3_devmirror(w_blob(w),(size_t)w->O*rb);
        if(dw){ void *ds=coli_k3_devmirror(w->s,(size_t)w->O*nsc*sizeof(float));
                if(ds){ mw->k3_dw=dw; mw->k3_ds=ds;
                        k3_mir_register((void*)w_blob(w),(size_t)w->O*rb);
                        k3_mir_register((void*)w->s,(size_t)w->O*nsc*sizeof(float)); } }
        mw->k3_dev_tried=1;
    }
    *blob = mw->k3_dw ? (const void*)mw->k3_dw : w_blob(w);
    *sc   = mw->k3_dw ? (const float*)mw->k3_ds : w->s;
    return 1;
}

/* Fused shared-expert gate/up + SiTU on the GPU.
 *
 * NOT bit-exact: CUDA's transcendentals differ from glibc's in the last ulp, so
 * this is gated on end-to-end PCC, not the reference hash. It removes the two
 * separate gate/up matmuls, their two downloads and one upload, and the whole
 * serial situf_ loop from the CPU. K3_SITU_GPU=1 to enable. */
static int g_situ_gpu = -1;
static int k3_situ_fused(float *sg, const float *x, const Moe *o, int C, int shi,
                         float b1, float b2){
    if(g_situ_gpu < 0){ const char *e=getenv("K3_SITU_GPU"); g_situ_gpu = e?atoi(e):1; }
    if(!g_situ_gpu || !g_k3_ready) return 0;
    const W *wg=&o->sh_gate, *wu=&o->sh_up;
    if(wg->fmt!=4 || wu->fmt!=4 || wg->gs!=wu->gs) return 0;
    if(wg->O!=shi || wu->O!=shi || wg->I!=wu->I) return 0;
    const void *pg,*pu; const float *sgs,*sus;
    if(!w_k3_ptrs(wg,&pg,&sgs) || !w_k3_ptrs(wu,&pu,&sus)) return 0;
    return coli_k3_gate_up_situ(sg, x, pg, sgs, pu, sus, C, wg->I, shi, wg->gs, b1, b2);
}
#endif

static void moe_forward(Model *m, Layer *l, int li, const float *x, int C, float *out){
    Cfg *c=&m->c; Moe *o=&l->moe;
    int E=c->n_experts, K=c->topk, LT=c->latent, MI=c->moe_inter;
    float *sco=falloc((int64_t)C*E);
    double tp0=now_s();
    int r2done=0;
#ifdef COLI_CUDA
    /* Runs for any C -- one call per token -- so the teacher-forced capture,
     * which is prefill only, measures this path and not the f32 fallback. */
    if(o->r_q8 && g_k3_ready){
        static int reg=0; static float dl=-1.f;
        if(dl<0){ const char *e=getenv("K3_ROUTER2_DELTA"); dl=e?(float)atof(e):0.01f; }
        if(!o->r_reg){
            /* int8 copy mirrored (read every token), f32 rows only registered --
             * the exact pass touches ~29 of 896 rows, so the SMMU path is fine
             * and 2.36 GB of mirror budget stays with the dense weights. */
            void *dq=coli_k3_devmirror(o->r_q8,(size_t)E*c->hidden);
            void *ds=coli_k3_devmirror(o->r_s8,(size_t)E*sizeof(float));
            void *db=coli_k3_devmirror(o->rbias,(size_t)E*sizeof(float));
            if(dq&&ds&&db&&coli_k3_register(o->router,(size_t)E*c->hidden*sizeof(float))){
                o->r_dq=dq; o->r_ds=(float*)ds; o->r_db=(float*)db; o->r_reg=1;
            } else o->r_reg=-1;
        }
        if(o->r_reg==1){
            r2done=1;
            for(int t=0;t<C && r2done;t++)
                r2done = coli_k3_router2(sco+(int64_t)t*E, x+(int64_t)t*c->hidden,
                                         o->r_dq,o->r_ds,o->router,o->r_db,
                                         E,c->hidden,K,dl);
        }
        (void)reg;
    }
#endif
    if(!r2done) w_matmul(sco,x,&o->router_w,C);
    m->t_router+=now_s()-tp0; tp0=now_s();
    int *idxs=malloc((size_t)C*K*sizeof(int)); float *wsels=falloc((int64_t)C*K);
    int *keff=malloc((size_t)C*sizeof(int));
    float *bias_sc=falloc(E); unsigned char *used=malloc((size_t)E);
    if(!idxs||!keff||!used){fprintf(stderr,"OOM moe sel\n");exit(1);}
    for(int t=0;t<C;t++){
        float *st=sco+(int64_t)t*E;
        for(int e=0;e<E;e++) st[e]=sigmoidf_(st[e]);
        int *idx=idxs+(int64_t)t*K; float *wsel=wsels+(int64_t)t*K;
        /* Selection was O(K^2 * E): each of the K picks rescanned all E experts
         * AND walked the already-chosen list to test membership, and recomputed
         * st[e]+rbias[e] on every pass. A used[] flag and one precomputed
         * biased score give the identical sequence -- same strict `>` so the
         * lowest index still wins ties, same raw-sigmoid weight -- for ~9x less
         * work. Bit-exact by construction: only the membership test and a
         * common subexpression changed. */
        for(int e=0;e<E;e++){ bias_sc[e]=st[e]+o->rbias[e]; used[e]=0; }
        /* Does safe int8 pruning of the router have any headroom? A two-pass
         * router only pays if the gap between the 16th and 17th biased score
         * exceeds the worst-case int8 error, because anything closer than that
         * must be re-scored exactly. Bound per row: (s/2)*||x||_1 with
         * s = max|w_row|/127. Printed once. */
        if(g_dense_census){
            static int shown=0;
            if(!shown){ shown=1;
                float *tmp=falloc(E);
                for(int e=0;e<E;e++) tmp[e]=bias_sc[e];
                for(int a2=0;a2<18;a2++) for(int b2=a2+1;b2<E;b2++)
                    if(tmp[b2]>tmp[a2]){ float t=tmp[a2]; tmp[a2]=tmp[b2]; tmp[b2]=t; }
                double l1=0; for(int i=0;i<c->hidden;i++) l1+=fabs(x[i]);
                float wmax=0;
                if(o->router){ const float *rw=o->router;
                    for(int64_t i=0;i<(int64_t)c->hidden;i++){ float a=fabsf(rw[i]); if(a>wmax)wmax=a; } }
                else if(o->router_w.fmt==1) wmax=o->router_w.s[0]*127.f;
                double bound=(wmax/127.0/2.0)*l1;
                /* How many experts sit within delta of the 16th score? That
                 * is the candidate count a two-pass router would have to
                 * re-score exactly, and it decides whether the second pass is
                 * cheap enough to be worth the first. */
                { double k16=tmp[15];
                  const double dl[5]={0.002,0.005,0.01,0.02,0.05};
                  char buf[256]; int off=0;
                  for(int q=0;q<5;q++){
                      int cnt=0;
                      for(int e=0;e<E;e++) if(bias_sc[e] >= k16-dl[q]) cnt++;
                      off+=snprintf(buf+off,sizeof(buf)-off,"%s d=%.3f:%d",q?"":"",dl[q],cnt); }
                  fprintf(stderr,"[K3/ROUTER] candidates within delta of k16 (of %d):%s\n",E,buf); }
                fprintf(stderr,"[K3/ROUTER] top1=%.5f k16=%.5f k17=%.5f gap(16,17)=%.6f "
                        "spread(1,16)=%.5f | ||x||_1=%.1f wmax=%.5f int8bound=%.5f ratio=%.1f\n",
                        tmp[0],tmp[15],tmp[16],tmp[15]-tmp[16],tmp[0]-tmp[15],
                        l1,wmax,bound,bound/((double)(tmp[15]-tmp[16])+1e-30));
                free(tmp);
            }
        }
        for(int kk=0;kk<K;kk++){
            int best=-1; float bv=-1e30f;
            for(int e=0;e<E;e++){
                float sv=bias_sc[e];
                if(!used[e]&&sv>bv){ bv=sv; best=e; }
            }
            used[best]=1;
            idx[kk]=best; wsel[kk]=st[best];          /* weight = RAW sigmoid score */
        }
        { float sm=0; for(int kk=0;kk<K;kk++) sm+=wsel[kk];
          for(int kk=0;kk<K;kk++) wsel[kk]/=(sm+1e-20f); }
        int Kt=K;
        /* K3_TOPP: drop the low-weight tail — the only lever that cuts expert
         * I/O AND compute proportionally. Weights renormalize over the kept
         * set (GLM's TOPP semantics). Quality-gate via K3_LOGITS. */
        if(g_k3_topp>0.f && g_k3_topp<1.f){
            for(int a2=1;a2<Kt;a2++){ int e=idx[a2]; float w2=wsel[a2]; int b2=a2-1;
                while(b2>=0&&wsel[b2]<w2){ idx[b2+1]=idx[b2]; wsel[b2+1]=wsel[b2]; b2--; }
                idx[b2+1]=e; wsel[b2+1]=w2; }
            float cum=0; int keep=Kt;
            for(int kk=0;kk<Kt;kk++){ cum+=wsel[kk]; if(cum>=g_k3_topp){ keep=kk+1; break; } }
            if(keep<Kt){
                float sm=0; for(int kk=0;kk<keep;kk++) sm+=wsel[kk];
                for(int kk=0;kk<keep;kk++) wsel[kk]/=(sm+1e-20f);
                Kt=keep;
            }
        }
        keff[t]=Kt;
        /* K3_ROUTE_STATS: which experts the router actually picks. Needed to
         * decide what can be pruned: at 1 bit each node still needs 106 GB of
         * experts plus ~30 GB of dense against 116 GB, so residency requires
         * dropping the cold tail. Counted on every rank -- the router is
         * replicated, so any rank sees the full 896-wide distribution. */
        if(m->route_hist)
            for(int kk=0;kk<Kt;kk++) m->route_hist[(int64_t)li*E+idx[kk]]++;
    }
    m->t_topk+=now_s()-tp0; tp0=now_s();
    float *z=falloc((int64_t)C*LT), *u=falloc((int64_t)C*LT);
    float *gate=falloc(MI), *up=falloc(MI), *hz=falloc(LT);
    w_matmul(z,x,&o->lat_down,C);
    m->t_latent+=now_s()-tp0;
    memset(u,0,(size_t)C*LT*sizeof(float));
    /* union across the chunk: each unique expert loads ONCE and applies to
     * every position that selected it (position lists via counting sort).
     * With QB-flat routing the dedup is modest (~15% at C=32), but the loads
     * arrive as one deep burst for the NVMe and the dense side above/below
     * batches perfectly. */
    {
        int *map=malloc((size_t)E*sizeof(int));
        int *uid=malloc((size_t)C*K*sizeof(int));
        int *pcnt=malloc((size_t)C*K*sizeof(int)), *pfirst=malloc((size_t)C*K*sizeof(int));
        int *poslist=malloc((size_t)C*K*sizeof(int)); float *wlist=falloc((int64_t)C*K);
        int *cur=malloc((size_t)C*K*sizeof(int));
        if(!map||!uid||!pcnt||!pfirst||!poslist||!cur){fprintf(stderr,"OOM moe union\n");exit(1);}
        for(int e=0;e<E;e++) map[e]=-1;
        int nu=0;
        int world=k3_net_world(), rank=k3_net_rank();
        for(int t=0;t<C;t++) for(int kk=0;kk<keff[t];kk++){
            int e=idxs[(int64_t)t*K+kk];
            /* Expert parallelism: every rank runs the SAME router and top-k
             * (same weights, same x, deterministic), so ownership needs no
             * communication — each rank simply drops the experts it does not
             * hold and the partial sums are reduced below. */
            if(world>1 && e%world!=rank) continue;
            if(map[e]<0){ map[e]=nu; uid[nu]=e; pcnt[nu]=0; nu++; }
            pcnt[map[e]]++;
        }
        int acc=0;
        for(int j=0;j<nu;j++){ pfirst[j]=acc; cur[j]=acc; acc+=pcnt[j]; }
        for(int t=0;t<C;t++) for(int kk=0;kk<keff[t];kk++){
            int e=idxs[(int64_t)t*K+kk];
            if(world>1 && e%world!=rank) continue;
            int j=map[e];
            poslist[cur[j]]=t; wlist[cur[j]]=wsels[(int64_t)t*K+kk]; cur[j]++;
        }
        /* keep loads in DISK-OFFSET order (experts are NOT id-ordered inside
         * the HF shards — measured 169/895); permute the list heads with the
         * ids. WILLNEED prefetch only for the buffered path. */
        for(int a2=0;a2<nu-1;a2++) for(int b2=a2+1;b2<nu;b2++){
            ERef *ea=&m->eref[(int64_t)li*E+uid[a2]], *eb=&m->eref[(int64_t)li*E+uid[b2]];
            if(eb->fd[0]<ea->fd[0]||(eb->fd[0]==ea->fd[0]&&eb->off[0]<ea->off[0])){
                int tt=uid[a2];uid[a2]=uid[b2];uid[b2]=tt;
                tt=pcnt[a2];pcnt[a2]=pcnt[b2];pcnt[b2]=tt;
                tt=pfirst[a2];pfirst[a2]=pfirst[b2];pfirst[b2]=tt; }
        }
        if(!g_k3_direct)
            for(int j=0;j<nu;j++){
                ERef *er=&m->eref[(int64_t)li*E+uid[j]];
                int64_t sizes[6]={m->e_w1p,m->e_w1s,m->e_w2p,m->e_w2s,m->e_w1p,m->e_w1s};
                if(er->contig){ if(er->fd[0]>=0) posix_fadvise(er->fd[0],er->off[0],m->e_slot,POSIX_FADV_WILLNEED); }
                else for(int k2=0;k2<6;k2++) if(er->fd[k2]>=0) posix_fadvise(er->fd[k2],er->off[k2],sizes[k2],POSIX_FADV_WILLNEED);
            }
        double te0=now_s();
        if(g_dense_census){ atomic_fetch_add_explicit(&g_exp_n,nu,memory_order_relaxed);
                            atomic_fetch_add_explicit(&g_exp_layers,1,memory_order_relaxed); }
        experts_apply_union(m,li,nu,uid,pfirst,pcnt,poslist,wlist,z,LT,C,u,gate,up,hz);
        m->t_expert+=now_s()-te0;
        free(map);free(uid);free(pcnt);free(pfirst);free(poslist);free(wlist);free(cur);
    }
    /* Sum the per-rank partial expert contributions. MUST precede the rmsnorm:
     * that norm is nonlinear, so reducing after it would not equal the
     * single-node result. 14 KB per token per layer — latency, not bandwidth. */
    /* Issue the reduce, then run the SHARED experts underneath it: they read x,
     * not the reduced accumulator, so they are independent. At 4 ranks the
     * collective is 3.21 ms of which only ~1.12 ms is transport -- the rest is
     * arrival jitter -- and the shared experts are ~0.15 s/token, enough to
     * cover most of it. Ordering is otherwise unchanged; out += sd still
     * happens after lat_up. */
    k3_net_allreduce_start(u,(size_t)C*LT);
    tp0=now_s();
    int shi=MI*c->n_shared;
    float *sg=falloc((int64_t)C*shi), *sd=falloc((int64_t)C*c->hidden);
    int fused=0;
#ifdef COLI_CUDA
    fused = k3_situ_fused(sg, x, o, C, shi, c->situ_b1, c->situ_b2);
#endif
    if(!fused){
        float *su=falloc((int64_t)C*shi);
        w_matmul(sg,x,&o->sh_gate,C); w_matmul(su,x,&o->sh_up,C);
        for(int64_t i=0;i<(int64_t)C*shi;i++) sg[i]=situf_(sg[i],su[i],c->situ_b1,c->situ_b2);
        free(su);
    }
    w_matmul(sd,sg,&o->sh_down,C);
    m->t_shared+=now_s()-tp0;
    { double na=now_s(); k3_net_allreduce_wait(); m->t_net_moe+=now_s()-na; }
    tp0=now_s();
    for(int t=0;t<C;t++)
        rmsnorm_(u+(int64_t)t*LT,u+(int64_t)t*LT,o->lat_norm,LT,c->eps);
    m->t_rnorm+=now_s()-tp0; tp0=now_s();
    w_matmul(out,u,&o->lat_up,C);
    m->t_latent+=now_s()-tp0;
    for(int64_t d=0;d<(int64_t)C*c->hidden;d++) out[d]+=sd[d];
    free(sco);free(idxs);free(wsels);free(keff);free(bias_sc);free(used);
    free(z);free(u);free(gate);free(up);free(hz);free(sg);free(sd);
}

static void dense_forward(Model *m, Layer *l, const float *x, int C, float *out){
    Cfg *c=&m->c; int DI=c->dense_inter;
    float *g=falloc((int64_t)C*DI), *u=falloc((int64_t)C*DI);
    w_matmul(g,x,&l->d_gate,C); w_matmul(u,x,&l->d_up,C);
    for(int64_t i=0;i<(int64_t)C*DI;i++) g[i]=situf_(g[i],u[i],c->situ_b1,c->situ_b2);
    w_matmul(out,g,&l->d_down,C);
    free(g);free(u);
}

/* ---------- a CHUNK of C tokens through the stack, layer-major: every dense
 * matmul batches over the chunk (weights stream from RAM once per chunk), the
 * MoE loads each unique expert once. Sequential state (KDA recurrence, MLA
 * cache, AttnRes bookkeeping) advances per token inside each layer, which is
 * exactly the original order — chunked results are bit-identical to C=1.
 * Returns the LAST position's logits (falloc'd), or NULL pre-head. ---------- */
static float *g_x0=NULL; static int g_x0_n=0;  /* K3_X0: injected inputs (validation) */
static FILE *g_lfp=NULL;                       /* K3_LOGITS: per-position logit dump */
static float *step_chunk(Model *m, const int *ids, int pos0, int C){
    Cfg *c=&m->c; int D=c->hidden;
    int nbmax=(c->n_layers+c->res_bs-1)/c->res_bs;
    float *hidden=falloc((int64_t)C*D), *bres=falloc((int64_t)C*nbmax*D);
    float *prefix=falloc((int64_t)C*D), *nrm=falloc((int64_t)C*D);
    float *att=falloc((int64_t)C*D), *mix=falloc(D), *mlp=falloc((int64_t)C*D);
    int nb=0;
    for(int t=0;t<C;t++){
        if(g_x0){
            if(pos0+t>=g_x0_n){ fprintf(stderr,"K3_X0: pos %d beyond %d injected rows\n",pos0+t,g_x0_n); exit(1); }
            memcpy(hidden+(int64_t)t*D,g_x0+(int64_t)(pos0+t)*D,D*sizeof(float));
        } else {
            char nm[512]; snprintf(nm,sizeof(nm),"%smodel.embed_tokens.weight",m->pfx);
            st_read_slice_f32(&m->S,nm,(int64_t)ids[t]*D,D,hidden+(int64_t)t*D,0);
        }
    }
    for(int i=0;i<c->n_layers;i++){
        Layer *l=&m->L[i];
        int snap=(i%c->res_bs==0);                    /* block boundary: same for all t */
        for(int t=0;t<C;t++){
            float *h=hidden+(int64_t)t*D, *p=prefix+(int64_t)t*D;
            memcpy(p,h,D*sizeof(float));              /* prefix_sum at entry */
            if(nb>0) res_mix(h,p,bres+(int64_t)t*nbmax*D,nb,D,l->attn_sw,c->eps);
            if(snap) memcpy(bres+(int64_t)t*nbmax*D+(int64_t)nb*D,p,D*sizeof(float));
            rmsnorm_(nrm+(int64_t)t*D,h,l->in_ln,D,c->eps);
        }
        int have_prefix=!snap;
        if(snap) nb++;
        double t0=now_s();
        if(l->kda) kda_forward(m,l,i,nrm,C,att);
        else       mla_forward(m,l,i,nrm,pos0,C,att);
        m->t_attn+=now_s()-t0;
        for(int t=0;t<C;t++){
            float *p=prefix+(int64_t)t*D, *a=att+(int64_t)t*D;
            if(have_prefix){ for(int d=0;d<D;d++) p[d]+=a[d]; }
            else           { memcpy(p,a,D*sizeof(float)); }
            res_mix(mix,p,bres+(int64_t)t*nbmax*D,nb,D,l->mlp_sw,c->eps);
            rmsnorm_(nrm+(int64_t)t*D,mix,l->post_ln,D,c->eps);
        }
        t0=now_s();
        if(l->sparse) moe_forward(m,l,i,nrm,C,mlp);
        else          dense_forward(m,l,nrm,C,mlp);
        m->t_moe+=now_s()-t0;
        for(int t=0;t<C;t++){
            float *p=prefix+(int64_t)t*D;
            for(int d=0;d<D;d++) p[d]+=mlp[(int64_t)t*D+d];
            memcpy(hidden+(int64_t)t*D,p,D*sizeof(float));
            if(m->trace) fwrite(hidden+(int64_t)t*D,sizeof(float),D,m->trace);
        }
    }
    float *logits=NULL;
    m->head_tok=-1;
    if(m->has_head){
        double t0=now_s();
        for(int t=0;t<C;t++){
            /* head only where needed: the chunk's last token (feeds sampling)
             * and every position when K3_LOGITS dumps teacher-forced logits */
            if(!g_lfp && t<C-1) continue;
            res_mix(mix,hidden+(int64_t)t*D,bres+(int64_t)t*nbmax*D,nb,D,m->out_sw,c->eps);
            rmsnorm_(mix,mix,m->final_norm,D,c->eps);
            if(m->trace) fwrite(mix,sizeof(float),D,m->trace);
            /* lm_head is [vocab, hidden] and every rank computed ALL of it --
             * 1.17 GB and 5.1 ms/token replicated four ways. Greedy sampling
             * only needs the argmax, so each rank can take a contiguous row
             * slice and the ranks then agree on a winner with ONE collective
             * per token (not per layer).
             *
             * Exact: row o's dot is independent of the other rows, so the
             * values match the replicated computation. sample_tok's greedy rule
             * is `if(lo[i]>lo[b])`, i.e. first index wins ties; a rank's local
             * scan uses the same strict >, and scanning ranks in ascending
             * order preserves it globally because the slices are ascending and
             * contiguous.
             *
             * temp>0 needs the whole vector for softmax/top-p, and K3_LOGITS
             * and the trace dump need it too, so those keep the full head. */
            int hw = k3_net_world();
            int shard = hw>1 && m->head_greedy && !g_lfp && !m->trace && t==C-1;
            if(shard){
                int V=c->vocab, per=(V+hw-1)/hw, rk=k3_net_rank();
                int v0=rk*per; if(v0>V) v0=V;
                int vn=V-v0; if(vn>per) vn=per;
                if(!m->lm_head_shard_ready){
                    m->lm_head_shard=w_rows(&m->lm_head,v0,vn>0?vn:1);
                    m->lm_head_shard_ready=1;
                }
                float bv=-INFINITY; int gi=v0;
                if(vn>0){
                    float *part=falloc(vn);
                    w_matmul(part,mix,&m->lm_head_shard,1);
                    int bi=0; for(int i=1;i<vn;i++) if(part[i]>part[bi]) bi=i;
                    bv=part[bi]; gi=v0+bi;
                    free(part);
                }
                /* zero-filled sum IS a gather: each rank writes only its slot */
                float red[2*K3_NET_MAX]; memset(red,0,sizeof(red));
                red[2*rk]=bv; red[2*rk+1]=(float)gi;
                k3_net_allreduce(red,(size_t)2*hw);
                int best=(int)red[1]; float bestv=red[0];
                for(int i=1;i<hw;i++) if(red[2*i]>bestv){ bestv=red[2*i]; best=(int)red[2*i+1]; }
                m->head_tok=best;
                logits=NULL;
                continue;
            }
            float *lo=falloc(c->vocab);
            w_matmul(lo,mix,&m->lm_head,1);
            if(g_lfp) fwrite(lo,sizeof(float),(size_t)c->vocab,g_lfp);
            if(t==C-1) logits=lo; else free(lo);
        }
        m->t_head+=now_s()-t0;
    }
    free(hidden);free(bres);free(prefix);free(nrm);free(att);free(mix);free(mlp);
    return logits;
}

static void kv_alloc(Model *m, int max_t){
    Cfg *c=&m->c; m->max_t=max_t;
    m->Lc=calloc(c->n_layers,sizeof(kvq*));
    m->Rc=calloc(c->n_layers,sizeof(kvq*));
    int nc=0;
    for(int i=0;i<c->n_layers;i++) if(!m->L[i].kda){
        m->Lc[i]=malloc((size_t)max_t*c->kv_lora*sizeof(kvq));
        m->Rc[i]=malloc((size_t)max_t*c->qk_rope*sizeof(kvq));
        if(!m->Lc[i]||!m->Rc[i]){ fprintf(stderr,"OOM kv cache\n"); exit(1); }
        nc++;
    }
    fprintf(stderr,"[K3] kv cache %d/%d layers x %d tok x %d el x %zub = %.2f GB\n",
            nc,c->n_layers,max_t,c->kv_lora+c->qk_rope,sizeof(kvq),
            (double)nc*max_t*(c->kv_lora+c->qk_rope)*sizeof(kvq)/1073741824.0);
}

typedef struct { float p; int id; } SampleProb;
static int sample_prob_desc(const void *a,const void *b){
    float d=((const SampleProb*)b)->p-((const SampleProb*)a)->p;
    return d>0?1:d<0?-1:0;
}
static int sample_tok(const float *lo, int V, float temp, float top_p){
    if(temp<=0.f){ int b=0; for(int i=1;i<V;i++) if(lo[i]>lo[b]) b=i; return b; }
    SampleProb *rank=malloc((size_t)V*sizeof(SampleProb)); float mx=lo[0];
    if(!rank){ fprintf(stderr,"OOM sampling\n"); exit(1); }
    for(int i=1;i<V;i++) if(lo[i]>mx) mx=lo[i];
    double sum=0;
    for(int i=0;i<V;i++){ float p=expf((lo[i]-mx)/temp); sum+=p; rank[i]=(SampleProb){p,i}; }
    qsort(rank,(size_t)V,sizeof(SampleProb),sample_prob_desc);
    double cut=(top_p>0.f&&top_p<1.f)?top_p*sum:sum, kept=0; int n=0;
    while(n<V&&kept<cut) kept+=rank[n++].p;
    double r=((double)rand()/RAND_MAX)*kept, acc=0; int pick=rank[0].id;
    for(int i=0;i<n;i++){ acc+=rank[i].p; if(acc>=r){ pick=rank[i].id; break; } }
    free(rank); return pick;
}

/* ---------- K3 XTML chat format (faithful to the shipped encoding_k3.py) --
 * Only <|open|>, <|close|>, <|sep|>, <|end_of_msg|> are special TOKENS; tag
 * names and attributes are ordinary text, encoded as the same standalone
 * segments as the reference (segment boundaries are token boundaries). A
 * turn renders as
 *   <|open|>message role="user"<|sep|>TEXT<|close|>message<|sep|><|end_of_msg|>
 * and the generation prompt opens the assistant message plus its structural
 * thinking channel:
 *   <|open|>message role="assistant"<|sep|><|open|>think<|sep|>
 * (K3_THINK=0 opens <response> directly = non-thinking mode). The model then
 * closes think, opens response, and finishes with <|end_of_msg|> (the eos). */
typedef struct { Tok *T; int *ids; int n, cap;
                 int sp_open, sp_close, sp_sep, sp_eom; } ChatB;
static void cb_special(ChatB *b, int id){
    if(b->n>=b->cap){ fprintf(stderr,"chat prompt too long\n"); exit(1); }
    b->ids[b->n++]=id;
}
static void cb_text(ChatB *b, const char *s){
    if(!*s) return;
    b->n+=tok_encode(b->T,s,(int)strlen(s),b->ids+b->n,b->cap-b->n);
}
static void cb_open(ChatB *b, const char *tag, const char *role){
    cb_special(b,b->sp_open); cb_text(b,tag);
    if(role){ cb_text(b," role"); cb_text(b,"=\""); cb_text(b,role); cb_text(b,"\""); }
    cb_special(b,b->sp_sep);
}
static void cb_close(ChatB *b, const char *tag){
    cb_special(b,b->sp_close); cb_text(b,tag); cb_special(b,b->sp_sep);
}
static int chat_special(Tok *T, const char *s){
    int l=(int)strlen(s);
    for(int i=0;i<T->nsp;i++)
        if(T->sp[i].len==l && !memcmp(T->sp[i].str,s,l)) return T->sp[i].id;
    return -1;
}
/* returns prompt length; sp[4] = {open, close, sep, end_of_msg} ids */
static int chat_build(Tok *T, const char *sys, const char *user, int thinking,
                      int *ids, int cap, int *sp){
    ChatB b={T,ids,0,cap,
        chat_special(T,"<|open|>"), chat_special(T,"<|close|>"),
        chat_special(T,"<|sep|>"),  chat_special(T,"<|end_of_msg|>")};
    if(b.sp_open<0||b.sp_close<0||b.sp_sep<0||b.sp_eom<0){
        fprintf(stderr,"chat: XTML special tokens not in tokenizer.json\n"); exit(1); }
    sp[0]=b.sp_open; sp[1]=b.sp_close; sp[2]=b.sp_sep; sp[3]=b.sp_eom;
    if(sys&&*sys){
        cb_open(&b,"message","system"); cb_text(&b,sys);
        cb_close(&b,"message"); cb_special(&b,b.sp_eom);
    }
    cb_open(&b,"message","user"); cb_text(&b,user);
    cb_close(&b,"message"); cb_special(&b,b.sp_eom);
    cb_open(&b,"message","assistant");
    cb_open(&b,thinking?"think":"response",NULL);
    return b.n;
}

static void chat_message(ChatB *b, const char *role, const char *text, int assistant){
    cb_open(b,"message",role);
    if(assistant) cb_open(b,"response",NULL);
    cb_text(b,text);
    if(assistant) cb_close(b,"response");
    cb_close(b,"message");
    cb_special(b,b->sp_eom);
}

static void chat_assistant(ChatB *b, const char *reasoning, const char *text){
    cb_open(b,"message","assistant");
    cb_open(b,"think",NULL); cb_text(b,reasoning); cb_close(b,"think");
    cb_open(b,"response",NULL); cb_text(b,text); cb_close(b,"response");
    cb_close(b,"message"); cb_special(b,b->sp_eom);
}

/* Internal gateway payload. Length framing keeps arbitrary UTF-8/newlines in
 * message content while preserving the segment boundaries required by K3's
 * rank-BPE chat template:
 *   K3CHAT1\n
 *   M <role> <utf8-bytes>\n<content> ...
 *   G <thinking>\n
 */
static int chat_build_wire(Tok *T, const char *wire, int nwire, int *thinking,
                           int *ids, int cap, int *sp){
    ChatB b={T,ids,0,cap,
        chat_special(T,"<|open|>"), chat_special(T,"<|close|>"),
        chat_special(T,"<|sep|>"),  chat_special(T,"<|end_of_msg|>")};
    if(b.sp_open<0||b.sp_close<0||b.sp_sep<0||b.sp_eom<0) return -1;
    sp[0]=b.sp_open; sp[1]=b.sp_close; sp[2]=b.sp_sep; sp[3]=b.sp_eom;
    const char *p=wire, *end=wire+nwire;
    if(nwire<8||memcmp(p,"K3CHAT1\n",8)) return -1;
    p+=8; *thinking=0;
    while(p<end){
        const char *nl=memchr(p,'\n',(size_t)(end-p));
        if(!nl) return -1;
        if(*p=='G'){
            int v=0;
            if(sscanf(p,"G %d",&v)!=1) return -1;
            *thinking=!!v; p=nl+1; break;
        }
        if(*p=='A'){
            int nr=-1, nt=-1;
            if(sscanf(p,"A %d %d",&nr,&nt)!=2||nr<0||nt<0||nl+1+nr+nt>end) return -1;
            char *reason=malloc((size_t)nr+1), *text=malloc((size_t)nt+1);
            if(!reason||!text){ fprintf(stderr,"OOM chat assistant\n"); exit(1); }
            memcpy(reason,nl+1,(size_t)nr); reason[nr]=0;
            memcpy(text,nl+1+nr,(size_t)nt); text[nt]=0;
            chat_assistant(&b,reason,text);
            free(reason); free(text); p=nl+1+nr+nt; continue;
        }
        char role[16]; int nb=-1;
        if(sscanf(p,"M %15s %d",role,&nb)!=2||nb<0||nl+1+nb>end) return -1;
        char *text=malloc((size_t)nb+1);
        if(!text){ fprintf(stderr,"OOM chat message\n"); exit(1); }
        memcpy(text,nl+1,(size_t)nb); text[nb]=0;
        const char *r=!strcmp(role,"developer")?"system":role;
        if(strcmp(r,"system")&&strcmp(r,"user")&&strcmp(r,"assistant")){ free(text); return -1; }
        chat_message(&b,r,text,!strcmp(r,"assistant"));
        free(text); p=nl+1+nb;
    }
    cb_open(&b,"message","assistant");
    cb_open(&b,*thinking?"think":"response",NULL);
    return b.n;
}

/* ---------- serve mode: shared openai_server.py protocol ---------- */
typedef struct {
    char id[64];
    int max_tok;
    float temp, top_p;
    char *payload;
    int plen;
} ServeReq;

static void model_state_reset(Model *m){
    Cfg *c=&m->c;
    for(int i=0;i<c->n_layers;i++){
        if(m->L[i].kda){
            memset(m->kstate[i],0,(size_t)c->kda_heads*c->kda_hd*c->kda_hd*sizeof(float));
            m->cwh[i]=0;
            memset(m->cwq[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
            memset(m->cwk[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
            memset(m->cwv[i],0,(size_t)c->kda_proj*c->conv_k*sizeof(float));
        }
        if(m->Lc&&m->Lc[i]) free(m->Lc[i]);
        if(m->Rc&&m->Rc[i]) free(m->Rc[i]);
    }
    free(m->Lc); free(m->Rc);
    m->Lc=NULL; m->Rc=NULL; m->max_t=0;
}

static int serve_stdin_readable(void){
    fd_set r; struct timeval tv={0,0};
    FD_ZERO(&r); FD_SET(0,&r);
    return select(1,&r,NULL,NULL,&tv)>0;
}

static int serve_read_req(ServeReq *q, const char *active){
    char line[512], cmd[16], id[64];
    if(!fgets(line,sizeof(line),stdin)) return -1;
    if(sscanf(line,"%15s %63s",cmd,id)<2) return 0;
    if(!strcmp(cmd,"CANCEL")||!strcmp(cmd,"STOP")) return active&&!strcmp(active,id);
    /* SET <key> <value>: retune a knob without paying the load again. Dense
     * weights come in at ~1.5 s/layer, so a process restart costs 145 s while a
     * warm measurement pass costs ~80 s -- an A/B that restarts per arm spends
     * more time loading than measuring. Only knobs that are read per request
     * (K3_CHUNK, K3_MAXT) or held in a global listed here actually change;
     * anything cached in a lazily-initialised static keeps its first value, so
     * the reply echoes what was applied and the harness can assert on it. */
    if(!strcmp(cmd,"SET")){
        char val[64]="";
        if(sscanf(line,"%*s %*s %63s",val)!=1){ printf("ERROR SET needs a value\n"); fflush(stdout); return 0; }
        setenv(id,val,1);
        int live = 1;
        if(!strcmp(id,"K3_KVNEON"))      g_kvneon=atoi(val);
        else if(!strcmp(id,"K3_MATT_BATCH")) g_mattb=atoi(val);
        else if(!strcmp(id,"K3_CHUNK")||!strcmp(id,"K3_MAXT")) ;      /* re-read per request */
        else live = 0;
        printf("SETOK %s=%s live=%d\n",id,val,live); fflush(stdout);
        return 0;
    }
    if(strcmp(cmd,"SUBMIT")) return 0;
    int slot, plen, max_tok; float temp, top_p;
    if(sscanf(line,"%*s %*s %d %d %d %f %f",&slot,&plen,&max_tok,&temp,&top_p)!=5||
       plen<0||plen>(1<<24)||max_tok<1){
        printf("ERROR %s bad submit header\n",id); fflush(stdout); return 0;
    }
    (void)slot;
    char *payload=malloc((size_t)plen+1);
    if(!payload){ printf("ERROR %s out of memory\n",id); fflush(stdout); return 0; }
    if(fread(payload,1,(size_t)plen,stdin)!=(size_t)plen){ free(payload); return -1; }
    (void)fgetc(stdin); payload[plen]=0;
    snprintf(q->id,sizeof(q->id),"%s",id);
    q->max_tok=max_tok; q->temp=temp; q->top_p=top_p;
    q->payload=payload; q->plen=plen;
    return 2;
}

static void serve_data(const char *id, const char *p, int n){
    if(n<=0) return;
    printf("DATA %s %d\n",id,n);
    fwrite(p,1,(size_t)n,stdout); fputc('\n',stdout); fflush(stdout);
}

static void serve_one(Model *m, Tok *T, ServeReq *q){
    int cap=65536, *ids=malloc((size_t)cap*sizeof(int)), np=0;
    if(!ids){ printf("ERROR %s out of memory\n",q->id); fflush(stdout); return; }
    int sp[4]={-1,-1,-1,-1}, chat=0, thinking=0;
    if(m->c.bos>=0) ids[np++]=m->c.bos;
    if(q->plen>=8&&!memcmp(q->payload,"K3CHAT1\n",8)){
        int n=chat_build_wire(T,q->payload,q->plen,&thinking,ids+np,cap-np,sp);
        if(n<0){ printf("ERROR %s invalid K3 chat payload\n",q->id); fflush(stdout); free(ids); return; }
        np+=n; chat=1;
    } else {
        np+=tok_encode(T,q->payload,q->plen,ids+np,cap-np);
    }
    int max_ctx=getenv("K3_MAXT")?atoi(getenv("K3_MAXT")):8192;
    if(np<1||np+q->max_tok>max_ctx){
        printf("ERROR %s CONTEXT_EXCEEDED prompt_tokens=%d requested=%d capacity=%d\n",
               q->id,np,q->max_tok,max_ctx);
        fflush(stdout); free(ids); return;
    }
    printf("ACCEPT %s %d\n",q->id,np); fflush(stdout);
#ifdef COLI_CUDA
    /* Sweep AFTER the first request: mirrors are made lazily on first use, and
     * only a completed prefill+decode has exercised every shape both paths see. */
    { static int reqs=0;
      if(reqs==1 && getenv("K3_FREE_HOST") && atoi(getenv("K3_FREE_HOST")))
          k3_host_reclaim_sweep();
      if(reqs<2) reqs++; }
#endif
    model_state_reset(m);
    kv_alloc(m,np+q->max_tok+8);
    int chunk=getenv("K3_CHUNK")?atoi(getenv("K3_CHUNK")):32;
    if(chunk<1) chunk=1; if(chunk>512) chunk=512;
    double t0=now_s(), a0=m->t_attn, e0=m->t_moe, d0=m->t_eload, h0=m->t_head;
    uint64_t hit0=m->hits, miss0=m->miss;
    float *lo=NULL;
    m->head_greedy = (q->temp <= 0.f);
    /* PROF2 has always taken its baseline BELOW the prefill loop, so every term
     * it reports is decode-only and prefill was 74% of a prompted request with
     * no instrument on it at all. Snapshot the same timers around the loop and
     * report the prefill deltas as PROF2P. */
    double pa0=m->t_attn, pe0=m->t_moe, pd0=m->t_eload, ph0=m->t_head;
    double pr0=m->t_router, ptop0=m->t_topk, pl0=m->t_latent;
    double ps0=m->t_shared, px0=m->t_expert, pn0=m->t_rnorm;
    double pkpr0=m->t_kproj, pkc0=m->t_kconv, pkh0=m->t_khead, pko0=m->t_kout;
    double pcj0=m->t_ctljoin, pcw0=g_ctl_secs;
    double pnk0=m->t_net_kda, pnm0=m->t_net_mla, pnx0=m->t_net_moe;
    double prm0=g_resmix_secs;
    double pmpr0=m->t_mproj, pma0=m->t_matt, pmo0=m->t_mout;
    double pnet0=k3_net_secs(); uint64_t pnc0=k3_net_calls();
    double pwall0=now_s();
    for(int i=0;i<np;i+=chunk){
        int C=np-i<chunk?np-i:chunk;
        free(lo); lo=step_chunk(m,ids+i,i,C);
    }
    { double pw=now_s()-pwall0;
      fprintf(stderr,"PROF2P wall=%.3f np=%d chunk=%d tok/s=%.2f | attn=%.3f moe=%.3f load=%.3f head=%.3f "
        "net=%.3f/%llu router=%.3f topk=%.3f latent=%.3f shared=%.3f expert=%.3f rnorm=%.3f "
        "kproj=%.3f kconv=%.3f khead=%.3f kout=%.3f mproj=%.3f matt=%.3f mout=%.3f "
        "ctlwork=%.3f ctljoin=%.3f netkda=%.3f netmla=%.3f netmoe=%.3f resmix=%.3f\n",
        pw,np,chunk,np/(pw>0?pw:1),
        m->t_attn-pa0,m->t_moe-pe0,m->t_eload-pd0,m->t_head-ph0,
        k3_net_secs()-pnet0,(unsigned long long)(k3_net_calls()-pnc0),
        m->t_router-pr0,m->t_topk-ptop0,m->t_latent-pl0,m->t_shared-ps0,
        m->t_expert-px0,m->t_rnorm-pn0,
        m->t_kproj-pkpr0,m->t_kconv-pkc0,m->t_khead-pkh0,m->t_kout-pko0,
        m->t_mproj-pmpr0,m->t_matt-pma0,m->t_mout-pmo0,
        g_ctl_secs-pcw0,m->t_ctljoin-pcj0,
        m->t_net_kda-pnk0,m->t_net_mla-pnm0,m->t_net_moe-pnx0,
        g_resmix_secs-prm0); }
    /* Decode-only snapshots.  The existing PROF line deliberately includes
     * prefill, which is useful for request accounting but obscures B=1 token
     * latency.  PROF2 excludes prefill and breaks the warm decode path into
     * terms that can actually guide kernel/collective work. */
    double da0=m->t_attn, de0=m->t_moe, dd0=m->t_eload, dh0=m->t_head;
    double dr0=m->t_router, dtop0=m->t_topk, dl0=m->t_latent;
    double ds0=m->t_shared, dx0=m->t_expert, dn0=m->t_rnorm;
    double dar0=g_ar_dur, das0=g_ar_spawn; long dac0=g_ar_calls;
    double dkpr0=m->t_kproj, dkc0=m->t_kconv, dkh0=m->t_khead, dko0=m->t_kout;
    double dcj0=m->t_ctljoin, dcw0=g_ctl_secs;
    double dnk0=m->t_net_kda, dnm0=m->t_net_mla, dnx0=m->t_net_moe;
    double dcs0=g_ctl_start, drm0=g_resmix_secs;
    double dmpr0=m->t_mproj, dmc0=m->t_mcache, dma0=m->t_matt, dmo0=m->t_mout;
    double dnet0=k3_net_secs(); uint64_t dnc0=k3_net_calls();
    int gen=0, limited=1, cancelled=0, xsup=0, xopen=0, xtl=0;
    char buf[512], xtag[64];
    double tg=now_s();
    for(int s=0;s<q->max_tok&&!cancelled;s++){
        int tk = (m->head_tok>=0) ? m->head_tok
                               : sample_tok(lo,m->c.vocab,q->temp,q->top_p);
        free(lo); lo=NULL;
        int eos=0; for(int i=0;i<m->c.n_eos;i++) if(tk==m->c.eos[i]) eos=1;
        int show=!eos;
        if(chat&&sp[0]>=0){
            if(tk==sp[0]||tk==sp[1]){
                xsup=1; xopen=(tk==sp[0]); xtl=0; show=0;
            } else if(tk==sp[2]){
                if(xsup){
                    xsup=0; xtag[xtl]=0;
                    if(xopen&&!strcmp(xtag,"response")&&thinking)
                        serve_data(q->id,"</think>",8);
                }
                show=0;
            } else if(xsup){
                int nb=tok_decode(T,&tk,1,buf,sizeof(buf)-1);
                if(xtl+nb<(int)sizeof(xtag)){ memcpy(xtag+xtl,buf,(size_t)nb); xtl+=nb; }
                show=0;
            } else if(tk==sp[3]) show=0;
        }
        if(show){
            int nb=tok_decode(T,&tk,1,buf,sizeof(buf)-1);
            serve_data(q->id,buf,nb);
        }
        if(!eos) gen++;
        while(serve_stdin_readable()){
            ServeReq queued={0};
            int r=serve_read_req(&queued,q->id);
            if(r<0){ cancelled=1; break; }
            if(r==1) cancelled=1;
            if(r==2){
                printf("ERROR %s engine busy\n",queued.id); fflush(stdout); free(queued.payload);
            }
        }
        if(cancelled){ limited=0; break; }
        if(eos){ limited=0; break; }
        if(s+1<q->max_tok) lo=step_chunk(m,&tk,np+s,1);
    }
    free(lo); free(ids);
    double dt=now_s()-t0, decode=now_s()-tg;
    uint64_t hits=m->hits-hit0, misses=m->miss-miss0, total=hits+misses;
    printf("DONE %s STAT %d %.3f %.1f %.2f %d %d\n",q->id,gen,
           decode>0?gen/decode:0.0,total?100.0*hits/total:0.0,rss_gb(),np,limited);
    double moe=m->t_moe-e0, disk=m->t_eload-d0;
    printf("PROF %.3f %d %d %.3f %.3f %.3f %.3f %.3f %d\n",
           dt,np,gen,disk,0.0,moe>disk?moe-disk:moe,m->t_attn-a0,m->t_head-h0,gen+1);
#ifdef COLI_CUDA
    { static int once=0; if(!once){ once=1; coli_k3_devmirror_report(); } }
#endif
    printf("PROF2 attn=%.3f moe=%.3f load=%.3f head=%.3f net=%.3f/%llu "
           "router=%.3f topk=%.3f latent=%.3f shared=%.3f expert=%.3f rnorm=%.3f "
           "kproj=%.3f kconv=%.3f khead=%.3f kout=%.3f "
           "mproj=%.3f mcache=%.3f matt=%.3f mout=%.3f ctlwork=%.3f ctljoin=%.3f "
           "netkda=%.3f netmla=%.3f netmoe=%.3f ctlstart=%.3f resmix=%.3f "
           "arwork=%.3f arspawn=%.3f arcalls=%ld\n",
           m->t_attn-da0,m->t_moe-de0,m->t_eload-dd0,m->t_head-dh0,
           k3_net_secs()-dnet0,(unsigned long long)(k3_net_calls()-dnc0),
           m->t_router-dr0,m->t_topk-dtop0,m->t_latent-dl0,
           m->t_shared-ds0,m->t_expert-dx0,m->t_rnorm-dn0,
           m->t_kproj-dkpr0,m->t_kconv-dkc0,m->t_khead-dkh0,m->t_kout-dko0,
           m->t_mproj-dmpr0,m->t_mcache-dmc0,m->t_matt-dma0,m->t_mout-dmo0,
           g_ctl_secs-dcw0,m->t_ctljoin-dcj0,
           m->t_net_kda-dnk0,m->t_net_mla-dnm0,m->t_net_moe-dnx0,g_ctl_start-dcs0,
           g_resmix_secs-drm0,
           g_ar_dur-dar0,g_ar_spawn-das0,g_ar_calls-dac0);
    if(getenv("K3_NET_TTFB") && atoi(getenv("K3_NET_TTFB")) && g_ttfb_calls)
        fprintf(stderr,"[K3/NET] exchanges=%llu  peer-not-ready=%.3f s (%.0f%%)  "
                "bytes-moving=%.3f s (%.0f%%)  -> %.0f us/exchange waiting, %.0f us moving\n",
                (unsigned long long)g_ttfb_calls, g_ttfb_wait,
                100.0*g_ttfb_wait/(g_ttfb_wait+g_ttfb_move+1e-9), g_ttfb_move,
                100.0*g_ttfb_move/(g_ttfb_wait+g_ttfb_move+1e-9),
                1e6*g_ttfb_wait/g_ttfb_calls, 1e6*g_ttfb_move/g_ttfb_calls);
    if(g_dense_census) fprintf(stderr,"[K3/PREFILL] S>1 dense calls: gemm=%ld generic-fmt=%ld\n",
            (long)atomic_load_explicit(&g_pf_gemm,memory_order_relaxed),
            (long)atomic_load_explicit(&g_pf_generic,memory_order_relaxed));
    if(g_dense_census) fprintf(stderr,"[K3/EXP] experts run=%ld over %ld layer-calls (%.3f per call)\n",
            (long)atomic_load_explicit(&g_exp_n,memory_order_relaxed),
            (long)atomic_load_explicit(&g_exp_layers,memory_order_relaxed),
            atomic_load_explicit(&g_exp_layers,memory_order_relaxed)
              ? (double)atomic_load_explicit(&g_exp_n,memory_order_relaxed)
                / (double)atomic_load_explicit(&g_exp_layers,memory_order_relaxed) : 0.0);
    if(g_dense_census) fprintf(stderr,"[K3/DENSE] decode dispatch: exact=%ld off=%ld fmt=%ld inpar=%ld\n",
            (long)atomic_load_explicit(&g_dp_k3,memory_order_relaxed),
            (long)atomic_load_explicit(&g_dp_off,memory_order_relaxed),
            (long)atomic_load_explicit(&g_dp_fmt,memory_order_relaxed),
            (long)atomic_load_explicit(&g_dp_par,memory_order_relaxed));
    fflush(stdout);
}

/* Dump the K3_ROUTE_STATS histogram. Called from BOTH exits: main's tail and
 * serve mode, which returns straight out of main and used to bypass the write
 * entirely -- losing exactly the long-running profile the option exists to
 * collect. Serve mode also rewrites after every request rather than only at
 * exit, because a server is usually killed rather than allowed to return, and
 * 333 KB against a multi-second request is free. */
static void k3_route_stats_write(const Model *m){
    const char *p = getenv("K3_ROUTE_STATS");
    if(!m->route_hist || !p) return;
    FILE *rf=fopen(p,"wb");
    if(!rf){ perror(p); return; }
    fwrite(m->route_hist,sizeof(uint32_t),
           (size_t)m->c.n_layers*m->c.n_experts,rf);
    fclose(rf);
}

static void serve_loop(Model *m, Tok *T){
    setvbuf(stdin,NULL,_IONBF,0);
    fputs("\x01\x01READY\x01\x01\n",stdout);
    printf("STAT 0 0.0 0.0 %.2f 0 0\n",rss_gb());
    fflush(stdout);
    for(;;){
        ServeReq q={0}; int r;
        do r=serve_read_req(&q,NULL); while(r==0);
        if(r<0){ k3_route_stats_write(m); return; }
        if(r==2){ serve_one(m,T,&q); free(q.payload); k3_route_stats_write(m); }
    }
}

int main(int argc, char **argv){
    /* Multi-node: K3_WORLD>1 shards the routed experts across ranks. Every
     * rank runs the identical dense forward and, after the per-layer
     * all-reduce, holds identical activations — so all ranks sample the same
     * token and stay in lockstep with no extra synchronisation. Only rank 0
     * writes to stdout. No-op when K3_WORLD is unset. */
    k3_net_init();
    int serving=getenv("SERVE")&&getenv("SERVE")[0]=='1';
    if(!serving&&argc<2){
        fprintf(stderr,"usage: %s <model_dir> [prompt] [--ids \"1 2 3\"] [--ngen N]\n",argv[0]);
        return 1;
    }
    const char *snap=serving?getenv("SNAP"):argv[1], *prompt=NULL, *idstr=NULL, *sysmsg=NULL, *wirepath=NULL;
    if(!snap||!*snap){ fprintf(stderr,"set SNAP=<Kimi K3 snapshot directory>\n"); return 1; }
    int ngen=32, chat=0;
    for(int i=serving?1:2;i<argc;i++){
        if(!strcmp(argv[i],"--ngen")&&i+1<argc) ngen=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ids")&&i+1<argc) idstr=argv[++i];
        else if(!strcmp(argv[i],"--chat")) chat=1;
        else if(!strcmp(argv[i],"--system")&&i+1<argc) sysmsg=argv[++i];
        else if(!strcmp(argv[i],"--wire-test")&&i+1<argc) wirepath=argv[++i];
        else if(!prompt) prompt=argv[i];
    }
    if(wirepath){
        char tp[2048]; snprintf(tp,sizeof(tp),"%s/tokenizer.json",snap);
        Tok wt; tok_load(&wt,tp);
        FILE *wf=fopen(wirepath,"rb"); if(!wf){ perror(wirepath); return 1; }
        fseek(wf,0,SEEK_END); long wn=ftell(wf); fseek(wf,0,SEEK_SET);
        if(wn<0||wn>(1<<24)){ fprintf(stderr,"wire payload too large\n"); fclose(wf); return 1; }
        char *wire=malloc((size_t)wn+1); int *wid=malloc(65536*sizeof(int));
        if(!wire||!wid){ fprintf(stderr,"OOM wire test\n"); return 1; }
        if(fread(wire,1,(size_t)wn,wf)!=(size_t)wn){ fprintf(stderr,"short wire read\n"); return 1; }
        fclose(wf); wire[wn]=0;
        int thinking=0, wsp[4], n=chat_build_wire(&wt,wire,(int)wn,&thinking,wid,65536,wsp);
        if(n<0){ fprintf(stderr,"invalid K3 chat wire payload\n"); return 1; }
        for(int i=0;i<n;i++) printf("%s%d",i?" ":"",wid[i]);
        printf("\n"); free(wire); free(wid); return 0;
    }
    float temp=getenv("COLI_TEMP")?(float)atof(getenv("COLI_TEMP")):0.f;
    if(getenv("K3_KVNEON")) g_kvneon=atoi(getenv("K3_KVNEON"));
    if(getenv("K3_FDOT")) g_fdot=atoi(getenv("K3_FDOT"));
    if(getenv("K3_MATT_BATCH")) g_mattb=atoi(getenv("K3_MATT_BATCH"));
    int nlayers=getenv("K3_LAYERS")?atoi(getenv("K3_LAYERS")):0;
    Model m;
    model_init(&m,snap,nlayers);
    k3_net_barrier();   /* line the ranks up before timing anything (see k3_net.h) */
    if(getenv("K3_TRACE")){
        m.trace=fopen(getenv("K3_TRACE"),"wb");
        if(!m.trace){ perror(getenv("K3_TRACE")); return 1; }
    }
    if(getenv("K3_X0")){       /* injected input rows [T,hidden] f32, bypasses embed */
        FILE *f=fopen(getenv("K3_X0"),"rb");
        if(!f){ perror(getenv("K3_X0")); return 1; }
        fseek(f,0,SEEK_END); long fn=ftell(f); fseek(f,0,SEEK_SET);
        g_x0_n=(int)(fn/((long)m.c.hidden*4));
        g_x0=falloc((int64_t)g_x0_n*m.c.hidden);
        if(fread(g_x0,4,(size_t)g_x0_n*m.c.hidden,f)!=(size_t)g_x0_n*m.c.hidden){ fprintf(stderr,"K3_X0 short read\n"); return 1; }
        fclose(f);
        fprintf(stderr,"[K3] K3_X0: %d injected input rows\n",g_x0_n);
    }
    /* tokenize */
    int ids[65536], np=0;
    Tok T; int has_tok=0;
    { char tp[2048]; snprintf(tp,sizeof(tp),"%s/tokenizer.json",snap);
      FILE *f=fopen(tp,"rb"); if(f){ fclose(f); tok_load(&T,tp); has_tok=1;
          fprintf(stderr,"[K3] tokenizer.json loaded (family=%s)\n",T.kimi?"kimi":(T.o200k?"o200k":"cl100k")); } }
    if(serving){
        if(!has_tok){ fprintf(stderr,"serve mode needs tokenizer.json\n"); return 1; }
        serve_loop(&m,&T);
        return 0;
    }
    int sp[4]={-1,-1,-1,-1};
    int think=getenv("K3_THINK")?atoi(getenv("K3_THINK")):1;
    if(idstr){
        const char *p=idstr;
        while(*p&&np<65536){ while(*p==' '||*p==',')p++; if(!*p)break; ids[np++]=(int)strtol(p,(char**)&p,10); }
    } else if(chat){
        if(!has_tok){ fprintf(stderr,"--chat needs tokenizer.json\n"); return 1; }
        if(!prompt){ fprintf(stderr,"--chat needs a user message\n"); return 1; }
        if(m.c.bos>=0) ids[np++]=m.c.bos;
        np+=chat_build(&T,sysmsg,prompt,think,ids+np,65536-np,sp);
        if(getenv("K3_CHAT_IDS")){
            fprintf(stderr,"[K3] chat ids:");
            for(int i=0;i<np;i++) fprintf(stderr," %d",ids[i]);
            fprintf(stderr,"\n");
        }
    } else if(prompt){
        if(!has_tok){ fprintf(stderr,"no tokenizer.json — pass --ids (generate one with tools/k3_tokenizer.py)\n"); return 1; }
        if(m.c.bos>=0) ids[np++]=m.c.bos;
        np+=tok_encode(&T,prompt,(int)strlen(prompt),ids+np,65536-np);
    } else { fprintf(stderr,"no prompt and no --ids\n"); return 1; }
    fprintf(stderr,"[K3] prompt: %d tokens | ngen %d | temp %.2f\n",np,ngen,temp);
    int max_t=getenv("K3_MAXT")?atoi(getenv("K3_MAXT")):np+ngen;
    kv_alloc(&m,max_t);
    if(getenv("K3_LOGITS")){
        g_lfp=fopen(getenv("K3_LOGITS"),"wb");
        if(!g_lfp){ perror(getenv("K3_LOGITS")); return 1; }
    }
    int chunk=getenv("K3_CHUNK")?atoi(getenv("K3_CHUNK")):32;
    if(chunk<1) chunk=1;
    if(chunk>512) chunk=512;
    if(m.trace && chunk>1){
        chunk=1;                       /* trace rows are token-major by contract */
        fprintf(stderr,"[K3] K3_TRACE set: prefill chunk forced to 1\n");
    }
    double t0=now_s(); float *lo=NULL;
    m.head_greedy = (temp <= 0.f);
    for(int i=0;i<np;i+=chunk){
        int Cc=np-i<chunk?np-i:chunk;
        if(lo) free(lo);
        lo=step_chunk(&m,ids+i,i,Cc);
        fprintf(stderr,"\r[K3] prefill %d/%d (%.1fs)",i+Cc,np,now_s()-t0);
    }
    if(g_lfp){ fclose(g_lfp); g_lfp=NULL; }
    fprintf(stderr,"\n[K3] prefill done in %.1fs (%.2f tok/s)\n",now_s()-t0,np/(now_s()-t0));
    if(!m.has_head||!lo){
        fprintf(stderr,"[K3] no head — trace written, stopping after prefill\n");
        if(m.trace) fclose(m.trace);
        return 0;
    }
    double tg=now_s(); int ntok=0;
    char buf[512];
    /* chat print filter: hide the XTML structure, label the channels.
     * Structural runs are <|open|>/<|close|> TAGTEXT <|sep|> — suppress them
     * and print a channel banner when the response channel opens. */
    int xsup=0, xopen=0; char xtag[64]; int xtl=0;
    if(chat&&think){ printf("[think] "); fflush(stdout); }
    for(int s=0;s<ngen;s++){
        int t = (m.head_tok>=0) ? m.head_tok : sample_tok(lo,m.c.vocab,temp,1.f);
        free(lo); lo=NULL;
        int is_eos=0; for(int e=0;e<m.c.n_eos;e++) if(t==m.c.eos[e]) is_eos=1;
        int show=1;
        if(chat&&sp[0]>=0){
            if(t==sp[0]||t==sp[1]){ xsup=1; xopen=(t==sp[0]); xtl=0; show=0; }
            else if(t==sp[2]){
                if(xsup){ xsup=0; xtag[xtl]=0;
                    if(xopen&&!strcmp(xtag,"response")){ printf("\n\n[response] "); fflush(stdout); }
                }
                show=0;
            } else if(xsup){
                if(has_tok){ int n2=tok_decode(&T,&t,1,buf,sizeof(buf)-1);
                    if(xtl+n2<(int)sizeof(xtag)){ memcpy(xtag+xtl,buf,n2); xtl+=n2; } }
                show=0;
            } else if(t==sp[3]) show=0;
        }
        if(show && k3_net_rank()==0){    /* every rank generates; only rank 0 speaks */
            if(has_tok){ int n2=tok_decode(&T,&t,1,buf,sizeof(buf)-1); fwrite(buf,1,n2,stdout); fflush(stdout); }
            else { printf("%d ",t); fflush(stdout); }
        }
        ntok++;
        if(is_eos){ fprintf(stderr,"\n[K3] eos\n"); break; }
        if(np+ntok>=max_t){ fprintf(stderr,"\n[K3] context full\n"); break; }
        lo=step_chunk(&m,&t,np+ntok-1,1);
        double el=now_s()-tg;
        fprintf(stderr,"  [tok %d: %.1fs/tok, hit %.0f%%, %.1f GB read]\n",
                ntok,el/ntok,100.0*m.hits/(m.hits+m.miss+1e-9),m.ebytes/1e9);
    }
    if(lo) free(lo);
    double dt=now_s()-tg;
    fprintf(stderr,"\n[K3] decode %d tokens in %.1fs (%.2f tok/s) | expert hit %.1f%% (%llu/%llu) | %.1f GB streamed\n",
            ntok,dt,ntok/dt,100.0*m.hits/(m.hits+m.miss+1e-9),
            (unsigned long long)m.hits,(unsigned long long)(m.hits+m.miss),m.ebytes/1e9);
    fprintf(stderr,"[K3] time: attn %.1fs moe %.1fs (eload %.1fs) head %.1fs | RSS %.1f GB\n",
            m.t_attn,m.t_moe,m.t_eload,m.t_head,rss_gb());
    fprintf(stderr,"[K3] moe split: router %.1fs topk %.1fs latent %.1fs experts %.1fs"
            " (of which eload %.1fs) rnorm %.1fs shared %.1fs\n",
            m.t_router,m.t_topk,m.t_latent,m.t_expert,m.t_eload,m.t_rnorm,m.t_shared);
    fprintf(stderr,"[K3] expert region: GPU kernel %.1fs over %llu calls (%.3f ms each);"
            " everything else in the region %.1fs\n",
            m.t_ekernel,(unsigned long long)m.n_ekernel,
            m.n_ekernel?1e3*m.t_ekernel/(double)m.n_ekernel:0.0,
            m.t_expert-m.t_eload-m.t_ekernel);
    fprintf(stderr,"[K3] kda split: proj %.1fs conv %.1fs heads %.1fs out+reduce %.1fs\n",
            m.t_kproj,m.t_kconv,m.t_khead,m.t_kout);
    if(k3_net_world()>1)
        fprintf(stderr,"[K3/NET] allreduce %.1fs over %llu calls (%.2f ms each)\n",
            k3_net_secs(),(unsigned long long)k3_net_calls(),
            k3_net_calls()?1e3*k3_net_secs()/(double)k3_net_calls():0.0);
#ifdef COLI_CUDA
    k3_cuda_report();
#endif
    if(m.route_hist && getenv("K3_ROUTE_STATS")){
        k3_route_stats_write(&m);
        fprintf(stderr,"[K3] route stats -> %s (%d layers x %d experts)\n",
                getenv("K3_ROUTE_STATS"),m.c.n_layers,m.c.n_experts);
    }
    if(m.trace) fclose(m.trace);
    return 0;
}
