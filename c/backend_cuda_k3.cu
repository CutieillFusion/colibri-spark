/* MXFP4 routed-expert GEMV for the Kimi-K3 engine.
 *
 * Why this file exists: measured on GB10, MoE is ~84% of K3's decode time and
 * expert *loading* is only ~14% of that -- the async loader hides NVMe well.
 * The cost is the routed-expert matmuls themselves, which quant.h runs on the
 * CPU at ~11 GB/s. backend_cuda.cu cannot help: its formats are f32/int8/int4/
 * int2/grouped-int4/E8, none of which is MXFP4, and re-encoding e2m1 (a
 * non-uniform grid {0,.5,1,1.5,2,3,4,6}) into uniform int4 would be lossy. So
 * the experts get their own kernel, reading the checkpoint's native layout.
 *
 * Layout (compressed-tensors "mxfp4-pack-quantized", mirrored from quant.h):
 *   packed [O, I/2]  u8 -- e2m1 nibbles, LOW nibble = even column, bit3 = sign
 *   scales [O, I/32] u8 -- ue8m0, w = v * 2^(s-127), decoded by the (s<<23)
 *                          bit trick so GPU and CPU agree exactly
 *
 * Zero-copy: GB10 is an integrated device, so cudaMalloc+memcpy of a streamed
 * expert would duplicate it in the SAME physical RAM and burn ~26 GB/token of
 * copies. Instead the engine registers each expert slot allocation once
 * (they are posix_memalign'd once and thereafter only swapped between the LRU
 * and the working set) and the kernels read host memory directly. Unlike
 * backend_cuda.cu's int4 path this never mutates the buffer, so aliasing it is
 * safe -- the CPU fallback keeps working on the same bytes.
 *
 * Byte loads, not uint4: s->buf is s->base + (file_offset % 4096), which is not
 * guaranteed 16-byte aligned. A misaligned vector load is undefined in CUDA,
 * and consecutive threads reading consecutive 16-byte spans coalesce into the
 * same transactions anyway.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "backend_cuda_k3.h"

#define K3_THREADS 128

static int    g_dev = -1;
static int    g_ready = 0;
static float *g_z = nullptr, *g_gate = nullptr, *g_hz = nullptr;
static int    g_latent = 0, g_inter = 0;
static cudaStream_t g_stream = nullptr;

__constant__ float c_mx4_lut[16];
static const float h_mx4_lut[16] = { 0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                    -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f };

static int ck(cudaError_t e, const char *what) {
    if (e == cudaSuccess) return 1;
    fprintf(stderr, "[K3/EXP] %s: %s\n", what, cudaGetErrorString(e));
    return 0;
}

/* w = v * 2^(s-127); identical bit trick to quant.h's mx4_scale. */
__device__ __forceinline__ float mx4_scale_dev(unsigned char s) {
    return __uint_as_float((unsigned int)s << 23);
}

__device__ __forceinline__ float blk_reduce(float v, float *sh) {
    #pragma unroll
    for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int off = 16; off; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;                                  /* meaningful in thread 0 */
}

/* One dot product of a packed MXFP4 row against x, split over the block. */
__device__ __forceinline__ float row_dot(const unsigned char *pk, const unsigned char *sc,
                                         const float *x, int ng) {
    float acc = 0.f;
    for (int g = threadIdx.x; g < ng; g += blockDim.x) {
        const unsigned char *b = pk + (g << 4);        /* 32 values = 16 bytes */
        const float *xs = x + (g << 5);
        float p = 0.f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            unsigned char c = b[k];
            p += xs[2 * k] * c_mx4_lut[c & 0xF] + xs[2 * k + 1] * c_mx4_lut[c >> 4];
        }
        acc += p * mx4_scale_dev(sc[g]);
    }
    return acc;
}

/* gate = SiTU(w1 @ z, w3 @ z). Both rows share z, so one block does both and
 * the activation is fused into the epilogue -- no round trip for `up`. */
__global__ void k3_gate_up_situ(float *__restrict__ gate,
                                const unsigned char *__restrict__ w1p,
                                const unsigned char *__restrict__ w1s,
                                const unsigned char *__restrict__ w3p,
                                const unsigned char *__restrict__ w3s,
                                const float *__restrict__ z,
                                int I, float beta1, float beta2) {
    int o = blockIdx.x, ng = I >> 5;
    size_t rb = (size_t)(I >> 1);
    __shared__ float sh[K3_THREADS / 32];
    float a1 = row_dot(w1p + (size_t)o * rb, w1s + (size_t)o * ng, z, ng);
    float g1 = blk_reduce(a1, sh);
    __syncthreads();
    float a3 = row_dot(w3p + (size_t)o * rb, w3s + (size_t)o * ng, z, ng);
    float g3 = blk_reduce(a3, sh);
    if (threadIdx.x == 0)
        gate[o] = beta1 * tanhf(g1 / beta1) * (1.f / (1.f + expf(-g1)))
                * beta2 * tanhf(g3 / beta2);
}

__global__ void k3_down(float *__restrict__ hz,
                        const unsigned char *__restrict__ w2p,
                        const unsigned char *__restrict__ w2s,
                        const float *__restrict__ gate, int I) {
    int o = blockIdx.x, ng = I >> 5;
    size_t rb = (size_t)(I >> 1);
    __shared__ float sh[K3_THREADS / 32];
    float a = row_dot(w2p + (size_t)o * rb, w2s + (size_t)o * ng, gate, ng);
    float t = blk_reduce(a, sh);
    if (threadIdx.x == 0) hz[o] = t;
}

/* ---------------- packed 2-bit variant (format "k3-w2") ------------------
 * Same ue8m0 block-32 scales, but the codes are 2-bit indices into the
 * sign-symmetric codebook {-4,-1,1,4} -- 8 bytes per group instead of 16, so
 * 2.25 bits/weight and 1.89x less to move per expert.
 *
 * Symmetry is the whole point: the L2-optimal 2-bit codebook is
 * sign-asymmetric and degenerates these models (0/12 coherence in
 * vllm-moet/docs/quality.md), while this set at the same L2 error matches the
 * 4-bit baseline. Packing order is column-major within the byte -- bits 0-1
 * are column 0 -- so the kernel shifts in column order with no permute. */
__constant__ float c_w2_lut[4];
static const float h_w2_lut[4] = { -4.f, -1.f, 1.f, 4.f };

__device__ __forceinline__ float row_dot_w2(const unsigned char *pk, const unsigned char *sc,
                                            const float *x, int ng) {
    float acc = 0.f;
    for (int g = threadIdx.x; g < ng; g += blockDim.x) {
        const unsigned char *b = pk + (g << 3);      /* 32 values = 8 bytes */
        const float *xs = x + (g << 5);
        float p = 0.f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            unsigned char c = b[k];
            p += xs[4*k+0] * c_w2_lut[ c        & 3]
               + xs[4*k+1] * c_w2_lut[(c >> 2)  & 3]
               + xs[4*k+2] * c_w2_lut[(c >> 4)  & 3]
               + xs[4*k+3] * c_w2_lut[(c >> 6)  & 3];
        }
        acc += p * mx4_scale_dev(sc[g]);
    }
    return acc;
}

__global__ void k3_w2_gate_up_situ(float *__restrict__ gate,
                                   const unsigned char *__restrict__ w1p,
                                   const unsigned char *__restrict__ w1s,
                                   const unsigned char *__restrict__ w3p,
                                   const unsigned char *__restrict__ w3s,
                                   const float *__restrict__ z,
                                   int I, float beta1, float beta2) {
    int o = blockIdx.x, ng = I >> 5;
    size_t rb = (size_t)(I >> 2);                    /* 2 bits/weight */
    __shared__ float sh[K3_THREADS / 32];
    float a1 = row_dot_w2(w1p + (size_t)o * rb, w1s + (size_t)o * ng, z, ng);
    float g1 = blk_reduce(a1, sh);
    __syncthreads();
    float a3 = row_dot_w2(w3p + (size_t)o * rb, w3s + (size_t)o * ng, z, ng);
    float g3 = blk_reduce(a3, sh);
    if (threadIdx.x == 0)
        gate[o] = beta1 * tanhf(g1 / beta1) * (1.f / (1.f + expf(-g1)))
                * beta2 * tanhf(g3 / beta2);
}

__global__ void k3_w2_down(float *__restrict__ hz,
                           const unsigned char *__restrict__ w2p,
                           const unsigned char *__restrict__ w2s,
                           const float *__restrict__ gate, int I) {
    int o = blockIdx.x, ng = I >> 5;
    size_t rb = (size_t)(I >> 2);
    __shared__ float sh[K3_THREADS / 32];
    float a = row_dot_w2(w2p + (size_t)o * rb, w2s + (size_t)o * ng, gate, ng);
    float t = blk_reduce(a, sh);
    if (threadIdx.x == 0) hz[o] = t;
}

/* ---------------- warp-per-row 2-bit kernels (the fast path) --------------
 * The block-per-row layout above measured 15 GB/s against 59 GB/s for the
 * dense path on the same GPU. Cause: one 896-byte row spread over 128 threads
 * is ~7 bytes/thread, plus shared memory and two __syncthreads for the block
 * reductions, plus byte-at-a-time loads.
 *
 * Here one WARP owns a row and eight warps share a block:
 *   - 56 bytes/thread instead of 7
 *   - reduction is pure __shfl_down: no shared memory, no block sync
 *   - 8-byte loads. Safe only for the w2 store, whose slots are 4096-aligned
 *     and whose w1p/w1s offsets are multiples of 16 -- the MXFP4 path reads
 *     HF shard offsets that carry no such guarantee, which is why it uses
 *     byte loads.
 *   - lane L takes groups L, L+32, ... so a warp's 32 lanes read 256
 *     contiguous bytes: fully coalesced.
 */
#define K3_WARPS 8
#define K3_FAST_THREADS (K3_WARPS * 32)

/* {-4,-1,1,4} from a 2-bit index by arithmetic, not a table lookup.
 * c_w2_lut lives in constant memory, which broadcasts at full speed only when
 * every thread in the warp reads the SAME address; here each thread decodes
 * different codes, so the access serialises. Two selects are cheaper:
 *   sign = idx&2 ? + : -        mag = (idx ^ (idx>>1)) & 1 ? 1 : 4
 *   idx 0->-4  1->-1  2->+1  3->+4
 */
/* Codebook in constant memory so K3_W1 can collapse it at init.
 * K3_W1=1 sets {-a,-a,+a,+a}: because the 2-bit codebook is sign-symmetric,
 * sign(2-bit value) == sign(the original MXFP4 value), so this is EXACTLY the
 * sign-based 1-bit quantization of the checkpoint -- testable with no repack
 * and no new storage. a defaults to the mean |w| of the 2-bit distribution
 * (0.77*1 + 0.23*4 = 1.69), which minimises L2 for a two-level quantiser.
 * Measured earlier: LUT vs arithmetic made no difference (31.5 vs 31.6 GB/s). */
__constant__ float c_w2v[4];
__device__ __forceinline__ float w2_val(unsigned int idx) { return c_w2v[idx & 3u]; }

__device__ __forceinline__ float warp_row_dot_w2(const unsigned char *__restrict__ pk,
                                                 const unsigned char *__restrict__ sc,
                                                 const float *__restrict__ x,
                                                 int ng, int lane) {
    float acc = 0.f;
    /* 16 bytes = TWO groups per load. ng is even for both K3 shapes (112 and
     * 96) and pk is 16-byte aligned, so lane L can take groups 2L, 2L+1 and
     * halve the load instructions.
     *
     * UNVALIDATED ON THIS DEPLOYMENT. The float4 activation load below is
     * bit-exact and mechanical, but this whole kernel is dormant here: the
     * cluster runs the 1-bit store (K3-w1, --bits 1), so k3_w2_* never
     * executes and no A/B can price it. Committed as an instruction-count
     * reduction on reasoning alone, which is weaker than everything else in
     * this file.
     *
     * What is NOT addressed: lane L reads xs at stride 64 floats, so every lane
     * hits the same shared bank -- a 32-way conflict that the float4 load makes
     * wider while making it 4x rarer. The 1-bit kernel solves this with
     * K3_W1_SHSTRIDE padding; doing the same here means changing the shared
     * layout and its store loop, which is not something to land blind. */
    for (int g = lane * 2; g < ng; g += 64) {
        uint4 v = *(const uint4 *)(pk + (g << 3));
        const unsigned int w[4] = { v.x, v.y, v.z, v.w };
        #pragma unroll
        for (int h = 0; h < 2; h++) {                 /* h-th group in the pair */
            if (g + h >= ng) break;
            const float *xs = x + ((g + h) << 5);
            float p = 0.f;
            #pragma unroll
            for (int q = 0; q < 2; q++) {             /* 4 bytes = 16 codes */
                unsigned int u = w[h * 2 + q];
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    unsigned int c = (u >> (8 * k)) & 0xFFu;
                    /* One byte codes four consecutive activations, so the four
                     * scalar reads were always one float4. xs is x + ((g+h)<<5),
                     * i.e. 128-byte aligned, and q*16 + k*4 is a multiple of
                     * four floats, so the address is 16-byte aligned. Same
                     * values, same order, same adds -- bit-exact, one load
                     * instruction instead of four. This is what the 1-bit
                     * kernel already does (warp_row_dot_w1 pulls eight float4
                     * per group). */
                    float4 xb = *(const float4 *)(xs + q * 16 + k * 4);
                    p += xb.x * w2_val( c        & 3u)
                       + xb.y * w2_val((c >> 2)  & 3u)
                       + xb.z * w2_val((c >> 4)  & 3u)
                       + xb.w * w2_val((c >> 6)  & 3u);
                }
            }
            acc += p * mx4_scale_dev(sc[g + h]);
        }
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;                                   /* lane 0 holds the total */
}

/* The activations dominate the traffic: one 8-byte weight group is multiplied
 * against 32 floats (128 B) of x, so x outweighs weights 16:1, and every warp
 * was re-reading the same 14 KB. Staging x in shared memory once per block
 * cuts that by the warp count. K3_SHM_MAX caps the shape we will do this for;
 * anything larger falls back to reading x from global. */
#define K3_SHM_MAX 4096
extern __shared__ float shx[];

__global__ void k3_w2_gate_up_fast(float *__restrict__ gate,
                                   const unsigned char *__restrict__ w1p,
                                   const unsigned char *__restrict__ w1s,
                                   const unsigned char *__restrict__ w3p,
                                   const unsigned char *__restrict__ w3s,
                                   const float *__restrict__ z,
                                   int I, int O, float beta1, float beta2) {
    for (int i = threadIdx.x; i < I; i += blockDim.x) shx[i] = z[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    int ng = I >> 5;
    size_t rb = (size_t)(I >> 2);
    float g1 = warp_row_dot_w2(w1p + (size_t)o * rb, w1s + (size_t)o * ng, shx, ng, lane);
    float g3 = warp_row_dot_w2(w3p + (size_t)o * rb, w3s + (size_t)o * ng, shx, ng, lane);
    if (lane == 0)
        gate[o] = beta1 * tanhf(g1 / beta1) * (1.f / (1.f + expf(-g1)))
                * beta2 * tanhf(g3 / beta2);
}

__global__ void k3_w2_down_fast(float *__restrict__ hz,
                                const unsigned char *__restrict__ w2p,
                                const unsigned char *__restrict__ w2s,
                                const float *__restrict__ gate, int I, int O) {
    for (int i = threadIdx.x; i < I; i += blockDim.x) shx[i] = gate[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    int ng = I >> 5;
    size_t rb = (size_t)(I >> 2);
    float t = warp_row_dot_w2(w2p + (size_t)o * rb, w2s + (size_t)o * ng, shx, ng, lane);
    if (lane == 0) hz[o] = t;
}

extern "C" int coli_k3_expert_w2(const void *w1p, const void *w1s,
                                 const void *w2p, const void *w2s,
                                 const void *w3p, const void *w3s,
                                 float *hz, const float *z,
                                 int latent, int inter, float beta1, float beta2) {
    if (!g_ready || latent != g_latent || inter != g_inter) return 0;
    if (!ck(cudaMemcpyAsync(g_z, z, (size_t)latent * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "z upload")) return 0;
    int gu_blocks = (inter + K3_WARPS - 1) / K3_WARPS;
    int dn_blocks = (latent + K3_WARPS - 1) / K3_WARPS;
    k3_w2_gate_up_fast<<<gu_blocks, K3_FAST_THREADS, latent*sizeof(float), g_stream>>>(
        g_gate, (const unsigned char *)w1p, (const unsigned char *)w1s,
        (const unsigned char *)w3p, (const unsigned char *)w3s, g_z, latent, inter, beta1, beta2);
    k3_w2_down_fast<<<dn_blocks, K3_FAST_THREADS, inter*sizeof(float), g_stream>>>(
        g_hz, (const unsigned char *)w2p, (const unsigned char *)w2s, g_gate, inter, latent);
    if (!ck(cudaMemcpyAsync(hz, g_hz, (size_t)latent * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "hz download")) return 0;
    if (!ck(cudaStreamSynchronize(g_stream), "expert sync")) return 0;
    return 1;
}

/* ---------------- batched 2-bit experts (one launch per layer) -----------
 * Per-expert dispatch costs two launches and a stream sync each: at 16 experts
 * x 92 layers that is ~2944 launches and ~1472 syncs per token, which showed up
 * as expert compute taking 0.86 s/token when the bytes moved imply ~0.24 s.
 * Batching a whole layer's experts into one grid removes the syncs and lets the
 * GPU overlap the small per-expert GEMVs. blockIdx.y selects the expert. */
#define K3_BATCH_MAX 64

typedef struct { const unsigned char *w1p,*w1s,*w2p,*w2s,*w3p,*w3s; } K3Expert;

static K3Expert *g_dev_ex = nullptr;
static float    *g_gate_b = nullptr, *g_hz_b = nullptr;
static int       g_batch_cap = 0;

__global__ void k3_w2_gate_up_situ_b(float *__restrict__ gate,
                                     const K3Expert *__restrict__ ex,
                                     const float *__restrict__ z,
                                     int I, int O, float beta1, float beta2) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = blockIdx.y;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    int ng = I >> 5; size_t rb = (size_t)(I >> 2);
    float g1 = warp_row_dot_w2(ex[j].w1p + (size_t)o * rb, ex[j].w1s + (size_t)o * ng, z, ng, lane);
    float g3 = warp_row_dot_w2(ex[j].w3p + (size_t)o * rb, ex[j].w3s + (size_t)o * ng, z, ng, lane);
    if (lane == 0)
        gate[(size_t)j * O + o] = beta1 * tanhf(g1 / beta1) * (1.f / (1.f + expf(-g1)))
                                * beta2 * tanhf(g3 / beta2);
}

__global__ void k3_w2_down_b(float *__restrict__ hz, const K3Expert *__restrict__ ex,
                             const float *__restrict__ gate, int I, int O) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = blockIdx.y;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    int ng = I >> 5; size_t rb = (size_t)(I >> 2);
    float t = warp_row_dot_w2(ex[j].w2p + (size_t)o * rb, ex[j].w2s + (size_t)o * ng,
                              gate + (size_t)j * I, ng, lane);
    if (lane == 0) hz[(size_t)j * O + o] = t;
}

static int ensure_batch(int n, int latent, int inter) {
    if (n <= g_batch_cap) return 1;
    if (g_dev_ex) { cudaFree(g_dev_ex); cudaFree(g_gate_b); cudaFree(g_hz_b); }
    if (!ck(cudaMalloc(&g_dev_ex, (size_t)n * sizeof(K3Expert)), "batch desc") ||
        !ck(cudaMalloc(&g_gate_b, (size_t)n * inter * sizeof(float)), "batch gate") ||
        !ck(cudaMalloc(&g_hz_b, (size_t)n * latent * sizeof(float)), "batch hz")) {
        g_batch_cap = 0; return 0;
    }
    g_batch_cap = n;
    return 1;
}

/* hz_all is [n][latent]; the caller still folds in the routing weights. */
extern "C" int coli_k3_expert_batch_w2(const void *const *w1p, const void *const *w1s,
                                       const void *const *w2p, const void *const *w2s,
                                       const void *const *w3p, const void *const *w3s,
                                       int n, float *hz_all, const float *z,
                                       int latent, int inter, float beta1, float beta2) {
    if (!g_ready || n < 1 || n > K3_BATCH_MAX) return 0;
    if (latent != g_latent || inter != g_inter) return 0;
    if (!ensure_batch(n, latent, inter)) return 0;

    K3Expert host[K3_BATCH_MAX];
    for (int j = 0; j < n; j++) {
        host[j].w1p = (const unsigned char *)w1p[j]; host[j].w1s = (const unsigned char *)w1s[j];
        host[j].w2p = (const unsigned char *)w2p[j]; host[j].w2s = (const unsigned char *)w2s[j];
        host[j].w3p = (const unsigned char *)w3p[j]; host[j].w3s = (const unsigned char *)w3s[j];
    }
    if (!ck(cudaMemcpyAsync(g_dev_ex, host, (size_t)n * sizeof(K3Expert),
                            cudaMemcpyHostToDevice, g_stream), "batch desc upload")) return 0;
    if (!ck(cudaMemcpyAsync(g_z, z, (size_t)latent * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "z upload")) return 0;
    dim3 gu((unsigned)((inter+K3_WARPS-1)/K3_WARPS), (unsigned)n);
    dim3 dn((unsigned)((latent+K3_WARPS-1)/K3_WARPS), (unsigned)n);
    k3_w2_gate_up_situ_b<<<gu, K3_FAST_THREADS, 0, g_stream>>>(g_gate_b, g_dev_ex, g_z,
                                                               latent, inter, beta1, beta2);
    k3_w2_down_b<<<dn, K3_FAST_THREADS, 0, g_stream>>>(g_hz_b, g_dev_ex, g_gate_b, inter, latent);
    if (!ck(cudaMemcpyAsync(hz_all, g_hz_b, (size_t)n * latent * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "hz download")) return 0;
    if (!ck(cudaStreamSynchronize(g_stream), "batch sync")) return 0;   /* ONE sync per layer */
    return 1;
}


/* ---------------- packed 1-bit experts (format "k3-w1") ------------------
 * One SIGN bit per weight, 8 per byte, keeping the ue8m0 block-32 scales:
 * 1.25 bits/weight, slot 4.92 MiB (vs 8.86 at 2-bit, 16.74 at MXFP4), expert
 * set 1447 -> 425 GB. Value = bit ? +a : -a.
 *
 * Viable because the codebook {-a,+a} is sign-symmetric BY CONSTRUCTION, which
 * is the property that decides whether these models survive quantisation --
 * not the L2 error, which is 0.64 here versus 0.38 at 2 bits. Verified
 * coherent on K3 before this was built, by collapsing the 2-bit codebook in
 * the kernel (K3_W1=1) -- no repack needed for the quality answer.
 *
 * `a` folds into the group scale, so the kernel just picks +/-x per bit and
 * multiplies once per group. 32 values = 4 bytes. */
__constant__ float c_w1a[1];

/* x is staged with a 33-float stride per 32-value group, not 32.
 *
 * At the natural stride, lane L owns group L+32k and reads shx[32L + 1024k + j],
 * so every lane in the warp lands on bank j: a 32-way shared-memory bank
 * conflict on each of the 32 loads per group. A probe that kept the loads but
 * removed the arithmetic ran at exactly the same speed as the real kernel
 * (0.0515 ms both), while a version that dropped only the shared reads ran 5x
 * faster -- the ALU was free and shared memory was the whole cost. Padding to
 * 33 makes lane L read bank (L + j) % 32, all 32 distinct. Same values in the
 * same order; 66 -> 168 GB/s of weight bytes.
 *
 * warp_row_dot_w2 has the identical indexing and therefore the identical
 * conflict. It is left alone only because the 2-bit store is not what this
 * deployment runs, so the fix could not be validated end to end here. */
/* Stride 36, not 33. 33 fixes the bank conflict for SCALAR loads but is not
 * 16-byte aligned, so it forbids vector loads. 36 floats = 144 B is 16-byte
 * aligned AND stays conflict-free for 128-bit accesses: the hardware splits a
 * warp's float4 loads into four phases of eight lanes, and eight lanes x four
 * banks each covers exactly the 32 banks. That turns the 32 scalar shared
 * loads per group into 8 float4 loads. Measured on the gate_up shape:
 * stride 33 scalar 0.0205 ms / 168 GB/s -> stride 36 vector 0.0153 ms /
 * 225 GB/s, against 0.0102 ms for a variant that reads no activation at all. */
#define K3_W1_SHSTRIDE 36
#define K3_W1_SHFLOATS(I) ((size_t)((I) >> 5) * K3_W1_SHSTRIDE)

__device__ __forceinline__ float warp_row_dot_w1(const unsigned char *__restrict__ pk,
                                                 const unsigned char *__restrict__ sc,
                                                 const float *__restrict__ x,
                                                 int ng, int lane) {
    float acc = 0.f;
    for (int g = lane; g < ng; g += 32) {
        unsigned int v = *(const unsigned int *)(pk + (g << 2));   /* 32 bits */
        /* 8 float4 loads cover the group's 32 activations; q0..q3 keep them in
         * the same order the four partial sums consumed them before. */
        const float4 *x4 = (const float4 *)(x + g * K3_W1_SHSTRIDE);
        float4 a0=x4[0],a1=x4[1],a2=x4[2],a3=x4[3],a4=x4[4],a5=x4[5],a6=x4[6],a7=x4[7];
        const float q0[8]={a0.x,a0.y,a0.z,a0.w,a1.x,a1.y,a1.z,a1.w};
        const float q1[8]={a2.x,a2.y,a2.z,a2.w,a3.x,a3.y,a3.z,a3.w};
        const float q2[8]={a4.x,a4.y,a4.z,a4.w,a5.x,a5.y,a5.z,a5.w};
        const float q3[8]={a6.x,a6.y,a6.z,a6.w,a7.x,a7.y,a7.z,a7.w};
        /* Branchless sign flip: XOR bit 31 when the weight bit is 0. The
         * obvious `bit ? x : -x` costs a select per weight and 32 serial loop
         * iterations per group -- versus 8 iterations x 4 values in the 2-bit
         * kernel -- which made 1-bit SLOWER than 2-bit despite moving half the
         * bytes (moe 22.4 -> 27.3 s). Four partial sums restore the ILP the
         * 2-bit path gets for free. */
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned int b0 = (~v >> (j))      & 1u, b1 = (~v >> (j + 8))  & 1u;
            unsigned int b2 = (~v >> (j + 16)) & 1u, b3 = (~v >> (j + 24)) & 1u;
            p0 += __uint_as_float(__float_as_uint(q0[j]) ^ (b0 << 31));
            p1 += __uint_as_float(__float_as_uint(q1[j]) ^ (b1 << 31));
            p2 += __uint_as_float(__float_as_uint(q2[j]) ^ (b2 << 31));
            p3 += __uint_as_float(__float_as_uint(q3[j]) ^ (b3 << 31));
        }
        acc += ((p0 + p1) + (p2 + p3)) * mx4_scale_dev(sc[g]);
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc;
}

__global__ void k3_w1_gate_up_fast(float *__restrict__ gate,
                                   const unsigned char *__restrict__ w1p,
                                   const unsigned char *__restrict__ w1s,
                                   const unsigned char *__restrict__ w3p,
                                   const unsigned char *__restrict__ w3s,
                                   const float *__restrict__ z,
                                   int I, int O, float beta1, float beta2) {
    for (int i = threadIdx.x; i < I; i += blockDim.x) shx[(i >> 5) * K3_W1_SHSTRIDE + (i & 31)] = z[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    int ng = I >> 5; size_t rb = (size_t)(I >> 3);
    float a = c_w1a[0];
    float g1 = warp_row_dot_w1(w1p + (size_t)o*rb, w1s + (size_t)o*ng, shx, ng, lane) * a;
    float g3 = warp_row_dot_w1(w3p + (size_t)o*rb, w3s + (size_t)o*ng, shx, ng, lane) * a;
    if (lane == 0)
        gate[o] = beta1 * tanhf(g1 / beta1) * (1.f / (1.f + expf(-g1)))
                * beta2 * tanhf(g3 / beta2);
}

__global__ void k3_w1_down_fast(float *__restrict__ hz,
                                const unsigned char *__restrict__ w2p,
                                const unsigned char *__restrict__ w2s,
                                const float *__restrict__ gate, int I, int O) {
    for (int i = threadIdx.x; i < I; i += blockDim.x) shx[(i >> 5) * K3_W1_SHSTRIDE + (i & 31)] = gate[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    int ng = I >> 5; size_t rb = (size_t)(I >> 3);
    float t = warp_row_dot_w1(w2p + (size_t)o*rb, w2s + (size_t)o*ng, shx, ng, lane) * c_w1a[0];
    if (lane == 0) hz[o] = t;
}


/* ---------------- batched 1-bit expert kernels -----------------------------
 * ncu on the single-expert path: Waves Per SM = 1.56. The grid barely fills the
 * machine once, so these kernels are tail-bound rather than bandwidth-bound --
 * DRAM and L2 throughput both sit near 19% and 7% of peak. Decode applies every
 * routed expert of a layer to the SAME z, so they are independent and can share
 * one launch: blockIdx.y selects the expert. About 4.6 experts a layer per rank
 * turns 1.56 waves into ~7 and collapses ~846 launches a token into ~184.
 *
 * Bit-exact: each (expert, row) pair runs exactly the arithmetic it ran alone,
 * in the same order. Only the launch geometry changes.
 *
 * The 2-bit store already had this (k3_w2_*_b); the 1-bit store is what this
 * deployment actually runs, and it did not. */
__global__ void k3_w1_gate_up_fast_b(float *__restrict__ gate_all,
                                     const K3Expert *__restrict__ ex,
                                     const float *__restrict__ z,
                                     int I, int O, float beta1, float beta2) {
    const K3Expert e = ex[blockIdx.y];
    for (int i = threadIdx.x; i < I; i += blockDim.x) shx[(i >> 5) * K3_W1_SHSTRIDE + (i & 31)] = z[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    int ng = I >> 5; size_t rb = (size_t)(I >> 3);
    float a = c_w1a[0];
    float g1 = warp_row_dot_w1(e.w1p + (size_t)o*rb, e.w1s + (size_t)o*ng, shx, ng, lane) * a;
    float g3 = warp_row_dot_w1(e.w3p + (size_t)o*rb, e.w3s + (size_t)o*ng, shx, ng, lane) * a;
    if (lane == 0)
        gate_all[(size_t)blockIdx.y * O + o] =
              beta1 * tanhf(g1 / beta1) * (1.f / (1.f + expf(-g1)))
            * beta2 * tanhf(g3 / beta2);
}

__global__ void k3_w1_down_fast_b(float *__restrict__ hz_all,
                                  const K3Expert *__restrict__ ex,
                                  const float *__restrict__ gate_all, int I, int O) {
    const K3Expert e = ex[blockIdx.y];
    const float *gate = gate_all + (size_t)blockIdx.y * I;
    for (int i = threadIdx.x; i < I; i += blockDim.x) shx[(i >> 5) * K3_W1_SHSTRIDE + (i & 31)] = gate[i];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    int ng = I >> 5; size_t rb = (size_t)(I >> 3);
    float t = warp_row_dot_w1(e.w2p + (size_t)o*rb, e.w2s + (size_t)o*ng, shx, ng, lane) * c_w1a[0];
    if (lane == 0) hz_all[(size_t)blockIdx.y * O + o] = t;
}

extern "C" int coli_k3_expert_w1(const void *w1p, const void *w1s,
                                 const void *w2p, const void *w2s,
                                 const void *w3p, const void *w3s,
                                 float *hz, const float *z,
                                 int latent, int inter, float beta1, float beta2) {
    if (!g_ready || latent != g_latent || inter != g_inter) return 0;
    if (!ck(cudaMemcpyAsync(g_z, z, (size_t)latent*sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "z upload")) return 0;
    int gu = (inter + K3_WARPS - 1) / K3_WARPS, dn = (latent + K3_WARPS - 1) / K3_WARPS;
    k3_w1_gate_up_fast<<<gu, K3_FAST_THREADS,
                         K3_W1_SHFLOATS(latent)*sizeof(float), g_stream>>>(
        g_gate, (const unsigned char*)w1p, (const unsigned char*)w1s,
        (const unsigned char*)w3p, (const unsigned char*)w3s, g_z, latent, inter, beta1, beta2);
    k3_w1_down_fast<<<dn, K3_FAST_THREADS,
                      K3_W1_SHFLOATS(inter)*sizeof(float), g_stream>>>(
        g_hz, (const unsigned char*)w2p, (const unsigned char*)w2s, g_gate, inter, latent);
    if (!ck(cudaMemcpyAsync(hz, g_hz, (size_t)latent*sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "hz download")) return 0;
    return ck(cudaStreamSynchronize(g_stream), "expert sync");
}

extern "C" int coli_k3_init(int device, int latent, int inter) {
    if (g_ready) return 1;
    if ((latent & 31) || (inter & 31)) {
        fprintf(stderr, "[K3/EXP] latent=%d inter=%d must be multiples of 32\n", latent, inter);
        return 0;
    }
    if (!ck(cudaSetDevice(device), "set device")) return 0;
    cudaDeviceProp prop{};
    if (!ck(cudaGetDeviceProperties(&prop, device), "device properties")) return 0;
    if (!prop.canMapHostMemory) {
        fprintf(stderr, "[K3/EXP] device %d cannot map host memory — expert GPU path unavailable\n", device);
        return 0;
    }
    if (!ck(cudaMemcpyToSymbol(c_mx4_lut, h_mx4_lut, sizeof(h_mx4_lut)), "lut upload")) return 0;
    if (!ck(cudaMemcpyToSymbol(c_w2_lut, h_w2_lut, sizeof(h_w2_lut)), "w2 lut upload")) return 0;
    {   float v[4] = { -4.f, -1.f, 1.f, 4.f };
        if (getenv("K3_W1") && atoi(getenv("K3_W1"))) {
            float a = getenv("K3_W1_A") ? (float)atof(getenv("K3_W1_A")) : 1.69f;
            v[0] = v[1] = -a; v[2] = v[3] = a;
            fprintf(stderr, "[K3/EXP] K3_W1: experts collapsed to 1 bit {%+.2f,%+.2f}\n", -a, a);
        }
        if (!ck(cudaMemcpyToSymbol(c_w2v, v, sizeof(v)), "w2v upload")) return 0;
        float a1 = getenv("K3_W1_A") ? (float)atof(getenv("K3_W1_A")) : 1.69f;
        if (!ck(cudaMemcpyToSymbol(c_w1a, &a1, sizeof(a1)), "w1a upload")) return 0;
    }
    if (!ck(cudaStreamCreate(&g_stream), "stream") ||
        !ck(cudaMalloc(&g_z, (size_t)latent * sizeof(float)), "z scratch") ||
        !ck(cudaMalloc(&g_gate, (size_t)inter * sizeof(float)), "gate scratch") ||
        !ck(cudaMalloc(&g_hz, (size_t)latent * sizeof(float)), "hz scratch")) return 0;
    g_dev = device; g_latent = latent; g_inter = inter; g_ready = 1;
    fprintf(stderr, "[K3/EXP] MXFP4 expert kernel ready on device %d (%s, integrated=%d)\n",
            device, prop.name, prop.integrated);
    return 1;
}

/* Map one expert-slot allocation for zero-copy. Called once per slot, at the
 * posix_memalign that creates it. Requires the device pointer to alias the host
 * one (true under UVA); anything else would need per-call translation, so we
 * refuse instead of silently reading the wrong memory. */
extern "C" int coli_k3_register(void *p, size_t bytes) {
    if (!g_ready) return 0;
    cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterMapped);
    if (e == cudaErrorHostMemoryAlreadyRegistered) return 1;
    if (!ck(e, "host register")) return 0;
    void *dp = nullptr;
    if (!ck(cudaHostGetDevicePointer(&dp, p, 0), "device pointer")) return 0;
    if (dp != p) {
        fprintf(stderr, "[K3/EXP] mapped pointer %p != host %p — refusing zero-copy\n", dp, p);
        cudaHostUnregister(p);
        return 0;
    }
    return 1;
}

/* Release a mapping made by coli_k3_register. The engine never needs this --
 * expert slots live for the life of the process -- but any caller that frees a
 * registered buffer MUST unregister first: the pages stay mapped after free(),
 * so the next malloc reusing that address fails with "memory range is already
 * mapped" and poisons every later CUDA call. That is exactly what made
 * tests/bench_k3_dense report only its first shape. */
extern "C" void coli_k3_unregister(void *p) {
    if (!g_ready || !p) return;
    cudaHostUnregister(p);
    cudaGetLastError();          /* clear; do not let it stick to later launches */
}

extern "C" int coli_k3_expert(const void *w1p, const void *w1s,
                              const void *w2p, const void *w2s,
                              const void *w3p, const void *w3s,
                              float *hz, const float *z,
                              int latent, int inter, float beta1, float beta2) {
    if (!g_ready || latent != g_latent || inter != g_inter) return 0;
    if (!ck(cudaMemcpyAsync(g_z, z, (size_t)latent * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "z upload")) return 0;
    k3_gate_up_situ<<<inter, K3_THREADS, 0, g_stream>>>(
        g_gate, (const unsigned char *)w1p, (const unsigned char *)w1s,
        (const unsigned char *)w3p, (const unsigned char *)w3s, g_z, latent, beta1, beta2);
    k3_down<<<latent, K3_THREADS, 0, g_stream>>>(
        g_hz, (const unsigned char *)w2p, (const unsigned char *)w2s, g_gate, inter);
    if (!ck(cudaMemcpyAsync(hz, g_hz, (size_t)latent * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "hz download")) return 0;
    if (!ck(cudaStreamSynchronize(g_stream), "expert sync")) return 0;
    return 1;
}


/* Bench-only: copy a buffer into DEVICE memory and hand back the pointer.
 * The kernels do not care whether a pointer is host-mapped or device-resident,
 * so this isolates the cost of zero-copy from the cost of the kernel itself. */
extern "C" void *coli_k3_devcopy(const void *src, size_t n) {
    void *d = nullptr;
    if (!ck(cudaMalloc(&d, n), "devcopy alloc")) return nullptr;
    if (!ck(cudaMemcpy(d, src, n, cudaMemcpyHostToDevice), "devcopy")) { cudaFree(d); return nullptr; }
    return d;
}

extern "C" void coli_k3_shutdown(void) {
    if (!g_ready) return;
    cudaFree(g_z); cudaFree(g_gate); cudaFree(g_hz);
    cudaStreamDestroy(g_stream);
    g_ready = 0;
}

/* ---------------- K3 dense GEMV (fmt 1 = int8, fmt 4 = int4-g64) ----------
 * backend_cuda.cu's quant_matmul carries the same three costs the expert
 * kernel had, and it measured 59 GB/s against this one's 90+:
 *   - one block per output row, reading x from GLOBAL every time, so the
 *     activation vector is re-read by every block (the 16:1 traffic problem)
 *   - a shared-memory block reduction with log2(256) = 8 __syncthreads
 *   - fmt=4 does an integer DIVIDE per element (i / gs) to find the group
 * Here: one warp per row, x staged in shared once per block, shuffle-only
 * reduction, and a shift instead of the divide (gs is a power of two).
 *
 * Weights are read zero-copy from the host W buffers -- measured free on this
 * integrated part -- so unlike coli_cuda_matmul this uploads nothing and
 * duplicates nothing, returning ~36 GB of RAM to the expert cache, which is
 * what actually binds decode.
 *
 * quant_matmul is left alone: it is shared with colibri/inkling and cannot be
 * regression-tested from here.
 */
extern __shared__ float shx_d[];

/* stage=1 copies x into shared; stage=0 reads it from global. Staging is only
 * a win when x is SMALL: at I=12288 it costs 49 KB/block and drops occupancy to
 * 1-2 blocks/SM, which measured 1.8x SLOWER than the stock kernel. The expert
 * kernel benefits because its x is 3584 floats (14 KB); dense x is 7168-12288. */
__global__ void k3_dense_i4g(float *__restrict__ y, const float *__restrict__ x,
                             const unsigned char *__restrict__ q4,
                             const float *__restrict__ scl,
                             int I, int O, int gsh, int ng, int stage) {
    const float *xv = x;
    if (stage) {
        for (int i = threadIdx.x; i < I; i += blockDim.x) shx_d[i] = x[i];
        __syncthreads();
        xv = shx_d;
    }
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    size_t rb = (size_t)((I + 1) >> 1);
    const unsigned char *w = q4 + (size_t)o * rb;
    const float *s = scl + (size_t)o * ng;
    int gsz = 1 << gsh;                       /* group size, power of two */
    float acc = 0.f;
    for (int g = lane; g < ng; g += 32) {
        const unsigned char *b = w + ((size_t)g << (gsh - 1));   /* gsz/2 bytes */
        const float *xs = xv + ((size_t)g << gsh);
        int n = I - (g << gsh); if (n > gsz) n = gsz;
        float p = 0.f;
        for (int k = 0; k < (n >> 1); k++) {
            unsigned char c = b[k];
            p += xs[2*k]   * (float)((int)(c & 0xF) - 8)
               + xs[2*k+1] * (float)((int)(c >> 4)  - 8);
        }
        acc += p * s[g];
    }
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[o] = acc;
}

__global__ void k3_dense_i8(float *__restrict__ y, const float *__restrict__ x,
                            const signed char *__restrict__ q8,
                            const float *__restrict__ scl, int I, int O, int stage) {
    const float *xv = x;
    if (stage) {
        for (int i = threadIdx.x; i < I; i += blockDim.x) shx_d[i] = x[i];
        __syncthreads();
        xv = shx_d;
    }
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int o = blockIdx.x * K3_WARPS + warp;
    if (o >= O) return;
    const signed char *w = q8 + (size_t)o * I;
    float acc = 0.f;
    for (int i = lane; i < I; i += 32) acc += xv[i] * (float)w[i];
    #pragma unroll
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[o] = acc * scl[o];
}

/* Arithmetic-order-compatible zero-copy kernels.  These deliberately mirror
 * backend_cuda.cu's quant_matmul decode loop and 256-thread tree reduction.
 * The faster warp-per-row kernels above change the summation order; their
 * relative error is tiny, but K3's close logits can still flip a greedy token.
 * Keeping the stock order lets us isolate the unified-memory benefit of not
 * duplicating resident weights without changing model arithmetic. */
__global__ void k3_dense_i4g_exact(float *__restrict__ y, const float *__restrict__ x,
                                   const unsigned char *__restrict__ q4,
                                   const float *__restrict__ scales,
                                   int I, int O, int gsh, int ng) {
    int o = blockIdx.x;
    if (o >= O) return;
    size_t rb = (size_t)((I + 1) >> 1);
    const unsigned char *w = q4 + (size_t)o * rb;
    const float *scl = scales + (size_t)o * ng;
    /* Preserve the stock 256-lane reduction tree on 128 physical threads:
     * s0 and s1 are exactly original partial[t] and partial[t+128], and their
     * addition is exactly the tree's first edge. This can double resident
     * blocks without changing a floating-point association. */
    float s0 = 0.f, s1 = 0.f;
    for (int i = threadIdx.x; i < I; i += 256) {
        unsigned char b = w[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 15);
        int g = i >> gsh;
        if (g >= ng) g = ng - 1;
        s0 += x[i] * (float)(n - 8) * scl[g];
    }
    for (int i = threadIdx.x + 128; i < I; i += 256) {
        unsigned char b = w[i >> 1];
        int n = (i & 1) ? (b >> 4) : (b & 15);
        int g = i >> gsh;
        if (g >= ng) g = ng - 1;
        s1 += x[i] * (float)(n - 8) * scl[g];
    }
    __shared__ float partial[128];
    partial[threadIdx.x] = s0+s1;
    __syncthreads();
    for (int n = 64; n >= 32; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float v=partial[threadIdx.x];
        #pragma unroll
        for (int n=16;n;n>>=1) v+=__shfl_down_sync(0xffffffffu,v,n);
        if (!threadIdx.x) y[o]=v;
    }
}

/* W-wide consecutive fold: 256/W physical threads, thread p owning logical
 * lanes W*p .. W*p+W-1 in W accumulators merged only at the very end.
 *
 * Exact because the stock 256-lane tree combines lane bit 7 FIRST and the low
 * bits LAST. Grouping lanes by their high bits therefore leaves exactly the
 * bottom log2(W) edges to replay, which is
 *     for (step = W/2; step >= 1; step >>= 1) V[j] += V[j+step], j < step
 * -- for W=4 that is (V0+V2)+(V1+V3), for W=8
 * ((V0+V4)+(V2+V6))+((V1+V5)+(V3+V7)) -- while each chain's own reduction over
 * the 256/W threads replays lane bits 7..log2(W) in the stock order.
 *
 * The point is that those W lanes are W CONSECUTIVE elements, so one step costs
 * W/4 float4 loads, W/2 bytes of int4 weight (a uint once W>=8) and, since gs is
 * a multiple of W, ONE group scale. W=4 is 4 loads per 4 elements against the
 * twelve the {t, t+128} fold needs.
 *
 * W=8 halves the loads again and is bit-exact, but measured SLOWER on mirrors
 * (shared gate 377 -> 296 GB/s, lat_up 379 -> 311): 32 threads is one warp, so
 * 24 blocks/SM reaches only 768 threads. W=4 is the sweet spot. */

/* Same arithmetic as k3_dense_i4g_exactW<W>, R rows per block.
 *
 * A short row underfills the one-row-per-block launch: lat_up is [7168, 3584],
 * so each block reads 1792 bytes and runs the 4-wide loop about 3.5 times with
 * only 64 threads. Each row keeps a PRIVATE 256-lane tree over its own
 * T = 256/W threads, so no summation order changes and the result stays
 * bit-exact -- this only raises warps per block, the one occupancy knob left
 * once W is pinned at 4 by the width sweep.
 *
 * Applied to every shape it was a wash (4.234 vs 4.258). Applied only to short
 * rows it takes latent from 1.596 to 1.555 s/100 and is neutral to slightly
 * positive end to end, verified 6/6 byte-identical. */


/* ---------------- two-pass router ------------------------------------------
 * The router is 2.36 GB/token and drives top-16-of-896 selection, where the
 * measured 16th-to-17th score gap is 0.0032. Quantizing it outright fails the
 * quality bar -- int8 scored PCC 0.985, fp16 0.994 -- because the error is
 * comparable to that gap and swaps experts.
 *
 * So quantize only the SEARCH and keep the DECISION exact: score all 896 rows
 * in int8, take the 16th best, then re-score in f32 only the rows within delta
 * of it. Measured candidate counts at the boundary are sparse -- 16 rows at
 * delta 0.002, 29 at 0.010, 78 at 0.020 -- against an int8 RMS score error of
 * ~0.0016, so delta 0.01 is past 6 sigma while touching 3.2% of the rows.
 * Bytes: 0.59 GB int8 + ~0.08 GB gathered f32 against 2.36 GB.
 *
 * Three kernels chained on one stream, so this costs two extra launches per
 * layer (~4 us each) and no extra round trip -- the host still gets one
 * download of 896 scores. */

/* 16th-largest of (sc + rbias), then every row within delta of it. One block,
 * K masked max-reductions -- 16 passes of a 256-way tree, a few hundred steps. */
/* The engine ranks by sigmoid(score) + rbias, not score + rbias. Sigmoid is
 * monotonic, but the bias is added AFTER it, so the two orderings differ and
 * selecting on the raw score picks the wrong candidates -- which measured as
 * PCC 0.985, indistinguishable from having no second pass at all. */
__device__ __forceinline__ float k3_rank(float raw, float bias) {
    return (float)(1.0 / (1.0 + exp(-(double)raw))) + bias;
}

__global__ void k3_router_cand(const float *__restrict__ sc,
                               const float *__restrict__ rbias,
                               int E, int K, float delta,
                               int *__restrict__ cand, int *__restrict__ ncand,
                               int cap) {
    __shared__ float sv[256];
    __shared__ int   si[256];
    __shared__ unsigned char taken[1024];
    __shared__ float kth;
    int p = threadIdx.x, T = blockDim.x;
    for (int e = p; e < E; e += T) taken[e] = 0;
    if (!p) { *ncand = 0; kth = -3.0e38f; }
    __syncthreads();
    for (int k = 0; k < K; k++) {
        float bv = -3.0e38f; int ba = -1;
        for (int e = p; e < E; e += T) {
            if (taken[e]) continue;
            float v = k3_rank(sc[e], rbias[e]);
            if (v > bv) { bv = v; ba = e; }
        }
        sv[p] = bv; si[p] = ba;
        __syncthreads();
        for (int n = T / 2; n >= 1; n >>= 1) {
            if (p < n && sv[p + n] > sv[p]) { sv[p] = sv[p + n]; si[p] = si[p + n]; }
            __syncthreads();
        }
        if (!p) { if (si[0] >= 0) taken[si[0]] = 1; kth = sv[0]; }
        __syncthreads();
    }
    float thr = kth - delta;
    for (int e = p; e < E; e += T) {
        if (k3_rank(sc[e], rbias[e]) >= thr) {
            int i = atomicAdd(ncand, 1);
            if (i < cap) cand[i] = e;
        }
    }
}

/* Exact f32 re-score of the candidate rows. One block per candidate. */
__global__ void k3_router_exact(float *__restrict__ sc, const float *__restrict__ x,
                                const float *__restrict__ w,
                                const int *__restrict__ cand,
                                const int *__restrict__ ncand, int I, int cap) {
    int j = blockIdx.x;
    int n = *ncand; if (n > cap) n = cap;
    if (j >= n) return;
    int o = cand[j];
    const float *wr = w + (size_t)o * I;
    int p = threadIdx.x, T = blockDim.x;
    float a = 0.f;
    for (int i = p; i < I; i += T) a += x[i] * wr[i];
    __shared__ float sh[256];
    sh[p] = a; __syncthreads();
    for (int nn = T / 2; nn >= 1; nn >>= 1) {
        if (p < nn) sh[p] += sh[p + nn];
        __syncthreads();
    }
    if (!p) sc[o] = sh[0];
}



/* ---------------- prefill: one weight read, C tokens ------------------------
 * coli_k3_dense returned 0 for S != 1, so every prefill matmul fell back to the
 * generic quant_matmul -- 1370 us a launch against 91 us for the exact kernel,
 * and nsys duly showed it eating 64% of GPU time. Prefill therefore got none of
 * the mirroring, the 4-wide fold, or the exact int4 path, and measured 235
 * ms/token against decode's ~200: batching made it WORSE, not better.
 *
 * This is what batching is supposed to buy. The block stages its weight row in
 * shared memory ONCE (rb bytes + ng scales -- 4 KB at I=7168, 19 KB at 33792)
 * and then walks the C activation vectors against it, so DRAM weight traffic
 * drops C-fold while x stays L2-resident (32 tokens x 7168 x 4B = 918 KB).
 *
 * Per (row, token) the arithmetic and its order are exactly the S==1 kernel's,
 * so prefill becomes numerically consistent with decode -- which the generic
 * path was not. */
template<int W>
__global__ void k3_dense_i4g_exactW_S(float *__restrict__ y, const float *__restrict__ x,
                                      const unsigned char *__restrict__ q4,
                                      const float *__restrict__ scales,
                                      int I, int O, int gsh, int ng, int S) {
    enum { T = 256 / W };
    int o = blockIdx.x;
    if (o >= O) return;
    size_t rb = (size_t)((I + 1) >> 1);

    extern __shared__ unsigned char shw[];
    unsigned char *w  = shw;
    float         *scl = (float *)(shw + ((rb + 15) & ~(size_t)15));
    for (size_t i = threadIdx.x; i < rb; i += blockDim.x) w[i] = q4[(size_t)o * rb + i];
    for (int g = threadIdx.x; g < ng; g += blockDim.x) scl[g] = scales[(size_t)o * ng + g];
    __syncthreads();

    __shared__ float sh[W][T];
    int p = (int)threadIdx.x;
    for (int t = 0; t < S; t++) {
        const float *xs = x + (size_t)t * I;
        float a[W];
        #pragma unroll
        for (int j = 0; j < W; j++) a[j] = 0.f;
        for (int i = p * W; i + W - 1 < I; i += 256) {
            int g = i >> gsh;
            if (g >= ng) g = ng - 1;
            float sc = scl[g];
            #pragma unroll
            for (int q = 0; q < W / 4; q++) {
                float4 xx = *(const float4 *)(xs + i + q * 4);
                unsigned char b0 = w[(i >> 1) + q * 2], b1 = w[(i >> 1) + q * 2 + 1];
                a[q*4+0] += xx.x * (float)((int)(b0 & 15) - 8) * sc;
                a[q*4+1] += xx.y * (float)((int)(b0 >> 4)  - 8) * sc;
                a[q*4+2] += xx.z * (float)((int)(b1 & 15) - 8) * sc;
                a[q*4+3] += xx.w * (float)((int)(b1 >> 4)  - 8) * sc;
            }
        }
        #pragma unroll
        for (int j = 0; j < W; j++) sh[j][p] = a[j];
        __syncthreads();
        for (int n = T / 2; n >= 1; n >>= 1) {
            if (p < n) {
                #pragma unroll
                for (int j = 0; j < W; j++) sh[j][p] += sh[j][p + n];
            }
            __syncthreads();
        }
        if (!p) {
            float v[W];
            #pragma unroll
            for (int j = 0; j < W; j++) v[j] = sh[j][0];
            #pragma unroll
            for (int step = W / 2; step >= 1; step >>= 1) {
                #pragma unroll
                for (int j = 0; j < step; j++) v[j] += v[j + step];
            }
            y[(size_t)t * O + o] = v[0];
        }
        __syncthreads();
    }
}


/* ---------------- prefill GEMM, warp-per-row ------------------------------
 * The first prefill kernel was a decode GEMV wearing a loop: 64 threads (two
 * warps) per block, a 256-lane tree reduction and ~7 block-wide barriers PER
 * TOKEN, so a 32-token chunk paid ~256 barriers per block. It recovered only
 * part of the gap -- prefill sits at 7.4 tok/s against a ~97 tok/s weight
 * floor, so the shape is the cost, not the bytes.
 *
 * This is the shape the expert path already uses (warp_row_dot_w1): one WARP
 * owns an output row, reduces with __shfl_down_sync, and never touches a block
 * barrier. Eight warps per block stage eight weight rows in shared (32 KB at
 * I=7168) and walk all C tokens against them.
 *
 * NOT bit-exact against decode: a 32-lane shuffle reduction sums in a different
 * order than the 256-lane tree. That is the point of a separate prefill
 * implementation -- gate it on the quality harness, not the reference hash. */
#define K3_PF_WARPS 8
__global__ void k3_dense_i4g_pf(float *__restrict__ y, const float *__restrict__ x,
                                const unsigned char *__restrict__ q4,
                                const float *__restrict__ scales,
                                int I, int O, int gsh, int ng, int S) {
    int warp = (int)(threadIdx.x >> 5), lane = (int)(threadIdx.x & 31);
    int o = blockIdx.x * K3_PF_WARPS + warp;
    size_t rb = (size_t)((I + 1) >> 1);
    size_t rbA = (rb + 15) & ~(size_t)15;

    extern __shared__ unsigned char shp[];
    unsigned char *w   = shp + (size_t)warp * rbA;
    float         *scl = (float *)(shp + (size_t)K3_PF_WARPS * rbA) + (size_t)warp * ng;
    if (o < O) {
        for (size_t i = lane; i < rb; i += 32) w[i] = q4[(size_t)o * rb + i];
        for (int g = lane; g < ng; g += 32) scl[g] = scales[(size_t)o * ng + g];
    }
    __syncwarp();
    if (o >= O) return;

    for (int t = 0; t < S; t++) {
        const float *xs = x + (size_t)t * I;
        float a = 0.f;
        for (int i = lane * 4; i + 3 < I; i += 128) {
            int g = i >> gsh; if (g >= ng) g = ng - 1;
            float sc = scl[g];
            float4 xx = *(const float4 *)(xs + i);
            unsigned char b0 = w[(i >> 1)], b1 = w[(i >> 1) + 1];
            a += xx.x * (float)((int)(b0 & 15) - 8) * sc;
            a += xx.y * (float)((int)(b0 >> 4)  - 8) * sc;
            a += xx.z * (float)((int)(b1 & 15) - 8) * sc;
            a += xx.w * (float)((int)(b1 >> 4)  - 8) * sc;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (!lane) y[(size_t)t * O + o] = a;
    }
}

/* ---------------- prefill GEMM, 2D tiled ----------------------------------
 * Both earlier prefill kernels were row-parallel: every block re-read the WHOLE
 * activation block. At O=3072, I=7168, S=32 that is
 *   weights  12.4 MB   (each row staged once -- already minimal)
 *   x       384 blocks x 918 KB = 2.8 GB   (226x the weights)
 * x is only 918 KB so it lives in L2, but 2.8 GB of L2 reads at the measured
 * 2.69 ms/call is ~1.05 TB/s -- the L2 roofline. That is why neither halving
 * the barriers (warp shape, +2%) nor cutting DRAM weight traffic moved it:
 * prefill was never DRAM-bound, it was bound on re-reading activations.
 *
 * So tile both operands. A block owns 32 output rows x 32 tokens and walks the
 * reduction in 256-element steps, staging x[32][256] and w[32][256/8] in
 * shared. x is now read once per ROW-TILE rather than once per ROW:
 *   (O/32) x S x I x 4 = 88 MB, a 32x cut, and weights stay at 12.4 MB.
 * Each staged weight nibble feeds 4 FMAs (one per token the warp owns), which
 * is also what lifts arithmetic intensity off the shared-memory port.
 *
 * Lane owns a row and warp owns 4 tokens, so ws[] is indexed by lane (padded
 * +1 uint32 to spread banks) and xs[] is warp-uniform (a broadcast, free).
 *
 * NOT bit-exact against decode -- the reduction runs in 256-element tiles in
 * thread-local order rather than a 256-lane tree. Gate on the quality harness. */
#define K3_TG_O 32
#define K3_TG_S 32
#define K3_TG_I 256
__global__ __launch_bounds__(256) void
k3_dense_i4g_tg(float *__restrict__ y, const float *__restrict__ x,
                const unsigned char *__restrict__ q4, const float *__restrict__ scales,
                int I, int O, int ng, int S) {
    int lane = (int)(threadIdx.x & 31), warp = (int)(threadIdx.x >> 5);
    int o0 = blockIdx.x * K3_TG_O, t0 = blockIdx.y * K3_TG_S;
    int o  = o0 + lane;
    size_t rb = (size_t)(I >> 1);

    __shared__ float        xs[K3_TG_S][K3_TG_I];
    __shared__ unsigned int ws[K3_TG_O][K3_TG_I / 8 + 1];   /* +1: bank spread   */
    __shared__ float        ss[K3_TG_O][K3_TG_I / 64 + 1];  /* +1: 5 is coprime  */

    float a[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) a[j] = 0.f;

    for (int i0 = 0; i0 < I; i0 += K3_TG_I) {
        for (int p = (int)threadIdx.x; p < K3_TG_S * (K3_TG_I / 4); p += 256) {
            int t = p / (K3_TG_I / 4), c = (p % (K3_TG_I / 4)) * 4, tg = t0 + t;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (tg < S) v = *(const float4 *)(x + (size_t)tg * I + i0 + c);
            *(float4 *)&xs[t][c] = v;
        }
        for (int p = (int)threadIdx.x; p < K3_TG_O * (K3_TG_I / 8); p += 256) {
            int r = p / (K3_TG_I / 8), c = p % (K3_TG_I / 8);
            ws[r][c] = (o0 + r < O)
                     ? *(const unsigned int *)(q4 + (size_t)(o0 + r) * rb + (i0 >> 1) + c * 4)
                     : 0u;
        }
        for (int p = (int)threadIdx.x; p < K3_TG_O * (K3_TG_I / 64); p += 256) {
            int r = p / (K3_TG_I / 64), c = p % (K3_TG_I / 64);
            int g = (i0 >> 6) + c; if (g >= ng) g = ng - 1;
            ss[r][c] = (o0 + r < O) ? scales[(size_t)(o0 + r) * ng + g] : 0.f;
        }
        __syncthreads();

        #pragma unroll 4
        for (int c = 0; c < K3_TG_I / 8; c++) {
            unsigned int wv = ws[lane][c];
            float        sc = ss[lane][c >> 3];
            int          k  = c * 8;
            #pragma unroll
            for (int n = 0; n < 8; n++) {
                float wq = (float)((int)((wv >> (n * 4)) & 15u) - 8) * sc;
                #pragma unroll
                for (int j = 0; j < 4; j++) a[j] += xs[warp * 4 + j][k + n] * wq;
            }
        }
        __syncthreads();
    }
    if (o >= O) return;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int t = t0 + warp * 4 + j;
        if (t < S) y[(size_t)t * O + o] = a[j];
    }
}

/* ---------------- fp16 dense GEMV (fmt 2) ---------------------------------
 * For the router. int8 per-row quantization ties the absolute error to the row
 * maximum -- about 2% of a typical weight when the row spans 4 sigma -- and
 * that was enough to move expert selections and score PCC 0.985. fp16 keeps
 * RELATIVE precision everywhere, ~0.05%, roughly 40x tighter, while still
 * halving the bytes against f32.
 *
 * Same block-per-row, 4-wide fold, 256-lane tree as the int4 kernel, so the
 * summation order matches the rest of the dense path. */
__global__ void k3_dense_f16(float *__restrict__ y, const float *__restrict__ x,
                             const __half *__restrict__ w, int I, int O) {
    enum { W = 4, T = 256 / W };
    int o = blockIdx.x;
    if (o >= O) return;
    const __half *wr = w + (size_t)o * I;
    int p = (int)threadIdx.x;
    float a[W];
    #pragma unroll
    for (int j = 0; j < W; j++) a[j] = 0.f;
    for (int i = p * W; i + W - 1 < I; i += 256) {
        float4 xx = *(const float4 *)(x + i);
        a[0] += xx.x * __half2float(wr[i + 0]);
        a[1] += xx.y * __half2float(wr[i + 1]);
        a[2] += xx.z * __half2float(wr[i + 2]);
        a[3] += xx.w * __half2float(wr[i + 3]);
    }
    __shared__ float sh[W][T];
    #pragma unroll
    for (int j = 0; j < W; j++) sh[j][p] = a[j];
    __syncthreads();
    for (int n = T / 2; n >= 1; n >>= 1) {
        if (p < n) {
            #pragma unroll
            for (int j = 0; j < W; j++) sh[j][p] += sh[j][p + n];
        }
        __syncthreads();
    }
    if (!p) {
        float v0 = sh[0][0], v1 = sh[1][0], v2 = sh[2][0], v3 = sh[3][0];
        v0 += v2; v1 += v3;
        y[o] = v0 + v1;
    }
}

template<int W, int R>
__global__ void k3_dense_i4g_exactR(float *__restrict__ y, const float *__restrict__ x,
                                    const unsigned char *__restrict__ q4,
                                    const float *__restrict__ scales,
                                    int I, int O, int gsh, int ng) {
    enum { T = 256 / W };
    int sub = (int)threadIdx.x / T;
    int o = blockIdx.x * R + sub;
    int p = (int)threadIdx.x % T;
    __shared__ float sh[R][W][T];
    if (o < O) {
        size_t rb = (size_t)((I + 1) >> 1);
        const unsigned char *w = q4 + (size_t)o * rb;
        const float *scl = scales + (size_t)o * ng;
        float a[W];
        #pragma unroll
        for (int j = 0; j < W; j++) a[j] = 0.f;
        for (int i = p * W; i + W - 1 < I; i += 256) {
            int g = i >> gsh;
            if (g >= ng) g = ng - 1;
            float sc = scl[g];
            #pragma unroll
            for (int q = 0; q < W / 4; q++) {
                float4 xx = *(const float4 *)(x + i + q * 4);
                unsigned char b0 = w[(i >> 1) + q * 2], b1 = w[(i >> 1) + q * 2 + 1];
                a[q*4+0] += xx.x * (float)((int)(b0 & 15) - 8) * sc;
                a[q*4+1] += xx.y * (float)((int)(b0 >> 4)  - 8) * sc;
                a[q*4+2] += xx.z * (float)((int)(b1 & 15) - 8) * sc;
                a[q*4+3] += xx.w * (float)((int)(b1 >> 4)  - 8) * sc;
            }
        }
        #pragma unroll
        for (int j = 0; j < W; j++) sh[sub][j][p] = a[j];
    }
    __syncthreads();
    for (int n = T / 2; n >= 1; n >>= 1) {
        if (o < O && p < n) {
            #pragma unroll
            for (int j = 0; j < W; j++) sh[sub][j][p] += sh[sub][j][p+n];
        }
        __syncthreads();
    }
    if (o < O && !p) {
        float v[W];
        #pragma unroll
        for (int j = 0; j < W; j++) v[j] = sh[sub][j][0];
        #pragma unroll
        for (int step = W / 2; step >= 1; step >>= 1) {
            #pragma unroll
            for (int j = 0; j < step; j++) v[j] += v[j + step];
        }
        y[o] = v[0];
    }
}

/* ---------------- exactW, ILP variant --------------------------------------
 * ncu on the shipped exactW<4>, all tensors mirrored, says nothing is
 * saturated:
 *     Memory 21.85%   Compute 21.85%   L1/TEX 26.68%   L2 14.99%
 *     achieved occupancy 115%, 55.3 active warps/SM
 * High occupancy with every pipe near a fifth of peak is the signature of a
 * LATENCY bound, not a bandwidth or compute bound. So the fix is more memory
 * work in flight per thread, not fewer bytes.
 *
 * Two changes, both leaving the arithmetic and its order untouched:
 *
 *   1. The weight pair was two separate 1-byte loads at consecutive addresses.
 *      i = 4p + 256k makes i>>1 even, so the pair is 2-byte aligned and can be
 *      one ushort load -- half the weight load instructions, same bytes.
 *   2. Unroll the reduction loop by 2. Iterations i and i+256 are independent
 *      up to the accumulator update, so this puts two float4 loads and two
 *      weight loads in flight instead of one, while each a[j] still accumulates
 *      strictly in increasing i. Bit-exact by construction.
 *
 * K3_DENSE_ILP=0 selects the original. */
/* The per-row computation, shared by the single and batched launches so the
 * two cannot drift apart numerically. Returns the row's value on thread 0. */
template<int W>
__device__ __forceinline__ float k3_exact_row_ilp(const float *__restrict__ x,
                                                  const unsigned char *__restrict__ w,
                                                  const float *__restrict__ scl,
                                                  int I, int gsh, int ng) {
    enum { T = 256 / W };
    int p = (int)threadIdx.x;
    float a[W];
    #pragma unroll
    for (int j = 0; j < W; j++) a[j] = 0.f;

    int i = p * W;
    for (; i + W - 1 + 256 < I; i += 512) {          /* two strides per trip */
        int g0 = i >> gsh;             if (g0 >= ng) g0 = ng - 1;
        int g1 = (i + 256) >> gsh;     if (g1 >= ng) g1 = ng - 1;
        float sc0 = scl[g0], sc1 = scl[g1];
        #pragma unroll
        for (int q = 0; q < W / 4; q++) {
            float4 x0 = *(const float4 *)(x + i + q * 4);
            float4 x1 = *(const float4 *)(x + i + 256 + q * 4);
            unsigned short p0 = *(const unsigned short *)(w + (i >> 1) + q * 2);
            unsigned short p1 = *(const unsigned short *)(w + ((i + 256) >> 1) + q * 2);
            unsigned int b0 = p0 & 0xffu, b1 = p0 >> 8, c0 = p1 & 0xffu, c1 = p1 >> 8;
            a[q*4+0] += x0.x * (float)((int)(b0 & 15) - 8) * sc0;
            a[q*4+1] += x0.y * (float)((int)(b0 >> 4)  - 8) * sc0;
            a[q*4+2] += x0.z * (float)((int)(b1 & 15) - 8) * sc0;
            a[q*4+3] += x0.w * (float)((int)(b1 >> 4)  - 8) * sc0;
            a[q*4+0] += x1.x * (float)((int)(c0 & 15) - 8) * sc1;
            a[q*4+1] += x1.y * (float)((int)(c0 >> 4)  - 8) * sc1;
            a[q*4+2] += x1.z * (float)((int)(c1 & 15) - 8) * sc1;
            a[q*4+3] += x1.w * (float)((int)(c1 >> 4)  - 8) * sc1;
        }
    }
    for (; i + W - 1 < I; i += 256) {                /* odd trip, if any */
        int g = i >> gsh; if (g >= ng) g = ng - 1;
        float sc = scl[g];
        #pragma unroll
        for (int q = 0; q < W / 4; q++) {
            float4 xx = *(const float4 *)(x + i + q * 4);
            unsigned short pp = *(const unsigned short *)(w + (i >> 1) + q * 2);
            unsigned int b0 = pp & 0xffu, b1 = pp >> 8;
            a[q*4+0] += xx.x * (float)((int)(b0 & 15) - 8) * sc;
            a[q*4+1] += xx.y * (float)((int)(b0 >> 4)  - 8) * sc;
            a[q*4+2] += xx.z * (float)((int)(b1 & 15) - 8) * sc;
            a[q*4+3] += xx.w * (float)((int)(b1 >> 4)  - 8) * sc;
        }
    }

    __shared__ float sh[W][T];
    #pragma unroll
    for (int j = 0; j < W; j++) sh[j][p] = a[j];
    __syncthreads();
    for (int n = T / 2; n >= 1; n >>= 1) {
        if (p < n) {
            #pragma unroll
            for (int j = 0; j < W; j++) sh[j][p] += sh[j][p+n];
        }
        __syncthreads();
    }
    float out = 0.f;
    if (!p) {
        float v[W];
        #pragma unroll
        for (int j = 0; j < W; j++) v[j] = sh[j][0];
        #pragma unroll
        for (int step = W / 2; step >= 1; step >>= 1) {
            #pragma unroll
            for (int j = 0; j < step; j++) v[j] += v[j + step];
        }
        out = v[0];
    }
    __syncthreads();                     /* sh must not be reused before all lanes read */
    return out;
}

template<int W>
__global__ void k3_dense_i4g_exactW_ilp(float *__restrict__ y, const float *__restrict__ x,
                                        const unsigned char *__restrict__ q4,
                                        const float *__restrict__ scales,
                                        int I, int O, int gsh, int ng) {
    int o = blockIdx.x;
    if (o >= O) return;
    size_t rb = (size_t)((I + 1) >> 1);
    float v = k3_exact_row_ilp<W>(x, q4 + (size_t)o * rb, scales + (size_t)o * ng, I, gsh, ng);
    if (!threadIdx.x) y[o] = v;
}

/* ---------------- batched dense: N tensors, one x, one sync -----------------
 * Every coli_k3_dense call is a full round trip: upload x, launch, download y,
 * cudaStreamSynchronize. The four KDA projections (q, k, v, g) multiply the
 * SAME x, so per layer that is four uploads of the same 28.7 KB and four
 * synchronisations.
 *
 * The arithmetic says how much that costs. Per rank the projections are
 * 4 x [3072, 7168] at 4032 B/row = 49.5 MB/layer, 3.42 GB/token over 69 KDA
 * layers. kproj measures 3.06 s/100 tokens = 30.6 ms/token, i.e. 112 GB/s,
 * against 234.5 GB/s for the same kernel on mirrored weights. At 235 GB/s the
 * bytes alone want ~53 us per call and the measurement says ~111 us, so more
 * than half of kproj is per-call overhead rather than the kernel.
 *
 * So upload x once, select the weight matrix with blockIdx.y, and synchronise
 * once. The downloads stay separate (the caller has four distinct
 * destinations) but they are async and cost only their enqueue.
 *
 * MEASURED WORSE, off by default (K3_DENSE_MULTI=1 to re-run the A/B):
 *
 *     e2e        5.182 -> 4.795 tok/s   -7.5%
 *     kproj      3.211 -> 4.437         +38%
 *     ctlwork    2.705 -> 3.984         +47%
 *
 * The launch count was never the problem. Four launches of 3072 blocks and one
 * of 12288 carry the same parallelism, but the four run as short kernels that
 * each stream one 12.4 MB tensor, while the batch interleaves four weight
 * streams through L2 at once. The tell is ctlwork: the CPU control thread that
 * runs concurrently got 47% SLOWER without doing anything different, so the
 * longer single kernel is holding the memory system against it. On a shared
 * LPDDR5X the GPU and CPU are contending, and a kernel that runs longer in one
 * stretch is worse for the pair than several that leave gaps -- even though the
 * per-call round trip really is more than half of kproj by the byte arithmetic.
 *
 * Eighth instance of the same pattern: removing work wins, moving it loses. */
struct K3DenseBatch { const unsigned char *q4[8]; const float *sc[8]; };

template<int W>
__global__ void k3_dense_i4g_exactW_multi(float *__restrict__ y, const float *__restrict__ x,
                                          K3DenseBatch b, int I, int O, int gsh, int ng) {
    int o = blockIdx.x, j = blockIdx.y;
    if (o >= O) return;
    size_t rb = (size_t)((I + 1) >> 1);
    float v = k3_exact_row_ilp<W>(x, b.q4[j] + (size_t)o * rb, b.sc[j] + (size_t)o * ng,
                                  I, gsh, ng);
    if (!threadIdx.x) y[(size_t)j * O + o] = v;
}

template<int W>
__global__ void k3_dense_i4g_exactW(float *__restrict__ y, const float *__restrict__ x,
                                    const unsigned char *__restrict__ q4,
                                    const float *__restrict__ scales,
                                    int I, int O, int gsh, int ng) {
    enum { T = 256 / W };
    int o = blockIdx.x;
    if (o >= O) return;
    size_t rb = (size_t)((I + 1) >> 1);
    const unsigned char *w = q4 + (size_t)o * rb;
    const float *scl = scales + (size_t)o * ng;
    int p = (int)threadIdx.x;
    float a[W];
    #pragma unroll
    for (int j = 0; j < W; j++) a[j] = 0.f;
    for (int i = p * W; i + W - 1 < I; i += 256) {
        int g = i >> gsh;
        if (g >= ng) g = ng - 1;
        float sc = scl[g];
        #pragma unroll
        for (int q = 0; q < W / 4; q++) {
            float4 xx = *(const float4 *)(x + i + q * 4);
            unsigned char b0 = w[(i >> 1) + q * 2], b1 = w[(i >> 1) + q * 2 + 1];
            a[q*4+0] += xx.x * (float)((int)(b0 & 15) - 8) * sc;
            a[q*4+1] += xx.y * (float)((int)(b0 >> 4)  - 8) * sc;
            a[q*4+2] += xx.z * (float)((int)(b1 & 15) - 8) * sc;
            a[q*4+3] += xx.w * (float)((int)(b1 >> 4)  - 8) * sc;
        }
    }
    __shared__ float sh[W][T];
    #pragma unroll
    for (int j = 0; j < W; j++) sh[j][p] = a[j];
    __syncthreads();
    for (int n = T / 2; n >= 1; n >>= 1) {            /* lane bits 7..log2(W) */
        if (p < n) {
            #pragma unroll
            for (int j = 0; j < W; j++) sh[j][p] += sh[j][p+n];
        }
        __syncthreads();
    }
    if (!p) {
        float v[W];
        #pragma unroll
        for (int j = 0; j < W; j++) v[j] = sh[j][0];
        #pragma unroll
        for (int step = W / 2; step >= 1; step >>= 1) {
            #pragma unroll
            for (int j = 0; j < step; j++) v[j] += v[j + step];
        }
        y[o] = v[0];
    }
}

__global__ void k3_dense_i8_exact(float *__restrict__ y, const float *__restrict__ x,
                                  const signed char *__restrict__ q8,
                                  const float *__restrict__ scales, int I, int O) {
    int o = blockIdx.x;
    if (o >= O) return;
    const signed char *w = q8 + (size_t)o * I;
    /* Same 256-lane fold the int4 path uses: s0 and s1 ARE original partial[t]
     * and partial[t+128], so s0+s1 is the tree's first edge and not one
     * floating-point association moves. Measured on the real shapes this is
     * worth 143.7 -> 166.7 GB/s on lm_head and 129 -> 152 on MLA q_b.
     *
     * Folds of 4 and 8 (64 and 32 threads), tiling 2 and 4 rows per CTA to
     * amortise the x re-read, and pairing lanes {2p, 2p+1} so one byte load
     * serves both nibbles were all built and measured exact; every one of them
     * was neutral or slower here, because at 64-71% of this part's 235 GB/s
     * achievable read bandwidth these GEMVs are bandwidth-bound, not
     * issue-bound. Do not re-try them without a different memory layout. */
    float s0 = 0.f, s1 = 0.f;
    for (int i = threadIdx.x; i < I; i += 256)
        s0 += x[i] * (float)w[i];
    for (int i = threadIdx.x + 128; i < I; i += 256)
        s1 += x[i] * (float)w[i];
    __shared__ float partial[128];
    partial[threadIdx.x] = s0 + s1;
    __syncthreads();
    for (int n = 64; n >= 32; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (threadIdx.x < 32) {
        float v=partial[threadIdx.x];
        #pragma unroll
        for (int n=16;n;n>>=1) v+=__shfl_down_sync(0xffffffffu,v,n);
        if (!threadIdx.x) y[o]=v*scales[o];
    }
}

/* Same 4-wide fold for int8. Four consecutive elements means one float4 of
 * activations and FOUR ADJACENT weight bytes -- a single uint when the row base
 * happens to be 4-byte aligned, which is two loads per four elements against
 * the eight the {t, t+128} fold needs. The alignment test is per-block uniform
 * (w depends only on blockIdx), so the branch is free. */
__global__ void k3_dense_i8_exact4(float *__restrict__ y, const float *__restrict__ x,
                                   const signed char *__restrict__ q8,
                                   const float *__restrict__ scales, int I, int O) {
    int o = blockIdx.x;
    if (o >= O) return;
    const signed char *w = q8 + (size_t)o * I;
    int p = (int)threadIdx.x;
    int aligned = (((uintptr_t)w & 3u) == 0);
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int i = p * 4; i + 3 < I; i += 256) {
        float4 xx = *(const float4 *)(x + i);
        int w0, w1, w2, w3;
        if (aligned) {
            unsigned int u = *(const unsigned int *)(w + i);
            w0 = (int)(signed char)(u & 0xFF);         w1 = (int)(signed char)((u >> 8)  & 0xFF);
            w2 = (int)(signed char)((u >> 16) & 0xFF); w3 = (int)(signed char)((u >> 24) & 0xFF);
        } else {
            w0 = w[i]; w1 = w[i+1]; w2 = w[i+2]; w3 = w[i+3];
        }
        a0 += xx.x * (float)w0; a1 += xx.y * (float)w1;
        a2 += xx.z * (float)w2; a3 += xx.w * (float)w3;
    }
    __shared__ float sh[4][64];
    sh[0][p] = a0; sh[1][p] = a1; sh[2][p] = a2; sh[3][p] = a3;
    __syncthreads();
    for (int n = 32; n >= 1; n >>= 1) {
        if (p < n) {
            sh[0][p] += sh[0][p+n]; sh[1][p] += sh[1][p+n];
            sh[2][p] += sh[2][p+n]; sh[3][p] += sh[3][p+n];
        }
        __syncthreads();
    }
    if (!p) y[o] = ((sh[0][0] + sh[2][0]) + (sh[1][0] + sh[3][0])) * scales[o];
}

static int    g_dense_attr = 0;
static int    g_dense_shmax = 0;
static float *g_dx = nullptr, *g_dy = nullptr;
static int    g_dx_cap = 0, g_dy_cap = 0;

/* x and y are ordinary host buffers the kernel cannot address, so stage them
 * through device scratch like the expert path does. Both are small next to the
 * weights (28 KB of x, <=640 KB of y) -- it is the WEIGHTS that must not be
 * copied, and those stay zero-copy. */
static int ensure_dense_scratch(int I, int O) {
    if (I > g_dx_cap) {
        if (g_dx) cudaFree(g_dx);
        if (!ck(cudaMalloc(&g_dx, (size_t)I * sizeof(float)), "dense x scratch")) { g_dx_cap = 0; return 0; }
        g_dx_cap = I;
    }
    if (O > g_dy_cap) {
        if (g_dy) cudaFree(g_dy);
        if (!ck(cudaMalloc(&g_dy, (size_t)O * sizeof(float)), "dense y scratch")) { g_dy_cap = 0; return 0; }
        g_dy_cap = O;
    }
    return 1;
}

/* ---- device mirrors for the exact dense path ------------------------------
 * The dense weights are read zero-copy so their bytes stay available to the
 * expert cache, and on this integrated part that was assumed free. It is not:
 * host-registered pages reach the GPU through the SMMU at 4 KB granularity,
 * while cudaMalloc'd memory uses large pages. Same DRAM, measurably different
 * rate on the SAME kernel (K3_DENSE_EXACT=1, rel=0.0e+00 either way):
 *
 *   shared gate 6144x7168 int4   153.5 -> 214.9 GB/s
 *   MLA q_b     18432x1536 int8  145.1 -> 239.2
 *   KDA q/k/v/g 12288x7168 int4  150.9 -> 180.6
 *   lm_head    163840x7168 int8  166.4 -> 153.1   (mirroring HURTS: 1.17 GB,
 *                                                  far past any cache)
 *
 * So mirror selectively, under a byte budget, and never for tensors bigger than
 * the point where it stops paying. Mirroring costs RAM the expert cache would
 * otherwise hold, which is why this is a budget and not a default-everything. */
static size_t g_devmir_left = 0, g_devmir_used = 0;
static size_t g_devmir_cap_skip = 0, g_devmir_budget_skip = 0;
static int    g_devmir_init = 0;

/* ---- live-settable knobs -------------------------------------------------
 * These were function-local `static int x = -1; if (x<0) x = getenv(...)`
 * latches, which is why the SET command reported live=0 for them. An A/B arm
 * therefore needed a process restart, and a restart at ctx 1963 costs 146 s of
 * load plus ~960 s refilling the expert cache -- 81% of an arm's wall clock
 * spent on warmup that is then discarded. Hoisted to file scope so one loaded
 * engine can serve every arm of a knob sweep. (K3_EXPERT_GB still cannot: the
 * cache is sized at init.) */
static int g_k_ilp = -1, g_k_multi = -1, g_k_pfshape = -1, g_k_i4w = -1, g_k_rthr = -1;
static int k3_knob(int *slot, const char *env, int dflt) {
    if (*slot < 0) { const char *e = getenv(env); *slot = e ? atoi(e) : dflt; }
    return *slot;
}
extern "C" int coli_k3_set_knob(const char *k, int v) {
    if      (!strcmp(k, "K3_DENSE_ILP"))     g_k_ilp     = v;
    else if (!strcmp(k, "K3_DENSE_MULTI"))   g_k_multi   = v;
    else if (!strcmp(k, "K3_PF_SHAPE"))      g_k_pfshape = v;
    else if (!strcmp(k, "K3_DENSE_I4W"))     g_k_i4w     = v;
    else if (!strcmp(k, "K3_DENSE_RTHRESH")) g_k_rthr    = v;
    else return 0;
    return 1;
}

extern "C" size_t coli_k3_devmirror_used(void) { return g_devmir_used; }
/* Which limit actually bound: the byte budget, or the per-tensor cap. */
extern "C" void coli_k3_devmirror_report(void) {
    fprintf(stderr, "[K3/EXP] dense mirrors: %.2f GB placed, %.2f GB skipped for budget, "
                    "%.2f GB skipped over the 2 GB cap\n",
            g_devmir_used/1e9, g_devmir_budget_skip/1e9, g_devmir_cap_skip/1e9);
}

/* Returns a device copy, or null when it declines (budget spent, too large,
 * or allocation failed). Never fails the caller: they keep the host pointer. */
extern "C" void *coli_k3_devmirror(const void *host, size_t bytes) {
    if (!g_ready || !host || !bytes) return nullptr;
    if (!g_devmir_init) {
        /* Defaulting this to zero disabled mirroring entirely, which measured
         * 3.86 tok/s against 4.28 with a budget set -- an 11% loss that stayed
         * invisible because the speed harness never forwarded the variable.
         * Auto-size instead: a fifth of what is free, capped, so the expert
         * cache keeps the rest. Only ~16 GB is eligible on this sharding, so
         * the cap is never reached and nothing is skipped for budget. */
        const char *e = getenv("K3_DENSE_DEV_GB");
        double gb;
        if (e) gb = atof(e);
        else {
            size_t fb = 0, tb = 0;
            /* A fifth of free was too tight: it placed 11.45 GB, skipped 4.16
             * for budget, and reached only 4.21 tok/s against 4.28 with the
             * whole eligible set mirrored. The eligible set is ~16 GB on this
             * sharding, so take a third and cap at 24. */
            gb = (cudaMemGetInfo(&fb, &tb) == cudaSuccess) ? (double)fb * 0.35 / 1e9 : 0.0;
            if (gb > 24.0) gb = 24.0;
        }
        g_devmir_left = (size_t)(gb * 1e9);
        g_devmir_init = 1;
        if (g_devmir_left)
            fprintf(stderr, "[K3/EXP] dense device-mirror budget %.1f GB\n", gb);
    }
    /* The cap used to be 64 MB because lm_head measured SLOWER mirrored. That
     * was an artefact of the 2-wide fold: with the 4-wide kernel the same
     * tensor goes 166.7 GB/s zero-copy -> 234.5 mirrored, so the large-page
     * mapping pays even for a 1.17 GB stream that cannot stay cache-resident.
     * The cap now only guards against a single tensor eating the whole budget. */
    if (bytes > (size_t)2048 * 1024 * 1024) { g_devmir_cap_skip += bytes; return nullptr; }
    if (bytes > g_devmir_left) { g_devmir_budget_skip += bytes; return nullptr; }
    void *d = nullptr;
    /* The mirror exists for large pages, not for device locality -- this GPU is
     * integrated, so cudaMalloc does not move the bytes anywhere, it just maps
     * them at a coarser granularity than the SMMU gives host-registered pages.
     * That means the tensor is now resident TWICE in the same LPDDR5X, which
     * the startup banner says outright, and 16.06 GB is held twice.
     *
     * cudaMallocManaged would be dereferenceable from both sides, so the host
     * copy could be freed and every pointer repointed at the managed one -- no
     * duplicate and, unlike free()ing the host copy outright, no way for a CPU
     * fallback to reach a dangling pointer.
     *
     * MEASURED WORSE ON BOTH COUNTS, so that route is closed:
     *
     *     e2e      5.206 -> 4.796 tok/s   -7.9%
     *     shared   3.509 -> 4.440
     *     kout     2.949 -> 3.212
     *     RSS      70.94 -> 86.21 GB
     *
     * Managed pages do not keep whatever mapping cudaMalloc gets here -- the
     * dense timers all regress. And RSS went UP 15.3 GB rather than down,
     * which says the duplication was never visible in RSS to begin with:
     * cudaMalloc'd device pages on this integrated part are physically
     * resident but not charged to the process, while managed pages are. So
     * "free the host copy and save 16 GB" cannot be validated by watching RSS,
     * and this particular way of doing it costs 8% besides.
     *
     * K3_DENSE_MANAGED=1 to re-run the A/B. */
    static int mgd = -1;
    if (mgd < 0) { const char *e = getenv("K3_DENSE_MANAGED"); mgd = e ? atoi(e) : 0; }
    /* No cudaMemAdvise/cudaMemPrefetchAsync: CUDA 13 changed both to take a
     * cudaMemLocation struct, and the cudaMemcpy below populates the pages
     * anyway. Keeping the call portable matters more than the hint. */
    if (mgd) {
        if (cudaMallocManaged(&d, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    } else if (cudaMalloc(&d, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    if (cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError(); cudaFree(d); return nullptr;
    }
    g_devmir_left -= bytes; g_devmir_used += bytes;
    return d;
}

/* Batched S==1 dense: N tensors sharing one x, one upload, one synchronise.
 * Returns 0 when it cannot help so the caller keeps its per-tensor loop. */
extern "C" int coli_k3_dense_multi(float *const *ys, const float *x,
                                   const void *const *ws, const float *const *scales,
                                   int n, int fmt, int I, int O, int gs) {
    if (!k3_knob(&g_k_multi, "K3_DENSE_MULTI", 0) || !g_ready || n < 2 || n > 8) return 0;
    if (fmt != 4 || I <= 0 || O <= 0) return 0;
    if (gs < 4 || (gs & (gs - 1)) || (I & (gs - 1)) || (I & 3)) return 0;
    static int i4w = -1;
    if (i4w < 0) { const char *e = getenv("K3_DENSE_I4W"); i4w = e ? atoi(e) : 4; }
    if (i4w != 4) return 0;
    { const char *fe = getenv("K3_DENSE_EXACT"); if (fe && !atoi(fe)) return 0; }
    if (!k3_knob(&g_k_ilp, "K3_DENSE_ILP", 1)) return 0;   /* shares the ILP row helper */
    int gsh = 0; while ((1 << gsh) < gs) gsh++;
    int ng = (I + gs - 1) / gs;
    if (!ensure_dense_scratch(I, O * n)) return 0;
    if (!ck(cudaMemcpyAsync(g_dx, x, (size_t)I * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "multi x")) return 0;
    K3DenseBatch b;
    for (int j = 0; j < n; j++) { b.q4[j] = (const unsigned char *)ws[j]; b.sc[j] = scales[j]; }
    for (int j = n; j < 8; j++) { b.q4[j] = b.q4[0]; b.sc[j] = b.sc[0]; }
    dim3 gr((unsigned)O, (unsigned)n);
    k3_dense_i4g_exactW_multi<4><<<gr, 64, 0, g_stream>>>(g_dy, g_dx, b, I, O, gsh, ng);
    if (!ck(cudaGetLastError(), "multi launch")) return 0;
    for (int j = 0; j < n; j++)
        if (!ck(cudaMemcpyAsync(ys[j], g_dy + (size_t)j * O, (size_t)O * sizeof(float),
                                cudaMemcpyDeviceToHost, g_stream), "multi y")) return 0;
    return ck(cudaStreamSynchronize(g_stream), "multi sync");
}

/* S==1 only (decode GEMV). Returns 0 when it cannot help, so the caller keeps
 * its existing path for prefill and for shapes whose x will not fit shared. */
extern "C" int coli_k3_dense(float *y, const float *x, const void *w, const float *scales,
                             int fmt, int S, int I, int O, int gs) {
    static int en = -1;
    if (en < 0) { const char *e = getenv("K3_DENSE_GPU"); en = e ? atoi(e) : 1; }
    if (!en && !getenv("K3_DENSE_BENCH")) return 0;
    if (!g_ready || S != 1 || I <= 0 || O <= 0) return 0;
    /* Only stage when x is small enough that shared memory does not throttle
     * occupancy. 16 KB keeps >=4 blocks/SM; beyond that, global + L1 wins. */
    int stage = ((size_t)I * sizeof(float) <= 16u * 1024u);
    { const char *fs = getenv("K3_DENSE_STAGE"); if (fs) stage = atoi(fs); }
    if (stage && (size_t)I * sizeof(float) > 96u * 1024u) stage = 0;
    size_t shbytes = stage ? (size_t)I * sizeof(float) : 0;
    if (fmt != 1 && fmt != 2 && fmt != 4) return 0;   /* 2 = fp16, no scales */
    int gsh = 0;
    if (fmt == 4) {
        if (gs <= 0 || (gs & (gs - 1))) return 0;        /* need a power of two */
        while ((1 << gsh) < gs) gsh++;
        if (I & (gs - 1)) return 0;                      /* whole groups only */
    }
    if (!g_dense_attr) {
        /* Query the opt-in limit rather than assuming: sm_121 caps dynamic
         * shared near 99 KB, and asking for 100 KB fails, leaving a STICKY
         * CUDA error that then breaks every later launch -- including
         * coli_cuda_matmul's, which made the stock path look like it was
         * declining when it was simply poisoned. */
        int mx = 0;
        cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (mx > 0) {
            cudaFuncSetAttribute(k3_dense_i4g, cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
            cudaFuncSetAttribute(k3_dense_i8,  cudaFuncAttributeMaxDynamicSharedMemorySize, mx);
        }
        g_dense_shmax = mx;
        cudaGetLastError();                               /* clear, do not inherit */
        g_dense_attr = 1;
    }
    if (stage && shbytes > (size_t)g_dense_shmax) { stage = 0; shbytes = 0; }
    if (!ensure_dense_scratch(I, O)) return 0;
    if (!ck(cudaMemcpyAsync(g_dx, x, (size_t)I * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "dense x upload")) return 0;
    if (fmt == 2) {                       /* fp16 weights, no scales */
        if (I & 3) return 0;
        k3_dense_f16<<<O, 64, 0, g_stream>>>(g_dy, g_dx, (const __half *)w, I, O);
        if (!ck(cudaGetLastError(), "f16 launch")) return 0;
        if (!ck(cudaMemcpyAsync(y, g_dy, (size_t)O * sizeof(float),
                                cudaMemcpyDeviceToHost, g_stream), "f16 download")) return 0;
        return ck(cudaStreamSynchronize(g_stream), "f16 sync");
    }
    int exact = 1;
    { const char *fe = getenv("K3_DENSE_EXACT"); if (fe) exact = atoi(fe); }
    int blocks = (O + K3_WARPS - 1) / K3_WARPS;
    static int i4w = -1;
    if (i4w < 0) { const char *e = getenv("K3_DENSE_I4W"); i4w = e ? atoi(e) : 4; }
    if (exact && fmt == 4) {
        int ng = (I + gs - 1) / gs;
        /* Short rows underfill one-row-per-block; four rows gives 256 threads
         * and the same bit-exact per-row tree. K3_DENSE_RTHRESH=0 disables. */
        static int rthr = -1;
        if (rthr < 0) { const char *e = getenv("K3_DENSE_RTHRESH"); rthr = e ? atoi(e) : 4096; }
        if (i4w == 4 && gs >= 4 && !(I & 3) && I <= rthr)
            k3_dense_i4g_exactR<4,4><<<(O + 3) / 4, 256, 0, g_stream>>>(
                g_dy, g_dx, (const unsigned char *)w, scales, I, O, gsh, ng);
        else if (i4w == 4 && gs >= 4 && !(I & 3)) {
            if (k3_knob(&g_k_ilp, "K3_DENSE_ILP", 1))
                k3_dense_i4g_exactW_ilp<4><<<O, 64, 0, g_stream>>>(
                    g_dy, g_dx, (const unsigned char *)w, scales, I, O, gsh, ng);
            else
                k3_dense_i4g_exactW<4><<<O, 64, 0, g_stream>>>(
                    g_dy, g_dx, (const unsigned char *)w, scales, I, O, gsh, ng);
        }
        else
            k3_dense_i4g_exact<<<O, 128, 0, g_stream>>>(
                g_dy, g_dx, (const unsigned char *)w, scales, I, O, gsh, ng);
    } else if (exact) {
        if (i4w == 4 && !(I & 3))
            k3_dense_i8_exact4<<<O, 64, 0, g_stream>>>(
                g_dy, g_dx, (const signed char *)w, scales, I, O);
        else
            k3_dense_i8_exact<<<O, 128, 0, g_stream>>>(
                g_dy, g_dx, (const signed char *)w, scales, I, O);
    } else if (fmt == 4) {
        int ng = (I + gs - 1) / gs;
        k3_dense_i4g<<<blocks, K3_FAST_THREADS, shbytes, g_stream>>>(
            g_dy, g_dx, (const unsigned char *)w, scales, I, O, gsh, ng, stage);
    } else {
        k3_dense_i8<<<blocks, K3_FAST_THREADS, shbytes, g_stream>>>(
            g_dy, g_dx, (const signed char *)w, scales, I, O, stage);
    }
    if (!ck(cudaGetLastError(), "dense launch")) return 0;
    if (!ck(cudaMemcpyAsync(y, g_dy, (size_t)O * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "dense y download")) return 0;
    return ck(cudaStreamSynchronize(g_stream), "dense sync");
}

/* Fused shared-expert gate/up + SiTU. NOT bit-exact (CUDA vs glibc
 * tanhf/expf); gate with tools/k3_quality.py, never the reference hash. */
__global__ void k3_dense_gate_up_situ(float *__restrict__ gate,
                                      const float *__restrict__ x,
                                      const unsigned char *__restrict__ w1p,
                                      const float *__restrict__ w1s,
                                      const unsigned char *__restrict__ w3p,
                                      const float *__restrict__ w3s,
                                      int I, int O, int gsh, int ng,
                                      float beta1, float beta2) {
    enum { W = 4, T = 256 / W };
    int o = blockIdx.x;
    if (o >= O) return;
    size_t rb = (size_t)((I + 1) >> 1);
    int p = (int)threadIdx.x;
    __shared__ float sh[2][W][T];
    for (int m = 0; m < 2; m++) {
        const unsigned char *w = (m ? w3p : w1p) + (size_t)o * rb;
        const float *scl      = (m ? w3s : w1s) + (size_t)o * ng;
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        for (int i = p * W; i + W - 1 < I; i += 256) {
            int g = i >> gsh;
            if (g >= ng) g = ng - 1;
            float sc = scl[g];
            float4 xx = *(const float4 *)(x + i);
            unsigned char c0 = w[i >> 1], c1 = w[(i >> 1) + 1];
            a0 += xx.x * (float)((int)(c0 & 15) - 8) * sc;
            a1 += xx.y * (float)((int)(c0 >> 4)  - 8) * sc;
            a2 += xx.z * (float)((int)(c1 & 15) - 8) * sc;
            a3 += xx.w * (float)((int)(c1 >> 4)  - 8) * sc;
        }
        sh[m][0][p] = a0; sh[m][1][p] = a1; sh[m][2][p] = a2; sh[m][3][p] = a3;
    }
    __syncthreads();
    for (int n = T / 2; n >= 1; n >>= 1) {
        if (p < n) {
            #pragma unroll
            for (int m = 0; m < 2; m++)
                #pragma unroll
                for (int j = 0; j < W; j++) sh[m][j][p] += sh[m][j][p + n];
        }
        __syncthreads();
    }
    if (!p) {
        float r[2];
        #pragma unroll
        for (int m = 0; m < 2; m++) {
            float v0 = sh[m][0][0], v1 = sh[m][1][0], v2 = sh[m][2][0], v3 = sh[m][3][0];
            v0 += v2; v1 += v3;      /* bit 1 */
            r[m] = v0 + v1;          /* bit 0 */
        }
        float g = r[0], u = r[1];
        /* The sigmoid's exponential dominates the divergence from the CPU:
         * measured over 200k samples, CUDA expf differs from glibc in 30.3% of
         * values while (float)exp((double)x) differs in 0.062% -- ~500x fewer.
         * tanhf stays CUDA's, which is the closer of the two there (7.1%
         * against 11.8% for the double form, because glibc's tanhf is itself
         * about 1 ulp off the correctly-rounded result). Same expression and
         * same left-to-right association as situf_ on the CPU. */
        float sig = (float)(1.0 / (1.0 + exp(-(double)g)));
        gate[o] = beta1 * tanhf(g / beta1) * sig
                * beta2 * tanhf(u / beta2);
    }
}


/* Batched fused gate/up + SiTU for prefill.
 *
 * The S>1 path used to loop tokens with a full cudaStreamSynchronize EACH
 * iteration -- 32 uploads, 32 launches, 32 downloads and 32 syncs for a
 * 32-token chunk, re-reading both weight matrices from DRAM every token. It was
 * the largest single prefill term at 14.4 s of 74.3.
 *
 * Here the block stages ITS w1 and w3 rows in shared memory once (2*rb bytes +
 * 2*ng scales; 8 KB at I=7168) and walks all C tokens against them. Per
 * (row, token) the arithmetic and its order are exactly the single-token
 * kernel's, so values do not move. */
__global__ void k3_dense_gate_up_situ_S(float *__restrict__ gate,
                                        const float *__restrict__ x,
                                        const unsigned char *__restrict__ w1p,
                                        const float *__restrict__ w1s,
                                        const unsigned char *__restrict__ w3p,
                                        const float *__restrict__ w3s,
                                        int I, int O, int gsh, int ng, int S,
                                        float beta1, float beta2) {
    enum { W = 4, T = 256 / W };
    int o = blockIdx.x;
    if (o >= O) return;
    size_t rb = (size_t)((I + 1) >> 1);

    extern __shared__ unsigned char shg[];
    unsigned char *a1 = shg;
    unsigned char *a3 = shg + ((rb + 15) & ~(size_t)15);
    float *s1 = (float *)(a3 + ((rb + 15) & ~(size_t)15));
    float *s3 = s1 + ng;
    for (size_t i = threadIdx.x; i < rb; i += blockDim.x) {
        a1[i] = w1p[(size_t)o * rb + i];
        a3[i] = w3p[(size_t)o * rb + i];
    }
    for (int g = threadIdx.x; g < ng; g += blockDim.x) {
        s1[g] = w1s[(size_t)o * ng + g];
        s3[g] = w3s[(size_t)o * ng + g];
    }
    __syncthreads();

    __shared__ float sh[2][W][T];
    int p = (int)threadIdx.x;
    for (int t = 0; t < S; t++) {
        const float *xs = x + (size_t)t * I;
        float g1[W], g3[W];
        #pragma unroll
        for (int j = 0; j < W; j++) { g1[j] = 0.f; g3[j] = 0.f; }
        for (int i = p * W; i + W - 1 < I; i += 256) {
            int g = i >> gsh; if (g >= ng) g = ng - 1;
            float c1 = s1[g], c3 = s3[g];
            float4 xx = *(const float4 *)(xs + i);
            unsigned char b0 = a1[(i >> 1)], b1 = a1[(i >> 1) + 1];
            g1[0] += xx.x * (float)((int)(b0 & 15) - 8) * c1;
            g1[1] += xx.y * (float)((int)(b0 >> 4)  - 8) * c1;
            g1[2] += xx.z * (float)((int)(b1 & 15) - 8) * c1;
            g1[3] += xx.w * (float)((int)(b1 >> 4)  - 8) * c1;
            unsigned char d0 = a3[(i >> 1)], d1 = a3[(i >> 1) + 1];
            g3[0] += xx.x * (float)((int)(d0 & 15) - 8) * c3;
            g3[1] += xx.y * (float)((int)(d0 >> 4)  - 8) * c3;
            g3[2] += xx.z * (float)((int)(d1 & 15) - 8) * c3;
            g3[3] += xx.w * (float)((int)(d1 >> 4)  - 8) * c3;
        }
        #pragma unroll
        for (int j = 0; j < W; j++) { sh[0][j][p] = g1[j]; sh[1][j][p] = g3[j]; }
        __syncthreads();
        for (int n = T / 2; n >= 1; n >>= 1) {
            if (p < n) {
                #pragma unroll
                for (int m = 0; m < 2; m++)
                    #pragma unroll
                    for (int j = 0; j < W; j++) sh[m][j][p] += sh[m][j][p + n];
            }
            __syncthreads();
        }
        if (!p) {
            float r[2];
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                float v0 = sh[m][0][0], v1 = sh[m][1][0], v2 = sh[m][2][0], v3 = sh[m][3][0];
                v0 += v2; v1 += v3; r[m] = v0 + v1;
            }
            float gg = r[0], uu = r[1];
            float sig = (float)(1.0 / (1.0 + exp(-(double)gg)));
            gate[(size_t)t * O + o] = beta1 * tanhf(gg / beta1) * sig
                                    * beta2 * tanhf(uu / beta2);
        }
        __syncthreads();
    }
}

extern "C" int coli_k3_gate_up_situ(float *gate, const float *x,
                                    const void *w1p, const float *w1s,
                                    const void *w3p, const float *w3s,
                                    int S, int I, int O, int gs,
                                    float beta1, float beta2) {
    if (!g_ready || I <= 0 || O <= 0 || S <= 0 || (I & 3)) return 0;
    if (gs < 4 || (gs & (gs - 1)) || (I & (gs - 1))) return 0;
    int gsh = 0; while ((1 << gsh) < gs) gsh++;
    if (S > 1) {                       /* prefill: one launch for the whole chunk */
        size_t rb = (size_t)((I + 1) >> 1), ng2 = (size_t)((I + gs - 1) / gs);
        size_t shb = 2 * ((rb + 15) & ~(size_t)15) + 2 * ng2 * sizeof(float);
        int mx = 0;
        cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        if (shb <= (size_t)mx) {
            if (shb > 48u * 1024u &&
                cudaFuncSetAttribute(k3_dense_gate_up_situ_S,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb)
                != cudaSuccess) { cudaGetLastError(); goto situ_pertoken; }
            if (!ensure_dense_scratch((int)((size_t)S * I), (int)((size_t)S * O))) return 0;
            if (!ck(cudaMemcpyAsync(g_dx, x, (size_t)S * I * sizeof(float),
                                    cudaMemcpyHostToDevice, g_stream), "situ x S")) return 0;
            k3_dense_gate_up_situ_S<<<O, 64, shb, g_stream>>>(
                g_dy, g_dx, (const unsigned char *)w1p, w1s,
                (const unsigned char *)w3p, w3s, I, O, gsh, (I + gs - 1) / gs, S,
                beta1, beta2);
            if (!ck(cudaGetLastError(), "situ S launch")) return 0;
            if (!ck(cudaMemcpyAsync(gate, g_dy, (size_t)S * O * sizeof(float),
                                    cudaMemcpyDeviceToHost, g_stream), "situ y S")) return 0;
            return ck(cudaStreamSynchronize(g_stream), "situ S sync");
        }
    }
situ_pertoken:
    if (!ensure_dense_scratch(I, O)) return 0;
    for (int t = 0; t < S; t++) {
        if (!ck(cudaMemcpyAsync(g_dx, x + (size_t)t * I, (size_t)I * sizeof(float),
                                cudaMemcpyHostToDevice, g_stream), "situ x")) return 0;
        k3_dense_gate_up_situ<<<O, 64, 0, g_stream>>>(
            g_dy, g_dx, (const unsigned char *)w1p, w1s,
            (const unsigned char *)w3p, w3s, I, O, gsh, (I + gs - 1) / gs, beta1, beta2);
        if (!ck(cudaGetLastError(), "situ launch")) return 0;
        if (!ck(cudaMemcpyAsync(gate + (size_t)t * O, g_dy, (size_t)O * sizeof(float),
                                cudaMemcpyDeviceToHost, g_stream), "situ y")) return 0;
        if (!ck(cudaStreamSynchronize(g_stream), "situ sync")) return 0;
    }
    return 1;
}

/* The candidate count varies far more per layer than one sample suggested:
 * measured 32, 51, 264, 119, 88, 85 at delta 0.01. A cap of 192 silently
 * dropped the overflow, so those layers never re-scored their true top-16 and
 * fell back to int8 quality -- PCC 0.9826. The cap is now the full expert
 * count, which cannot overflow; the average is ~106 of 896, so the exact pass
 * still touches ~12% of the rows. */
#define K3_RT_CAP 1024
static int *g_rt_cand = nullptr, *g_rt_n = nullptr;

/* Chained on one stream: int8 scores, candidate selection, exact re-score.
 * Two extra launches per layer and no extra round trip. */
extern "C" int coli_k3_router2(float *scores, const float *x,
                               const void *q8, const float *s8,
                               const float *wf32, const float *rbias,
                               int E, int I, int K, float delta) {
    if (!g_ready || E <= 0 || E > 1024 || I <= 0) return 0;
    if (!ensure_dense_scratch(I, E)) return 0;
    if (!g_rt_cand && !ck(cudaMalloc(&g_rt_cand, K3_RT_CAP * sizeof(int)), "rt cand")) return 0;
    int cap = E < K3_RT_CAP ? E : K3_RT_CAP;
    if (!g_rt_n && !ck(cudaMalloc(&g_rt_n, sizeof(int)), "rt n")) return 0;
    if (!ck(cudaMemcpyAsync(g_dx, x, (size_t)I * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "rt x")) return 0;
    k3_dense_i8_exact<<<E, 128, 0, g_stream>>>(g_dy, g_dx, (const signed char *)q8, s8, I, E);
    k3_router_cand<<<1, 256, 0, g_stream>>>(g_dy, rbias, E, K, delta,
                                            g_rt_cand, g_rt_n, cap);
    k3_router_exact<<<E, 256, 0, g_stream>>>(g_dy, g_dx, wf32,
                                                     g_rt_cand, g_rt_n, I, cap);
    if (!ck(cudaGetLastError(), "rt launch")) return 0;
    if (!ck(cudaMemcpyAsync(scores, g_dy, (size_t)E * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "rt scores")) return 0;
    /* How many rows actually needed re-scoring? If this ever reaches the cap the
     * overflow is silently dropped and the true top-K may not be re-scored. */
    { static int nrep = 0; int hn = 0;
      if (nrep < 6) {
          cudaMemcpyAsync(&hn, g_rt_n, sizeof(int), cudaMemcpyDeviceToHost, g_stream);
          cudaStreamSynchronize(g_stream);
          fprintf(stderr, "[K3/RT] candidates=%d cap=%d%s\n", hn, cap,
                  hn > cap ? "  *** OVER CAP ***" : "");
          nrep++;
      } }
    return ck(cudaStreamSynchronize(g_stream), "rt sync");
}

/* Batched 1-bit expert application. Mirrors coli_k3_expert_batch_w2 but drives
 * the w1 (sign-bit) kernels, which is what a 1-bit store actually runs. */
extern "C" int coli_k3_expert_batch_w1(const void *const *w1p, const void *const *w1s,
                                       const void *const *w2p, const void *const *w2s,
                                       const void *const *w3p, const void *const *w3s,
                                       int n, float *hz_all, const float *z,
                                       int latent, int inter, float beta1, float beta2) {
    if (!g_ready || n < 1 || n > K3_BATCH_MAX) return 0;
    if (latent != g_latent || inter != g_inter) return 0;
    if (!ensure_batch(n, latent, inter)) return 0;

    K3Expert host[K3_BATCH_MAX];
    for (int j = 0; j < n; j++) {
        host[j].w1p = (const unsigned char *)w1p[j]; host[j].w1s = (const unsigned char *)w1s[j];
        host[j].w2p = (const unsigned char *)w2p[j]; host[j].w2s = (const unsigned char *)w2s[j];
        host[j].w3p = (const unsigned char *)w3p[j]; host[j].w3s = (const unsigned char *)w3s[j];
    }
    if (!ck(cudaMemcpyAsync(g_dev_ex, host, (size_t)n * sizeof(K3Expert),
                            cudaMemcpyHostToDevice, g_stream), "w1 batch desc")) return 0;
    if (!ck(cudaMemcpyAsync(g_z, z, (size_t)latent * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "w1 batch z")) return 0;
    dim3 gu((unsigned)((inter + K3_WARPS - 1) / K3_WARPS), (unsigned)n);
    dim3 dn((unsigned)((latent + K3_WARPS - 1) / K3_WARPS), (unsigned)n);
    k3_w1_gate_up_fast_b<<<gu, K3_FAST_THREADS,
                           K3_W1_SHFLOATS(latent)*sizeof(float), g_stream>>>(
        g_gate_b, g_dev_ex, g_z, latent, inter, beta1, beta2);
    k3_w1_down_fast_b<<<dn, K3_FAST_THREADS,
                        K3_W1_SHFLOATS(inter)*sizeof(float), g_stream>>>(
        g_hz_b, g_dev_ex, g_gate_b, inter, latent);
    if (!ck(cudaGetLastError(), "w1 batch launch")) return 0;
    if (!ck(cudaMemcpyAsync(hz_all, g_hz_b, (size_t)n * latent * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "w1 batch hz")) return 0;
    return ck(cudaStreamSynchronize(g_stream), "w1 batch sync");
}

/* Prefill path: fmt 4 with S>1. Weight row staged once per block, C tokens
 * walked against it. K3_PREFILL_GPU=0 falls back to the generic kernel. */
extern "C" int coli_k3_dense_s(float *y, const float *x, const void *w, const float *scales,
                               int fmt, int S, int I, int O, int gs) {
    static int en = -1;
    if (en < 0) { const char *e = getenv("K3_PREFILL_GPU"); en = e ? atoi(e) : 1; }
    if (!en || !g_ready || fmt != 4 || S < 2 || I <= 0 || O <= 0) return 0;
    if (gs <= 0 || (gs & (gs - 1)) || (I & (gs - 1)) || (I & 3)) return 0;
    int gsh = 0; while ((1 << gsh) < gs) gsh++;
    int ng = (I + gs - 1) / gs;
    size_t rb = (size_t)((I + 1) >> 1);
    size_t shb = ((rb + 15) & ~(size_t)15) + (size_t)ng * sizeof(float);
    int mx = 0;
    cudaDeviceGetAttribute(&mx, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (shb > (size_t)mx) return 0;                    /* row will not fit; caller falls back */
    if (shb > 48u * 1024u) {
        if (cudaFuncSetAttribute(k3_dense_i4g_exactW_S<4>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb)
            != cudaSuccess) { cudaGetLastError(); return 0; }
    }
    if (!ensure_dense_scratch((int)((size_t)S * I), (int)((size_t)S * O))) return 0;
    if (!ck(cudaMemcpyAsync(g_dx, x, (size_t)S * I * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "prefill x")) return 0;
    if (g_k_pfshape < 0) {
        const char *e = getenv("K3_PF_SHAPE");
        /* DEFAULT IS tree, the bit-exact shape, chosen on quality not speed.
         * Against the exact path on the rank-symmetric decode gate (26
         * token-selecting positions):
         *     tree + NEON   PCC 0.999547   top-1 100.0%   PASS
         *     warp + NEON   diverged       top-1  38.5%
         *     tile + NEON   PCC 0.992058   top-1 100.0%   FAIL
         * Both non-exact shapes miss the >= 0.999 bar -- tile drags the pair to
         * 0.992, warp flips a token within 24 steps. tree costs 8.2% of prefill
         * against tile and leaves decode untouched, so 5.2 tok/s short-prompt
         * and 4.61 at ctx 1963 are unaffected.
         *
         * K3_PF_SHAPE=tile restores the faster prefill for anyone who accepts
         * coherent-but-not-equivalent output. The text is sound either way; it
         * simply is not within 0.999 of the exact path. */
        g_k_pfshape = !e                  ? 0
                    : !strcmp(e, "tree")  ? 0
                    : !strcmp(e, "warp")  ? 1 : 2;
    }
    int shape = g_k_pfshape;
    size_t rbA = ((rb + 15) & ~(size_t)15);
    size_t shw = (size_t)K3_PF_WARPS * rbA + (size_t)K3_PF_WARPS * ng * sizeof(float);
    /* 2D tile: needs a g64 store and a reduction length that divides evenly. */
    if (shape == 2 && gs == 64 && (I % K3_TG_I) == 0) {
        dim3 gr((O + K3_TG_O - 1) / K3_TG_O, (S + K3_TG_S - 1) / K3_TG_S);
        k3_dense_i4g_tg<<<gr, 256, 0, g_stream>>>(
            g_dy, g_dx, (const unsigned char *)w, scales, I, O, ng, S);
        if (!ck(cudaGetLastError(), "prefill tile launch")) return 0;
        goto pf_done;
    }
    if (shape && shw <= (size_t)mx) {
        if (shw > 48u * 1024u &&
            cudaFuncSetAttribute(k3_dense_i4g_pf,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shw)
            != cudaSuccess) { cudaGetLastError(); goto pf_tree; }
        k3_dense_i4g_pf<<<(O + K3_PF_WARPS - 1) / K3_PF_WARPS, K3_PF_WARPS * 32, shw, g_stream>>>(
            g_dy, g_dx, (const unsigned char *)w, scales, I, O, gsh, ng, S);
        if (!ck(cudaGetLastError(), "prefill warp launch")) return 0;
        goto pf_done;
    }
pf_tree:
    k3_dense_i4g_exactW_S<4><<<O, 64, shb, g_stream>>>(
        g_dy, g_dx, (const unsigned char *)w, scales, I, O, gsh, ng, S);
pf_done:;
    if (!ck(cudaGetLastError(), "prefill launch")) return 0;
    if (!ck(cudaMemcpyAsync(y, g_dy, (size_t)S * O * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "prefill y")) return 0;
    return ck(cudaStreamSynchronize(g_stream), "prefill sync");
}
