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
     * halve the load instructions. */
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
                    const float *xb = xs + q * 16 + k * 4;
                    p += xb[0] * w2_val( c        & 3u)
                       + xb[1] * w2_val((c >> 2)  & 3u)
                       + xb[2] * w2_val((c >> 4)  & 3u)
                       + xb[3] * w2_val((c >> 6)  & 3u);
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

extern "C" size_t coli_k3_devmirror_used(void) { return g_devmir_used; }
/* Which limit actually bound: the byte budget, or the per-tensor cap. */
extern "C" void coli_k3_devmirror_report(void) {
    fprintf(stderr, "[K3/EXP] dense mirrors: %.2f GB placed, %.2f GB skipped for budget, "
                    "%.2f GB skipped over the 64 MB cap\n",
            g_devmir_used/1e9, g_devmir_budget_skip/1e9, g_devmir_cap_skip/1e9);
}

/* Returns a device copy, or null when it declines (budget spent, too large,
 * or allocation failed). Never fails the caller: they keep the host pointer. */
extern "C" void *coli_k3_devmirror(const void *host, size_t bytes) {
    if (!g_ready || !host || !bytes) return nullptr;
    if (!g_devmir_init) {
        const char *e = getenv("K3_DENSE_DEV_GB");
        double gb = e ? atof(e) : 0.0;
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
    if (cudaMalloc(&d, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    if (cudaMemcpy(d, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaGetLastError(); cudaFree(d); return nullptr;
    }
    g_devmir_left -= bytes; g_devmir_used += bytes;
    return d;
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
    if (fmt != 1 && fmt != 4) return 0;
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
    int exact = 1;
    { const char *fe = getenv("K3_DENSE_EXACT"); if (fe) exact = atoi(fe); }
    int blocks = (O + K3_WARPS - 1) / K3_WARPS;
    static int i4w = -1;
    if (i4w < 0) { const char *e = getenv("K3_DENSE_I4W"); i4w = e ? atoi(e) : 4; }
    if (exact && fmt == 4) {
        int ng = (I + gs - 1) / gs;
        if (i4w == 4 && gs >= 4 && !(I & 3))
            k3_dense_i4g_exactW<4><<<O, 64, 0, g_stream>>>(
                g_dy, g_dx, (const unsigned char *)w, scales, I, O, gsh, ng);
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
