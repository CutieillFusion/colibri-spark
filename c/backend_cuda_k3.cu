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

extern "C" int coli_k3_expert_w2(const void *w1p, const void *w1s,
                                 const void *w2p, const void *w2s,
                                 const void *w3p, const void *w3s,
                                 float *hz, const float *z,
                                 int latent, int inter, float beta1, float beta2) {
    if (!g_ready || latent != g_latent || inter != g_inter) return 0;
    if (!ck(cudaMemcpyAsync(g_z, z, (size_t)latent * sizeof(float),
                            cudaMemcpyHostToDevice, g_stream), "z upload")) return 0;
    k3_w2_gate_up_situ<<<inter, K3_THREADS, 0, g_stream>>>(
        g_gate, (const unsigned char *)w1p, (const unsigned char *)w1s,
        (const unsigned char *)w3p, (const unsigned char *)w3s, g_z, latent, beta1, beta2);
    k3_w2_down<<<latent, K3_THREADS, 0, g_stream>>>(
        g_hz, (const unsigned char *)w2p, (const unsigned char *)w2s, g_gate, inter);
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
    int o = blockIdx.x, j = blockIdx.y, ng = I >> 5;
    size_t rb = (size_t)(I >> 2);
    __shared__ float sh[K3_THREADS / 32];
    float a1 = row_dot_w2(ex[j].w1p + (size_t)o * rb, ex[j].w1s + (size_t)o * ng, z, ng);
    float g1 = blk_reduce(a1, sh);
    __syncthreads();
    float a3 = row_dot_w2(ex[j].w3p + (size_t)o * rb, ex[j].w3s + (size_t)o * ng, z, ng);
    float g3 = blk_reduce(a3, sh);
    if (threadIdx.x == 0)
        gate[(size_t)j * O + o] = beta1 * tanhf(g1 / beta1) * (1.f / (1.f + expf(-g1)))
                                * beta2 * tanhf(g3 / beta2);
}

__global__ void k3_w2_down_b(float *__restrict__ hz, const K3Expert *__restrict__ ex,
                             const float *__restrict__ gate, int I, int O) {
    int o = blockIdx.x, j = blockIdx.y, ng = I >> 5;
    size_t rb = (size_t)(I >> 2);
    __shared__ float sh[K3_THREADS / 32];
    float a = row_dot_w2(ex[j].w2p + (size_t)o * rb, ex[j].w2s + (size_t)o * ng,
                         gate + (size_t)j * I, ng);
    float t = blk_reduce(a, sh);
    if (threadIdx.x == 0) hz[(size_t)j * O + o] = t;
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
    dim3 gu((unsigned)inter, (unsigned)n), dn((unsigned)latent, (unsigned)n);
    k3_w2_gate_up_situ_b<<<gu, K3_THREADS, 0, g_stream>>>(g_gate_b, g_dev_ex, g_z,
                                                          latent, inter, beta1, beta2);
    k3_w2_down_b<<<dn, K3_THREADS, 0, g_stream>>>(g_hz_b, g_dev_ex, g_gate_b, inter, latent);
    if (!ck(cudaMemcpyAsync(hz_all, g_hz_b, (size_t)n * latent * sizeof(float),
                            cudaMemcpyDeviceToHost, g_stream), "hz download")) return 0;
    if (!ck(cudaStreamSynchronize(g_stream), "batch sync")) return 0;   /* ONE sync per layer */
    return 1;
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

extern "C" void coli_k3_shutdown(void) {
    if (!g_ready) return;
    cudaFree(g_z); cudaFree(g_gate); cudaFree(g_hz);
    cudaStreamDestroy(g_stream);
    g_ready = 0;
}
