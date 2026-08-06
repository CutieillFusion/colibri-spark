#ifndef COLIBRI_BACKEND_CUDA_K3_H
#define COLIBRI_BACKEND_CUDA_K3_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* MXFP4 routed-expert GEMV for Kimi-K3. Separate from backend_cuda.h because
 * MXFP4 is not one of that backend's formats and the expert lifecycle is
 * different: weights stream in from the shards rather than being resident.
 * See backend_cuda_k3.cu for the layout and the zero-copy rationale. */

int  coli_k3_init(int device, int latent, int inter);

/* Map one expert-slot allocation for zero-copy reads. Call once per slot, with
 * the whole posix_memalign'd region. Returns 0 if the mapping cannot alias the
 * host pointer, in which case the caller must stay on the CPU path. */
int  coli_k3_register(void *p, size_t bytes);
void coli_k3_unregister(void *p);

/* hz[latent] = w2 @ SiTU(w1 @ z, w3 @ z), reading the slot's native MXFP4.
 * The caller still does u += wk * hz, so this is a drop-in for the three
 * matmul_mxfp4 calls in expert_apply(). Returns 0 if unavailable. */
int  coli_k3_expert(const void *w1p, const void *w1s,
                    const void *w2p, const void *w2s,
                    const void *w3p, const void *w3s,
                    float *hz, const float *z,
                    int latent, int inter, float beta1, float beta2);

/* Same contract, packed 2-bit codes ({-4,-1,1,4} indices) instead of e2m1.
 * Scales are unchanged ue8m0, so only the code unpacking differs. */
int  coli_k3_expert_w2(const void *w1p, const void *w1s,
                       const void *w2p, const void *w2s,
                       const void *w3p, const void *w3s,
                       float *hz, const float *z,
                       int latent, int inter, float beta1, float beta2);

/* One launch for a whole layer's experts instead of two-plus-a-sync each.
 * hz_all is [n][latent]; the caller folds in the routing weights. Max 64. */
int  coli_k3_expert_batch_w2(const void *const *w1p, const void *const *w1s,
                             const void *const *w2p, const void *const *w2s,
                             const void *const *w3p, const void *const *w3s,
                             int n, float *hz_all, const float *z,
                             int latent, int inter, float beta1, float beta2);

/* Bench-only: device-resident copy, to separate zero-copy cost from kernel cost. */
void *coli_k3_devcopy(const void *src, size_t n);

/* Dense GEMV for K3's own w_matmul. Weights are read zero-copy from the host
 * buffers (no upload or duplicate), with stock-compatible reduction order by
 * default so decode remains bit-exact. S==1 only; returns 0 when it declines
 * so the caller keeps its old path. */
int  coli_k3_dense(float *y, const float *x, const void *w, const float *scales,
                   int fmt, int S, int I, int O, int gs);

/* 1-bit variant: one sign bit per weight, 8 per byte, same ue8m0 scales. */
int  coli_k3_expert_w1(const void *w1p, const void *w1s,
                       const void *w2p, const void *w2s,
                       const void *w3p, const void *w3s,
                       float *hz, const float *z,
                       int latent, int inter, float beta1, float beta2);

/* Optional device copy of a dense weight/scale blob, under K3_DENSE_DEV_GB.
 * Returns null when it declines; the caller then keeps its zero-copy pointer. */
void  *coli_k3_devmirror(const void *host, size_t bytes);
size_t coli_k3_devmirror_used(void);
void   coli_k3_devmirror_report(void);

int  coli_k3_gate_up_situ(float *gate, const float *x,
                          const void *w1p, const float *w1s,
                          const void *w3p, const float *w3s,
                          int S, int I, int O, int gs, float beta1, float beta2);

/* Two-pass router: int8 search, exact f32 decision near the top-K boundary.
 * rbias and wf32 must be device-visible (mirrored or host-registered). */
int  coli_k3_router2(float *scores, const float *x,
                     const void *q8, const float *s8,
                     const float *wf32, const float *rbias,
                     int E, int I, int K, float delta);

/* Batched 1-bit expert application: one launch for all n experts of a layer.
 * Bit-exact -- only the launch geometry differs from coli_k3_expert_w1. */
int  coli_k3_expert_batch_w1(const void *const *w1p, const void *const *w1s,
                             const void *const *w2p, const void *const *w2s,
                             const void *const *w3p, const void *const *w3s,
                             int n, float *hz_all, const float *z,
                             int latent, int inter, float beta1, float beta2);

void coli_k3_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
