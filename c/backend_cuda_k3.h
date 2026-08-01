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

void coli_k3_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
