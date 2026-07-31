/* Validate coli_cuda_matmul against the CPU kernels kimi_k3.c would otherwise
 * run, for exactly the three formats w_load() produces (0=f32, 1=int8 per-row,
 * 4=int4-g64). This is the correctness gate for the K3 CUDA path: it needs no
 * checkpoint, so it can run while the 1.4 TB model is still staging, and it is
 * the only thing that actually answers "does the GPU agree with the CPU on
 * sm_121".
 *
 * Tolerance is relative RMS, not exact equality: the GPU accumulates in a
 * different order (and may use tensor cores), so bitwise agreement is not the
 * contract. Anything above ~1e-3 means a real layout/scale mismatch rather
 * than reassociation noise.
 *
 *   make -C c tests/test_k3_cuda CUDA=1 ARCH=native CUDA_ARCH=native
 *   ./c/tests/test_k3_cuda
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../quant.h"
#include "../backend_cuda.h"

static uint32_t rs = 12345;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return (float)((rs >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static double rel_rms(const float *a, const float *b, int n) {
    double se = 0, ss = 0;
    for (int i = 0; i < n; i++) { double d = (double)a[i] - b[i]; se += d * d; ss += (double)a[i] * a[i]; }
    return ss > 0 ? sqrt(se / ss) : sqrt(se);
}

static int check(const char *tag, const float *ref, const float *got, int n, double tol) {
    double r = rel_rms(ref, got, n);
    int ok = r <= tol && isfinite(r);
    printf("  %-22s rel_rms=%.3e  %s\n", tag, r, ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    const int I = 512, O = 256, gs = 64;
    int dev = 0;
    if (!coli_cuda_init(&dev, 1)) { fprintf(stderr, "coli_cuda_init failed\n"); return 77; }
    printf("device 0 integrated=%d\n", coli_cuda_device_integrated(0));

    float *x = malloc(sizeof(float) * 8 * I);
    for (int i = 0; i < 8 * I; i++) x[i] = frand();

    int ok = 1;
    for (int si = 0; si < 2; si++) {
        int S = si ? 8 : 1;                 /* decode (GEMV) and a small prefill (GEMM) */
        float *ref = malloc(sizeof(float) * S * O), *got = malloc(sizeof(float) * S * O);
        printf("S=%d\n", S);

        /* fmt 0: f32 */
        {
            float *w = malloc(sizeof(float) * (size_t)O * I);
            for (int i = 0; i < O * I; i++) w[i] = frand() * 0.05f;
            matmul(ref, x, w, S, I, O);
            ColiCudaTensor *t = NULL;
            memset(got, 0, sizeof(float) * S * O);
            if (!coli_cuda_matmul(&t, got, x, w, NULL, 0, S, I, O, 0, 0)) { printf("  fmt0 matmul refused\n"); ok = 0; }
            else ok &= check("fmt0 f32", ref, got, S * O, 1e-4);
            coli_cuda_tensor_free(t); free(w);
        }
        /* fmt 1: int8 with one scale per output row */
        {
            int8_t *q8 = malloc((size_t)O * I);
            float *s = malloc(sizeof(float) * O);
            for (int o = 0; o < O; o++) { s[o] = 0.002f + 0.001f * (o % 7);
                for (int i = 0; i < I; i++) q8[(size_t)o * I + i] = (int8_t)((int)(frand() * 127)); }
            matmul_q(ref, x, q8, s, S, I, O);
            ColiCudaTensor *t = NULL;
            memset(got, 0, sizeof(float) * S * O);
            if (!coli_cuda_matmul(&t, got, x, q8, s, 1, S, I, O, 0, 0)) { printf("  fmt1 matmul refused\n"); ok = 0; }
            else ok &= check("fmt1 int8", ref, got, S * O, 1e-3);
            coli_cuda_tensor_free(t); free(q8); free(s);
        }
        /* fmt 4: int4 grouped, zero-point 8, low nibble first (w_addrow's layout) */
        {
            int rb = (I + 1) / 2, ng = (I + gs - 1) / gs;
            uint8_t *q4 = malloc((size_t)O * rb);
            float *s = malloc(sizeof(float) * (size_t)O * ng);
            for (int o = 0; o < O; o++) {
                for (int g = 0; g < ng; g++) s[(size_t)o * ng + g] = 0.003f + 0.0005f * ((o + g) % 5);
                for (int b = 0; b < rb; b++) q4[(size_t)o * rb + b] =
                    (uint8_t)(((int)(frand() * 7 + 8) & 0xF) | (((int)(frand() * 7 + 8) & 0xF) << 4));
            }
            matmul_i4_grouped(ref, x, q4, s, S, I, O, gs);
            ColiCudaTensor *t = NULL;
            memset(got, 0, sizeof(float) * S * O);
            if (!coli_cuda_matmul(&t, got, x, q4, s, 4, S, I, O, 0, gs)) { printf("  fmt4 matmul refused\n"); ok = 0; }
            else ok &= check("fmt4 int4-g64", ref, got, S * O, 1e-3);
            coli_cuda_tensor_free(t); free(q4); free(s);
        }
        free(ref); free(got);
    }
    free(x);
    coli_cuda_shutdown();
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
