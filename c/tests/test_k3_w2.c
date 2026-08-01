/* Validate the packed-2-bit expert kernel against a CPU reference.
 *
 * The warp-per-row rewrite changed both the lane->group mapping and the
 * reduction, so its output legitimately differs from the block-per-row version
 * in the last bits -- and at greedy decode that can flip a token. Comparing
 * generated text cannot separate "benign reassociation" from "wrong answer",
 * so this computes the expected hz in double on the CPU and checks the kernel
 * against it directly. Needs no checkpoint.
 *
 *   ./c/tests/test_k3_w2
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../backend_cuda_k3.h"

static const float LUT[4] = { -4.f, -1.f, 1.f, 4.f };

static uint32_t rs = 4242;
static uint32_t rnd(void) { rs = rs * 1664525u + 1013904223u; return rs; }
static float frand(void) { return (float)((rnd() >> 8) & 0xFFFF) / 32768.0f - 1.0f; }

static float mx4_scale(unsigned char s) {
    union { uint32_t u; float f; } b; b.u = (uint32_t)s << 23; return b.f;
}

/* One packed row against x, accumulated in double so the reference is not
 * itself order-sensitive. */
static double ref_row(const uint8_t *pk, const uint8_t *sc, const float *x, int I) {
    double acc = 0;
    for (int g = 0; g < I / 32; g++) {
        double p = 0;
        for (int k = 0; k < 8; k++) {
            uint8_t c = pk[g * 8 + k];
            for (int j = 0; j < 4; j++)
                p += (double)x[g * 32 + 4 * k + j] * LUT[(c >> (2 * j)) & 3];
        }
        acc += p * (double)mx4_scale(sc[g]);
    }
    return acc;
}

static float situ(float g, float u, float b1, float b2) {
    return b1 * tanhf(g / b1) * (1.f / (1.f + expf(-g))) * b2 * tanhf(u / b2);
}

int main(void) {
    const int latent = 512, inter = 256;          /* multiples of 32 */
    const float b1 = 4.0f, b2 = 25.0f;            /* K3's situ betas */
    if (!coli_k3_init(0, latent, inter)) { fprintf(stderr, "init failed\n"); return 77; }

    size_t w1p_n = (size_t)inter * (latent / 4), w1s_n = (size_t)inter * (latent / 32);
    size_t w2p_n = (size_t)latent * (inter / 4), w2s_n = (size_t)latent * (inter / 32);
    uint8_t *w1p = malloc(w1p_n), *w1s = malloc(w1s_n);
    uint8_t *w2p = malloc(w2p_n), *w2s = malloc(w2s_n);
    uint8_t *w3p = malloc(w1p_n), *w3s = malloc(w1s_n);
    float *z = malloc(sizeof(float) * latent), *hz = malloc(sizeof(float) * latent);
    for (size_t i = 0; i < w1p_n; i++) { w1p[i] = (uint8_t)rnd(); w3p[i] = (uint8_t)rnd(); }
    for (size_t i = 0; i < w2p_n; i++) w2p[i] = (uint8_t)rnd();
    /* exponents near 127 so values land in a sane range */
    for (size_t i = 0; i < w1s_n; i++) { w1s[i] = 120 + (rnd() % 9); w3s[i] = 120 + (rnd() % 9); }
    for (size_t i = 0; i < w2s_n; i++) w2s[i] = 120 + (rnd() % 9);
    for (int i = 0; i < latent; i++) z[i] = frand();

    /* CPU reference: hz = w2 @ SiTU(w1@z, w3@z) */
    float *gate = malloc(sizeof(float) * inter);
    for (int o = 0; o < inter; o++) {
        double a1 = ref_row(w1p + (size_t)o * (latent / 4), w1s + (size_t)o * (latent / 32), z, latent);
        double a3 = ref_row(w3p + (size_t)o * (latent / 4), w3s + (size_t)o * (latent / 32), z, latent);
        gate[o] = situ((float)a1, (float)a3, b1, b2);
    }
    double *ref = malloc(sizeof(double) * latent);
    for (int o = 0; o < latent; o++)
        ref[o] = ref_row(w2p + (size_t)o * (inter / 4), w2s + (size_t)o * (inter / 32), gate, inter);

    memset(hz, 0, sizeof(float) * latent);
    if (!coli_k3_expert_w2(w1p, w1s, w2p, w2s, w3p, w3s, hz, z, latent, inter, b1, b2)) {
        fprintf(stderr, "coli_k3_expert_w2 refused\n"); return 1;
    }

    double se = 0, ss = 0, worst = 0;
    for (int o = 0; o < latent; o++) {
        double d = hz[o] - ref[o];
        se += d * d; ss += ref[o] * ref[o];
        double rel = fabs(ref[o]) > 1e-6 ? fabs(d) / fabs(ref[o]) : fabs(d);
        if (rel > worst) worst = rel;
    }
    double rel_rms = sqrt(se / ss);
    /* fp32 accumulation over 512 terms vs a double reference: ~1e-6 expected.
     * Anything at 1e-2 or above is a layout/reduction bug, not rounding. */
    int ok = rel_rms < 1e-4;
    printf("rel_rms=%.3e  worst_elem=%.3e  %s\n", rel_rms, worst, ok ? "PASS" : "FAIL");
    coli_k3_shutdown();
    return ok ? 0 : 1;
}
