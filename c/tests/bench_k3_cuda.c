/* CPU vs GPU for w_matmul at REAL Kimi-K3 tensor shapes.
 *
 * The correctness gate (test_k3_cuda.c) uses toy dimensions. This one answers
 * the question that actually decides whether the CUDA path is worth using:
 * does it beat the 20 Cortex-X925 cores at the shapes K3 really runs, and does
 * the answer differ between decode (S=1, a GEMV) and prefill (S>>1, a GEMM)?
 *
 * Expectation going in: at S=1 every matmul is memory-bandwidth-bound, and on
 * GB10 the GPU shares the SAME LPDDR5X as the CPU (~273 GB/s), so there is no
 * bandwidth to win and the speedup should be ~1x. At large S the work becomes
 * compute-bound and the GPU should pull away. Measuring rather than asserting.
 *
 *   ./c/tests/bench_k3_cuda
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../quant.h"
#include "../backend_cuda.h"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static uint32_t rs = 999;
static float frand(void){ rs = rs*1664525u + 1013904223u; return (float)((rs>>8)&0xFFFF)/32768.0f - 1.0f; }

typedef struct { const char *name; int O, I; } Shape;

/* Straight from model.safetensors.index.json; gs=64 matches w_load's int4-g64. */
static Shape SHAPES[] = {
    {"KDA q/k/v/g_proj", 12288,  7168},
    {"KDA o_proj",        7168, 12288},
    {"MLA q_b_proj",     18432,  1536},
    {"LatentMoE lat_up",  7168,  3584},
    {"shared gate/up",    6144,  7168},
    {"lm_head",         163840,  7168},
};

int main(void){
    const int gs = 64;
    int dev = 0;
    if(!coli_cuda_init(&dev,1)){ fprintf(stderr,"coli_cuda_init failed\n"); return 77; }
    printf("%-20s %7s %6s %6s %10s %10s %8s\n",
           "shape","O","I","S","CPU ms","GPU ms","speedup");

    int Ss[] = {1, 64, 512};
    for(size_t si=0; si<sizeof(SHAPES)/sizeof(*SHAPES); si++){
        int O=SHAPES[si].O, I=SHAPES[si].I;
        int rb=(I+1)/2, ng=(I+gs-1)/gs;
        uint8_t *q4 = malloc((size_t)O*rb);
        float   *s  = malloc(sizeof(float)*(size_t)O*ng);
        if(!q4||!s){ fprintf(stderr,"OOM on %s\n",SHAPES[si].name); return 1; }
        for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(rs=rs*1664525u+1013904223u);
        for(size_t i=0;i<(size_t)O*ng;i++) s[i]=0.003f;

        ColiCudaTensor *t=NULL;
        for(size_t k=0;k<sizeof(Ss)/sizeof(*Ss);k++){
            int S=Ss[k];
            float *x=malloc(sizeof(float)*(size_t)S*I), *y=malloc(sizeof(float)*(size_t)S*O);
            for(size_t i=0;i<(size_t)S*I;i++) x[i]=frand();

            int iters = S==1 ? 20 : (S<=64 ? 5 : 2);
            matmul_i4_grouped(y,x,q4,s,S,I,O,gs);                 /* warm */
            double c0=now_s();
            for(int r=0;r<iters;r++) matmul_i4_grouped(y,x,q4,s,S,I,O,gs);
            double cms=(now_s()-c0)*1e3/iters;

            double gms=-1;
            if(coli_cuda_matmul(&t,y,x,q4,s,4,S,I,O,0,gs)){       /* warm + upload */
                double g0=now_s();
                for(int r=0;r<iters;r++) coli_cuda_matmul(&t,y,x,q4,s,4,S,I,O,0,gs);
                gms=(now_s()-g0)*1e3/iters;
            }
            if(gms>0) printf("%-20s %7d %6d %6d %10.2f %10.2f %7.2fx\n",
                             k?"":SHAPES[si].name,O,I,S,cms,gms,cms/gms);
            else      printf("%-20s %7d %6d %6d %10.2f %10s %8s\n",
                             k?"":SHAPES[si].name,O,I,S,cms,"refused","-");
            free(x); free(y);
        }
        coli_cuda_tensor_free(t);
        free(q4); free(s);
    }
    coli_cuda_shutdown();
    return 0;
}
