/* Dense GEMV: stock quant_matmul vs the K3 kernel, at REAL Kimi-K3 shapes.
 *
 * Iterating this through full-model runs was slow (4 min/load) and noisy. The
 * expert kernel only got fixed once there was a microbenchmark to iterate on;
 * this is the same tool for the dense path.
 *
 * Reports GB/s per variant and checks they agree numerically, so a "faster"
 * result can't be a wrong one. Shapes are exactly what K3 runs at decode.
 *
 *   K3_DENSE_STAGE=0|1  force shared-memory staging off/on (default: auto)
 *   ./c/tests/bench_k3_dense
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "../backend_cuda.h"
#include "../backend_cuda_k3.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                         return t.tv_sec + t.tv_nsec*1e-9; }
static uint32_t rs=11; static uint32_t rnd(void){ rs=rs*1664525u+1013904223u; return rs; }

typedef struct { const char *name; int O, I, fmt; } Shape;

static Shape SH[] = {
    {"KDA q/k/v/g  int4", 12288,  7168, 4},
    {"KDA o_proj   int4",  7168, 12288, 4},
    {"shared gate  int4",  6144,  7168, 4},
    {"lat_up       int4",  7168,  3584, 4},
    {"MLA q_b      int8", 18432,  1536, 1},
    {"lm_head      int8", 163840, 7168, 1},
};

int main(void){
    const int gs = 64;
    int dev = 0;
    if(!coli_cuda_init(&dev,1)){ fprintf(stderr,"cuda init failed\n"); return 77; }
    if(!coli_k3_init(0, 3584, 3072)){ fprintf(stderr,"k3 init failed\n"); return 77; }
    printf("%-20s %7s %6s %10s %10s %9s %9s\n",
           "shape","O","I","stock ms","k3 ms","stock GB/s","k3 GB/s");

    for(size_t si=0; si<sizeof(SH)/sizeof(*SH); si++){
        int O=SH[si].O, I=SH[si].I, fmt=SH[si].fmt;
        size_t wb = (fmt==4) ? (size_t)O*((I+1)/2) : (size_t)O*I;
        size_t nsc= (fmt==4) ? (size_t)O*((I+gs-1)/gs) : (size_t)O;
        uint8_t *w; float *sc, *x, *y1, *y2;
        if(posix_memalign((void**)&w,4096,wb) || posix_memalign((void**)&sc,4096,nsc*sizeof(float))){
            fprintf(stderr,"alloc %s\n",SH[si].name); return 1; }
        x=malloc(sizeof(float)*I); y1=malloc(sizeof(float)*O); y2=malloc(sizeof(float)*O);
        for(size_t i=0;i<wb;i++) w[i]=(uint8_t)rnd();
        for(size_t i=0;i<nsc;i++) sc[i]=0.002f;
        for(int i=0;i<I;i++) x[i]=0.01f*((i%17)-8);
        coli_k3_register(w,wb); coli_k3_register(sc,nsc*sizeof(float));

        /* stock: uploads on first call, reuses after -- so warm it first */
        ColiCudaTensor *t=NULL;
        int okstock = coli_cuda_matmul(&t,y1,x,w,sc,fmt,1,I,O,0,fmt==4?gs:0);
        double ms1=-1;
        if(okstock){
            for(int i=0;i<5;i++) coli_cuda_matmul(&t,y1,x,w,sc,fmt,1,I,O,0,fmt==4?gs:0);
            int n = (wb > (size_t)200e6) ? 10 : 60;
            double a=now();
            for(int i=0;i<n;i++) coli_cuda_matmul(&t,y1,x,w,sc,fmt,1,I,O,0,fmt==4?gs:0);
            ms1=(now()-a)*1e3/n;
        }
        double ms2=-1;
        int okk3 = coli_k3_dense(y2,x,w,sc,fmt,1,I,O,fmt==4?gs:0);
        if(okk3){
            for(int i=0;i<5;i++) coli_k3_dense(y2,x,w,sc,fmt,1,I,O,fmt==4?gs:0);
            int n = (wb > (size_t)200e6) ? 10 : 60;
            double a=now();
            for(int i=0;i<n;i++) coli_k3_dense(y2,x,w,sc,fmt,1,I,O,fmt==4?gs:0);
            ms2=(now()-a)*1e3/n;
        }
        double gb=(wb+nsc*4)/1e9;
        printf("%-20s %7d %6d %10.3f %10.3f %9.1f %9.1f",
               SH[si].name,O,I, ms1, ms2,
               ms1>0?gb/(ms1/1e3):0.0, ms2>0?gb/(ms2/1e3):0.0);
        if(okstock&&okk3){                       /* a faster wrong answer is not faster */
            double se=0,ss=0;
            for(int o=0;o<O;o++){ double d=y1[o]-y2[o]; se+=d*d; ss+=(double)y1[o]*y1[o]; }
            double rel=ss>0?sqrt(se/ss):sqrt(se);
            printf("  rel=%.1e %s", rel, rel<1e-4?"OK":"MISMATCH");
        } else printf("  %s", okk3?"stock declined":"k3 declined");
        printf("\n");
        coli_cuda_tensor_free(t);
        /* w and sc were cudaHostRegister'd above. free() alone leaves the pages
         * MAPPED, so the next shape's malloc lands on a still-registered
         * address and from there every CUDA call fails with "memory range is
         * already mapped" -- which is why this benchmark used to report only
         * its first row and decline all the rest. Unregister before freeing. */
        coli_k3_unregister(w); coli_k3_unregister(sc);
        free(w); free(sc); free(x); free(y1); free(y2);
    }
    coli_k3_shutdown();
    coli_cuda_shutdown();
    return 0;
}
