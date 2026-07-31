/* Does matmul_i4_grouped actually use the cores? Validity check for
 * bench_k3_cuda's CPU baseline: if OpenMP silently ran single-threaded, the
 * measured GPU speedup would be meaningless. One shape, S=1, nothing else.
 *
 *   OMP_NUM_THREADS=1  ./c/tests/probe_omp_scaling
 *   OMP_NUM_THREADS=20 ./c/tests/probe_omp_scaling
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../quant.h"

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

int main(void){
    const int O=12288, I=7168, gs=64, S=1, iters=20;   /* KDA q_proj */
    int rb=(I+1)/2, ng=(I+gs-1)/gs;
    uint8_t *q4=malloc((size_t)O*rb); float *s=malloc(sizeof(float)*(size_t)O*ng);
    float *x=malloc(sizeof(float)*I), *y=malloc(sizeof(float)*O);
    uint32_t r=7; for(size_t i=0;i<(size_t)O*rb;i++) q4[i]=(uint8_t)(r=r*1664525u+1013904223u);
    for(size_t i=0;i<(size_t)O*ng;i++) s[i]=0.003f;
    for(int i=0;i<I;i++) x[i]=0.01f*(i%17);
    matmul_i4_grouped(y,x,q4,s,S,I,O,gs);
    double t0=now_s();
    for(int k=0;k<iters;k++) matmul_i4_grouped(y,x,q4,s,S,I,O,gs);
    double ms=(now_s()-t0)*1e3/iters;
    size_t bytes=(size_t)O*rb + (size_t)O*ng*4;
#ifdef _OPENMP
    printf("omp_max_threads=%d  ", omp_get_max_threads());
#else
    printf("NO OPENMP  ");
#endif
    printf("%.2f ms  -> %.1f GB/s effective (%.1f MB of weights)\n",
           ms, bytes/1e6/ms, bytes/1e6);
    return 0;
}
