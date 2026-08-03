/* Throughput of the packed-1-bit routed-expert kernel at REAL Kimi-K3 dims.
 *
 * This is the kernel the four-Spark B1 config actually runs (K3_W1_DIR), and
 * PROF2 attributes 4.22 s per 100 tokens to it -- ~42 ms/token to move roughly
 * 1.8 GB of expert slots, i.e. ~43 GB/s against a measured 235 GB/s achievable
 * read bandwidth on this part. bench_k3_w2 only covers the 2-bit kernel, so
 * this measures the 1-bit one and splits the cost three ways:
 *
 *   hot   -- one slot reused, L2-resident: kernel cost with no memory pressure
 *   cold  -- cycling distinct slots, which is what the engine does
 *   device-- same kernel, weights in cudaMalloc memory instead of zero-copy
 *            host memory, isolating the mapped-host path from the kernel
 *
 * K3_BL / K3_BI override latent/inter; a tiny shape exposes the fixed per-call
 * cost (two launches, two memcpyAsync, one stream sync), measured at 11 us.
 *
 *   make CUDA=1 backend_cuda_k3.o && gcc -O3 -march=native -DCOLI_CUDA \
 *     tests/bench_k3_w1.c backend_cuda_k3.o -o tests/bench_k3_w1 -lm \
 *     -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart -lstdc++
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include "../backend_cuda_k3.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                         return t.tv_sec + t.tv_nsec*1e-9; }
static uint32_t rs=7; static uint32_t rnd(void){ rs=rs*1664525u+1013904223u; return rs; }

int main(void){
    /* Overridable so a tiny shape can expose the fixed per-call cost: two
     * kernel launches, two memcpyAsync and one cudaStreamSynchronize. */
    int latent = getenv("K3_BL") ? atoi(getenv("K3_BL")) : 3584;
    int inter  = getenv("K3_BI") ? atoi(getenv("K3_BI")) : 3072;
    if(!coli_k3_init(0,latent,inter)){ fprintf(stderr,"init failed\n"); return 77; }
    size_t w1p_n=(size_t)inter*(latent/8), w1s_n=(size_t)inter*(latent/32);
    size_t w2p_n=(size_t)latent*(inter/8), w2s_n=(size_t)latent*(inter/32);
    size_t slot=2*(w1p_n+w1s_n)+w2p_n+w2s_n;
    uint8_t *w1p,*w1s,*w2p,*w2s,*w3p,*w3s; float *z,*hz;
    if(posix_memalign((void**)&w1p,4096,w1p_n)||posix_memalign((void**)&w1s,4096,w1s_n)||
       posix_memalign((void**)&w2p,4096,w2p_n)||posix_memalign((void**)&w2s,4096,w2s_n)||
       posix_memalign((void**)&w3p,4096,w1p_n)||posix_memalign((void**)&w3s,4096,w1s_n)){
        fprintf(stderr,"alloc\n"); return 1; }
    z=malloc(sizeof(float)*latent); hz=malloc(sizeof(float)*latent);
    for(size_t i=0;i<w1p_n;i++){ w1p[i]=(uint8_t)rnd(); w3p[i]=(uint8_t)rnd(); }
    for(size_t i=0;i<w2p_n;i++) w2p[i]=(uint8_t)rnd();
    for(size_t i=0;i<w1s_n;i++){ w1s[i]=125; w3s[i]=125; }
    for(size_t i=0;i<w2s_n;i++) w2s[i]=125;
    for(int i=0;i<latent;i++) z[i]=0.01f*(i%13);
    coli_k3_register(w1p,w1p_n); coli_k3_register(w1s,w1s_n);
    coli_k3_register(w2p,w2p_n); coli_k3_register(w2s,w2s_n);
    coli_k3_register(w3p,w1p_n); coli_k3_register(w3s,w1s_n);

    printf("expert slot %.3f MiB\n", slot/1048576.0);
    for(int i=0;i<20;i++) coli_k3_expert_w1(w1p,w1s,w2p,w2s,w3p,w3s,hz,z,latent,inter,4.f,25.f);
    const int N=300;
    double t0=now();
    for(int i=0;i<N;i++) coli_k3_expert_w1(w1p,w1s,w2p,w2s,w3p,w3s,hz,z,latent,inter,4.f,25.f);
    double ms=(now()-t0)*1e3/N;
    printf("hot  (1 slot reused)  %.3f ms/expert  %6.1f GB/s\n", ms, (slot/1e9)/(ms/1e3));

    void *d1p=coli_k3_devcopy(w1p,w1p_n), *d1s=coli_k3_devcopy(w1s,w1s_n);
    void *d2p=coli_k3_devcopy(w2p,w2p_n), *d2s=coli_k3_devcopy(w2s,w2s_n);
    void *d3p=coli_k3_devcopy(w3p,w1p_n), *d3s=coli_k3_devcopy(w3s,w1s_n);
    if(d1p&&d1s&&d2p&&d2s&&d3p&&d3s){
        for(int i=0;i<20;i++) coli_k3_expert_w1(d1p,d1s,d2p,d2s,d3p,d3s,hz,z,latent,inter,4.f,25.f);
        double t1=now();
        for(int i=0;i<N;i++) coli_k3_expert_w1(d1p,d1s,d2p,d2s,d3p,d3s,hz,z,latent,inter,4.f,25.f);
        double ms2=(now()-t1)*1e3/N;
        printf("hot  device-resident  %.3f ms/expert  %6.1f GB/s\n", ms2, (slot/1e9)/(ms2/1e3));
    }
    {   /* COLD: cycle distinct slots so nothing stays cache-resident -- this is
         * the number that should match the engine's `expert` PROF2 term. */
        const int NE=48;
        uint8_t **c1p=malloc(NE*sizeof(void*)), **c1s=malloc(NE*sizeof(void*));
        uint8_t **c2p=malloc(NE*sizeof(void*)), **c2s=malloc(NE*sizeof(void*));
        uint8_t **c3p=malloc(NE*sizeof(void*)), **c3s=malloc(NE*sizeof(void*));
        int okc=1;
        for(int j=0;j<NE&&okc;j++){
            if(posix_memalign((void**)&c1p[j],4096,w1p_n)||posix_memalign((void**)&c1s[j],4096,w1s_n)||
               posix_memalign((void**)&c2p[j],4096,w2p_n)||posix_memalign((void**)&c2s[j],4096,w2s_n)||
               posix_memalign((void**)&c3p[j],4096,w1p_n)||posix_memalign((void**)&c3s[j],4096,w1s_n)){ okc=0; break; }
            memcpy(c1p[j],w1p,w1p_n); memcpy(c1s[j],w1s,w1s_n);
            memcpy(c2p[j],w2p,w2p_n); memcpy(c2s[j],w2s,w2s_n);
            memcpy(c3p[j],w3p,w1p_n); memcpy(c3s[j],w3s,w1s_n);
            coli_k3_register(c1p[j],w1p_n); coli_k3_register(c1s[j],w1s_n);
            coli_k3_register(c2p[j],w2p_n); coli_k3_register(c2s[j],w2s_n);
            coli_k3_register(c3p[j],w1p_n); coli_k3_register(c3s[j],w1s_n);
        }
        if(okc){
            for(int i=0;i<NE;i++) coli_k3_expert_w1(c1p[i],c1s[i],c2p[i],c2s[i],c3p[i],c3s[i],hz,z,latent,inter,4.f,25.f);
            double tc=now(); const int K=NE*4;
            for(int i=0;i<K;i++){ int j=i%NE;
                coli_k3_expert_w1(c1p[j],c1s[j],c2p[j],c2s[j],c3p[j],c3s[j],hz,z,latent,inter,4.f,25.f); }
            double msc=(now()-tc)*1e3/K;
            printf("cold (%d slots cycled) %.3f ms/expert  %6.1f GB/s\n", NE, msc, (slot/1e9)/(msc/1e3));
            printf("-> 4 experts x 92 layers = %.2f s/100 tokens\n", 100*4*92*msc/1e3);
        }
    }
    coli_k3_shutdown();
    return 0;
}
