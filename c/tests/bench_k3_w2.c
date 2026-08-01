/* Throughput of the packed-2-bit expert kernel at REAL Kimi-K3 dimensions.
 *
 * Full-model A/Bs kept getting confounded (page-cache warmth, a concurrent
 * 766 GB pack, the shard extraction). This times the kernel alone: one call
 * reads exactly one expert slot, 9,289,728 B, so GB/s falls straight out.
 *
 * Reference points measured on this GPU: the DENSE int4 GEMV path achieves
 * ~59 GB/s, and the original block-per-row expert kernel implied ~15 GB/s in
 * the full-model run. Peak is 273 GB/s.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "../backend_cuda_k3.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                         return t.tv_sec + t.tv_nsec*1e-9; }
static uint32_t rs=7; static uint32_t rnd(void){ rs=rs*1664525u+1013904223u; return rs; }

int main(void){
    const int latent=3584, inter=3072;           /* K3 LatentMoE */
    if(!coli_k3_init(0,latent,inter)){ fprintf(stderr,"init failed\n"); return 77; }
    size_t w1p_n=(size_t)inter*(latent/4), w1s_n=(size_t)inter*(latent/32);
    size_t w2p_n=(size_t)latent*(inter/4), w2s_n=(size_t)latent*(inter/32);
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
    /* mirror the engine: slots are registered once for zero-copy */
    coli_k3_register(w1p,w1p_n); coli_k3_register(w1s,w1s_n);
    coli_k3_register(w2p,w2p_n); coli_k3_register(w2s,w2s_n);
    coli_k3_register(w3p,w1p_n); coli_k3_register(w3s,w1s_n);

    for(int i=0;i<20;i++) coli_k3_expert_w2(w1p,w1s,w2p,w2s,w3p,w3s,hz,z,latent,inter,4.f,25.f);
    const int N=300;
    double t0=now();
    for(int i=0;i<N;i++) coli_k3_expert_w2(w1p,w1s,w2p,w2s,w3p,w3s,hz,z,latent,inter,4.f,25.f);
    double ms=(now()-t0)*1e3/N;
    printf("expert slot %.2f MiB | %.3f ms/expert | %.1f GB/s\n",
           slot/1048576.0, ms, (slot/1e9)/(ms/1e3));
    printf("-> 16 experts x 92 layers = %.2f s/token of expert compute\n", 16*92*ms/1e3);
    /* Same kernel, DEVICE-resident weights: isolates zero-copy from the kernel. */
    void *d1p=coli_k3_devcopy(w1p,w1p_n), *d1s=coli_k3_devcopy(w1s,w1s_n);
    void *d2p=coli_k3_devcopy(w2p,w2p_n), *d2s=coli_k3_devcopy(w2s,w2s_n);
    void *d3p=coli_k3_devcopy(w3p,w1p_n), *d3s=coli_k3_devcopy(w3s,w1s_n);
    if(d1p&&d1s&&d2p&&d2s&&d3p&&d3s){
        for(int i=0;i<20;i++) coli_k3_expert_w2(d1p,d1s,d2p,d2s,d3p,d3s,hz,z,latent,inter,4.f,25.f);
        double t1=now();
        for(int i=0;i<N;i++) coli_k3_expert_w2(d1p,d1s,d2p,d2s,d3p,d3s,hz,z,latent,inter,4.f,25.f);
        double ms2=(now()-t1)*1e3/N;
        printf("device-resident:  %.3f ms/expert | %.1f GB/s\n", ms2, (slot/1e9)/(ms2/1e3));
    }
    /* Batched: 16 experts, ONE sync. Isolates fixed per-call overhead. */
    {
        const int B=16;
        const void *p1[16],*s1[16],*p2[16],*s2[16],*p3[16],*s3[16];
        float *hzb=malloc(sizeof(float)*(size_t)B*latent);
        for(int j=0;j<B;j++){ p1[j]=w1p; s1[j]=w1s; p2[j]=w2p; s2[j]=w2s; p3[j]=w3p; s3[j]=w3s; }
        for(int i=0;i<5;i++) coli_k3_expert_batch_w2(p1,s1,p2,s2,p3,s3,B,hzb,z,latent,inter,4.f,25.f);
        double t2=now(); const int M=40;
        for(int i=0;i<M;i++) coli_k3_expert_batch_w2(p1,s1,p2,s2,p3,s3,B,hzb,z,latent,inter,4.f,25.f);
        double ms3=(now()-t2)*1e3/M/B;
        printf("batched x%d:       %.3f ms/expert | %.1f GB/s\n", B, ms3, (slot/1e9)/(ms3/1e3));
        printf("-> 16 experts x 92 layers = %.2f s/token of expert compute\n", 16*92*ms3/1e3);
        free(hzb);
    }
    coli_k3_shutdown();
    return 0;
}
