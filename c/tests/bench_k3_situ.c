/* Isolate the fused shared-expert gate/up+SiTU kernel BEFORE any engine run.
 * Reference = two coli_k3_dense calls + the CPU situf_, exactly what
 * moe_forward does. Checks S=1 and S>1 separately, because the S>1 token loop
 * and the kernel are different suspects. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "../backend_cuda_k3.h"
static uint32_t rs=9; static uint32_t rnd(void){ rs=rs*1664525u+1013904223u; return rs; }
static float sigmoidf_(float x){ return 1.f/(1.f+expf(-x)); }
static float situf_(float g,float u,float b1,float b2){
    return b1*tanhf(g/b1)*sigmoidf_(g) * b2*tanhf(u/b2);
}
int main(void){
    const int I=7168, O=6144, gs=64;      /* real shared-expert shape */
    const float b1=4.f, b2=25.f;
    if(!coli_k3_init(0,3584,3072)){ fprintf(stderr,"init failed\n"); return 77; }
    size_t wb=(size_t)O*((I+1)/2), nsc=(size_t)O*((I+gs-1)/gs);
    uint8_t *w1,*w3; float *s1,*s3;
    if(posix_memalign((void**)&w1,4096,wb)||posix_memalign((void**)&w3,4096,wb)||
       posix_memalign((void**)&s1,4096,nsc*4)||posix_memalign((void**)&s3,4096,nsc*4)){
        fprintf(stderr,"alloc\n"); return 1; }
    for(size_t i=0;i<wb;i++){ w1[i]=(uint8_t)rnd(); w3[i]=(uint8_t)rnd(); }
    for(size_t i=0;i<nsc;i++){ s1[i]=0.002f; s3[i]=0.0017f; }
    coli_k3_register(w1,wb); coli_k3_register(w3,wb);
    coli_k3_register(s1,nsc*4); coli_k3_register(s3,nsc*4);

    int Ss[2]={1,8};
    for(int si=0; si<2; si++){
        int S=Ss[si];
        float *x=malloc((size_t)S*I*4), *ref=malloc((size_t)S*O*4);
        float *got=malloc((size_t)S*O*4), *g=malloc((size_t)S*O*4), *u=malloc((size_t)S*O*4);
        for(int t=0;t<S;t++) for(int i=0;i<I;i++) x[(size_t)t*I+i]=0.01f*(((i+3*t)%23)-11);
        /* reference: same calls moe_forward makes */
        int ok1=1;
        for(int t=0;t<S;t++){
            ok1 &= coli_k3_dense(g+(size_t)t*O,x+(size_t)t*I,w1,s1,4,1,I,O,gs);
            ok1 &= coli_k3_dense(u+(size_t)t*O,x+(size_t)t*I,w3,s3,4,1,I,O,gs);
        }
        for(size_t i=0;i<(size_t)S*O;i++) ref[i]=situf_(g[i],u[i],b1,b2);
        int ok2=coli_k3_gate_up_situ(got,x,w1,s1,w3,s3,S,I,O,gs,b1,b2);
        if(!ok1||!ok2){ printf("S=%-2d  DECLINED (ref=%d fused=%d)\n",S,ok1,ok2); continue; }
        double worst=0,ss=0,se=0; size_t wi=0;
        for(size_t i=0;i<(size_t)S*O;i++){
            double d=fabs((double)ref[i]-(double)got[i]);
            if(d>worst){ worst=d; wi=i; }
            se+=d*d; ss+=(double)ref[i]*ref[i];
        }
        printf("S=%-2d  worst |diff| = %.6g at %zu (ref=%.6g fused=%.6g)  rel=%.2e\n",
               S, worst, wi, ref[wi], got[wi], ss>0?sqrt(se/ss):0.0);
        free(x);free(ref);free(got);free(g);free(u);
    }
    coli_k3_unregister(w1); coli_k3_unregister(w3);
    coli_k3_unregister(s1); coli_k3_unregister(s3);
    coli_k3_shutdown(); return 0;
}
