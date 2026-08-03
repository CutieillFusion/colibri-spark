/* Unit tests for the tensor-parallel head-slicing support in kimi_k3.c:
 * the bf16 KV codec, the row compactor, and the head-shard formula.
 *
 * All three share a failure mode that motivated these tests: they go wrong
 * QUIETLY. A KV codec that mangles NaN yields plausible finite numbers; a
 * compactor that copies the wrong rows yields fluent text from the wrong
 * weights; a shard formula that disagrees with the forward makes a rank
 * multiply weights it did not load. None of them crash, so an end-to-end run
 * cannot be trusted to reveal them -- exactly what happened during bring-up,
 * where a dangling scale pointer surfaced only via an expert-call count.
 *
 * Build:  make tests/test_k3_slice && ./tests/test_k3_slice
 */
#define main k3_main_unused
#include "../kimi_k3.c"
#undef main

static int g_fail = 0;
#define CHECK(cond, ...) do{ if(!(cond)){ \
    fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); g_fail++; } }while(0)

/* ---------- 1. bf16 KV codec ---------- */

static void test_kv_special(void){
    /* Inf/NaN must survive the round trip. The original (u+0x8000)>>16 carried
     * out of the mantissa into exponent and sign: 0x7fffffff decoded as -0,
     * 0xffffffff as +0 and 0x7f800001 as +Inf, silently turning a numerical
     * failure into finite-looking output. */
    const uint32_t nans[] = { 0x7fffffffu, 0xffffffffu, 0x7f800001u,
                              0xff800001u, 0x7fc00000u };
    for(unsigned i=0;i<sizeof nans/sizeof*nans;i++){
        union{uint32_t u;float f;}b; b.u=nans[i];
        float r = kv_dec(kv_enc(b.f));
        CHECK(isnan(r), "NaN 0x%08x decoded to %g, not NaN", nans[i], r);
    }
    union{uint32_t u;float f;}pi,ni; pi.u=0x7f800000u; ni.u=0xff800000u;
    float rp=kv_dec(kv_enc(pi.f)), rn=kv_dec(kv_enc(ni.f));
    CHECK(isinf(rp) && !signbit(rp), "+Inf decoded to %g", rp);
    CHECK(isinf(rn) &&  signbit(rn), "-Inf decoded to %g", rn);
}

static void test_kv_exact(void){
    /* Values already representable in bf16 must be returned bit-exact --
     * anything else means the rounding term is corrupting clean inputs. */
    const float ex[] = { 0.f, -0.f, 1.f, -1.f, 0.5f, -0.5f, 2.f, 256.f, -1024.f };
    for(unsigned i=0;i<sizeof ex/sizeof*ex;i++){
        float r=kv_dec(kv_enc(ex[i]));
        CHECK(memcmp(&r,&ex[i],sizeof r)==0, "%g did not round-trip exactly (got %g)", ex[i], r);
    }
}

static void test_kv_accuracy(void){
    /* bf16 keeps 8 mantissa bits, so worst-case relative error is 2^-9 with
     * round-to-nearest. Also assert the error is UNBIASED: round-half-up (the
     * first implementation) drifts upward on ties, which accumulates over a
     * cache that is written every token. */
    double se=0, sx=0, bias=0; int n=0; float worst=0;
    unsigned seed=12345u;
    for(int i=0;i<400000;i++){
        seed = seed*1103515245u + 12345u;
        float x = ((float)(seed>>8) / (float)(1u<<24)) * 8.f - 4.f;
        float r = kv_dec(kv_enc(x));
        double e = (double)r - x;
        se += e*e; sx += (double)x*x; bias += e; n++;
        if(fabsf(x) > 1e-6f){ float rel=(float)(fabs(e)/fabsf(x)); if(rel>worst) worst=rel; }
    }
    double relrms = sqrt(se/sx);
    CHECK(relrms < 3e-3, "rel RMS %.6f exceeds bf16 expectation", relrms);
    CHECK(worst <= 1.0f/256.0f + 1e-6f, "worst relative error %.6f exceeds 2^-8", worst);
    CHECK(fabs(bias/n) < 1e-5, "mean signed error %.3e suggests biased rounding", bias/n);
}

/* ---------- 2. w_keep_rows ---------- */

static void test_keep_rows_fmt(int fmt){
    /* Build a tensor whose every byte encodes its own (row,col) so a
     * misaligned copy cannot coincidentally look correct. */
    const int O=16, I=128, gs=64, R0=4, NR=6;
    const int rb=I/2, ng=I/gs;
    W w; memset(&w,0,sizeof w);
    w.O=O; w.I=I; w.fmt=fmt; w.gs=gs;
    if(fmt==0){
        w.f=falloc((int64_t)O*I);
        for(int r=0;r<O;r++) for(int c=0;c<I;c++) w.f[r*I+c]=(float)(r*1000+c);
    } else if(fmt==1){
        w.q8=malloc((size_t)O*I); w.s=falloc(O);
        for(int r=0;r<O;r++){ w.s[r]=(float)(r+1);
            for(int c=0;c<I;c++) w.q8[r*I+c]=(int8_t)((r*7+c)&0x7f); }
    } else {
        w.q4=malloc((size_t)O*rb); w.s=falloc((int64_t)O*ng);
        for(int r=0;r<O;r++){
            for(int g=0;g<ng;g++) w.s[r*ng+g]=(float)(r*10+g);
            for(int c=0;c<rb;c++) w.q4[r*rb+c]=(uint8_t)((r*13+c)&0xff); }
    }
    w_keep_rows(&w,R0,NR);
    CHECK(w.O==NR, "fmt=%d: O is %d, expected %d", fmt, w.O, NR);
    CHECK(w.I==I,  "fmt=%d: I changed to %d", fmt, w.I);
    for(int r=0;r<NR;r++){
        int src=R0+r;
        if(fmt==0){
            for(int c=0;c<I;c++)
                CHECK(w.f[r*I+c]==(float)(src*1000+c),
                      "fmt=0: row %d col %d is %g, expected %g",
                      r,c,w.f[r*I+c],(double)(src*1000+c));
        } else if(fmt==1){
            CHECK(w.s[r]==(float)(src+1), "fmt=1: scale %d is %g, expected %g",
                  r,w.s[r],(double)(src+1));
            for(int c=0;c<I;c++)
                CHECK(w.q8[r*I+c]==(int8_t)((src*7+c)&0x7f),
                      "fmt=1: row %d col %d mismatched", r, c);
        } else {
            for(int g=0;g<ng;g++)
                CHECK(w.s[r*ng+g]==(float)(src*10+g),
                      "fmt=4: scale row %d group %d is %g, expected %g",
                      r,g,w.s[r*ng+g],(double)(src*10+g));
            for(int c=0;c<rb;c++)
                CHECK(w.q4[r*rb+c]==(uint8_t)((src*13+c)&0xff),
                      "fmt=4: row %d byte %d mismatched", r, c);
        }
    }
    w_free_host(&w);
}

static void test_keep_rows_identity(void){
    /* NR==O must be a no-op, including leaving the buffer pointer usable --
     * this is the path every single-node load takes. */
    const int O=8, I=64;
    W w; memset(&w,0,sizeof w); w.O=O; w.I=I; w.fmt=0;
    w.f=falloc((int64_t)O*I);
    for(int i=0;i<O*I;i++) w.f[i]=(float)i;
    w_keep_rows(&w,0,O);
    CHECK(w.O==O, "identity compaction changed O to %d", w.O);
    for(int i=0;i<O*I;i++) CHECK(w.f[i]==(float)i, "identity compaction altered element %d", i);
    w_free_host(&w);
}

/* ---------- 3. k3_head_shard ---------- */

static void test_head_shard_partition(void){
    /* The ranks' head ranges must EXACTLY tile [0,H): every head owned once,
     * none twice. A gap means weights nobody multiplies; an overlap means a
     * head counted twice in the all-reduce. Both produce wrong numbers rather
     * than a crash, and neither is visible in a coherence check. */
    const int worlds[] = {1,2,3,4,8};
    const int heads[]  = {96,64,24,7,1};
    int saved_w=g_net_world, saved_r=g_net_rank, saved_tp=g_k3_tp_attn;
    g_k3_tp_attn = 1;
    for(unsigned wi=0; wi<sizeof worlds/sizeof*worlds; wi++){
        for(unsigned hi=0; hi<sizeof heads/sizeof*heads; hi++){
            int W_=worlds[wi], H=heads[hi];
            int *own=calloc((size_t)H,sizeof(int));
            g_net_world=W_;
            for(int r=0;r<W_;r++){
                g_net_rank=r;
                int h0,hn; k3_head_shard(H,&h0,&hn);
                CHECK(h0>=0 && hn>=0 && h0+hn<=H,
                      "world=%d H=%d rank=%d: range [%d,%d) out of bounds",W_,H,r,h0,h0+hn);
                for(int h=h0;h<h0+hn;h++) own[h]++;
            }
            for(int h=0;h<H;h++)
                CHECK(own[h]==1, "world=%d H=%d: head %d owned by %d ranks (expected 1)",
                      W_,H,h,own[h]);
            free(own);
        }
    }
    /* TP disabled: every rank must see the whole range regardless of world. */
    g_k3_tp_attn=0; g_net_world=4; g_net_rank=2;
    int h0,hn; k3_head_shard(96,&h0,&hn);
    CHECK(h0==0 && hn==96, "TP off: expected [0,96), got [%d,%d)",h0,h0+hn);
    g_net_world=saved_w; g_net_rank=saved_r; g_k3_tp_attn=saved_tp;
}

static void test_head_shard_matches_forward(void){
    /* kda_forward/mla_forward derive their split from the same formula; if the
     * loader and the forward ever diverge, a rank loads one slice and
     * multiplies another. Recompute the forward's expression independently
     * here so a change to either side breaks this test. */
    int saved_w=g_net_world, saved_r=g_net_rank, saved_tp=g_k3_tp_attn;
    g_k3_tp_attn=1;
    for(int W_=1; W_<=8; W_++){
        g_net_world=W_;
        for(int H=1; H<=128; H++){
            for(int r=0;r<W_;r++){
                g_net_rank=r;
                int h0,hn; k3_head_shard(H,&h0,&hn);
                /* verbatim from the forwards */
                int e0=0, e1=H;
                if(W_>1){ int per=(H+W_-1)/W_; e0=r*per; e1=e0+per;
                          if(e1>H) e1=H; if(e0>H) e0=H; }
                CHECK(h0==e0 && hn==e1-e0,
                      "world=%d H=%d rank=%d: loader [%d,%d) vs forward [%d,%d)",
                      W_,H,r,h0,h0+hn,e0,e1);
            }
        }
    }
    g_net_world=saved_w; g_net_rank=saved_r; g_k3_tp_attn=saved_tp;
}

/* ---------- 4. fused B=1 KDA control projections ---------- */

static void test_kda_control_b1_exact(void){
    /* The optimized path shares one OpenMP team for f_a, beta and f_b, and may
     * execute concurrently with CUDA q/k/v/g. Its per-row arithmetic must stay
     * byte-identical to the original three strict-f32 matmul calls. */
    enum { D=64, HD=8, PN=12, HN=3, P=20, H=8, P0=4, H0=2 };
    Kda a; memset(&a,0,sizeof a);
    a.fa.f=falloc((int64_t)HD*D); a.fb.f=falloc((int64_t)PN*HD);
    a.bp.f=falloc((int64_t)HN*D);
    float *x=falloc(D), *tr=falloc(HD), *tf=falloc(HD);
    float *gr=fcalloc(P), *gf=fcalloc(P), *br=fcalloc(H), *bf=fcalloc(H);
    for(int i=0;i<D;i++) x[i]=(float)((i*17)%31-15)/19.f;
    for(int i=0;i<HD*D;i++) a.fa.f[i]=(float)((i*13)%37-18)/23.f;
    for(int i=0;i<PN*HD;i++) a.fb.f[i]=(float)((i*11)%29-14)/17.f;
    for(int i=0;i<HN*D;i++) a.bp.f[i]=(float)((i*7)%41-20)/27.f;
    matmul(tr,x,a.fa.f,1,D,HD);
    matmul(gr+P0,tr,a.fb.f,1,HD,PN);
    matmul(br+H0,x,a.bp.f,1,D,HN);
    KdaCtrlJob j={&a,x,tf,gf,bf,D,HD,P0,PN,H0,HN};
    kda_control_b1(&j);
    CHECK(!memcmp(tr,tf,sizeof(float)*HD),"KDA f_a fused output changed");
    CHECK(!memcmp(gr,gf,sizeof(float)*P), "KDA f_b fused output changed");
    CHECK(!memcmp(br,bf,sizeof(float)*H), "KDA beta fused output changed");
    free(x);free(tr);free(tf);free(gr);free(gf);free(br);free(bf);
    w_free_host(&a.fa);w_free_host(&a.fb);w_free_host(&a.bp);
}

int main(void){
    test_kv_special();
    test_kv_exact();
    test_kv_accuracy();
    test_keep_rows_fmt(0);
    test_keep_rows_fmt(1);
    test_keep_rows_fmt(4);
    test_keep_rows_identity();
    test_head_shard_partition();
    test_head_shard_matches_forward();
    test_kda_control_b1_exact();
    if(g_fail){ fprintf(stderr,"test_k3_slice: %d failure(s)\n",g_fail); return 1; }
    printf("test_k3_slice: all checks passed\n");
    return 0;
}
