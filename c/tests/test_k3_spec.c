/* Unit tests for Kimi-K3 speculative decoding's two quiet failure modes:
 * state rollback/replay and greedy accept-prefix indexing.
 *
 * The real KDA state is ~434 MB and needs checkpoint weights to advance, so
 * this test builds a dimensionally small Model and drives the same three
 * sequential state classes with a deterministic toy token step. Restore plus
 * replay must be byte-identical to plain one-token-at-a-time advancement,
 * including KDA recurrence storage, all three convolution windows, the MLA
 * prefix retained by logical truncation, and state_pos itself.
 *
 * Build: make tests/test_k3_spec && ./tests/test_k3_spec
 */
#define main k3_main_unused
#include "../kimi_k3.c"
#undef main

static int g_fail=0;
#define CHECK(c,...) do{ if(!(c)){ fprintf(stderr,"FAIL %s:%d: ",__FILE__,__LINE__); \
    fprintf(stderr,__VA_ARGS__); fputc('\n',stderr); g_fail++; } }while(0)

static void toy_init(Model *m){
    memset(m,0,sizeof(*m));
    Cfg *c=&m->c;
    c->n_layers=3; c->kda_heads=2; c->kda_hd=3; c->kda_proj=6; c->conv_k=4;
    c->kv_lora=5; c->qk_rope=2;
    c->is_kda[0]=1; c->is_kda[2]=1;
    m->L=calloc((size_t)c->n_layers,sizeof(Layer));
    m->kstate=calloc((size_t)c->n_layers,sizeof(float*));
    m->cwq=calloc((size_t)c->n_layers,sizeof(float*));
    m->cwk=calloc((size_t)c->n_layers,sizeof(float*));
    m->cwv=calloc((size_t)c->n_layers,sizeof(float*));
    m->Lc=calloc((size_t)c->n_layers,sizeof(kvq*));
    m->Rc=calloc((size_t)c->n_layers,sizeof(kvq*));
    m->max_t=16;
    for(int l=0;l<c->n_layers;l++){
        m->L[l].kda=c->is_kda[l];
        if(c->is_kda[l]){
            size_t nk=(size_t)c->kda_heads*c->kda_hd*c->kda_hd;
            size_t nc=(size_t)c->kda_proj*c->conv_k;
            m->kstate[l]=falloc(nk); m->cwq[l]=falloc(nc);
            m->cwk[l]=falloc(nc); m->cwv[l]=falloc(nc);
            for(size_t i=0;i<nk;i++) m->kstate[l][i]=(float)(100*l+(int)i)*0.01f;
            for(size_t i=0;i<nc;i++){
                m->cwq[l][i]=(float)(200*l+(int)i)*0.02f;
                m->cwk[l][i]=(float)(300*l+(int)i)*0.03f;
                m->cwv[l][i]=(float)(400*l+(int)i)*0.04f;
            }
        } else {
            m->Lc[l]=calloc((size_t)m->max_t*c->kv_lora,sizeof(kvq));
            m->Rc[l]=calloc((size_t)m->max_t*c->qk_rope,sizeof(kvq));
        }
    }
}

static void toy_free(Model *m){
    for(int l=0;l<m->c.n_layers;l++){
        free(m->kstate[l]); free(m->cwq[l]); free(m->cwk[l]); free(m->cwv[l]);
        free(m->Lc[l]); free(m->Rc[l]);
    }
    free(m->kstate); free(m->cwq); free(m->cwk); free(m->cwv);
    free(m->Lc); free(m->Rc); free(m->L);
}

static void toy_step(Model *m, int token){
    Cfg *c=&m->c; int pos=m->state_pos;
    for(int l=0;l<c->n_layers;l++){
        if(m->L[l].kda){
            size_t nk=(size_t)c->kda_heads*c->kda_hd*c->kda_hd;
            for(size_t i=0;i<nk;i++)
                m->kstate[l][i]=m->kstate[l][i]*0.75f+(float)(token+7*l+(int)i)*0.125f;
            float *w[3]={m->cwq[l],m->cwk[l],m->cwv[l]};
            for(int q=0;q<3;q++) for(int d=0;d<c->kda_proj;d++){
                float *p=w[q]+(size_t)d*c->conv_k;
                memmove(p,p+1,(size_t)(c->conv_k-1)*sizeof(float));
                p[c->conv_k-1]=(float)(token*13+l*5+q*3+d);
            }
        } else {
            for(int i=0;i<c->kv_lora;i++) m->Lc[l][(size_t)pos*c->kv_lora+i]=kv_enc((float)(token*11+i));
            for(int i=0;i<c->qk_rope;i++) m->Rc[l][(size_t)pos*c->qk_rope+i]=kv_enc((float)(token*17+i));
        }
    }
    m->state_pos++;
}

static void check_state_equal(const Model *a, const Model *b){
    const Cfg *c=&a->c;
    CHECK(a->state_pos==b->state_pos,"state_pos %d != %d",a->state_pos,b->state_pos);
    for(int l=0;l<c->n_layers;l++){
        if(a->L[l].kda){
            size_t nk=(size_t)c->kda_heads*c->kda_hd*c->kda_hd*sizeof(float);
            size_t nc=(size_t)c->kda_proj*c->conv_k*sizeof(float);
            CHECK(!memcmp(a->kstate[l],b->kstate[l],nk),"layer %d KDA state differs",l);
            CHECK(!memcmp(a->cwq[l],b->cwq[l],nc),"layer %d q window differs",l);
            CHECK(!memcmp(a->cwk[l],b->cwk[l],nc),"layer %d k window differs",l);
            CHECK(!memcmp(a->cwv[l],b->cwv[l],nc),"layer %d v window differs",l);
        } else {
            size_t nl=(size_t)a->state_pos*c->kv_lora*sizeof(kvq);
            size_t nr=(size_t)a->state_pos*c->qk_rope*sizeof(kvq);
            CHECK(!memcmp(a->Lc[l],b->Lc[l],nl),"layer %d MLA L prefix differs",l);
            CHECK(!memcmp(a->Rc[l],b->Rc[l],nr),"layer %d MLA R prefix differs",l);
        }
    }
}

static void test_restore_replay(void){
    Model got,plain; toy_init(&got); toy_init(&plain);
    toy_step(&got,3); toy_step(&got,5);
    toy_step(&plain,3); toy_step(&plain,5);
    K3StateSnapshot s; k3_snapshot_init(&s,&got); k3_snapshot_take(&s,&got);
    /* rejected verify span */
    toy_step(&got,91); toy_step(&got,92); toy_step(&got,93); toy_step(&got,94);
    k3_snapshot_restore(&got,&s);
    CHECK(got.state_pos==2,"restore did not logically truncate MLA to 2 (got %d)",got.state_pos);
    /* n=3 accepted tokens: replay versus three ordinary C=1 steps */
    const int accepted[]={7,11,13};
    for(unsigned i=0;i<sizeof accepted/sizeof*accepted;i++){
        toy_step(&got,accepted[i]); toy_step(&plain,accepted[i]);
    }
    check_state_equal(&got,&plain);
    k3_snapshot_free(&s); toy_free(&got); toy_free(&plain);
}

static void test_accept_prefix(void){
    enum { V=6, R=4 }; float rows[R*V];
    for(int i=0;i<R*V;i++) rows[i]=-100.f;
    rows[0*V+2]=9.f; rows[1*V+3]=8.f; rows[2*V+4]=7.f; rows[3*V+1]=6.f;
    int draft[]={2,3,1};
    CHECK(k3_accept_prefix(rows,V,draft,3)==2,"expected two-token accepted prefix");
    draft[0]=5;
    CHECK(k3_accept_prefix(rows,V,draft,3)==0,"first mismatch must accept zero");
    int all[]={2,3,4};
    CHECK(k3_accept_prefix(rows,V,all,3)==3,"all matching draft tokens must be accepted");
    CHECK(k3_accept_prefix(rows,V,all,0)==0,"empty draft must accept zero");
}

static void test_prompt_lookup(void){
    int ctx[]={8,9,1,2,3,40,41,1,2,3}, out[4];
    int n=k3_prompt_lookup(NULL,ctx,10,out,2);
    CHECK(n==2,"prompt lookup returned %d tokens, expected 2",n);
    CHECK(n>=2&&out[0]==40&&out[1]==41,"prompt lookup chose wrong continuation");
}

int main(void){
    int sw=g_net_world,sr=g_net_rank,st=g_k3_tp_attn;
    g_net_world=1; g_net_rank=0; g_k3_tp_attn=1;
    test_restore_replay(); test_accept_prefix(); test_prompt_lookup();
    g_net_world=sw; g_net_rank=sr; g_k3_tp_attn=st;
    if(g_fail){ fprintf(stderr,"test_k3_spec: %d failure(s)\n",g_fail); return 1; }
    puts("test_k3_spec: rollback/replay, accept-prefix, prompt lookup passed");
    return 0;
}
