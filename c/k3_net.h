#ifndef COLIBRI_K3_NET_H
#define COLIBRI_K3_NET_H
/* Hierarchical TCP all-reduce for multi-node Kimi-K3.
 *
 * Why sockets and not MPI/NCCL: the payload is one MoE latent accumulator per
 * layer -- 3584 floats = 14 KB -- so ~93 reductions and ~1.3 MB per token. On
 * any of these links that is negligible wire time; what costs is round trips.
 * A dependency-free reduction keeps Colibri's "pure C, zero deps" property and
 * is within a few hundred microseconds of anything fancier at this size.
 *
 * Why hierarchical: this cluster is NOT a 4-way fabric. Each Spark spends both
 * its 200G ports on one partner, so it is two isolated pairs
 * (spark1<->spark2, spark3<->spark4) bridged only by 1 GbE. A flat star rooted
 * at rank 0 would drag every far-pair rank's payload across that bridge --
 * 4 messages per reduction here. Reducing inside each pair first and crossing
 * once cuts bridge traffic by the group size and keeps the slow hop off the
 * critical path for all but one rank.
 *
 *   phase 1  members -> group leader        (intra-pair, 200 GbE)
 *   phase 2  leaders  <-> root              (cross-pair, 1 GbE, ONE exchange)
 *   phase 3  leader   -> members            (intra-pair, 200 GbE)
 *
 * On the exact 4-rank/2-per-group Spark topology, K3_NET_RD2=1 adds a direct
 * Ethernet socket between ranks 1 and 3.  Both ranks in each fast pair first
 * exchange-add, then the two corresponding ranks exchange-add across pairs.
 * That recursive-doubling special case takes two phases and has no broadcast;
 * the launcher enables it by default for four nodes only.
 *
 * Set K3_GROUP_SIZE=1 or =K3_WORLD to degenerate back to a flat star.
 *
 * Environment (the launcher computes these per rank):
 *   K3_WORLD        ranks total; unset or 1 -> every collective is a no-op
 *   K3_RANK         this rank, 0..world-1
 *   K3_GROUP_SIZE   ranks per group (default = world, i.e. flat)
 *   K3_UP_HOST      address of this rank's upstream: its group leader if it is
 *                   a member, rank 0 if it is a non-root leader. Unset on
 *                   rank 0. Use the FAST address for intra-group links --
 *                   `ssh sparkN` names resolve to tailscale, not the fabric,
 *                   so pass 10.10.12.x / 10.10.34.x here.
 *   K3_MASTER_PORT  TCP port every leader listens on (default 29555)
 *   K3_NET_RD2      enable the specialized 4-rank/2-group two-phase path
 *   K3_CROSS_HOST   lower-pair Ethernet address for ranks 2 and 3 in RD2
 */
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#define K3_NET_MAX 16

static int    g_net_rank = 0, g_net_world = 1, g_net_gsize = 1;
static int    g_net_leader = 0, g_net_isleader = 1;
static int    g_net_fd[K3_NET_MAX];        /* downstream: members, and (root) peer leaders */
static int    g_net_up = -1;               /* upstream: leader, or root for a leader */
static int    g_net_cross = -1;            /* optional rank r <-> r+2 direct bridge */
static int    g_net_rd2 = 0;               /* two-phase 4-rank recursive doubling */
static int    g_net_split = 1;             /* halve the bridge payload; K3_NET_SPLIT=0 off */
/* Time-to-first-byte across the exchange. netkda is 416 us/layer where the
 * wire time after the payload split should be ~150; splitting the wait into
 * "peer was not ready" and "bytes moving" says whether the rest is arrival
 * jitter (a balance problem) or transport (a fabric problem). K3_NET_TTFB=1. */
static int    g_net_ttfb = 0;
static double g_ttfb_wait = 0, g_ttfb_move = 0;
static uint64_t g_ttfb_calls = 0;

static float *g_net_scratch = NULL;
static size_t g_net_scratch_n = 0;
static double g_net_secs = 0;              /* time parked in collectives */
static uint64_t g_net_calls = 0;

static inline int k3_net_rank(void)  { return g_net_rank; }
static inline int k3_net_world(void) { return g_net_world; }
static inline double k3_net_secs(void) { return g_net_secs; }
static inline uint64_t k3_net_calls(void) { return g_net_calls; }

static double k3_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                            return t.tv_sec + t.tv_nsec*1e-9; }

static int k3_send_all(int fd, const void *p, size_t n) {
    const char *b = (const char *)p;
    while (n) { ssize_t k = send(fd, b, n, 0);
        if (k <= 0) { if (errno == EINTR) continue; return 0; }
        b += k; n -= (size_t)k; }
    return 1;
}
static int k3_recv_all(int fd, void *p, size_t n) {
    char *b = (char *)p;
    while (n) { ssize_t k = recv(fd, b, n, 0);
        if (k <= 0) { if (k < 0 && errno == EINTR) continue; return 0; }
        b += k; n -= (size_t)k; }
    return 1;
}
static void k3_nodelay(int fd) {
    int one = 1;
    /* Every message is a synchronous round trip; Nagle would add up to 40 ms
     * per reduction, which at ~93 reductions/token is fatal. */
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    /* Halving the bridge payload took netkda down only 12%, so what is left is
     * latency, not wire time: the split costs two round trips per reduction and
     * there are ~185 reductions in a token. The exchange loop already spins in
     * user space with MSG_DONTWAIT, so the remaining wait is NIC-to-socket
     * interrupt latency, which is exactly what busy polling removes. Best
     * effort -- SO_BUSY_POLL needs CAP_NET_ADMIN on some kernels, and a failure
     * here is not worth reporting. K3_NET_BUSYPOLL=0 disables, or set it to the
     * microsecond budget to spend per receive. */
#ifdef SO_BUSY_POLL
    const char *bp = getenv("K3_NET_BUSYPOLL");
    int us = bp ? atoi(bp) : 50;
    if (us > 0) setsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &us, sizeof(us));
#endif
}

static int k3_connect_retry(const char *host, int port) {
    struct addrinfo hints, *res = NULL; char ps[16];
    snprintf(ps, sizeof(ps), "%d", port);
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    int fd = -1;
    /* Ranks come up together under srun; retry so we do not race the listen. */
    for (int t = 0; t < 900 && fd < 0; t++) {
        if (res) { freeaddrinfo(res); res = NULL; }
        if (getaddrinfo(host, ps, &hints, &res)) { usleep(100000); continue; }
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(fd, res->ai_addr, res->ai_addrlen)) { close(fd); fd = -1; usleep(100000); }
    }
    if (res) freeaddrinfo(res);
    if (fd < 0) { fprintf(stderr, "[K3/NET] connect %s:%d failed\n", host, port); exit(1); }
    k3_nodelay(fd);
    return fd;
}

static int k3_net_init(void) {
    const char *w = getenv("K3_WORLD");
    g_net_world = w ? atoi(w) : 1;
    if (g_net_world <= 1) { g_net_world = 1; return 1; }
    if (g_net_world > K3_NET_MAX) { fprintf(stderr,"[K3/NET] world %d > %d\n",g_net_world,K3_NET_MAX); exit(1); }
    g_net_rank  = getenv("K3_RANK") ? atoi(getenv("K3_RANK")) : 0;
    g_net_gsize = getenv("K3_GROUP_SIZE") ? atoi(getenv("K3_GROUP_SIZE")) : g_net_world;
    if (g_net_gsize < 1) g_net_gsize = 1;
    g_net_leader   = (g_net_rank / g_net_gsize) * g_net_gsize;
    g_net_isleader = (g_net_rank == g_net_leader);
    int port = getenv("K3_MASTER_PORT") ? atoi(getenv("K3_MASTER_PORT")) : 29555;
    g_net_rd2 = getenv("K3_NET_RD2") ? atoi(getenv("K3_NET_RD2")) : 0;
    g_net_split = getenv("K3_NET_SPLIT") ? atoi(getenv("K3_NET_SPLIT")) : 1;
    g_net_ttfb  = getenv("K3_NET_TTFB") ? atoi(getenv("K3_NET_TTFB")) : 0;
    if (g_net_rd2 && (g_net_world != 4 || g_net_gsize != 2)) {
        fprintf(stderr,"[K3/NET] K3_NET_RD2 requires world=4, group=2\n"); exit(1);
    }
    for (int i = 0; i < K3_NET_MAX; i++) g_net_fd[i] = -1;

    /* How many inbound connections this rank owns: its group members, plus --
     * on the root -- one per peer leader. */
    int nmembers = 0;
    for (int r = g_net_leader + 1; r < g_net_leader + g_net_gsize && r < g_net_world; r++) nmembers++;
    int npeers = 0;
    if (g_net_rank == 0)
        for (int L = g_net_gsize; L < g_net_world; L += g_net_gsize) npeers++;
    int nacc = (g_net_isleader ? nmembers : 0) + npeers;

    int ls = -1;
    if (nacc > 0) {
        ls = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        struct sockaddr_in a; memset(&a, 0, sizeof(a));
        a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons((uint16_t)port);
        if (bind(ls,(struct sockaddr*)&a,sizeof(a)) || listen(ls, K3_NET_MAX)) {
            fprintf(stderr,"[K3/NET] bind/listen :%d: %s\n",port,strerror(errno)); exit(1); }
    }

    /* Members connect up before leaders do, and every listener is already
     * bound above, so there is no ordering deadlock: a leader can accept its
     * members and then dial the root while the root is still accepting. */
    if (!g_net_isleader) {
        const char *up = getenv("K3_UP_HOST");
        if (!up) { fprintf(stderr,"[K3/NET] rank %d: K3_UP_HOST unset\n",g_net_rank); exit(1); }
        g_net_up = k3_connect_retry(up, port);
        if (!k3_send_all(g_net_up,&g_net_rank,sizeof(g_net_rank))) { fprintf(stderr,"[K3/NET] hello\n"); exit(1); }
    }
    for (int i = 0; i < nacc; i++) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) { fprintf(stderr,"[K3/NET] accept: %s\n",strerror(errno)); exit(1); }
        int peer = -1;
        if (!k3_recv_all(fd,&peer,sizeof(peer)) || peer < 0 || peer >= g_net_world) {
            fprintf(stderr,"[K3/NET] bad hello\n"); exit(1); }
        k3_nodelay(fd);
        g_net_fd[peer] = fd;
    }
    if (g_net_isleader && g_net_rank != 0) {
        const char *up = getenv("K3_UP_HOST");
        if (!up) { fprintf(stderr,"[K3/NET] leader %d: K3_UP_HOST unset\n",g_net_rank); exit(1); }
        g_net_up = k3_connect_retry(up, port);
        if (!k3_send_all(g_net_up,&g_net_rank,sizeof(g_net_rank))) { fprintf(stderr,"[K3/NET] hello\n"); exit(1); }
    }
    if (ls >= 0) close(ls);

    /* The 4-Spark topology has Ethernet on every node, not just the two pair
     * leaders.  Add one direct r<->r+2 socket so the all-reduce can do an
     * intra-pair exchange followed by two cross-pair exchanges in parallel.
     * This removes the third (leader->member) phase from the critical path.
     * Keep it opt-in: the generic hierarchy remains valid for every topology. */
    if (g_net_rd2) {
        int cp = port + 1;
        if (g_net_rank < 2) {
            int cs = socket(AF_INET, SOCK_STREAM, 0), one = 1;
            setsockopt(cs, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
            struct sockaddr_in a; memset(&a,0,sizeof(a));
            a.sin_family=AF_INET; a.sin_addr.s_addr=INADDR_ANY; a.sin_port=htons((uint16_t)cp);
            if (bind(cs,(struct sockaddr*)&a,sizeof(a)) || listen(cs,1)) {
                fprintf(stderr,"[K3/NET] cross bind/listen :%d: %s\n",cp,strerror(errno)); exit(1); }
            g_net_cross=accept(cs,NULL,NULL); close(cs);
            if (g_net_cross<0) { fprintf(stderr,"[K3/NET] cross accept: %s\n",strerror(errno)); exit(1); }
            int peer=-1;
            if (!k3_recv_all(g_net_cross,&peer,sizeof(peer)) || peer!=g_net_rank+2) {
                fprintf(stderr,"[K3/NET] bad cross hello\n"); exit(1); }
            k3_nodelay(g_net_cross);
        } else {
            const char *ch=getenv("K3_CROSS_HOST");
            if (!ch) { fprintf(stderr,"[K3/NET] rank %d: K3_CROSS_HOST unset\n",g_net_rank); exit(1); }
            g_net_cross=k3_connect_retry(ch,cp);
            if (!k3_send_all(g_net_cross,&g_net_rank,sizeof(g_net_rank))) {
                fprintf(stderr,"[K3/NET] cross hello\n"); exit(1); }
        }
    }
    fprintf(stderr,"[K3/NET] rank %d/%d group=%d leader=%d%s%s ready\n",
            g_net_rank,g_net_world,g_net_gsize,g_net_leader,g_net_isleader?" (leader)":"",
            g_net_rd2?" rd2":"");
    return g_net_world;
}

static void k3_reduce_from(int fd, float *v, size_t n) {
    if (g_net_scratch_n < n) {
        g_net_scratch = (float*)realloc(g_net_scratch, n*sizeof(float));
        if (!g_net_scratch) { fprintf(stderr,"[K3/NET] OOM scratch\n"); exit(1); }
        g_net_scratch_n = n;
    }
    if (!k3_recv_all(fd,g_net_scratch,n*sizeof(float))) { fprintf(stderr,"[K3/NET] recv\n"); exit(1); }
    for (size_t i = 0; i < n; i++) v[i] += g_net_scratch[i];
}

/* Exchange pair-local sums in both directions at once.  The two-group
 * topology used by the Spark cluster is full duplex, but the generic rooted
 * reduction below serialises its cross-group traffic: peer -> root, then the
 * completed root result -> peer.  Here both leaders send their local sum and
 * receive the other pair's sum concurrently, then perform the same final
 * floating-point add locally.  MSG_DONTWAIT plus poll keeps this safe for
 * prefill payloads larger than the socket send buffer (two blocking sends
 * could otherwise deadlock).
 */

static void k3_exchange_add(int fd, float *v, size_t n) {
    if (g_net_scratch_n < n) {
        g_net_scratch = (float*)realloc(g_net_scratch, n*sizeof(float));
        if (!g_net_scratch) { fprintf(stderr,"[K3/NET] OOM scratch\n"); exit(1); }
        g_net_scratch_n = n;
    }
    const char *tx = (const char*)v;
    char *rx = (char*)g_net_scratch;
    size_t bytes=n*sizeof(float), ns=0, nr=0;
    double t_enter = g_net_ttfb ? k3_now() : 0, t_first = 0;
    while (ns<bytes || nr<bytes) {
        int progress=0;
        if (ns<bytes) {
            ssize_t k=send(fd,tx+ns,bytes-ns,MSG_DONTWAIT|MSG_NOSIGNAL);
            if (k>0) { ns+=(size_t)k; progress=1; }
            else if (k<0 && errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) {
                fprintf(stderr,"[K3/NET] exchange send\n"); exit(1); }
        }
        if (nr<bytes) {
            ssize_t k=recv(fd,rx+nr,bytes-nr,MSG_DONTWAIT);
            if (k>0) { if (g_net_ttfb && !t_first) t_first = k3_now();
                       nr+=(size_t)k; progress=1; }
            else if (k==0) { fprintf(stderr,"[K3/NET] exchange EOF\n"); exit(1); }
            else if (errno!=EAGAIN && errno!=EWOULDBLOCK && errno!=EINTR) {
                fprintf(stderr,"[K3/NET] exchange recv\n"); exit(1); }
        }
        if (!progress) {
            struct pollfd p={.fd=fd,.events=0,.revents=0};
            if (ns<bytes) p.events|=POLLOUT;
            if (nr<bytes) p.events|=POLLIN;
            int pr;
            do pr=poll(&p,1,-1); while (pr<0 && errno==EINTR);
            if (pr<0) { fprintf(stderr,"[K3/NET] exchange poll\n"); exit(1); }
        }
    }
    for (size_t i=0;i<n;i++) v[i]+=g_net_scratch[i];
    if (g_net_ttfb) {
        double now = k3_now();
        g_ttfb_wait += (t_first ? t_first : now) - t_enter;   /* peer not ready */
        g_ttfb_move += t_first ? now - t_first : 0;           /* bytes moving */
        g_ttfb_calls++;
    }
}

/* v[0..n) := elementwise sum over all ranks. Blocking; every rank must call it
 * the same number of times in the same order. */
static void k3_net_allreduce(float *v, size_t n) {
    if (g_net_world <= 1) return;
    double t0 = k3_now();

    if (g_net_rd2) {
        /* Recursive doubling specialized to the two isolated fast pairs.
         * Phase 1 leaves the pair sum on BOTH ranks; phase 2 uses two separate
         * Ethernet links concurrently, so every rank obtains the global sum
         * without a leader broadcast. */
        int pairfd = g_net_isleader ? g_net_fd[g_net_rank+1] : g_net_up;
        k3_exchange_add(pairfd,v,n);
        /* After phase 1 BOTH ranks of a pair hold the identical pair sum, and
         * the old phase 2 had both of them push that same full vector across
         * the 1 GbE bridge. Half of that is redundant: let the pair leader
         * carry the first half of the vector across and the member carry the
         * second, then swap the two completed halves back over the 200 GbE
         * pair link, which is ~14x faster and effectively free.
         *
         * Bridge bytes per rank drop from n to n/2. Bit-exact: an element of
         * the first half is still ((v0+v1)+(v2+v3)) -- the same two additions
         * in the same order, just evaluated on one rank of the pair instead of
         * redundantly on both -- and the swap moves finished values, not
         * partial sums. Falls back to the full exchange for short vectors,
         * where a second round trip costs more than the halved payload saves
         * (the barrier reduces a single float). */
        size_t half = n / 2;
        if (g_net_split && n >= 1024 && half > 0) {
            if (g_net_isleader) {
                k3_exchange_add(g_net_cross, v, half);
                if (!k3_send_all(pairfd, v, half*sizeof(float)) ||
                    !k3_recv_all(pairfd, v+half, (n-half)*sizeof(float))) {
                    fprintf(stderr,"[K3/NET] split swap (leader)\n"); exit(1); }
            } else {
                k3_exchange_add(g_net_cross, v+half, n-half);
                if (!k3_recv_all(pairfd, v, half*sizeof(float)) ||
                    !k3_send_all(pairfd, v+half, (n-half)*sizeof(float))) {
                    fprintf(stderr,"[K3/NET] split swap (member)\n"); exit(1); }
            }
        } else {
            k3_exchange_add(g_net_cross,v,n);
        }
        g_net_secs += k3_now() - t0; g_net_calls++;
        return;
    }

    /* 1. intra-group: members -> leader (fast link) */
    if (g_net_isleader) {
        for (int r = g_net_leader+1; r < g_net_leader+g_net_gsize && r < g_net_world; r++)
            k3_reduce_from(g_net_fd[r], v, n);
    } else if (!k3_send_all(g_net_up, v, n*sizeof(float))) {
        fprintf(stderr,"[K3/NET] send to leader\n"); exit(1);
    }

    /* 2. across groups: leaders only — the single hop over the slow bridge */
    if (g_net_isleader) {
        int ngroups=(g_net_world+g_net_gsize-1)/g_net_gsize;
        if (ngroups==2) {
            int fd = g_net_rank==0 ? g_net_fd[g_net_gsize] : g_net_up;
            k3_exchange_add(fd,v,n);
        } else if (g_net_rank == 0) {
            for (int L = g_net_gsize; L < g_net_world; L += g_net_gsize)
                k3_reduce_from(g_net_fd[L], v, n);
            for (int L = g_net_gsize; L < g_net_world; L += g_net_gsize)
                if (!k3_send_all(g_net_fd[L], v, n*sizeof(float))) {
                    fprintf(stderr,"[K3/NET] bcast to leader %d\n",L); exit(1); }
        } else {
            if (!k3_send_all(g_net_up, v, n*sizeof(float)) ||
                !k3_recv_all(g_net_up, v, n*sizeof(float))) {
                fprintf(stderr,"[K3/NET] leader exchange\n"); exit(1); }
        }
    }

    /* 3. intra-group: leader -> members (fast link) */
    if (g_net_isleader) {
        for (int r = g_net_leader+1; r < g_net_leader+g_net_gsize && r < g_net_world; r++)
            if (!k3_send_all(g_net_fd[r], v, n*sizeof(float))) {
                fprintf(stderr,"[K3/NET] bcast to member %d\n",r); exit(1); }
    } else if (!k3_recv_all(g_net_up, v, n*sizeof(float))) {
        fprintf(stderr,"[K3/NET] recv from leader\n"); exit(1);
    }

    g_net_secs += k3_now() - t0; g_net_calls++;
}

/* Rendezvous, then zero the counters. Ranks finish loading at different times
 * (measured: 23 s apart on a 2-node run), and without this the first reduction
 * absorbs that skew and shows up as enormous per-call network cost — which is
 * startup jitter, not wire time. Call once after the model is resident. */
static void k3_net_barrier(void) {
    if (g_net_world <= 1) return;
    float x = 0.f;
    k3_net_allreduce(&x, 1);
    g_net_secs = 0; g_net_calls = 0;
}

/* ---- non-blocking form: issue on a worker thread, wait later ---------------
 * Measured at 4 ranks: the collective costs 3.21 ms/call in-engine but only
 * ~1.12 ms of that is transport -- the rest is rank arrival jitter, because
 * top-16 routing lands unevenly across the e%world shards (expert counts per
 * rank differed 10676..11392) and every barrier pays the max.
 *
 * Neither part can be made much faster, but both can be HIDDEN: the shared
 * experts depend on x, not on the reduced accumulator, so they are legitimate
 * independent work to run underneath. A worker thread is enough -- one
 * collective is in flight at a time, the main thread never touches these
 * sockets meanwhile, and it keeps the blocking implementation as the single
 * source of truth. */
static pthread_t g_ar_th;
static float    *g_ar_v = NULL;
static size_t    g_ar_n = 0;
static int       g_ar_active = 0;

static void *k3_ar_worker(void *unused) {
    (void)unused;
    /* Do NOT split the bridge payload here. Splitting halves the bytes but adds
     * a second round trip, and this collective is the one that is already
     * hidden underneath the shared experts -- halving a wait that was already
     * covered buys nothing while the extra trip is real. Measured: netmoe went
     * 0.974 -> 1.054 s/100 when the split applied here, against netkda -12.4%
     * and netmla -19% at the exposed sites. Payload reduction pays on exposed
     * collectives only. */
    int save = g_net_split; g_net_split = 0;
    k3_net_allreduce(g_ar_v, g_ar_n);
    g_net_split = save;
    return NULL;
}

static void k3_net_allreduce_start(float *v, size_t n) {
    if (g_net_world <= 1) return;
    g_ar_v = v; g_ar_n = n;
    if (pthread_create(&g_ar_th, NULL, k3_ar_worker, NULL) != 0) {
        k3_net_allreduce(v, n);          /* fall back to blocking */
        return;
    }
    g_ar_active = 1;
}

static void k3_net_allreduce_wait(void) {
    if (g_net_world <= 1 || !g_ar_active) return;
    pthread_join(g_ar_th, NULL);
    g_ar_active = 0;
}

static void k3_net_finalize(void) {
    if (g_net_world <= 1) return;
    for (int i = 0; i < K3_NET_MAX; i++) if (g_net_fd[i] >= 0) close(g_net_fd[i]);
    if (g_net_up >= 0) close(g_net_up);
    if (g_net_cross >= 0) close(g_net_cross);
    free(g_net_scratch); g_net_scratch = NULL; g_net_scratch_n = 0;
}

#endif
