/* Isolate the all-reduce cost from model load imbalance.
 *
 * The 2-node run reports 4.05 ms per reduction, but that number includes
 * whatever skew exists between ranks arriving at each layer. This drives the
 * SAME k3_net.h code path with no model behind it, so what it measures is
 * purely transport + reduction. The gap between the two is the imbalance.
 *
 * Matters because tensor-parallel dense would add ~93 more reductions per
 * token: at 4 ms each that costs more than sharding the dense set saves, while
 * at ~0.5 ms it is nearly free. Measure before building.
 *
 *   rank0: K3_WORLD=2 K3_RANK=0 ./bench_k3_net
 *   rank1: K3_WORLD=2 K3_RANK=1 K3_UP_HOST=10.10.12.1 ./bench_k3_net
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../k3_net.h"

int main(void) {
    k3_net_init();
    if (k3_net_world() < 2) { fprintf(stderr, "needs K3_WORLD>=2\n"); return 1; }

    /* 3584 floats = the Kimi-K3 latent accumulator, the real payload. */
    const int sizes[] = {1, 3584, 7168, 28672};
    const char *names[] = {"1 float (barrier)", "3584 (K3 latent)", "7168 (hidden)", "28672 (C=8)"};

    for (size_t s = 0; s < sizeof(sizes)/sizeof(*sizes); s++) {
        int n = sizes[s];
        float *v = (float *)malloc((size_t)n * sizeof(float));
        for (int i = 0; i < n; i++) v[i] = 1.0f;

        for (int w = 0; w < 50; w++) k3_net_allreduce(v, n);   /* warm + sync */
        k3_net_barrier();                                       /* zero counters */

        const int iters = 500;
        for (int i = 0; i < iters; i++) k3_net_allreduce(v, n);
        double ms = 1e3 * k3_net_secs() / (double)k3_net_calls();
        if (k3_net_rank() == 0)
            printf("%-20s %6d floats (%6.1f KB)  %7.3f ms/call  %6.2f MB/s effective\n",
                   names[s], n, n * 4 / 1024.0, ms,
                   (n * 4.0 / 1e6) / (ms / 1e3));
        k3_net_barrier();
        free(v);
    }
    /* Correctness: after allreduce every element must equal world size. */
    {
        float x = 1.0f;
        k3_net_allreduce(&x, 1);
        if (k3_net_rank() == 0)
            printf("sum check: %.1f (expect %d) %s\n", x, k3_net_world(),
                   x == (float)k3_net_world() ? "PASS" : "FAIL");
    }
    k3_net_finalize();
    return 0;
}
