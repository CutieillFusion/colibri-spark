#!/usr/bin/env python3
"""Hierarchical TP all-reduce: RD2 mapped onto two NCCL communicators.

The four Sparks are two 200 GbE RoCE islands (spark1+spark2, spark3+spark4)
joined only by a 1 GbE LAN. NCCL's netName is per-communicator and
all-or-nothing, so a flat 4-rank communicator spans both islands and every
byte -- including the intra-island legs that could have used RoCE -- goes over
the 1 GbE. Measured on this cluster, 14 KB (one K3 all-reduce):

    flat 4-rank, across islands            5.507 ms
    2-rank intra-island, forced Socket     2.861 ms
    2-rank intra-island, net_name="IB"     0.020 ms     <- 143x

Recursive doubling splits one 4-way all-reduce into two 2-way ones:

    phase 1   all-reduce within the island pair   -> RoCE, 0.020 ms
    phase 2   all-reduce across the pairs         -> 1 GbE, ~2.9 ms

which sums to the same value (addition is associative) at ~2.9 ms instead of
5.5. This is the same decomposition c/k3_net.h:315-331 implements over raw
TCP; there it costs far less than NCCL's Socket path does, so this is an
improvement rather than a match.

torch exposes ProcessGroupNCCL.Options().config.net_name in this build, so no
vLLM or NCCL patch is needed -- just two extra process groups.

Enable with K3_HIER_AR=1.
"""

import os
import socket

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

# Physical islands. Override with K3_ISLANDS="a,b|c,d" if the wiring changes.
DEFAULT_ISLANDS = "spark1,spark2|spark3,spark4"
_state = {}


def _island_map() -> dict[str, int]:
    spec = os.environ.get("K3_ISLANDS", DEFAULT_ISLANDS)
    m = {}
    for i, part in enumerate(spec.split("|")):
        for h in part.split(","):
            if h.strip():
                m[h.strip()] = i
    return m


def _build_groups(tp):
    """Create the island and cross groups. Every rank must call this."""
    world = tp.world_size
    if world != 4:
        logger.warning("k3_w1: hierarchical all-reduce wants TP=4, got %d; "
                       "leaving the flat path in place", world)
        return None

    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname(), group=tp.device_group)
    imap = _island_map()
    isl = [imap.get(h, -1) for h in hosts]
    if sorted(isl) != [0, 0, 1, 1]:
        logger.warning("k3_w1: hosts %s do not form two pairs under %s; "
                       "leaving the flat path in place", hosts, imap)
        return None

    members = {0: [r for r in range(world) if isl[r] == 0],
               1: [r for r in range(world) if isl[r] == 1]}
    rank = tp.rank_in_group
    my_isl = isl[rank]
    pos = members[my_isl].index(rank)

    # RoCE for the intra-island pair. Every rank has to enter new_group for
    # each group, in the same order, even the ones not in it.
    ib = dist.ProcessGroupNCCL.Options()
    ib.config.net_name = "IB"
    island_pg = None
    for i in (0, 1):
        g = dist.new_group(members[i], backend="nccl", pg_options=ib)
        if i == my_isl:
            island_pg = g

    # The cross pair necessarily spans the bridge, so Socket.
    sk = dist.ProcessGroupNCCL.Options()
    sk.config.net_name = "Socket"
    cross_pg = None
    for p in range(2):
        pair = [members[0][p], members[1][p]]
        g = dist.new_group(pair, backend="nccl", pg_options=sk)
        if p == pos:
            cross_pg = g

    logger.info("k3_w1: hierarchical all-reduce ready -- hosts %s, "
                "island %s (IB), cross %s (Socket)",
                hosts, members[my_isl], [members[0][pos], members[1][pos]])
    return island_pg, cross_pg


def apply():
    if os.environ.get("K3_HIER_AR", "0") != "1":
        return False
    import vllm.distributed.parallel_state as ps

    orig = ps.GroupCoordinator._all_reduce_out_place

    def hier(self, input_: torch.Tensor) -> torch.Tensor:
        # Only the TP group; PP/DP/EP collectives keep the default path.
        if getattr(self, "group_name", None) != "tp":
            return orig(self, input_)
        if "pgs" not in _state:
            try:
                _state["pgs"] = _build_groups(self)
            except Exception as e:            # never take the model down
                logger.warning("k3_w1: hierarchical setup failed (%s); "
                               "using the flat path", e)
                _state["pgs"] = None
        pgs = _state["pgs"]
        if pgs is None:
            return orig(self, input_)
        island, cross = pgs
        out = input_.contiguous()
        dist.all_reduce(out, group=island)
        dist.all_reduce(out, group=cross)
        return out

    ps.GroupCoordinator._all_reduce_out_place = hier
    logger.info("k3_w1: hierarchical TP all-reduce installed (K3_HIER_AR=1)")
    return True
