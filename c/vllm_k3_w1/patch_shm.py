#!/usr/bin/env python3
"""Skip the shared-memory same-node probe, which deadlocks this cluster.

vLLM builds a MessageQueue per process group and asks in_the_same_node_as
which peers can use shared memory instead of TCP
(device_communicators/shm_broadcast.py:1022). That probe has the source rank
create a POSIX shm segment and broadcast its name, with the whole
create-and-broadcast wrapped in `contextlib.suppress(OSError)`
(parallel_state.py:2196). If the source's create fails, it swallows the error
and never broadcasts -- while every other rank is already blocked inside
broadcast_object_list. That is a permanent hang, and it is where a four-node
TP=4 K3 wedges: rank 0 sits in the broadcast, the others never enter it.

Here the answer is known without asking: each Spark has exactly one GB10, so
no two ranks ever share a node and every peer is remote. Returning that
directly skips the probe, the collective, and the deadlock. The MessageQueue
then uses its TCP path for all readers, which is what it would have chosen
anyway.

Guarded by K3_ONE_GPU_PER_NODE=1 so it cannot silently mislead a machine with
several GPUs per host, where same-node ranks are real and shm is the fast path.
"""

import os

import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


def apply():
    if os.environ.get("K3_ONE_GPU_PER_NODE", "0") != "1":
        return False
    import vllm.distributed.parallel_state as ps

    orig = ps.in_the_same_node_as

    def no_shm_probe(pg, source_rank: int = 0):
        try:
            if isinstance(pg, dist.ProcessGroup):
                world = dist.get_world_size(pg)
            else:
                world = pg.world_size
        except Exception:
            return orig(pg, source_rank)
        # Only the source is on the source's node.
        return [i == source_rank for i in range(world)]

    ps.in_the_same_node_as = no_shm_probe
    logger.info("k3_w1: same-node shm probe skipped "
                "(K3_ONE_GPU_PER_NODE=1); all peers treated as remote")
    return True
