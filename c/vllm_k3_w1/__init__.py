"""Kimi-K3 1-bit expert support for vLLM.

Importing this package registers the `k3_w1` quantization method, after which
vLLM accepts `--quantization k3_w1` for KimiK3ForConditionalGeneration.
"""

import os as _os

from . import (
    patch_hier_ar,
    patch_latent,
    patch_loader,
    patch_rank_order,
    patch_shm,
    patch_warmup,
)
from .dense_method import KimiK3DenseLinearMethod
from .loader import K3W1Store
from .method import KimiK3OneBitConfig, KimiK3OneBitMoEMethod

# Importing the package must be enough to make the latent projections
# quantizable; see patch_latent for why that needs a patch at all.
patch_latent.apply()

# The expert loader only makes sense when a store is configured; without
# K3_W1_DIR the checkpoint's own MXFP4 experts are the only source, and
# skipping them would leave the params empty.

if _os.environ.get("K3_W1_DIR"):
    patch_rank_order.apply()
    patch_loader.apply()
    patch_warmup.apply()

# Off unless K3_HIER_AR=1; see patch_hier_ar for the measurements.
def _local_rank_log():
    """Mirror vLLM's log to a host-visible file on every rank.

    The per-rank memory decomposition ("Actual usage is X GiB for consumed
    memory (weights + non-torch), Y GiB for peak activation") is emitted by
    each worker but forwarded to the HEAD's log by Ray. When the head becomes
    unreachable during load -- which is the normal failure here -- that data
    is lost even though every rank computed it. /k3w1 is a host bind mount, so
    writing there survives both an unreachable head and the container exiting.
    """
    import logging
    import os
    import socket

    try:
        d = os.environ.get("K3_LOG_DIR", "/k3w1/logs")
        os.makedirs(d, exist_ok=True)
        path = f"{d}/rank-{socket.gethostname()}-{os.getpid()}.log"
        fh = logging.FileHandler(path, mode="a")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        lg = logging.getLogger("vllm")
        lg.addHandler(fh)
        lg.setLevel(logging.INFO)
    except OSError:
        pass


_local_rank_log()
patch_shm.apply()
patch_hier_ar.apply()

def register():
    """vLLM general-plugin entry point.

    Importing this module already registers k3_w1 and applies the patches;
    this exists so vLLM's plugin loader can trigger that import inside worker
    processes it spawns, where our own import never ran.
    """
    return None


__all__ = [
    "K3W1Store",
    "KimiK3DenseLinearMethod",
    "KimiK3OneBitConfig",
    "KimiK3OneBitMoEMethod",
    "patch_hier_ar",
    "patch_latent",
    "patch_loader",
    "patch_rank_order",
    "patch_shm",
    "patch_warmup",
    "register",
]
