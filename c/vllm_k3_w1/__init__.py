"""Kimi-K3 1-bit expert support for vLLM.

Importing this package registers the `k3_w1` quantization method, after which
vLLM accepts `--quantization k3_w1` for KimiK3ForConditionalGeneration.
"""

from . import patch_hier_ar, patch_latent, patch_loader
from .dense_method import KimiK3DenseLinearMethod
from .method import KimiK3OneBitConfig, KimiK3OneBitMoEMethod
from .loader import K3W1Store

# Importing the package must be enough to make the latent projections
# quantizable; see patch_latent for why that needs a patch at all.
patch_latent.apply()

# The expert loader only makes sense when a store is configured; without
# K3_W1_DIR the checkpoint's own MXFP4 experts are the only source, and
# skipping them would leave the params empty.
import os as _os
if _os.environ.get("K3_W1_DIR"):
    patch_loader.apply()

# Off unless K3_HIER_AR=1; see patch_hier_ar for the measurements.
patch_hier_ar.apply()

def register():
    """vLLM general-plugin entry point.

    Importing this module already registers k3_w1 and applies the patches;
    this exists so vLLM's plugin loader can trigger that import inside worker
    processes it spawns, where our own import never ran.
    """
    return None


__all__ = ["register", "KimiK3OneBitConfig", "KimiK3OneBitMoEMethod",
           "KimiK3DenseLinearMethod", "K3W1Store", "patch_latent", "patch_loader", "patch_hier_ar"]
