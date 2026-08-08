"""Kimi-K3 1-bit expert support for vLLM.

Importing this package registers the `k3_w1` quantization method, after which
vLLM accepts `--quantization k3_w1` for KimiK3ForConditionalGeneration.
"""

from . import patch_latent
from .dense_method import KimiK3DenseLinearMethod
from .method import KimiK3OneBitConfig, KimiK3OneBitMoEMethod
from .loader import K3W1Store

# Importing the package must be enough to make the latent projections
# quantizable; see patch_latent for why that needs a patch at all.
patch_latent.apply()

__all__ = ["KimiK3OneBitConfig", "KimiK3OneBitMoEMethod",
           "KimiK3DenseLinearMethod", "K3W1Store", "patch_latent"]
