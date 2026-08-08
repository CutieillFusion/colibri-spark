"""Kimi-K3 1-bit expert support for vLLM.

Importing this package registers the `k3_w1` quantization method, after which
vLLM accepts `--quantization k3_w1` for KimiK3ForConditionalGeneration.
"""

from .dense_method import KimiK3DenseLinearMethod
from .method import KimiK3OneBitConfig, KimiK3OneBitMoEMethod
from .loader import K3W1Store

__all__ = ["KimiK3OneBitConfig", "KimiK3OneBitMoEMethod",
           "KimiK3DenseLinearMethod", "K3W1Store"]
