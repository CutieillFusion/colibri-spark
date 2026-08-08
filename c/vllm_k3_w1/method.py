#!/usr/bin/env python3
"""vLLM quantization method for the Kimi-K3 1-bit ("k3-w1") expert store.

Registers out-of-tree via `register_quantization_config`, so no vLLM fork is
needed -- import this module before constructing the engine and pass
`--quantization k3_w1`.

Why this exists: K3's checkpoint ships MXFP4 experts at 17,547,264 B each,
which is 362 GB/node for our 224-expert shard against 121 GB of unified
memory. The 1-bit store is 5,160,960 B/expert = 106 GB, the only format that
is fully RAM-resident on a DGX Spark. Everything else about the model already
runs on sm_121 (KDA decode and the MLA kv-cache fusions are compiled for
12.0f; AttnRes falls back to Triton).

Format is exactly what c/backend_cuda_k3.cu runs -- see kernels.py. The one
piece that is not in the checkpoint is the global amplitude `a`, which our
engine reads from K3_W1_A (default 1.69); it is carried here as a config
field so a store packed with a different `a` stays self-describing.
"""

from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from .dense_method import KimiK3DenseLinearMethod, bits_for_prefix
from .kernels import GROUP, situ_and_mul, w1_gemv, w1_grouped_gemm

def _defer_experts() -> bool:
    """Whether to allocate expert parameters empty and fill them later.

    On by default; K3_W1_EAGER_EXPERTS=1 restores the ordinary behaviour for
    a single-layer test where the peak does not matter.
    """
    import os
    return os.environ.get("K3_W1_EAGER_EXPERTS", "0") != "1"


DEFAULT_AMPLITUDE = 1.69
BLOCK_M = 16

# Below this many tokens the grouped GEMM is padding a handful of real rows up
# to BLOCK_M to feed tl.dot, and the GEMV path -- which needs no block-sorting
# pass at all -- wins. Measured per full MoE layer on one EP rank of four (224
# resident experts, 16-of-896 routing, H=3584 I=3072), bench_crossover.py:
#
#     T      1      2      4      8     16     64
#     gemv   0.371  0.617  0.891  1.480 3.019  12.365 ms
#     group  0.461  0.738  1.083  1.910 3.560   9.711 ms
#
# so the crossover sits between 16 and 64. Note this depends on the routing
# being spread: with few distinct local experts, grouped can serve several
# selections from one slot read and wins much earlier.
GEMV_MAX_TOKENS = 32


@register_quantization_config("k3_w1")
class KimiK3OneBitConfig(QuantizationConfig):
    """Two unrelated quantizations under one config, because K3 needs both.

    Routed experts: 1-bit sign-only with UE8M0 group-32 scales inherited from
    the checkpoint (kernels.py).

    Everything else: int4-g64 with f32 scales, or int8 per row for the MLA
    projections and lm_head (dense_kernels.py). The checkpoint's `ignore` list
    exempts these from its own MXFP4, so they arrive bf16 -- 114.4 GB across
    the model, which does not fit next to 106.4 GB of experts on a 121 GB
    node. They are quantized after loading rather than left alone.
    """

    def __init__(self, amplitude: float = DEFAULT_AMPLITUDE,
                 group_size: int = GROUP, dense_bits: int = 4,
                 mla_bits: int = 8, head_bits: int = 8) -> None:
        super().__init__()
        if group_size != GROUP:
            raise ValueError(
                f"k3_w1 scales are inherited byte-identically from the MXFP4 "
                f"checkpoint, which is group-32; got {group_size}"
            )
        self.amplitude = float(amplitude)
        self.group_size = group_size
        # Dense weights are a different quantization entirely -- signed int4 on
        # group-64 with f32 scales, or int8 per row. Defaults mirror the
        # engine's K3_BITS / K3_MLA_BITS / K3_HEAD_BITS. 16 disables it, which
        # is only viable if the non-expert set fits in bf16 (it does not, at
        # 114.4 GB across the model).
        self.dense_bits = int(dense_bits)
        self.mla_bits = int(mla_bits)
        self.head_bits = int(head_bits)

    def __repr__(self) -> str:
        return (f"KimiK3OneBitConfig(amplitude={self.amplitude}, "
                f"dense_bits={self.dense_bits}, mla_bits={self.mla_bits}, "
                f"head_bits={self.head_bits})")

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "k3_w1"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton-only; no tensor-core or tcgen05 dependency. GB10 is 121.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KimiK3OneBitConfig":
        # --quantization k3_w1 hands us an empty dict, so the env is the only
        # way to tune this from a launch script. Names mirror the engine's.
        import os
        def _e(k, d):
            return int(os.environ.get(k, config.get(k.lower().replace("k3_", ""), d)))
        return cls(
            amplitude=float(os.environ.get(
                "K3_W1_A", config.get("amplitude", DEFAULT_AMPLITUDE))),
            group_size=config.get("group_size", GROUP),
            dense_bits=_e("K3_BITS", 4),
            mla_bits=_e("K3_MLA_BITS", 8),
            head_bits=_e("K3_HEAD_BITS", 8),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, RoutedExperts):
            return KimiK3OneBitMoEMethod(self, layer.moe_config)
        if isinstance(layer, LinearBase):
            # The vision tower is small (0.9 GB) and not what this is for;
            # leaving it in bf16 keeps it out of the numerics story.
            if "vision" in prefix or "mm_projector" in prefix:
                return UnquantizedLinearMethod()
            bits = bits_for_prefix(prefix, self.dense_bits, self.mla_bits,
                                   self.head_bits)
            if bits >= 16:
                return UnquantizedLinearMethod()
            return KimiK3DenseLinearMethod(bits)
        return None


class KimiK3OneBitMoEMethod(FusedMoEMethodBase):
    """Sign-bit expert GEMMs, dequantized inside a Triton grouped GEMM.

    Deliberately not a modular kernel: `moe_kernel` stays None, so
    `is_monolithic` is False and RoutedExperts calls `apply` directly. The
    modular-kernel path exists to compose dispatch/combine backends for
    large-batch EP, none of which applies at B=1 on four nodes.
    """

    def __init__(self, quant_config: KimiK3OneBitConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if hidden_size % GROUP or intermediate_size_per_partition % GROUP:
            raise ValueError(
                f"k3_w1 needs both dims divisible by {GROUP}: "
                f"hidden={hidden_size} inter={intermediate_size_per_partition}"
            )
        w13_rows = self.moe.w13_num_shards * intermediate_size_per_partition

        # Allocated EMPTY and materialized later, by patch_loader, once the
        # dense weights have been quantized. vLLM allocates every parameter
        # before loading anything, so holding the real 106.4 GB of experts
        # here would sit alongside 38 GB of not-yet-quantized bf16 dense --
        # 144 GB against ~122 available. Deferring costs nothing: the store is
        # read after load_weights either way.
        layer.k3_expert_shapes = {
            "w13_qweight": (num_experts, w13_rows, hidden_size // 8),
            "w2_qweight": (num_experts, hidden_size,
                           intermediate_size_per_partition // 8),
            "w13_scales": (num_experts, w13_rows, hidden_size // GROUP),
            "w2_scales": (num_experts, hidden_size,
                          intermediate_size_per_partition // GROUP),
        }
        defer = _defer_experts()

        def _mk(shape):
            return torch.nn.Parameter(
                torch.empty(0 if defer else shape, dtype=torch.uint8),
                requires_grad=False)

        # Packed sign bits: 8 weights per byte, input-index order, LSB first.
        w13_qweight = _mk(layer.k3_expert_shapes["w13_qweight"])
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = _mk(layer.k3_expert_shapes["w2_qweight"])
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        # UE8M0 exponents, kept as raw bytes rather than materialized floats:
        # one byte per 32 inputs is 1/4 the memory of an fp16 scale and the
        # kernel does the exp2 anyway.
        w13_scales = _mk(layer.k3_expert_shapes["w13_scales"])
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = _mk(layer.k3_expert_shapes["w2_scales"])
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        layer.k3_amplitude = self.quant_config.amplitude

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        # No modular kernel, so no FusedMoEQuantConfig to hand one.
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Nothing to repack: the store's layout is already the kernel's layout.
        # That is the point of packing offline rather than at load time.
        return

    def _act_config(self) -> ApplyMoEActivationConfig:
        return ApplyMoEActivationConfig(
            activation_situ_beta=self.moe.activation_situ_beta,
            activation_situ_linear_beta=self.moe.activation_situ_linear_beta,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        # Routed output only, never a (shared, routed) tuple. MoERunner
        # ._apply_quant_method already runs the shared experts around this call
        # -- SharedExpertsOrder.NO_OVERLAP before, MULTI_STREAM_OVERLAPPED
        # after -- and reads _shared_experts.output itself
        # (runner/moe_runner.py:587-623). The shared_experts argument is here
        # only so a modular kernel can overlap them internally; running them
        # here as well would double-apply them.
        return self._forward_routed(layer, x, topk_weights, topk_ids)

    def _forward_routed(self, layer, x, topk_weights, topk_ids) -> torch.Tensor:
        T, H = x.shape
        top_k = topk_ids.shape[1]
        I = layer.w13_qweight.shape[1] // self.moe.w13_num_shards
        use_gemv = T <= GEMV_MAX_TOKENS

        if use_gemv:
            # No block-sorting pass at all: each (token, slot) program reads
            # its expert straight out of topk_ids.
            def gemm(a, packed, scale, out, tk, mul):
                w1_gemv(a, packed, scale, topk_ids, topk_weights,
                        layer.expert_map, out, tk, layer.k3_amplitude, mul)
        else:
            sorted_ids, expert_ids, npad = moe_align_block_size(
                topk_ids, BLOCK_M, layer.global_num_experts, layer.expert_map
            )

            def gemm(a, packed, scale, out, tk, mul):
                w1_grouped_gemm(a, packed, scale, sorted_ids, expert_ids, npad,
                                topk_weights, out, tk, layer.k3_amplitude, mul,
                                block_m=BLOCK_M)

        # gate/up, then SITU, then down. The routing weight is applied on the
        # down GEMM (not the up) so it multiplies the post-activation value,
        # matching every other vLLM fused-MoE kernel.
        inter = torch.zeros(
            (T * top_k, layer.w13_qweight.shape[1]), dtype=x.dtype, device=x.device
        )
        gemm(x, layer.w13_qweight, layer.w13_scales, inter, top_k, False)

        act = layer.activation
        if act == MoEActivation.SITU and self.moe.activation_situ_beta is not None:
            h = torch.empty((T * top_k, I), dtype=x.dtype, device=x.device)
            apply_moe_activation(act, h, inter,
                                 activation_config=self._act_config())
        else:
            h = situ_and_mul(
                inter, self.moe.activation_situ_beta or 1.0,
                self.moe.activation_situ_linear_beta,
            )

        # top_k=1 on the down GEMM: `h` already has one row per (token, slot),
        # so the row index is the pair index rather than the token index.
        down = torch.zeros((T * top_k, H), dtype=x.dtype, device=x.device)
        gemm(h, layer.w2_qweight, layer.w2_scales, down, 1, True)
        return down.view(T, top_k, H).sum(dim=1)
