#!/usr/bin/env python3
"""Load the 1-bit expert store instead of the checkpoint's MXFP4 experts.

The checkpoint carries 1446 GB of MXFP4 expert tensors that we do not want and
cannot fit. The 1-bit store is a separate flat file the engine already uses.
So two things have to happen during weight loading:

  1. skip every `...experts.<E>.<w1|w2|w3>.weight_{packed,scale}` tensor, so
     vLLM neither reads 1.4 TB nor tries to force MXFP4 into our uint8 params
  2. fill w13_qweight / w13_scales / w2_qweight / w2_scales from the store

Config comes from the environment, matching the engine's own names:
    K3_W1_DIR    directory holding experts.w2   (required)
    K3_W1_SHARD  "r/N", the store's own e%N==r packing; omit for a full store

Slot addressing is c/kimi_k3.c:1455-1480 -- see loader.K3W1Store.
"""

import os
import re

import torch

from vllm.logger import init_logger

from .loader import K3W1Store

logger = init_logger(__name__)

EXPERT_RE = re.compile(r"\.experts\.\d+\.(w1|w2|w3)\.weight_(packed|scale)$")
_applied = False


def is_expert_tensor(name: str) -> bool:
    return bool(EXPERT_RE.search(name))


class _Filler:
    """Resolves store slots for a layer and copies them into its parameters."""

    def __init__(self):
        d = os.environ.get("K3_W1_DIR")
        if not d:
            raise RuntimeError("K3_W1_DIR is not set; nowhere to read experts from")
        self.dir = d
        sh = os.environ.get("K3_W1_SHARD", "")
        if sh:
            r, n = sh.split("/")
            self.s_rank, self.s_world = int(r), int(n)
        else:
            # Derive from the runtime rank rather than a per-node env: Ray
            # decides which host gets which rank, so baking "1/4" into
            # spark2's launch would silently mis-map if it were assigned
            # rank 2. Each node mounts its own K3-w1-r<rank> at the same
            # container path and this picks the matching packing.
            self.s_rank, self.s_world = _rank_world()
        self._store = None
        self.filled = 0

    def store(self, hidden, inter, experts_per_rank):
        if self._store is None:
            self._store = K3W1Store(self.dir, hidden, inter, experts_per_rank)
            logger.info("k3_w1: %s", self._store)
        return self._store

    def local_to_store_index(self, global_e: int) -> int:
        """Global expert id -> index within this store.

        A full store (no K3_W1_SHARD) is indexed by the global id. A store
        packed as e%N==r holds only those experts, contiguously, so the index
        is e//N -- the same arithmetic k3_launch.sh sets up with K3_W2_SHARD.
        """
        if self.s_world == 1:
            return global_e
        if global_e % self.s_world != self.s_rank:
            raise KeyError(f"expert {global_e} is not in shard "
                           f"{self.s_rank}/{self.s_world}")
        return global_e // self.s_world


def _rank_world() -> tuple[int, int]:
    """(rank, world) of the group the expert shard follows.

    A single-process run gets (0, 1), which selects a full unsharded store.
    """
    try:
        from vllm.distributed import parallel_state as ps

        g = ps.get_ep_group()
        return g.rank_in_group, g.world_size
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


def _global_ids_for(layer) -> list[int]:
    """Global expert ids this rank owns, in local-slot order."""
    emap = getattr(layer, "expert_map", None)
    shapes = getattr(layer, "k3_expert_shapes", None)
    n_local = (shapes["w13_qweight"][0] if shapes
               else layer.w13_qweight.shape[0])
    if emap is None:
        return list(range(n_local))
    ids = [-1] * n_local
    m = emap.tolist()
    for g, loc in enumerate(m):
        if loc >= 0:
            ids[loc] = g
    if any(i < 0 for i in ids):
        raise RuntimeError("expert_map does not cover every local slot")
    return ids


def quantize_dense_now(model) -> int:
    """Run the dense post-load hooks early, before experts are materialized.

    Ordering is the whole point. vLLM's sequence is create_weights for every
    parameter, then load, then process_weights_after_loading. Left alone that
    means 38 GB of bf16 dense is still resident when the 106.4 GB of experts
    appear -- 144 GB against ~122. Quantizing dense here drops it to 13 GB
    first, so the peak is 119.4 GB instead. vLLM's own later call is a no-op
    because the bf16 weight is already gone.
    """
    from .dense_method import KimiK3DenseLinearMethod

    n = 0
    for _, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, KimiK3DenseLinearMethod) and \
                not hasattr(mod, "k3_qweight"):
            qm.process_weights_after_loading(mod)
            n += 1
    return n


def materialize_experts(layer):
    """Allocate the expert parameters deferred by create_weights."""
    shapes = getattr(layer, "k3_expert_shapes", None)
    if not shapes:
        return
    for name, shape in shapes.items():
        p = getattr(layer, name)
        if tuple(p.shape) != tuple(shape):
            p.data = torch.empty(shape, dtype=torch.uint8, device=p.data.device)


def fill_layer(filler: _Filler, layer, moe_ordinal: int):
    """Populate one RoutedExperts layer from the store."""
    materialize_experts(layer)
    E, w13_rows, hb = layer.w13_qweight.shape
    hidden = hb * 8
    inter = w13_rows // 2
    # Slots per store "row": a sharded store holds global/N experts per MoE
    # layer, a full store holds all of them. Taken from the model rather than
    # hardcoded so a truncated config still addresses the store correctly.
    g_experts = int(getattr(layer, "global_num_experts", E * filler.s_world))
    st = filler.store(hidden, inter,
                      g_experts // filler.s_world if filler.s_world > 1
                      else g_experts)

    ids = _global_ids_for(layer)
    for slot, g in enumerate(ids):
        s = st.read_slot(moe_ordinal, filler.local_to_store_index(g))
        w13p, w13s = layer.w13_qweight[slot], layer.w13_scales[slot]
        w13p[:inter].copy_(torch.from_numpy(s["w1p"]))
        w13p[inter:].copy_(torch.from_numpy(s["w3p"]))
        w13s[:inter].copy_(torch.from_numpy(s["w1s"]))
        w13s[inter:].copy_(torch.from_numpy(s["w3s"]))
        layer.w2_qweight[slot].copy_(torch.from_numpy(s["w2p"]))
        layer.w2_scales[slot].copy_(torch.from_numpy(s["w2s"]))
    filler.filled += len(ids)


def apply_expert_map_shard():
    """Own the experts this node's store actually holds, not the ones its rank
    implies.

    Ray decides which host gets which rank, and it does not follow hostname
    order -- observed rank 2 on spark4 and rank 3 on spark3, while the stores
    mounted there are shard 3 and shard 2. Every expert would then be loaded
    from the wrong slot, silently.

    Rather than fight the assignment, permute the ownership: build this rank's
    expert_map from the shard its local store holds (K3_W1_SHARD). That stays
    correct because expert ownership only has to be a bijection -- each expert
    still has exactly one owner, and routing follows expert_map rather than
    rank arithmetic. Without K3_W1_SHARD this is a no-op.
    """
    sh = os.environ.get("K3_W1_SHARD", "")
    if not sh:
        return
    s_rank, s_world = (int(x) for x in sh.split("/"))
    import vllm.model_executor.layers.fused_moe.expert_map_manager as emm

    orig = emm.determine_expert_map

    def patched(ep_size, ep_rank, *a, **kw):
        if ep_size == s_world and ep_rank != s_rank:
            logger.info("k3_w1: rank %d owns shard %d (local store), not %d",
                        ep_rank, s_rank, ep_rank)
            ep_rank = s_rank
        return orig(ep_size, ep_rank, *a, **kw)

    emm.determine_expert_map = patched


def apply_placement():
    """Keep round-robin expert placement, which is what the store is packed for.

    The store holds e%N==r packed contiguously at e//N (k3_launch.sh's
    K3_W2_SHARD), which is exactly vLLM's `round_robin` mapping:
    expert_map[arange(rank, global, world)] = arange(0, local). But
    determine_expert_placement_strategy silently downgrades round_robin to
    `linear` unless num_expert_group > 1, and K3 has num_expert_group=1. Under
    linear, rank 0 would want globals 0..223 while its store holds
    0,4,8,...,892 -- every expert would be wrong, and nothing would crash.

    The gate does not apply to us: with num_expert_group=1 and topk_group=1
    the grouping is a no-op, and we take neither the all2all path nor the
    routing-table path (our apply() reads expert_map directly). Redundant
    experts and EPLB are still refused, since those genuinely change the
    mapping.
    """
    import vllm.model_executor.layers.fused_moe.expert_map_manager as emm

    orig = emm.determine_expert_placement_strategy

    def patched(expert_placement_strategy, moe_parallel_config,
                num_expert_group, num_redundant_experts, enable_eplb):
        if (expert_placement_strategy == "round_robin"
                and num_redundant_experts == 0 and not enable_eplb):
            return "round_robin"
        return orig(expert_placement_strategy, moe_parallel_config,
                    num_expert_group, num_redundant_experts, enable_eplb)

    emm.determine_expert_placement_strategy = patched
    logger.info("k3_w1: round-robin expert placement forced (matches store)")


def apply():
    """Patch K3's load_weights. Idempotent; call before model construction."""
    global _applied
    if _applied:
        return True
    apply_placement()
    apply_expert_map_shard()
    import vllm.models.kimi_k3.nvidia.model as k3

    cls = k3.KimiK3ForConditionalGeneration
    orig = cls.load_weights

    def load_weights(self, weights):
        skipped = [0]

        def keep(it):
            for name, w in it:
                if is_expert_tensor(name):
                    skipped[0] += 1
                    continue
                yield name, w

        loaded = orig(self, keep(weights))
        logger.info("k3_w1: skipped %d MXFP4 expert tensors", skipped[0])

        nq = quantize_dense_now(self)
        logger.info("k3_w1: quantized %d dense layers before expert fill", nq)

        filler = _Filler()
        n = 0
        for name, mod in self.named_modules():
            if not hasattr(mod, "w13_qweight"):
                continue
            m = re.search(r"layers\.(\d+)\.", name)
            if not m:
                continue
            # Store slots are indexed by MoE-layer ordinal, which counts only
            # sparse layers. first_k_dense_replace dense layers come first.
            ordinal = int(m.group(1)) - _first_dense(self)
            fill_layer(filler, mod, ordinal)
            n += 1
        logger.info("k3_w1: filled %d MoE layers, %d expert slots from %s",
                    n, filler.filled, filler.dir)
        if n == 0:
            raise RuntimeError("no MoE layers were filled -- is k3_w1 active?")

        # Tell vLLM these are loaded so its completeness check passes.
        for name, mod in self.named_modules():
            if hasattr(mod, "w13_qweight"):
                for p in ("w13_qweight", "w13_scales", "w2_qweight",
                          "w2_scales"):
                    loaded.add(f"{name}.{p}") if isinstance(loaded, set) else None
        return loaded

    cls.load_weights = load_weights
    _applied = True
    logger.info("k3_w1: expert loader installed")
    return True


def _first_dense(model) -> int:
    cfg = getattr(model, "config", None)
    for obj in (getattr(cfg, "text_config", None), cfg):
        v = getattr(obj, "first_k_dense_replace", None)
        if v is not None:
            return int(v)
    return 1
