#!/bin/bash
# Per-node launcher: Kimi-K3, TP=4 + EP across all four Sparks, 1-bit experts.
#
# Shape follows /nas/vllm-moet/run_moet_pp2tp2rdma_node.sh, which is the
# configuration already proven to run a large MoE across these four boxes:
# spark1 is the ray head and the serve driver, the rest join as workers, and
# bootstrap and cross-stage traffic use the 192.168.0.0/24 management LAN.
# K3_TP_RDMA=1 selects RoCE for the two intra-island TP communicators while
# keeping world and pipeline communicators on Socket.
#
# Each node mounts its OWN expert shard at the same container path; the loader
# picks the matching packing from the runtime rank (patch_loader._rank_world).
#
#   ./run_k3_node.sh            start
#   ./run_k3_node.sh stop       tear down
set -uo pipefail

HEAD_IP=192.168.0.159                 # spark1 on enP7s7
HOST=$(hostname)
RUN_ID=${K3_RUN_ID:-adhoc}
RUNTIME_ROOT=${K3_RUNTIME_ROOT:-/home/norquistdylan/k3w1/runtime}
CACHE_ROOT=${K3_CACHE_ROOT:-/home/norquistdylan/k3w1/cache}
LOG_DIR=$RUNTIME_ROOT/$RUN_ID/$HOST
CACHE_DIR=$CACHE_ROOT/$HOST
mkdir -p "$LOG_DIR/ray" "$CACHE_DIR/torchinductor" \
  "$CACHE_DIR/triton" "$CACHE_DIR/cuda"
# The base moet image is already on all four nodes and the built tree works
# from PYTHONPATH, so there is no 26 GB image to ship.
IMG=${K3_IMG:-vllm-moet-sm120:v024rdma2}
MODEL=${K3_MODEL:-/k3data/K3-dense}
NAME=k3-$( [ "$HOST" = spark1 ] && echo head || echo worker )

if [ "${1:-}" = stop ]; then
  sg docker -c "docker rm -f k3-head k3-worker 2>/dev/null" >/dev/null 2>&1
  echo "[$(hostname)] stopped"; exit 0
fi

# TP=2 x PP=2. The PP stage a node serves is FIXED by which repacked store it
# holds (that store contains only that stage's 46 MoE layers); the TP rank
# within the stage is not, since expert ownership only has to be a bijection
# (patch_loader.apply_expert_map_shard). spark1/spark2 hold stage 0,
# spark3/spark4 stage 1 -- matching the rank order Ray was observed to assign.
case "$HOST" in
  spark1) STAGE=0; TPR=0 ;; spark2) STAGE=0; TPR=1 ;;
  spark3) STAGE=1; TPR=0 ;; spark4) STAGE=1; TPR=1 ;;
  *) echo "unknown host $HOST"; exit 1 ;;
esac
RANK=$((STAGE * 2 + TPR))
STORE=/home/norquistdylan/k3data/K3-w1-pp${STAGE}t${TPR}
[ -f "$STORE/experts.w2" ] || { echo "no store at $STORE"; exit 1; }
MYIP=$(ip -o -4 addr show enP7s7 | grep -oE '192\.168\.0\.[0-9]+' | head -1)

# One line, deliberately: this string is handed to `sg docker -c`, which
# feeds it to sh -- embedded newlines become separate commands.
# --memory caps the container via cgroup so a runaway load is killed INSIDE
# it rather than starving the host. Without it spark1 wedges during expert
# fill -- the kernel still answers ICMP but sshd cannot complete a handshake,
# which has cost several reboots. 115g of 121 leaves the OS room to stay
# reachable; --memory-swap equal to it disables container swap.
COMMON="--memory=${K3_MEM_LIMIT:-115g} --memory-swap=${K3_MEM_LIMIT:-115g} --network host --gpus all --privileged --shm-size 16g -v /nas:/nas -v ${K3_WORK:-/home/norquistdylan/k3vllm}:/work -v ${K3_PLUGIN:-/home/norquistdylan/k3w1}:/k3w1 -v $LOG_DIR/ray:/k3ray -v /home/norquistdylan/k3data:/k3data:ro -v $STORE:/k3store:ro -v /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1"

# PYTHONPATH carries both the vLLM build and our plugin; importing
# vllm_k3_w1 is what registers k3_w1 and installs the loader patches, so it
# has to happen inside every worker process, not just the driver.
if [ "${K3_TP_RDMA:-0}" = 1 ]; then
  NCCL_IB_DISABLE_DEFAULT=0
else
  NCCL_IB_DISABLE_DEFAULT=1
fi
ENV="-e PYTHONPATH=/work:/k3w1 -e VLLM_HOST_IP=$MYIP -e MASTER_ADDR=$HEAD_IP -e GLOO_SOCKET_IFNAME=enP7s7 -e VLLM_DISTRIBUTED_TIMEOUT_SECONDS=${K3_DIST_TIMEOUT:-3600} -e TORCH_DISTRIBUTED_TIMEOUT=3600 -e NCCL_SOCKET_IFNAME=enP7s7 -e NCCL_IB_GID_INDEX=3 -e RAY_memory_monitor_refresh_ms=0 -e RAY_gcs_rpc_server_reconnect_timeout_s=900 -e RAY_health_check_timeout_ms=120000 -e RAY_health_check_period_ms=30000 -e RAY_health_check_failure_threshold=20 -e RAY_raylet_heartbeat_timeout_milliseconds=120000 -e RAY_TMPDIR=/k3ray -e K3_RAY_CPUS=${K3_RAY_CPUS:-0} -e TORCHINDUCTOR_CACHE_DIR=/k3w1/cache/$HOST/torchinductor -e TRITON_CACHE_DIR=/k3w1/cache/$HOST/triton -e CUDA_CACHE_PATH=/k3w1/cache/$HOST/cuda -e PYTHONFAULTHANDLER=1 -e K3_RUN_ID=$RUN_ID -e K3_LOG_DIR=/k3w1/runtime/$RUN_ID/$HOST -e K3_W1_DIR=/k3store -e K3_W1_SHARD=$TPR/2 -e K3_W1_STAGE_LOCAL=1 -e VLLM_PP_LAYER_PARTITION=47,46 -e VLLM_PLUGINS=k3_w1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false -e K3_MODEL_DIR=$MODEL -e K3_RESIDENT_EXPERTS=${K3_RESIDENT_EXPERTS:-0} -e K3_STREAM_SLOTS=${K3_STREAM_SLOTS:-16} -e K3_MOE_TOKEN_CHUNK=${K3_MOE_TOKEN_CHUNK:-0} -e K3_QUANT_EMBED=${K3_QUANT_EMBED:-1} -e K3_QUANT_GATE=${K3_QUANT_GATE:-0} -e K3_ONE_GPU_PER_NODE=1 -e K3_BITS=${K3_BITS:-4} -e K3_GROUP=${K3_GROUP:-64} -e K3_MLA_BITS=${K3_MLA_BITS:-8} -e K3_HEAD_BITS=${K3_HEAD_BITS:-8} -e K3_HIER_AR=${K3_HIER_AR:-0} -e VLLM_RDMA_ISLANDS_ENABLE=${K3_TP_RDMA:-0} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-$NCCL_IB_DISABLE_DEFAULT}"
ENV="$ENV -e K3_RANK_IPS=${K3_RANK_IPS:-192.168.0.159,192.168.0.230,192.168.0.200,192.168.0.33}"
ENV="$ENV -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${K3_EXEC_TIMEOUT:-7200} -e VLLM_ENGINE_ITERATION_TIMEOUT_S=${K3_EXEC_TIMEOUT:-7200}"
ENV="$ENV -e K3_ENFORCE_EAGER=${K3_ENFORCE_EAGER:-1} -e K3_PROFILE=${K3_PROFILE:-0}"
if [ "${K3_NATIVE:-0}" = 1 ]; then
  ENV="$ENV -e K3_NATIVE=1 -e VLLM_PLUGINS= -e K3_W1_DIR= -e K3_W1_SHARD= -e K3_W1_STAGE_LOCAL=0"
fi

sg docker -c "docker rm -f $NAME 2>/dev/null" >/dev/null 2>&1

# Drop page cache before starting. vLLM sizes the KV cache from the free
# memory it sees at worker init, and leftover cache from a previous load
# counted against it -- 112.66 GiB free with cache present vs ~117 without,
# which is the whole margin. Costs nothing: the next load re-reads what it
# needs.
sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true

ENV="$ENV -e HEAD_IP=$HEAD_IP -e MYIP=$MYIP -e K3_MODEL=$MODEL -e K3_MAXLEN=${K3_MAXLEN:-8192} -e K3_MNBT=${K3_MNBT:-512} -e K3_UTIL=${K3_UTIL:-0.918} -e K3_KV_BYTES=${K3_KV_BYTES:-} -e K3_KV_DTYPE=${K3_KV_DTYPE:-auto}"

if [ "$HOST" = spark1 ]; then
  echo "[head spark1] ray head + K3 TP=4 serve, store=$STORE, model=$MODEL"
  INNER=/k3w1/vllm_k3_w1/k3_head_inner.sh
else
  echo "[worker $(hostname)] joining $HEAD_IP as $MYIP, store=$STORE"
  INNER=/k3w1/vllm_k3_w1/k3_worker_inner.sh
fi

sg docker -c "docker run -d --name $NAME $COMMON $ENV --entrypoint bash $IMG $INNER"
echo "[$(hostname)] container $NAME started ($INNER)"
echo "[$HOST] run_id=$RUN_ID logs=$LOG_DIR cache=$CACHE_DIR"
