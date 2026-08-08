#!/bin/bash
# Per-node launcher: Kimi-K3, TP=4 + EP across all four Sparks, 1-bit experts.
#
# Shape follows /nas/vllm-moet/run_moet_pp2tp2rdma_node.sh, which is the
# configuration already proven to run a large MoE across these four boxes:
# spark1 is the ray head and the serve driver, the rest join as workers, and
# every rank talks over enP7s7 (the 192.168.0.0/24 management LAN). The two
# 200 GbE RoCE islands cannot carry a 4-rank communicator -- netName is
# per-communicator and all-or-nothing -- so a flat TP=4 has to run on the
# 1 GbE. That is the thing to optimize after it works, not before.
#
# Each node mounts its OWN expert shard at the same container path; the loader
# picks the matching packing from the runtime rank (patch_loader._rank_world).
#
#   ./run_k3_node.sh            start
#   ./run_k3_node.sh stop       tear down
set -uo pipefail

HEAD_IP=192.168.0.159                 # spark1 on enP7s7
# The base moet image is already on all four nodes and the built tree works
# from PYTHONPATH, so there is no 26 GB image to ship.
IMG=${K3_IMG:-vllm-moet-sm120:v024rdma2}
MODEL=${K3_MODEL:-/k3data/K3-dense}
NAME=k3-$( [ "$(hostname)" = spark1 ] && echo head || echo worker )

if [ "${1:-}" = stop ]; then
  sg docker -c "docker rm -f k3-head k3-worker 2>/dev/null" >/dev/null 2>&1
  echo "[$(hostname)] stopped"; exit 0
fi

# TP=2 x PP=2. The PP stage a node serves is FIXED by which repacked store it
# holds (that store contains only that stage's 46 MoE layers); the TP rank
# within the stage is not, since expert ownership only has to be a bijection
# (patch_loader.apply_expert_map_shard). spark1/spark2 hold stage 0,
# spark3/spark4 stage 1 -- matching the rank order Ray was observed to assign.
case "$(hostname)" in
  spark1) STAGE=0; TPR=0 ;; spark2) STAGE=0; TPR=1 ;;
  spark3) STAGE=1; TPR=0 ;; spark4) STAGE=1; TPR=1 ;;
  *) echo "unknown host $(hostname)"; exit 1 ;;
esac
RANK=$((STAGE * 2 + TPR))
STORE=/home/norquistdylan/k3data/K3-w1-pp${STAGE}t${TPR}
[ -f "$STORE/experts.w2" ] || { echo "no store at $STORE"; exit 1; }
MYIP=$(ip -o -4 addr show enP7s7 | grep -oE '192\.168\.0\.[0-9]+' | head -1)

# One line, deliberately: this string is handed to `sg docker -c`, which
# feeds it to sh -- embedded newlines become separate commands.
COMMON="--network host --gpus all --privileged --shm-size 16g -v /nas:/nas -v /tmp/vllm-main:/work -v /tmp/k3w1:/k3w1 -v /home/norquistdylan/k3data:/k3data:ro -v $STORE:/k3store:ro -v /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1"

# PYTHONPATH carries both the vLLM build and our plugin; importing
# vllm_k3_w1 is what registers k3_w1 and installs the loader patches, so it
# has to happen inside every worker process, not just the driver.
ENV="-e PYTHONPATH=/work:/k3w1 -e VLLM_HOST_IP=$MYIP -e MASTER_ADDR=$HEAD_IP -e GLOO_SOCKET_IFNAME=enP7s7 -e NCCL_SOCKET_IFNAME=enP7s7 -e NCCL_IB_GID_INDEX=3 -e RAY_memory_monitor_refresh_ms=0 -e K3_W1_DIR=/k3store -e K3_W1_SHARD=$TPR/2 -e K3_W1_STAGE_LOCAL=1 -e VLLM_PP_LAYER_PARTITION=47,46 -e VLLM_PLUGINS=k3_w1 -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false -e K3_MODEL_DIR=$MODEL -e K3_ONE_GPU_PER_NODE=1 -e K3_BITS=${K3_BITS:-4} -e K3_MLA_BITS=${K3_MLA_BITS:-8} -e K3_HEAD_BITS=${K3_HEAD_BITS:-8} -e K3_HIER_AR=${K3_HIER_AR:-0} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"

sg docker -c "docker rm -f $NAME 2>/dev/null" >/dev/null 2>&1

ENV="$ENV -e HEAD_IP=$HEAD_IP -e MYIP=$MYIP -e K3_MODEL=$MODEL -e K3_MAXLEN=${K3_MAXLEN:-8192} -e K3_MNBT=${K3_MNBT:-2048} -e K3_UTIL=${K3_UTIL:-0.95}"

if [ "$(hostname)" = spark1 ]; then
  echo "[head spark1] ray head + K3 TP=4 serve, store=$STORE, model=$MODEL"
  INNER=/k3w1/vllm_k3_w1/k3_head_inner.sh
else
  echo "[worker $(hostname)] joining $HEAD_IP as $MYIP, store=$STORE"
  INNER=/k3w1/vllm_k3_w1/k3_worker_inner.sh
fi

sg docker -c "docker run -d --name $NAME $COMMON $ENV --entrypoint bash $IMG $INNER"
echo "[$(hostname)] container $NAME started ($INNER)"
