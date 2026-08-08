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

case "$(hostname)" in
  spark1) RANK=0 ;; spark2) RANK=1 ;; spark3) RANK=2 ;; spark4) RANK=3 ;;
  *) echo "unknown host $(hostname)"; exit 1 ;;
esac
STORE=/home/norquistdylan/k3data/K3-w1-r$RANK
[ -f "$STORE/experts.w2" ] || { echo "no store at $STORE"; exit 1; }
MYIP=$(ip -o -4 addr show enP7s7 | grep -oE '192\.168\.0\.[0-9]+' | head -1)

COMMON="--network host --gpus all --privileged --ipc=host --shm-size 8g
  -v /nas:/nas -v /tmp/vllm-main:/work -v /tmp/k3w1:/k3w1
  -v /home/norquistdylan/k3data:/k3data:ro -v $STORE:/k3store:ro
  -v /dev/infiniband:/dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1"

# PYTHONPATH carries both the vLLM build and our plugin; importing
# vllm_k3_w1 is what registers k3_w1 and installs the loader patches, so it
# has to happen inside every worker process, not just the driver.
ENV="-e PYTHONPATH=/work:/k3w1 -e VLLM_HOST_IP=$MYIP -e MASTER_ADDR=$HEAD_IP
  -e GLOO_SOCKET_IFNAME=enP7s7 -e NCCL_SOCKET_IFNAME=enP7s7
  -e NCCL_IB_DISABLE=1 -e RAY_memory_monitor_refresh_ms=0
  -e K3_W1_DIR=/k3store -e VLLM_PLUGINS=k3_w1
  -e PYTHONDONTWRITEBYTECODE=1 -e TOKENIZERS_PARALLELISM=false"

sg docker -c "docker rm -f $NAME 2>/dev/null" >/dev/null 2>&1

if [ "$(hostname)" = spark1 ]; then
  echo "[head spark1] ray head + K3 TP=4 serve, store=$STORE"
  sg docker -c "docker run -d --name $NAME $COMMON $ENV --entrypoint bash $IMG -c '
    set -e
    python3 -c \"import vllm_k3_w1\" || exit 1
    ray start --head --node-ip-address=$HEAD_IP --port=6379 --num-gpus=1 \
      --disable-usage-stats --object-store-memory 1000000000
    while [ \$(ray status 2>/dev/null | grep -c \"1.0/1.0 GPU\\|GPU\") -lt 1 ]; do sleep 2; done
    echo WAITING_FOR_WORKERS
    for i in \$(seq 120); do
      n=\$(python3 -c \"import ray;ray.init(address=\\\"auto\\\");print(len(ray.nodes()))\" 2>/dev/null || echo 0)
      [ \"\$n\" -ge 4 ] && break; sleep 5
    done
    echo RAY_NODES=\$n
    exec python3 -m vllm.entrypoints.openai.api_server \
      --model $MODEL --served-model-name kimi-k3 \
      --quantization k3_w1 --trust-remote-code \
      --tensor-parallel-size 4 --enable-expert-parallel \
      --expert-placement-strategy round_robin \
      --distributed-executor-backend ray \
      --max-model-len ${K3_MAXLEN:-8192} \
      --max-num-seqs 1 --max-num-batched-tokens ${K3_MNBT:-2048} \
      --limit-mm-per-prompt '"'"'{"image":0,"video":0}'"'"' \
      --gpu-memory-utilization ${K3_UTIL:-0.97} \
      --enforce-eager \
      --host 0.0.0.0 --port 8000
  '"
else
  echo "[worker $(hostname)] joining $HEAD_IP as $MYIP, store=$STORE"
  sg docker -c "docker run -d --name $NAME $COMMON $ENV --entrypoint bash $IMG -c '
    python3 -c \"import vllm_k3_w1\" || exit 1
    exec ray start --address=$HEAD_IP:6379 --node-ip-address=$MYIP \
      --num-gpus=1 --object-store-memory 1000000000 --block
  '"
fi
echo "[$(hostname)] container $NAME started"
