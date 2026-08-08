#!/bin/bash
# Runs inside the head container on spark1: ray head, wait for all four nodes,
# then serve. Kept as its own file rather than inlined in the docker run
# command -- the nesting of bash-in-docker-in-sg-in-ssh makes quoting the
# single most likely thing to break, and it breaks silently.
set -uo pipefail

HEAD_IP=${HEAD_IP:-192.168.0.159}
MODEL=${K3_MODEL:-/k3data/K3-dense}

echo "[head] importing plugin"
# Installed, not just importable: VLLM_PLUGINS resolves an entry point, and
# vLLM loads general plugins inside the engine-core and Ray worker processes
# (v1/engine/core.py:117, models/registry.py:1506) where our PYTHONPATH import
# never ran. No compilation -- this is pure Python.
pip install -e /k3w1/vllm_k3_w1 --no-deps -q 2>&1 | tail -2
python3 -c "import vllm_k3_w1; print('k3_w1 importable')" || exit 1

ray start --head --node-ip-address="$HEAD_IP" --port=6379 --num-gpus=1 \
  --disable-usage-stats --object-store-memory 200000000 || exit 1

echo "[head] waiting for 4 ray nodes"
for i in $(seq 180); do
  n=$(python3 -c "
import ray
ray.init(address='auto', logging_level='ERROR')
print(sum(1 for x in ray.nodes() if x['Alive']))
" 2>/dev/null || echo 0)
  [ "${n:-0}" -ge 4 ] && break
  sleep 5
done
echo "[head] ray nodes alive: ${n:-0}"
[ "${n:-0}" -ge 4 ] || { echo "[head] cluster never reached 4 nodes"; exit 1; }

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name kimi-k3 \
  --quantization k3_w1 --trust-remote-code \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --expert-placement-strategy round_robin \
  --distributed-executor-backend ray \
  --max-model-len "${K3_MAXLEN:-8192}" \
  --max-num-seqs 1 --max-num-batched-tokens "${K3_MNBT:-2048}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --gpu-memory-utilization "${K3_UTIL:-0.95}" \
  --enforce-eager \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port 8000
