#!/bin/bash
# Runs inside the head container on spark1: ray head, wait for all four nodes,
# then serve. Kept as its own file rather than inlined in the docker run
# command -- the nesting of bash-in-docker-in-sg-in-ssh makes quoting the
# single most likely thing to break, and it breaks silently.
set -uo pipefail

HEAD_IP=${HEAD_IP:-192.168.0.159}
MODEL=${K3_MODEL:-/k3data/K3-dense}
mkdir -p "$K3_LOG_DIR"
exec >>"$K3_LOG_DIR/head.log" 2>&1

echo "[head] checking vLLM source"
# Installed, not just importable: VLLM_PLUGINS resolves an entry point, and
# vLLM loads general plugins inside the engine-core and Ray worker processes
# (v1/engine/core.py:117, models/registry.py:1506) where our PYTHONPATH import
# never ran. No compilation -- this is pure Python.
# Only remove the image's own vLLM if it actually shadows /work. On some
# nodes dist-packages started winning after `pip install -e` (ImportError:
# cannot import name 'ApplyMoEActivationConfig'); on others the partial tree
# there is load-bearing and removing it breaks platform detection.
python3 -c "import vllm,sys; sys.exit(0 if vllm.__file__.startswith('/work') else 1)" 2>/dev/null || {
  echo "[setup] dist-packages vllm shadows /work; removing it"
  rm -rf /usr/local/lib/python3.12/dist-packages/vllm \
         /usr/local/lib/python3.12/dist-packages/vllm-*.dist-info 2>/dev/null
}
python3 -c "import vllm; assert vllm.__file__.startswith('/work'), vllm.__file__; print('vllm', vllm.__version__, 'from /work')" || exit 1
if [ "${K3_NATIVE:-0}" != 1 ]; then
  pip install -e /k3w1/vllm_k3_w1 --no-deps -q 2>&1 | tail -2
  python3 -c "import vllm_k3_w1; print('k3_w1 plugin importable')" || exit 1
else
  python3 -c "from vllm.model_executor.layers.quantization import get_quantization_config; print(get_quantization_config('k3_w1'))" || exit 1
fi

ray start --head --node-ip-address="$HEAD_IP" --port=6379 --num-gpus=1 \
  --num-cpus="${K3_RAY_CPUS:-0}" \
  --disable-usage-stats --object-store-memory 200000000 --include-dashboard=false || exit 1

(
  while :; do
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    grep -E 'MemAvailable|SwapFree' /proc/meminfo
    cat /proc/pressure/memory
    sleep 5
  done
) >>"$K3_LOG_DIR/host-memory.log" 2>&1 &

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
  --reasoning-parser kimi_k3 \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 \
  --enable-expert-parallel \
  --expert-placement-strategy round_robin \
  --distributed-executor-backend ray \
  --max-model-len "${K3_MAXLEN:-8192}" \
  --max-num-seqs 1 --max-num-batched-tokens "${K3_MNBT:-2048}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --gpu-memory-utilization "${K3_UTIL:-0.95}" \
  ${K3_KV_BYTES:+--kv-cache-memory-bytes "$K3_KV_BYTES"} \
  --kv-cache-dtype "${K3_KV_DTYPE:-auto}" \
  --no-enable-prefix-caching \
  --enforce-eager \
  --kernel-config '{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false, "enable_jit_warmup": false}' \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port 8000
