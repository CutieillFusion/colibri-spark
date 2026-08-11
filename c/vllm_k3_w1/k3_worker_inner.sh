#!/bin/bash
# Runs inside a worker container: import the plugin so the registration exists
# in this process image, then join the ray head and block.
set -uo pipefail
mkdir -p "$K3_LOG_DIR"
exec > >(tee -a "$K3_LOG_DIR/worker.log") 2>&1
(
  while :; do
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    grep -E 'MemAvailable|SwapFree' /proc/meminfo
    cat /proc/pressure/memory
    sleep 5
  done
) >>"$K3_LOG_DIR/host-memory.log" 2>&1 &
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
pip install -e /k3w1/vllm_k3_w1 --no-deps -q 2>&1 | tail -2
python3 -c "import vllm; assert vllm.__file__.startswith('/work'), vllm.__file__; print('vllm', vllm.__version__, 'from /work')" || exit 1
python3 -c "import vllm_k3_w1; print('k3_w1 importable')" || exit 1
# Wait for the head's GCS before joining. `ray start` gives up after 60 s,
# and the head now runs a pip install before `ray start --head`, so a fixed
# sleep in the launcher is not enough -- all three workers died this way.
for i in $(seq 120); do
  (echo > /dev/tcp/${HEAD_IP:-192.168.0.159}/6379) >/dev/null 2>&1 && break
  sleep 5
done
echo "[worker] GCS reachable, joining"
exec ray start --address="${HEAD_IP:-192.168.0.159}:6379" \
  --node-ip-address="$MYIP" --num-gpus=1 \
  --num-cpus="${K3_RAY_CPUS:-0}" \
  --object-store-memory 200000000 --block
