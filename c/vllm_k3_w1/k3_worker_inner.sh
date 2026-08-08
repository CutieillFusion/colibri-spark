#!/bin/bash
# Runs inside a worker container: import the plugin so the registration exists
# in this process image, then join the ray head and block.
set -uo pipefail
# Installed, not just importable: VLLM_PLUGINS resolves an entry point, and
# vLLM loads general plugins inside the engine-core and Ray worker processes
# (v1/engine/core.py:117, models/registry.py:1506) where our PYTHONPATH import
# never ran. No compilation -- this is pure Python.
pip install -e /k3w1/vllm_k3_w1 --no-deps -q 2>&1 | tail -2
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
  --object-store-memory 1000000000 --block
