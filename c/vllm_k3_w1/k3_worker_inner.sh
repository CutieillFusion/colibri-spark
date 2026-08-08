#!/bin/bash
# Runs inside a worker container: import the plugin so the registration exists
# in this process image, then join the ray head and block.
set -uo pipefail
python3 -c "import vllm_k3_w1; print('k3_w1 registered')" || exit 1
exec ray start --address="${HEAD_IP:-192.168.0.159}:6379" \
  --node-ip-address="$MYIP" --num-gpus=1 \
  --object-store-memory 1000000000 --block
