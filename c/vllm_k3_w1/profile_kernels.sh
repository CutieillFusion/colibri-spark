#!/bin/bash
set -euo pipefail

OUT=${K3_PROFILE_DIR:-/k3w1/profiles}
NCU=${NCU_BIN:-/usr/local/cuda-13.0/bin/ncu}
PYTHON=${K3_PYTHON:-python3}
mkdir -p "$OUT"

"$NCU" --target-processes all --replay-mode kernel \
  --kernel-name 'regex:_k3_w1_gemm_kernel' --launch-count 8 \
  --section SpeedOfLight --section Occupancy --section LaunchStats \
  --csv --log-file "$OUT/prefill.csv" \
  "$PYTHON" /k3w1/vllm_k3_w1/profile_workload.py prefill

"$NCU" --target-processes all --replay-mode kernel \
  --kernel-name 'regex:_k3_w1_gemv_kernel' --launch-count 8 \
  --section SpeedOfLight --section Occupancy --section LaunchStats \
  --csv --log-file "$OUT/decode.csv" \
  "$PYTHON" /k3w1/vllm_k3_w1/profile_workload.py decode
