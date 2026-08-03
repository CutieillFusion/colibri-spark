#!/usr/bin/env bash
# Launch multi-node Kimi-K3 with a topology-aware all-reduce.
#
# The cluster is two isolated 200 GbE pairs (spark1<->spark2, spark3<->spark4)
# bridged only by 1 GbE, so ranks are grouped by pair: each member reduces to
# its pair leader over the fabric, and only the two leaders talk across the
# bridge. Intra-group links MUST use the 10.10.x fabric addresses -- the
# `sparkN` hostnames resolve to tailscale over the 1 GbE, which would silently
# put the fast path on the slow wire.
#
#   rank 0 = spark1 (leader, root)   rank 1 = spark2 -> 10.10.12.1
#   rank 2 = spark3 (leader)         rank 3 = spark4 -> 10.10.34.1
#   rank 2 -> rank 0 over 192.168.0.159 (the 1 GbE bridge, one hop per layer)
#
# Usage:  ./k3_launch.sh <world:2|4> "<prompt>" [ngen]
set -uo pipefail
WORLD=${1:-2}; PROMPT=${2:-"The capital of France is"}; NGEN=${3:-8}
BIN=${K3_BIN:-$HOME/colibri-spark/c/kimi_k3}
SNAP=${K3_SNAP:-/tmp/Kimi-K3}
PORT=${K3_MASTER_PORT:-29555}
S="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

HOSTS=(spark1 spark2 spark3 spark4)
UP=("" 10.10.12.1 192.168.0.159 10.10.34.1)     # per-rank upstream address
CROSS=("" "" 192.168.0.159 192.168.0.230)        # r2->r0 and r3->r1 Ethernet

COMMON="K3_WORLD=$WORLD K3_GROUP_SIZE=2 K3_MASTER_PORT=$PORT \
K3_GPUS=0 K3_EXPERT_GPU=1 K3_EXPERT_GB=${K3_EGB:-28} K3_GPU_GB=${K3_GGB:-40} K3_MAXT=512 K3_TP_ATTN=${K3_TP_ATTN:-1} OMP_NUM_THREADS=${K3_OMP_THREADS:-19}"
if [ "$WORLD" = 4 ]; then RD2=${K3_NET_RD2:-1}; else RD2=${K3_NET_RD2:-0}; fi
COMMON="$COMMON K3_NET_RD2=$RD2"

# This list is EXPLICIT: a variable exported in the caller's shell does NOT reach
# the ranks unless it is named here, and the run still looks completely normal
# when one is dropped -- K3_TP_ATTN was silently lost this way once (caught only
# because a collective count failed to change), and K3_ROUTE_STATS after it
# (collected all run, wrote nothing). Anything the engine reads via getenv and a
# caller may want to set has to be forwarded below.
# Each rank writes to its own node-local /tmp, so per-rank paths do not collide.
for v in K3_ROUTE_STATS K3_BITS K3_MLA_BITS K3_HEAD_BITS K3_THINK K3_TRACE \
         K3_DENSE_GPU K3_DENSE_STAGE K3_DENSE_EXACT; do
  eval "val=\${$v:-}"
  [ -n "$val" ] && COMMON="$COMMON $v=$val"
done

# Per-rank 2-bit expert store. Rank 0 keeps the FULL store: expert parallelism
# means it only ever reads e%world==0, and full-store indexing already places
# those correctly, so it needs no compacted shard. Every other rank carries just
# its slice (K3_W2_SHARD=r/N), which is what keeps the set inside one node.
W2=(${K3_W2_R0:-} ${K3_W2_R1:-} ${K3_W2_R2:-} ${K3_W2_R3:-})
# 1-bit stores use the same per-rank scheme; K3_W1_DIR selects the narrower
# slot sizing in the engine. Rank 0 keeps the FULL store (its e%world==0
# experts are already correctly placed by full-store indexing).
W1=(${K3_W1_R0:-} ${K3_W1_R1:-} ${K3_W1_R2:-} ${K3_W1_R3:-})
W2SHARD=("" "1/$WORLD" "2/$WORLD" "3/$WORLD")

pids=()
for ((r=0; r<WORLD; r++)); do
  env_r="$COMMON K3_RANK=$r"
  [ -n "${UP[$r]}" ] && env_r="$env_r K3_UP_HOST=${UP[$r]}"
  [ "$RD2" != 0 ] && [ -n "${CROSS[$r]}" ] && env_r="$env_r K3_CROSS_HOST=${CROSS[$r]}"
  if [ -n "${W1[$r]:-}" ]; then
    env_r="$env_r K3_W1_DIR=${W1[$r]}"
    [ "$r" != 0 ] && env_r="$env_r K3_W2_SHARD=${W2SHARD[$r]}"
  elif [ -n "${W2[$r]:-}" ]; then
    env_r="$env_r K3_W2_DIR=${W2[$r]}"
    [ "$r" != 0 ] && env_r="$env_r K3_W2_SHARD=${W2SHARD[$r]}"
  fi
  # rank 0 keeps stdout (only it prints tokens); the rest log to files
  if [ "$r" = 0 ]; then
    $S "${HOSTS[$r]}" "$env_r $BIN $SNAP '$PROMPT' --ngen $NGEN" 2>&1 | sed "s/^/[r0] /" &
  else
    $S "${HOSTS[$r]}" "$env_r $BIN $SNAP '$PROMPT' --ngen $NGEN" \
        > /tmp/k3-rank$r.log 2>&1 &
  fi
  pids+=($!)
done
wait "${pids[0]}"
rc=$?
for p in "${pids[@]:1}"; do wait "$p" 2>/dev/null; done
echo "[launch] rank0 exit=$rc; other ranks logged to /tmp/k3-rank*.log (on this host)"
exit $rc
