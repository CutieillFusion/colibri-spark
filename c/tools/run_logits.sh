#!/usr/bin/env bash
# Teacher-forced logit capture across the four ranks (--ngen $NGEN), for quality A/B.
TAG=$1; PORT=$2; shift 2; EXTRA="$*"; NGEN=${NGEN:-0}
S="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
BIN=/home/norquistdylan/colibri-spark/c/kimi_k3
P="Explain how a B-tree index works, then write the insert function in C. Include a discussion of node splitting, the difference between leaf and internal nodes, and why databases prefer B-trees over binary search trees for on-disk storage. Then describe how a write-ahead log interacts with index updates during crash recovery."
C="K3_WORLD=4 K3_GROUP_SIZE=2 K3_NET_RD2=1 K3_MASTER_PORT=$PORT K3_GPUS=0 K3_EXPERT_GPU=1 \
K3_EXPERT_GB=80 K3_GPU_GB=30 K3_MAXT=512 K3_TP_ATTN=1 OMP_NUM_THREADS=10 \
K3_DENSE_GPU=1 K3_DENSE_EXACT=1 K3_KDA_OVERLAP=1 K3_DENSE_DEV_GB=18 K3_DENSE_I4W=4 $EXTRA"
$S spark1 "$C K3_RANK=0 K3_W1_DIR=/home/norquistdylan/k3data/K3-w1 K3_LOGITS=/tmp/logits-$TAG.bin $BIN /home/norquistdylan/k3data/Kimi-K3 '$P' --ngen $NGEN" > /tmp/lg-$TAG-r0.out 2>&1 & p0=$!
sleep 2
$S spark2 "$C K3_RANK=1 K3_LOGITS=/tmp/logits-$TAG-r1.bin K3_UP_HOST=10.10.12.1 K3_W1_DIR=/home/norquistdylan/k3data/K3-w1-r1 K3_W2_SHARD=1/4 $BIN /home/norquistdylan/k3data/Kimi-K3 '$P' --ngen $NGEN" > /tmp/lg-$TAG-r1.out 2>&1 &
$S spark3 "$C K3_RANK=2 K3_LOGITS=/tmp/logits-$TAG-r2.bin K3_UP_HOST=192.168.0.159 K3_CROSS_HOST=192.168.0.159 K3_W1_DIR=/home/norquistdylan/k3data/K3-w1-r2 K3_W2_SHARD=2/4 $BIN /home/norquistdylan/k3data/Kimi-K3 '$P' --ngen $NGEN" > /tmp/lg-$TAG-r2.out 2>&1 &
$S spark4 "$C K3_RANK=3 K3_LOGITS=/tmp/logits-$TAG-r3.bin K3_UP_HOST=10.10.34.1 K3_CROSS_HOST=192.168.0.230 K3_W1_DIR=/home/norquistdylan/k3data/K3-w1-r3 K3_W2_SHARD=3/4 $BIN /home/norquistdylan/k3data/Kimi-K3 '$P' --ngen $NGEN" > /tmp/lg-$TAG-r3.out 2>&1 &
wait
