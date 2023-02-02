#!/usr/bin/env bash

CONFIG=$1
WORKDIR=$2
CKPT=$3
GPUS=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-$((29500 + $RANDOM % 29))}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


if command -v torchrun &> /dev/null
then
  echo "Using torchrun mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    torchrun --nnodes=$NNODES \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/fscil.py $CONFIG $WORKDIR $CKPT --launcher pytorch ${@:5}
else
  echo "Using launch mode."
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/fscil.py $CONFIG $WORKDIR $CKPT --launcher pytorch ${@:5}
fi
