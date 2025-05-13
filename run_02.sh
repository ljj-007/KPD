#! /bin/bash

# 显卡选择
GPU_IDS=${4-"0"} # 使用第四个参数指定显卡编号，默认为 "0"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1} # 每个节点的GPU数量

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="./"
CMD="torchrun ${DISTRIBUTED_ARGS} ./02_get_teacher_layers.py $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
${CMD}