#! /bin/bash

# 显卡选择
GPU_IDS=${4-"0,1"} # 使用第四个参数指定显卡编号，默认为 "0"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

MASTER_ADDR=localhost
MASTER_PORT=${2-2113}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-2}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"."}
CKPT_NAME="Qwen-2.5-7B"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
PEFT_CKPT_NAME="sft_8B"
PEFT_CKPT="./results/qwen/train/sft/sft_7B/e10-bs4-lr5e-05-G1-N2-NN1-lora-256-64-0.1/17510"
MP_SIZE=4
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/qwen/"
# hp
EVAL_BATCH_SIZE=16
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen/gen/"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names dolly"
OPTS+=" --num-workers 0"
OPTS+=" --gen-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# lora
OPTS+=" --peft lora"
OPTS+=" --peft-name ${PEFT_CKPT_NAME}"
OPTS+=" --peft-path ${PEFT_CKPT}"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed 10"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type gen"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/generate.py ${OPTS} $@"


echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}