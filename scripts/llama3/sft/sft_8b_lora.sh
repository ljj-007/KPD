#! /bin/bash

# 显卡选择
GPU_IDS=${4-"0"} # 使用第四个参数指定显卡编号，默认为 "0"
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

MASTER_ADDR=localhost # 分布式训练的主节点地址，默认为 localhost。
MASTER_PORT=${2-2013} # 主节点端口，默认为传入的第二个参数，若未传入则为 2012。
NNODES=1 # 总节点数，默认为 1。
NODE_RANK=0 # 当前节点的排名，默认为 0。
GPUS_PER_NODE=${3-1} # 每个节点上的 GPU 数量，默认为传入的第三个参数，若未传入则为 1。

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"."} # 基础路径，默认为传入的第一个参数，若未传入则为 "/home/MiniLLM"。
CKPT_NAME="llama3.1-8b" # 检查点名称，默认为 "llama-3.1-8b"。
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/" # 检查点路径。
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/llama3/" # 数据目录
# hp
BATCH_SIZE=2 # 训练时的批量大小，默认为 4。
LR=5e-5 # 学习率，默认为 0.0005。
GRAD_ACC=1 # 梯度累积步数，默认为 1。
EVAL_BATCH_SIZE=8 # 验证时的批量大小，默认为 8。
# length
MAX_LENGTH=512 # 最大序列长度，默认为 512。
# runtime
SAVE_PATH="${BASE_PATH}/results/llama3/train/sft/sft_8B" # 结果保存路径。
# seed
SEED=20 # 随机种子，默认为 20。
SEED_ORDER=10 # 顺序种子，默认为 10。


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama3"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 10"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num 1"
OPTS+=" --save ${SAVE_PATH}"
# lora
OPTS+=" --peft lora"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"
# type
OPTS+=" --type lm"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG="" # NCCL调试信息（空字符串表示不输出调试信息）。
export WANDB_DISABLED=True # 禁用 Weights & Biases，设置为 True。
export TF_CPP_MIN_LOG_LEVEL=3 # 设定 TensorFlow 的日志级别，3 表示只输出错误信息。
export PYTHONPATH=${BASE_PATH} # 设定 Python 路径，使用 BASE_PATH。
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@" # 生成最终运行的命令，包含分布式参数和训练选项。

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
