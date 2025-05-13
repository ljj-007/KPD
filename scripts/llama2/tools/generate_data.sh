#! /bin/bash

MASTER_ADDR=localhost # 分布式训练主节点的地址，默认为 localhost。
MASTER_PORT=${2-2113} # 分布式训练主节点的端口，默认值为 2113，也可以通过脚本的第二个参数进行指定。
NNODES=1 # 节点总数，默认为 1。
NODE_RANK=0 # 当前节点的排名，默认为 0。
GPUS_PER_NODE=${3-1} # 每个节点上的 GPU 数量，默认值为 1，可以通过脚本的第三个参数指定。

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT" # 构建用于 torchrun 的分布式训练参数。

# model
BASE_PATH=${1-"."} # 基础路径，默认值为 ./，可以通过脚本的第一个参数指定。
CKPT_NAME="llama2" # 模型检查点的名称。
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/" # 模型检查点的路径。
PEFT_CKPT_NAME="sft_7B" # Lora保存名字。
PEFT_CKPT="${BASE_PATH}/results/llama2/train/sft/${PEFT_CKPT_NAME}/" # LoRA保存路径
MP_SIZE=4 # 模型并行度。
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/llama2/" # 数据目录。
# hp
EVAL_BATCH_SIZE=16 # 评估批次大小。
# runtime
SAVE_PATH="${BASE_PATH}/results/llama2/gen/" # 生成结果的保存路径。


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type llama2"
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


export TOKENIZERS_PARALLELISM=false # 禁用 tokenizers 的并行化。
export PYTHONIOENCODING=utf-8 # 设置 Python 输入输出编码为 UTF-8。
export PYTHONPATH=${BASE_PATH} # 设置 Python 模块搜索路径。
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/generate.py ${OPTS} $@" # 组合所有选项和参数形成最终的命令。


echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}" # 输出命令和 PYTHONPATH 以供调试。
mkdir -p ${SAVE_PATH} # 创建保存路径（如果不存在）。
${CMD} # 执行命令 ${CMD}
