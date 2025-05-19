#!/bin/bash

MASTER_PORT=2060
DEVICE=${1-"1"}
ckpt=${2}

# dolly eval
CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/qwen/eval/eval_pkd_gkd/eval_main_dolly_lora.sh ./ ${MASTER_PORT} 1 qwen
CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/qwen/eval/eval_pkd_gkd/eval_main_self_inst_lora.sh ./ ${MASTER_PORT} 1 qwen
CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/qwen/eval/eval_pkd_gkd/eval_main_vicuna_lora.sh ./ ${MASTER_PORT} 1 qwen
CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/qwen/eval/eval_pkd_gkd/eval_main_sinst_lora.sh ./ ${MASTER_PORT} 1 qwen
CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/qwen/eval/eval_pkd_gkd/eval_main_uinst_lora.sh ./ ${MASTER_PORT} 1 qwen
# for seed in 10 20 30 40 50
# do
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openqwen/eval/eval_main_dolly_lora.sh ./ ${MASTER_PORT} 1 openqwen-3B
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openqwen/eval/eval_main_self_inst_lora.sh ./ ${MASTER_PORT} 1 openqwen-3B
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openqwen/eval/eval_main_vicuna_lora.sh ./ ${MASTER_PORT} 1 openqwen-3B
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openqwen/eval/eval_main_sinst_lora.sh ./ ${MASTER_PORT} 1 openqwen-3B
#     CUDA_VISIBLE_DEVICES=${DEVICE} bash ./scripts/openqwen/eval/eval_main_uinst_lora.sh ./ ${MASTER_PORT} 1 openqwen-3B
# done