#!/bin/bash

NUM_GPUS=2
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

torchrun --nproc_per_node=$NUM_GPUS \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         /data2/jun/my-distill/examples/ddp_exapmle.py

# python -m torch.distributed.launch \
#     --nproc_per_node=$NUM_GPUS \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     /data2/jun/my-distill/examples/ddp_exapmle.py