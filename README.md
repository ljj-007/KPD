# PKD


## 备注
model-type: 模型名, 不加大小, 基本旨在数据集地址那里使用。比如写llama2。


## 数据集准备
./scripts/llama/tools/process_data_dolly.sh

./scripts/llama/tools/process_data_pretrain.sh


## 推理
./scripts/llama2/tools/generate_data.sh


## SFT
./scripts/llama2/sft/sft_7b_lora.sh