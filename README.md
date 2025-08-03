# KPD: Knowledge Probing Distillation Framework

**K**nowledge‑**P**robing‑based precise **D**istillation for Large Language Models

## ✨ Key Features

- **Token‑level Need Discovery** – Identify *what* the student has not learned via uncertainty‑based probing.
- **Neuron Attribution Mapping** – Locate *where* the teacher stores the missing knowledge using integrated‑gradient attributions.
- **Focused Layer‑to‑Layer Transfer** – Distil only the relevant information through isometric layer mapping and KL‑divergence minimisation.

## 📑 Table of Contents

1. Overview
2. Architecture
3. Project Layout
4. Installation
5. Data Preparation
6. Probe & Distil
7. Evaluation

## 🏖️ 1. Overview

KPD is a three‑stage framework that **learns exactly what the student model lacks and transfers only the necessary knowledge** from a large teacher model, resulting in faster and more parameter‑efficient training.

1. **Student Probing** Compute token‑level prediction uncertainty and pick the *key tokens* with the highest entropy.
2. **Teacher Probing** Run integrated gradients on the teacher to obtain neuron attributions for these key tokens and select the most responsible layers.
3. **Targeted Distillation** Match each selected teacher layer to a proportionally mapped student layer (isometric mapping) and minimise the KL divergence between their intermediate logits.

![](./framework.png)

## 🚁 2. Architecture

student (N layers)              teacher (M layers)
      │                                │
      │ ① uncertainty probing          │ ② attribution probing
      ▼                                ▼
 key tokens  ─────────►  responsible teacher layers
      │                                │
      └────────── ③ isometric mapping ─┘
                     & KL minimisation

## 💺 3. Project Layout

The structure of the folder is shown below:

```csharp
KPD
├── configs/                 # YAML/JSON experiment configs
├── data/                    # Raw & processed datasets
├── data_utils/              # Data loaders & preprocessing scripts
├── distillm/                # Offical DistillM reproduction
├── EasyEdit/                # Neuron knowledge editing toolkit
├── examples/                # End‑to‑end usage examples
├── minillm/                 # Official MiniLLM reproduction
├── abkd/                    # Official abkd reproduction
├── disitllm-2/              # Official distillm-2 reproduction
├── mpu/                     # Parallel‑aware Transformer ops
├── probe_data/              # Student probing intermediate files
├── probe_teacher_data/      # Teacher probing intermediate files
├── results/                 # Checkpoints & evaluation outputs
├── scripts/                 # Shell helpers & slurm jobs
├── tools/                   # Misc utility scripts
└── README.md
```

Introduction to the structure of the folder:

- /configs: The configuration file for this project is in it.
- /data: The data for this project is stored in this directory.
- /data_utils: The data processing flow for this project is placed in this directory.
- /distillm: The official reproduction of the Distillm method is in this folder.
- /EasyEdit: A library of tools for cumulative neuronal knowledge detection on large models. (Limited to the size, we add the whole EasyEdit codes and resources in the Supplemental Materials)
- /minillm:  The official reproduction of the Minillm method is in this folder.
- /abkd:  The official reproduction of the ABKD method is in this folder.
- /distillm-2:  The official reproduction of the Distillm-2 method is in this folder.
- /mpu: The Transformer library for parallel computing.
- /probe_data: Probe dataset for probing teacher models.
- /probe_teacher_data: Results after probing the teacher model.
- /results: The path where the results of the model training are stored.
- /tools: Some of the tools commonly used in the code.

## 🎄 4. Installation

Ensure **Python ≥ 3.9** and **CUDA 11.7+** are available, then install dependencies:

```shell
pip install -r requirements.txt
```

## 🍧 5. Data Preparation

Run *the provided scripts to preprocess Dolly‑15k and the pre‑training mixtures for each backbone:*

```shell
# LLaMA 2 (7B)
./scripts/llama2/tools/process_data_dolly.sh
./scripts/llama2/tools/process_data_pretrain.sh

# LLaMA 3 (8B)
./scripts/llama3/tools/process_data_dolly.sh
./scripts/llama3/tools/process_data_pretrain.sh

# Qwen 2.5 (7B)
./scripts/qwen/tools/process_data_dolly.sh
./scripts/qwen/tools/process_data_pretrain.sh
```

## 🍬 6. Probe & Distillation

Below is a minimal example for *KD + PKD* on a LLaMA‑3.2‑1B student.

- 01 Probe the student model

```shell
python 01_probe_student.py
```

- 02 Probe the teacher model

```shell
chmod 777 ./run_02.sh
./run_02.sh  # calls 02_probe_teacher.py internally
```

- 03 Distill from teacher model to student model, different backbone distillation methods can be selected by modifying the type parameter in run_pkd.sh

```shell
chmod 777 ./run_pkd
./run_pkd   # classic knowledge distillation
```

## ☕ 7. Evaluation

After training, evaluate on all benchmarks:

```shell
./scripts/llama2/eval/eval_pkd_kd/run_eval.sh
./scripts/llama3/eval/eval_pkd_kd/run_eval.sh
./scripts/qwen/eval/eval_pkd_kd/run_eval.sh
```

The evaluation script prints accuracy, perplexity, and average score across *Dolly, Self‑Instr, SInSt, UInSt,* and *Vicuna*.
