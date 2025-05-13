import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"文件夹 '{path}' 已创建。")
    else:
        print(f"文件夹 '{path}' 已存在。")


def get_data(data_path):
    """ 读取数据 """
    train_data_path = os.path.join(data_path, "train.jsonl")
    valid_data_path = os.path.join(data_path, "valid.jsonl")
    train_data, valid_data = [], []
    with open(train_data_path, "r", encoding="utf-8") as rf:
        lines = list(rf.readlines())
        for line in lines:
            train_data.append(json.loads(line))
    with open(valid_data_path, "r", encoding="utf-8") as rf:
        lines = list(rf.readlines())
        for line in lines:
            valid_data.append(json.loads(line))
    return train_data, valid_data


def get_model(model_path):
    """ 读取模型 """
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def get_probe_data(d, model, tokenizer):
    """ 输入一条数据, 输出这条数据相关的探测数据, 用于对教师模型的关键层进行探测 """
    probe_data = []
    text = d["prompt"] + d["output"]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # split_index = torch.nonzero(inputs["input_ids"][0] == 0).squeeze().cpu().item()
    inputs_prompt = tokenizer(d["prompt"], return_tensors="pt").to(model.device)
    inputs_len = len(inputs["input_ids"][0]) # 总长度
    inputs_prompt_len = len(inputs_prompt["input_ids"][0]) # 输入长度
    # print(tokenizer.decode(inputs["input_ids"][0]))
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits # [batch_size, seq_len, vocab_size]

    shifted_logits = logits[:, :-1, :]
    shifted_labels = inputs.input_ids[:, 1:]

    probs = torch.softmax(shifted_logits, dim=-1)
    gt_probs = torch.gather(probs, 2, shifted_labels.unsqueeze(-1)).squeeze(-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)  # shape: [batch_size, seq_len-1]
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    for i in range(inputs_prompt_len - 1, len(gt_probs[0])):
        # print(f"Token: {tokens[i+1]:<15} | P: {gt_probs[0][i].item():.4f} | Entropy: {entropy[0][i].item():.4f}")
        if len(tokens[i+1]) == 1:
            gt_probs[0][i] = 1
    lowest_prob_indices = torch.argsort(gt_probs[0][inputs_prompt_len - 1: len(gt_probs[0])], dim=-1, descending=False) + inputs_prompt_len - 1
    lowest_len = min(5, len(lowest_prob_indices))
    for i in range(lowest_len):
        probe_b = {}
        probe_b["prompt"] = inputs["input_ids"][0][:lowest_prob_indices[i] + 1]
        probe_b["output"] = inputs["input_ids"][0][lowest_prob_indices[i] + 1]
        probe_b["prompt"] = tokenizer.decode(probe_b["prompt"])
        probe_b["output"] = tokenizer.decode(probe_b["output"])
        # print(probe_b["prompt"])
        # print(probe_b["output"])
        probe_data.append(probe_b)
        # print(tokens[lowest_prob_indices[i] + 1])
        # print(gt_probs[0][lowest_prob_indices[i]])
        # print("*"*38)
    return probe_data


if __name__ == "__main__":
    model_path = "./checkpoints/Qwen-2.5-1.5B" # NOTE: Change
    data_path = "./processed_data/dolly/full/qwen/" # NOTE: Change
    model, tokenizer = get_model(model_path)
    train_data, valid_data = get_data(data_path)
    # probe_data_list = process_data_parallel(train_data, model, tokenizer)
    probe_data_list = []
    for d in tqdm(train_data):
        probe_data = get_probe_data(d, model, tokenizer)
        probe_data_list.append(probe_data)
    probe_data_dir = "./probe_data/dolly"
    create_directory_if_not_exists(probe_data_dir)
    probe_data_path = os.path.join(probe_data_dir, "train_Qwen-2.5-1.5B.json") # NOTE: Change
    with open(probe_data_path, "w", encoding="utf-8") as f:
        whole_data = []
        for idx, probe_data in enumerate(probe_data_list):
            json_data = {}
            json_data["id"] = idx + 1
            json_data["probes"] = []
            for probe_d in probe_data:
                json_data["probes"].append({
                    "prompt": probe_d["prompt"],
                    "output": probe_d["output"]
                })
            whole_data.append(json_data)
        json.dump(whole_data, f, ensure_ascii=False, indent=4)