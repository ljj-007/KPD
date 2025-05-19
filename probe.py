import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import json
import argparse
import random

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed

from data_utils.prompt_datasets import PromptDataset
from data_utils.lm_datasets import LMTrainDataset
from EasyEdit.easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./checkpoints/llama2-7b/")
    parser.add_argument("--ds-config-path", type=str, default="./configs/deepspeed/ds_config.json")
    parser.add_argument("--data-path", type=str, default="./processed_data/dolly/full/llama2/")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--min-prompt-length", type=int, default=128)
    parser.add_argument("--model-type", type=str, default="llama2")
    parser.add_argument("--json-data", type=bool, default=True)
    parser.add_argument("--bin-data", type=bool, default=False)
    parser.add_argument("--txt-data", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    return args


def get_data(data_path):
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
    for d in train_data[:10]:
        print(d)


def probe_model(args):

    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map=None)
    with open(args.ds_config_path, "r") as f:
        ds_config = json.load(f)
    ds_config["steps_per_print"] = 1
    ds_config["zero_optimization"]["stage"] = 0
    model, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        mpu=None,
        config_params=ds_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    train_data, valid_data = get_data(args.data_path)
    # data = PromptDataset(args, tokenizer, "valid", data_path=args.data_path)
    # rng_sample = random.Random(args.seed)
    # data["train"] = LMTrainDataset(args, tokenizer, args.data_path, "train", -1, 1, rng_sample)
    # data["dev"] = LMTrainDataset(args, tokenizer, args.data_path, "valid", -1, 1, rng_sample)
    # print(len(data["train"])) # 14011
    # print(len(data["dev"])) # 1000
    
    kn = KnowledgeNeurons(model, tokenizer, model_type="llama")

    tmp_results = []
    for d in train_data:
        tmp_results += kn.get_coarse_neurons(prompt=d["prompt"], ground_truth=d["output"], batch_size=1, steps=10, adaptive_threshold=0.3)
        print(tmp_results) # [30, 8026], [31, 2161]
        break

    



if __name__ == "__main__":
    args = init_parser()
    probe_model(args)