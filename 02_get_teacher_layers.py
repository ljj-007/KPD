import json
import time
import argparse
import gc
import random
random.seed(10)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import deepspeed
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from EasyEdit.easyeditor.models.kn.knowledge_neurons.knowledge_neurons import KnowledgeNeurons


def move_inputs_to_device(tokenizer, model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(model.device) for k, v in inputs.items()}


def get_model(args):
    """ huggingface + accelerate + deepspeed """

    with init_empty_weights():
        config = AutoConfig.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_config(config)

    model = load_checkpoint_and_dispatch(
        model,
        args.model_path,
        device_map="auto",
        dtype=torch.float16,
    )

    with open(args.ds_config_path, "r") as f:
        ds_config = json.load(f)
    ds_config["steps_per_print"] = 1
    ds_config["zero_optimization"]["stage"]=0

    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        config_params=ds_config,
        optimizer=None,
        lr_scheduler=None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    return model_engine, tokenizer


def get_data(data_path):
    with open(data_path, "r", encoding="utf-8") as rf:
        probe_data_list = json.load(rf)
    return probe_data_list


def init_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-name", )
    parser.add_argument("--model-path", type=str, default="./checkpoints/llama2-7b") # NOTE: Change
    parser.add_argument("--ds-config-path", type=str, default="./configs/deepspeed/ds_config.json")
    parser.add_argument("--data-path", type=str, default="./probe_data/dolly/train_TinyLlama-1.1B.json") # NOTE: Change
    parser.add_argument("--probe-teacher-data-path", type=str, default="./probe_teacher_data/llama2/train.json") # NOTE: Change
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--min-prompt-length", type=int, default=128)
    parser.add_argument("--model-type", type=str, default="llama2") # NOTE: Change
    parser.add_argument("--json-data", type=bool, default=True)
    parser.add_argument("--bin-data", type=bool, default=False)
    parser.add_argument("--txt-data", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--local_rank", type=int, default=-1, help="deepspeed local rank")
    parser.add_argument("--deepspeed", action="store_true", help="deepspeed")
    args = parser.parse_args()
    return args


def probe_teacher_model(args):
    model, tokenizer = get_model(args)
    probe_data_list = get_data(args.data_path)
    probe_teacher_data = []
    random.shuffle(probe_data_list)
    probe_data_list = probe_data_list[:100]
    with open(args.probe_teacher_data_path, "a", encoding="utf-8") as af:
        for idx, probe_data in enumerate(probe_data_list):
            kn = KnowledgeNeurons(model, tokenizer, model_type="llama", device=model.device)
            start_time = time.time()
            results = []
            try:
                for probe in probe_data["probes"]:
                    prompt = probe["prompt"]
                    output = probe["output"]
                    results += kn.get_coarse_neurons(prompt=prompt, ground_truth=output, batch_size=1, steps=5, adaptive_threshold=0.3)
                end_time = time.time()
                print(f"probe {idx} data")
                print(f"one data time:{end_time - start_time}s")
                print(results)
                save_ans = {}
                save_ans["id"] = idx
                save_ans["probes"] = probe_data["probes"]
                save_ans["layers"] = results
                af.write(json.dumps(save_ans))
                af.write("\n")
                # probe_teacher_data.append(save_ans)
            except:
                continue
            del results
            kn.model.zero_grad()
            model.zero_grad()
            del kn
            torch.cuda.empty_cache()
    # with open(args.probe_teacher_data_path, "w", encoding="utf-8") as wf:
    #     json.dump(probe_teacher_data, wf, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    args = init_parser()
    probe_teacher_model(args)