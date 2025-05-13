import os
import json

import fire
import torch
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.utils.prompter import Prompter
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_data(data_path):
    data = []
    with open(data_path, "r", encoding="utf-8") as rf:
        data = json.load(rf)
    new_data = []
    for d in data:
        new_d = {}
        new_d["instruction"] = d["instruction"]
        new_d["input"] = d["question"]
        new_data.append(new_d)
    return new_data


def main(
    model_path: str = "/data2/jun/models/llama-2-7b-hf",
    lora_path: str = "/data2/jun/LLaMA-Factory/saves/gsm8k/llama2-7b-warm0.03",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": device},
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    def evaluate(
        instruction,
        input=None,
        temperature=0.9,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        # print("输入是:")
        # print(prompt)
        # print("------------------------------------------------")
        # prompt = instruction + input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=True,
            **kwargs,
        )

        # with torch.no_grad():
        #     model.eval()
        #     model_outputs = model(input_ids, use_cache=False)
        #     logits = model_outputs.logits
        #     print(logits)
        #     exit()

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        # print("输出是")
        # print(output)
        # print("---------------------------------")
        return prompter.get_response(output)

    # instruction=["Please answer the arithmetic question."]
    # inputs = ["Josh decides to try flipping a house.  He buys a house for $80,000 and then puts in $50,000 in repairs.  This increased the value of the house by 150%.  How much profit did he make?"]
    # instruction = instruction[0]
    # inputs = inputs[0]
    # ans = evaluate(
    #     instruction=instruction,
    #     input=inputs,
    #     temperature=0.1,
    #     top_p=0.75,
    #     top_k=40,
    #     num_beams=4,
    #     max_new_tokens=128,
    # )
    # print(ans)

    data = get_data("/data2/jun/datasets/gsm8k/test.json")
    for d in tqdm(data):
        ans = evaluate(
            instruction=d["instruction"],
            input=d["input"],
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
        )
        # print(ans)



if __name__ == "__main__":
    fire.Fire(main)