from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained("/data2/jun/my-distill/checkpoints/llama2")

print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)

model = AutoModelForCausalLM.from_pretrained("/data2/jun/my-distill/checkpoints/llama2",
                                            torch_dtype=torch.float16,
                                            device_map="auto")
model.to(device)

# prompt = "Hey, are you conscious? Can you talk to me?"
# prompt = "What is the capital of China?"
instruction = "What is the capital of China?"
prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")
inputs.to(device)


generate_ids = model.generate(inputs.input_ids, max_length=60)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(output)