import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
from util import compute_score
import sys

MAX_INT = sys.maxsize


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)
    tokenizer = llm.get_tokenizer()
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_gt_responses = []
    problem_prompt_with_input = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
    problem_prompt_without_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )
    # print('promt =====', problem_prompt)
    def get_input(query):
        if query.find('\n') == -1:
            return ''
        return '\n'.join(query.split('\n')[1:])

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            # item = {'instruction': item['question'].split('\n')[0],
            # 'input': get_input(item['question']),
            # 'output': item['answer']}
            # if "input" not in item or len(item["input"]) == 0:
            #     temp_instr = problem_prompt_without_input.format(instruction=item["instruction"])
            # else:
            #     temp_instr = problem_prompt_with_input.format(instruction=item["instruction"], input=item["input"])
            # temp_instr += " Let's think step by step and output the final answer within \\boxed{}. "
            prompt_key = 'question' if 'question' in item else "problem"
            answer_key = 'answer' if 'answer' in item else 'solution'
            messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer WITHIN \\boxed{}."},
            {"role": "user", "content": item[prompt_key]}
            ]
            # 用 chat 模板生成 prompt
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            gsm8k_ins.append(prompt)
            # temp_ans = item['output'].split('#### ')[1]
            # temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_gt_responses.append(item[answer_key])

    # gsm8k_ins = gsm8k_ins[start:end]
    # gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    # stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction",
    #                "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048,)
    print('sampleing =====', sampling_params)
    result = []
    res_completions = []
    cnt = 0
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_gt_responses)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)
            cnt += 1

    invalid_outputs = []
    for idx, (prompt, completion, gt_response) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_gt_responses)):
        score = compute_score(completion, gt_response)
        # y_pred = extract_answer_number(completion)
        if score < 2:
            result.append(score)
            temp = {'question': prompt, 'output': completion, 'gt_response': gt_response}
            # invalid_outputs.append(temp)
        else:
            result.append(0)
            temp = {'question': prompt, 'output': completion, 'gt_response': gt_response}
            invalid_outputs.append(temp)
    acc = sum(result) / len(result)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    # print('start===', start, ', end====', end)
    print('length====', len(result), ', acc====', acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-1.5B-Instruct')  # model path
    parser.add_argument("--data_file", type=str, default='datasets/openai/gsm8k/main/test.jsonl')  # data path
    parser.add_argument("--start", type=int, default=0)  # start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=3200)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size,
               tensor_parallel_size=args.tensor_parallel_size)


# python scripts/qwen/eval/eval_gsm8k.py --model results/qwen2.5-7B/train/sft/sft_7B/e2-bs16-lr5e-06-G1-N8-NN1/6092 --data_file datasets/openai/gsm8k/main/test.jsonl