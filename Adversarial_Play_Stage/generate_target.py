from vllm import LLM, SamplingParams
import json
import tqdm
import math
import argparse
from transformers import AutoTokenizer
import random
random.seed(8888)

max_batch_size = 60000

data = []

class BatchIterator:
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i : i + self.batch_size]

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

def generate_prompt(prompt,tokenizer):
    summary = [
        {"role":"user","content":prompt},
    ]
    messages = tokenizer.apply_chat_template(summary, tokenize = False) + '<|start_header_id|>assistant<|end_header_id|>\n\n'
    return messages

def test(question_path, save_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with open(question_path, 'r', encoding='utf-8') as g:
        data = json.load(g)

    generate_prompt_fn = generate_prompt
    batching_inputs = tqdm.tqdm(
            BatchIterator(data, max_batch_size),
            desc="Batched inference",
        )

    stop = ['USER:']
    sampling_params = SamplingParams(temperature=0.95, max_tokens=4096, stop = stop)
    llm = LLM(model=model_path, tensor_parallel_size=4)
    output_text = []

    for batched_inputs in zip(batching_inputs):
        messages = [
                generate_prompt_fn(
                    prompt = ex_input['Example'],
                    tokenizer=tokenizer
                )
                for ex_input in batched_inputs[0]
            ]
        completions_list = [[] for _ in range(len(messages))]
        outputs = llm.generate(messages, sampling_params)

        for idx, output in enumerate(outputs):
            # import pdb;pdb.set_trace()
            generated_text = output.outputs[0].text
            completions_list[idx].append(generated_text)
        
        for task, prompt,completions in zip(batched_inputs[0],messages, completions_list):
                output_text.append({"theme task":task['theme task'],
                                    'Strategy of the PDF':task['Strategy of the PDF'],
                                    "Method Definition": task["Method Definition"],
                                    "Example": task['Example'],
                                    'Source': task['Source'],
                                'prompt': prompt, 
                                'completions': completions[0]})

    with open(save_path, 'w') as f:
        json.dump(output_text, f,indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--question_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    parser.add_argument('--save_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    parser.add_argument('--model_path', type=str,
                        default="",
                        help='model name in the hub or local path')
    args = parser.parse_args()
    test(args.question_path, args.save_path, args.model_path)