import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import math
import argparse
import os

def moderate(chat,tokenizer,model,device):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

def test(question_path, save_path, model_path, group_rank, group_size):
    model_id = model_path
    device = "cuda"
    dtype = torch.bfloat16
    with open(question_path, 'r', encoding='utf-8') as g:
        data = json.load(g)
    data = data[group_rank::group_size]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    
    save_file = []
    for item in tqdm(data):
        format = [
        {"role": "user", "content": item['theme task']},
        {"role": "assistant", "content": item['completions']},
        ]
        result = moderate(format, tokenizer, model, device)
        save_file.append({"theme task":item['theme task'],
                          'Strategy of the PDF':item['Strategy of the PDF'],
                          "Method Definition": item["Method Definition"],
                          "Example": item['Example'],
                          'Source': item['Source'],
                         "prompt":item['prompt'],
                         "completions": item['completions'],
                         'judge': result})
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + f'guard_{group_rank}.json', 'w') as f:
        json.dump(save_file, f,indent=2)

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
    # parser.add_argument('--N', type=int, default=30)
    parser.add_argument('--group_rank', type=int, default=0)
    parser.add_argument('--group_size', type=int, default=1) 
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.group_rank}'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()
    test(args.question_path, args.save_path, args.model_path, args.group_rank, args.group_size)