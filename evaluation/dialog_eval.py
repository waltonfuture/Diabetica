import pandas as pd
import argparse
import torch
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def internlm_output(content, model, tokenizer):
    response, history = model.chat(tokenizer, content, history=[], max_new_tokens=2048)
    return response

def qwen_chat_output(content, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
        repetition_penalty=1.1
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def yi_chat_output(content, model, tokenizer):
    temp = "<|im_start|>user\n{contents}<|im_end|>\n<|im_start|>assistant\n"
    text = temp.format(contents=content)
    #print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
        do_sample=True,
        repetition_penalty=1.1,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def llama3_output(content, model, tokenizer):
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
        ]
    prompts = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([prompts], return_tensors="pt").to("cuda")
    input_length = model_inputs.input_ids.shape[1]
    generated_ids = model.generate(**model_inputs, max_new_tokens=2048,eos_token_id=terminators,)
    output = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
    return output


def main(args):
    model_path = args.path
    if "internlm" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()
        model_output = internlm_output
    elif "llama3" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
        model_output = llama3_output
    elif "yi" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        model_output = yi_chat_output
    elif "qwen" in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_output = qwen_chat_output
    model = model.eval()
    output_file_path=f'./{args.model}_eval_rules.jsonl'
    file_path = './data/120_benchmark.xlsx'

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        df = pd.read_excel(file_path)

        questions_correct = df['question'].tolist()
        rulehuman_correct = df['rulehuman'].tolist()
        for i, question in enumerate(tqdm(questions_correct)):
            output = {}
            response = model_output(question, model, tokenizer)
            output["question"] = question
            output["response"] = response
            output["rulehuman"] = rulehuman_correct[i]
            json.dump(output, output_file, ensure_ascii=False)
            output_file.write('\n')
            output_file.flush()
        output_file.close()
    print(model_path)

            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default ='qwen', type = str)
    parser.add_argument("--path", default = "", type = str)
    args = parser.parse_args()
    main(args)