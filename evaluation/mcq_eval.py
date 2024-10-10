import pandas as pd
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def internlm_output(content, model, tokenizer):
    response, history = model.chat(tokenizer, content, history=[], max_new_tokens=2048)
    return response

def yi_chat_output(content, model, tokenizer):
    temp = "<|im_start|>user\n{contents}<|im_end|>\n<|im_start|>assistant\n"
    text = temp.format(contents=content)
    inputs = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    input_length = inputs.shape[1]
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=32)
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

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
        max_new_tokens=32,
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
    generated_ids = model.generate(**model_inputs, max_new_tokens=256,eos_token_id=terminators,)
    output = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
    return output

def main(args):
    model_path = args.path
    if args.model == "internlm":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()
        model_output = internlm_output
    elif args.model == "llama3":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to('cuda')
        model_output = llama3_output
    elif args.model == "yi":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        model_output = yi_chat_output
    elif args.model == "qwen":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_output = qwen_chat_output
    model = model.eval()
    scores = {}
    mcq_list = ['./evaluation/data/ZhiCheng_MCQ_A1.csv','./evaluation/data/ZhiCheng_MCQ_A2.csv'] # for Chinese
    # mcq_list = ['./data/ZhiCheng_MCQ_A1_translated.csv','./data/ZhiCheng_MCQ_A2_translated.csv'] # for English
    for i, file_path in enumerate(mcq_list):
        file_name = file_path[-6:-4]
        df = pd.read_csv(file_path)
        merged_df = pd.DataFrame(columns=['question', 'response','answer'])
        addPrompt = '\n请根据题干和选项，给出唯一的最佳答案。输出内容仅为选项英文字母，不要输出任何其他内容。不要输出汉字。'

        for index, row in tqdm(df.iterrows()):
            output = {}
            response = model_output(row['question'] + addPrompt, model, tokenizer)
            output["question"] = [row['question']]
            output["response"] = [response]
            output['answer'] = row['answer']
            df_output = pd.DataFrame(output)
            merged_df = pd.concat([merged_df, df_output])

        # calculate MCQ score
        merged_df['responseTrimmed'] = merged_df['response'].apply(lambda x: re.search(r'[ABCDE]', x).group() if re.search(r'[ABCDE]', x) else None)
        merged_df['check'] = merged_df.apply(lambda row: 1 if row['responseTrimmed'] == row['answer'][0] else 0, axis=1)
        score = merged_df['check'].mean()
        name = f"{args.model}{file_name}"
        scores[name] = score

    print(scores)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default ='qwen', type = str)
    parser.add_argument("--path", default = "", type = str)
    args = parser.parse_args()
    main(args)
