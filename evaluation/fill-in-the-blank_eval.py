import pandas as pd
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from bert_score import score
from collections import Counter
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

single_prompt = r'''请回答以下填空题，要求结合上下文在（）补充内容，仅输出填空内容，不输出解释：
<question>
'''

def compute_bleu(ref, candi):
    reference_words = word_tokenize(ref.lower())  
    candidate_words = word_tokenize(candi.lower())
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference_words], candidate_words, smoothing_function=smoothie)
    return bleu_score

def lcs_length(x, y):
    """
    Compute the length of the Longest Common Subsequence between two strings.
    """
    table = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] == y[j]:
                table[i+1][j+1] = table[i][j] + 1
            else:
                table[i+1][j+1] = max(table[i+1][j], table[i][j+1])
    return table[-1][-1]

def rouge_l(reference, summary):
    """
    Compute the ROUGE-L score between a reference and a summary.
    """
    lcs = lcs_length(reference, summary)
    precision = lcs / len(summary)
    recall = lcs / len(reference)
    
    if precision + recall == 0:
        return 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
def rouge_n(reference, summary, n=1):
    """
    Compute the ROUGE-N score between a reference and a summary.
    """
    def get_ngrams(text, n):
        return [tuple(text[i:i+n]) for i in range(len(text)-n+1)]

    ref_ngrams = Counter(get_ngrams(reference, n))
    summary_ngrams = Counter(get_ngrams(summary, n))

    intersection_ngrams = sum((ref_ngrams & summary_ngrams).values())
    total_ref_ngrams = sum(ref_ngrams.values())
    total_summary_ngrams = sum(summary_ngrams.values())

    if total_summary_ngrams == 0 or total_ref_ngrams == 0:
        return 0

    precision = intersection_ngrams / total_summary_ngrams
    recall = intersection_ngrams / total_ref_ngrams

    if precision + recall == 0:
        return 0

    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def rouge_1(reference, summary):
    """
    Compute the ROUGE-1 score.
    """
    return rouge_n(reference, summary, n=1)

def rouge_2(reference, summary):
    """
    Compute the ROUGE-2 score.
    """
    return rouge_n(reference, summary, n=2)

def compute_bert_score(refs, hyps, model_type='bert-base-multilingual-cased'):
    """
    Compute BERT Score between a list of reference texts and a list of hypothesis texts.
    """
    P, R, F1 = score(hyps, refs, lang='zh', model_type=model_type, verbose=True)
    return P, R, F1

def default(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

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
    input_file_path = f'./data/fill-in-the-blank.csv'

    data = pd.read_csv(input_file_path, encoding='gbk') 
    questions_list = data.loc[:, "question"].tolist()
    answers_list = data.loc[:, "answer"].tolist()
    bert_scores, rouge_scores, bleu_scores, rouge1, rouge2 = [], [], [], [], []
    for index, question in tqdm(enumerate(questions_list)):
        text = single_prompt.replace('<question>', question)
        output = model_output(text, model, tokenizer)
        answer = answers_list[index]
        _, _, F1 = compute_bert_score([answer], [output])
        bleu_score = compute_bleu(answer, output)
        rl = rouge_l(answer, output)
        r1 = rouge_1(answer, output)
        r2 = rouge_2(answer, output)
        bert_scores.append(float(F1.mean()))
        bleu_scores.append(bleu_score)
        rouge_scores.append(rl)
        rouge1.append(r1)
        rouge2.append(r2)

    print(f"bert score: {sum(bert_scores)/len(bert_scores)}")
    print(f"rouge score: {sum(rouge_scores)/len(rouge_scores)}")
    print(f"rouge1 score: {sum(rouge1)/len(rouge1)}")
    print(f"rouge2 score: {sum(rouge2)/len(rouge2)}")
    print(f"bleu score: {sum(bleu_scores)/len(bleu_scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default ='qwen', type = str)
    parser.add_argument("--path", default = "", type = str)
    args = parser.parse_args()
    main(args)