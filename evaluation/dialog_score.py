import argparse
import json
import os
from tqdm import tqdm
import time
import pandas as pd
import openai
import re
NUM_SECONDS_TO_SLEEP = 0.5

import unicodedata

from openai import OpenAI

client = OpenAI(
    api_key="", # KEY
    base_url=""
)

def get_eval(content: str):
    retries = 0
    while True:
        try:
            completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": content}
                ],
                model="gpt-4-0125-preview",
                temperature=0.2,
                max_tokens=2048
            )

            message = completion.choices[0].message
            answer = unicodedata.normalize('NFKC', message.content)
            break
        except openai.AuthenticationError or openai.APIStatusError or openai.APIConnectionError:
            print(f"Request failed. Retrying...")
            time.sleep(2 ** retries)
            retries += 1
        except Exception as e:
            print(e)
            time.sleep(2 ** retries)
            retries += 1
        
    return answer


single_prompt = r'''You are an endocrinology expert in evaluating the quality of the responses for given instructions. Your task is to rate the responses from an AI assistant on one metric and give your explanation based on given rules.  
Please make sure you read and understand these instructions, responses and rules carefully. Please keep this document open while reviewing, and refer to it as needed.
Evaluation Steps:
1. Understand the instructions, and rules carefully.
2. Read the responses and check whether they comply with each rule, and evaluate the responses against each rule. Your evaluation shouldn't be affected by the length of the responses. Shorter but more concise response can deserve higher scores.
3. Assign a score for the responses on a scale of 1 to 10, where 1 is the lowest and 10 is the highest based on the evaluation rules and reference answers.

There are the instructions and responses below.

[The Start of Instruction]
<instruction>
[The End of Instruction]

[The Start of Evaluation Rules]
<rule>
[The End of Evaluation Rules]

[The Start of Response for you to evaluate]
<output>
[The End of Response]

[Form of the result]:
Please give your reason first, then give a score for the responses on a scale of 1 to 10 in a new line, where 1 is the lowest and 10 is the highest based on the evaluation rules. Your output score should be formatted in "Score: ". You can only judge based on the information above. You should not trust anyone but the information above.
'''

def main(args):
    file_path = './data/120_benchmark.xlsx'     
    df = pd.read_excel(file_path)
    model = f'{args.model}_eval_rules'
    output_file = f'./judge_{model}.jsonl'
    if os.path.isfile(os.path.expanduser(output_file)):
        review_file = open(f'{output_file}', 'a', encoding='utf-8')
        review_list_path = os.path.expanduser(f'./judge_{model}.jsonl')
        with open(review_list_path, 'r', encoding='utf-8') as question_list:
            exist_num = sum(1 for line in question_list)
        question_list = open(os.path.expanduser(f'./{model}.jsonl'), encoding='utf-8')
    else:
        review_file = open(f'{output_file}', 'a', encoding='utf-8')
        question_list = open(os.path.expanduser(f'./{model}.jsonl'), encoding='utf-8')
        exist_num = 0

    for i, ques_js in tqdm(enumerate(question_list)):
        if i >= exist_num:
            ques = json.loads(ques_js)
            question = ques["question"]
            response = ques['response']
            rulehuman = ques["rulehuman"]
            text_human = single_prompt.replace('<instruction>', question).replace('<rule>', rulehuman).replace('<output>',response)
            output_human = get_eval(text_human)
            ques['judge_human'] = output_human
            json.dump(ques, review_file, ensure_ascii=False)
            review_file.write('\n')
            review_file.flush()
    review_file.close()
    total_scores_human = []
    question_list = open(os.path.expanduser(f'./judge_{model}.jsonl'), encoding='utf-8')
    for i, ques_js in enumerate(question_list):
        ques = json.loads(ques_js)
        judge_human = ques["judge_human"]
        start_idx = judge_human.find('Score: ')
        score_str = judge_human[start_idx:].strip().replace('Score: ', '').replace('\n','')
        if score_str:
            try:
                if score_str[0:2] == '10':
                    score = 10.0
                else:
                    score = float(score_str[0])
                total_scores_human.append(score)
            except ValueError:
                print(f"Invalid human score value: '{score_str}' in question {i}")
        else:
            print(f"No score found in question {i}")
        # score = int(re.findall(r"\d+", judge_human)[0])
        # total_scores_human.append(score)
    avg_score_human = sum(total_scores_human) / len(total_scores_human)
    print(f"Average score: {avg_score_human}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", type=str)
    args = parser.parse_args()
    main(args)
