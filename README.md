# Diabetica (SCI-FM @ ICLR 2025)
<div align="center">
<h2>
    Diabetica: Adapting Large Language Model to Enhance Multiple Medical Tasks in Diabetes Care and Management
</h2>


</div>

<p align="center">
⬇️ <a href="https://huggingface.co/WaltonFuture/Diabetica-7B" target="_blank">7B Model</a> ｜⬇️ <a href="https://huggingface.co/WaltonFuture/Diabetica-1.5B" target="_blank">1.5B Model</a> ｜⬇️ <a href="https://huggingface.co/WaltonFuture/Diabetica-o1" target="_blank">o1 like Model</a> ｜📃 <a href="https://arxiv.org/pdf/2409.13191" target="_blank">Paper</a> <br>
</p>

## News

🔥 [2025.3] We release the SFT datasets for Diabetica and Diabetica-o1. Please check them at [Diabetica-SFT](https://huggingface.co/datasets/WaltonFuture/Diabetica-SFT) and [Diabetica-o1-SFT](https://huggingface.co/datasets/WaltonFuture/Diabetica-o1-SFT).

🔥 [2025.3] We release an o1 like model ([Diabetica-o1](https://huggingface.co/WaltonFuture/Diabetica-o1)). 

🔥 [2025.3] Our paper is accepted by SCI-FM @ ICLR 2025.

🔥 [2024.9] We release the Diabetica family ([7B](https://huggingface.co/WaltonFuture/Diabetica-7B) and [1.5B](https://huggingface.co/WaltonFuture/Diabetica-1.5B) models) and benchmarks.


## Introduction

Hello! Welcome to the repository for [Diabetica](https://arxiv.org/pdf/2409.13191). 

Our study introduced a reproducible framework for developing a specialized LLM capable of handling various diabetes tasks. We present three key contributions: 

- High-performance domain-specific model: Compared with previous generic LLMs, our model Diabetica, showed superior performance across a broad range of diabetes-related tasks, including diagnosis, treatment recommendations, medication management, lifestyle advice, patient education, and so on.

- Reproducible framework: We offered a detailed method for creating specialized medical LLMs using open-source models, curated disease-specific datasets, and fine-tuning techniques. This approach can be adapted to other medical fields, potentially accelerating AI-assisted care development.

- Comprehensive evaluation: We designed comprehensive benchmarks and conducted clinical trials to validate the model's effectiveness in clinical applications. This ensured our model's practical utility and sets a new standard for evaluating AI tools in diabetes care.


<div align=center>
<img src="assets/procedure.jpg"  width = "90%" alt="Diabetica" align=center/>
</div>


## Performance

Compared with popular open-source models and closed-source models (including GPT-4 and Claude-3.5), Diabetica showed impressive performance on diabetes benchmarks. Here, we present some of the results.

- **Multiple-choice questions**: Diabetica-7B has an 87.2% accuracy level, significantly surpassing all the other models, including GPT-4 and Claude-3.5.

<div align=center>
<img src="assets/mcq.png"  width = "80%" alt="Diabetica" align=center/>
</div>

- **Fill-in-the-blank questions**: The performance of Diabetica-7B is superior to all other open-sourced models with similar sizes across all metrics. It is also comparable with state-of-the-art close-source models, such as GPT-4 and Claude-3.5.

<div align=center>
<img src="assets/FB.jpg"  width = "80%" alt="Diabetica" align=center/>
</div>

- **Open-ended dialogues**: Diabetica-7B outperforms other similarly sized open-sourced LLMs by using fine-tuning through a self-distillation pipeline, without the need for RLHF.

<div align=center>
<img src="assets/dialog.jpg"  width = "80%" alt="Diabetica" align=center/>
</div>


## Model

### Model Access

Our models are now available on Huggingface.

| Model          | Backbone           | Checkpoint    |
| -------------- | ------------------ | ------------- |
| Diabetica-7B  | Qwen2-7B-Instruct  | [HF Link](https://huggingface.co/WaltonFuture/Diabetica-7B) |
| Diabetica-1.5B  | Qwen2-1.5B-Instruct  | [HF Link](https://huggingface.co/WaltonFuture/Diabetica-1.5B) |
| Diabetica-o1  | Qwen2.5-7B-Instruct  | [HF Link](https://huggingface.co/WaltonFuture/Diabetica-o1) |

### Setup

```bash
pip install -r requirements.txt
```

### Model Inference

```bash
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" # the device to load the model onto
model_path = 'WaltonFuture/Diabetica-7B'

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def model_output(content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=2048,
        do_sample=True,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

prompt = "Hello! Please tell me something about diabetes."

response = model_output(prompt)
print(response)
```

### Inference with Web Demo

```bash
python web_demo.py
```

## Data

Our datasets are now available on Huggingface.

| Dataset          |  Link    |
| -------------- | ------------- |
| Diabetica-SFT  |  [HF Link](https://huggingface.co/datasets/WaltonFuture/Diabetica-SFT) |
| Diabetica-o1-SFT  |  [HF Link](https://huggingface.co/WaltonFuture/Diabetica-1.5B) |


## Evaluation

### Multiple Choice Questions

Evaluation script for the MCQ benchmark.

```bash
bash scripts/MCQ.sh
```

### Fill-in-the-Blank Questions

Evaluation script for the Fill-in-the-Blank benchmark.

```bash
bash scripts/Fill-in-the-Blank.sh
```


### Open-Ended Dialogues

Evaluation script for the Open-ended Dialog benchmark.

```bash
bash scripts/Dialog.sh
```

## Acknowledgment

The Diabetica family is built upon the amazing [Qwen2](https://huggingface.co/collections/Qwen/qwen2-6659360b33528ced941e557f) family.

This repository is built upon [HuatuoGPT-II](https://github.com/FreedomIntelligence/HuatuoGPT-II).

## Contact

Please contact Lai Wei (waltonfuture@sjtu.edu.cn) or Zhen Ying (zying16@fudan.edu.cn) if needed.

## Citation
```
@article{wei2024adapted,
  title={An adapted large language model facilitates multiple medical tasks in diabetes care},
  author={Wei, Lai and Ying, Zhen and He, Muyang and Chen, Yutong and Yang, Qian and Hong, Yanzhe and Lu, Jiaping and Li, Xiaoying and Huang, Weiran and Chen, Ying},
  journal={arXiv preprint arXiv:2409.13191},
  year={2024}
}
```

