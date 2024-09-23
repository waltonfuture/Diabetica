import torch
import gradio as gr
import random
from transformers import TextIteratorStreamer
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    

def model_output(history, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    messages += history
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generation_config = model.generation_config
    generation_config.max_new_tokens=2048

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    Thread(target=model.generate, kwargs=dict(
                inputs=model_inputs.input_ids, streamer=streamer,
                generation_config=generation_config,
            )).start()

    return streamer

model_path = "WaltonFuture/Diabetica-7B"
model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
        ).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)
    
def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = model_output(history_openai_format, model, tokenizer)

    partial_message = ""
    for chunk in response:
        if chunk != '<':
            partial_message += chunk
            yield partial_message

gr.ChatInterface(predict).launch(share=True, share_server_address="my-gpt-wrapper.com:7000")