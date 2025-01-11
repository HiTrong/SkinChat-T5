# ====== Import Libraries ======
from simplet5 import SimpleT5
import torch
import transformers
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration

# ====== model path ======
model_path = "./model/"
new_tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
simplet5 = SimpleT5()
simplet5.model = model
simplet5.load_model(model_path, use_gpu=True)
simplet5.tokenizer = new_tokenizer

def get_response(prompt):
    prompt = "Trả lời câu hỏi này: " + prompt
    prompt = prompt.replace('\n','<newline>')
    response = simplet5.predict(prompt.replace('<newline>','\n'), skip_special_tokens=False)[0].replace('<pad> ','').replace('</s>', '').replace('<newline>','\n')
    return response