# Import Libraries
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import faiss
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


index = faiss.read_index("./db/index.faiss")

with open("./db/mapping_data.json", "r", encoding="utf-8") as file:
    mapping_dict = json.load(file)


# Load embedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
model.to(device)

# Embedding Function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embedding(question):
    encoded_input = tokenizer(question, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy()


k = 10
query_vector = embedding("Chiều nay đi uống cà phê không friend")
faiss.normalize_L2(query_vector)
distances, indices = index.search(query_vector, k)

print("Khoảng cách đến các vector gần nhất:", distances)
print("Các câu hỏi gần nhất:\n", "\n".join([mapping_dict[str(i)]["Question"] for i in indices[0]]))