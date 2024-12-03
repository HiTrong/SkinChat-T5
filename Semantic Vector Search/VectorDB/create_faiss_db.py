# Import Libraries
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import faiss

# Tạo chỉ mục với khoảng cách Euclidean
index = faiss.IndexFlatL2(384)

# Read csv
df = pd.read_csv("./dataset/skinchat_train.csv")

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


# Embedding data and add to db
mapping_dict = {
}
count = 0
for i in tqdm(range(df.shape[0]), desc="Creating vectordb - Faiss"):
    question = str(df["Question"].iloc[i])
    answer = str(df["Answer"].iloc[i])
    if question is None or answer is None or question == "nan" or answer == "nan":
        continue
    data = embedding(question)
    faiss.normalize_L2(data)
    index.add(data)
    mapping_dict[count] = {
        "Question": question,
        "Answer": answer
    }
    count += 1
with open("./db/mapping_data.json", "w", encoding="utf-8") as file:
    json.dump(mapping_dict, file, ensure_ascii=False, indent=4)
    
faiss.write_index(index, "./db/index.faiss")

print("Hoàn tất!")
    
