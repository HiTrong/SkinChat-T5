# ====== Import Libraries ======
import numpy as np
import pandas as pd
import os
import json
import timm
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import faiss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load VectorDB
image_index = faiss.read_index("./db/vectordb/image_index.faiss")
with open("./db/vectordb/image_mapping_data.json", "r", encoding="utf-8") as file:
    image_mapping_dict = json.load(file)
    
# Load embedding
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
model = timm.create_model('resnet50', pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Embedding Function
def embedding_wpath(path):
    image = Image.open(path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.forward_features(input_tensor)  # Sử dụng feature extraction của TIMM
        embedding = embedding.mean(dim=(2, 3))
        return embedding.cpu().numpy()
    
def embedding_wpil(image):
    image = image.convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.forward_features(input_tensor)  # Sử dụng feature extraction của TIMM
        embedding = embedding.mean(dim=(2, 3))
        return embedding.cpu().numpy()
    
# Find similar image
def Find_similar_image(image, k=5):
    result = []
    query_vector = embedding_wpil(image)
    faiss.normalize_L2(query_vector)
    distances, indices = image_index.search(query_vector, k)
    for i in range(len(indices[0])):
        id = str(indices[0][i])
        result.append(image_mapping_dict[id])
    return result
        
# Diagnosis Function
def diagnosis(image, k=5):
    result = Find_similar_image(image)
    image_path = result[0]["image_path"]
    diagnosis_name = result[0]["sick_name"]
    another_diagnosis = ""
    for i in range(1, len(result)):
        if result[i]["sick_name"] not in another_diagnosis and result[i]["sick_name"] != diagnosis_name:
            another_diagnosis += result[i]["sick_name"] + ", "
    another_diagnosis = another_diagnosis.strip()
    another_diagnosis += "..."
    print(image_path)
    return image_path, diagnosis_name, another_diagnosis
    