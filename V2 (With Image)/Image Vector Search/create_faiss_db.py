# Import Libraries
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

# Tạo chỉ mục với khoảng cách Euclidean
index = faiss.IndexFlatL2(2048)

# Load sick configuration
with open("skin_sick.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    
# Chuẩn hóa hình ảnh
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
# Load ResNet-50 từ TIMM
model = timm.create_model('resnet50', pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Embedding Function
def embedding(path):
    image = Image.open(path).convert('RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model.forward_features(input_tensor)  # Sử dụng feature extraction của TIMM
        embedding = embedding.mean(dim=(2, 3))
        return embedding.cpu().numpy()

# Embedding data and add to db
mapping_dict = {
}
count = 0
for sick_name in data:
    sick_config = data[sick_name]
    file_list = [f for f in os.listdir(sick_config["img_folder"]) if os.path.isfile(os.path.join(sick_config["img_folder"], f))]
    for img_name in tqdm(file_list, desc=f"Embedding {sick_name}"):
        path = os.path.join(sick_config["img_folder"], img_name)
        emb = embedding(path)
        faiss.normalize_L2(emb)
        index.add(emb)
        mapping_dict[count] = {
            "image_path": path,
            "sick_name": sick_name
        }
        count += 1

with open("./db/vectordb/image_mapping_data.json", "w", encoding="utf-8") as file:
    json.dump(mapping_dict, file, ensure_ascii=False, indent=4)

faiss.write_index(index, "./db/vectordb/image_index.faiss")

print("Hoàn tất!")


