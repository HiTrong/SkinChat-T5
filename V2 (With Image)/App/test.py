import json, os
from pathlib import Path

with open("skin_sick.json", "r", encoding='utf-8') as f:
    data = json.load(f)
    
for sick_name in data:
    folder_path = Path(data[sick_name]["illustrate_folder"])
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Thư mục '{folder_path}' đã được tạo thành công.")