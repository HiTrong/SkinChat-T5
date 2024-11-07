import numpy as np 
import pandas as pd
import queue
import re
import time
from datetime import datetime
import os
import google.generativeai as genai
from tqdm.auto import tqdm
import argparse
from pathlib import Path

# Config API function
def config_API(API_KEY):
    genai.configure(api_key=API_KEY)
    
# Request function
def request(API_KEY, model_name="gemini-1.0-pro", prompt=None):
    try:
        if prompt is not None:
            config_API(API_KEY)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        else:
            print("Prompt must be NOT None!")
            return None
    except:
        print(f"LOGGING: Request failed at {datetime.now()}")
        return None

# clear response function
def clear_response(response):
    # Thay thế \n\n bằng \n
    response = re.sub(r'\n\n+', '\n', response)
    
    # Loại bỏ markdown như ** và *
    response = re.sub(r'\*\*|\*', '', response)
    
    return response

# extract text in response [START]...[END]
def extract_response(response):
    # Sử dụng regex để tìm tất cả các chuỗi nằm giữa [START] và [END]
    pattern = r'\[START\](.*?)\[END\]'
    conversations = re.findall(pattern, response, re.DOTALL)  # re.DOTALL để match cả xuống dòng

    # Loại bỏ các khoảng trắng thừa đầu và cuối chuỗi
    conversations = [conv.strip() for conv in conversations]
    
    return conversations

# Get Queue
def get_queue(API:str):
    API_queue = queue.Queue()
    model_list = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro"
    ]
    for model in model_list:
        API_queue.put((model, API))
    return API_queue

# Get model with API
def get_item(API_queue):
    model, api = API_queue.get()
    API_queue.put((model, api))
    return model, api

# Get Dataframe
def get_df(input_path):
    df = pd.read_csv(input_path)
    return df

# Save csv function
def save_csv(data:dict, path:str):
    df = pd.DataFrame(data)
    df = df.drop_duplicates().dropna()
    df = df.sample(frac=1)
    df.to_csv(path, index=False)
    
    
# Augmentation Function
def aug(API, input_csv, output):
    MAX_LIMIT_ERROR = 80
    error = 0
    data_dict = {
        "Question": [],
        "Answer": []
    }
    TEMPLATE = """Hướng dẫn: Bạn có vai trò tăng cường dữ liệu dựa trên nội dung do người dùng cung cấp. Hãy tăng cường dữ liệu bằng cách đổi cách sử dụng từ, đa dạng cách hỏi, cách trình bày hoặc tạo ra nhiều kịch bản liên quan cho thật là đa dạng.
Lưu ý: Mỗi đoạn hội thoại đều được bắt đầu bằng [START] và kết thúc là [END] không cần markdown, trình bày càng đa dạng càng tốt.
Nội dung cung cấp: {}
Hãy tăng cường dữ liệu lên 10 lần
Thứ kết quả tôi mong muốn có format như sau:
[START]Question: ...
Answer: ...[END]
Tiếp tục như vậy đến khi đủ yêu cầu.
"""
    API_queue = get_queue(API)
    df = get_df(input_csv)
    output_path = os.path.join(output, os.path.basename(input_csv))
    for i in tqdm(range(df.shape[0]), desc=f"Data Augmentation for {os.path.basename(input_csv)}"):
        question = df["Question"].iloc[i]
        answer = df["Answer"].iloc[i]
        str_supply = f"Question: {question}\nAnswer: {answer}"
        prompt = TEMPLATE.format(str_supply)
        while True:
            model, api = get_item(API_queue)
            response = request(api, model, prompt)
            if response is not None:
                break
            error += 1
            if error > MAX_LIMIT_ERROR:
                raise Exception("Reached MAX_LIMIT_ERROR!")
            time.sleep(10)
        response = clear_response(response)
        qas = extract_response(response)
        for qa in qas:
            index = qa.find("Answer: ")
            if index != -1:
                data_dict["Question"].append(qa[:index].strip().replace("Question: ",""))
                data_dict["Answer"].append(qa[index:].strip().replace("Answer: ",""))
        save_csv(data_dict, output_path)
        time.sleep(1)
    print("DONE!")

# Main
def main():
    parser = argparse.ArgumentParser(description="Argument parser")
    
    parser.add_argument("--API", type=str, required=True, help="Gemini API")
    parser.add_argument("--input_csv", type=Path, required=True, help="Đường dẫn tới file csv")
    parser.add_argument("--output", type=Path, required=True, help="Đường dẫn tới thư mục output")
    
    arg = parser.parse_args()
    
    name_csv = os.path.basename(arg.input_csv)

    if os.path.exists(os.path.join(arg.output, name_csv)):
        print("Data has already been augmented!")
        return
    
    aug(arg.API, arg.input_csv, arg.output)
    
    
if __name__ == "__main__":
    main()