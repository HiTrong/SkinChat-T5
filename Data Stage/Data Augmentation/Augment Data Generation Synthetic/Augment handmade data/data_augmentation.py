import numpy as np 
import pandas as pd
import queue
import re
import random
import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai

API_list = [
    "API-KEY1",
    "API-KEY2",
    "API-KEY3",
    "API-KEY4",
    "API-KEY5",
]

model_list = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro"
]

API_queue = queue.Queue()
for model in model_list:
    for api in API_list:
        API_queue.put((model, api, 1100))
        
# Get model with API
def get_item():
    model, api, number_of_requests = API_queue.get()
    if number_of_requests > 1:
        API_queue.put((model, api, number_of_requests-1))
    else:
        print(f"Model {model} của API {api} đã hết lượt request!")
    return model, api

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
        print("\nSomething went wrong! Try again!")
        return None
    
# clear response function
def clear_response(response):
    # Thay thế \n\n bằng \n
    response = re.sub(r'\n\n+', '\n', response)
    
    # Loại bỏ markdown như ** và *
    response = re.sub(r'\*\*|\*', '', response)
    
    return response

template = """Hướng dẫn: Bạn có vai trò tăng cường dữ liệu dựa trên nội dung do người dùng cung cấp. Hãy tăng cường dữ liệu bằng cách đổi cách sử dụng từ, cách trình bày hoặc tạo ra nhiều kịch bản liên quan cho thật là đa dạng.
Lưu ý: Mỗi đoạn hội thoại đều được bắt đầu bằng [START] và kết thúc là [END] không cần markdown, trình bày càng đa dạng càng tốt.
Nội dung cung cấp: {}
Hãy tăng cường dữ liệu lên 10 lần
Thứ kết quả tôi mong muốn có format như sau:
[START]Question: ...
Answer: ...[END]
Tiếp tục như vậy đến khi đủ yêu cầu.
"""

df = pd.read_csv("./Data_Synthetic_handmade.csv")
df = df[["Question", "Answer"]]
df = df.sample(frac=1)


# extract text in response [START]...[END]
def extract_response(response):
    # Sử dụng regex để tìm tất cả các chuỗi nằm giữa [START] và [END]
    pattern = r'\[START\](.*?)\[END\]'
    conversations = re.findall(pattern, response, re.DOTALL)  # re.DOTALL để match cả xuống dòng

    # Loại bỏ các khoảng trắng thừa đầu và cuối chuỗi
    conversations = [conv.strip() for conv in conversations]
    
    return conversations

# Save csv function
def save_csv(data:dict, path:str):
    df = pd.DataFrame(data)
    df = df.drop_duplicates().dropna()
    df = df.sample(frac=1)
    df.to_csv(path, index=False)
    
    
train_dict = {
    "Question": [],
    "Answer": []
}
valid_dict = {
    "Question": [],
    "Answer": []
}
test_dict = {
    "Question": [],
    "Answer": []
}

def aug(data):
    question, answer = data
    prompt = template.format(f"Question: {question}\nAnswer: {answer}")
    while True:
        model, api = get_item()
        response = request(api, model, prompt)
        if response is not None:
            break
        time.sleep(2)
    response = clear_response(response)
    qas = extract_response(response)
    for i in range(len(qas)):
        qa = qas[i].strip()
        index = qa.find("Answer: ")
        if index != -1:
            if i <= 7:
                train_dict["Question"].append(qa[:index].strip().replace("Question: ",""))
                train_dict["Answer"].append(qa[index:].strip().replace("Answer: ",""))
            elif i == 8:
                valid_dict["Question"].append(qa[:index].strip().replace("Question: ",""))
                valid_dict["Answer"].append(qa[index:].strip().replace("Answer: ",""))
                if random.randint(1, 100) > 90:
                    train_dict["Question"].append(qa[:index].strip().replace("Question: ",""))
                    train_dict["Answer"].append(qa[index:].strip().replace("Answer: ",""))
            elif i == 9:
                test_dict["Question"].append(qa[:index].strip().replace("Question: ",""))
                test_dict["Answer"].append(qa[index:].strip().replace("Answer: ",""))
                if random.randint(1, 100) > 90:
                    train_dict["Question"].append(qa[:index].strip().replace("Question: ",""))
                    train_dict["Answer"].append(qa[index:].strip().replace("Answer: ",""))
                
def data_augmentation():
    batch = []
    for i in tqdm(range(df.shape[0]), desc="Data Augmentation"):
        try:
            if len(batch) < 10 and i < df.shape[0] - 1:
                q = df["Question"].iloc[i]
                a = df["Answer"].iloc[i]
                batch.append((q,a))
            else:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    results = list(executor.map(aug, batch))
                save_csv(train_dict, "train_data_augmentation.csv")
                save_csv(valid_dict, "valid_data_augmentation.csv")
                save_csv(test_dict, "test_data_augmentation.csv")
                time.sleep(10)
                batch = []
        except:
            continue
    print("DONE!")
        
data_augmentation()

