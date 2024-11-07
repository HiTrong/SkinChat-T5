import numpy as np
import pandas as pd

data_dict = {
    "Question": [],
    "Answer": []
}

while True:
    input_str = input()
    if (input_str == ""):
        print("Bạn chưa nhập gì!")
    else:
        if "Question: " in input_str:
            data_dict["Question"].append(input_str.replace("Question: ","").strip())
        if "Answer: " in input_str:
            data_dict["Answer"].append(input_str.replace("Answer: ", "").strip())
    if len(data_dict["Answer"]) == len(data_dict["Question"]):
        df = pd.DataFrame(data_dict)
        df = df.dropna().drop_duplicates()
        df.to_csv("hello.csv", index=False)
