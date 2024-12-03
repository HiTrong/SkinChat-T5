# ====== Import Libraries ======
import numpy as np
import pandas as pd
import joblib
import json
import random
import re
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import faiss
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load VectorDB
index = faiss.read_index("./db/index.faiss")
with open("./db/mapping_data.json", "r", encoding="utf-8") as file:
    mapping_dict = json.load(file)

# Load model (Unrelated and Correct)
correct_model = joblib.load("./model/correct_model.pkl")
unrelated_model = joblib.load("./model/unrelated_model.pkl")

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

# Evaluate Function
def calculate_f1_score(true_answer, predicted_answer):
    # Tiền xử lý: loại bỏ dấu câu và đưa về chữ thường
    true_tokens = re.findall(r'\w+', true_answer.lower())
    pred_tokens = re.findall(r'\w+', predicted_answer.lower())
    
    # Tìm các từ chung giữa hai câu
    common_tokens = set(true_tokens) & set(pred_tokens)
    
    if not common_tokens:
        return 0.0
    
    # Tính Precision và Recall
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(true_tokens)
    
    # Tính F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def calculate_rouge_scores(true_answer, predicted_answer):
    # Khởi tạo ROUGE Scorer với ROUGE-1, ROUGE-2, và ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(true_answer, predicted_answer)
    
    # Trích xuất từng chỉ số ROUGE và trả về kết quả
    rouge_1 = scores['rouge1'].fmeasure
    rouge_2 = scores['rouge2'].fmeasure
    rouge_l = scores['rougeL'].fmeasure
    
    return rouge_1, rouge_2, rouge_l


# Search simmilar question
def Find_similar_qa(question, k = 10):
    similar_question = []
    query_vector = embedding(question)
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, k)
    for i in range(len(indices[0])):
        id = str(indices[0][i])
        yield mapping_dict[id]


unrelated_responses = [
    "Dường như câu hỏi của bạn không liên quan lắm đến vấn đề da liễu. Vui lòng chỉ hỏi những vấn đề da liễu nhé!",
    "Có vẻ như câu hỏi của bạn không thuộc lĩnh vực da liễu. Bạn có thể hỏi về các vấn đề khác liên quan đến da không?",
    "Xin lỗi, tôi chỉ có thể hỗ trợ các câu hỏi liên quan đến da liễu. Vui lòng đặt câu hỏi về da liễu nhé!",
    "Rất tiếc, câu hỏi của bạn không liên quan đến chủ đề da liễu. Hãy hỏi tôi về các vấn đề da liễu nhé!",
    "Dường như câu hỏi này không thuộc phạm vi chuyên môn của tôi. Bạn có thể hỏi về các vấn đề da liễu không?",
    "Tôi xin lỗi, nhưng câu hỏi của bạn không thuộc lĩnh vực da liễu. Hãy thử hỏi về các vấn đề liên quan đến da nhé!",
    "Có vẻ như câu hỏi không liên quan đến da liễu. Vui lòng hỏi về các chủ đề da liễu để tôi có thể hỗ trợ bạn tốt nhất.",
    "Xin lỗi, tôi chỉ có thể giải đáp các thắc mắc về da liễu. Bạn có thể hỏi về vấn đề khác liên quan đến da không?",
    "Tôi rất tiếc, câu hỏi này không nằm trong lĩnh vực da liễu. Vui lòng đặt câu hỏi về các vấn đề da liễu nhé!",
    "Có vẻ như câu hỏi này không phù hợp với chủ đề da liễu. Hãy đặt câu hỏi liên quan đến các vấn đề về da nhé!",
    "Xin lỗi, tôi chỉ hỗ trợ về lĩnh vực da liễu. Vui lòng hỏi về da để tôi có thể giúp bạn tốt hơn!",
    "Câu hỏi này có vẻ không liên quan đến da liễu. Bạn có thể hỏi về các vấn đề da để được hỗ trợ nhé!",
    "Tôi xin lỗi, nhưng câu hỏi này không thuộc phạm vi da liễu. Hãy thử hỏi tôi về da và các vấn đề liên quan nhé!",
    "Có vẻ như câu hỏi của bạn không phù hợp với chủ đề da liễu. Hãy hỏi tôi về các vấn đề da liễu nhé!",
    "Xin lỗi, câu hỏi của bạn không thuộc lĩnh vực da liễu. Bạn có thể đặt câu hỏi khác về da để được giải đáp nhé!",
    "Tôi rất tiếc, câu hỏi của bạn không phù hợp với lĩnh vực da liễu. Vui lòng hỏi về các vấn đề da nhé!",
    "Có vẻ như câu hỏi này không thuộc lĩnh vực da liễu. Bạn có thể đặt câu hỏi khác liên quan đến da không?",
    "Xin lỗi, tôi chỉ có thể trả lời các câu hỏi về da liễu. Bạn có thể hỏi về các vấn đề liên quan đến da không?",
    "Rất tiếc, câu hỏi của bạn không thuộc chủ đề da liễu. Hãy thử hỏi về các vấn đề da để tôi có thể hỗ trợ bạn.",
    "Dường như câu hỏi này không thuộc lĩnh vực da liễu. Bạn có thể hỏi tôi về các vấn đề liên quan đến da nhé!"
]

wrong_responses = [
    "Dường như tôi không thể đưa ra câu trả lời chính xác cho câu hỏi này của bạn. Tuy nhiên để phục vụ bạn tốt nhất có thể, tôi xin được phép nhận thông tin email của bạn. Chúng tôi sẽ liên hệ chuyên gia của SkinChat để trả lời cho bạn sớm nhất có thể! Xin cảm ơn!",
    "Xin lỗi, có vẻ như câu trả lời của tôi chưa đáp ứng được kỳ vọng của bạn. Nếu bạn có thể cung cấp email, chúng tôi sẽ sắp xếp để chuyên gia của SkinChat hỗ trợ bạn chi tiết hơn!",
    "Rất xin lỗi vì câu trả lời chưa làm bạn hài lòng. Nếu bạn cung cấp email, chúng tôi sẽ chuyển câu hỏi của bạn đến chuyên gia da liễu và liên hệ lại sớm nhất!",
    "Có lẽ câu trả lời của tôi chưa hoàn toàn phù hợp. Để hỗ trợ bạn tốt hơn, bạn có thể để lại email, và đội ngũ chuyên gia sẽ phản hồi chi tiết hơn!",
    "Xin lỗi nếu câu trả lời của tôi chưa đạt yêu cầu. Nếu bạn đồng ý cung cấp email, chúng tôi sẽ liên hệ để giải đáp đầy đủ hơn từ chuyên gia da liễu!",
    "Tôi rất tiếc nếu câu trả lời này chưa hoàn toàn chính xác. Hãy cho phép tôi nhận email của bạn để chúng tôi có thể yêu cầu chuyên gia phản hồi nhanh chóng!",
    "Rất xin lỗi vì câu trả lời của tôi chưa đạt được mong đợi của bạn. Nếu bạn có thể cung cấp email, chúng tôi sẽ nhanh chóng liên hệ chuyên gia để hỗ trợ bạn tốt hơn!",
    "Dường như câu trả lời của tôi chưa chính xác. Nếu bạn vui lòng để lại email, chúng tôi sẽ yêu cầu chuyên gia tư vấn chi tiết cho bạn.",
    "Xin lỗi, có thể câu trả lời của tôi chưa đủ chi tiết. Hãy để lại email của bạn, và chúng tôi sẽ liên hệ chuyên gia da liễu hỗ trợ sớm nhất!",
    "Có vẻ câu trả lời của tôi chưa làm bạn hài lòng. Nếu bạn có thể cung cấp email, chúng tôi sẽ chuyển câu hỏi của bạn đến chuyên gia để hỗ trợ thêm.",
    "Tôi xin lỗi nếu câu trả lời này chưa phù hợp. Để giúp bạn tốt nhất, hãy cung cấp email để chúng tôi có thể đưa chuyên gia liên hệ lại sớm!",
    "Rất xin lỗi nếu câu trả lời này chưa đáp ứng mong đợi của bạn. Hãy để lại email để chúng tôi có thể nhờ chuyên gia phản hồi bạn sớm nhất!",
    "Tôi rất tiếc nếu câu trả lời chưa đầy đủ. Nếu bạn cung cấp email, đội ngũ chuyên gia sẽ liên hệ để hỗ trợ thêm cho bạn!",
    "Xin lỗi, có vẻ câu trả lời của tôi chưa thỏa mãn nhu cầu của bạn. Hãy cung cấp email để chuyên gia da liễu phản hồi sớm cho bạn!",
    "Rất tiếc nếu câu trả lời chưa chính xác. Bạn vui lòng cho tôi biết email để đội ngũ chuyên gia của SkinChat có thể hỗ trợ thêm.",
    "Xin lỗi vì câu trả lời của tôi chưa làm bạn hài lòng. Nếu có thể, hãy cho phép tôi nhận email để chúng tôi nhờ chuyên gia phản hồi đầy đủ hơn!",
    "Có vẻ câu trả lời của tôi chưa đủ chi tiết. Để phục vụ bạn tốt hơn, bạn có thể để lại email để chúng tôi nhờ chuyên gia hỗ trợ.",
    "Rất xin lỗi nếu câu trả lời chưa đáp ứng mong đợi. Bạn có thể cho tôi email để chúng tôi nhờ chuyên gia phản hồi chính xác hơn.",
    "Xin lỗi vì câu trả lời này có thể chưa đạt yêu cầu. Nếu bạn đồng ý cung cấp email, chúng tôi sẽ liên hệ chuyên gia để giúp bạn.",
    "Tôi rất tiếc nếu câu trả lời này chưa làm bạn hài lòng. Hãy cho tôi biết email của bạn, và chúng tôi sẽ nhờ chuyên gia SkinChat hỗ trợ thêm!",
    "Xin lỗi, tôi chưa được học hoặc huấn luyện để trả lời câu hỏi này. Tuy nhiên, bạn có thể để lại email và chúng tôi sẽ nhờ chuyên gia của SkinChat liên hệ để hỗ trợ bạn sớm nhất!",
    "Rất tiếc, tôi chưa được huấn luyện về câu hỏi này. Hãy cho phép tôi nhận email của bạn để chúng tôi có thể yêu cầu chuyên gia hỗ trợ bạn kỹ càng hơn.",
    "Tôi rất xin lỗi, có vẻ câu hỏi này nằm ngoài phạm vi kiến thức của tôi. Bạn có thể cung cấp email để chúng tôi sắp xếp chuyên gia tư vấn thêm cho bạn.",
    "Có vẻ câu hỏi này không thuộc phạm vi kiến thức của tôi. Nếu bạn cung cấp email, chúng tôi sẽ liên hệ chuyên gia để hỗ trợ bạn chi tiết hơn.",
    "Xin lỗi, tôi chưa được đào tạo để trả lời câu hỏi này. Bạn có thể để lại email, và chúng tôi sẽ kết nối bạn với chuyên gia của SkinChat.",
    "Tôi rất tiếc vì câu hỏi này không nằm trong phạm vi đã được huấn luyện. Nếu bạn vui lòng cung cấp email, chuyên gia của chúng tôi sẽ liên hệ để hỗ trợ bạn.",
    "Xin lỗi, tôi chưa có đủ kiến thức để trả lời câu hỏi này. Hãy để lại email của bạn, và chuyên gia của chúng tôi sẽ giải đáp giúp bạn.",
    "Rất tiếc, câu hỏi của bạn vượt quá khả năng hiện tại của tôi. Vui lòng cho tôi biết email, và chúng tôi sẽ nhờ chuyên gia SkinChat phản hồi lại sớm nhất.",
    "Tôi chưa được huấn luyện để trả lời câu hỏi này, rất xin lỗi bạn. Hãy để lại email, và chúng tôi sẽ sắp xếp chuyên gia liên hệ để hỗ trợ thêm.",
    "Có vẻ như câu hỏi này nằm ngoài phạm vi hiểu biết của tôi. Bạn có thể cung cấp email, chúng tôi sẽ nhờ chuyên gia phản hồi lại bạn chi tiết nhất.",
    "Rất xin lỗi vì tôi chưa được đào tạo để trả lời câu hỏi này. Nếu bạn vui lòng để lại email, chúng tôi sẽ sắp xếp chuyên gia phản hồi cho bạn.",
    "Xin lỗi, tôi chưa được học về câu hỏi này. Bạn có thể để lại email, chúng tôi sẽ liên hệ chuyên gia hỗ trợ bạn chi tiết hơn.",
    "Có lẽ câu hỏi này nằm ngoài kiến thức hiện tại của tôi. Nếu bạn để lại email, chúng tôi sẽ yêu cầu chuyên gia SkinChat phản hồi giúp bạn.",
    "Tôi rất tiếc vì chưa có khả năng trả lời câu hỏi này. Hãy để lại email của bạn, và chúng tôi sẽ nhờ chuyên gia giải đáp thêm cho bạn.",
    "Xin lỗi, câu hỏi này không thuộc phạm vi đã học của tôi. Bạn có thể cung cấp email, và chúng tôi sẽ kết nối bạn với chuyên gia.",
    "Rất xin lỗi, tôi chưa được huấn luyện để trả lời câu hỏi này. Vui lòng cho tôi biết email để chúng tôi nhờ chuyên gia tư vấn thêm cho bạn.",
    "Xin lỗi, tôi chưa có kiến thức về câu hỏi này. Hãy để lại email, chúng tôi sẽ nhờ chuyên gia SkinChat phản hồi nhanh chóng cho bạn.",
    "Tôi rất tiếc vì chưa có kiến thức để trả lời câu hỏi này. Nếu bạn để lại email, chuyên gia của chúng tôi sẽ hỗ trợ bạn chi tiết hơn.",
    "Có vẻ câu hỏi này nằm ngoài phạm vi hiểu biết của tôi. Bạn vui lòng để lại email, chúng tôi sẽ nhờ chuyên gia của SkinChat phản hồi lại sớm nhất.",
    "Xin lỗi, tôi chưa được đào tạo về câu hỏi này. Hãy cho tôi biết email của bạn, và chúng tôi sẽ nhờ chuyên gia SkinChat phản hồi chi tiết."
]

# Evaluate the response
def Evaluate_response(question, response):
    related = False
    for similar_question in Find_similar_qa(question):
        f1 = calculate_f1_score(similar_question["Question"], question)
        rouge_1, rouge_2, rouge_l = calculate_rouge_scores(similar_question["Question"], question)
        if unrelated_model.predict([[f1, rouge_1, rouge_2, rouge_l]])[0] == 1:
            related = True
            f1 = calculate_f1_score(similar_question["Answer"], response)
            rouge_1, rouge_2, rouge_l = calculate_rouge_scores(similar_question["Answer"], response)
            f1_2 = calculate_f1_score(question, response)
            rouge_1_2, rouge_2_2, rouge_l_2 = calculate_rouge_scores(question, response)
            if correct_model.predict([[f1, rouge_1, rouge_2, rouge_l, f1_2, rouge_1_2, rouge_2_2, rouge_l_2]])[0] == 1:
                return False, response
    if related:
        return True, random.choice(wrong_responses)
    else:
        return False, random.choice(unrelated_responses)
    
    
    


    
            
