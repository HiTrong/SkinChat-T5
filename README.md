# SkinChat - T5

[![Python](https://img.shields.io/badge/3.10-green?style=flat-square&logo=Python&label=Python&labelColor=green&color=grey)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/11.8-black?style=flat-square&logo=PyTorch&logoColor=red&label=Torch&labelColor=orange&color=grey)](https://pytorch.org/)
[![SimpleT5](https://img.shields.io/badge/T5%20Base-black?style=flat-square&logo=Google&logoColor=red&label=SimpleT5&labelColor=black&color=grey)](https://pypi.org/project/simplet5/)
[![HuggingFace](https://img.shields.io/badge/transformers-black?style=flat-square&logo=HuggingFace&logoColor=red&label=HuggingFace&labelColor=yellow&color=grey)](https://pypi.org/project/transformers/)
[![Gemini API](https://img.shields.io/badge/Free_API-black?style=flat-square&logo=Google&logoColor=white&label=Gemini&labelColor=blue&color=grey)](https://ai.google.dev/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Application-red?logo=Streamlit&logoColor=Red&labelColor=white)](https://docs.streamlit.io/)
[![Apache 2.0 License](https://img.shields.io/badge/Apache_2.0-blue?style=flat-square&logo=License&logoColor=red&label=License&labelColor=blue&color=grey)](https://www.apache.org/licenses/LICENSE-2.0)

Đây là project phục vụ cho báo cáo tiểu luận chuyên ngành với đề tài tìm hiểu mô hình ngôn ngữ lớn và ứng dụng xây dựng Chatbot.

Với mục tiêu xây dựng một **chatbot** tư vấn trả lời câu hỏi về **da liễu** dựa trên mô hình ngôn ngữ lớn, chúng ta sẽ tiến hành một số bước như sau:

| Giai đoạn | Thao tác |
|-----------|----------|
| Giai đoạn 1 | Tìm kiếm và thu thập các nguồn dữ liệu uy tín |
| Giai đoạn 2 | Tăng cường dữ liệu và sinh dữ liệu tổng hợp thông qua API Gemini |
| Giai đoạn 3 | Xử lý dữ liệu và xây dựng bộ Tokenizer |
| Giai đoạn 4 | Huấn luyện mô hình |
| Giai đoạn 5 | Đánh giá và so sánh mô hình |
| Giai đoạn 6 | Tích hợp mô hình vào ứng dụng chatbot với streamlit |

## 📄 Giai đoạn 1, 2, 3

Các nguồn dữ liệu được chúng tôi thu thập từ các trang web uy tín bao gồm: 

- [Bệnh viện 115](https://benhvien115.com.vn/)
- [Bệnh viện Nhi Trung Ương](http://benhviennhi.org.vn/)
- [eDoctor](https://edoctor.io/)
- [Chuyên mục Da liễu - VnExpress](https://vnexpress.net/suc-khoe/cac-benh/da-lieu)
- [Ivie](https://ivie.vn/)
- [VietSkin](https://www.vietskin.vn/)
- [Hướng dẫn chẩn đoán điều trị Da liễu - KCB](https://kcb.vn/upload/2005611/20210723/Huong-dan-chan-doan-dieu-tri-Da-lieu.pdf)

Tất cả các dữ liệu đó sẽ được sinh dữ liệu tổng hợp và tăng cường dữ liệu thông qua một mô hình ngôn ngữ đóng vai trò làm **"thầy giáo"**. Ở đây, chúng tôi lựa chọn nhóm mô hình **Gemini** của **Google AI** vì **Google AI Studio** cho phép người dùng thông qua **API** để sử dụng trong một giới hạn nhất định. 

Dữ liệu được xử lý và đưa về dạng:
```
source_text:
Trả lời câu hỏi này: Tôi có thể làm gì để giảm đau do bệnh zona không, thưa bác sĩ?

target_text:
Bạn có thể sử dụng thuốc giảm đau theo chỉ định của bác sĩ, cũng như các liệu pháp tại chỗ như kem b...
```

**Tokenizer** tiếng việt được huấn luyện lại từ **Tokenizer T5** nguyên mẫu của **T5-base** dựa trên dữ liệu đã xử lý trên với số từ vựng là `20000`.

## 🦾 Giai đoạn 4
