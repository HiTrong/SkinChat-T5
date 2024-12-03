# ====== Import Libraries ======
import streamlit as st
import matplotlib.pyplot as plt
import time
import os
import base64
import pandas as pd
from streamlit_option_menu import option_menu
from MyT5 import get_response
from Semantic_vectordb import Evaluate_response
from utils import extract_emails, send_question_email

# ====== Page Settings ======
st.set_page_config(
        page_title="Chatbot - LLM",  
        page_icon=":globe_with_meridians:",  
        layout="centered", 
        initial_sidebar_state="auto" 
    )


# ====== CSS Settings ======
def skinchat_css():    
    with open("./css/skinchat_styles.css", encoding="utf-8") as f:
        css = f.read()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
def normal_css():    
    with open("./css/homepage_styles.css", encoding="utf-8") as f:
        css = f.read()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    
normal_css()


# ====== Sidebar ======
with st.sidebar:    
    st.image("./img/logongang.png")
    selected = option_menu("Menu",["Giới thiệu","Mô hình ngôn ngữ lớn","Chatbot","Điều khoản sử dụng"],
                           icons=['house','star','chat','flag'], menu_icon="cast", default_index=0,
                           styles={
                                "container": {"font-family": "Monospace"},
                                "icon": {"color":"#71738d"}, 
                                "nav-link": {"--hover-color": "#d2cdfa","font-family": "Monospace"},
                                "nav-link-selected": {"font-family": "Monospace","background-color": "#6694ed"},
                            }
                           )
    st.markdown(
        """
        <div class="sidebar-signature">
            Copyright © 2024 TRONGVO & PHUONGANH
        </div>
        """,
        unsafe_allow_html=True
    )
    
    
    
# ====== Giới thiệu Page ======
if selected == "Giới thiệu":
    st.markdown(""" # Đề tài: Tìm Hiểu Về Mô Hình Ngôn Ngữ Lớn và Ứng Dụng Trong Xây Dựng Chatbot  
*Thực hiện bởi:* Võ Hoài Trọng và Nguyễn Thị Phương Anh  
*Chuyên ngành:* Kỹ thuật Dữ liệu  

## Mục tiêu của đề tài  
Đề tài này nhằm nghiên cứu và tìm hiểu về các mô hình ngôn ngữ lớn (LLM) và ứng dụng của chúng trong việc xây dựng chatbot cung cấp thông tin và tư vấn về các vấn đề da liễu.

## Quy trình nghiên cứu

1. *Lựa chọn mô hình ngôn ngữ:*  
   Nghiên cứu và xác định mô hình ngôn ngữ lớn phù hợp với yêu cầu của ứng dụng chatbot da liễu.

2. *Thu thập dữ liệu:*  
   Thu thập và chuẩn bị tập dữ liệu chất lượng để huấn luyện mô hình, đảm bảo tính đa dạng và độ chính xác của các thông tin liên quan đến da liễu.

3. *Huấn luyện mô hình:*  
   Tiến hành huấn luyện mô hình ngôn ngữ với tập dữ liệu đã chuẩn bị.

4. *Đánh giá hiệu quả:*  
   Đo lường và đánh giá hiệu suất của mô hình dựa trên các tiêu chí như độ chính xác và khả năng phản hồi thông tin.

5. *So sánh với các mô hình khác:*  
   So sánh kết quả của mô hình đã huấn luyện với các mô hình khác để xác định ưu điểm và nhược điểm của mô hình đã chọn.

## Kết quả  
Sản phẩm cuối cùng của đề tài là một chatbot thông minh, tích hợp mô hình ngôn ngữ đã huấn luyện, có khả năng tư vấn và cung cấp thông tin chính xác về các vấn đề da liễu cho người dùng.

---

## Giới thiệu SkinChat: 

***SkinChat***  là một ứng dụng chatbot được phát triển bởi Võ Hoài Trọng và Nguyễn Thị Phương Anh, sinh viên chuyên ngành Kỹ thuật Dữ liệu.  

Ứng dụng này là sản phẩm của tiểu luận chuyên ngành, với mục tiêu cung cấp công cụ tư vấn và hỏi đáp về các vấn đề da liễu cho người dùng.  

Chatbot ***SkinChat*** sử dụng công nghệ xử lý ngôn ngữ tự nhiên và học máy tiên tiến để giải đáp các thắc mắc liên quan đến chăm sóc và điều trị da.  
Với giao diện thân thiện, ứng dụng mang lại sự tiện lợi, hỗ trợ người dùng tiếp cận thông tin chăm sóc sức khỏe da mọi lúc, mọi nơi.  

Tuy nhiên, do thông tin từ chatbot có thể chưa hoàn toàn chính xác, người dùng cần kiểm tra kỹ lưỡng trước khi áp dụng các biện pháp điều trị.  

Mục tiêu của ***SkinChat*** là trở thành một công cụ hữu ích cho cộng đồng, giúp nâng cao nhận thức và hỗ trợ quá trình chăm sóc sức khỏe da.  

Ứng dụng sẽ không ngừng được cải thiện nhằm đáp ứng tốt hơn nhu cầu của người dùng.
## Hướng dẫn sử dụng ứng dụng Chatbot - SkinChat

Để sử dụng ứng dụng, bạn vui lòng thực hiện theo các bước sau:

1. Nhấn vào mục **Chatbot**.
2. Gõ câu hỏi vào khung chat.

**Lưu ý:**
- Các câu hỏi cần liên quan đến chủ đề da liễu.
- Câu hỏi của bạn phải tuân thủ các điều khoản và chính sách của chúng tôi.

Sau khi hoàn thành hai bước trên, bạn sẽ nhận được câu trả lời từ chatbot hoặc chuyên gia tư vấn.""")
    
# ====== Điều khoản sử dụng Page ======
if selected == "Điều khoản sử dụng":
    st.markdown("""## Điều khoản sử dụng SkinChat

1. *Giới thiệu về dịch vụ:*  
   ***SkinChat*** cung cấp dịch vụ tư vấn liên quan đến các vấn đề da liễu, nhằm hỗ trợ người dùng hiểu rõ hơn về các tình trạng da của mình và cung cấp thông tin tham khảo. 

2. *Trách nhiệm của người dùng:*  
   Người dùng chịu hoàn toàn trách nhiệm đối với các hành động y tế dựa trên thông tin từ SkinChat.  
   Ứng dụng chỉ cung cấp thông tin tham khảo về da liễu, và người dùng cần tham khảo ý kiến của các chuyên gia da liễu trước khi thực hiện bất kỳ phương pháp điều trị nào. 

3. *Sử dụng đúng mục đích:*  
   Người dùng phải sử dụng ***SkinChat*** đúng mục đích tư vấn về da liễu.  
   Mọi hành vi lạm dụng ứng dụng để phát tán thông tin sai lệch hoặc gây hại đều bị cấm. 

4. *Tuyên bố miễn trừ trách nhiệm:*  
   ***SkinChat*** không phải là dịch vụ chăm sóc sức khỏe chuyên nghiệp và không thay thế cho các ý kiến y tế từ bác sĩ.  
   Người dùng cần phải tham khảo và tuân thủ hướng dẫn từ các chuyên gia y tế trước khi đưa ra quyết định điều trị. 

5. *Quyền sửa đổi và ngừng dịch vụ:*  
   Nhà phát triển có quyền thay đổi, chỉnh sửa nội dung hoặc ngừng cung cấp dịch vụ bất kỳ lúc nào mà không cần thông báo trước.  
   Người dùng có trách nhiệm theo dõi các thay đổi trong điều khoản sử dụng. 

6. *Quyền sở hữu trí tuệ:*  
   Toàn bộ thông tin, nội dung, và công nghệ được sử dụng trong ***SkinChat*** thuộc quyền sở hữu trí tuệ của nhà phát triển.  
   Việc sao chép, phân phối hoặc sử dụng bất hợp pháp sẽ bị xử lý theo quy định pháp luật.""")
    st.markdown("""## Chính sách bảo mật SkinChat

1. ***Thu thập dữ liệu cá nhân:***
   Chúng tôi thu thập thông tin cá nhân như email và lịch sử cuộc trò chuyện để phục vụ cho việc vận hành ứng dụng và cải thiện trải nghiệm người dùng.  
   Thông tin này không được thu thập vì mục đích kinh doanh hay quảng cáo.

2. ***Lưu trữ và bảo vệ dữ liệu:***
   Thông tin cá nhân của người dùng được lưu trữ trên các hệ thống bảo mật cao, đảm bảo rằng chỉ những nhân viên có quyền truy cập mới có thể xem xét dữ liệu này.  
   Chúng tôi cam kết không sử dụng thông tin cá nhân ngoài mục đích vận hành ứng dụng.

3. ***Quyền của người dùng:***  
   Người dùng có toàn quyền kiểm soát dữ liệu cá nhân của mình.  
   Bạn có thể yêu cầu xóa, chỉnh sửa hoặc xuất dữ liệu bất cứ lúc nào bằng cách liên hệ với đội ngũ hỗ trợ.

4. ***Chia sẻ thông tin với bên thứ ba:***  
   Trong trường hợp cần thiết để cải thiện trải nghiệm dịch vụ hoặc hỗ trợ tư vấn chuyên sâu, chúng tôi có thể chia sẻ thông tin với các chuyên gia da liễu.  
   Việc chia sẻ này sẽ chỉ được thực hiện khi có sự đồng ý của người dùng và đảm bảo tuân thủ quy định bảo mật.

5. ***Bảo mật thông tin nhạy cảm:***  
   Chúng tôi áp dụng các biện pháp bảo mật đặc biệt để bảo vệ những thông tin này, bao gồm mã hóa dữ liệu và kiểm soát truy cập nghiêm ngặt, đảm bảo rằng thông tin này không bị truy cập trái phép hoặc rò rỉ ra ngoài.

6. ***Cơ chế khiếu nại và liên hệ:***  
   Nếu có bất kỳ thắc mắc hoặc khiếu nại liên quan đến bảo mật dữ liệu, người dùng có thể liên hệ với chúng tôi qua email hỗ trợ: ***skinchat.support@gmail.com***.  
   Chúng tôi sẽ phản hồi và giải quyết các vấn đề trong thời gian sớm nhất.""")
    
# ====== Large Language Model Page ======
if selected == "Mô hình ngôn ngữ lớn":
    if "get_model" not in st.session_state:
        st.session_state.get_model = False
    if "get_dataset" not in st.session_state:
        st.session_state.get_dataset = False
    if "get_training" not in st.session_state:
        st.session_state.get_training = False
    if "get_evaluation" not in st.session_state:
        st.session_state.get_evaluation = False
    if "get_comparation" not in st.session_state:
        st.session_state.get_comparation = False
        
    tab_labels = ["Chọn mô hình", "Chọn tập dữ liệu", "Huấn luyện", "Đánh giá mô hình", "So sánh mô hình"]
    tabs = st.tabs(tab_labels)
    
    with tabs[0]:
        col1, col2 = st.columns([3, 1])  # Tỉ lệ cột (col1 chiếm 3 phần, col2 chiếm 1 phần)

        with col1:
            st.write("- **Mô hình T5** là một mô hình Transformer mạnh mẽ, được thiết kế để xử lý các tác vụ NLP (Natural Language Processing) như dịch ngôn ngữ, tóm tắt văn bản, và tạo câu trả lời cho câu hỏi.")

        with col2:
            choose_model = st.button("Chọn mô hình này", key="T5")
                
        if choose_model or st.session_state.get_model == True:
            st.success("Bạn đã chọn mô hình T5!")
            st.markdown("""| Thành phần                | Siêu tham số                                      | Giá trị         |
|---------------------------|--------------------------------------------------|-----------------|
| **Embedding**             | Vocabulary size                                  | **<Custom>**    |
|                           | Embedding dimension                              | 768             |
| **Encoder**               | Number of encoder layers                         | 12              |
|                           | Attention heads                                  | 12              |
|                           | Attention dimension                              | 768             |
|                           | Feed-forward hidden size                         | 3072            |
|                           | Dropout rate                                     | 0.1             |
|                           | Relative attention bias embedding size           | 32              |
|                           | Final layer normalization                        |   LayerNorm     |
| **Decoder**               | Number of decoder layers                         | 12              |
|                           | Attention heads                                  | 12              |
|                           | Attention dimension                              | 768             |
|                           | Feed-forward hidden size                         | 3072            |
|                           | Dropout rate                                     | 0.1             |
|                           | Relative attention bias embedding size           | 32              |
|                           | Final layer normalization                        |   LayerNorm     |
| **Attention**             | Query, Key, Value, Output projection dimensions  | 768             |
|                           | Cross-attention                                  | Yes             |
| **Feed-Forward Network**  | DenseReluDense (input to hidden) dimension       | 768 -> 3072     |
|                           | DenseReluDense (hidden to output) dimension      | 3072 -> 768     |
| **LM Head**               | Output vocabulary size                           | **<Custom>**    |
|                           | Output dimension                                 | 768             |
""")
            st.session_state.get_model = True
        
    with tabs[1]:
        if st.session_state.get_model == False:
            st.warning("Bạn chưa hoàn thành thao tác cần thiết!")
        else:
            option = st.radio("Chọn cách tải dataset", ("Tải dataset lên", "Chọn dataset có sẵn"))

            if option == "Tải dataset lên":
                uploaded_file = st.file_uploader("Chọn dataset để tải lên", type=["csv"])
                if uploaded_file is not None:
                    st.write("Bạn đã tải lên dataset:", uploaded_file.name)
                    st.warning("Chức năng này hiện tại không hoạt động! Vui lòng chọn tập dữ liệu có sẵn!")

            elif option == "Chọn dataset có sẵn":
                available_datasets = ["Skinchat_dataset.csv"]
                dataset_choice = st.selectbox("Chọn dataset có sẵn", available_datasets)
                st.success(f"Bạn đã chọn dataset: {dataset_choice}")
                st.markdown("""
                            Đây là tập dữ liệu hỏi đáp về 65 loại bệnh da liễu được chia làm làm 3 tập dữ liệu:
                            - Dữ liệu huấn luyện: 220697 câu hỏi-đáp
                            - Dữ liệu xác thực: 130 câu hỏi-đáp (Mỗi loại bệnh gồm 2 câu hỏi)
                            - Dữ liệu kiểm thử: 130 câu hỏi-đáp (Mỗi loại bệnh gồm 2 câu hỏi)
                            """)
                st.session_state.get_dataset = True
            
    with tabs[2]:
        if st.session_state.get_model == False or st.session_state.get_dataset == False:
            st.warning("Bạn chưa hoàn thành thao tác cần thiết!")
        else: 
            st.markdown("Các siêu tham số tối ưu cho huấn luyện:")
            params = {
                "Parameter": [
                    "Số epoch",
                    "Source max token length",
                    "Target max token length",
                    "Batch size",
                    "Max epochs",
                    "Use GPU",
                    "Precision",
                    "Optimizer",
                    "Learning rate",
                    "Learning rate scheduler",
                    "Weight decay",
                    "Gradient clipping",
                ],
                "Value": [
                    20,
                    100,
                    200,
                    24,
                    5,
                    True,
                    16,
                    "AdamW",
                    "5 × 10⁻⁵",
                    "Linear scheduler với warm-up",
                    0.01,
                    1.0,
                ],
            }

            params_df = pd.DataFrame(params)
            st.table(params_df)
            
            start_train = st.button("Bắt đầu huấn luyện", key="start_train")
            progress_bar = st.progress(0)
                
            text_empty = st.empty()
            pyplot_empty = st.empty()
            if start_train:
                epochs = range(1, 21)
                train_loss = [
                    2.8112, 1.93, 1.7054, 1.564, 1.4577, 1.3586, 1.2814, 1.2171, 1.158, 1.1031, 
                    1.0473, 1.0004, 0.9588, 0.9214, 0.8861, 0.8513, 0.8113, 0.7769, 0.7501, 0.7205
                ]
                val_loss = [
                    1.4982, 1.1801, 1.0517, 0.9555, 0.8898, 0.8571, 0.8279, 0.7905, 0.7691, 0.7557,
                    0.7525, 0.757, 0.7523, 0.7479, 0.7404, 0.743, 0.75, 0.7581, 0.7519, 0.7586
                ]
                
                
                for i, epoch in enumerate(epochs):
                    progress = epoch / len(epochs)
                    progress_bar.progress(progress) 
                    time.sleep(0.05)  
                    with text_empty:
                        st.write(f"Epoch {epoch}/{len(epochs)} - Training Loss: {train_loss[i]:.4f} - Validation Loss: {val_loss[i]:.4f}")
                    with pyplot_empty:
                        # Vẽ đồ thị
                        plt.figure(figsize=(10, 6))
                        plt.plot(epochs[:i], train_loss[:i], label='Train Loss', marker='o')
                        plt.plot(epochs[:i], val_loss[:i], label='Validation Loss', marker='o')
                        plt.xlabel('Epochs')
                        plt.ylabel('Loss')
                        plt.title('Train and Validation Loss over Epochs')
                        plt.legend()
                        plt.grid(True)
                        st.pyplot(plt)
                st.success("Huấn luyện hoàn tất!")
                st.session_state.get_training = True
                
            
    with tabs[3]:
        if st.session_state.get_model == False or st.session_state.get_dataset == False or st.session_state.get_training == False:
            st.warning("Bạn chưa hoàn thành thao tác cần thiết!")
        else:                    
            st.write("### Kết quả đánh giá mô hình:")
            st.write("""
                | Metric               | Score   |
                |----------------------|---------|
                | Average F1 Score     | 0.5830  |
                | Average BLEU Score   | 0.3943  |
                | Average ROUGE-1 Score| 0.7540  |
                | Average ROUGE-2 Score| 0.5528  |
                | Average ROUGE-L Score| 0.6273  |
            """)
            st.session_state.get_evaluation = True
            
    with tabs[4]:
        if st.session_state.get_model == False or st.session_state.get_dataset == False or st.session_state.get_training == False or st.session_state.get_evaluation == False:
            st.warning("Bạn chưa hoàn thành thao tác cần thiết!")
        else:
            st.markdown("""
            ### So sánh các mô hình

            | **Model**            | **Tổng số tham số** | **Số câu trả lời đúng**    | **Tốc độ suy luận với tài nguyên nhỏ** |
            |----------------------|---------------------|----------------------------|----------------------------------------|
            | **SkinChat (T5)**     | **213.5M**          | 83.8 (109/130)             | **Rất nhanh**                         |
            | PhoGPT                | 4B                  | 73.8 (96/130)              | Trung bình                            |
            | ChatGPT-3.5           | 175B                | **94.6 (123/130)**         | Không thể vận hành                    |
            | Gemma-2-2B-it         | 2B                  | 46.2 (60/130)              | Nhanh                                 |
            """)
    
    
# ====== Chatbot Page ======
if selected == "Chatbot":
    skinchat_css()
    st.title("SkinChat")
    
    if "receive_mail_state" not in st.session_state:
        st.session_state.receive_mail_state = False
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({'role':'assistant', 'content':"**Xin chào, tôi là bác sĩ AI - SkinChat. Hãy hỏi những câu hỏi về da liễu để được hỗ trợ nhé!**"})
        
    for message in st.session_state.messages:
        if message['role'] == "assistant":
            with st.chat_message(message['role'], avatar="👩‍⚕️"):
                st.markdown(message['content'])
        else:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                
    if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây!"):
        # input
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content':prompt})
        mail_list = extract_emails(prompt)
        if st.session_state.receive_mail_state == False or len(mail_list)==0:
            # Get response
            with st.chat_message("assistant", avatar="👩‍⚕️"):
                message_placeholder = st.empty()
                text = ""
                for c in "Bác sĩ AI đang đọc và trả lời câu hỏi của bạn...":
                    text += c 
                    message_placeholder.markdown(text)
                    time.sleep(0.01)
                response = get_response(prompt)
                
                b, response = Evaluate_response(prompt, response)
                st.session_state.receive_mail_state = b
                
                text = ""
                for c in response:
                    text += c 
                    message_placeholder.markdown(text)
                    time.sleep(0.003)
            st.session_state.messages.append({'role':'assistant', 'content':response})
        else:
            if len(mail_list) > 0:
                email = mail_list[0]
                check = send_question_email(email, st.session_state.messages[len(st.session_state.messages)-3]['content'])
                if check:
                    with st.chat_message("assistant", avatar="👩‍⚕️"):
                        st.markdown("Câu hỏi đã được gửi tới chuyên gia! Bạn vui lòng check mail sau 3-5 ngày để nhận được câu trả lời từ chuyên gia nhé!")
                    st.session_state.messages.append({'role':'assistant', 'content':"Câu hỏi đã được gửi tới chuyên gia! Bạn vui lòng check mail sau 3-5 ngày để nhận được câu trả lời từ chuyên gia nhé!"})
                else:
                    with st.chat_message("assistant", avatar="👩‍⚕️"):
                        st.markdown("Đã có lỗi với chức năng gửi mail đến chuyên gia! Vui lòng thử lại sau!")
                    st.session_state.messages.append({'role':'assistant', 'content':"Đã có lỗi với chức năng gửi mail đến chuyên gia! Vui lòng thử lại sau!"})
            st.session_state.receive_mail_state = False
            

# Thêm logo ở góc dưới bên trái từ thư mục
footer_logo_path = "./img/logodoc.png"  # Đường dẫn đến logo ở góc dưới
if os.path.exists(footer_logo_path):  # Kiểm tra xem tệp có tồn tại không
    footer_logo_base64 = None
    with open(footer_logo_path, "rb") as image_file:
        footer_logo_base64 = base64.b64encode(image_file.read()).decode()
    st.markdown(f'<img class="logo" src="data:image/png;base64,{footer_logo_base64}" alt="Logo" />', unsafe_allow_html=True)