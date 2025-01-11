# ====== Import Libraries ======
import streamlit as st
import matplotlib.pyplot as plt
import time
import os
import base64
from PIL import Image
import pandas as pd
from streamlit_option_menu import option_menu
from MyT5 import get_response
from Semantic_vectordb import Evaluate_response
from Image_vectordb import diagnosis
from utils import extract_emails, send_question_email, illustrate

# ====== Page Settings ======
st.set_page_config(
        page_title="Chatbot - LLM",  
        page_icon=":globe_with_meridians:",  
        layout="centered", 
        initial_sidebar_state="auto" 
    )

# ====== Diagnosis Template ======
diagnosis_template = """**Chẩn đoán:** {}

{}

**Chẩn đoán khác:** {}"""


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
    st.image("./img/logo/logongang.png")
    selected = option_menu("Menu",["Giới thiệu","Chatbot","Điều khoản sử dụng"],
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
    st.markdown("""# Giới thiệu SkinChat: 

***SkinChat***  là một ứng dụng chatbot được phát triển bởi Võ Hoài Trọng và Nguyễn Thị Phương Anh, sinh viên chuyên ngành Kỹ thuật Dữ liệu.  

Ứng dụng này là sản phẩm của tiểu luận chuyên ngành, với mục tiêu cung cấp công cụ tư vấn và hỏi đáp về các vấn đề da liễu cho người dùng.  

Chatbot ***SkinChat*** sử dụng công nghệ xử lý ngôn ngữ tự nhiên và học máy tiên tiến để giải đáp các thắc mắc liên quan đến chăm sóc và điều trị da.  
Với giao diện thân thiện, ứng dụng mang lại sự tiện lợi, hỗ trợ người dùng tiếp cận thông tin chăm sóc sức khỏe da mọi lúc, mọi nơi.  

Tuy nhiên, do thông tin từ chatbot có thể chưa hoàn toàn chính xác, người dùng cần kiểm tra kỹ lưỡng trước khi áp dụng các biện pháp điều trị.  

Mục tiêu của ***SkinChat*** là trở thành một công cụ hữu ích cho cộng đồng, giúp nâng cao nhận thức và hỗ trợ quá trình chăm sóc sức khỏe da.  

Ứng dụng sẽ không ngừng được cải thiện nhằm đáp ứng tốt hơn nhu cầu của người dùng.

# Hướng dẫn sử dụng ứng dụng Chatbot - SkinChat

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
    
    
# ====== Chatbot Page ======
if selected == "Chatbot":
    skinchat_css()
    st.title("SkinChat")
    
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = None
    
    if "receive_mail_state" not in st.session_state:
        st.session_state.receive_mail_state = False
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({'type':'text',
                                          'role':'assistant', 
                                          'content':"**Xin chào, tôi là bác sĩ AI - SkinChat. Hãy hỏi những câu hỏi về da liễu để được hỗ trợ nhé!**", 
                                          'illustration':None
                                          })
        
    for message in st.session_state.messages:
        if message['role'] == "assistant":
            with st.chat_message(message['role'], avatar="👩‍⚕️"):
                if message['type'] == "text":
                    st.markdown(message['content'])
                    if message['illustration'] is not None:
                        with st.expander("**Ảnh minh họa** (*Lưu ý:* Có thể gây khó chịu!)"):
                            st.image(message['illustration'])
                else:
                    st.markdown("""Dựa trên hình ảnh của bạn, chúng tôi sẽ dựa các hình ảnh triệu chứng tương tự để đưa ra chuẩn đoán cho bạn. **Tuy nhiên**, xin lưu ý rằng đây chỉ là những kết quả dựa trên phân tích hình ảnh và không thay thế cho sự tư vấn hoặc chẩn đoán y tế chính thức. Để có kết quả chính xác và đáng tin cậy, chúng tôi khuyến khích bạn nên tham khảo ý kiến từ các chuyên gia y tế hoặc bác sĩ.""")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(message['illustration'])
                    with col2:
                        st.markdown(message['content'])
                    
        else:
            with st.chat_message(message['role']):
                if message['type'] == "text":
                    st.markdown(message['content'])
                else:
                    with st.expander("**Ảnh upload của bạn**"):
                        st.image(message['illustration'])
                        
                
    if prompt := st.chat_input("Nhập câu hỏi của bạn tại đây!"):
        # input
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({'type':'text', 'role':'user', 'content':prompt})
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
                illustration = illustrate(response)
                
                text = ""
                for c in response:
                    text += c 
                    message_placeholder.markdown(text)
                    time.sleep(0.003)
                if illustration is not None:
                    with st.expander("**Ảnh minh họa** (*Lưu ý:* Có thể gây khó chịu!)"):
                        st.image(illustration)
            st.session_state.messages.append({'type':'text', 'role':'assistant', 'content':response, 'illustration':illustration})
        else:
            if len(mail_list) > 0:
                email = mail_list[0]
                check = send_question_email(email, st.session_state.messages[len(st.session_state.messages)-3]['content'])
                if check:
                    with st.chat_message("assistant", avatar="👩‍⚕️"):
                        st.markdown("Câu hỏi đã được gửi tới chuyên gia! Bạn vui lòng check mail sau 3-5 ngày để nhận được câu trả lời từ chuyên gia nhé!")
                    st.session_state.messages.append({'type':'text','role':'assistant', 'content':"Câu hỏi đã được gửi tới chuyên gia! Bạn vui lòng check mail sau 3-5 ngày để nhận được câu trả lời từ chuyên gia nhé!", 'illustration':None})
                else:
                    with st.chat_message("assistant", avatar="👩‍⚕️"):
                        st.markdown("Đã có lỗi với chức năng gửi mail đến chuyên gia! Vui lòng thử lại sau!")
                    st.session_state.messages.append({'type':'text','role':'assistant', 'content':"Đã có lỗi với chức năng gửi mail đến chuyên gia! Vui lòng thử lại sau!", 'illustration': None})
            st.session_state.receive_mail_state = False
            
    if uploaded_file := st.file_uploader(
            "**[NEW]** Chuẩn đoán thông qua ảnh", type=["png", "jpg", "jpeg"]
        ):
        if st.session_state.file_uploaded is None:
            st.session_state.file_uploaded = ""
            image = Image.open(uploaded_file)
            image_path, diagnosis_name, another_diagnosis = diagnosis(image)
            
            # User 
            with st.chat_message("user"):
                with st.expander("**Ảnh upload của bạn**"):
                    st.image(uploaded_file)
            st.session_state.messages.append({'type':'image', 'role':'user', 'content':"", 'illustration':uploaded_file})

            
            # Assistant
            text = diagnosis_template.format(diagnosis_name, get_response(diagnosis_name + "là gì?"),another_diagnosis)
            with st.chat_message("assistant", avatar="👩‍⚕️"):
                st.markdown("""Dựa trên hình ảnh của bạn, chúng tôi sẽ dựa các hình ảnh triệu chứng tương tự để đưa ra chẩn đoán cho bạn. **Tuy nhiên**, xin lưu ý rằng đây chỉ là những kết quả dựa trên phân tích hình ảnh và không thay thế cho sự tư vấn hoặc chẩn đoán y tế chính thức. Để có kết quả chính xác và đáng tin cậy, chúng tôi khuyến khích bạn nên tham khảo ý kiến từ các chuyên gia y tế hoặc bác sĩ.""")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_path)
                with col2:
                    st.markdown(text)
            st.session_state.messages.append({'type':'image', 'role':'assistant', 'content':text, 'illustration':image_path})
    else:
        st.session_state.file_uploaded = None
            
            

# Thêm logo ở góc dưới bên trái từ thư mục
footer_logo_path = "./img/logo/logodoc.png"  # Đường dẫn đến logo ở góc dưới
if os.path.exists(footer_logo_path):  # Kiểm tra xem tệp có tồn tại không
    footer_logo_base64 = None
    with open(footer_logo_path, "rb") as image_file:
        footer_logo_base64 = base64.b64encode(image_file.read()).decode()
    st.markdown(f'<img class="logo" src="data:image/png;base64,{footer_logo_base64}" alt="Logo" />', unsafe_allow_html=True)