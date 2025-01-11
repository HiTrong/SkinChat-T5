import re
import os
import random
import json
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

with open("./sick_config/skin_sick.json", "r", encoding="utf-8") as f:
    skin_dict = json.load(f)

def illustrate(response):
    for sick_name in skin_dict:
        sick_config = skin_dict[sick_name]
        for keyword in sick_config["Keywords"]:
            if keyword in response.lower():
                file_names = [file for file in os.listdir(sick_config["illustrate_folder"]) if os.path.isfile(os.path.join(sick_config["illustrate_folder"], file))]
                if len(file_names)==0:
                    return None
                return os.path.join(sick_config["illustrate_folder"], random.choice(file_names))
    return None

def extract_emails(text):
    # Định dạng của email
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # Tìm tất cả các chuỗi phù hợp với định dạng email
    emails = re.findall(email_pattern, text)
    return emails

def send_question_email(user_email, question):
    # Cấu hình thông tin email
    sender_email = "skinchat01@gmail.com"  # Địa chỉ email gửi
    # receiver_email = "21133004@student.hcmute.edu.vn"  # Địa chỉ email nhận
    receiver_email = "trongvo250403@gmail.com"  # Địa chỉ email nhận
    password = "rrhw jlhy vbjf fsnl"  # Mật khẩu email của bạn
    
    # Tạo email
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "SkinChat - Support: Cần hỗ trợ trả lời câu hỏi từ người dùng"
    
    body = f"""
    <html>
    <head>
        <style>
            .content {{
                font-family: Arial, sans-serif;
                color: #333;
            }}
            .header {{
                background-color: #005acf;
                color: white;
                padding: 10px;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
            }}
            .message {{
                margin: 20px;
            }}
            .footer {{
                margin: 20px;
                font-size: 12px;
                color: #666;
            }}
            .button {{
                display: inline-block;
                margin: 20px 0;
                padding: 10px 20px;
                font-size: 16px;
                color: white;
                background-color: #f33900;
                text-decoration: none;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="content">
            <div class="header">
                SkinChat - Support
            </div>
            <div class="message">
                <p>Xin chào bác sĩ Hoài Trọng,</p>
                <p>Hệ thống SkinChat vừa nhận được một câu hỏi từ người dùng mà chúng tôi không thể trả lời. Chúng tôi cần sự trợ giúp của chuyên gia để phản hồi và hỗ trợ người dùng này.</p>
                <p><strong>Thời gian:</strong></p>
                <p>{datetime.now().strftime("%d/%m/%Y")}</p>
                <p><strong>Câu hỏi từ người dùng:</strong></p>
                <p>{question}</p>
                <p>Vui lòng kiểm tra và gửi phản hồi đến người dùng sớm nhất có thể.</p>
                <a href="mailto:{user_email}" class="button">Trả lời người dùng</a>
            </div>
            <div class="footer">
                <p>Trân trọng,<br>Đội ngũ hỗ trợ SkinChat Support</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Đính kèm nội dung email HTML vào email
    message.attach(MIMEText(body, "html"))
    
    # Gửi email qua Gmail
    check = True
    try:
        # Kết nối đến server SMTP của Gmail
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Bắt đầu mã hóa TLS
        server.login(sender_email, password)  # Đăng nhập vào email
        text = message.as_string()  # Chuyển đổi email thành chuỗi
        server.sendmail(sender_email, receiver_email, text)  # Gửi email
        print("Email đã được gửi thành công!")
    except Exception as e:
        print(f"Gặp lỗi khi gửi email: {e}")
        check = False
    finally:
        server.quit()  # Đóng kết nối
        return check
