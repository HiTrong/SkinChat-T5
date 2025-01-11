import streamlit as st

# Tạo tiêu đề cho ứng dụng
st.title("Chat App với Expander cho từng tin nhắn")

# Khởi tạo session state để lưu trữ tin nhắn trong phiên làm việc
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tạo phần nhập tin nhắn
user_input = st.text_input("Nhập tin nhắn của bạn:", key="user_input")

# Xử lý khi người dùng gửi tin nhắn
if st.button("Gửi"):
    if user_input:
        # Lưu tin nhắn vào session state
        st.session_state.messages.append(user_input)
        # Xóa nội dung ô nhập tin nhắn sau khi gửi

# Hiển thị từng tin nhắn trong expander
st.subheader("Lịch sử tin nhắn:")
for i, message in enumerate(st.session_state.messages):
    with st.expander(f"Tin nhắn {i+1}", expanded=False):
        st.write(message)
