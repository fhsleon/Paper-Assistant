import streamlit as st
import requests
import html

API_URL = "http://localhost:8000"


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "files" not in st.session_state:
        st.session_state.files = []
    
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False
    
    if "files_loaded" not in st.session_state:
        st.session_state.files_loaded = False
    
    if "uploading" not in st.session_state:
        st.session_state.uploading = False
    
    if "pending_message" not in st.session_state:
        st.session_state.pending_message = None


def get_file_list():
    if st.session_state.files_loaded:
        return
    
    try:
        response = requests.get(API_URL + "/api/files", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.session_state.files = data.get("files", [])
            st.session_state.files_loaded = True
    except:
        pass


def refresh_files():
    st.session_state.files_loaded = False
    get_file_list()


def escape_html(text):
    return html.escape(str(text))


def upload_pdf_file(file):
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(API_URL + "/api/upload", files=files, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            return True, data.get("filename", "")
        else:
            try:
                error_msg = response.json().get("detail", "上传失败")
            except:
                error_msg = "上传失败，状态码: " + str(response.status_code)
            return False, error_msg
    except Exception as e:
        return False, "连接错误: " + str(e)


def load_pdf_file(filename):
    try:
        response = requests.post(API_URL + "/api/load/" + filename, timeout=60)
        if response.status_code == 200:
            return True, ""
        else:
            try:
                error_msg = response.json().get("detail", "加载失败")
            except:
                error_msg = "加载失败，状态码: " + str(response.status_code)
            return False, error_msg
    except Exception as e:
        return False, "连接错误: " + str(e)


def delete_pdf_file(filename):
    try:
        response = requests.delete(API_URL + "/api/files/" + filename, timeout=10)
        if response.status_code == 200:
            return True, ""
        else:
            try:
                error_msg = response.json().get("detail", "删除失败")
            except:
                error_msg = "删除失败，状态码: " + str(response.status_code)
            return False, error_msg
    except Exception as e:
        return False, "连接错误: " + str(e)


def clear_chat_history():
    try:
        requests.post(API_URL + "/api/history/clear", timeout=5)
        st.session_state.messages = []
    except:
        pass


def clear_all_data():
    try:
        requests.post(API_URL + "/api/clear", timeout=5)
        st.session_state.messages = []
        st.session_state.files = []
        st.session_state.files_loaded = False
    except:
        pass


def send_chat_message(message):
    try:
        data = {"message": message}
        response = requests.post(API_URL + "/api/chat", json=data, timeout=120)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            return "请求失败，请重试"
    except Exception as e:
        return "连接错误: " + str(e)


st.set_page_config(
    page_title="论文智能助手",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session()
get_file_list()

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        width: 280px;
        min-width: 280px;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        padding: 1rem;
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
        height: calc(100vh - 140px);
        overflow-y: auto;
    }
    
    .message-row {
        display: flex;
        margin: 16px 0;
        align-items: flex-start;
    }
    
    .message-user {
        flex-direction: row-reverse;
    }
    
    .message-assistant {
        flex-direction: row;
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
    }
    
    .avatar-user {
        background: #007AFF;
        margin-left: 12px;
    }
    
    .avatar-assistant {
        background: #8E44AD;
        margin-right: 12px;
    }
    
    .bubble {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 12px;
        line-height: 1.6;
        word-wrap: break-word;
    }
    
    .bubble-user {
        background: #007AFF;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .bubble-assistant {
        background: #f5f5f5;
        color: #333;
        border-bottom-left-radius: 4px;
    }
    
    .input-container {
        position: fixed;
        bottom: 0;
        left: 280px;
        right: 0;
        background: white;
        padding: 16px 24px;
        border-top: 1px solid #eee;
    }
    
    .input-box {
        max-width: 900px;
        margin: 0 auto;
    }
    
    .empty-state {
        text-align: center;
        padding: 80px 20px;
        color: #999;
    }
    
    .empty-state h2 {
        margin-bottom: 12px;
        color: #666;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 8px;
        background: #f8f8f8;
    }
    
    .file-item-active {
        background: #e8f4fd;
    }
    
    .file-name {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 13px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active {
        background: #34C759;
    }
    
    .status-inactive {
        background: #ccc;
    }
    
    .stSpinner > div {
        font-size: 14px !important;
        white-space: nowrap;
    }
    
    .stSpinner {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📚 论文智能助手")
    st.markdown("---")
    
    st.markdown("**📤 上传论文**")
    uploaded_file = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        label_visibility="collapsed",
        key="upload",
        disabled=st.session_state.uploading
    )
    
    if uploaded_file is not None and not st.session_state.uploading:
        st.session_state.uploading = True
        
        with st.spinner("上传中..."):
            success, msg = upload_pdf_file(uploaded_file)
            
            if success:
                st.success("上传成功: " + msg)
                refresh_files()
                st.session_state.uploading = False
                st.rerun()
            else:
                st.error(msg)
                st.session_state.uploading = False
    
    st.markdown("---")
    st.markdown("**📁 论文列表**")
    
    if len(st.session_state.files) > 0:
        for file_info in st.session_state.files:
            filename = file_info["name"]
            is_loaded = file_info.get("loaded", False)
            
            safe_filename = escape_html(filename)
            
            if len(filename) > 18:
                display_name = safe_filename[:18] + "..."
            else:
                display_name = safe_filename
            
            if is_loaded:
                item_class = "file-item file-item-active"
                dot_class = "status-dot status-active"
            else:
                item_class = "file-item"
                dot_class = "status-dot status-inactive"
            
            st.markdown(f"""
            <div class="{item_class}">
                <div class="{dot_class}"></div>
                <span class="file-name">{display_name}</span>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("加载", key="load_" + filename, use_container_width=True):
                    with st.spinner("加载中..."):
                        success, err_msg = load_pdf_file(filename)
                        if success:
                            refresh_files()
                            st.rerun()
                        else:
                            st.error(err_msg)
            
            with col2:
                if st.button("删除", key="delete_" + filename, use_container_width=True):
                    success, err_msg = delete_pdf_file(filename)
                    if success:
                        refresh_files()
                        st.rerun()
                    else:
                        st.error(err_msg)
    else:
        st.info("暂无论文")
    
    st.markdown("---")
    
    if st.button("🗑️ 清空对话", use_container_width=True):
        clear_chat_history()
        st.rerun()
    
    if st.button("⚠️ 清空所有", use_container_width=True):
        clear_all_data()
        st.rerun()

current_paper_name = None
for file_info in st.session_state.files:
    if file_info.get("loaded") == True:
        current_paper_name = file_info["name"]
        break

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if len(st.session_state.messages) > 0:
    for msg in st.session_state.messages:
        safe_content = escape_html(msg["content"])
        
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message-row message-user">
                <div class="avatar avatar-user">😄</div>
                <div class="bubble bubble-user">{safe_content}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-row message-assistant">
                <div class="avatar avatar-assistant">🤖</div>
                <div class="bubble bubble-assistant">{safe_content}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    if current_paper_name is not None:
        safe_paper = escape_html(current_paper_name)
        if len(current_paper_name) > 20:
            paper_display = safe_paper[:20] + "..."
        else:
            paper_display = safe_paper
        welcome_text = "当前已加载: " + paper_display
    else:
        welcome_text = "上传论文开始提问，或直接对话"
    
    st.markdown(f"""
    <div class="empty-state">
        <h2>👋 欢迎使用论文智能助手</h2>
        <p>{welcome_text}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="input-container"><div class="input-box">', unsafe_allow_html=True)

if current_paper_name is not None:
    safe_paper_short = escape_html(current_paper_name)
    if len(current_paper_name) > 15:
        paper_short = safe_paper_short[:15] + "..."
    else:
        paper_short = safe_paper_short
    placeholder_text = "针对《" + paper_short + "》提问..."
else:
    placeholder_text = "输入你的问题..."

user_input = st.chat_input(placeholder_text, key="input")

if user_input is not None and st.session_state.pending_message is None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.pending_message = user_input
    st.rerun()

if st.session_state.pending_message is not None:
    with st.spinner("思考中..."):
        reply = send_chat_message(st.session_state.pending_message)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.pending_message = None
    st.rerun()

st.markdown('</div></div>', unsafe_allow_html=True)
