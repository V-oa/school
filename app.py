import streamlit as st
from rag_engine import init_qa_chain

st.set_page_config(
    page_title="商丘师范学院 - SQNU RAG", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, sans-serif;
        background-color: #0E1117 !important;
        color: #E2E8F0 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D !important;
    }
    
    div[data-testid="stChatMessage"] {
        background-color: #1F2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        padding: 15px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    div[data-testid="stStatusWidget"] {
        border: 1px solid #3B82F6 !important;
        background-color: #1D4ED822 !important;
        border-radius: 8px !important;
    }
    
    .stButton>button {
        background-color: #21262D !important;
        color: #C9D1D9 !important;
        border: 1px solid #30363D !important;
        border-radius: 6px !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: #58A6FF !important;
        color: #58A6FF !important;
        background-color: #30363D !important;
    }
    
    .stChatFloatingInputContainer {
        background-color: #0E1117 !important;
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #58A6FF; font-weight: 600; letter-spacing: 1px;'>SQNU RAG ENGINE</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8B949E; font-size: 12px; margin-top: -10px;'>校园知识库增强检索系统</p>", unsafe_allow_html=True)
    st.markdown("<div style='border-bottom: 1px solid #30363D; margin: 15px 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("<p style='color: #8B949E; font-size: 13px; margin-bottom: 5px;'>📊 ENVIROMENT</p>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 12px; font-size: 14px; color: #C9D1D9; line-height: 1.6;'>
            • <b>CORE:</b> DeepSeek-Chat<br>
            • <b>VECTOR:</b> BAAI/bge-small-zh<br>
            • <b>DATASET:</b> 商丘师院 student_handbook<br>
            • <b>STATUS:</b> <span style='color: #39D353;'>● Connected</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    if st.button("🧹 Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #484F58; font-size: 11px;'>SYSTEM VERSION V2.6.0<br>© 2026 SQNU LAB</p>", unsafe_allow_html=True)

st.markdown("<h1 style='color: #F0F6FC; font-weight: 600; font-size: 32px; letter-spacing: -0.5px;'>商丘师范学院知识库智能问答系统</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8B949E; font-size: 15px; margin-top: -8px;'>基于 RAG 架构的数字化校园政策与教务管理规定智能解构平台</p>", unsafe_allow_html=True)
st.markdown("<div style='border-bottom: 1px solid #21262D; margin: 20px 0;'></div>", unsafe_allow_html=True)

@st.cache_resource
def load_rag_system():
    return init_qa_chain()

try:
    rag_chain = load_rag_system()
except Exception as e:
    st.error(f"❌ 系统大脑初始化失败: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("输入校务或政策问题进行智能检索..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=True) as status:
            try:
                answer = rag_chain.invoke(user_input)
                status.update(label="Response generated from knowledge base:", state="complete", expanded=False)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"请求失败，原因: {e}")