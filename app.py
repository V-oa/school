import streamlit as st
from rag_engine import init_qa_chain

st.set_page_config(
    page_title="商丘师范学院 - 智能问答系统", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    header {visibility: hidden;} 
    footer {visibility: hidden;}
    .stChatFloatingInputContainer {background-color: transparent !important;}
    div[data-testid="stChatMessage"] {
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #4F8BF0;'>🎓 控制台</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📋 运行状态")
    st.info("📂 **当前知识库**：\n《商丘师范学院学生手册》")
    st.success("🤖 **核心引擎**：\nDeepSeek-Chat (RAG增强)")
    st.warning("📍 **适用校区**：\n睢阳校区 / 💡 文化路校区")
    
    st.markdown("---")
    st.markdown("### ⚙️ 系统操作")
    if st.button("🧹 清空当前聊天历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
    st.caption("<p style='text-align: center; color: gray;'>©️ 2026 商丘师范学院大作业演示</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 10])
with col1:
    st.markdown("<h1 style='font-size: 50px; margin-top: -10px;'>🎓</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='color: #1E3A8A; margin-top: -15px;'>商丘师范学院知识库智能问答系统</h1>", unsafe_allow_html=True)

st.markdown("<p style='color: #555555; font-size: 16px; margin-top: -10px;'>基于 RAG 检索增强技术 · 数字化校园教务与政策智能查询平台</p>", unsafe_allow_html=True)
st.markdown("---")

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

if user_input := st.chat_input("请输入你想咨询的商丘师院政策（例如：挂科几门取消学位？）"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.status("🔍 正在检索校园数据库并组织语言...", expanded=True) as status:
            try:
                answer = rag_chain.invoke(user_input)
                status.update(label="✅ 检索完成！结合官方手册为您解答：", state="complete", expanded=False)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                status.update(label="❌ 检索失败", state="error")
                st.error(f"请求失败，原因: {e}")