import streamlit as st
from rag_engine import init_qa_chain

st.set_page_config(
    page_title="商丘师范学院-智能问答系统", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("## 🎓 系统控制台")
    st.markdown("---")
    st.info("💡 **当前数据库**：\n《商丘师范学院学生手册》")
    st.success("🤖 **核心引擎**：\nDeepSeek-Chat (RAG增强)")
    st.warning("📍 **演示校区**：\n睢阳校区 / 文化路校区")
    
    if st.button("🧹 清空聊天历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("🎓 商丘师范学院知识库智能问答系统")
st.subheader("基于 RAG 技术的校园政策与教务管理规定智能查询平台")
st.markdown("---")

@st.cache_resource
def load_rag_system():
    return init_qa_chain()

try:
    rag_chain = load_rag_system()
except Exception as e:
    st.error(f"❌ 系统初始化失败: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("请输入你想咨询的商丘师院政策"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索..."):
            try:
                answer = rag_chain.invoke(user_input)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"请求失败: {e}")