import streamlit as st
from rag_engine import init_qa_chain


st.set_page_config(
    page_title="商丘师范学院-智能问答系统", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    
    #MainMenu {visibility: hidden;}
    
   
    footer {visibility: hidden;}
    

    stDeployButton {display:none;}
    [data-testid="stStatusWidget"] {display:none !important;}
    .st-emotion-cache-12fm60b {display:none !important;} 
    .st-emotion-cache-6q9sum {display:none !important;}
    header {visibility: hidden;} 
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    try:
        st.image("https://www.sqnu.edu.cn/images/logo.png")
    except Exception:
        pass
    st.markdown("## 🎓 系统控制台")
    st.markdown("---")
    st.info("💡 **当前数据库**：\n《商丘师范学院学生手册》")
    st.success("🤖 **核心引擎**：\nDeepSeek-Chat (RAG增强)")
    st.warning("📍 **演示校区**：\n睢阳校区 / 梁园校区")
    
    st.markdown("---")
    
    if st.button("🧹 清空聊天历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.caption("©️ 2026 商丘师范学院郭双瑞毕业论文演示")


st.title("🎓 商丘师范学院知识库智能问答系统")
st.subheader("基于 RAG 技术的校园政策与教务管理规定智能查询平台")
st.markdown("---")


@st.cache_resource
def load_rag_system():
    return init_qa_chain()

try:
    rag_chain = load_rag_system()
except Exception as e:
    st.error(f"❌ 系统初始化失败，请检查网络或 API Key 配置。错误详情: {e}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input := st.chat_input("请输入你想咨询的商丘师院政策（如：挂科几门取消学位？）"):
    
   
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索校园知识库并组织语言..."):
            try:
                
                answer = rag_chain.invoke(user_input)
                
                
                st.markdown(answer)
                
                
                st.toast("回答生成完毕！数据源自官方手册。", icon="✅")
                
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"请求失败，请检查网络状况: {e}")