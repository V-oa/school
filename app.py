import streamlit as st
from rag_engine import init_qa_chain


st.set_page_config(
    page_title="智能问答系统", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)


with st.sidebar:
    
    
    st.markdown("## 🎓 系统控制台")
    st.markdown("---")
    st.info("📖 **当前数据库**：\n《某高校学生手册》")
    st.success("🛜 **大语言模型**：\nDeepseek-v4-flash")
    st.warning("🏫 **学校校区**：\n睢阳校区 / 梁园校区")
    
    st.markdown("---")
   
    if st.button("❌ 清空聊天历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.caption("©️ 2447301104 毕业设计演示")


st.title("🎓 数据库智能问答系统")
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


if user_input := st.chat_input("请输入你想咨询的学校政策（如：挂科几门取消学位？）"):
    
   
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
   
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索数据库并组织语言..."):
            try:
               
                answer = rag_chain.invoke(user_input)
                
                
                st.markdown(answer)
                
                
                st.toast("回答生成完毕！数据源自官方手册。", icon="✅")
                
               
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"请求失败，请检查网络状况: {e}")