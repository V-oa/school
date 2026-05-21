import streamlit as st
from rag_engine import init_qa_chain

st.set_page_config(
    page_title="高校智能问答系统", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_rag_system():
    return init_qa_chain()

try:
    dual_engine = load_rag_system()
except Exception as e:
    st.error(f"❌ 系统底层初始化失败，请检查网络或 API 秘钥配置。错误详情: {e}")
    st.stop()


with st.sidebar:
    st.markdown("## ⚙️MENU(SUDENT)")
    st.markdown("---")
    

    system_mode = st.selectbox(
        " 🌟当前运行模式:",
        ["高校数据库模式", "日常推理模式"],
        index=0
    )
    
    st.markdown("---")
    if system_mode == "高校数据库模式":
        st.success("📖 **当前状态**：校园手册 AI \n(回答严格遵循学校官方规章)")
    else:
        st.info("📖 **当前状态**：日常全能 AI \n(适合常识回答、日常闲聊)")
        
    st.warning("🛜 **推理模型**：\nDeepSeek-v4-pro")
    st.markdown("---")
   
    if st.button("❌ 清空当前聊天历史", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.caption("©️ 2447301104 毕业设计演示")



st.title("🎓 高校大模型智能问答系统")
st.subheader("基于RAG技术的校园政策与教务管理规定智能查询平台")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if system_mode == "高校数据库模式":
   
    placeholder_tips = "✨ 当前处于【高校数据库模式】，请输入您要查询的校园政策（如：挂科几门取消学位？）"
else:
  
    placeholder_tips = "✨ 当前处于【日常推理模式】，可以向 AI 提问任何事情..."


if user_input := st.chat_input(placeholder_tips):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        with st.spinner(f"🔍 正在切换至 {system_mode} 并组织语言..."):
            try:
               
                answer = dual_engine.invoke(user_input, mode=system_mode)
                st.markdown(answer)
                st.toast(f"已通过 {system_mode} 渲染输出。", icon="✅")
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"⚠️ 系统在处理您的请求时出现异常: {e}")