import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 保持你的环境变量配置一致
os.environ["OPENAI_API_KEY"] = "sk-267e9a64cda14959b2a1d74949a92043"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# 定义全局组件变量，供动态路由类内部安全调用
_global_retriever = None
_global_llm = None
_global_strict_prompt = None

def format_docs(docs):
    """把检索出来的文档片段拼接成纯文本"""
    return "\n\n".join(doc.page_content for doc in docs)

class IntelligentRagProxy:
    """
    创新架构亮点：双轨制智能语义路由拦截器
    通过判断底层向量检索的召回内容是否为空，动态在【严谨RAG模式】与【开放域闲聊模式】间灵活切换
    """
    def invoke(self, user_input: str) -> str:
        global _global_retriever, _global_llm, _global_strict_prompt
        
        if _global_retriever is None or _global_llm is None:
            return "系统尚未完全初始化，请刷新页面重试。"
            
        # 1. 预先在后台执行轻量级检索
        retrieved_docs = _global_retriever.invoke(user_input)
        context_text = format_docs(retrieved_docs)
        
        # 2. 语义分流路由核心逻辑判断
        if not context_text.strip() or len(context_text.strip()) < 5:
            # 【分支A】：未匹配到本地关联知识（属于日常闲聊、打招呼或通用常识，如西瓜大还是苹果大、几点了）
            # 此时直接跳过RAG，放开束缚，让DeepSeek扮演极高情商的老学长
            casual_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "你是一个专门为师生服务的全能高校智能助理（身份通常是校园里热情幽默的老学长或老学姐）。\n"
                    "当前用户的输入属于日常问候、通用常识、无厘头闲聊或时间查询（由于本地校规手册无此类信息，此处已为你解除知识库限制）。\n"
                    "请【完全忽略】任何关于学生手册或学校知识库的字眼，不要说‘抱歉未查询到相关信息’之类的话！\n"
                    "直接调用你丰富的原生大模型常识储备，用一种严谨有温度、贴近大学生活的校园口吻，直接并完美地回答用户的问题。\n"
                    "例如：用户问西瓜还是苹果大，你要直接告诉他西瓜大；问几点了，提醒他看手机。"
                )),
                ("human", "{input}")
            ])
            casual_chain = casual_prompt | _global_llm | StrOutputParser()
            return casual_chain.invoke({"input": user_input})
            
        else:
            # 【分支B】：精准命中本地学生手册文本（属于挂科、转专业、奖学金、宿舍断电等校务硬性规章）
            # 此时启动严格的严谨RAG生成轨，确保校规输出百分之百权威准确
            rag_chain = _global_strict_prompt | _global_llm | StrOutputParser()
            return rag_chain.invoke({"context": context_text, "input": user_input})


def init_qa_chain():
    global _global_retriever, _global_llm, _global_strict_prompt
    
    # 1. 加载本地知识库文本
    pdf_path = "data/shangqiu_handbook.txt"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"未找到文件：{pdf_path}")
        
    loader = TextLoader(pdf_path, encoding="utf-8")
    docs = loader.load()
    
    # 2. 文本高密度切片
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    splits = text_splitter.split_documents(docs)
    
    # 3. 构建高维向量数据库并生成索引
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # 建立高级检索适配器（已锁定标准k参数）
    _global_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. 初始化底层大语言模型驱动核心
    # 将温度稍微调到 0.2，既保证校规的严谨，又能给闲聊带来灵动的校园生命力
    _global_llm = ChatOpenAI(model="deepseek-chat", temperature=0.2)
    
    # 5. 定义针对严肃校务政策的系统级标准约束 Prompt
    strict_system_prompt = (
        "你是一个高校智能助理。请严格根据以下学校官方文件的内容回答学生的问题。\n"
        "请用条理清晰的要点形式回答。如果文件中没有提到对应政策，请礼貌地回答：'抱歉，在学校公开的知识库中未查询到相关政策信息，建议同学咨询辅导员哦。'\n\n"
        "【学校官方文件参考内容】:\n{context}"
    )
    _global_strict_prompt = ChatPromptTemplate.from_messages([
        ("system", strict_system_prompt),
        ("human", "{input}"),
    ])
    
    # 6. 返回代理实例，无缝契合前端 app.py 的 .invoke() 链式调用
    return IntelligentRagProxy()