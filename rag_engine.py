import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 环境变量保持不变
os.environ["OPENAI_API_KEY"] = "sk-267e9a64cda14959b2a1d74949a92043"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"

# 全局组件变量
_retriever = None
_llm = None
_strict_prompt = None

def format_docs(docs):
    """把检索出来的文档片段拼接成纯文本"""
    return "\n\n".join(doc.page_content for doc in docs)

def init_qa_chain():
    global _retriever, _llm, _strict_prompt
    
    pdf_path = "data/shangqiu_handbook.txt"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"未找到文件：{pdf_path}")
        
    loader = TextLoader(pdf_path, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    splits = text_splitter.split_documents(docs)
    
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # 严格锁定的检索参数
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    _llm = ChatOpenAI(model="deepseek-chat", temperature=0.3)
    
    # 场景A：匹配到知识库时的标准提示词
    strict_system_prompt = (
        "你是一个高校智能助理。请严格根据以下学校官方文件的内容回答学生的问题。\n"
        "如果文件中没有提到对应政策，请礼貌地回答：'抱歉，在学校公开的知识库中未查询到相关信息，建议咨询辅导员。'\n"
        "请用条理清晰的要点形式回答。\n\n"
        "【学校文件内容】:\n{context}"
    )
    _strict_prompt = ChatPromptTemplate.from_messages([
        ("system", strict_system_prompt),
        ("human", "{input}"),
    ])
    
    # 返回拦截代理代理实例
    return IntelligentRagProxy()

class IntelligentRagProxy:
    """智能动态路由：根据检索结果是否为空切断或启动RAG"""
    def invoke(self, user_input: str) -> str:
        global _retriever, _llm, _strict_prompt
        
        # 💡【核心修正点】：把 get_relevant_documents 换成新版规范的 invoke
        retrieved_docs = _retriever.invoke(user_input)
        context_text = format_docs(retrieved_docs)
        
        # 路由拦截分支判断
        if not context_text.strip():
            # 分支1：无相关本地知识 ➔ 解锁大模型原生大脑，自由闲聊
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "你是一个高校全能智能助理（老学长/老学姐）。\n"
                    "现在用户在和你问候、查询当前时间或闲聊通用生活常识（例如问苹果和西瓜哪个大）。\n"
                    "请【完全脱离】学校文件的限制，直接利用你丰富的常识储备给出一个符合客观事实、幽默且有校园温度的精彩回答。"
                )),
                ("human", "{input}")
            ])
            chat_chain = chat_prompt | _llm | StrOutputParser()
            return chat_chain.invoke({"input": user_input})
            
        else:
            # 分支2：存在本地参考内容 ➔ 强依附RAG规章制度回答
            rag_chain = _strict_prompt | _llm | StrOutputParser()
            return rag_chain.invoke({"context": context_text, "input": user_input})