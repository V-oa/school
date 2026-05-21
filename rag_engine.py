import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


os.environ["OPENAI_API_KEY"] = "sk-267e9a64cda14959b2a1d74949a92043"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"


_global_retriever = None
_global_llm_strict = None
_global_llm_casual = None
_global_strict_prompt = None
_global_casual_prompt = None

def format_docs(docs):
    """将向量库检索到的 Document 对象片段拼接为纯文本"""
    return "\n\n".join(doc.page_content for doc in docs)


class DualEngineManager:
   
    def invoke(self, user_input: str, mode: str) -> str:
        global _global_retriever, _global_llm_strict, _global_llm_casual, _global_strict_prompt, _global_casual_prompt
        
       
        if mode == "日常推理模式":
            casual_chain = _global_casual_prompt | _global_llm_casual | StrOutputParser()
            return casual_chain.invoke({"input": user_input})
            
       
        else:
            if _global_retriever is None:
                return "❌ 本地高校知识库未加载成功，请检查 data/shangqiu_handbook.txt 是否存在。"
            
   
            retrieved_docs = _global_retriever.invoke(user_input)
            context_text = format_docs(retrieved_docs)
            
            rag_chain = _global_strict_prompt | _global_llm_strict | StrOutputParser()
            return rag_chain.invoke({"context": context_text, "input": user_input})


def init_qa_chain():
    """系统初始化"""
    global _global_retriever, _global_llm_strict, _global_llm_casual, _global_strict_prompt, _global_casual_prompt
    
    
    pdf_path = "data/shangqiu_handbook.txt"
    if os.path.exists(pdf_path):
        loader = TextLoader(pdf_path, encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        splits = text_splitter.split_documents(docs)
        
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        _global_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    
    _global_llm_strict = ChatOpenAI(model="deepseek-chat", temperature=0.1,max_tokens=100)
    
    strict_system_prompt = (
        "你是一个高校智能助理。请严格根据以下学校官方文件的内容回答学生的问题。\n"
        "请用条理清晰的要点形式回答。如果参考内容中没有提到相关信息，请礼貌地回答：'抱歉，在学校公开的知识库中未查询到相关政策信息，建议同学详细咨询辅导员老师。'\n\n"
        "【学校官方文件参考内容】:\n{context}"
    )
    _global_strict_prompt = ChatPromptTemplate.from_messages([
        ("system", strict_system_prompt),
        ("human", "{input}"),
    ])
    
  
    _global_llm_casual = ChatOpenAI(model="deepseek-chat", temperature=0.7,max_tokens=100)
    casual_system_prompt = (
        "你是一个由 DeepSeek 驱动的高校智能助理，你的名字叫做ovo，现在化身为学生们的伙伴，你的回答温暖并且沉稳认真，很少有废话。\n"
        "当前用户在与你进行日常沟通，你可以帮他们解答日常常识、解决生活问题、写代码、日常吐槽等。\n"
        "此时【完全脱离学校手册的限制】，你可以调用你作为通用人工智能的一切智慧直接回答！不要说‘抱歉未查到’这种死板的话。"
    )
    _global_casual_prompt = ChatPromptTemplate.from_messages([
        ("system", casual_system_prompt),
        ("human", "{input}")
    ])
    
    return DualEngineManager()