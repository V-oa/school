import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


os.environ["OPENAI_API_KEY"] = "sk-267e9a64cda14959b2a1d74949a92043"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1" 


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
    
   
    _retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    _llm = ChatOpenAI(model="deepseek-chat", temperature=0.3) 
    
   
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
    
   
    return IntelligentRagProxy()

class IntelligentRagProxy:
    """智能动态拦截代理：判断是否命中本地知识库"""
    def invoke(self, user_input: str) -> str:
        global _retriever, _llm, _strict_prompt
        
       
        retrieved_docs = _retriever.get_relevant_documents(user_input)
        context_text = format_docs(retrieved_docs)
        
       
        if not context_text.strip():
     
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个高校全能智能助理（老学长/老学姐）。现在用户在和你问候或闲聊通用常识（例如问时间、比水果大小）。请完全用你自身的丰富常识和幽默风趣的语言回答，保持关怀与温度。"),
                ("human", "{input}")
            ])
            chat_chain = chat_prompt | _llm | StrOutputParser()
            return chat_chain.invoke({"input": user_input})
            
        else:
     
            rag_chain = _strict_prompt | _llm | StrOutputParser()
            return rag_chain.invoke({"context": context_text, "input": user_input})