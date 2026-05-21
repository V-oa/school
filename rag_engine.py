import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


os.environ["OPENAI_API_KEY"] = "sk-267e9a64cda14959b2a1d74949a92043"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1" 


def format_docs(docs):
    """把检索出来的文档片段拼接成纯文本"""
    return "\n\n".join(doc.page_content for doc in docs)

def init_qa_chain():
    
    
    
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.1)
    
    
    system_prompt = (
        "你是一个高校智能助理。请严格根据以下学校官方文件的内容回答学生的问题。\n"
        "如果文件中没有提到相关内容，请礼貌地回答：'抱歉，在学校公开的知识库中未查询到相关信息。'\n"
        "请用条理清晰的要点形式回答。\n\n"
        "【学校文件内容】:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain