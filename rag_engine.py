import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


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
    

    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    

    retriever = vectorstore.as_retriever(search_kwargs={"1": 3})
    

    llm = ChatOpenAI(model="deepseek-chat", temperature=0.1)
    

    system_prompt = (
        "你是一个专门为全校师生服务的高校全能智能助理。\n"
        "请结合以下由本地向量数据库检索到的学校官方规章制度内容，灵活回答用户的问题。\n\n"
        "【本地官方规章参考内容】:\n{context}\n\n"
        "【核心回答准则（请务必严格遵守）】:\n"
        "1. 如果用户问的是关于挂科、学籍、宿舍、转专业、毕业论文、生活起居等【学校具体政策或校园生活】：\n"
        "   - 请务必严格根据【本地官方规章参考内容】进行条理清晰的要点形式回答，禁止胡乱编造任何校规。\n"
        "   - 如果参考内容中完全没有提到相关的校园政策，请礼貌且委婉地回答：'抱歉，在学校公开的知识库中未查询到相关信息，建议同学咨询一下辅导员哦。'\n\n"
        "2. 如果用户问的是【通用生活常识】（如“苹果大还是西瓜大”）、【日常问候】（如“你好”、“在吗”）、或【闲聊交流】：\n"
        "   - 此时这些问题与本地学校文件不相关，请你【完全忽略】知识库的限制！\n"
        "   - 直接调用你作为通用大语言模型自身的智慧和常识储备，给出一个符合客观事实的贴心回答，千万不要拒绝回答！\n\n"
        "始终保持热情、充满正能量的校园助理语气。"
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