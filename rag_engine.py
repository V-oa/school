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
        "你是一个专门为全校师生服务的高校全能智能助理（老学长/老学姐）。\n"
        "你需要根据用户的输入，在【校务场景】与【生活闲聊场景】之间做出清醒的判断：\n\n"
        "【本地官方规章参考内容（仅用于回答校务及宿舍问题）】:\n{context}\n\n"
        "【核心行为准则（请务必严格遵守）】:\n"
        "1. 针对【通用常识与日常闲聊】：\n"
        "   - 当用户问的是‘你好’、‘在吗’、‘现在几点了’、‘苹果大还是西瓜大’等任何日常问候、通用常识或轻松的闲聊交互时；\n"
        "   - 此时与学校官方文件内容完全无关！请你【立刻彻底放开所有知识库限制】！\n"
        "   - 绝对不允许说‘抱歉，在学校公开的知识库中未查询到相关信息’！\n"
        "   - 请直接调用你强大的原生知识储备，用幽默、风趣、充满老学长关怀的口吻直接回答他（例如直接明确告诉他西瓜比苹果大得多）。\n\n"
        "2. 针对【学校具体政策】：\n"
        "   - 当用户问的是关于挂科、学籍、转专业、毕业论文、宿舍作息等学校具体规章制度时；\n"
        "   - 请务必严格、严谨地根据【本地官方规章参考内容】提取要点进行规范回答，严禁胡乱编造任何校规条文。\n"
        "   - 只有当用户问的是【学校政策问题】且在参考内容中完全没有提到时，才允许礼貌地回答：'抱歉，在学校公开的知识库中未查询到相关信息，建议同学咨询一下辅导员哦。'\n\n"
        "始终保持热情、充满正能量的校园老学长助理语气。"
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