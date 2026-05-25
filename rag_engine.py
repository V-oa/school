import os
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings



os.environ["OPENAI_API_KEY"] = "sk-267e9a64cda14959b2a1d74949a92043"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com"



_global_retriever = None
_global_llm_strict = None
_global_llm_casual = None
_global_strict_prompt = None
_global_casual_prompt = None


def format_docs(docs):
    """将检索结果拼接为文本"""
    return "\n\n".join(doc.page_content for doc in docs)


class DualEngineManager:

    def invoke(self, user_input: str, mode: str) -> str:
        global _global_retriever
        global _global_llm_strict
        global _global_llm_casual
        global _global_strict_prompt
        global _global_casual_prompt


        if mode == "日常推理模式":

            casual_chain = (
                _global_casual_prompt
                | _global_llm_casual
                | StrOutputParser()
            )

            return casual_chain.invoke({
                "input": user_input
            })


        else:

            if _global_retriever is None:
                return "❌ 本地高校知识库未加载成功，请检查 chroma_db 是否存在。"

         
            retrieved_docs = _global_retriever.invoke(user_input)

    
            context_text = format_docs(retrieved_docs)

   
            rag_chain = (
                _global_strict_prompt
                | _global_llm_strict
                | StrOutputParser()
            )

            return rag_chain.invoke({
                "context": context_text,
                "input": user_input
            })


def init_qa_chain():
    """初始化系统"""

    global _global_retriever
    global _global_llm_strict
    global _global_llm_casual
    global _global_strict_prompt
    global _global_casual_prompt



    if os.path.exists("chroma_db"):

        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5"
        )

        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )

        _global_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )



    _global_llm_strict = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0.1,
        max_tokens=100
    )

    strict_system_prompt = (
        "你是高校助手。"
        "仅依据给定条文，以要点形式严谨回答。\n\n"

        "请用条理清晰的要点形式回答。"
        "如果参考内容中没有提到相关信息，"
        "请礼貌地回答："
        "‘抱歉，在学校公开的知识库中未查询到相关政策信息，建议同学详细咨询辅导员老师。’\n\n"

        "【学校官方文件参考内容】:\n"
        "{context}"
    )

    _global_strict_prompt = ChatPromptTemplate.from_messages([
        ("system", strict_system_prompt),
        ("human", "{input}")
    ])


    _global_llm_casual = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0.7,
        max_tokens=100
    )

    casual_system_prompt = (
        "你是由 DeepSeek 驱动的高校助手。"
        "你的名字叫 ovo。"
        "你现在化身为学生们的伙伴。"
        "你的回答温暖、认真、简洁。\n\n"

        "当前用户正在与你进行日常交流。"
        "你可以回答："
        "日常常识、代码问题、学习问题、生活问题等。\n\n"

        "此时完全脱离学校手册限制，"
        "你可以自由使用你的通用人工智能能力进行回答。"
    )

    _global_casual_prompt = ChatPromptTemplate.from_messages([
        ("system", casual_system_prompt),
        ("human", "{input}")
    ])

    return DualEngineManager()