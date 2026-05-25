from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


loader = TextLoader("data/shangqiu_handbook.txt", encoding="utf-8")
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40
)

splits = text_splitter.split_documents(docs)


embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="chroma_db"
)

vectorstore.persist()

print("知识库构建完成！")