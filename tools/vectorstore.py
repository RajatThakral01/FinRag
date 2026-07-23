from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        _vectorstore = Chroma(
            persist_directory="./chroma_db",
            collection_name="financial_10k",
            embedding_function=embeddings,
        )
    return _vectorstore