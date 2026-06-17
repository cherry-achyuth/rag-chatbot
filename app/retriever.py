from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_PATH, EMBEDDINNG_MODEL

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDINNG_MODEL
    )

    vectordb = Chroma(
        persist_directory= CHROMA_PATH,
        embedding_function=embeddings
    )
    return vectordb

def retrieve_documents(query, k =3):
    vectordb = load_vectorstore()

    docs = vectordb.similarity_search(
        query=query,
        k=k
    )
    return docs