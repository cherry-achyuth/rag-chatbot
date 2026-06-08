from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

def load_documents(folder_path):
    documents = []
    pdf_files = Path(folder_path).glob("*.pdf")

    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        documents.extend(loader.load())
    return documents

from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap = 20
    )
    chunks = splitter.split_documents(documents)
    return chunks

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings

from langchain_community.vectorstores import Chroma

def create_vector_store(chunks):
    embedding_models = get_embedding_model()
    vectordb = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_models,
        persist_directory = "vectordb"
    )
    vectordb.persist()
    print("vector db created successfully")

if __name__ == "__main__":
    docs = load_documents("data/documents")
    chunks = split_documents(docs)
    create_vector_store(chunks)