import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

def build_or_load_index(docs, index_path="models/faiss_index.pkl", embedding_model="gemma:2b"):
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, OllamaEmbeddings(model=embedding_model),allow_dangerous_deserialization=True)
    
    embedding = OllamaEmbeddings(model=embedding_model)
    vectordb = FAISS.from_documents(docs, embedding)
    vectordb.save_local(index_path)
    return vectordb


