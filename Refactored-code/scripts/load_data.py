from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_data(file_path, chunk_size=1000, chunk_overlap=50):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)
