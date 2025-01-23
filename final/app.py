



from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re
import streamlit as st

with open('eds_data.txt', 'r') as file:
    eds_data = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = [Document(page_content=eds_data)]
split_docs = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

rag_qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="gemma:2b"),
    retriever=vectorstore.as_retriever(),
)

base_model = Ollama(model="gemma:2b")

def extract_code_snippet(query, documentation):
    query = query.lower()
    pattern_map = {
        "button": r"<button.*?>.*?<\/button>",
        "container": r"<div class=\"eds-container.*?>.*?<\/div>",
        "form": r"<form.*?>.*?<\/form>",
        "login": r"<div class=\"eds-form.*?>.*?<\/div>",
    }

    for keyword, pattern in pattern_map.items():
        if keyword in query:
            matches = re.findall(pattern, documentation, re.DOTALL | re.IGNORECASE)
            if matches:
                return "\n".join(matches)

    return None

def is_ui_question(query):
    ui_keywords = ["eds-", "class", "button", "container", "alert", "card", "grid", "form", "navbar", "<", "login"]
    return any(keyword in query.lower() for keyword in ui_keywords)

def get_response(query):
    if is_ui_question(query):
        code_snippet = extract_code_snippet(query, eds_data)
        if code_snippet:
            return code_snippet

        response = rag_qa_chain.run(query)
        if not response or "context does not provide information" in response.lower():
            return "The EDS documentation may lack direct context for your query. Please refine your question."
        return response

    return base_model(query)

st.title("EDS Query System")
st.write("Ask questions about the EDS UI library or general topics!")

query = st.text_input("Enter your question:")

if query:
    response = get_response(query)
    st.write("Response:", response)
