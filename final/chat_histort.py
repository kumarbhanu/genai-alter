from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

file_path = "eds_data.txt"
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embedding = OllamaEmbeddings(model="codegemma:2b")
vectordb = FAISS.from_documents(docs, embedding)

llm = Ollama(model="codegemma:2b")
template = """
Use the following pieces of context to answer the question at the end.

If the question asks for a button, input, table, or any HTML code, provide the relevant information in plain text. If the question contains HTML code, convert it into the equivalent EDS code. Provide the answer with examples in EDS-specific syntax where applicable.
If the question asks for HTML or code, please convert it to EDS format add specific eds.
If the question is not related to components, answer it in plain text using the context provided.
If you don't know the answer, just say that you don't know and avoid making up an answer.

Use three sentences maximum. Always say "Thanks for asking!" at the end.

### Context:
{context}

### Question: 
{question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

st.title("Document Q&A with Ollama gemmaL2b and FAISS")
st.write("This app uses a preloaded text file to answer your questions. Ask away!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask a question about the document:")
threshold = st.slider("Set similarity threshold (0.0 - 1.0):", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

if question:
    with st.spinner("Thinking..."):
        result = qa_chain({"query": question})
        filtered_docs = [doc for doc in result["source_documents"] if doc.metadata.get("similarity", 1.0) >= threshold]
        st.session_state.chat_history.append((question, result["result"]))

    st.write("### Answer:")
    st.write(result["result"])

    st.write("### Chat History:")
    for q, a in st.session_state.chat_history:
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")

    with st.expander("View Source Documents"):
        for i, doc in enumerate(filtered_docs):
            st.write(f"#### Document {i + 1}:")
            st.write(doc.page_content)
