import streamlit as st
from scripts.load_data import load_and_split_data
from scripts.build_index import build_or_load_index
from scripts.qa_chain import initialize_qa_chain


DATA_FILE = "data/eds_data.txt"
INDEX_FILE = "models/faiss_index.pkl"


st.write("Loading and processing data...")
docs = load_and_split_data(DATA_FILE)


st.write("Building or loading FAISS index...")
vectordb = build_or_load_index(docs, INDEX_FILE)


st.write("Initializing QA chain...")
qa_chain = initialize_qa_chain(vectordb)


st.title("Document Q&A with Ollama gemma:2b and FAISS")
st.write("Ask questions about the document and get precise answers!")


question = st.text_input("Ask a question about the document:")
if question:
    with st.spinner("Thinking..."):
        try:
          
            result = qa_chain({"query": question})
            
         
            st.write("### Answer:")
            st.write(result.get("result", "No answer found."))  
            
          
            source_docs = result.get("source_documents", [])
            if source_docs:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(source_docs):
                        st.write(f"#### Document {i + 1}:")
                        st.write(doc.page_content)
            else:
                st.write("No source documents available.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
