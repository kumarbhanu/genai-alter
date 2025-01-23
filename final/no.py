# from langchain.vectorstores import FAISS
# from langchain.embeddings import OllamaEmbeddings
# from langchain.llms import Ollama
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import streamlit as st

# # Load `.txt` file directly with TextLoader
# file_path = "eds_data.txt"  # Replace with the path to your file
# loader = TextLoader(file_path, encoding="utf-8")
# documents = loader.load()

# # Split the text into smaller chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# docs = text_splitter.split_documents(documents)

# # Initialize the embedding function and FAISS vector database
# embedding = OllamaEmbeddings(model="gemma:2b")
# vectordb = FAISS.from_documents(docs, embedding)

# # Define the Ollama model and prompt
# llm = Ollama(model="gemma:2b")
# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum. Always say "Thanks for asking!" at the end.
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

# # Create a RetrievalQA chain
# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

# # Streamlit UI
# st.title("Document Q&A with Ollama gemmaL2b and FAISS")
# st.write("This app uses a preloaded text file to answer your questions. Ask away!")

# # User Input for Question
# question = st.text_input("Ask a question about the document:")
# if question:
#     with st.spinner("Thinking..."):
#         # Run the chain to get the answer
#         result = qa_chain({"query": question})
    
#     # Display results
#     st.write("### Answer:")
#     st.write(result["result"])
    
#     # Optionally display source documents
#     with st.expander("View Source Documents"):
#         for i, doc in enumerate(result["source_documents"]):
#             st.write(f"#### Document {i + 1}:")
#             st.write(doc.page_content)

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

embedding = OllamaEmbeddings(model="gemma:2b")
vectordb = FAISS.from_documents(docs, embedding)

llm = Ollama(model="gemma:2b")

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
    You are an AI assistant trained to answer questions about the EDS UI Library.
    Here is the context:
    {context}
    
    Chat History:
    {chat_history}
    
    Question:
    {question}
    
    Provide a concise and informative answer.
    """
)

class ChatHistoryQAChain:
    def __init__(self, llm, retriever, prompt_template):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template

    def __call__(self, inputs):
        inputs["chat_history"] = inputs.get("chat_history", "")
        context_docs = self.retriever.get_relevant_documents(inputs["query"])
        context = "\n".join([doc.page_content for doc in context_docs])
        prompt = self.prompt_template.format(
            context=context,
            chat_history=inputs["chat_history"],
            question=inputs["query"],
        )
        response = self.llm(prompt)
        return {"result": response, "source_documents": context_docs}

qa_chain = ChatHistoryQAChain(
    llm=llm, retriever=vectordb.as_retriever(), prompt_template=QA_CHAIN_PROMPT
)

st.title("EDS-Aware GenAI App")
st.write("Get tailored responses for your company's EDS UI Library.")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = ""

question = st.text_input("Ask about the EDS UI Library:")

if question:
    st.session_state["chat_history"] += f"User: {question}\n"
    with st.spinner("Processing..."):
        result = qa_chain({"query": question, "chat_history": st.session_state["chat_history"]})
        answer = result["result"]

    st.session_state["chat_history"] += f"AI: {answer}\n"
    st.write("### Answer:")
    st.write(answer)

    with st.expander("View Source Documents"):
        for i, doc in enumerate(result["source_documents"]):
            st.write(f"Document {i + 1}: {doc.page_content}")

    with st.expander("View Chat History"):
        st.text(st.session_state["chat_history"])
