pip install langchain transformers ipywidgets


-------------------------------

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import ipywidgets as widgets
from IPython.display import display
-------------------------------------------

# Hardcoded text content
hardcoded_text = """
This is an example document. It contains multiple lines of text and is used to test the retrieval-based QA system.
Feel free to replace this with your actual text content.
The goal is to demonstrate how the system handles hardcoded text input directly in the code.
"""

# Create documents from the hardcoded text
documents = [{"page_content": hardcoded_text}]
---------------------------------------------------
# Split the text into smaller chunks for better processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
----------------------------------------------
# Initialize the Hugging Face embedding model and FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = FAISS.from_documents(docs, embedding_model)
---------------------------------------------------------
llm_pipeline = pipeline(  "text-generation",
    model="EleutherAI/gpt-neo-2.7B",
    device=0,  # GPU usage
    pad_token_id=50256)  # Or max_new_tokens=50

# Continue with the rest of your code
llm = HuggingFacePipeline(pipeline=llm_pipeline)
---------------------------------------------------
template = """
Use the following pieces of context to answer the question at the end.

If the question asks for a button, input, table, or any HTML code, provide the relevant information in plain text. Clearly explain how to use the component in EDS or standard HTML with an example.

If the input contains HTML code, identify the EDS-specific equivalent or alternative for the HTML. Provide the answer with examples in EDS-specific syntax.

If the question is not related to components, answer it in plain text using the context provided. 
If you don't know the answer, just say that you don't know and avoid making up an answer.

Use three sentences maximum. Always say "Thanks for asking!" at the end.

Context:
{context}

Question: {question}

Helpful Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
--------------------------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
-----------------------------


 result = qa_chain({"query": "button in eds"})
            print("\n### Answer:")
            print(result["result"])



# from langchain.schema import Document

# documents = [{"page_content": hardcoded_text, "metadata": {"key": "value"}}]

# # Convert to LangChain Document objects
# docs = [Document(page_content=doc["page_content"], metadata=doc.get("metadata", {})) for doc in documents]

# # Split documents
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
# split_docs = text_splitter.split_documents(docs)