from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def initialize_qa_chain(vectordb, llm_model="gemma:2b"):
    llm = Ollama(model=llm_model)
 

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
    qa_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    
 
    return RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,  
        chain_type_kwargs={"prompt": qa_prompt}
    )
