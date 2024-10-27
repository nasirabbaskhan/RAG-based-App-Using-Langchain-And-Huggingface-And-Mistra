
# import for LLM
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# import for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
# import for streamlit prompt template
from langchain_core.prompts import ChatPromptTemplate
# import for streamlit app
import streamlit as st
# import for env 
from dotenv import load_dotenv
import os
# import for similarty checking
# from torch import cosine_similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

load_dotenv()


huggingfacehub_api_token =  os.getenv("HUGGINGFACE_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# RAG
loader = PyPDFLoader("research.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)

# embaddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embaddings = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)

vector = FAISS.from_documents(documents, embedding=embaddings)
retriever = vector.as_retriever()



#1 creating the inference of HuggingFaceEndpoint
llm  = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
    model=""
)




prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Provide the most accurate response based on the question:
    <context>
    {context}
    </context>
    
    Question: {input}

    
    """
)

# creating chaining 
document_chain = create_stuff_documents_chain(llm, prompt)


chain = create_retrieval_chain(retriever, document_chain)

# Relevance threshold for similarity
RELEVANCE_THRESHOLD = 0.7

# Relevance check function
def is_relevant_question(question, retriever):
    # Generate embedding for the question
    question_embedding = embaddings.embed_query(question)

    # Retrieve the most relevant document (if any)
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return False  # No relevant docs found

    # Embed the content of the top retrieved document
    top_doc_embedding = embaddings.embed_query(docs[0].page_content)

    # Convert embeddings to NumPy arrays for cosine similarity calculation
    question_embedding_np = np.array(question_embedding).reshape(1, -1)
    top_doc_embedding_np = np.array(top_doc_embedding).reshape(1, -1)

    # Check the similarity between question and the top document's embedding
    similarity_score = cosine_similarity(question_embedding_np, top_doc_embedding_np)[0][0]

    return similarity_score >= RELEVANCE_THRESHOLD

# invoking to get response
def get_response_from_model(user_input):
    response= chain.invoke({"input":user_input})  
    return response
   
    

# # streamlit app

st.title("Hugging Fasce RAG basd App")
user_input = st.text_input("Ask That You Want:")

submit = st.button("Ask the Question")

if submit:
    if is_relevant_question(user_input, retriever):
        response = get_response_from_model(user_input)
        st.subheader("The Response is:")
        st.write(response)
    else:
        st.subheader("The Response is:")
        st.write("I am trained on the research paper 'Attention is All You Need,' so please ask a relevant question.")