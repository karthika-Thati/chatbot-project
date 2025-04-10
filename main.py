import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.schema import Document
import os
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound, PermissionDenied
from typing import List
import google.generativeai as genai

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("AIzaSyC2x_PUIGEbUQZITVZpzmtwchMkwHMX0Qo")

def create_chatbot(knowledge_file):
    loader = TextLoader(knowledge_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro') #Check model availability.
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
    except NotFound as e:
        st.error(f"Error: Gemini 1.5 Pro model not found. Please verify that the Generative Language API is enabled in your Google Cloud project, and that the model name is correct. Error details: {e}")
        return None
    except PermissionDenied as e:
        st.error(f"Error: Permission denied. Please check your API key and service account permissions. Error details: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}. Please check your API key and Google Cloud project setup. Error details: {e}")
        return None

    if llm is None:
        return None

    retriever = DummyRetriever(texts=texts)

    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs={"prompt": PROMPT})
    return qa

class DummyRetriever(BaseRetriever):
    texts: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        relevant_docs = []
        for doc in self.texts:
            if query.lower() in doc.page_content.lower():
                relevant_docs.append(doc)
        return relevant_docs

def run_streamlit_app(knowledge_file):
    st.title("Custom Knowledge Base Chatbot (Gemini 1.5 Pro)")

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = create_chatbot(knowledge_file)

    query = st.text_input("Enter your question:")

    if st.button("Submit"):
        if query:
            if st.session_state.chatbot is not None:
                result = st.session_state.chatbot({"query": query})
                st.write("Answer:", result["result"])
            else:
                st.error("Chatbot initialization failed. Check error messages above.")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    knowledge_file = "knowledge_base.txt"
    run_streamlit_app(knowledge_file)