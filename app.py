import os
import streamlit as st
from loader import load_website
from embedding import embeddings
from splitter import split_documents
from vector_store import create_vectorstore
from chatbot import ChatGroq
from retriever import get_retriever
from langchain_classic.chains import retrieval_qa

st.title("website chatbot")

if "qa.chain" not in st.session_state:
    st.session_state.qa_chain = None

website_url = st.text_input("Enter Website URL")

if st.button("Load Website"):
    if website_url:
        with st.spinner("loading and processing..."):
            docs = load_website(website_url)
            chunks = split_documents(docs)
            vector_store = create_vectorstore(chunks, embeddings) 
            retriever = get_retriever(vector_store)

            st.session_state.qa_chain = retrieval_qa.from_chain_type(
                model = ChatGroq,
                chain_type = "stuff",
                retriever = get_retriever
            )
            st.success("Website Loaded Successfully")
    else:
            st.error("Please Enter Valid URL")

question = st.text_input("Ask Question")

if st.button("Ask"):
    if st.session_state.qa_chain is None:
        st.error("Please load a website first")
    elif not question:
        st.error("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(
                {"query" : question}
            )
            st.write(response["result"])
