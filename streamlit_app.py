import streamlit as st
import PyPDF2
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# Load QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
llm = HuggingFacePipeline(pipeline=qa_pipeline)
chain = load_qa_chain(llm=llm, chain_type="stuff")

st.title("PDF Question Answering App")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    reader = PyPDF2.PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    st.success("PDF uploaded and text extracted!")

    question = st.text_input("Enter your question:")
    if question:
        docs = [Document(page_content=full_text)]
        answer = chain.run(input_documents=docs, question=question)
        st.write("**Answer:**", answer)
      
