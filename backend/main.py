"""
Streamlit UI for AI-Powered Document Q&A System
Handles:
- PDF upload
- Knowledge base creation
- Question answering
"""

import os
import tempfile
import streamlit as st

from rag_pipeline import build_vector_store, answer_question

st.set_page_config(page_title="AI Document Q&A (RAG)", layout="wide")

st.title("📄 AI-Powered Document Q&A System")
st.write("Upload PDFs and ask questions using Retrieval-Augmented Generation (RAG)")

# Session state to track vector store
if "kb_built" not in st.session_state:
    st.session_state.kb_built = False

# ---- PDF Upload ----
st.header("1️⃣ Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

pdf_paths = []

if uploaded_files:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            pdf_paths.append(tmp.name)

# ---- Build Knowledge Base ----
st.header("2️⃣ Build Knowledge Base")

if st.button("Build Knowledge Base"):
    if not pdf_paths:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Processing PDFs and building vector store..."):
            try:
                build_vector_store(pdf_paths)
                st.session_state.kb_built = True
                st.success("Knowledge base built successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

# ---- Question Answering ----
st.header("3️⃣ Ask a Question")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not st.session_state.kb_built:
        st.warning("Please build the knowledge base first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = answer_question(question)
                st.subheader("✅ Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
