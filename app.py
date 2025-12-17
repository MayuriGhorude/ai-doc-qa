import tempfile
import streamlit as st

from rag_pipeline import (
    build_vector_store,
    answer_question_with_sources,
    extract_structured_resume_info
)

st.set_page_config(page_title="AI Document Q&A (RAG)", layout="wide")

st.title("📄 AI-Powered Document Q&A System")
st.write("Upload PDFs and ask questions using Retrieval-Augmented Generation (RAG)")

# -------------------------------
# Session State
# -------------------------------
if "kb_built" not in st.session_state:
    st.session_state.kb_built = False

# -------------------------------
# PDF Upload
# -------------------------------
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

# -------------------------------
# Build Knowledge Base
# -------------------------------
st.header("2️⃣ Build Knowledge Base")

if st.button("Build Knowledge Base"):
    if not pdf_paths:
        st.error("Please upload at least one PDF.")
    else:
        with st.spinner("Building vector store..."):
            build_vector_store(pdf_paths)
            st.session_state.kb_built = True
            st.success("✅ Knowledge base built successfully!")

# -------------------------------
# Question Answering
# -------------------------------
st.header("3️⃣ Ask a Question")

question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if not st.session_state.kb_built:
        st.warning("Please build the knowledge base first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer, sources = answer_question_with_sources(question)

            st.subheader("✅ Answer")
            st.write(answer)

            st.subheader("📌 Sources")
            for i, doc in enumerate(sources, start=1):
                page = doc.metadata.get("page", "N/A")
                content = doc.page_content[:300].replace("\n", " ")
                st.markdown(
                    f"**Source {i} (Page {page + 1}):** {content}..."
                )

# -------------------------------
# Resume Structured Extraction
# -------------------------------
st.header("4️⃣ Extract Resume Info")

if st.button("Extract Resume Info"):
    if not st.session_state.kb_built:
        st.warning("Please build the knowledge base first.")
    else:
        with st.spinner("Extracting resume info..."):
            output = extract_structured_resume_info()
            st.text(output)
