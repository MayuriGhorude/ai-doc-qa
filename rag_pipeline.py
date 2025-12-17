"""
RAG Pipeline for AI-Powered Document Q&A System

Features:
- PDF ingestion
- Text chunking
- Embedding creation
- Vector store (ChromaDB)
- Clean Question Answering (RAG)
- Multi-turn chat memory (custom, stable)
- Deterministic Resume Extraction (ATS-style)

Fixes:
- UTF-8 / emoji issues (Windows)
- LangChain v1.x compatibility
- No repeated / mixed answers
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from transformers import pipeline

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# ==================================================
# GLOBAL OBJECTS
# ==================================================

vector_store = None

# Simple custom chat memory (LangChain-independent)
chat_history = []  # list of (question, answer)

# Load embeddings once
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load LLM once
hf_pipeline = pipeline(
    "text2text-generation",
    model=LLM_MODEL,
    max_length=200,
    do_sample=False,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# ==================================================
# UTILITIES
# ==================================================

def clean_text(text: str) -> str:
    """Remove problematic unicode characters (Windows safe)"""
    if not text:
        return ""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def format_docs(docs) -> str:
    """Convert Document objects to plain text"""
    return "\n\n".join(doc.page_content for doc in docs)

# ==================================================
# BUILD VECTOR STORE
# ==================================================

def build_vector_store(pdf_paths: List[str]):
    global vector_store, chat_history

    documents = []

    for pdf in pdf_paths:
        if not os.path.exists(pdf):
            raise FileNotFoundError(f"{pdf} not found")

        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())

    if not documents:
        raise ValueError("No text found in PDFs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    for doc in chunks:
        doc.page_content = clean_text(doc.page_content)

    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vector_store.persist()

    # Reset chat memory on new upload
    chat_history = []

    return True

# ==================================================
# QUESTION ANSWERING (RAG + CHAT MEMORY)
# ==================================================

QA_PROMPT = PromptTemplate.from_template(
    """Answer the question using the context and previous conversation.
If the answer is not found, say "I don't know".

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:"""
)

def answer_question_with_sources(question: str):
    global vector_store, chat_history

    if vector_store is None:
        raise RuntimeError("Vector store not built yet")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    context_text = format_docs(docs)

    # Keep last 3 interactions only
    history_text = ""
    for q, a in chat_history[-3:]:
        history_text += f"Q: {q}\nA: {a}\n"

    answer = llm.invoke(
        QA_PROMPT.format(
            context=context_text,
            chat_history=history_text,
            question=clean_text(question)
        )
    )

    # Save to memory
    chat_history.append((question, answer))

    return answer, docs

# ==================================================
# RESUME STRUCTURED EXTRACTION (ATS-STYLE, NO LLM)
# ==================================================

def extract_structured_resume_info():
    """
    Deterministic resume extraction using rules
    (No LLM hallucination)
    """
    global vector_store

    if vector_store is None:
        raise RuntimeError("Vector store not built yet")

    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke("skills education projects experience")

    text = format_docs(docs).lower()

    FRONTEND = ["html", "css", "javascript", "react", "bootstrap", "material-ui"]
    BACKEND = ["node", "php", "python", "java", "express"]
    DATABASE = ["mysql", "mongodb"]
    AI_ML = ["langchain", "rag", "llm"]

    skills = {
        "Frontend": sorted({s for s in FRONTEND if s in text}),
        "Backend": sorted({s for s in BACKEND if s in text}),
        "Database": sorted({s for s in DATABASE if s in text}),
        "AI/ML": sorted({s for s in AI_ML if s in text})
    }

    projects = []
    for line in text.splitlines():
        if "application" in line or "project" in line:
            projects.append(line.strip())

    output = "SKILLS:\n"
    for k, v in skills.items():
        if v:
            output += f"{k}: {', '.join(v)}\n"

    output += "\nPROJECTS:\n"
    for p in sorted(set(projects)):
        output += f"- {p[:120]}\n"

    return output
