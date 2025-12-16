"""
This file handles:
- PDF ingestion
- Text chunking
- Embedding creation
- Vector store (ChromaDB)
- Question answering using modern RAG (LCEL)
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


from transformers import pipeline

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

vector_store = None


def build_vector_store(pdf_paths: List[str]):
    """Load PDFs, split text, create embeddings and store in ChromaDB"""
    global vector_store

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

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DB_DIR
    )

    vector_store.persist()
    return True


def answer_question(question: str) -> str:
    """Answer question using RAG (Retriever + LLM)"""
    global vector_store

    if vector_store is None:
        raise RuntimeError("Vector store not built yet")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template(
        """Use the following context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""
    )

    hf_pipeline = pipeline(
        "text2text-generation",
        model=LLM_MODEL,
        max_length=512
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain.invoke(question)
