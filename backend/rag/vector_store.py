import os
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from backend.config import CHROMA_PERSIST_DIR

def process_pdf(file_path: str, embeddings) -> Chroma:
    reader = PdfReader(file_path)
    docs = [
        Document(page_content=page.extract_text(), metadata={"page": i+1})
        for i, page in enumerate(reader.pages)
        if page.extract_text().strip()
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    persist_dir = Path(CHROMA_PERSIST_DIR)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="pdf_rag"
    )

def get_vectorstore(embeddings) -> Chroma:
    persist_dir = Path(CHROMA_PERSIST_DIR)
    if persist_dir.exists() and (persist_dir / "chroma.sqlite3").exists():
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name="pdf_rag"
        )
    return None