import os
import shutil
import gc
import time
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from backend.config import CHROMA_PERSIST_DIR

def _force_reset_chroma(dir_path: str):
    """Безопасно очищает папку ChromaDB, обходя блокировки файлов в Docker"""
    path = Path(dir_path)
    if not path.exists():
        return
        
    gc.collect()
    time.sleep(0.2)
    shutil.rmtree(path, ignore_errors=True)
    time.sleep(0.1)
  
    if path.exists():
        backup = Path(f"{dir_path}_tmp_{int(time.time())}")
        try:
            path.rename(backup)
            path.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(backup, ignore_errors=True)
        except Exception:
            new_path = path / f"session_{int(time.time())}"
            new_path.mkdir(parents=True, exist_ok=True)
            os.environ["CHROMA_PERSIST_DIR"] = str(new_path)
    else:
        path.mkdir(parents=True, exist_ok=True)

def process_pdf(file_path: str, embeddings) -> Chroma:
    reader = PdfReader(file_path)
    docs = [
        Document(page_content=page.extract_text(), metadata={"page": i + 1})
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

    _force_reset_chroma(CHROMA_PERSIST_DIR)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", CHROMA_PERSIST_DIR),
        collection_name="pdf_rag"
    )
