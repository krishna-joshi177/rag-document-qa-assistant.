"""Document loading + chunking. Supports PDF, DOCX, TXT, MD, CSV."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}


def _load_pdf(data: bytes, name: str) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": name, "page": i}))
    return docs


def _load_docx(data: bytes, name: str) -> list[Document]:
    import docx

    d = docx.Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:  # keep table content too
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    text = "\n".join(parts).strip()
    return [Document(page_content=text, metadata={"source": name})] if text else []


def _load_text(data: bytes, name: str) -> list[Document]:
    text = data.decode("utf-8", errors="ignore").strip()
    return [Document(page_content=text, metadata={"source": name})] if text else []


def _load_csv(data: bytes, name: str) -> list[Document]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8", errors="ignore")))
    docs = []
    for i, row in enumerate(reader, start=1):
        text = "\n".join(f"{k}: {v}" for k, v in row.items() if k)
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": name, "row": i}))
    return docs


_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_text,
    ".md": _load_text,
    ".csv": _load_csv,
}


def load_bytes(data: bytes, filename: str) -> list[Document]:
    """Load a single file given its raw bytes (used for web uploads)."""
    ext = Path(filename).suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return _LOADERS[ext](data, Path(filename).name)


def load_paths(paths: Iterable[str | Path]) -> list[Document]:
    """Load files or every supported file inside given folders."""
    docs: list[Document] = []
    for p in map(Path, paths):
        files = [f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS] if p.is_dir() else [p]
        for f in files:
            docs.extend(load_bytes(f.read_bytes(), f.name))
    return docs


def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks
