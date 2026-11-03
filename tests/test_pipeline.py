"""Offline tests (no Azure calls) using LangChain fake models:  pytest -q"""
import io

import docx
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from rag import RAGAssistant, Settings
from rag.ingest import load_bytes, split_documents
from rag.pipeline import format_context

TEXT = b"Employees get 24 days of paid leave.\n\nHybrid staff work from office 3 days a week."


def make_assistant(tmp_path, responses):
    s = Settings(index_dir=str(tmp_path / "idx"), chunk_size=60, chunk_overlap=0, top_k=2, fetch_k=5)
    return RAGAssistant(s, llm=FakeListChatModel(responses=responses),
                        embeddings=DeterministicFakeEmbedding(size=64))


def test_loaders_txt_and_docx():
    assert "24 days" in load_bytes(TEXT, "p.txt")[0].page_content
    d = docx.Document(); d.add_paragraph("Reimbursement within 30 days")
    buf = io.BytesIO(); d.save(buf)
    assert "30 days" in load_bytes(buf.getvalue(), "p.docx")[0].page_content


def test_split_and_context_format():
    chunks = split_documents(load_bytes(TEXT, "p.txt"), chunk_size=60, chunk_overlap=0)
    assert len(chunks) >= 2 and all("chunk_id" in c.metadata for c in chunks)
    assert format_context(chunks).startswith("[1] (source: p.txt)")


def test_end_to_end_with_history_and_persistence(tmp_path):
    a = make_assistant(tmp_path, ["24 days [1]", "standalone query", "3 days [2]"])
    assert a.add_documents(load_bytes(TEXT, "p.txt")) > 0
    r1 = a.ask("How many leave days?")
    assert r1.answer == "24 days [1]" and r1.sources
    r2 = a.ask("and office days?")           # follow-up -> condensed query used
    assert r2.search_query == "standalone query"
    a.save()
    b = make_assistant(tmp_path, ["ok"])
    assert b.load() and b.ready
