"""Core RAG pipeline: index -> (condense) -> retrieve (MMR) -> grounded answer with citations."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from .config import Settings
from .ingest import split_documents
from .prompts import answer_prompt, condense_prompt

NOT_FOUND = "I couldn't find this in the uploaded documents."
MAX_HISTORY_TURNS = 6  # keep prompt small


def build_azure_models(s: Settings) -> tuple[BaseChatModel, Embeddings]:
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

    s.validate()
    common = dict(azure_endpoint=s.azure_endpoint, api_key=s.azure_api_key, api_version=s.api_version)
    llm_kwargs = dict(azure_deployment=s.chat_deployment, max_retries=3, **common)
    if s.temperature is not None:
        llm_kwargs["temperature"] = s.temperature
    llm = AzureChatOpenAI(**llm_kwargs)
    emb = AzureOpenAIEmbeddings(azure_deployment=s.embedding_deployment, **common)
    return llm, emb


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        blocks.append(f"[{i}] (source: {source_label(d)})\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def source_label(d: Document) -> str:
    m = d.metadata
    label = m.get("source", "unknown")
    if "page" in m:
        label += f", page {m['page']}"
    if "row" in m:
        label += f", row {m['row']}"
    return label


@dataclass
class Answer:
    question: str
    search_query: str
    answer: str
    sources: list[Document] = field(default_factory=list)


class RAGAssistant:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ):
        self.s = settings or Settings()
        if llm is None or embeddings is None:
            az_llm, az_emb = build_azure_models(self.s)
            llm, embeddings = llm or az_llm, embeddings or az_emb
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore: FAISS | None = None
        self.history: list[BaseMessage] = []
        self._condense = condense_prompt | self.llm | StrOutputParser()
        self._answer = answer_prompt | self.llm | StrOutputParser()

    # ---------------- Indexing ----------------
    def add_documents(self, docs: list[Document]) -> int:
        chunks = split_documents(docs, self.s.chunk_size, self.s.chunk_overlap)
        if not chunks:
            return 0
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vectorstore.add_documents(chunks)
        return len(chunks)

    def save(self, path: str | None = None) -> None:
        if self.vectorstore:
            self.vectorstore.save_local(path or self.s.index_dir)

    def load(self, path: str | None = None) -> bool:
        p = Path(path or self.s.index_dir)
        if not (p / "index.faiss").exists():
            return False
        # Safe here: we only load indexes this app created itself.
        self.vectorstore = FAISS.load_local(str(p), self.embeddings, allow_dangerous_deserialization=True)
        return True

    def reset(self, delete_disk: bool = False) -> None:
        self.vectorstore = None
        self.history.clear()
        if delete_disk:
            shutil.rmtree(self.s.index_dir, ignore_errors=True)

    @property
    def ready(self) -> bool:
        return self.vectorstore is not None

    # ---------------- Retrieval ----------------
    def _standalone_query(self, question: str) -> str:
        if not self.history:
            return question
        q = self._condense.invoke({"question": question, "chat_history": self.history[-MAX_HISTORY_TURNS * 2:]})
        return q.strip() or question

    def retrieve(self, query: str) -> list[Document]:
        if not self.vectorstore:
            raise RuntimeError("No documents indexed yet. Upload documents first.")
        # MMR = relevant AND diverse chunks (avoids near-duplicate context)
        k = self.s.top_k
        fetch_k = max(self.s.fetch_k, k)
        return self.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

    def _prepare(self, question: str) -> tuple[str, list[Document], dict]:
        query = self._standalone_query(question)
        docs = self.retrieve(query)
        inputs = {
            "question": question,
            "context": format_context(docs) if docs else "(no relevant context found)",
            "chat_history": self.history[-MAX_HISTORY_TURNS * 2:],
        }
        return query, docs, inputs

    def _remember(self, question: str, answer: str) -> None:
        self.history.extend([HumanMessage(question), AIMessage(answer)])

    # ---------------- Q&A ----------------
    def ask(self, question: str) -> Answer:
        query, docs, inputs = self._prepare(question)
        answer = NOT_FOUND if not docs else self._answer.invoke(inputs).strip()
        self._remember(question, answer)
        return Answer(question, query, answer, docs)

    def ask_stream(self, question: str) -> tuple[Iterator[str], list[Document]]:
        """Returns (token stream, sources). History is updated once the stream finishes."""
        query, docs, inputs = self._prepare(question)

        def gen() -> Iterator[str]:
            if not docs:
                self._remember(question, NOT_FOUND)
                yield NOT_FOUND
                return
            parts = []
            for tok in self._answer.stream(inputs):
                parts.append(tok)
                yield tok
            self._remember(question, "".join(parts).strip())

        return gen(), docs

    def clear_history(self) -> None:
        self.history.clear()
