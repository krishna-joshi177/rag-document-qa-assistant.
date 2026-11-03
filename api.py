"""REST API:  uvicorn api:app --reload   (docs at http://localhost:8000/docs)"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag import RAGAssistant, Settings
from rag.ingest import load_bytes
from rag.pipeline import source_label

app = FastAPI(title="RAG Document Q&A API")
assistant = RAGAssistant(Settings())
assistant.load()


class Question(BaseModel):
    question: str
    reset_history: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "index_ready": assistant.ready}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    total = 0
    for f in files:
        try:
            total += assistant.add_documents(load_bytes(await f.read(), f.filename))
        except ValueError as e:
            raise HTTPException(400, str(e))
    assistant.save()
    return {"files": [f.filename for f in files], "chunks_indexed": total}


@app.post("/ask")
def ask(q: Question):
    if not assistant.ready:
        raise HTTPException(400, "No documents indexed. POST /upload first.")
    if q.reset_history:
        assistant.clear_history()
    a = assistant.ask(q.question)
    return {
        "answer": a.answer,
        "search_query": a.search_query,
        "sources": [{"id": i, "source": source_label(d), "text": d.page_content}
                    for i, d in enumerate(a.sources, 1)],
    }


@app.delete("/index")
def clear_index():
    assistant.reset(delete_disk=True)
    return {"status": "cleared"}
