# RAG-based Document Q&A Assistant
**Python · Azure OpenAI · LangChain · FAISS · Streamlit · FastAPI**

A GenAI assistant that retrieves context from uploaded documents and generates grounded, cited answers
using Azure OpenAI, LangChain and Retrieval-Augmented Generation (RAG).

## Features
- Upload **PDF, DOCX, TXT, MD, CSV** (page/row-level metadata kept for citations)
- Recursive chunking with overlap → Azure OpenAI embeddings → **FAISS** vector index (saved to disk)
- **MMR retrieval** for relevant *and* diverse chunks
- **Prompt engineering:** follow-up questions rewritten into standalone queries, strict grounding,
  inline `[n]` citations, explicit "not found" fallback, conflict handling
- Multi-turn chat memory, streaming answers, source viewer
- 3 interfaces: **Streamlit UI**, **REST API**, **CLI** + offline unit tests

## Architecture
```
Upload → Load (pypdf/python-docx) → Split → Embed (Azure OpenAI) → FAISS
Question → Condense w/ history (LLM) → MMR retrieve top-k → Grounded prompt → Azure GPT → Answer + sources
```

## Setup
1. In Azure OpenAI / AI Foundry, deploy a **chat model** (e.g. gpt-4o-mini) and an **embedding model** (e.g. text-embedding-3-small).
2. Install & configure:
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # Windows: copy .env.example .env
# edit .env -> endpoint, key, deployment names
```

## Run
```bash
streamlit run app.py                     # Web UI  -> http://localhost:8501
uvicorn api:app --reload                 # REST API -> http://localhost:8000/docs
python cli.py ingest data/sample         # CLI: index files/folders
python cli.py chat                       # CLI: chat
pytest -q                                # offline tests, no Azure needed
```

### API example
```bash
curl -F "files=@data/sample/company_policy.txt" http://localhost:8000/upload
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question":"What is the certification reimbursement process?"}'
```

## Tuning (.env)
| Variable | Default | Purpose |
|---|---|---|
| CHUNK_SIZE / CHUNK_OVERLAP | 1000 / 150 | Chunk granularity |
| TOP_K | 4 | Chunks sent to the LLM |
| FETCH_K | 20 | MMR candidate pool |
| TEMPERATURE | 0.1 | Leave empty for reasoning models (o-series) that reject temperature |

## Project structure
```
rag/config.py     settings from .env
rag/ingest.py     loaders + chunking
rag/prompts.py    condense + grounded-answer prompts
rag/pipeline.py   RAGAssistant (index, retrieve, answer, stream, memory)
app.py / api.py / cli.py   interfaces
tests/            offline tests with fake LLM/embeddings
```
