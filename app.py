"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from rag import RAGAssistant, Settings
from rag.ingest import SUPPORTED_EXTENSIONS, load_bytes
from rag.pipeline import source_label

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄", layout="wide")
st.title("📄 RAG Document Q&A Assistant")
st.caption("Azure OpenAI • LangChain • FAISS — answers grounded in your documents, with citations")


@st.cache_resource(show_spinner=False)
def get_assistant() -> RAGAssistant:
    a = RAGAssistant(Settings())
    a.load()  # reuse a previously saved index if present
    return a


try:
    assistant = get_assistant()
except Exception as e:  # missing / wrong Azure config
    st.error(f"Configuration error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Sidebar: upload & settings ----------------
with st.sidebar:
    st.header("1️⃣ Upload documents")
    files = st.file_uploader(
        "PDF, DOCX, TXT, MD, CSV",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if st.button("Build / update index", type="primary", disabled=not files, use_container_width=True):
        with st.spinner("Reading, chunking and embedding…"):
            total_chunks, errors = 0, []
            for f in files:
                try:
                    total_chunks += assistant.add_documents(load_bytes(f.getvalue(), f.name))
                except Exception as e:
                    errors.append(f"{f.name}: {e}")
            assistant.save()
        if total_chunks:
            st.success(f"Indexed {total_chunks} chunks from {len(files) - len(errors)} file(s).")
        for err in errors:
            st.error(err)

    st.header("2️⃣ Retrieval settings")
    assistant.s.top_k = st.slider("Chunks per answer (k)", 1, 10, assistant.s.top_k)
    assistant.s.fetch_k = st.slider("MMR candidate pool", 5, 50, assistant.s.fetch_k)
    show_sources = st.toggle("Show retrieved sources", value=True)

    st.divider()
    st.write("Index status:", "✅ Ready" if assistant.ready else "⏳ Empty")
    c1, c2 = st.columns(2)
    if c1.button("New chat", use_container_width=True):
        assistant.clear_history()
        st.session_state.messages = []
        st.rerun()
    if c2.button("Clear index", use_container_width=True):
        assistant.reset(delete_disk=True)
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources") and show_sources:
            with st.expander("Sources"):
                for i, (label, text) in enumerate(m["sources"], 1):
                    st.markdown(f"**[{i}] {label}**")
                    st.caption(text[:500] + ("…" if len(text) > 500 else ""))

if question := st.chat_input("Ask a question about your documents…", disabled=not assistant.ready):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            stream, docs = assistant.ask_stream(question)
            answer = st.write_stream(stream)
            sources = [(source_label(d), d.page_content) for d in docs]
            if sources and show_sources:
                with st.expander("Sources"):
                    for i, (label, text) in enumerate(sources, 1):
                        st.markdown(f"**[{i}] {label}**")
                        st.caption(text[:500] + ("…" if len(text) > 500 else ""))
        except Exception as e:
            answer, sources = f"⚠️ Error: {e}", []
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

if not assistant.ready:
    st.info("👈 Upload documents and click **Build / update index** to start.")
