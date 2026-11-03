"""Command line:
   python cli.py ingest data/sample          # build index from files/folders
   python cli.py chat                        # interactive Q&A
"""
import argparse

from rag import RAGAssistant, Settings
from rag.ingest import load_paths
from rag.pipeline import source_label


def main():
    p = argparse.ArgumentParser(description="RAG Document Q&A Assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--reset", action="store_true", help="rebuild index from scratch")
    sub.add_parser("chat")
    args = p.parse_args()

    a = RAGAssistant(Settings())
    if args.cmd == "ingest":
        if not args.reset:
            a.load()
        n = a.add_documents(load_paths(args.paths))
        a.save()
        print(f"Indexed {n} chunks -> {a.s.index_dir}/")
        return

    if not a.load():
        raise SystemExit("No index found. Run: python cli.py ingest <files/folders>")
    print("Ask questions (type 'exit' to quit, 'reset' to clear history)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            a.clear_history(); print("History cleared.\n"); continue
        if not q:
            continue
        stream, docs = a.ask_stream(q)
        print("Assistant: ", end="", flush=True)
        for tok in stream:
            print(tok, end="", flush=True)
        print("\nSources:", "; ".join(f"[{i}] {source_label(d)}" for i, d in enumerate(docs, 1)), "\n")


if __name__ == "__main__":
    main()
