"""RAG-based Document Q&A Assistant (Azure OpenAI + LangChain)."""
from .config import Settings
from .pipeline import RAGAssistant

__all__ = ["Settings", "RAGAssistant"]
