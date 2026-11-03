"""Central configuration loaded from environment variables / .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val


@dataclass
class Settings:
    azure_endpoint: str | None = field(default_factory=lambda: _get("AZURE_OPENAI_ENDPOINT"))
    azure_api_key: str | None = field(default_factory=lambda: _get("AZURE_OPENAI_API_KEY"))
    api_version: str = field(default_factory=lambda: _get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    chat_deployment: str | None = field(default_factory=lambda: _get("AZURE_OPENAI_CHAT_DEPLOYMENT"))
    embedding_deployment: str | None = field(default_factory=lambda: _get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))

    chunk_size: int = field(default_factory=lambda: int(_get("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(_get("CHUNK_OVERLAP", "150")))
    top_k: int = field(default_factory=lambda: int(_get("TOP_K", "4")))
    fetch_k: int = field(default_factory=lambda: int(_get("FETCH_K", "20")))
    # Set TEMPERATURE= (empty) for reasoning models that don't accept temperature.
    temperature: float | None = field(
        default_factory=lambda: float(_get("TEMPERATURE")) if _get("TEMPERATURE") else None
    )
    index_dir: str = field(default_factory=lambda: _get("INDEX_DIR", "vector_index"))

    def validate(self) -> None:
        missing = [
            name
            for name, val in {
                "AZURE_OPENAI_ENDPOINT": self.azure_endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_api_key,
                "AZURE_OPENAI_CHAT_DEPLOYMENT": self.chat_deployment,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.embedding_deployment,
            }.items()
            if not val
        ]
        if missing:
            raise ValueError(
                "Missing Azure OpenAI settings: " + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )
