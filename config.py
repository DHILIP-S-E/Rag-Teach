"""
Configuration management for the RAG system.
Loads settings from .env and provides typed access to all config values.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "rag_documents")

    # Embedding
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # LLM (Google Gemini)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.1-pro-preview")

    # Chunking
    CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 64  # tokens

    # Retrieval
    TOP_K: int = 5  # number of chunks to retrieve
    RERANK_TOP_K: int = 3  # after re-ranking, keep this many

    # Embedding cache directory
    CACHE_DIR: str = os.path.join(os.path.dirname(__file__), "embedding_cache")

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing/invalid configuration issues."""
        issues = []
        if not cls.QDRANT_URL:
            issues.append("QDRANT_URL is not set")
        if not cls.QDRANT_API_KEY:
            issues.append("QDRANT_API_KEY is not set")
        if not cls.GEMINI_API_KEY:
            issues.append("GEMINI_API_KEY is not set (needed for LLM generation)")
        return issues
