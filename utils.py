"""
Shared utilities for the RAG pipeline.
"""

import hashlib
import json
import os
from config import Config


def compute_text_hash(text: str) -> str:
    """Stable hash for a text string, used for deduplication and cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_cache_dir() -> str:
    """Create the embedding cache directory if it doesn't exist."""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    return Config.CACHE_DIR


def load_json_cache(cache_key: str) -> dict | None:
    """Load a cached JSON object by key. Returns None on miss."""
    cache_dir = ensure_cache_dir()
    path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_json_cache(cache_key: str, data: dict) -> None:
    """Persist a JSON-serializable object to disk cache."""
    cache_dir = ensure_cache_dir()
    path = os.path.join(cache_dir, f"{cache_key}.json")
    with open(path, "w") as f:
        json.dump(data, f)
