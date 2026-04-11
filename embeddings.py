"""
Embedding module with provider switching and disk caching.

Supports two providers:
- HuggingFace (sentence-transformers): runs locally, free, no API key needed
- Gemini: Google's embedding model, requires Gemini API key

Embeddings are cached to disk keyed by (model + text hash) so repeated
ingestion of the same documents doesn't recompute embeddings.
"""

import numpy as np
from config import Config
from utils import compute_text_hash, load_json_cache, save_json_cache


class EmbeddingModel:
    """Unified interface for generating embeddings regardless of provider."""

    def __init__(
        self,
        provider: str | None = None,
        model_name: str | None = None,
    ):
        self.provider = provider or Config.EMBEDDING_PROVIDER
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self._model = None  # lazy-loaded
        self._dimension: int | None = None

    def _load_model(self):
        """Lazy-load the underlying model on first use."""
        if self._model is not None:
            return

        if self.provider == "huggingface":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            # Try new API first, fall back to old one
            if hasattr(self._model, "get_embedding_dimension"):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()

        elif self.provider == "gemini":
            from google import genai
            self._model = genai.Client(api_key=Config.GEMINI_API_KEY)
            # Gemini text-embedding-004 produces 768-dim vectors
            self._dimension = 768
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    @property
    def dimension(self) -> int:
        self._load_model()
        return self._dimension

    def embed_texts(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Returns a numpy array of shape (len(texts), dimension).
        Uses disk cache to avoid recomputing known embeddings.
        """
        self._load_model()

        results = [None] * len(texts)
        texts_to_compute: list[tuple[int, str]] = []  # (original_index, text)

        # Check cache first
        if use_cache:
            for i, text in enumerate(texts):
                cache_key = f"emb_{self.provider}_{self.model_name}_{compute_text_hash(text)}"
                cached = load_json_cache(cache_key)
                if cached is not None:
                    results[i] = cached["embedding"]
                else:
                    texts_to_compute.append((i, text))
        else:
            texts_to_compute = list(enumerate(texts))

        # Compute missing embeddings in batch
        if texts_to_compute:
            indices, batch_texts = zip(*texts_to_compute)
            batch_texts = list(batch_texts)

            if self.provider == "huggingface":
                embeddings = self._model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                embeddings = embeddings.tolist()

            elif self.provider == "gemini":
                # New google-genai SDK: client.models.embed_content
                response = self._model.models.embed_content(
                    model="text-embedding-004",
                    contents=batch_texts,
                )
                embeddings = [e.values for e in response.embeddings]

            # Store computed embeddings in results and cache
            for idx, emb, text in zip(indices, embeddings, batch_texts):
                results[idx] = emb
                if use_cache:
                    cache_key = f"emb_{self.provider}_{self.model_name}_{compute_text_hash(text)}"
                    save_json_cache(cache_key, {"embedding": emb})

        return np.array(results, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (dimension,)."""
        return self.embed_texts([query], use_cache=False)[0]
