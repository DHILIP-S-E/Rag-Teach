"""
Retriever module with hybrid search and re-ranking.

Implements two retrieval strategies:
1. Vector search (semantic similarity via Qdrant)
2. BM25 keyword search (term-frequency based)

Hybrid search combines both to handle cases where:
- Vector search misses exact keyword matches (e.g. acronyms, IDs)
- Keyword search misses semantic paraphrases

Re-ranking uses a lightweight cross-encoder-style scoring to
surface the most relevant chunks from the combined candidate set.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from embeddings import EmbeddingModel
from vector_store import VectorStore
from config import Config


class Retriever:
    """Hybrid retriever combining vector search with BM25 keyword matching."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def vector_search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Pure vector similarity search."""
        top_k = top_k or Config.TOP_K
        query_embedding = self.embedding_model.embed_query(query)
        return self.vector_store.search(
            query_vector=query_embedding.tolist(),
            top_k=top_k,
        )

    def bm25_rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-rank candidates using BM25 keyword relevance.
        This catches cases where vector search returned semantically
        similar but keyword-irrelevant results, and boosts candidates
        that contain the exact query terms.
        """
        top_k = top_k or Config.RERANK_TOP_K

        if not candidates:
            return []

        # Tokenize candidates for BM25
        corpus = [doc["text"].lower().split() for doc in candidates]
        bm25 = BM25Okapi(corpus)

        # Score each candidate against the query
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Combine vector score (already in candidates) with BM25 score.
        # Normalize both to [0,1] range, then weight: 0.6 vector + 0.4 keyword
        vector_scores = np.array([c.get("score", 0.0) for c in candidates])
        vs_min, vs_max = vector_scores.min(), vector_scores.max()
        if vs_max > vs_min:
            vector_scores_norm = (vector_scores - vs_min) / (vs_max - vs_min)
        else:
            vector_scores_norm = np.ones_like(vector_scores)

        bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
        if bm25_max > bm25_min:
            bm25_scores_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_scores_norm = np.zeros_like(bm25_scores)

        combined = 0.6 * vector_scores_norm + 0.4 * bm25_scores_norm

        # Sort by combined score descending
        ranked_indices = np.argsort(combined)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            result = candidates[idx].copy()
            result["combined_score"] = float(combined[idx])
            result["bm25_score"] = float(bm25_scores[idx])
            results.append(result)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_reranking: bool = True,
    ) -> list[dict]:
        """
        Full retrieval pipeline:
        1. Fetch top_k * 2 candidates via vector search (over-fetch for re-ranking)
        2. Re-rank with BM25 hybrid scoring
        3. Return top_k final results
        """
        final_k = top_k or Config.RERANK_TOP_K

        if use_reranking:
            # Over-fetch candidates for re-ranking pool
            candidates = self.vector_search(query, top_k=final_k * 3)
            return self.bm25_rerank(query, candidates, top_k=final_k)
        else:
            return self.vector_search(query, top_k=final_k)
