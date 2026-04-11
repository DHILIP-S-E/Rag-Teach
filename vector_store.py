"""
Vector store module using Qdrant Cloud.

Handles:
- Collection creation with proper vector configuration
- Upserting document chunks with embeddings and metadata
- Similarity search (cosine)
- Collection management (list, delete, check existence)

Qdrant was chosen over FAISS because it provides:
- Built-in persistence (cloud-hosted)
- Metadata filtering
- Payload storage alongside vectors
- Production-ready scaling
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from config import Config
from ingestion import Document


class VectorStore:
    """Interface to the Qdrant vector database."""

    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or Config.QDRANT_COLLECTION
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
        )

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if it doesn't already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> int:
        """
        Insert or update document chunks with their embeddings.
        Returns the number of points upserted.
        """
        points = []
        for doc, embedding in zip(documents, embeddings):
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": doc.text,
                    "source": doc.metadata.get("source", ""),
                    "page": doc.metadata.get("page", 0),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                },
            ))

        # Qdrant supports batch upsert; send in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        return len(points)

    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[dict]:
        """
        Perform cosine similarity search.
        Returns list of dicts with keys: text, source, page, chunk_index, score.
        """
        top_k = top_k or Config.TOP_K

        # Optional: filter by source document
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("source", ""),
                "page": hit.payload.get("page", 0),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "score": hit.score,
            }
            for hit in results.points
        ]

    def get_collection_info(self) -> dict | None:
        """Return collection stats, or None if it doesn't exist."""
        try:
            # Use count() — it's always reliable, unlike info.points_count
            # which can be None for newly-created collections
            count_result = self.client.count(
                collection_name=self.collection_name,
                exact=True,
            )
            return {
                "name": self.collection_name,
                "points_count": count_result.count,
            }
        except Exception:
            return None

    def delete_collection(self) -> bool:
        """Delete the entire collection. Use with caution."""
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception:
            return False
