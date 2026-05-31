"""Qdrant vector store for email embeddings."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from config.settings import config


class EmailVectorStore:
    def __init__(self):
        self.client = QdrantClient(host=config.qdrant.host, port=config.qdrant.port)
        self.collection = config.qdrant.collection
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=config.models.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """Store email chunks with their embeddings."""
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            points.append(PointStruct(
                id=chunk["id"],
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "message_id": chunk["message_id"],
                    "subject": chunk["subject"],
                    "sender": chunk["sender"],
                    "date": chunk["date"],
                    "category": chunk["category"],
                    "chunk_index": chunk["chunk_index"],
                },
            ))

        # Upsert in batches of 100
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i : i + 100],
            )

    def search(self, query_embedding: list[float], limit: int = 5) -> list[dict]:
        """Search for similar email chunks."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=limit,
        )

        return [
            {
                "score": r.score,
                "text": r.payload.get("text", ""),
                "subject": r.payload.get("subject", ""),
                "sender": r.payload.get("sender", ""),
                "date": r.payload.get("date", ""),
                "category": r.payload.get("category", ""),
                "message_id": r.payload.get("message_id", ""),
            }
            for r in results.points
        ]

    def delete_by_message_id(self, message_id: str):
        """Delete all chunks for a given email (deduplication on re-run)."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="message_id",
                            match=MatchValue(value=message_id),
                        )
                    ]
                ),
            )
        except Exception:
            pass  # Collection might be empty or not exist yet

    def get_collection_info(self) -> dict:
        """Get collection statistics."""
        info = self.client.get_collection(self.collection)
        return {
            "vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }
