# embedding_store.py
# ─────────────────────────────────────────────────────────────
# Handles: Embedding generation + ChromaDB persistence
# ─────────────────────────────────────────────────────────────

import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from document_processor import DocumentChunk
import config

logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────

class RetrievedChunk:
    """A chunk returned from a similarity search, with its relevance score."""
    def __init__(self, content: str, metadata: dict, score: float):
        self.content  = content
        self.metadata = metadata
        self.score    = score      # cosine similarity (0–1, higher = more relevant)

    def __repr__(self):
        return (
            f"RetrievedChunk(score={self.score:.3f}, "
            f"source={self.metadata.get('source','?')})"
        )


# ── Embedding Store ────────────────────────────────────────────

class EmbeddingStore:
    """
    Manages the ChromaDB vector store.

    Why ChromaDB?
    - Runs fully local (no API cost for vector storage)
    - Persists to disk between sessions
    - Built-in cosine similarity search
    - Native metadata filtering support

    Why sentence-transformers?
    - Free, runs on CPU
    - `all-MiniLM-L6-v2` is fast (14ms/query) and accurate for semantic search
    - No OpenAI key required for embeddings
    """

    def __init__(
        self,
        persist_dir: str   = config.CHROMA_PERSIST_DIR,
        collection:  str   = config.CHROMA_COLLECTION,
        model_name:  str   = config.EMBEDDING_MODEL,
    ):
        logger.info(f"Loading embedding model: {model_name}")
        self.embed_model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(
            path     = persist_dir,
            settings = Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name     = collection,
            metadata = {"hnsw:space": "cosine"},   # cosine similarity metric
        )
        logger.info(
            f"ChromaDB ready | dir={persist_dir} | "
            f"collection={collection} | "
            f"docs_stored={self.collection.count()}"
        )

    # ── Ingestion ──────────────────────────────────────────────

    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector for a single string."""
        return self.embed_model.encode(text, normalize_embeddings=True).tolist()

    def add_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        """
        Embed and store chunks in ChromaDB.
        Uses batching to avoid memory spikes on large knowledge bases.
        Returns number of NEW chunks added (skips duplicates by chunk_id).
        """
        if not chunks:
            logger.warning("add_chunks called with empty list.")
            return 0

        existing_ids = set(self.collection.get()["ids"])
        new_chunks   = [c for c in chunks if c.chunk_id not in existing_ids]

        if not new_chunks:
            logger.info("All chunks already in store. Skipping.")
            return 0

        added = 0
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]
            texts      = [c.content     for c in batch]
            ids        = [c.chunk_id    for c in batch]
            metadatas  = [c.metadata    for c in batch]
            embeddings = self.embed_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

            self.collection.add(
                ids        = ids,
                documents  = texts,
                embeddings = embeddings,
                metadatas  = metadatas,
            )
            added += len(batch)
            logger.info(f"  Stored batch {i//batch_size + 1} → {len(batch)} chunks")

        logger.info(f"Ingestion complete. Added {added} new chunks.")
        return added

    # ── Retrieval ──────────────────────────────────────────────

    def search(
        self,
        query:       str,
        top_k:       int   = config.TOP_K_RESULTS,
        min_score:   float = config.SIMILARITY_THRESHOLD,
        filter_meta: Optional[Dict] = None,
    ) -> List[RetrievedChunk]:
        """
        Semantic similarity search.

        Args:
            query:       Natural language query
            top_k:       Max results to return
            min_score:   Minimum cosine similarity (0–1).
                         ChromaDB returns *distance* (lower = closer),
                         so we convert: score = 1 - distance.
            filter_meta: Optional ChromaDB where-clause e.g. {"source": "manual.pdf"}

        Returns:
            List of RetrievedChunk sorted by score (highest first).
        """
        if self.collection.count() == 0:
            logger.warning("Vector store is empty. Ingest documents first.")
            return []

        query_embedding = self.embed_text(query)

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results":        min(top_k, self.collection.count()),
            "include":          ["documents", "metadatas", "distances"],
        }
        if filter_meta:
            kwargs["where"] = filter_meta

        results = self.collection.query(**kwargs)

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = round(1.0 - dist, 4)    # convert distance → similarity
            if score >= min_score:
                retrieved.append(RetrievedChunk(
                    content  = doc,
                    metadata = meta,
                    score    = score,
                ))

        retrieved.sort(key=lambda x: x.score, reverse=True)
        logger.info(
            f"Search for '{query[:60]}...' → "
            f"{len(retrieved)}/{top_k} results above threshold"
        )
        return retrieved

    # ── Utilities ──────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete and recreate the collection (use carefully!)."""
        logger.warning("Resetting ChromaDB collection!")
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name     = self.collection.name,
            metadata = {"hnsw:space": "cosine"},
        )

    def get_sources(self) -> List[str]:
        """Return list of unique source filenames in the store."""
        data = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in data["metadatas"]}
        return sorted(sources)
