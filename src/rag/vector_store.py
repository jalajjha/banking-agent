"""
ChromaDB vector store wrapper for RAG pipeline.
Handles document ingestion (embedding + storage) and similarity search
with retry logic and graceful fallbacks.
"""

from dataclasses import dataclass
from typing import Any

import chromadb
from langchain_core.documents import Document
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.rag.document_loader import load_policy_documents
from src.rag.chunker import chunk_documents
from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "cred_policies"


@dataclass
class SearchResult:
    """A single vector search result with content, metadata, and relevance score."""

    content: str
    metadata: dict[str, Any]
    relevance_score: float
    chunk_id: str


class VectorStore:
    """ChromaDB-based vector store with retry logic and graceful fallbacks."""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._persist_dir = persist_directory or settings.chroma_persist_dir
        self._collection_name = collection_name
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None

    def _get_client(self) -> chromadb.ClientAPI:
        """Initialize or return the ChromaDB client."""
        if self._client is None:
            try:
                # ChromaDB v1.x API
                self._client = chromadb.PersistentClient(
                    path=self._persist_dir,
                )
            except (TypeError, AttributeError):
                # Fallback for older ChromaDB versions
                self._client = chromadb.Client()
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        """Get or create the ChromaDB collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def ingest(self, chunks: list[Document] | None = None) -> int:
        """
        Ingest document chunks into the vector store.

        If no chunks are provided, loads and chunks policy documents
        from the default policies directory.

        Args:
            chunks: Pre-chunked Document objects. If None, loads from policies dir.

        Returns:
            Number of chunks ingested.
        """
        if chunks is None:
            logger.info("loading_and_chunking_policies")
            raw_docs = load_policy_documents()
            chunks = chunk_documents(raw_docs)

        if not chunks:
            logger.warning("no_chunks_to_ingest")
            return 0

        collection = self._get_collection()

        # Prepare data for ChromaDB
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get("chunk_id", f"chunk_{len(ids)}")
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            # ChromaDB requires metadata values to be str, int, float, or bool
            clean_meta = {
                k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                for k, v in chunk.metadata.items()
            }
            metadatas.append(clean_meta)

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("ingestion_complete", chunks_ingested=len(ids))
        return len(ids)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def search(
        self,
        query: str,
        k: int | None = None,
        min_relevance: float = 0.3,
    ) -> list[SearchResult]:
        """
        Search the vector store for relevant document chunks.

        Args:
            query: The search query string.
            k: Number of top results to return. Defaults to settings.top_k_results.
            min_relevance: Minimum relevance score threshold (0–1).
                          Results below this are filtered out.

        Returns:
            List of SearchResult objects sorted by relevance (highest first).
            Returns empty list if retrieval fails after retries.
        """
        top_k = k or settings.top_k_results
        collection = self._get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []

        if not results["documents"] or not results["documents"][0]:
            logger.warning("no_search_results", query=query[:100])
            return search_results

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to relevance score: 1 - (distance / 2)
            relevance = 1.0 - (distance / 2.0)

            if relevance >= min_relevance:
                search_results.append(
                    SearchResult(
                        content=doc,
                        metadata=meta,
                        relevance_score=round(relevance, 4),
                        chunk_id=meta.get("chunk_id", "unknown"),
                    )
                )

        # Sort by relevance (highest first)
        search_results.sort(key=lambda r: r.relevance_score, reverse=True)

        logger.info(
            "search_complete",
            query=query[:100],
            results_found=len(search_results),
            top_score=(
                search_results[0].relevance_score if search_results else 0
            ),
        )

        return search_results

    def get_fallback_response(self) -> str:
        """Return a safe fallback message when retrieval fails."""
        return (
            "I don't have sufficient information in my policy documents "
            "to answer this safely. Please contact our support team at "
            "1800-XXX-XXXX or visit the nearest Cred branch for assistance."
        )


# Singleton instance
vector_store = VectorStore()


if __name__ == "__main__":
    """CLI entrypoint to ingest policy documents into the vector store."""
    from src.utils.logging_config import setup_logging

    setup_logging()
    logger.info("starting_policy_ingestion")
    count = vector_store.ingest()
    logger.info("ingestion_finished", total_chunks=count)
