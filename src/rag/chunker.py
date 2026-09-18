"""
Text chunker for RAG pipeline.
Splits documents into smaller, overlapping chunks for embedding and retrieval.
Preserves source metadata on every chunk.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split documents into smaller chunks using recursive character splitting.

    Args:
        documents: List of LangChain Document objects to chunk.
        chunk_size: Maximum characters per chunk. Defaults to settings.chunk_size.
        chunk_overlap: Overlap between consecutive chunks. Defaults to settings.chunk_overlap.

    Returns:
        List of chunked Document objects, each with inherited metadata
        plus a unique chunk_id.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []

    for doc in documents:
        doc_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            # Add chunk-level metadata for source attribution
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata.get('source', 'unknown')}::chunk_{idx}"
            )
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(doc_chunks)
            chunks.append(chunk)

        logger.info(
            "document_chunked",
            source=doc.metadata.get("source", "unknown"),
            num_chunks=len(doc_chunks),
        )

    logger.info(
        "chunking_complete",
        total_documents=len(documents),
        total_chunks=len(chunks),
        chunk_size=size,
        chunk_overlap=overlap,
    )

    return chunks
