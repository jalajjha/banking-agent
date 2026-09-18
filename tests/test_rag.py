"""
Unit tests for the RAG pipeline — document loading, chunking, and vector store.
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.rag.document_loader import load_policy_documents
from src.rag.chunker import chunk_documents


# ════════════════════════════════════════════════════════════════
# DOCUMENT LOADER TESTS
# ════════════════════════════════════════════════════════════════


class TestDocumentLoader:
    """Tests for the policy document loader."""

    def test_loads_policy_documents(self):
        """Test that policy documents are loaded from the data directory."""
        docs = load_policy_documents()
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_documents_have_metadata(self):
        """Test that loaded documents include source metadata."""
        docs = load_policy_documents()
        for doc in docs:
            assert "source" in doc.metadata
            assert "source_path" in doc.metadata
            assert doc.metadata["document_type"] == "policy"

    def test_documents_have_content(self):
        """Test that loaded documents contain text content."""
        docs = load_policy_documents()
        for doc in docs:
            assert len(doc.page_content) > 0

    def test_loads_expected_policy_files(self):
        """Test that all expected policy files are loaded."""
        docs = load_policy_documents()
        sources = {doc.metadata["source"] for doc in docs}
        expected_files = {
            "credit_card_policy.txt",
            "loan_policy.txt",
            "kyc_aml_policy.txt",
            "dispute_resolution_policy.txt",
            "account_closure_policy.txt",
        }
        assert expected_files.issubset(sources)

    def test_raises_for_missing_directory(self):
        """Test that FileNotFoundError is raised for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            load_policy_documents(Path("/nonexistent/path"))


# ════════════════════════════════════════════════════════════════
# CHUNKER TESTS
# ════════════════════════════════════════════════════════════════


class TestChunker:
    """Tests for the document chunker."""

    def _make_doc(self, content: str, source: str = "test.txt") -> Document:
        return Document(
            page_content=content,
            metadata={"source": source, "document_type": "policy"},
        )

    def test_chunks_single_document(self):
        """Test that a single document is chunked into multiple pieces."""
        long_text = "This is a test sentence. " * 100  # ~2500 chars
        doc = self._make_doc(long_text)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_short_document_stays_whole(self):
        """Test that a short document stays as one chunk."""
        short_text = "Short policy text."
        doc = self._make_doc(short_text)
        chunks = chunk_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1

    def test_chunks_have_metadata(self):
        """Test that chunks inherit and extend metadata."""
        text = "Sample text. " * 50
        doc = self._make_doc(text, source="sample_policy.txt")
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "sample_policy.txt"
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "total_chunks" in chunk.metadata

    def test_chunk_ids_are_unique(self):
        """Test that all chunk IDs are unique within a document."""
        text = "Policy information. " * 100
        doc = self._make_doc(text)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_multiple_documents_chunked(self):
        """Test chunking multiple documents produces chunks from all."""
        docs = [
            self._make_doc("Document one content. " * 50, "doc1.txt"),
            self._make_doc("Document two content. " * 50, "doc2.txt"),
        ]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        sources = {c.metadata["source"] for c in chunks}
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources

    def test_empty_input_returns_empty(self):
        """Test that empty input returns empty chunks list."""
        chunks = chunk_documents([])
        assert chunks == []
