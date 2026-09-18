"""
Document loader for RAG pipeline.
Loads policy .txt files from the data/policies/ directory and returns
LangChain Document objects with source metadata.
"""

from pathlib import Path

from langchain_core.documents import Document

from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_policy_documents(
    policies_dir: Path | None = None,
) -> list[Document]:
    """
    Load all .txt policy files from the specified directory.

    Args:
        policies_dir: Directory containing policy .txt files.
                      Defaults to settings.policies_dir.

    Returns:
        List of LangChain Document objects with metadata including
        source file name and document ID.

    Raises:
        FileNotFoundError: If the policies directory does not exist.
    """
    directory = policies_dir or settings.policies_dir

    if not directory.exists():
        raise FileNotFoundError(
            f"Policies directory not found: {directory}"
        )

    documents: list[Document] = []
    txt_files = sorted(directory.glob("*.txt"))

    if not txt_files:
        logger.warning("no_policy_files_found", directory=str(directory))
        return documents

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            doc = Document(
                page_content=content,
                metadata={
                    "source": file_path.name,
                    "source_path": str(file_path),
                    "document_type": "policy",
                },
            )
            documents.append(doc)
            logger.info(
                "document_loaded",
                source=file_path.name,
                characters=len(content),
            )
        except Exception as e:
            logger.error(
                "document_load_error",
                source=file_path.name,
                error=str(e),
            )

    logger.info("total_documents_loaded", count=len(documents))
    return documents
