"""
Centralized configuration management using Pydantic BaseSettings.
Loads values from .env file with sensible defaults.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ── LLM Configuration ──
    openai_api_key: str = Field(default="", description="OpenAI API key")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model name")

    # ── Vector Store ──
    chroma_persist_dir: str = Field(
        default="./chroma_db", description="ChromaDB persistence directory"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model"
    )

    # ── Application ──
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Runtime environment")

    # ── RAG Settings ──
    chunk_size: int = Field(default=500, description="Text chunk size for RAG")
    chunk_overlap: int = Field(default=50, description="Chunk overlap for RAG")
    top_k_results: int = Field(
        default=5, description="Number of top results to retrieve"
    )

    # ── Paths ──
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def policies_dir(self) -> Path:
        return self.data_dir / "policies"

    @property
    def customers_file(self) -> Path:
        return self.data_dir / "customers.json"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()
