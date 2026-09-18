"""
Policy search tool for CrewAI agents.
Queries the vector store to retrieve relevant policy document chunks
with source attribution.
"""

from crewai.tools import tool
from pydantic import BaseModel, Field

from src.rag.vector_store import vector_store, SearchResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PolicySearchInput(BaseModel):
    """Input schema for the policy search tool."""

    query: str = Field(
        ..., description="The policy-related question or search query"
    )
    num_results: int = Field(
        default=5,
        ge=3,
        le=5,
        description="Number of relevant policy chunks to retrieve (3–5)",
    )


class PolicyChunkResult(BaseModel):
    """A single retrieved policy chunk with source attribution."""

    content: str
    source: str
    chunk_id: str
    relevance_score: float


class PolicySearchOutput(BaseModel):
    """Structured output from a policy search."""

    query: str
    results: list[PolicyChunkResult]
    total_results: int
    has_relevant_results: bool
    fallback_message: str | None = None


@tool("search_policy_documents")
def search_policy_documents(query: str, num_results: int = 5) -> str:
    """
    Search Cred Financial Services policy documents for relevant information.

    Use this tool to find answers to questions about credit card policies,
    loan terms, KYC/AML requirements, dispute resolution procedures,
    and account closure rules.

    Args:
        query: The policy-related question to search for.
        num_results: Number of top results to retrieve (3–5). Default is 5.

    Returns:
        A formatted string with the relevant policy excerpts and their sources.
        If no relevant results are found, returns a safe fallback message.
    """
    try:
        results: list[SearchResult] = vector_store.search(
            query=query, k=num_results
        )

        if not results:
            fallback = vector_store.get_fallback_response()
            logger.warning(
                "policy_search_no_results",
                query=query[:100],
            )
            return fallback

        # Format results with source attribution
        formatted_parts: list[str] = []
        formatted_parts.append(f"Found {len(results)} relevant policy excerpts:\n")

        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "Unknown")
            chunk_id = result.chunk_id
            score = result.relevance_score

            formatted_parts.append(
                f"--- Result {i} [Source: {source}] "
                f"[Chunk: {chunk_id}] "
                f"[Relevance: {score:.2%}] ---\n"
                f"{result.content}\n"
            )

        logger.info(
            "policy_search_success",
            query=query[:100],
            results_count=len(results),
        )

        return "\n".join(formatted_parts)

    except Exception as e:
        logger.error(
            "policy_search_error",
            query=query[:100],
            error=str(e),
        )
        return vector_store.get_fallback_response()
