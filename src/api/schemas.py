"""
Pydantic v2 request/response schemas for the FastAPI application.
All API inputs and outputs are validated through these models.
"""

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query request from the client."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The customer's question or request",
        examples=["What is the credit card late payment policy?"],
    )
    customer_id: str | None = Field(
        default=None,
        description="Customer ID for account-specific queries (e.g., 'CUST001')",
        pattern=r"^CUST\d{3}$",
        examples=["CUST001"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for conversation continuity",
    )


class SourceReference(BaseModel):
    """A source reference from a retrieved document chunk."""

    source: str = Field(..., description="Source document name")
    chunk_id: str = Field(default="", description="Chunk identifier")
    relevance_score: float = Field(
        default=0.0, description="Relevance score (0–1)"
    )


class QueryResponse(BaseModel):
    """Successful response to a customer query."""

    status: str = Field(default="success", description="Response status")
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The agent's response")
    agent_used: str = Field(
        ..., description="Which agent handled the query (policy/record/support)"
    )
    confidence: str = Field(
        default="medium", description="Confidence level: high, medium, low"
    )
    sources: list[SourceReference] = Field(
        default_factory=list, description="Source references"
    )
    request_id: str = Field(default="", description="Unique request tracking ID")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (latency, token usage, etc.)",
    )


class ErrorResponse(BaseModel):
    """Standardized error response envelope."""

    status: str = Field(default="error", description="Always 'error'")
    error_code: str = Field(
        ..., description="Machine-readable error code"
    )
    message: str = Field(
        ..., description="Human-readable error message"
    )
    request_id: str = Field(default="", description="Request tracking ID")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    service: str = Field(default="cred-support-agent")
    version: str = Field(default="1.0.0")
