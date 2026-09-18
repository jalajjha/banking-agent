"""
Token usage and latency metrics tracker.
Tracks per-request and cumulative LLM usage for cost control and observability.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request lifecycle."""

    request_id: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    vector_db_queries: int = 0
    tool_calls: int = 0

    @property
    def latency_ms(self) -> float:
        """Calculate total request latency in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    def finalize(self) -> None:
        """Mark the request as completed and log metrics."""
        self.end_time = time.time()
        logger.info(
            "request_metrics",
            request_id=self.request_id,
            latency_ms=round(self.latency_ms, 2),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            llm_calls=self.llm_calls,
            vector_db_queries=self.vector_db_queries,
            tool_calls=self.tool_calls,
        )

    def record_llm_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record token usage from an LLM call."""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.llm_calls += 1

    def record_vector_query(self) -> None:
        """Record a vector database query."""
        self.vector_db_queries += 1

    def record_tool_call(self) -> None:
        """Record a tool invocation."""
        self.tool_calls += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a dictionary."""
        return {
            "request_id": self.request_id,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "vector_db_queries": self.vector_db_queries,
            "tool_calls": self.tool_calls,
        }


class TokenTracker:
    """Global token usage tracker across all requests."""

    def __init__(self) -> None:
        self._cumulative_prompt_tokens: int = 0
        self._cumulative_completion_tokens: int = 0
        self._total_requests: int = 0

    def track_request(self, metrics: RequestMetrics) -> None:
        """Accumulate metrics from a completed request."""
        self._cumulative_prompt_tokens += metrics.prompt_tokens
        self._cumulative_completion_tokens += metrics.completion_tokens
        self._total_requests += 1

    @property
    def cumulative_stats(self) -> dict[str, int]:
        return {
            "total_requests": self._total_requests,
            "cumulative_prompt_tokens": self._cumulative_prompt_tokens,
            "cumulative_completion_tokens": self._cumulative_completion_tokens,
            "cumulative_total_tokens": (
                self._cumulative_prompt_tokens + self._cumulative_completion_tokens
            ),
        }


# Singleton instance
token_tracker = TokenTracker()
