"""
API routers for the Cred Support Agent.
Defines the /query endpoint that validates input, runs guardrails,
dispatches to the crew orchestrator, and sanitizes output.
"""

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    ErrorResponse,
    HealthResponse,
)
from src.guardrails.input_validator import validate_input
from src.guardrails.output_sanitizer import sanitize_output
from src.agents.crew import cred_crew
from src.utils.logging_config import get_logger, set_request_id, get_request_id
from src.utils.token_tracker import RequestMetrics, token_tracker

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        403: {"model": ErrorResponse, "description": "Access denied"},
        422: {"model": ErrorResponse, "description": "Input validation failed"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Support"],
    summary="Submit a banking support query",
    description=(
        "Send a banking-related query to the Cred Support Agent. "
        "The system will route your query to the appropriate specialist "
        "(policy, records, or general support) and return a grounded response."
    ),
)
async def process_query(request: QueryRequest) -> QueryResponse:
    """
    Main query endpoint.

    Flow:
    1. Set request ID for tracking
    2. Validate input (guardrails)
    3. Route to CrewAI orchestrator
    4. Sanitize output (PII masking)
    5. Return structured response
    """
    # Step 1: Initialize request tracking
    request_id = set_request_id()
    metrics = RequestMetrics(request_id=request_id)

    logger.info(
        "query_received",
        query=request.query[:100],
        customer_id=request.customer_id,
        request_id=request_id,
    )

    try:
        # Step 2: Input validation (guardrails)
        validation = validate_input(
            query=request.query,
            customer_id=request.customer_id,
        )

        if not validation.is_valid:
            logger.warning(
                "query_rejected",
                reason=validation.rejection_reason,
                risk_flags=validation.risk_flags,
            )

            error_code = "VALIDATION_ERROR"
            status_code = 400

            if "prompt_injection" in validation.risk_flags:
                error_code = "SECURITY_VIOLATION"
                status_code = 403
            elif "cross_customer_access" in validation.risk_flags:
                error_code = "ACCESS_DENIED"
                status_code = 403
            elif "out_of_domain" in validation.risk_flags:
                error_code = "OUT_OF_DOMAIN"
                status_code = 400

            raise HTTPException(
                status_code=status_code,
                detail={
                    "status": "error",
                    "error_code": error_code,
                    "message": validation.rejection_reason,
                    "request_id": request_id,
                },
            )

        # Step 3: Process through CrewAI orchestrator
        logger.info("dispatching_to_crew", query=validation.sanitized_query[:100])

        agent_response = cred_crew.process_query(
            query=validation.sanitized_query,
            customer_id=request.customer_id,
        )

        metrics.record_tool_call()

        # Step 4: Sanitize output (PII masking)
        sanitized_answer = sanitize_output(agent_response.answer)

        # Step 5: Finalize metrics
        metrics.finalize()
        token_tracker.track_request(metrics)

        logger.info(
            "query_completed",
            agent_used=agent_response.agent_used,
            latency_ms=metrics.latency_ms,
        )

        return QueryResponse(
            query=request.query,
            answer=sanitized_answer,
            agent_used=agent_response.agent_used,
            confidence=agent_response.confidence,
            sources=[],
            request_id=request_id,
            metadata=metrics.to_dict(),
        )

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise

    except Exception as e:
        # Catch-all: never crash the server
        logger.error(
            "query_processing_error",
            error=str(e),
            error_type=type(e).__name__,
        )

        metrics.finalize()

        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "error_code": "INTERNAL_ERROR",
                "message": (
                    "An unexpected error occurred while processing your request. "
                    "Please try again or contact support at 1800-XXX-XXXX."
                ),
                "request_id": request_id,
            },
        )
