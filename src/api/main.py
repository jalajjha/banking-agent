"""
FastAPI application entry point for the Cred Domain Support Agent.
Configures CORS, logging middleware, global exception handling,
and mounts the API routers.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers import router
from src.api.schemas import ErrorResponse
from src.utils.logging_config import setup_logging, get_logger, set_request_id
from src.rag.vector_store import vector_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.
    Initializes logging and ingests policy documents into the vector store
    at startup.
    """
    # ── Startup ──
    setup_logging()
    logger.info("application_starting", service="cred-support-agent")

    # Ingest policy documents into vector store
    try:
        chunk_count = vector_store.ingest()
        logger.info("vector_store_ready", chunks_indexed=chunk_count)
    except Exception as e:
        logger.error("vector_store_init_error", error=str(e))
        # Continue startup — the agent will use fallback responses

    yield

    # ── Shutdown ──
    logger.info("application_shutting_down")


# ── FastAPI App ──

app = FastAPI(
    title="Cred Domain Support Agent",
    description=(
        "Production-grade multi-agent RAG system for banking & FinTech "
        "customer support. Powered by CrewAI, FastAPI, and ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ──

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging Middleware ──


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Middleware that logs every request with a unique request ID,
    tracks latency, and ensures structured logging.
    """
    request_id = set_request_id()
    start_time = time.time()

    logger.info(
        "request_started",
        method=request.method,
        path=str(request.url.path),
        request_id=request_id,
    )

    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            "request_completed",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            "request_failed",
            method=request.method,
            path=str(request.url.path),
            error=str(e),
            latency_ms=round(latency_ms, 2),
            request_id=request_id,
        )
        raise


# ── Global Exception Handler ──


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch-all exception handler.
    Ensures the server never crashes due to an unhandled exception.
    Returns a standardized JSON error envelope.
    """
    logger.error(
        "unhandled_exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=str(request.url.path),
    )

    error = ErrorResponse(
        error_code="INTERNAL_SERVER_ERROR",
        message=(
            "An unexpected error occurred. Please try again later or "
            "contact Cred support at 1800-XXX-XXXX."
        ),
        request_id=request.headers.get("X-Request-ID", "unknown"),
    )

    return JSONResponse(
        status_code=500,
        content=error.model_dump(),
    )


# ── Mount Routers ──

app.include_router(router, prefix="", tags=["API"])
