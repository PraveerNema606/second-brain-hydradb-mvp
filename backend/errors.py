"""
Typed errors + FastAPI handlers for the Second Brain MVP.

All downstream failures (HydraDB, LLM, network/timeout) raise one of these
exceptions. main.py registers a single handler per type so the user always
gets a small, predictable JSON shape:

    { "detail": "<friendly message>", "error_type": "<short tag>" }

This means routes/services don't need to translate errors into HTTP
themselves — they just `raise`. It also means we never leak an exception
class name, full stack trace, prompt content, or API key into the
response body.
"""

from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from logging_config import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    """
    Base class for everything we surface to the API client.

    Subclasses set their own:
        status_code     -> HTTP status returned
        error_type      -> short machine-readable tag for the UI to switch on
        default_detail  -> user-facing message when caller doesn't pass one
    """

    status_code: int = 500
    error_type: str = "internal_error"
    default_detail: str = "Something went wrong."

    def __init__(
        self,
        detail: str = "",
        *,
        log_context: str = "",
        upstream_status: Optional[int] = None,
    ):
        # `detail` is shown to the user.
        # `log_context` is printed to stdout for operators but never returned.
        # `upstream_status` carries the HTTP status from the upstream service
        #   so the retry framework can decide whether to retry without
        #   knowing the error message content.
        super().__init__(detail or self.default_detail)
        self.detail = detail or self.default_detail
        self.log_context = log_context
        self.upstream_status: Optional[int] = upstream_status


class HydraDBError(AppError):
    status_code = 502
    error_type = "hydradb_error"
    default_detail = "The knowledge backend is unavailable. Please try again."


class WorkspaceTenantError(AppError):
    """
    Raised when a workspace-scoped HydraDB operation cannot resolve a
    hydradb_sub_tenant_id. Callers MUST fail closed — never fall back to
    the legacy env / default tenant (that would cross workspace data).
    """

    status_code = 502
    error_type = "workspace_tenant_error"
    default_detail = "Could not resolve workspace HydraDB tenant."


class LLMError(AppError):
    status_code = 502
    error_type = "llm_error"
    default_detail = "The language model is unavailable. Please try again."


class UpstreamTimeoutError(AppError):
    status_code = 504
    error_type = "upstream_timeout"
    default_detail = "An upstream service timed out. Please try again."


class RateLimitedError(AppError):
    status_code = 429
    error_type = "rate_limited"
    default_detail = "Too many requests. Please slow down."


# ---------------------------------------------------------------------- #
# FastAPI handler
# ---------------------------------------------------------------------- #
def _payload(exc: AppError) -> Dict[str, Any]:
    return {"detail": exc.detail, "error_type": exc.error_type}


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Single handler for every AppError subclass."""
    # Log shape: operators see the error type + their own log_context, never
    # the raw stack and never user data.
    logger.warning(
        'app_error',
        extra={'error_type': exc.error_type, 'log_context': exc.log_context or exc.detail},
    )
    return JSONResponse(status_code=exc.status_code, content=_payload(exc))
