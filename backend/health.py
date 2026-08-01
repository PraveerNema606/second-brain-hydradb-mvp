"""
Health check system for the Second Brain MVP.

Endpoints (registered on the returned `router`):
    GET /api/health/live   → liveness probe   (always 200 while process runs)
    GET /api/health/ready  → readiness probe  (200 req deps ok; 503 otherwise)
    GET /api/health        → full diagnostics (always 200; inspect 'status')

Dependency model:
    Required  (failure → overall "unhealthy"):  hydradb, llm
    Optional  (failure → overall "degraded"):   slack, scheduler, database

Adding a future service:
    1. Subclass HealthCheck and implement async check() → HealthResult
    2. Call register_health_check(MyCheck()) before the app starts serving.
       The three endpoints will automatically include it.

Timeout:
    Each check is capped at CHECK_TIMEOUT_SECONDS (3 s).  A hung external
    service cannot hang the endpoint.

Structured log lines:
    {"event": "health_check_started",   "check": "..."}
    {"event": "health_check_completed", "check": "...", "status": "..."}
    {"event": "health_check_failed",    "check": "...", "reason": "..."}
"""

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from logging_config import get_logger as _get_logger

_logger = _get_logger(__name__)

# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------
STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_ERROR = "error"
STATUS_NOT_CONFIGURED = "not_configured"

# Maximum wall-clock time per individual check.
CHECK_TIMEOUT_SECONDS = 3.0


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class HealthResult:
    status: str  # one of the STATUS_* constants above
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"status": self.status}
        if self.latency_ms is not None:
            d["latency_ms"] = round(self.latency_ms, 1)
        if self.message:
            d["message"] = self.message
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class HealthCheck(ABC):
    name: str = "unknown"
    required: bool = True  # True → failure causes "unhealthy"; False → "degraded"

    @abstractmethod
    async def check(self) -> HealthResult: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_registry: List[HealthCheck] = []


def register_health_check(check: HealthCheck) -> None:
    """Register a health check.  Call before the app starts serving."""
    _registry.append(check)


def _get_registry() -> List[HealthCheck]:
    return list(_registry)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------
def _log(event: str, check_name: str, **kwargs: Any) -> None:
    _logger.info(event, extra={"check": check_name, **kwargs})


# ---------------------------------------------------------------------------
# Per-check runner with timeout
# ---------------------------------------------------------------------------
async def _run_check(check: HealthCheck) -> Tuple[str, HealthResult]:
    """Run one check within the timeout budget.  Never raises."""
    _log("health_check_started", check.name)
    try:
        result = await asyncio.wait_for(check.check(), timeout=CHECK_TIMEOUT_SECONDS)
        _log("health_check_completed", check.name, status=result.status)
        return check.name, result
    except asyncio.TimeoutError:
        _log("health_check_failed", check.name, reason=f"timed out after {CHECK_TIMEOUT_SECONDS}s")
        return check.name, HealthResult(
            status=STATUS_ERROR,
            message=f"check timed out after {CHECK_TIMEOUT_SECONDS}s",
        )
    except Exception as exc:
        _log("health_check_failed", check.name, reason=str(exc)[:200])
        return check.name, HealthResult(status=STATUS_ERROR, message=str(exc)[:200])


# ---------------------------------------------------------------------------
# Overall status aggregation
# ---------------------------------------------------------------------------
def _aggregate_status(
    checks: List[HealthCheck],
    results: Dict[str, HealthResult],
) -> str:
    has_required_error = False
    has_degraded = False

    for check in checks:
        r = results.get(check.name)
        if r is None:
            continue
        if r.status == STATUS_ERROR:
            if check.required:
                has_required_error = True
            else:
                has_degraded = True
        elif r.status == STATUS_WARNING:
            has_degraded = True
        # STATUS_NOT_CONFIGURED is neutral — does not degrade overall status.

    if has_required_error:
        return "unhealthy"
    if has_degraded:
        return "degraded"
    return "healthy"


# ---------------------------------------------------------------------------
# Concrete check: HydraDB
# ---------------------------------------------------------------------------
class HydraHealthCheck(HealthCheck):
    name = "hydradb"
    required = True

    async def check(self) -> HealthResult:
        import requests  # noqa: PLC0415  (deferred import — safe for tests)

        api_key = os.getenv("HYDRADB_API_KEY")
        tenant_id = os.getenv("HYDRADB_TENANT_ID")
        if not api_key or not tenant_id:
            return HealthResult(
                status=STATUS_ERROR,
                message="HYDRADB_API_KEY or HYDRADB_TENANT_ID not configured",
            )

        base_url = os.getenv("HYDRADB_BASE_URL", "https://api.hydradb.com").rstrip("/")
        # No hard-coded default tenant — readiness must use an explicitly
        # configured probe sub-tenant (typically HYDRADB_SUB_TENANT_ID).
        sub_tenant_id = (os.getenv("HYDRADB_SUB_TENANT_ID") or "").strip()
        if not sub_tenant_id:
            return HealthResult(
                status=STATUS_ERROR,
                message="HYDRADB_SUB_TENANT_ID not configured",
            )
        url = f"{base_url}/recall/full_recall"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "tenant_id": tenant_id,
            "sub_tenant_id": sub_tenant_id,
            "query": "health check",
            "top_k": 1,
        }

        start = time.monotonic()
        try:
            resp = await asyncio.to_thread(lambda: requests.post(url, headers=headers, json=body, timeout=2.5))
            latency_ms = (time.monotonic() - start) * 1000

            if resp.status_code == 401:
                return HealthResult(
                    status=STATUS_ERROR,
                    latency_ms=latency_ms,
                    message="authentication failed",
                )
            if resp.status_code == 403:
                return HealthResult(
                    status=STATUS_ERROR,
                    latency_ms=latency_ms,
                    message="forbidden",
                )
            if resp.status_code >= 500:
                return HealthResult(
                    status=STATUS_ERROR,
                    latency_ms=latency_ms,
                    message=f"server error {resp.status_code}",
                )
            return HealthResult(status=STATUS_OK, latency_ms=latency_ms)

        except requests.exceptions.Timeout:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                message="request timed out",
            )
        except requests.exceptions.ConnectionError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                message=f"connection error: {exc}",
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                message=str(exc)[:200],
            )


# ---------------------------------------------------------------------------
# Concrete check: LLM
# ---------------------------------------------------------------------------
class LLMHealthCheck(HealthCheck):
    name = "llm"
    required = True

    async def check(self) -> HealthResult:
        import openai  # noqa: PLC0415

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return HealthResult(status=STATUS_ERROR, message="OPENAI_API_KEY not configured")

        base_url = os.getenv("OPENAI_BASE_URL") or None
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        start = time.monotonic()
        try:

            def _call() -> None:
                client = (
                    openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
                )
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=1,
                    temperature=0.0,
                )

            await asyncio.to_thread(_call)
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(status=STATUS_OK, latency_ms=latency_ms)

        except openai.AuthenticationError:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                message="authentication failed",
            )
        except openai.PermissionDeniedError:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                message="permission denied",
            )
        except openai.APITimeoutError:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                message="request timed out",
            )
        except openai.APIConnectionError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                message=f"connection error: {type(exc).__name__}",
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_ERROR,
                latency_ms=latency_ms,
                message=type(exc).__name__,
            )


# ---------------------------------------------------------------------------
# Concrete check: Slack
# ---------------------------------------------------------------------------
class SlackHealthCheck(HealthCheck):
    name = "slack"
    required = False

    async def check(self) -> HealthResult:
        from slack_sdk import WebClient  # noqa: PLC0415
        from slack_sdk.errors import SlackApiError  # noqa: PLC0415

        token = os.getenv("SLACK_BOT_TOKEN")
        if not token:
            return HealthResult(
                status=STATUS_NOT_CONFIGURED,
                message="SLACK_BOT_TOKEN not configured",
            )

        start = time.monotonic()
        try:

            def _call():
                return WebClient(token=token).auth_test()

            resp = await asyncio.to_thread(_call)
            latency_ms = (time.monotonic() - start) * 1000

            if not resp.get("ok"):
                error = resp.get("error", "unknown")
                return HealthResult(
                    status=STATUS_WARNING,
                    latency_ms=latency_ms,
                    message=f"auth.test failed: {error}",
                )
            return HealthResult(status=STATUS_OK, latency_ms=latency_ms)

        except SlackApiError as exc:
            latency_ms = (time.monotonic() - start) * 1000
            resp_data = getattr(exc, "response", None) or {}
            err = resp_data.get("error", str(exc)) if isinstance(resp_data, dict) else str(exc)
            return HealthResult(
                status=STATUS_WARNING,
                latency_ms=latency_ms,
                message=f"slack error: {err}",
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return HealthResult(
                status=STATUS_WARNING,
                latency_ms=latency_ms,
                message=str(exc)[:200],
            )


# ---------------------------------------------------------------------------
# Concrete check: Scheduler
# ---------------------------------------------------------------------------
class SchedulerHealthCheck(HealthCheck):
    name = "scheduler"
    required = False

    async def check(self) -> HealthResult:
        import scheduler as _sched  # noqa: PLC0415

        instance = _sched._scheduler

        if instance is None:
            if not _sched.auto_ingest_enabled():
                return HealthResult(
                    status=STATUS_OK,
                    message="scheduler disabled (AUTO_INGEST=false)",
                    extra={"jobs": 0},
                )
            return HealthResult(
                status=STATUS_WARNING,
                message="scheduler not running but AUTO_INGEST=true",
                extra={"jobs": 0},
            )

        if not instance.running:
            return HealthResult(
                status=STATUS_WARNING,
                message="scheduler initialized but not running",
                extra={"jobs": 0},
            )

        jobs = instance.get_jobs()
        return HealthResult(
            status=STATUS_OK,
            extra={"jobs": len(jobs)},
        )


# ---------------------------------------------------------------------------
# Placeholder: future database (Supabase etc.)
# ---------------------------------------------------------------------------
class DatabaseHealthCheckPlaceholder(HealthCheck):
    name = "database"
    required = False

    async def check(self) -> HealthResult:
        return HealthResult(status=STATUS_NOT_CONFIGURED)


# ---------------------------------------------------------------------------
# Populate default registry (called once at import time)
# ---------------------------------------------------------------------------
def _init_default_registry() -> None:
    register_health_check(HydraHealthCheck())
    register_health_check(LLMHealthCheck())
    register_health_check(SlackHealthCheck())
    register_health_check(SchedulerHealthCheck())
    register_health_check(DatabaseHealthCheckPlaceholder())


_init_default_registry()


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------
router = APIRouter()


@router.get("/api/health/live")
async def liveness() -> Dict[str, str]:
    """
    Liveness probe — always 200 while the process is running.
    Kubernetes / load-balancer: if this fails, restart the pod.
    """
    return {"status": "alive"}


@router.get("/api/health/ready")
async def readiness() -> JSONResponse:
    """
    Readiness probe — 200 only when required dependencies are healthy.
    Kubernetes / load-balancer: remove from rotation until this passes.
    """
    required = [c for c in _get_registry() if c.required]
    pairs = await asyncio.gather(*[_run_check(c) for c in required])

    checks_out: Dict[str, str] = {}
    all_ok = True
    for name, result in pairs:
        checks_out[name] = result.status
        if result.status == STATUS_ERROR:
            all_ok = False

    status_code = 200 if all_ok else 503
    body = {"status": "ready" if all_ok else "not_ready", "checks": checks_out}
    return JSONResponse(content=body, status_code=status_code)


@router.get("/api/health")
async def detailed_health() -> Dict[str, Any]:
    """
    Full diagnostic snapshot across all registered checks.
    Always returns HTTP 200; callers inspect the top-level 'status' field.
    """
    all_checks = _get_registry()
    pairs = await asyncio.gather(*[_run_check(c) for c in all_checks])

    results: Dict[str, HealthResult] = dict(pairs)
    overall = _aggregate_status(all_checks, results)

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {name: r.as_dict() for name, r in results.items()},
    }
