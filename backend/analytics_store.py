"""
Phase 15: analytics event persistence + aggregation.

Two responsibilities:

  1. `emit_event` -- record a single analytics row. Wrapped in
     try/except: a Supabase outage or rate limit must NEVER fail the
     surrounding product operation (query / ingest / etc).

  2. `aggregate_*` helpers -- read recent events and compute small
     summary dicts the UI renders. All workspace-scoped. We compute
     on-read because the volumes are tiny (events per workspace per
     day are at most thousands, usually dozens). When the math becomes
     interesting later we can add materialized rollups; for now,
     don't.

Mirrors the defensive style of `memory_store.py`: returns 0 / None /
empty list / empty dict on any failure, with a structured warning
log. Never raises to callers.

The same table also serves the ingestion-analytics requirement (C)
and the retrieval-analytics requirement (B) -- different `kind`
values, same row shape.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from logging_config import get_logger
from supabase_client import get_supabase

logger = get_logger(__name__)


_VALID_KINDS = (
    "query_completed",
    "ingest_completed",
    "memory_extracted",
    "retrieval_failure",
)


# ---------------------------------------------------------------------- #
# Emit
# ---------------------------------------------------------------------- #


def emit_event(
    *,
    workspace_id: str,
    kind: str,
    source_kind: Optional[str] = None,
    latency_ms: Optional[int] = None,
    success: Optional[bool] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Insert one analytics row. Returns True on success, False on any
    failure (logged). Callers SHOULD ignore the return value and
    never branch on it -- analytics must be a fire-and-forget signal
    from the product's perspective.

    `kind` is required; the rest is per-event optional. `payload`
    holds anything kind-specific (e.g. retrieval mode, recency hit,
    sources count). We don't validate `payload` shape here -- the
    aggregation helpers below tolerate missing keys.
    """
    if not workspace_id or not kind:
        return False
    if kind not in _VALID_KINDS:
        return False
    row: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "kind": kind,
        "payload": payload or {},
    }
    if source_kind is not None:
        row["source_kind"] = source_kind
    if latency_ms is not None:
        try:
            row["latency_ms"] = max(0, int(latency_ms))
        except (TypeError, ValueError):
            pass
    if success is not None:
        row["success"] = bool(success)
    try:
        client = get_supabase()
        client.table("analytics_events").insert(row).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "analytics_emit_failed",
            extra={
                "workspace_id": workspace_id,
                "kind": kind,
                "error": type(e).__name__,
            },
        )
        return False
    return True


# ---------------------------------------------------------------------- #
# Aggregations
# ---------------------------------------------------------------------- #
# All of these read a window's worth of events and project them into
# small UI-friendly dicts. The default window is 7 days; longer
# windows are accepted but capped at 90 days so an accidental open
# query can't pull the whole table.

_MAX_WINDOW_DAYS = 90
_DEFAULT_WINDOW_DAYS = 7


def _window_start(days: int) -> str:
    """ISO 8601 cutoff for `created_at >= now() - days`."""
    safe = max(1, min(int(days or _DEFAULT_WINDOW_DAYS), _MAX_WINDOW_DAYS))
    return (datetime.now(timezone.utc) - timedelta(days=safe)).isoformat()


def _list_events(
    *,
    workspace_id: str,
    kinds: Optional[List[str]] = None,
    days: int = _DEFAULT_WINDOW_DAYS,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Workspace-scoped read of recent events. Used by every
    aggregator below. Returns `[]` on Supabase error so callers
    degrade gracefully.

    `limit` is a defensive cap. 5000 rows handles ~weeks of activity
    for a single workspace; if a workspace ever exceeds this we'll
    add a paged-aggregate path. Until then, the cap is fine.
    """
    if not workspace_id:
        return []
    cutoff = _window_start(days)
    try:
        client = get_supabase()
        q = (
            client.table("analytics_events")
            .select("id, kind, source_kind, latency_ms, success, " "payload, created_at")
            .eq("workspace_id", workspace_id)
            .gte("created_at", cutoff)
        )
        if kinds:
            safe = [k for k in kinds if k in _VALID_KINDS]
            if not safe:
                return []
            q = q.in_("kind", safe)
        resp = q.order("created_at", desc=True).limit(limit).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "analytics_list_failed",
            extra={
                "workspace_id": workspace_id,
                "error": type(e).__name__,
            },
        )
        return []
    return list(getattr(resp, "data", None) or [])


def aggregate_query_stats(
    *,
    workspace_id: str,
    days: int = _DEFAULT_WINDOW_DAYS,
) -> Dict[str, Any]:
    """
    Summary stats for `query_completed` events:

      {
        "count":              int,
        "avg_latency_ms":     float | None,
        "p50_latency_ms":     int   | None,
        "p95_latency_ms":     int   | None,
        "empty_result_count": int,        # answers with 0 sources
        "memory_hit_count":   int,        # answers that included a memory
        "by_source": {
            "slack":     int,             # answers with >=1 Slack source
            "gmail":     int,
            "memory":    int,
            "mixed":     int,             # answers with >=2 distinct kinds
        },
        "recency_rerank_count": int,
      }

    All counts are over the window. Latency stats use median +
    p95 (cheap on the wire vs avg-only, more useful for tuning).
    """
    events = _list_events(
        workspace_id=workspace_id,
        kinds=["query_completed"],
        days=days,
    )
    out: Dict[str, Any] = {
        "count": len(events),
        "avg_latency_ms": None,
        "p50_latency_ms": None,
        "p95_latency_ms": None,
        "empty_result_count": 0,
        "memory_hit_count": 0,
        "by_source": {
            "slack": 0,
            "gmail": 0,
            "memory": 0,
            "mixed": 0,
        },
        "recency_rerank_count": 0,
    }
    if not events:
        return out
    latencies: List[int] = []
    for ev in events:
        latency_ms = ev.get("latency_ms")
        if isinstance(latency_ms, int) and latency_ms >= 0:
            latencies.append(latency_ms)
        payload = ev.get("payload") or {}
        if payload.get("sources_count", 0) == 0:
            out["empty_result_count"] += 1
        if payload.get("memory_hit"):
            out["memory_hit_count"] += 1
        if payload.get("retrieval_mode") == "recency":
            out["recency_rerank_count"] += 1
        # Each query event records which source kinds appeared in
        # the answer (computed at emit time). We distinguish "mixed"
        # to surface cross-source utility.
        source_kinds = set(payload.get("source_kinds") or [])
        if len(source_kinds) >= 2:
            out["by_source"]["mixed"] += 1
        for sk in source_kinds:
            if sk in out["by_source"]:
                out["by_source"][sk] += 1
    if latencies:
        latencies.sort()
        n = len(latencies)
        out["avg_latency_ms"] = round(sum(latencies) / n, 2)
        out["p50_latency_ms"] = latencies[n // 2]
        # p95 with the standard ceil-style index; on small n this
        # collapses to the last element which is fine.
        out["p95_latency_ms"] = latencies[min(n - 1, int(n * 0.95))]
    return out


def aggregate_ingest_stats(
    *,
    workspace_id: str,
    days: int = _DEFAULT_WINDOW_DAYS,
) -> Dict[str, Any]:
    """
    Summary stats for `ingest_completed` events:

      {
        "runs":              int,
        "messages_uploaded": int,         # sum across runs
        "failed_runs":       int,
        "by_source": {
            "slack": {"runs": int, "uploaded": int, "failed": int},
            "gmail": {"runs": int, "uploaded": int, "failed": int},
        },
        "last_ingest_at":    ISO str | None,
      }
    """
    events = _list_events(
        workspace_id=workspace_id,
        kinds=["ingest_completed"],
        days=days,
    )
    out: Dict[str, Any] = {
        "runs": len(events),
        "messages_uploaded": 0,
        "failed_runs": 0,
        "by_source": {
            "slack": {"runs": 0, "uploaded": 0, "failed": 0},
            "gmail": {"runs": 0, "uploaded": 0, "failed": 0},
        },
        "last_ingest_at": None,
    }
    if not events:
        return out
    # events come back newest-first from _list_events.
    out["last_ingest_at"] = events[0].get("created_at")
    for ev in events:
        sk = ev.get("source_kind")
        payload = ev.get("payload") or {}
        uploaded = int(payload.get("messages_uploaded") or 0)
        out["messages_uploaded"] += uploaded
        if ev.get("success") is False:
            out["failed_runs"] += 1
        if sk in out["by_source"]:
            bucket = out["by_source"][sk]
            bucket["runs"] += 1
            bucket["uploaded"] += uploaded
            if ev.get("success") is False:
                bucket["failed"] += 1
    return out


def aggregate_retrieval_failure_stats(
    *,
    workspace_id: str,
    days: int = _DEFAULT_WINDOW_DAYS,
) -> Dict[str, Any]:
    """
    Lightweight surface for retrieval failures over the window.
    Returns:
      {"count": int, "recent": [{"reason": str, "created_at": str}, ...]}
    Recent is capped at 10 so the UI can render a short list.
    """
    events = _list_events(
        workspace_id=workspace_id,
        kinds=["retrieval_failure"],
        days=days,
    )
    recent = []
    for ev in events[:10]:
        payload = ev.get("payload") or {}
        recent.append(
            {
                "reason": str(payload.get("reason") or "unknown")[:200],
                "created_at": ev.get("created_at"),
            }
        )
    return {"count": len(events), "recent": recent}


def recent_activity(
    *,
    workspace_id: str,
    days: int = _DEFAULT_WINDOW_DAYS,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Recent event feed. One mixed list newest-first, capped at
    `limit`. Used by the UI as the "what happened recently" card.

    Returns the raw events (already workspace-scoped via
    _list_events).
    """
    safe_limit = max(1, min(int(limit or 20), 100))
    events = _list_events(workspace_id=workspace_id, days=days, limit=safe_limit)
    return events[:safe_limit]
