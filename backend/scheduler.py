"""
Optional background ingestion for the Second Brain MVP.

Phase 4 update: workspace-isolated ingestion.
Phase 11 update: Gmail sweep added next to the Slack sweep.

If AUTO_INGEST=true in the environment, FastAPI startup spins up an
APScheduler BackgroundScheduler that runs one combined ingest pass
every AUTO_INGEST_INTERVAL_MINUTES. Each pass:

  1. Iterates every active workspace with a Slack installation and
     runs slack_oauth.run_workspace_ingest with that workspace's
     bot_token, selected channel_ids, and HydraDB sub_tenant_id.
  2. Iterates every active (workspace, Gmail connection) pair that
     has at least one selected label, and runs
     gmail_oauth.run_workspace_gmail_ingest with sync_mode="auto"
     (incremental when a watermark exists, full otherwise -- see
     run_workspace_gmail_ingest for details).
  3. On a clean Slack pass, stamps hydradb_last_sync_at on the
     workspace row. (Gmail tracks its own per-(connection, label)
     watermark in gmail_ingestion_state; no workspace-level stamp.)

Failures in one workspace MUST NOT cancel the pass for other
workspaces -- and failures in one Gmail connection MUST NOT cancel
the pass for the workspace's other Gmail connections or for Slack.
Each unit of work is wrapped in its own try/except, the error is
logged, and the loop continues.

Notes:
- Uses an in-process scheduler. State (next-run time, last-run time) is
  not persisted across restarts, which is fine for a single-instance MVP.
- The job runs synchronously inside the scheduler's worker thread. A run
  that takes longer than the interval will not overlap with itself
  because we set max_instances=1.
- We DO NOT auto-run on startup. The first run happens after the
  configured interval, so a freshly-started server doesn't immediately
  hammer Slack / HydraDB. Set AUTO_INGEST_RUN_ON_STARTUP=true to override.
- The legacy global ingestion entry point (ingestion.ingest_slack.main)
  is still importable and still works for ad-hoc CLI runs, but the
  scheduler no longer uses it -- calling it would write into the env
  default sub-tenant, defeating workspace isolation.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler

from gmail_oauth import run_workspace_gmail_ingest
from logging_config import get_logger
from retry import RetryExhausted
from slack_oauth import run_workspace_ingest
from supabase_client import (
    list_active_workspaces_with_gmail,
    list_active_workspaces_with_slack,
    mark_workspace_synced,
)

logger = get_logger(__name__)


JOB_ID = "second-brain-ingestion"


def auto_ingest_enabled() -> bool:
    return os.getenv("AUTO_INGEST", "").strip().lower() in ("1", "true", "yes", "on")


def run_on_startup_enabled() -> bool:
    return os.getenv("AUTO_INGEST_RUN_ON_STARTUP", "").strip().lower() in ("1", "true", "yes", "on")


def interval_minutes() -> int:
    """Read the configured interval, with a sensible floor."""
    try:
        value = int(os.getenv("AUTO_INGEST_INTERVAL_MINUTES", "15"))
    except ValueError:
        value = 15
    # Floor at 1 minute so a typo can't accidentally hammer Slack.
    return max(1, value)


def _slack_sweep() -> Dict[str, Any]:
    """
    Run the Slack ingest sweep. Returns a per-workspace summary.

    Extracted from run_all_workspaces_once so the Gmail sweep can run
    independently and the test suite can mock either side in isolation.
    Behavior is unchanged from Phase 4: per-workspace try/except, clean
    runs get hydradb_last_sync_at stamped.
    """
    workspaces = list_active_workspaces_with_slack()
    summary: Dict[str, Any] = {
        "workspaces_total": len(workspaces),
        "workspaces_run": 0,
        "workspaces_skipped": 0,
        "workspaces_failed": 0,
    }
    if not workspaces:
        logger.info("scheduler_no_workspaces")
        return summary

    for ws in workspaces:
        workspace_id = ws.get("workspace_id") or ""
        bot_token = ws.get("bot_token") or ""
        channel_ids = ws.get("channel_ids") or []
        sub_tenant = (ws.get("hydradb_sub_tenant_id") or "").strip()

        if not channel_ids:
            # Connected but nothing selected -- nothing to ingest yet.
            summary["workspaces_skipped"] += 1
            logger.info(
                "scheduler_skip_no_channels",
                extra={"workspace_id": workspace_id},
            )
            continue

        if not sub_tenant:
            # Defensive: every workspace should have a sub_tenant_id
            # after the Phase 4 migration backfill. If we somehow see
            # one without, refuse rather than route to the global
            # bucket.
            summary["workspaces_skipped"] += 1
            logger.warning(
                "scheduler_skip_no_sub_tenant",
                extra={"workspace_id": workspace_id},
            )
            continue

        try:
            result = run_workspace_ingest(
                workspace_id=workspace_id,
                bot_token=bot_token,
                channel_ids=channel_ids,
                hydradb_sub_tenant_id=sub_tenant,
            )
        except Exception as e:  # noqa: BLE001
            summary["workspaces_failed"] += 1
            logger.error(
                "scheduler_workspace_failed",
                extra={
                    "workspace_id": workspace_id,
                    "error": type(e).__name__,
                },
                exc_info=True,
            )
            continue

        summary["workspaces_run"] += 1
        # Stamp last-synced only when the run finished cleanly. A
        # partial failure (some channels OK, others bad) still counts --
        # the alternative would mean we never mark a workspace as
        # synced if a single bad channel persists.
        if result.get("failures", 0) == 0:
            mark_workspace_synced(workspace_id=workspace_id)

    logger.info("scheduler_slack_sweep_complete", extra=summary)
    return summary


def _gmail_sweep() -> Dict[str, Any]:
    """
    Run the Gmail ingest sweep across every active (workspace,
    connection) pair with at least one selected label.

    Each connection is isolated: an exception or a permanent Gmail
    failure for connection A never blocks connection B (even within
    the same workspace), and never blocks the Slack sweep above
    (the caller runs Slack first).

    Returns a summary dict. The runner itself returns rich per-run
    metadata (sync_mode, duration_ms, refresh_token_used,
    incremental/full counts); we aggregate into top-level totals for
    log readability, but log the full per-connection record so
    operators can drill in.
    """
    rows = list_active_workspaces_with_gmail()
    summary: Dict[str, Any] = {
        "connections_total": len(rows),
        "connections_run": 0,
        "connections_failed": 0,
        "messages_uploaded": 0,
        "incremental_label_count": 0,
        "full_label_count": 0,
        "invalidations": 0,
        "refresh_tokens_used": 0,
    }
    if not rows:
        logger.info("scheduler_no_gmail_connections")
        return summary

    for row in rows:
        workspace_id = row.get("workspace_id") or ""
        connection = row.get("connection") or {}
        sub_tenant = (row.get("hydradb_sub_tenant_id") or "").strip()
        label_ids = row.get("selected_label_ids") or []
        connection_id = connection.get("id") or ""

        if not sub_tenant or not label_ids or not connection_id:
            # Defensive: list_active_workspaces_with_gmail already
            # filters these but a race with a concurrent delete could
            # surface here. Skip silently rather than crashing.
            continue

        try:
            result = run_workspace_gmail_ingest(
                workspace_id=workspace_id,
                connection=connection,
                label_ids=label_ids,
                hydradb_sub_tenant_id=sub_tenant,
                sync_mode="auto",
            )
        except Exception as e:  # noqa: BLE001
            summary["connections_failed"] += 1
            logger.error(
                "scheduler_gmail_connection_failed",
                extra={
                    "workspace_id": workspace_id,
                    "connection_id": connection_id,
                    "error": type(e).__name__,
                },
                exc_info=True,
            )
            continue

        summary["connections_run"] += 1
        summary["messages_uploaded"] += int(result.get("messages_uploaded") or 0)
        summary["incremental_label_count"] += int(result.get("incremental_label_count") or 0)
        summary["full_label_count"] += int(result.get("full_label_count") or 0)
        summary["invalidations"] += int(result.get("invalidations") or 0)
        if result.get("refresh_token_used"):
            summary["refresh_tokens_used"] += 1

        # Per-connection record for the operator. No PII (no email,
        # no subject, no body); just the metadata.
        logger.info(
            "scheduler_gmail_connection_complete",
            extra={
                "workspace_id": workspace_id,
                "connection_id": connection_id,
                "sync_mode_requested": result.get("sync_mode_requested"),
                "duration_ms": result.get("duration_ms"),
                "labels_processed": result.get("labels_processed"),
                "labels_skipped": result.get("labels_skipped"),
                "labels_failed": result.get("labels_failed"),
                "messages_uploaded": result.get("messages_uploaded"),
                "messages_failed": result.get("messages_failed"),
                "incremental_label_count": result.get("incremental_label_count"),
                "full_label_count": result.get("full_label_count"),
                "invalidations": result.get("invalidations"),
                "refresh_token_used": result.get("refresh_token_used"),
            },
        )

    logger.info("scheduler_gmail_sweep_complete", extra=summary)
    return summary


def run_all_workspaces_once() -> dict:
    """
    Run one ingestion pass across every active workspace -- Slack
    first, then Gmail. Returns a small summary dict.

    Per-workspace + per-connection errors are caught and logged so a
    single bad token or a single dead channel never aborts the whole
    sweep.

    The Slack portion of the return shape is unchanged from Phase 4
    (existing tests check it directly). The Gmail portion is namespaced
    under "gmail" so the legacy keys (workspaces_total etc.) remain
    Slack-shaped.
    """
    slack_summary = _slack_sweep()
    gmail_summary = _gmail_sweep()
    combined: Dict[str, Any] = dict(slack_summary)
    combined["gmail"] = gmail_summary
    return combined


def _job_wrapper() -> None:
    """
    Run one ingestion pass. Swallow exceptions so a single failure
    never kills the scheduler thread -- APScheduler would otherwise
    stop scheduling the job.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    logger.info("scheduler_job_start", extra={"started_at": started_at})
    try:
        run_all_workspaces_once()
        logger.info("scheduler_job_finished")
    except RetryExhausted as e:
        logger.error('scheduler_job_retry_exhausted', extra={'error': str(e)})
    except Exception as e:  # noqa: BLE001
        logger.error(
            "scheduler_job_failed",
            extra={"error": type(e).__name__},
            exc_info=True,
        )


_scheduler: Optional[BackgroundScheduler] = None


def start_scheduler() -> None:
    """Start the scheduler if AUTO_INGEST is enabled."""
    global _scheduler
    if not auto_ingest_enabled():
        logger.info("scheduler_disabled", extra={"reason": "AUTO_INGEST=false"})
        return

    if _scheduler is not None:
        logger.info("scheduler_already_started")
        return

    minutes = interval_minutes()
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _job_wrapper,
        trigger="interval",
        minutes=minutes,
        id=JOB_ID,
        max_instances=1,  # never run two ingestions in parallel
        coalesce=True,  # if we fall behind, run once not N times
        replace_existing=True,
    )
    _scheduler.start()
    logger.info("scheduler_started", extra={"interval_minutes": minutes})

    if run_on_startup_enabled():
        logger.info("scheduler_startup_run_queued")
        _scheduler.add_job(_job_wrapper, id=f"{JOB_ID}-startup")


def stop_scheduler() -> None:
    """Stop the scheduler cleanly on shutdown."""
    global _scheduler
    if _scheduler is None:
        return
    logger.info("scheduler_stopping")
    _scheduler.shutdown(wait=False)
    _scheduler = None
