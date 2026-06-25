"""
Realtime Slack event -> HydraDB pipeline.

Phase 5 update: workspace-aware.

Called from main.py's /slack/events handler as a FastAPI BackgroundTask
after we've already returned 200 to Slack. The handler passes the FULL
Slack payload (not just the inner event) so we can read the team_id
and route to the right workspace.

Routing:
    payload["team_id"]
        -> slack_installations row     (supabase_client.get_slack_installation_by_team_id)
        -> workspace_id + bot_token
        -> hydradb_sub_tenant_id       (supabase_client.ensure_workspace_sub_tenant)
        -> get_channel_ingest_settings(...)  # is_selected + include_bot_messages
        -> ingest using the workspace's bot_token + sub_tenant_id

Events from a Slack team we don't have an installation for are silently
ignored. Events from selected-but-archived or unselected channels are
ignored. We DO NOT fall back to the env default SLACK_BOT_TOKEN or
SLACK_CHANNEL_IDS for realtime events.

Re-uses the existing builders from ingestion.ingest_slack so the
document format and stable-key dedupe are identical to the polling
path.

Concurrency notes:
- Slack delivers events one at a time per channel but globally many can
  arrive at once. FastAPI runs BackgroundTasks sequentially per request,
  but multiple requests can fire BackgroundTasks in parallel.
- The IngestionState file is rewritten atomically (write-temp-then-rename)
  but read-modify-write isn't safe without a lock. We use a module-level
  threading.Lock to serialize the realtime path. The polling CLI runs in
  its own process and is not affected.
"""

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hydradb_client import HydraDBClient, summarize_upload_response
from ingestion.ingest_slack import (
    _record_successful_uploads,
    build_message_file,
    build_thread_file,
    fetch_channel_name,
)
from ingestion.ingestion_state import (
    IngestionState,
    stable_key_for_message,
    stable_key_for_thread,
)
from ingestion.normalize import is_noise
from ingestion.slack_client import SlackClientWrapper
from logging_config import get_logger
from supabase_client import (
    ensure_workspace_sub_tenant,
    get_channel_ingest_settings,
    get_slack_installation_by_team_id,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------------- #
def _realtime_enabled() -> bool:
    return os.getenv("REALTIME_INGEST_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


# Path to the same state file the polling CLI uses.
_BACKEND_DIR = Path(__file__).resolve().parent
STATE_PATH = _BACKEND_DIR / "data" / "ingestion_state.json"


# ---------------------------------------------------------------------- #
# Idempotency: dedupe Slack event retries (~3 retries within 1 hour).
#
# Phase 7 hardening
# -----------------
# Phase 5 stored seen event_ids in this module-level dict only. That
# survived in-process retries but not restarts and not multi-worker
# deployments. Phase 7 promotes the in-memory map to a HOT-PATH cache
# in front of a durable Supabase-backed claim:
#
#   1. Look up event_id in the in-memory dict (microseconds). If
#      present and fresh, drop -- this is the common case for retries
#      that arrive faster than the round-trip to Supabase.
#   2. Otherwise call supabase_client.claim_slack_event_id, which does
#      an INSERT ... ON CONFLICT DO NOTHING into slack_event_seen.
#      True = we claimed it (process); False = another worker already
#      claimed it (drop).
#   3. On success populate the in-memory cache so subsequent retries
#      short-circuit at step 1.
#
# If Supabase is unreachable, claim_slack_event_id falls back to
# returning True (process the event); the in-memory cache still
# prevents same-process duplicates. That's a deliberate trade-off:
# during an outage we'd rather process the occasional duplicate than
# drop every Slack event.
# ---------------------------------------------------------------------- #
_SEEN_EVENT_TTL = 60 * 60  # 1 hour matches Slack's retry window
_SEEN_EVENT_MAX = 5000  # cap memory
_seen_event_ids: Dict[str, float] = {}
_seen_lock = threading.Lock()


def _event_already_seen_in_memory(event_id: str) -> bool:
    """
    Hot-path check: is event_id already in the in-process dedupe cache?
    Returns True if so (caller should drop). Side effect: records the
    event_id in the cache so subsequent calls within the TTL also
    return True even if this caller is the first one.
    """
    if not event_id:
        return False
    now = time.monotonic()
    with _seen_lock:
        # Drop stale entries lazily -- cheaper than a background sweeper.
        if len(_seen_event_ids) > _SEEN_EVENT_MAX:
            cutoff = now - _SEEN_EVENT_TTL
            for k, t in list(_seen_event_ids.items()):
                if t < cutoff:
                    _seen_event_ids.pop(k, None)
        seen_at = _seen_event_ids.get(event_id)
        if seen_at is not None and now - seen_at < _SEEN_EVENT_TTL:
            return True
        _seen_event_ids[event_id] = now
        return False


def _event_already_seen(event_id: str) -> bool:
    """
    Phase 7: two-tier dedupe. Returns True if this event has already
    been claimed (either in-process or in Supabase).

    Flow:
      - In-process hit  -> True (drop). Records seen.
      - In-process miss -> claim against Supabase. If Supabase says we
                           lost the race (someone else already claimed
                           it), update the in-process map and return
                           True. Otherwise we own the event and return
                           False (caller proceeds).
    """
    if not event_id:
        return False

    # Stage 1: in-process cache. Marks the event_id as seen as a side
    # effect (so two concurrent threads handling Slack retries in the
    # same process can't both claim it).
    if _event_already_seen_in_memory(event_id):
        return True

    # Stage 2: durable claim. Lazy import to avoid a hard dependency at
    # module import time -- supabase_client pulls in network libs.
    from supabase_client import claim_slack_event_id  # noqa: PLC0415

    claimed = claim_slack_event_id(event_id=event_id)
    if claimed:
        # We own this event_id. The in-process cache was already
        # populated by _event_already_seen_in_memory above.
        return False

    # Lost the race against another worker. Remember it so any further
    # retries that hit THIS worker short-circuit at stage 1.
    return True


# ---------------------------------------------------------------------- #
# Bot identity cache (per workspace, keyed by bot_token)
# ---------------------------------------------------------------------- #
# Phase 4/5 isolation: each workspace has its own Slack app installation
# with its own bot user_id. We cache one entry per bot_token so the
# common case (auth.test on first event per workspace, then in-process
# hit thereafter) doesn't double-call Slack.
_bot_user_id_by_token: Dict[str, str] = {}
_bot_user_id_lock = threading.Lock()
_BOT_USER_ID_CACHE_MAX = 256  # cap for multi-tenant deployments with many workspaces


def _resolve_bot_user_id(slack: SlackClientWrapper, bot_token: str) -> Optional[str]:
    """
    Resolve and cache the bot's own user_id for the given bot_token.
    Returns None on failure -- the caller falls back to the bot_id
    field on the event (which still catches the common case).
    """
    with _bot_user_id_lock:
        cached = _bot_user_id_by_token.get(bot_token)
        if cached is not None:
            return cached or None
        # Evict oldest half when at capacity (dict is insertion-ordered).
        if len(_bot_user_id_by_token) >= _BOT_USER_ID_CACHE_MAX:
            evict = list(_bot_user_id_by_token)[: _BOT_USER_ID_CACHE_MAX // 2]
            for k in evict:
                _bot_user_id_by_token.pop(k, None)
        try:
            resp = slack.client.auth_test()
            uid = resp.get("user_id") or ""
            if not isinstance(uid, str):
                uid = ""
        except Exception as e:  # noqa: BLE001
            logger.warning(
                'realtime_auth_test_failed',
                extra={'error': type(e).__name__},
            )
            uid = ""
        _bot_user_id_by_token[bot_token] = uid
    return uid or None


# ---------------------------------------------------------------------- #
# Single shared lock around state read-modify-write for the realtime path
# ---------------------------------------------------------------------- #
_state_lock = threading.Lock()

# Stable keys currently being uploaded — prevents two concurrent webhook
# deliveries for the same message from both passing the has() check before
# either one finishes writing to state. Protected by _state_lock.
_in_flight: set = set()


# ---------------------------------------------------------------------- #
# Public entry point: handle one Slack payload (full envelope)
# ---------------------------------------------------------------------- #
def process_slack_event(payload: Dict[str, Any]) -> None:
    """
    Handle one Slack event payload (the full envelope from /slack/events).

    Phase 5: the FULL payload is passed in (not just `payload["event"]`)
    so we can read `team_id` and route to the workspace that owns this
    Slack installation. We do NOT fall back to the env default
    SLACK_BOT_TOKEN or SLACK_CHANNEL_IDS for realtime events.

    Safe to call from a BackgroundTask. Catches everything so a single
    bad event never crashes the webhook worker.
    """
    if not _realtime_enabled():
        logger.debug('realtime_disabled')
        return

    if not isinstance(payload, dict):
        logger.debug('realtime_payload_not_dict')
        return

    # Phase 7: wrap the inner processing with retry + dead-letter. The
    # inner function uses primitives (Slack API, HydraDB, IngestionState
    # file I/O) that can fail transiently; retries with backoff handle
    # the common cases. A permanent failure emits a dead_letter event
    # AND, if Sentry is configured, an exception capture.
    from observability import emit_dead_letter  # noqa: PLC0415
    from retry import retry_with_backoff  # noqa: PLC0415

    workspace_for_logs = payload.get("team_id") or _resolve_team_id(payload) or "unknown"

    def _on_giveup(err: BaseException) -> None:
        emit_dead_letter(
            kind="realtime_event",
            workspace_id=workspace_for_logs,
            error=err,
            context={
                "event_id": payload.get("event_id") or "",
                "event_type": (payload.get("event") or {}).get("type") or "",
            },
        )

    try:
        retry_with_backoff(
            _process_slack_payload_inner,
            payload,
            attempts=3,
            initial_delay=0.5,
            max_delay=4.0,
            # Retry on broad Exception so transient HTTP errors from
            # Slack or HydraDB get a second chance, but NOT on
            # KeyboardInterrupt / SystemExit (which inherit BaseException).
            retry_on=(Exception,),
            on_giveup=_on_giveup,
            op_name="realtime_event",
        )
    except Exception as e:  # noqa: BLE001
        # We logged + dead-lettered in on_giveup. Swallow here so a
        # single event never crashes the webhook worker.
        logger.error(
            'realtime_event_handler_failed',
            extra={'error': type(e).__name__},
        )


def _resolve_team_id(payload: Dict[str, Any]) -> str:
    """
    Find the Slack team_id in the event envelope.

    Order of preference:
      1. payload["team_id"]                    (single-workspace install)
      2. payload["authorizations"][0]["team_id"]
         (Slack Connect / shared channels — preferred when present)
      3. payload["event"]["team"]              (some message events)

    The Slack-recommended path is `authorizations[0].team_id` when it's
    present (it identifies the team THIS event is for, even if other
    teams can also see the channel). We fall back to the older fields
    for backwards compatibility with older Slack payloads.
    """
    auths = payload.get("authorizations")
    if isinstance(auths, list) and auths:
        first = auths[0] or {}
        if isinstance(first, dict):
            tid = (first.get("team_id") or "").strip()
            if tid:
                return tid

    tid = (payload.get("team_id") or "").strip()
    if tid:
        return tid

    event = payload.get("event") or {}
    if isinstance(event, dict):
        return (event.get("team") or "").strip()
    return ""


def _process_slack_payload_inner(payload: Dict[str, Any]) -> None:
    event = payload.get("event") or {}
    if not isinstance(event, dict):
        return

    event_type = event.get("type")
    if event_type != "message":
        logger.debug(
            'realtime_event_ignored',
            extra={'event_type': event_type},
        )
        return

    subtype = event.get("subtype")

    # Edits and deletes are handled before the workspace/channel routing
    # below because they need the same clients (slack, hydra) already
    # resolved. We defer to after that block — see the routing calls below.

    if is_noise(event):
        logger.debug(
            'realtime_event_ignored',
            extra={'reason': 'noise', 'subtype': subtype},
        )
        return

    channel_id = (event.get("channel") or "").strip()
    if not channel_id:
        logger.debug('realtime_event_no_channel')
        return

    # ----- Workspace routing -----
    # team_id -> installation -> workspace_id + bot_token + sub_tenant.
    # Without a known installation, drop the event silently. A Slack app
    # can be installed in multiple Slack workspaces; we only act on the
    # ones we have a row for.
    team_id = _resolve_team_id(payload)
    if not team_id:
        logger.debug('realtime_event_no_team_id')
        return

    installation = get_slack_installation_by_team_id(slack_team_id=team_id)
    if not installation:
        logger.debug(
            'realtime_unknown_team',
            extra={'team_id': team_id},
        )
        return

    workspace_id = (installation.get("workspace_id") or "").strip()
    bot_token = (installation.get("bot_token") or "").strip()
    if not workspace_id or not bot_token:
        logger.warning(
            'realtime_installation_incomplete',
            extra={'team_id': team_id, 'has_workspace_id': bool(workspace_id), 'has_bot_token': bool(bot_token)},
        )
        return

    # ----- Channel-selection gate -----
    # The user must have explicitly opted this channel in via the Slack
    # settings panel. Unselected channels are dropped before we touch
    # the Slack API. We also read include_bot_messages here so the
    # bot-loop guard below can allow opted-in bots through.
    channel_settings = get_channel_ingest_settings(
        workspace_id=workspace_id,
        slack_channel_id=channel_id,
    )
    if not channel_settings or not channel_settings["is_selected"]:
        logger.debug(
            'realtime_channel_not_selected',
            extra={'workspace_id': workspace_id, 'channel_id': channel_id},
        )
        return
    include_bot_messages = channel_settings.get("include_bot_messages", False)

    # ----- Resolve workspace HydraDB sub-tenant -----
    sub_tenant = ensure_workspace_sub_tenant(workspace_id=workspace_id)
    if not sub_tenant:
        logger.warning(
            'realtime_no_sub_tenant',
            extra={'workspace_id': workspace_id},
        )
        return

    # ----- Build per-workspace clients -----
    # Each workspace has its own Slack bot token AND its own HydraDB
    # sub-tenant. Constructing them per-event is cheap; the alternative
    # (process-wide caching keyed by token) would only help under
    # sustained load, which Phase 5 doesn't target.
    slack = SlackClientWrapper(token=bot_token)
    hydra = HydraDBClient(sub_tenant_id=sub_tenant)

    # ----- Bot-loop guard -----
    bot_user_id = _resolve_bot_user_id(slack, bot_token)
    is_bot_message = bool(event.get("bot_id")) or subtype == "bot_message"
    if is_bot_message and not include_bot_messages:
        logger.debug('realtime_event_ignored', extra={'reason': 'bot_message_not_opted_in'})
        return
    # Always drop the workspace's own bot (loop prevention), even when opted in.
    if bot_user_id and event.get("user") == bot_user_id:
        logger.debug('realtime_event_ignored', extra={'reason': 'own_bot_user'})
        return
    # Edit/delete events carry no top-level text; exempt them before the
    # empty-text guard so they reach the handlers below.
    if subtype not in ("message_changed", "message_deleted"):
        if not (event.get("text") or "").strip():
            logger.debug('realtime_event_ignored', extra={'reason': 'empty_text'})
            return

    channel_name = fetch_channel_name(slack, channel_id)

    # ----- Route edit/delete subtypes -----
    if subtype == "message_changed":
        _handle_message_changed(slack, hydra, channel_id, channel_name, event, workspace_id=workspace_id)
        return
    if subtype == "message_deleted":
        _handle_message_deleted(channel_id, channel_name, event, hydra, slack, workspace_id=workspace_id)
        return

    # ----- Distinguish standalone vs thread -----
    # A message belongs to a thread if `thread_ts` is set AND differs
    # from its own `ts`. A thread parent appears as a normal message
    # event the first time it's posted (no thread_ts). When someone
    # replies, the reply has thread_ts != ts -- we then re-upload the
    # whole thread doc (parent + all replies) so the stored markdown
    # always represents the current thread state.
    ts = event.get("ts") or ""
    thread_ts = event.get("thread_ts")
    is_reply = bool(thread_ts) and thread_ts != ts

    if is_reply:
        _ingest_thread(
            slack,
            hydra,
            channel_id,
            channel_name,
            thread_ts,
            event,
            workspace_id=workspace_id,
        )
    else:
        _ingest_standalone(
            slack,
            hydra,
            channel_id,
            channel_name,
            event,
            workspace_id=workspace_id,
        )


# ---------------------------------------------------------------------- #
# Ingestion paths
# ---------------------------------------------------------------------- #
def _ingest_standalone(
    slack: SlackClientWrapper,
    hydra: HydraDBClient,
    channel_id: str,
    channel_name: str,
    message: Dict[str, Any],
    *,
    workspace_id: str = "",
    force_reupload: bool = False,
) -> None:
    """Build + upload a single standalone-message doc."""
    ts = message.get("ts", "")
    stable_key = stable_key_for_message(channel_id, ts)

    with _state_lock:
        state = IngestionState(STATE_PATH)
        if (not force_reupload and state.has(stable_key)) or stable_key in _in_flight:
            logger.debug(
                'realtime_already_ingested',
                extra={'stable_key': stable_key},
            )
            return
        _in_flight.add(stable_key)

    try:
        prepared = build_message_file(message, channel_id, channel_name, slack)
        logger.info(
            'realtime_uploading',
            extra={'stable_key': stable_key, 'doc_type': 'message'},
        )

        response = hydra.upload_knowledge([prepared])
        ok, _bad = summarize_upload_response(
            response if isinstance(response, dict) else {},
            batch_size=1,
        )
        if ok < 1:
            logger.warning(
                'realtime_upload_failed',
                extra={'stable_key': stable_key},
            )
            return

        with _state_lock:
            # IngestionState.locked() acquires an OS-level advisory lock, loads
            # fresh state from disk (cross-process safe), applies mutations, then
            # saves and releases automatically on exit.
            with IngestionState.locked(STATE_PATH) as state:
                _record_successful_uploads(state, [prepared], response or {})
                state.touch_last_ingested()

        # Phase 12: extract structured memory. Realtime path mirrors the
        # batch ingest hook -- any failure here MUST NOT block ingestion.
        if workspace_id:
            try:
                _extract_memory_from_prepared(workspace_id, prepared)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "realtime_memory_extract_failed",
                    extra={
                        "workspace_id": workspace_id,
                        "stable_key": stable_key,
                        "error": type(e).__name__,
                    },
                )
    finally:
        with _state_lock:
            _in_flight.discard(stable_key)


def _ingest_thread(
    slack: SlackClientWrapper,
    hydra: HydraDBClient,
    channel_id: str,
    channel_name: str,
    thread_ts: str,
    triggering_event: Dict[str, Any],
    *,
    workspace_id: str = "",
) -> None:
    """
    Fetch the full thread (parent + all replies) and re-upload the
    consolidated thread doc. Safe under stable-key dedupe: the doc's
    stable_key is derived from (channel_id, thread_ts), so re-uploading
    overwrites or updates the same logical document.
    """
    # Pull the full thread from Slack so the doc reflects current state.
    replies = slack.fetch_thread_replies(
        channel_id=channel_id,
        thread_ts=thread_ts,
    )
    if not replies:
        logger.debug(
            'realtime_thread_no_replies',
            extra={'thread_ts': thread_ts},
        )
        return

    parent = replies[0]
    if is_noise(parent):
        logger.debug('realtime_thread_parent_noise')
        return

    prepared = build_thread_file(parent, replies, channel_id, channel_name, slack)
    stable_key = prepared["stable_key"]
    logger.info(
        'realtime_uploading',
        extra={
            'stable_key': stable_key,
            'doc_type': 'thread',
            'reply_count': len(replies),
        },
    )

    response = hydra.upload_knowledge([prepared])
    ok, _bad = summarize_upload_response(
        response if isinstance(response, dict) else {},
        batch_size=1,
    )
    if ok < 1:
        logger.warning(
            'realtime_upload_failed',
            extra={'stable_key': stable_key},
        )
        return

    with _state_lock:
        with IngestionState.locked(STATE_PATH) as state:
            # _record_successful_uploads is overwrite-by-stable-key inside
            # state.entries, so re-uploads correctly refresh the snippet /
            # uploaded_at / source_id without creating duplicates.
            _record_successful_uploads(state, [prepared], response or {})
            state.touch_last_ingested()

    # Phase 12: extract structured memory. Re-extracting on a thread
    # update is fine -- the persistence layer's unique key dedupes
    # repeats, and any new action items / decisions added in the
    # latest replies will surface as new rows.
    if workspace_id:
        try:
            _extract_memory_from_prepared(workspace_id, prepared)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "realtime_memory_extract_failed",
                extra={
                    "workspace_id": workspace_id,
                    "stable_key": stable_key,
                    "error": type(e).__name__,
                },
            )


def _hydradb_delete_by_stable_key(
    hydra: HydraDBClient,
    stable_key: str,
) -> bool:
    """
    Look up the HydraDB source_id for `stable_key` and delete it.
    Returns True if the delete was attempted (source_id was known),
    False if source_id was null (delete skipped).
    """
    with _state_lock:
        entry = IngestionState(STATE_PATH).get(stable_key)
    source_id = (entry or {}).get("source_id")
    if not source_id:
        logger.debug(
            "realtime_hydradb_delete_skipped_no_source_id",
            extra={"stable_key": stable_key},
        )
        return False
    hydra.delete_knowledge([source_id])
    return True


def _handle_message_changed(
    slack: SlackClientWrapper,
    hydra: HydraDBClient,
    channel_id: str,
    channel_name: str,
    event: Dict[str, Any],
    *,
    workspace_id: str = "",
) -> None:
    """
    Re-upload the new version of an edited message, then delete the stale
    HydraDB document.  The delete is intentionally deferred until after a
    successful re-upload so a transient upload failure cannot leave a data hole.
    """
    updated = event.get("message")
    if not isinstance(updated, dict):
        logger.debug("realtime_message_changed_no_message")
        return
    if is_noise(updated):
        return

    thread_ts = updated.get("thread_ts")
    if thread_ts:
        thread_stable_key = stable_key_for_thread(channel_id, thread_ts)
        with _state_lock:
            old_entry = IngestionState(STATE_PATH).get(thread_stable_key)
        old_source_id = (old_entry or {}).get("source_id")
        old_uploaded_at = (old_entry or {}).get("uploaded_at")

        _ingest_thread(slack, hydra, channel_id, channel_name, thread_ts, updated, workspace_id=workspace_id)

        with _state_lock:
            new_entry = IngestionState(STATE_PATH).get(thread_stable_key)
        if (new_entry or {}).get("uploaded_at") != old_uploaded_at and old_source_id:
            hydra.delete_knowledge([old_source_id])
    else:
        msg_ts = updated.get("ts", "")
        msg_stable_key = stable_key_for_message(channel_id, msg_ts)
        with _state_lock:
            old_entry = IngestionState(STATE_PATH).get(msg_stable_key)
        old_source_id = (old_entry or {}).get("source_id")
        old_uploaded_at = (old_entry or {}).get("uploaded_at")

        _ingest_standalone(
            slack,
            hydra,
            channel_id,
            channel_name,
            updated,
            workspace_id=workspace_id,
            force_reupload=True,
        )

        with _state_lock:
            new_entry = IngestionState(STATE_PATH).get(msg_stable_key)
        if (new_entry or {}).get("uploaded_at") != old_uploaded_at and old_source_id:
            hydra.delete_knowledge([old_source_id])


def _handle_message_deleted(
    channel_id: str,
    channel_name: str,
    event: Dict[str, Any],
    hydra: HydraDBClient,
    slack: SlackClientWrapper,
    *,
    workspace_id: str = "",
) -> None:
    """
    Handle a message_deleted event.

    - Standalone message or thread parent: delete from HydraDB and remove from state.
    - Thread reply: re-upload the thread so the stored doc reflects current state
      (Slack's replies API excludes the deleted reply), then delete the stale HydraDB
      doc using delete-after-reupload to avoid a data hole on upload failure.
    """
    deleted_ts = event.get("deleted_ts") or (event.get("previous_message") or {}).get("ts")
    if not deleted_ts:
        logger.debug("realtime_message_deleted_no_ts")
        return

    previous = event.get("previous_message") or {}
    thread_ts = previous.get("thread_ts")
    is_reply = bool(thread_ts) and thread_ts != deleted_ts

    if is_reply:
        # Re-upload the thread (Slack returns replies excluding the deleted one),
        # then delete the now-stale HydraDB doc only if re-upload succeeded.
        thread_stable_key = stable_key_for_thread(channel_id, thread_ts)
        with _state_lock:
            old_entry = IngestionState(STATE_PATH).get(thread_stable_key)
        old_source_id = (old_entry or {}).get("source_id")
        old_uploaded_at = (old_entry or {}).get("uploaded_at")

        _ingest_thread(slack, hydra, channel_id, channel_name, thread_ts, event, workspace_id=workspace_id)

        with _state_lock:
            new_entry = IngestionState(STATE_PATH).get(thread_stable_key)
        if (new_entry or {}).get("uploaded_at") != old_uploaded_at and old_source_id:
            hydra.delete_knowledge([old_source_id])
        logger.info(
            "realtime_reply_deleted_thread_refreshed",
            extra={"thread_stable_key": thread_stable_key, "deleted_ts": deleted_ts},
        )
        return

    # Standalone message or thread parent: remove the whole doc.
    stable_key = stable_key_for_message(channel_id, deleted_ts)

    # Delete from HydraDB first (while source_id is still in state).
    _hydradb_delete_by_stable_key(hydra, stable_key)

    # Remove from state regardless of whether HydraDB delete succeeded.
    with _state_lock:
        with IngestionState.locked(STATE_PATH) as state:
            removed = state.remove(stable_key)

    if removed:
        logger.info("realtime_message_deleted", extra={"stable_key": stable_key})
    else:
        logger.debug("realtime_message_deleted_not_tracked", extra={"stable_key": stable_key})


def _extract_memory_from_prepared(
    workspace_id: str,
    prepared: Dict[str, Any],
) -> None:
    """
    Shared helper for the realtime path: pull out structured memory
    from a freshly-uploaded Slack doc and persist it. Slack ts is a
    unix-seconds string; convert to ISO so the timestamptz column
    accepts it.
    """
    from datetime import datetime as _dt  # noqa: PLC0415
    from datetime import timezone as _tz

    from memory_store import extract_and_persist  # noqa: PLC0415

    stable_key = prepared.get("stable_key") or ""
    if not stable_key:
        return
    ts = prepared.get("ts") or prepared.get("timestamp")
    try:
        source_iso = _dt.fromtimestamp(float(ts), tz=_tz.utc).isoformat() if ts else None
    except (TypeError, ValueError):
        source_iso = None
    extract_and_persist(
        workspace_id=workspace_id,
        source_kind="slack",
        source_stable_key=stable_key,
        source_timestamp=source_iso,
        text=prepared.get("content") or "",
        default_owner=prepared.get("user_name") or None,
    )


# ---------------------------------------------------------------------- #
# Status snapshot for the admin endpoint
# ---------------------------------------------------------------------- #
def admin_status_snapshot(scheduler_enabled: bool) -> Dict[str, Any]:
    """
    Light read-only snapshot of ingestion-related state.
    Called from /api/admin/status; cheap (single file read).
    """
    state = IngestionState(STATE_PATH)
    return {
        "realtime_ingest_enabled": _realtime_enabled(),
        "scheduler_enabled": scheduler_enabled,
        "last_ingested_at": state.get_last_ingested_at(),
        "total_docs": state.total_docs(),
        "channels_tracked": sum(1 for k in state.channels.keys() if k != "_meta"),
    }
