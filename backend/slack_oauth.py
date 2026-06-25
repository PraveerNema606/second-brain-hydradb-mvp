"""
Slack Connect (Phase 3) — OAuth state, OAuth code exchange, and the
per-workspace ingestion runner.

This module deliberately keeps everything related to Slack-OAuth-by-
workspace in one place so Phase 3 can be reviewed (and rolled back) as
a single unit. It depends only on:
    - the Slack Web API (via slack_sdk's WebClient)
    - supabase_client.py (for installation + channel CRUD)
    - existing ingestion primitives (SlackClientWrapper, process_channel,
      upload_in_batches, IngestionState) — reused unchanged so we don't
      diverge from the prototype's ingestion behavior.

We do NOT touch ingest_slack.main() — it still works as the env-driven
prototype CLI. The per-workspace flow re-uses the same primitives with
an explicit token + channel list instead of reading them from env.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from logging_config import get_logger
from oauth_common import make_oauth_state as _core_make_state
from oauth_common import verify_oauth_state as _core_verify_state

logger = get_logger(__name__)

# Default scopes we request from Slack. Bots need channels:history /
# groups:history to read messages, *:read to enumerate channels, and
# users:read so display names can be resolved during ingestion. Kept
# minimal — no posting, no DMs.
DEFAULT_SCOPES = (
    "channels:history",
    "channels:read",
    "files:read",
    "groups:history",
    "groups:read",
    "users:read",
)


# ---------------------------------------------------------------------- #
# Env access (helpers wrapped so tests can monkeypatch without import-
# time evaluation freezing the values).
# ---------------------------------------------------------------------- #


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _client_id() -> str:
    return _env("SLACK_CLIENT_ID")


def _client_secret() -> str:
    return _env("SLACK_CLIENT_SECRET")


def _redirect_uri() -> str:
    return _env("SLACK_REDIRECT_URI")


def _state_secret() -> str:
    """
    Resolve the HMAC key used to sign OAuth state. We deliberately keep
    this separate from SUPABASE_JWT_SECRET so a leak of one doesn't
    compromise the other.
    """
    return _env("SLACK_OAUTH_STATE_SECRET")


def slack_oauth_configured() -> bool:
    """True iff all three OAuth env values are present."""
    return bool(_client_id() and _client_secret() and _redirect_uri())


# ---------------------------------------------------------------------- #
# OAuth state — HMAC-signed token binding workspace_id + user_id + nonce
# ---------------------------------------------------------------------- #
# Thin wrappers around oauth_common. The shared crypto lives there so
# a single fix applies to both Slack and Gmail; the connector-specific
# secret lookup and fail-closed guard stay here.


def make_oauth_state(workspace_id: str, user_id: str) -> str:
    """
    Build a tamper-evident state token binding this OAuth attempt to a
    specific workspace + user. Includes a short expiry so a stolen state
    can't be replayed forever, and a random nonce so two consecutive
    calls produce different tokens.

    Format: base64url(payload) "." base64url(signature)
    """
    secret = _state_secret()
    if not secret:
        # Fail closed — if we can't sign, we can't safely issue states.
        raise RuntimeError("SLACK_OAUTH_STATE_SECRET is not set.")
    return _core_make_state(secret, workspace_id, user_id)


def verify_oauth_state(state: str) -> Optional[Dict[str, Any]]:
    """
    Validate a state token returned by Slack. Returns the decoded
    payload dict on success, or None on any failure (bad format, bad
    signature, expired, missing secret). Never raises — callers branch
    on None.
    """
    return _core_verify_state(_state_secret(), state)


# ---------------------------------------------------------------------- #
# Building the Connect-Slack URL
# ---------------------------------------------------------------------- #


def build_connect_url(*, workspace_id: str, user_id: str) -> str:
    """
    Build the full Slack OAuth v2 authorize URL the frontend should
    redirect the user to. Includes a signed state binding this attempt
    to the current workspace + user.
    """
    state = make_oauth_state(workspace_id, user_id)
    qs = urlencode(
        {
            "client_id": _client_id(),
            "scope": ",".join(DEFAULT_SCOPES),
            "user_scope": "",
            "redirect_uri": _redirect_uri(),
            "state": state,
        }
    )
    return f"https://slack.com/oauth/v2/authorize?{qs}"


# ---------------------------------------------------------------------- #
# OAuth code exchange
# ---------------------------------------------------------------------- #
# Slack's v2 oauth.access response shape:
#   {
#     "ok": true,
#     "access_token": "xoxb-...",         <-- the bot token
#     "scope": "channels:history,...",
#     "bot_user_id": "U...",
#     "team": {"id": "T...", "name": "..."},
#     ...
#   }
# Documented at https://api.slack.com/methods/oauth.v2.access.


def exchange_code(code: str) -> Optional[Dict[str, Any]]:
    """
    Exchange an OAuth code for a bot token via Slack's oauth.v2.access.
    Returns the parsed JSON on success, None on any failure.
    """
    try:
        resp = requests.post(
            "https://slack.com/api/oauth.v2.access",
            data={
                "client_id": _client_id(),
                "client_secret": _client_secret(),
                "code": code,
                "redirect_uri": _redirect_uri(),
            },
            timeout=15,
        )
    except requests.RequestException as e:
        logger.warning("slack_oauth_exchange_request_failed", extra={"error": type(e).__name__})
        return None
    if not resp.ok:
        logger.warning("slack_oauth_exchange_http_error", extra={"status": resp.status_code})
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        logger.warning("slack_oauth_exchange_not_ok", extra={"error": (data or {}).get("error")})
        return None
    return data


def installation_from_oauth_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Project a Slack oauth.v2.access response into the row we store in
    public.slack_installations. Defensive: missing fields collapse to
    empty strings rather than throwing — the upsert can still proceed.
    """
    team = data.get("team") or {}
    return {
        "slack_team_id": (team.get("id") or "").strip(),
        "slack_team_name": (team.get("name") or "").strip(),
        "bot_user_id": (data.get("bot_user_id") or "").strip(),
        "bot_token": (data.get("access_token") or "").strip(),
        "scopes": (data.get("scope") or "").strip(),
    }


def revoke_bot_token(bot_token: str) -> bool:
    """
    Call Slack auth.revoke to invalidate the bot token server-side.
    Returns True if Slack confirmed revocation, False on any failure.
    Fire-and-forget: callers should proceed with local cleanup regardless.
    """
    if not bot_token:
        return False
    try:
        resp = requests.post(
            "https://slack.com/api/auth.revoke",
            headers={"Authorization": f"Bearer {bot_token}"},
            timeout=10,
        )
        data = resp.json()
        ok = bool((data or {}).get("ok"))
        if not ok:
            logger.warning(
                "slack_revoke_not_ok",
                extra={"error": (data or {}).get("error")},
            )
        return ok
    except Exception as e:  # noqa: BLE001
        logger.warning("slack_revoke_failed", extra={"error": type(e).__name__})
        return False


# ---------------------------------------------------------------------- #
# Listing channels from Slack (after Connect, so the picker can populate)
# ---------------------------------------------------------------------- #


def list_slack_channels(bot_token: str) -> List[Dict[str, Any]]:
    """
    Enumerate channels the bot can see via conversations.list. Returns
    a list of dicts ready to upsert into slack_channels. We paginate
    transparently — Slack caps each page at 1000.

    Per-channel fields:
        slack_channel_id  str
        name              str
        is_archived       bool
        is_private        bool
        member_count      int      (0 when Slack omits num_members)
        topic             str      (topic.value, empty when missing)
        purpose           str      (purpose.value, empty when missing)

    Failures return an empty list and log a warning. The caller decides
    whether to surface that to the user.
    """
    if not bot_token:
        return []
    client = WebClient(token=bot_token)
    out: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    pages = 0
    try:
        while True:
            kwargs: Dict[str, Any] = {
                "exclude_archived": False,
                "limit": 1000,
                "types": "public_channel,private_channel",
            }
            if cursor:
                kwargs["cursor"] = cursor
            resp = client.conversations_list(**kwargs)
            for ch in resp.get("channels", []) or []:
                cid = (ch.get("id") or "").strip()
                if not cid:
                    continue
                # Slack returns topic/purpose as nested {value, creator,
                # last_set} dicts. We only need the .value string.
                topic_obj = ch.get("topic") or {}
                purpose_obj = ch.get("purpose") or {}
                # `num_members` may be missing on private channels the
                # bot isn't a member of; default to 0 so we don't try to
                # insert NULL into a NOT NULL integer column.
                num_members = ch.get("num_members")
                try:
                    member_count = int(num_members) if num_members is not None else 0
                except (TypeError, ValueError):
                    member_count = 0
                out.append(
                    {
                        "slack_channel_id": cid,
                        "name": (ch.get("name") or "").strip(),
                        "is_archived": bool(ch.get("is_archived")),
                        "is_private": bool(ch.get("is_private")),
                        "member_count": member_count,
                        "topic": ((topic_obj.get("value") or "").strip() if isinstance(topic_obj, dict) else ""),
                        "purpose": ((purpose_obj.get("value") or "").strip() if isinstance(purpose_obj, dict) else ""),
                    }
                )
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or ""
            pages += 1
            # Safety: Slack workspaces with >10k channels are exotic and
            # we'd rather return a partial list than spin forever.
            if not cursor or pages >= 20:
                break
    except SlackApiError as e:
        logger.warning("slack_conversations_list_failed", extra={"error": str(e)})
        return out
    return out


# ---------------------------------------------------------------------- #
# Per-workspace ingestion runner
# ---------------------------------------------------------------------- #
# Re-uses the existing ingestion primitives (process_channel,
# upload_in_batches, IngestionState) but threads through an explicit
# bot_token + channel id list instead of reading from env. The state
# file path is shared with the prototype on purpose — moving state into
# Supabase is explicitly out of scope for Phase 3.


def run_workspace_ingest(
    *,
    workspace_id: str,
    bot_token: str,
    channel_ids: List[str],
    channel_bot_messages: Optional[Dict[str, bool]] = None,
    hydradb_sub_tenant_id: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Run a synchronous ingestion pass for one workspace's selected
    channels. Returns a small stats dict.

    Phase 4: when `hydradb_sub_tenant_id` is provided we route all
    HydraDB uploads to that sub-tenant so workspaces are isolated.
    When omitted (legacy CLI path / older tests) we fall back to the
    HYDRADB_SUB_TENANT_ID env default. The /api/slack/ingest route
    resolves the workspace's sub-tenant via ensure_workspace_sub_tenant
    before scheduling this runner.

    Synchronous on purpose: the caller wires this into a FastAPI
    BackgroundTask so the request returns immediately and the run
    proceeds in the worker. If anything raises, we log and return a
    failure dict — never propagate to the caller.
    """
    if not bot_token or not channel_ids:
        return {
            "channels_processed": 0,
            "files_prepared": 0,
            "successes": 0,
            "failures": 0,
            "skipped": 0,
        }

    # Lazy imports so a missing slack_sdk dep at startup doesn't tank
    # the whole module (and so the test suite can monkeypatch).
    from hydradb_client import HydraDBClient
    from ingestion.ingest_slack import (
        STATE_PATH,
        process_channel,
        upload_in_batches,
    )
    from ingestion.ingestion_state import IngestionState
    from ingestion.slack_client import SlackClientWrapper
    from supabase_client import mark_channel_bot_removed

    slack = SlackClientWrapper(token=bot_token)
    # Phase 4: route uploads to the workspace's HydraDB sub-tenant.
    # Falling back to the env default would silently leak this
    # workspace's documents into another tenant — surface that as a
    # log warning so an operator can see the misconfiguration.
    if hydradb_sub_tenant_id:
        hydra = HydraDBClient(sub_tenant_id=hydradb_sub_tenant_id)
    else:
        logger.warning(
            "workspace_ingest_no_sub_tenant",
            extra={"workspace_id": workspace_id},
        )
        hydra = HydraDBClient()
    state = IngestionState(STATE_PATH)

    total_files = 0
    total_success = 0
    total_failure = 0
    total_skipped = 0
    processed = 0

    # Phase 7: per-channel retry + dead-letter. process_channel +
    # upload_in_batches both touch the network; transient failures
    # deserve a second attempt. Permanent failures emit dead_letter
    # so an operator can replay them later.
    from observability import emit_dead_letter  # noqa: PLC0415
    from retry import retry_with_backoff  # noqa: PLC0415

    for channel_id in channel_ids:

        def _process_one() -> Dict[str, Any]:
            include_bot = (channel_bot_messages or {}).get(channel_id, False)
            return process_channel(slack, channel_id, state, force=force, include_bot_messages=include_bot)

        def _on_giveup(err: BaseException, _channel_id: str = channel_id) -> None:
            emit_dead_letter(
                kind="slack_ingest_channel",
                workspace_id=workspace_id,
                error=err,
                context={
                    "channel_id": _channel_id,
                    "stage": "process_channel",
                },
            )

        try:
            result = retry_with_backoff(
                _process_one,
                attempts=3,
                initial_delay=1.0,
                max_delay=8.0,
                retry_on=(Exception,),
                on_giveup=_on_giveup,
                op_name="slack_ingest_channel",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "workspace_ingest_channel_error",
                extra={
                    "workspace_id": workspace_id,
                    "channel_id": channel_id,
                    "error": type(e).__name__,
                },
            )
            total_failure += 1
            # Only update bot_removed if we know the cause; leave the flag
            # unchanged for generic network failures so a prior removal
            # notice isn't silently cleared.
            if channel_id in slack.bot_removed_channels:
                mark_channel_bot_removed(
                    workspace_id=workspace_id,
                    channel_id=channel_id,
                    removed=True,
                )
            continue

        # After every successful attempt: set or clear bot_removed.
        mark_channel_bot_removed(
            workspace_id=workspace_id,
            channel_id=channel_id,
            removed=(channel_id in slack.bot_removed_channels),
        )
        processed += 1
        total_files += len(result.get("files") or [])
        total_skipped += result.get("skipped_count", 0)

        files = result.get("files") or []
        if not files:
            newest = result.get("newest_ts_seen")
            if newest:
                state.set_last_synced_ts(result["channel_id"], newest)
                state.save_locked()
            continue

        # Phase 7: upload_in_batches has its own internal per-batch
        # retry behavior inside the HydraDB client, so we don't double-
        # wrap here. Failures are reflected in stats["failures"].
        try:
            stats = upload_in_batches(hydra, files, state)
        except Exception as e:  # noqa: BLE001
            emit_dead_letter(
                kind="slack_ingest_upload",
                workspace_id=workspace_id,
                error=e,
                context={"channel_id": channel_id, "file_count": len(files)},
            )
            total_failure += 1
            continue

        # Phase 12: extract structured memory from every Slack doc we
        # just uploaded. Defensive: any failure here MUST NOT block
        # the ingest pass -- the second-brain layer is augmenting,
        # never blocking. We piggyback on the same workspace_id +
        # source_stable_key the chunks already carry.
        try:
            from memory_store import extract_and_persist  # noqa: PLC0415

            for f in files:
                stable_key = f.get("stable_key") or ""
                if not stable_key:
                    continue
                ts = f.get("ts") or f.get("timestamp")
                # Slack ts is a unix-seconds string; convert to ISO so
                # the memory row's source_timestamp column (timestamptz)
                # accepts it cleanly. Fall back to None on parse error.
                try:
                    source_iso = (
                        datetime.fromtimestamp(
                            float(ts),
                            tz=timezone.utc,
                        ).isoformat()
                        if ts
                        else None
                    )
                except (TypeError, ValueError):
                    source_iso = None
                extract_and_persist(
                    workspace_id=workspace_id,
                    source_kind="slack",
                    source_stable_key=stable_key,
                    source_timestamp=source_iso,
                    text=f.get("content") or "",
                    default_owner=f.get("user_name") or None,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "slack_memory_extract_failed",
                extra={
                    "workspace_id": workspace_id,
                    "channel_id": channel_id,
                    "error": type(e).__name__,
                },
            )
        total_success += stats.get("successes", 0)
        total_failure += stats.get("failures", 0)

        newest = result.get("newest_ts_seen")
        if newest and stats.get("failures", 0) == 0:
            state.set_last_synced_ts(result["channel_id"], newest)
            state.save_locked()

    logger.info(
        "workspace_ingest_complete",
        extra={
            "workspace_id": workspace_id,
            "channels_processed": processed,
            "files_prepared": total_files,
            "successes": total_success,
            "failures": total_failure,
            "skipped": total_skipped,
        },
    )

    # Phase 15: emit analytics. Defensive.
    try:
        from analytics_store import emit_event  # noqa: PLC0415

        emit_event(
            workspace_id=workspace_id,
            kind="ingest_completed",
            source_kind="slack",
            success=total_failure == 0,
            payload={
                "channels_processed": processed,
                "files_prepared": total_files,
                "messages_uploaded": total_success,
                "failures": total_failure,
                "skipped": total_skipped,
            },
        )
    except Exception:  # noqa: BLE001
        pass

    return {
        "channels_processed": processed,
        "files_prepared": total_files,
        "successes": total_success,
        "failures": total_failure,
        "skipped": total_skipped,
    }
