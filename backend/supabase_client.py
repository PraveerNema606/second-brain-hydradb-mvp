"""
Server-side Supabase client for Phase 1 workspace lookups.

Uses the official `supabase` Python client with the project's
service_role key so it bypasses row-level security — needed for
membership checks on routes the user has not yet been authorized for.

Service-role credentials MUST NEVER reach the frontend. They belong in
the backend .env only.

Phase 1 surface:
    get_workspace_membership(user_id, workspace_id) -> role | None
    list_user_workspaces(user_id)                   -> list[dict]

Errors are swallowed and logged — they translate to "no membership" /
empty list on the caller side. The caller decides how to surface that
(403 for require_workspace, empty UI for the workspaces listing).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from logging_config import get_logger
from supabase import Client, create_client

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """
    Build and cache a Supabase client at first use.

    Lazy: importing this module does NOT touch the env or open a
    connection. The client is materialized only when something asks
    for it. This keeps the smoke import check happy in environments
    where Supabase credentials aren't set.
    """
    url = (os.getenv("SUPABASE_URL") or "").strip()
    key = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
    return create_client(url, key)


def reset_supabase_cache() -> None:
    """Test hook: clear the cached client so env changes take effect."""
    get_supabase.cache_clear()


def get_workspace_membership(*, user_id: str, workspace_id: str) -> Optional[str]:
    """
    Return the role ('owner' | 'admin' | 'member') if the user is a
    member of the workspace, or None otherwise.

    Any error (network, schema, etc.) is logged and treated as "no
    membership". We deliberately do NOT propagate DB error detail to
    the caller — the caller only needs the access verdict.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("workspace_members")
            .select("role")
            .eq("user_id", user_id)
            .eq("workspace_id", workspace_id)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_membership_lookup_failed',
            extra={
                'user_id': user_id,
                'workspace_id': workspace_id,
                'error': type(e).__name__,
            },
        )
        return None

    rows = getattr(resp, "data", None) or []
    if not rows:
        return None

    role = rows[0].get("role")
    # Defensive: PostgREST should always return a non-empty role since the
    # column is NOT NULL, but if a schema change ever produces a null we'd
    # rather refuse access than crash the request.
    if not isinstance(role, str) or not role.strip():
        return None
    return role


def list_user_workspaces(*, user_id: str) -> List[Dict[str, Any]]:
    """
    Return the workspaces the user belongs to, with their role.

    Shape:
        [
            {"id": str, "name": str, "slug": str, "role": str},
            ...
        ]

    Order is whatever PostgREST returns (insertion order for now). Errors
    return an empty list — the frontend handles "no workspaces yet" the
    same way regardless of cause.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("workspace_members")
            .select("role, workspace:workspaces(id, name, slug)")
            .eq("user_id", user_id)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_workspaces_failed',
            extra={'user_id': user_id, 'error': type(e).__name__},
        )
        return []

    out: List[Dict[str, Any]] = []
    for row in getattr(resp, "data", []) or []:
        ws = row.get("workspace") or {}
        ws_id = ws.get("id")
        if not ws_id:
            continue
        out.append(
            {
                "id": ws_id,
                "name": ws.get("name") or "",
                "slug": ws.get("slug") or "",
                "role": row.get("role") or "member",
            }
        )
    return out


# =====================================================================
# Phase 2: chat sessions, chat messages, saved answers.
# =====================================================================
# All of these run via the service-role client (RLS bypassed) and ALWAYS
# scope by workspace_id + user_id explicitly. The HTTP layer
# (require_workspace) has already verified the caller is a member of
# the workspace; these helpers don't re-check, they just trust the
# arguments. The database RLS policies are defense-in-depth for the day
# something other than the backend talks to the DB.


# ---------- chat_sessions --------------------------------------------- #


def list_chat_sessions(
    *,
    workspace_id: str,
    user_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Return the caller's chat sessions in this workspace, newest first.

    We scope to the caller's own sessions (user_id) rather than the whole
    workspace — chat threads are personal even inside a shared workspace.
    Saved answers (below) are shared because they're explicit bookmarks.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("chat_sessions")
            .select("id, title, created_at, updated_at")
            .eq("workspace_id", workspace_id)
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_sessions_failed',
            extra={
                'workspace_id': workspace_id,
                'user_id': user_id,
                'error': type(e).__name__,
            },
        )
        return []
    return list(getattr(resp, "data", []) or [])


def create_chat_session(
    *,
    workspace_id: str,
    user_id: str,
    title: str,
) -> Optional[Dict[str, Any]]:
    """
    Insert a new chat session row. Returns the inserted row or None on
    failure. Title is trimmed and capped here so callers don't have to.
    """
    safe_title = (title or "").strip() or "New chat"
    if len(safe_title) > 200:
        safe_title = safe_title[:200]
    try:
        client = get_supabase()
        resp = (
            client.table("chat_sessions")
            .insert(
                {
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "title": safe_title,
                }
            )
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_create_session_failed',
            extra={
                'workspace_id': workspace_id,
                'user_id': user_id,
                'error': type(e).__name__,
            },
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def get_chat_session(
    *,
    session_id: str,
    workspace_id: str,
    user_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch a single session by id, scoped to (workspace_id, user_id) so
    you can't read another user's session inside your workspace, nor
    cross workspaces. Returns None if not found / not accessible.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("chat_sessions")
            .select("id, title, created_at, updated_at")
            .eq("id", session_id)
            .eq("workspace_id", workspace_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_session_failed',
            extra={
                'session_id': session_id,
                'workspace_id': workspace_id,
                'user_id': user_id,
                'error': type(e).__name__,
            },
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


# ---------- chat_messages --------------------------------------------- #

_ALLOWED_MESSAGE_ROLES = ("user", "assistant")


def list_chat_messages(
    *,
    session_id: str,
    workspace_id: str,
    user_id: str,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """
    Return the messages in a session, oldest first.

    We re-confirm the session belongs to the caller before reading
    messages so a stolen session_id from another user can't be used to
    drain their chat. The call costs one extra round-trip but is cheap.
    """
    if not get_chat_session(
        session_id=session_id,
        workspace_id=workspace_id,
        user_id=user_id,
    ):
        return []
    try:
        client = get_supabase()
        resp = (
            client.table("chat_messages")
            .select("id, role, content, sources, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_messages_failed',
            extra={
                'session_id': session_id,
                'workspace_id': workspace_id,
                'user_id': user_id,
                'error': type(e).__name__,
            },
        )
        return []
    return list(getattr(resp, "data", []) or [])


def create_chat_message(
    *,
    session_id: str,
    workspace_id: str,
    user_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Append a message to a session. The caller must own the session — we
    verify that here so a forged session_id can't be used to write into
    another user's thread. Touches updated_at on the session so the
    sessions list re-orders correctly.
    """
    if role not in _ALLOWED_MESSAGE_ROLES:
        logger.warning(
            'supabase_create_message_bad_role',
            extra={'role': role, 'user_id': user_id},
        )
        return None
    if not get_chat_session(
        session_id=session_id,
        workspace_id=workspace_id,
        user_id=user_id,
    ):
        return None
    try:
        client = get_supabase()
        resp = (
            client.table("chat_messages")
            .insert(
                {
                    "session_id": session_id,
                    "workspace_id": workspace_id,
                    "user_id": user_id,
                    "role": role,
                    "content": content or "",
                    "sources": sources if sources is not None else None,
                }
            )
            .execute()
        )
        # Touch the session's updated_at so it floats to the top of the list.
        client.table("chat_sessions").update({"updated_at": "now()"}).eq("id", session_id).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_create_message_failed',
            extra={
                'session_id': session_id,
                'workspace_id': workspace_id,
                'user_id': user_id,
                'error': type(e).__name__,
            },
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


# ---------- saved_answers --------------------------------------------- #


def list_saved_answers(
    *,
    workspace_id: str,
    user_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    List saved answers in the workspace, newest first.

    Scoped to the caller's own saves (user_id) by default — that matches
    the previous localStorage semantics (each user only saw their own
    bookmarks). RLS would also allow other workspace members to read
    each other's saves; we deliberately filter to user_id here to avoid
    silently exposing a teammate's saved-answer list when nobody asked
    for it. Sharing-across-members is a follow-on UX decision.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("saved_answers")
            .select("id, question, answer, sources, mode, filters, debug, " "created_at")
            .eq("workspace_id", workspace_id)
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_saved_failed',
            extra={
                'workspace_id': workspace_id,
                'user_id': user_id,
                'error': type(e).__name__,
            },
        )
        return []
    return list(getattr(resp, "data", []) or [])


def create_saved_answer(
    *,
    workspace_id: str,
    user_id: str,
    question: str,
    answer: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    mode: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Insert a saved-answer row. Returns the inserted row (with its new id
    and created_at) or None on failure.
    """
    row = {
        "workspace_id": workspace_id,
        "user_id": user_id,
        "question": (question or "")[:5000],
        "answer": answer or "",
        "sources": sources if sources is not None else None,
        "mode": (mode or None),
        "filters": filters if filters is not None else None,
        "debug": debug if debug is not None else None,
    }
    try:
        client = get_supabase()
        resp = client.table("saved_answers").insert(row).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_create_saved_failed',
            extra={
                'workspace_id': workspace_id,
                'user_id': user_id,
                'error': type(e).__name__,
            },
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def delete_saved_answer(
    *,
    saved_id: str,
    workspace_id: str,
    user_id: str,
) -> bool:
    """
    Delete a saved answer the caller owns. Scoped by workspace_id +
    user_id so a stolen id from a teammate's workspace can't be used to
    delete their bookmarks. Returns True when at least one row was
    removed.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("saved_answers")
            .delete()
            .eq("id", saved_id)
            .eq("workspace_id", workspace_id)
            .eq("user_id", user_id)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_delete_saved_failed',
            extra={
                'saved_id': saved_id,
                'workspace_id': workspace_id,
                'user_id': user_id,
                'error': type(e).__name__,
            },
        )
        return False
    return bool(getattr(resp, "data", []) or [])


# =====================================================================
# Phase 13: shared_links — public read-only share URLs for saved
# answers. The `share_token` is an opaque random string minted by
# secrets.token_urlsafe(32); see main.py for the routes.
# =====================================================================


def create_share_link(
    *,
    workspace_id: str,
    saved_answer_id: str,
    created_by: str,
    share_token: str,
    expires_at: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Insert a shared-link row. Returns the inserted row or None on
    failure. The route is responsible for verifying that
    saved_answer_id belongs to workspace_id BEFORE calling this --
    we don't double-check here because the route already has the
    workspace context and the FK enforces existence.
    """
    if not workspace_id or not saved_answer_id or not created_by or not share_token:
        return None
    row: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "saved_answer_id": saved_answer_id,
        "created_by": created_by,
        "share_token": share_token,
    }
    if expires_at:
        row["expires_at"] = expires_at
    try:
        client = get_supabase()
        resp = client.table("shared_links").insert(row).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_create_share_link_failed',
            extra={
                'workspace_id': workspace_id,
                'error': type(e).__name__,
            },
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def get_share_link_by_token(*, share_token: str) -> Optional[Dict[str, Any]]:
    """
    PUBLIC read path: look up a share link by its token. Returns
    the full row or None when:
      - the token doesn't exist
      - the link has been revoked (revoked_at is not null)
      - the link has expired (expires_at < now())

    The "not found" semantics intentionally collapse all three so
    an attacker probing tokens can't distinguish "wrong" from
    "revoked" from "expired".
    """
    if not share_token:
        return None
    try:
        client = get_supabase()
        resp = client.table("shared_links").select("*").eq("share_token", share_token).limit(1).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_share_link_failed',
            extra={'error': type(e).__name__},
        )
        return None
    rows = getattr(resp, "data", None) or []
    if not rows:
        return None
    row = rows[0]
    if row.get("revoked_at"):
        return None
    expires_at = row.get("expires_at")
    if expires_at:
        # Compare as ISO strings (lexical comparison is correct for
        # ISO 8601 with the same timezone offset). For robust
        # cross-timezone handling we'd parse, but our backend always
        # writes UTC ISO strings.
        try:
            from datetime import datetime, timezone

            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > exp:
                return None
        except (TypeError, ValueError):
            # Bad timestamp on the row -> treat as expired/invalid.
            return None
    return row


def get_saved_answer(
    *,
    saved_id: str,
    workspace_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch one saved answer by id. When workspace_id is supplied the
    lookup is also bound to it (defense-in-depth). The public share
    route uses this WITHOUT a user_id filter because the share token
    itself is the credential; the route still pins workspace_id from
    the share_link row so a leaked token can never read another
    workspace's data.
    """
    if not saved_id:
        return None
    try:
        client = get_supabase()
        q = (
            client.table("saved_answers")
            .select("id, workspace_id, question, answer, sources, mode, " "created_at")
            .eq("id", saved_id)
        )
        if workspace_id:
            q = q.eq("workspace_id", workspace_id)
        resp = q.limit(1).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_saved_answer_failed',
            extra={'error': type(e).__name__},
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def list_share_links_for_workspace(
    *,
    workspace_id: str,
    user_id: Optional[str] = None,
    saved_answer_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    List shared links in a workspace, newest first. Filter by
    user_id (== created_by) when the UI wants only the caller's own
    shares; filter by saved_answer_id when the UI wants "is THIS
    answer currently shared?". Always workspace-scoped.
    """
    if not workspace_id:
        return []
    try:
        client = get_supabase()
        q = (
            client.table("shared_links")
            .select("id, share_token, saved_answer_id, created_by, " "expires_at, revoked_at, created_at")
            .eq("workspace_id", workspace_id)
        )
        if user_id:
            q = q.eq("created_by", user_id)
        if saved_answer_id:
            q = q.eq("saved_answer_id", saved_answer_id)
        resp = q.order("created_at", desc=True).limit(100).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_share_links_failed',
            extra={
                'workspace_id': workspace_id,
                'error': type(e).__name__,
            },
        )
        return []
    return list(getattr(resp, "data", []) or [])


def revoke_share_link(
    *,
    share_token: str,
    workspace_id: str,
    user_id: str,
) -> bool:
    """
    Revoke a share by setting revoked_at = now(). Scoped to
    (workspace_id, created_by) so one user can't revoke another's
    share. Returns True when at least one row was updated.

    Idempotent: revoking an already-revoked link is a no-op success.
    """
    if not share_token or not workspace_id or not user_id:
        return False
    try:
        client = get_supabase()
        resp = (
            client.table("shared_links")
            .update({"revoked_at": "now()"})
            .eq("share_token", share_token)
            .eq("workspace_id", workspace_id)
            .eq("created_by", user_id)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_revoke_share_failed',
            extra={
                'workspace_id': workspace_id,
                'error': type(e).__name__,
            },
        )
        return False
    return bool(getattr(resp, "data", []) or [])


# =====================================================================
# Phase 3: Slack installations + Slack channels.
# =====================================================================
# Bot tokens (the most sensitive piece of data this app stores so far)
# live in slack_installations.bot_token. RLS denies all authenticated
# access to that table; only the service-role client used here can read
# it. We never serialize bot_token in any HTTP response — see main.py.


# ---------- slack_installations -------------------------------------- #


def upsert_slack_installation(
    *,
    workspace_id: str,
    slack_team_id: str,
    slack_team_name: str,
    bot_user_id: str,
    bot_token: str,
    scopes: str,
    installed_by: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Insert or update the workspace's Slack installation row. Returns the
    row on success, None on failure. Idempotent on workspace_id.

    Schema notes (the production schema this targets)
    -------------------------------------------------
    The production `slack_installations` table uses the column name
    `scope` (singular) -- NOT `scopes`. Sending `scopes` makes
    PostgREST return 400 "Could not find the 'scopes' column ..." and
    the caller sees `persist_failed` on the frontend. This function
    sends `scope`, which matches the canonical Slack OAuth v2 response
    field as well.

    Also populates `installed_by` when the caller passes the verified
    Supabase user id from the OAuth state. The column is nullable so
    older callers continue to work.

    Error logging
    -------------
    A PostgREST error carries a structured body (code, message, hint,
    details). On failure we log all of those so an operator sees the
    real cause -- previously we only logged `type(e).__name__`, which
    flattened every Supabase failure to `APIError` and gave zero
    debugging signal. We DELIBERATELY DO NOT log `bot_token` or any
    other token material; only column-shape metadata flows into logs.
    """
    payload: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "slack_team_id": slack_team_id,
        "slack_team_name": slack_team_name,
        "bot_user_id": bot_user_id,
        "bot_token": bot_token,
        # Production column is `scope` (singular). Keep the public
        # Python kwarg name `scopes` for backwards compat with all
        # callers; only the persisted column name changes.
        "scope": scopes,
    }
    if installed_by:
        payload["installed_by"] = installed_by

    try:
        client = get_supabase()
        # `upsert` with on_conflict=workspace_id matches the
        # `unique (workspace_id)` constraint, so re-running Connect
        # for the same workspace updates the existing row.
        resp = client.table("slack_installations").upsert(payload, on_conflict="workspace_id").execute()
    except Exception as e:  # noqa: BLE001
        # Extract structured PostgREST fields when possible so the
        # cause is visible in production logs. .json() exists on
        # postgrest.exceptions.APIError; fall back to repr() if a
        # different exception type came through (transport errors,
        # for instance).
        err_extra: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "error": type(e).__name__,
        }
        body = None
        try:
            body = e.json()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            body = None
        if isinstance(body, dict):
            # Whitelisted keys only; the body NEVER contains the
            # bot_token (PostgREST echoes back the column name that
            # failed, not the values we sent) but we still pull just
            # these specific keys to avoid leaking unexpected fields.
            for key in ("code", "message", "hint", "details"):
                if key in body and body[key]:
                    err_extra[f"pg_{key}"] = str(body[key])[:300]
        else:
            err_extra["error_repr"] = repr(e)[:300]
        logger.warning('supabase_upsert_installation_failed', extra=err_extra)
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def get_slack_installation(
    *,
    workspace_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the installation row for a workspace, including bot_token.
    Caller MUST keep bot_token server-side — never return it from an
    API endpoint.
    """
    try:
        client = get_supabase()
        resp = client.table("slack_installations").select("*").eq("workspace_id", workspace_id).limit(1).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_installation_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def get_slack_installation_public(
    *,
    workspace_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Same as get_slack_installation but WITHOUT bot_token, suitable for
    sending to the frontend. The presence of a return value means
    "Slack is connected for this workspace" — the frontend uses that
    to swap the Connect button for the channel picker.
    """
    row = get_slack_installation(workspace_id=workspace_id)
    if not row:
        return None
    return {
        "slack_team_id": row.get("slack_team_id"),
        "slack_team_name": row.get("slack_team_name"),
        "bot_user_id": row.get("bot_user_id"),
        "scopes": row.get("scopes"),
        "connected_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def delete_slack_installation(*, workspace_id: str) -> bool:
    """
    Hard-delete the Slack installation for a workspace.
    Cascades to slack_channels via FK (on delete cascade).
    Returns True if a row was deleted, False if not found or on error.
    """
    try:
        client = get_supabase()
        resp = client.table("slack_installations").delete().eq("workspace_id", workspace_id).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "supabase_delete_slack_installation_failed",
            extra={"workspace_id": workspace_id, "error": type(e).__name__},
        )
        return False
    rows = getattr(resp, "data", None) or []
    return bool(rows)


# ---------- slack_channels ------------------------------------------- #


def upsert_slack_channels(
    *,
    workspace_id: str,
    channels: List[Dict[str, Any]],
    installation_id: Optional[str] = None,
) -> int:
    """
    Upsert a batch of channels for a workspace. Each item must have at
    least `slack_channel_id` and `name`; every other field falls back
    to a safe default so a partial payload doesn't break the insert.
    Existing rows are updated in place — selection state (`is_selected`)
    is NEVER clobbered by this function.

    Production-schema alignment
    ---------------------------
    The production `slack_channels` table carries several columns the
    old code never populated (`installation_id`, `is_private`,
    `member_count`, `topic`, `purpose`, `last_seen_at`). Of those,
    `installation_id` is a foreign key against `slack_installations.id`
    that the production schema treats as required — sending the row
    WITHOUT it triggers a NOT-NULL violation and PostgREST returns
    400. The caller MUST pass `installation_id` when it has the
    installation row in hand (the /api/slack/channels route does).
    When omitted (older tests, fresh dev DB), we just don't include
    the column; the dev schema treats it as nullable.

    Error logging
    -------------
    On failure we extract the structured PostgREST error body (code,
    message, hint, details) and log it. The previous code logged just
    `type(e).__name__`, which flattened every Supabase failure to
    `APIError` and gave operators no debugging signal. Channel IDs are
    safe to log; channel names are NOT (a renamed-for-privacy channel
    leaking via logs would be embarrassing).

    Returns the number of rows touched.
    """
    if not channels:
        return 0

    now_iso = _utc_iso_now()
    rows: List[Dict[str, Any]] = []
    for c in channels:
        cid = (c.get("slack_channel_id") or "").strip()
        if not cid:
            continue
        row: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "slack_channel_id": cid,
            "name": (c.get("name") or "").strip(),
            "is_archived": bool(c.get("is_archived")),
            "is_private": bool(c.get("is_private")),
            # Slack omits num_members for archived or no-bot-member
            # private channels. Default to 0 so we never insert NULL
            # into a NOT NULL integer column.
            "member_count": int(c.get("member_count") or 0),
            "topic": (c.get("topic") or ""),
            "purpose": (c.get("purpose") or ""),
            "last_seen_at": now_iso,
        }
        # The installation_id FK is required by the production schema
        # and unused by the dev schema. Set it only when provided so
        # we don't try to insert into a column that doesn't exist on
        # older databases.
        if installation_id:
            row["installation_id"] = installation_id
        rows.append(row)
    if not rows:
        return 0

    try:
        client = get_supabase()
        # ignore_duplicates=False so an existing row gets its
        # metadata refreshed if Slack tells us anything changed.
        resp = (
            client.table("slack_channels")
            .upsert(
                rows,
                on_conflict="workspace_id,slack_channel_id",
                ignore_duplicates=False,
            )
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        # Extract the structured PostgREST body when possible so the
        # cause (e.g. "column does not exist", "violates not-null
        # constraint") is visible in production logs. The body NEVER
        # echoes back row values, so it's safe to log.
        err_extra: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "error": type(e).__name__,
            "count": len(rows),
        }
        body = None
        try:
            body = e.json()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            body = None
        if isinstance(body, dict):
            for key in ("code", "message", "hint", "details"):
                if key in body and body[key]:
                    err_extra[f"pg_{key}"] = str(body[key])[:300]
        else:
            err_extra["error_repr"] = repr(e)[:300]
        logger.warning('supabase_upsert_channels_failed', extra=err_extra)
        return 0
    return len(getattr(resp, "data", []) or [])


def _utc_iso_now() -> str:
    """Return current UTC time as ISO 8601 -- format Postgres accepts."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def list_workspace_channels(
    *,
    workspace_id: str,
) -> List[Dict[str, Any]]:
    """
    Return every known channel for a workspace, ordered by name.
    Includes selection state so the picker can hydrate checkboxes.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("slack_channels")
            .select("slack_channel_id, name, is_selected, is_archived, bot_removed, include_bot_messages, updated_at")
            .eq("workspace_id", workspace_id)
            .order("name", desc=False)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_channels_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return []
    return list(getattr(resp, "data", []) or [])


def mark_channel_bot_removed(
    *,
    workspace_id: str,
    channel_id: str,
    removed: bool,
) -> None:
    """
    Set or clear the bot_removed flag on a single channel row.
    No-ops silently if the channel isn't tracked yet (e.g. before the
    first upsert_slack_channels call).
    """
    try:
        client = get_supabase()
        (
            client.table("slack_channels")
            .update({"bot_removed": removed})
            .eq("workspace_id", workspace_id)
            .eq("slack_channel_id", channel_id)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "supabase_mark_bot_removed_failed",
            extra={
                "workspace_id": workspace_id,
                "channel_id": channel_id,
                "removed": removed,
                "error": type(e).__name__,
            },
        )


def get_channel_ingest_settings(
    *,
    workspace_id: str,
    slack_channel_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Return {is_selected, include_bot_messages} for one channel, or None
    if the channel row doesn't exist yet.

    Used by the realtime path to check selection and bot opt-in in one
    query instead of two.
    """
    if not workspace_id or not slack_channel_id:
        return None
    try:
        client = get_supabase()
        resp = (
            client.table("slack_channels")
            .select("is_selected, include_bot_messages")
            .eq("workspace_id", workspace_id)
            .eq("slack_channel_id", slack_channel_id)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "supabase_get_channel_settings_failed",
            extra={
                "workspace_id": workspace_id,
                "slack_channel_id": slack_channel_id,
                "error": type(e).__name__,
            },
        )
        return None
    rows = getattr(resp, "data", None) or []
    if not rows:
        return None
    return {
        "is_selected": bool(rows[0].get("is_selected")),
        "include_bot_messages": bool(rows[0].get("include_bot_messages")),
    }


def list_selected_channel_settings(
    *,
    workspace_id: str,
) -> List[Dict[str, Any]]:
    """
    Return [{slack_channel_id, include_bot_messages}] for every channel
    marked is_selected=True in the workspace.

    Used by the ingest route and scheduler to build the channel list and
    per-channel bot-message settings in one query.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("slack_channels")
            .select("slack_channel_id, include_bot_messages")
            .eq("workspace_id", workspace_id)
            .eq("is_selected", True)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "supabase_list_selected_channel_settings_failed",
            extra={"workspace_id": workspace_id, "error": type(e).__name__},
        )
        return []
    out: List[Dict[str, Any]] = []
    for row in getattr(resp, "data", None) or []:
        cid = (row.get("slack_channel_id") or "").strip()
        if cid:
            out.append(
                {
                    "slack_channel_id": cid,
                    "include_bot_messages": bool(row.get("include_bot_messages")),
                }
            )
    return out


def list_selected_channel_ids(
    *,
    workspace_id: str,
) -> List[str]:
    """
    Return just the channel IDs the workspace has marked for ingestion.
    Convenience for the ingest route — no point pulling unselected rows.
    """
    try:
        client = get_supabase()
        resp = (
            client.table("slack_channels")
            .select("slack_channel_id")
            .eq("workspace_id", workspace_id)
            .eq("is_selected", True)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_selected_channels_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return []
    out: List[str] = []
    for row in getattr(resp, "data", []) or []:
        cid = (row.get("slack_channel_id") or "").strip()
        if cid:
            out.append(cid)
    return out


def set_selected_channels(
    *,
    workspace_id: str,
    selected_ids: List[str],
    bot_message_ids: Optional[List[str]] = None,
) -> bool:
    """
    Replace the selected set and, optionally, the per-channel bot-message
    opt-in. Sets is_selected=true on every row whose slack_channel_id
    appears in `selected_ids`, is_selected=false on every other row.

    When `bot_message_ids` is provided (including an empty list), the same
    pattern applies: include_bot_messages=true for channels in the list,
    false for all others. When omitted, include_bot_messages is untouched.

    Channel rows must already exist (upserted by the previous
    /api/slack/channels GET) — this function intentionally does NOT create
    missing rows. Returns True on success, False on any failure.
    """
    try:
        client = get_supabase()

        # Two queries, one for each value of is_selected. We do this
        # explicitly rather than via two SQL UPDATE statements because
        # PostgREST doesn't support a single "set X to value Y where in
        # set / else value Z" expression. Each branch is constant-time
        # in row count so even hundreds of channels stay snappy.
        if selected_ids:
            client.table("slack_channels").update({"is_selected": True}).eq("workspace_id", workspace_id).in_(
                "slack_channel_id",
                selected_ids,
            ).execute()
            # Unselect everything NOT in the set.
            client.table("slack_channels").update({"is_selected": False}).eq("workspace_id", workspace_id).not_.in_(
                "slack_channel_id",
                selected_ids,
            ).execute()
        else:
            # Empty selection — unselect everything.
            client.table("slack_channels").update({"is_selected": False}).eq("workspace_id", workspace_id).execute()

        if bot_message_ids is not None:
            # Only set include_bot_messages=True for channels that are also
            # selected, so unselected channels can never carry a stale True.
            selected_set = set(selected_ids)
            effective_bot_ids = [cid for cid in bot_message_ids if cid in selected_set]
            if effective_bot_ids:
                client.table("slack_channels").update({"include_bot_messages": True}).eq(
                    "workspace_id", workspace_id
                ).in_("slack_channel_id", effective_bot_ids).execute()
                client.table("slack_channels").update({"include_bot_messages": False}).eq(
                    "workspace_id", workspace_id
                ).not_.in_("slack_channel_id", effective_bot_ids).execute()
            else:
                client.table("slack_channels").update({"include_bot_messages": False}).eq(
                    "workspace_id", workspace_id
                ).execute()

    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_set_selected_channels_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__, 'selected_count': len(selected_ids)},
        )
        return False
    return True


# =====================================================================
# Phase 4: per-workspace HydraDB sub-tenant isolation.
# =====================================================================
# Sub-tenants are stamped at signup time by the handle_new_user trigger
# (see phase4_hydradb_workspace_isolation.sql). The helpers below
# expose that field to the Python side and provide a lazy-create path
# for any pre-Phase-4 rows that may still have a NULL/blank value
# (defense in depth — the migration backfills them, but the lazy path
# means a fresh deploy CAN'T mis-route an ingest if anything goes
# sideways).


def _derived_sub_tenant_id(workspace_id: str) -> str:
    """
    Python-side mirror of the SQL derive_hydradb_sub_tenant_id() helper.
    Stays in sync with phase4_hydradb_workspace_isolation.sql — if you
    change the format there, change it here too.
    """
    if not workspace_id:
        return ""
    return "ws_" + workspace_id.replace("-", "")[:12]


def get_workspace_sub_tenant_id(*, workspace_id: str) -> Optional[str]:
    """
    Return the workspace's hydradb_sub_tenant_id, or None if the row
    doesn't exist or has no sub-tenant. Falls through to
    ensure_workspace_sub_tenant on a blank value — see that function
    for the lazy-create behavior.
    """
    if not workspace_id:
        return None
    try:
        client = get_supabase()
        resp = client.table("workspaces").select("hydradb_sub_tenant_id").eq("id", workspace_id).limit(1).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_sub_tenant_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return None
    rows = getattr(resp, "data", None) or []
    if not rows:
        return None
    value = (rows[0].get("hydradb_sub_tenant_id") or "").strip()
    return value or None


def ensure_workspace_sub_tenant(*, workspace_id: str) -> Optional[str]:
    """
    Make sure the workspace has a hydradb_sub_tenant_id and return it.

    Order of operations:
      1. SELECT the current value.
      2. If non-empty, return it (the common case after Phase 4
         migration + trigger updates).
      3. If empty, compute the deterministic derived value, write it
         back, and return it.

    Returns None on any DB error. Callers MUST handle None — they
    typically refuse the operation rather than fall back to the global
    env sub-tenant, which would leak data across workspaces.
    """
    existing = get_workspace_sub_tenant_id(workspace_id=workspace_id)
    if existing:
        return existing

    derived = _derived_sub_tenant_id(workspace_id)
    if not derived:
        return None
    try:
        client = get_supabase()
        client.table("workspaces").update({"hydradb_sub_tenant_id": derived}).eq("id", workspace_id).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_ensure_sub_tenant_write_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__, 'derived': derived},
        )
        return None
    logger.info(
        'workspace_sub_tenant_lazy_created',
        extra={'workspace_id': workspace_id, 'sub_tenant_id': derived},
    )
    return derived


def mark_workspace_synced(
    *,
    workspace_id: str,
    sync_at: Optional[str] = None,
) -> bool:
    """
    Stamp the workspace's hydradb_last_sync_at to `sync_at` (ISO 8601
    string) or to the DB's now() if omitted. Called after a successful
    ingestion pass so operators can see which workspaces are warm.

    Returns True on success, False on any DB error. Failures are best-
    effort — a missed timestamp update never blocks ingestion.
    """
    if not workspace_id:
        return False
    payload = {
        "hydradb_last_sync_at": sync_at if sync_at else "now()",
    }
    try:
        client = get_supabase()
        client.table("workspaces").update(payload).eq(
            "id",
            workspace_id,
        ).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_mark_workspace_synced_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return False
    return True


def list_active_workspaces_with_slack() -> List[Dict[str, Any]]:
    """
    Return every workspace with hydradb_status='active' that ALSO has a
    Slack installation row. Used by the scheduler to iterate workspaces
    for the periodic ingest.

    Shape:
        [
            {
                "workspace_id":          str,
                "hydradb_sub_tenant_id": str,
                "bot_token":             str,
                "channel_ids":           List[str],   # selected only
            },
            ...
        ]

    A workspace with no selected channels is INCLUDED in the result
    (with channel_ids=[]) so the scheduler can log "skipped — no
    channels selected" rather than silently dropping it. The scheduler
    decides whether to ingest based on len(channel_ids) > 0.
    """
    try:
        client = get_supabase()
        # Pull workspaces + their installation + their selected channels
        # in three queries (PostgREST embeds would work too but the
        # service-role client makes plain queries cheaper to reason
        # about).
        ws_resp = (
            client.table("workspaces")
            .select("id, hydradb_sub_tenant_id, hydradb_status")
            .eq("hydradb_status", "active")
            .execute()
        )
        all_workspaces = getattr(ws_resp, "data", None) or []

        if not all_workspaces:
            return []

        # Pull every installation in one shot then index by workspace.
        # We deliberately fetch ALL installations (not filtered to the
        # active workspace ids above) — the Slack installations table
        # is small and a single round-trip beats N queries.
        inst_resp = client.table("slack_installations").select("id, workspace_id, bot_token").execute()
        installations: Dict[str, Dict[str, str]] = {}
        for row in getattr(inst_resp, "data", None) or []:
            wid_key = (row.get("workspace_id") or "").strip()
            if wid_key:
                installations[wid_key] = {
                    "bot_token": (row.get("bot_token") or "").strip(),
                    "installation_id": (row.get("id") or "").strip(),
                }

        # Same for the selected channels (include include_bot_messages for
        # the ingestion runner to pass through to process_channel).
        chan_resp = (
            client.table("slack_channels")
            .select("workspace_id, slack_channel_id, include_bot_messages")
            .eq("is_selected", True)
            .execute()
        )
        channels_by_ws: Dict[str, List[str]] = {}
        bot_messages_by_ws: Dict[str, Dict[str, bool]] = {}
        for row in getattr(chan_resp, "data", None) or []:
            ws = row.get("workspace_id") or ""
            cid = (row.get("slack_channel_id") or "").strip()
            if not ws or not cid:
                continue
            channels_by_ws.setdefault(ws, []).append(cid)
            bot_messages_by_ws.setdefault(ws, {})[cid] = bool(row.get("include_bot_messages"))
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_active_workspaces_failed',
            extra={'error': type(e).__name__},
        )
        return []

    out: List[Dict[str, Any]] = []
    for ws in all_workspaces:
        wid = ws.get("id")
        install = installations.get(wid) or {}
        token = install.get("bot_token") or ""
        install_id = install.get("installation_id") or ""
        sub_tenant = (ws.get("hydradb_sub_tenant_id") or "").strip()
        if not wid or not token or not sub_tenant:
            # Skip workspaces without Slack connected or without a
            # materialized sub-tenant. The scheduler doesn't need to
            # know they exist.
            continue
        out.append(
            {
                "workspace_id": wid,
                "hydradb_sub_tenant_id": sub_tenant,
                "bot_token": token,
                "installation_id": install_id,
                "channel_ids": channels_by_ws.get(wid, []),
                "channel_bot_messages": bot_messages_by_ws.get(wid, {}),
            }
        )
    return out


# =====================================================================
# Phase 11: scheduled Gmail ingestion.
# =====================================================================
# Mirrors list_active_workspaces_with_slack() in spirit but for Gmail.
# One workspace can have MULTIPLE Gmail connections (personal mailbox
# + shared inbox + ...), so the shape is one row per (workspace,
# connection) instead of one row per workspace -- the scheduler
# iterates connections, not workspaces, so each connection's failures
# stay isolated.


def list_active_workspaces_with_gmail() -> List[Dict[str, Any]]:
    """
    Return one row per active (workspace, Gmail connection) pair that
    has BOTH a connection AND at least one selected label. Used by the
    scheduler for the periodic Gmail sweep.

    Shape:
        [
            {
                "workspace_id":          str,
                "hydradb_sub_tenant_id": str,
                "connection": {                # full row WITH tokens
                    "id":             str,
                    "email":          str,
                    "access_token":   str,     # may be expired -- _authed_request refreshes
                    "refresh_token":  str,
                    ...
                },
                "selected_label_ids":    List[str],
            },
            ...
        ]

    Connections with no selected labels are NOT returned -- there's
    nothing for the scheduler to do for them. Connections that don't
    join to an active workspace with a materialized sub_tenant_id are
    also dropped (same defensive policy as the Slack version: we
    never route to the global HydraDB bucket).
    """
    try:
        client = get_supabase()

        # Step 1: active workspaces -> sub_tenant_id.
        ws_resp = (
            client.table("workspaces")
            .select("id, hydradb_sub_tenant_id, hydradb_status")
            .eq("hydradb_status", "active")
            .execute()
        )
        all_workspaces = getattr(ws_resp, "data", None) or []
        if not all_workspaces:
            return []

        active_ws: Dict[str, str] = {}
        for ws in all_workspaces:
            wid = ws.get("id")
            sub_tenant = (ws.get("hydradb_sub_tenant_id") or "").strip()
            if wid and sub_tenant:
                active_ws[wid] = sub_tenant

        if not active_ws:
            return []

        # Step 2: all Gmail connections (tokens included -- this is
        # service-role-only code, never reachable from a user request).
        # Same shape get_gmail_connection returns.
        conn_resp = client.table("gmail_connections").select("*").execute()
        connections = getattr(conn_resp, "data", None) or []
        if not connections:
            return []

        # Step 3: selected labels per connection.
        label_resp = (
            client.table("gmail_labels").select("gmail_connection_id, label_id").eq("is_selected", True).execute()
        )
        labels_by_conn: Dict[str, List[str]] = {}
        for row in getattr(label_resp, "data", None) or []:
            cid = row.get("gmail_connection_id") or ""
            lid = (row.get("label_id") or "").strip()
            if cid and lid:
                labels_by_conn.setdefault(cid, []).append(lid)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_active_workspaces_with_gmail_failed',
            extra={'error': type(e).__name__},
        )
        return []

    out: List[Dict[str, Any]] = []
    for conn in connections:
        wid = conn.get("workspace_id")
        if not wid or wid not in active_ws:
            continue
        cid = conn.get("id")
        if not cid:
            continue
        labels = labels_by_conn.get(cid, [])
        if not labels:
            # No selected labels -- nothing to sync. Skip silently so
            # the scheduler doesn't log "skipped" for every fresh
            # Connect that hasn't picked any labels yet.
            continue
        out.append(
            {
                "workspace_id": wid,
                "hydradb_sub_tenant_id": active_ws[wid],
                "connection": conn,
                "selected_label_ids": labels,
            }
        )
    return out


def get_gmail_ingestion_state_map(
    *,
    workspace_id: str,
    gmail_connection_id: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Return the per-label sync watermark map for one (workspace,
    connection). Shape:

        { label_id: {"last_history_id": str|None, "last_synced_at": str|None}, ... }

    Empty dict when nothing is recorded yet (first sync ever). The
    runner reads this to decide between incremental and full sync per
    label.
    """
    if not workspace_id or not gmail_connection_id:
        return {}
    try:
        client = get_supabase()
        resp = (
            client.table("gmail_ingestion_state")
            .select("label_id, last_history_id, last_synced_at")
            .eq("workspace_id", workspace_id)
            .eq("gmail_connection_id", gmail_connection_id)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_gmail_ingestion_state_failed',
            extra={
                'workspace_id': workspace_id,
                'connection_id': gmail_connection_id,
                'error': type(e).__name__,
            },
        )
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in getattr(resp, "data", None) or []:
        lid = (row.get("label_id") or "").strip()
        if not lid:
            continue
        out[lid] = {
            "last_history_id": row.get("last_history_id"),
            "last_synced_at": row.get("last_synced_at"),
        }
    return out


def get_gmail_connection_sync_summary(
    *,
    workspace_id: str,
    gmail_connection_id: str,
) -> Dict[str, Any]:
    """
    Lightweight UI projection: for one connection, return
        {"last_synced_at": <max across labels or None>,
         "labels_synced":  <count of labels with a non-null last_synced_at>}

    Safe to surface publicly -- contains no tokens, no email body,
    no message ids.
    """
    state = get_gmail_ingestion_state_map(
        workspace_id=workspace_id,
        gmail_connection_id=gmail_connection_id,
    )
    last: Optional[str] = None
    synced_count = 0
    for _lid, row in state.items():
        ts = row.get("last_synced_at")
        if ts:
            synced_count += 1
            if last is None or str(ts) > str(last):
                last = ts
    return {"last_synced_at": last, "labels_synced": synced_count}


# =====================================================================
# Phase 5: workspace-aware realtime Slack Events ingestion.
# =====================================================================
# Two helpers the /slack/events handler needs:
#
#   1. get_slack_installation_by_team_id — given Slack's team_id from
#      the event payload, find which workspace owns that Slack
#      installation. This is how we map a webhook hit to a workspace
#      WITHOUT trusting any client-supplied workspace id.
#
#   2. is_channel_selected_for_workspace — given (workspace_id,
#      channel_id), return True iff that channel is marked is_selected
#      in slack_channels. Lets us drop events from channels the user
#      hasn't opted into without an extra round-trip per event.


def get_slack_installation_by_team_id(
    *,
    slack_team_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the installation row for a Slack team, including bot_token.
    Used by the realtime webhook to map team_id -> workspace_id.

    Returns None if no installation matches — the caller treats that
    as "Slack workspace not connected here" and silently drops the
    event (Slack apps can be installed in multiple Slack workspaces;
    we just ignore events from teams we don't know about).
    """
    if not slack_team_id:
        return None
    try:
        client = get_supabase()
        resp = client.table("slack_installations").select("*").eq("slack_team_id", slack_team_id).limit(1).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_installation_by_team_failed',
            extra={'slack_team_id': slack_team_id, 'error': type(e).__name__},
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def is_channel_selected_for_workspace(
    *,
    workspace_id: str,
    slack_channel_id: str,
) -> bool:
    """
    True iff slack_channels has a row for (workspace_id, channel_id)
    with is_selected=True.

    A missing row (i.e. the user has never refreshed channels, or this
    is a brand-new channel) reads as "not selected" so we err on the
    side of NOT ingesting random channels. Once the user opens the
    Slack settings panel the channel list refreshes and they can opt
    it in.
    """
    if not workspace_id or not slack_channel_id:
        return False
    try:
        client = get_supabase()
        resp = (
            client.table("slack_channels")
            .select("is_selected")
            .eq("workspace_id", workspace_id)
            .eq("slack_channel_id", slack_channel_id)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_is_channel_selected_failed',
            extra={'workspace_id': workspace_id, 'slack_channel_id': slack_channel_id, 'error': type(e).__name__},
        )
        return False
    rows = getattr(resp, "data", None) or []
    if not rows:
        return False
    return bool(rows[0].get("is_selected"))


# =====================================================================
# Phase 7: durable Slack event dedupe.
# =====================================================================
# Backed by public.slack_event_seen (see phase7_production_hardening.sql).
# claim_slack_event_id returns True on the FIRST claim and False on
# duplicates -- the dedupe primitive for /slack/events.
#
# We use insert-on-conflict-do-nothing as the atomicity primitive.
# Postgres serializes the constraint check, so the first insert
# returns a row; a duplicate inserted by a concurrent worker conflicts
# and returns no row. The caller branches on that.


def claim_slack_event_id(
    *,
    event_id: str,
    workspace_id: Optional[str] = None,
) -> bool:
    """
    Atomically claim a Slack event_id for processing.

    Returns:
        True  -- this caller is the FIRST to see this event_id;
                 caller should proceed to process the event.
        False -- the event_id was already claimed by another delivery
                 (a Slack retry, or another worker handled it);
                 caller should drop the event.

    Failures fall back to returning True (i.e. "process the event"),
    which means a DB outage degrades us to the previous in-memory
    behavior rather than dropping all webhook deliveries.
    """
    if not event_id:
        return False

    payload: Dict[str, Any] = {"event_id": event_id}
    if workspace_id:
        payload["workspace_id"] = workspace_id

    try:
        client = get_supabase()
        resp = (
            client.table("slack_event_seen").insert(payload, returning="representation")
            # ignore_duplicates=True -> ON CONFLICT DO NOTHING.
            # When the row already exists, .data comes back as an
            # empty list and we treat that as "duplicate".
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        # Likely a unique-constraint violation surfaced as a generic
        # exception by the supabase client. We can't tell apart "real
        # DB outage" from "duplicate row" here, so we fail OPEN
        # (return True) only when we can't confirm dedupe -- the
        # downside (processing a duplicate during a Supabase outage)
        # is less bad than the upside (degrading to in-memory behavior
        # rather than dropping events). The "duplicate" path emits a
        # specific event so we can spot it in logs.
        error_name = type(e).__name__
        error_msg = str(e).lower()
        if "duplicate" in error_msg or "23505" in error_msg or "unique" in error_msg:
            logger.debug(
                'supabase_claim_event_duplicate',
                extra={'event_id': event_id},
            )
            return False
        logger.warning(
            'supabase_claim_event_failed',
            extra={'event_id': event_id, 'error': error_name},
        )
        return True

    rows = getattr(resp, "data", None) or []
    return bool(rows)


def cleanup_slack_event_seen(*, retain_hours: int = 24) -> int:
    """
    Drop dedupe rows older than `retain_hours`. Call from a periodic
    job (or ad-hoc); we don't auto-cleanup on every request because
    the table stays small (Slack's retry window is 1h).

    Returns the number of rows removed, or 0 on error.
    """
    try:
        client = get_supabase()
        resp = client.rpc(
            "cleanup_slack_event_seen",
            {"retain_hours": int(retain_hours)},
        ).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_cleanup_event_seen_failed',
            extra={'error': type(e).__name__},
        )
        return 0
    data = getattr(resp, "data", None)
    try:
        return int(data) if data is not None else 0
    except (TypeError, ValueError):
        return 0


# =====================================================================
# Phase 8: Gmail connector helpers.
# =====================================================================
# Mirror the Slack helpers above (upsert/get installation, label CRUD,
# ingestion-state bookkeeping). Two variants of "read a connection":
#
#   get_gmail_connection         — full row INCLUDING tokens. Internal
#                                  use only (gmail_oauth.py needs them).
#   get_gmail_connection_public  — projection WITHOUT tokens. Safe to
#                                  return from API routes.
#
# Token fields never leave the backend in any response body.


_GMAIL_PUBLIC_FIELDS = (
    "id",
    "workspace_id",
    "google_user_id",
    "email",
    "scopes",
    "status",
    "created_at",
    "updated_at",
    "token_expiry",
)


def _gmail_public_projection(row: Dict[str, Any]) -> Dict[str, Any]:
    """Strip token fields. Used by every API-facing helper."""
    out = {k: row.get(k) for k in _GMAIL_PUBLIC_FIELDS}
    # `connected_at` is a nicer name for the frontend than the
    # generic created_at; keep both so callers don't need to map.
    out["connected_at"] = row.get("created_at")
    return out


def upsert_gmail_connection(
    *,
    workspace_id: str,
    google_user_id: str,
    email: str,
    access_token: str,
    refresh_token: str,
    token_expiry: Optional[str],
    scopes: str,
    status: str = "active",
) -> Optional[Dict[str, Any]]:
    """
    Insert or update a Gmail connection. Returns the full row
    (INCLUDING tokens) so the caller can use it for the first ingest
    without an extra read.

    Reconnecting the same Google account in the same workspace updates
    in place via the (workspace_id, google_user_id) UNIQUE constraint.

    Notes:
      - Google only re-issues a refresh_token when prompt=consent was
        passed AND the user hasn't been here recently. If `refresh_token`
        is blank on update, preserve the previously-stored one rather
        than overwriting with empty.
    """
    if not workspace_id or not google_user_id:
        return None

    # Read the existing row (if any) so we can preserve refresh_token
    # when Google didn't send a fresh one on reconnect.
    existing: Optional[Dict[str, Any]] = None
    try:
        client = get_supabase()
        sel = (
            client.table("gmail_connections")
            .select("*")
            .eq("workspace_id", workspace_id)
            .eq("google_user_id", google_user_id)
            .limit(1)
            .execute()
        )
        rows = getattr(sel, "data", None) or []
        existing = rows[0] if rows else None
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_gmail_connection_for_upsert_failed',
            extra={'error': type(e).__name__},
        )

    effective_refresh = (refresh_token or "").strip()
    if not effective_refresh and existing:
        effective_refresh = (existing.get("refresh_token") or "").strip()

    payload: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "google_user_id": google_user_id,
        "email": email or "",
        "access_token": access_token or "",
        "refresh_token": effective_refresh,
        "scopes": scopes or "",
        "status": status,
    }
    if token_expiry:
        payload["token_expiry"] = token_expiry

    try:
        client = get_supabase()
        resp = client.table("gmail_connections").upsert(payload, on_conflict="workspace_id,google_user_id").execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_upsert_gmail_connection_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return None

    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def get_gmail_connection(
    *,
    connection_id: str,
    workspace_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Return the full Gmail connection row, tokens included. Server-side
    callers only -- the API surface uses get_gmail_connection_public.
    """
    if not connection_id or not workspace_id:
        return None
    try:
        client = get_supabase()
        resp = (
            client.table("gmail_connections")
            .select("*")
            .eq("id", connection_id)
            .eq("workspace_id", workspace_id)
            .limit(1)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_get_gmail_connection_failed',
            extra={'connection_id': connection_id, 'error': type(e).__name__},
        )
        return None
    rows = getattr(resp, "data", None) or []
    return rows[0] if rows else None


def get_gmail_connection_public(
    *,
    connection_id: str,
    workspace_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Public projection of a Gmail connection -- safe to return from
    API routes. Token fields are stripped.
    """
    row = get_gmail_connection(
        connection_id=connection_id,
        workspace_id=workspace_id,
    )
    if not row:
        return None
    return _gmail_public_projection(row)


def list_gmail_connections_public(
    *,
    workspace_id: str,
) -> List[Dict[str, Any]]:
    """List public projections for every Gmail connection in a workspace."""
    if not workspace_id:
        return []
    try:
        client = get_supabase()
        resp = (
            client.table("gmail_connections").select("*").eq("workspace_id", workspace_id).order("created_at").execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_gmail_connections_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return []
    rows = getattr(resp, "data", None) or []
    return [_gmail_public_projection(r) for r in rows]


def delete_gmail_connection(
    *,
    connection_id: str,
    workspace_id: str,
) -> bool:
    """
    Delete a Gmail connection (cascades to labels + ingestion state via
    ON DELETE CASCADE). Returns True if a row was actually deleted.
    """
    if not connection_id or not workspace_id:
        return False
    try:
        client = get_supabase()
        resp = (
            client.table("gmail_connections")
            .delete()
            .eq("id", connection_id)
            .eq("workspace_id", workspace_id)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_delete_gmail_connection_failed',
            extra={'connection_id': connection_id, 'error': type(e).__name__},
        )
        return False
    rows = getattr(resp, "data", None) or []
    return bool(rows)


def update_gmail_connection_tokens(
    *,
    connection_id: str,
    workspace_id: str,
    access_token: str,
    token_expiry: Optional[str] = None,
) -> bool:
    """
    Persist a refreshed access_token (and its new expiry) back to the
    connection row. Called by the ingestion runner after a successful
    refresh_access_token call so the new token survives restarts.

    workspace_id is REQUIRED for defense in depth -- a buggy caller
    that somehow ended up holding another workspace's connection_id
    must not be able to overwrite that workspace's tokens.
    """
    if not connection_id or not workspace_id or not access_token:
        return False
    payload: Dict[str, Any] = {"access_token": access_token}
    if token_expiry:
        payload["token_expiry"] = token_expiry
    try:
        client = get_supabase()
        client.table("gmail_connections").update(payload).eq(
            "id",
            connection_id,
        ).eq("workspace_id", workspace_id).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_update_gmail_tokens_failed',
            extra={'connection_id': connection_id, 'error': type(e).__name__},
        )
        return False
    return True


# ---------- Gmail labels --------------------------------------------------


def upsert_gmail_labels(
    *,
    workspace_id: str,
    gmail_connection_id: str,
    labels: List[Dict[str, Any]],
) -> int:
    """
    Insert/refresh the label rows for a connection. is_selected is
    PRESERVED on update via the unique key (gmail_connection_id, label_id)
    -- the upsert specifies only label_id/name/type, never is_selected.

    Returns the number of rows we sent (best effort -- the supabase
    client doesn't always return updated_count reliably).
    """
    if not workspace_id or not gmail_connection_id or not labels:
        return 0

    rows: List[Dict[str, Any]] = []
    for raw in labels:
        lid = (raw.get("label_id") or "").strip()
        if not lid:
            continue
        rows.append(
            {
                "workspace_id": workspace_id,
                "gmail_connection_id": gmail_connection_id,
                "label_id": lid,
                "name": (raw.get("name") or "").strip(),
                "type": (raw.get("type") or "user").strip(),
            }
        )
    if not rows:
        return 0

    try:
        client = get_supabase()
        client.table("gmail_labels").upsert(
            rows,
            on_conflict="gmail_connection_id,label_id",
        ).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_upsert_gmail_labels_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return 0
    return len(rows)


def list_gmail_labels(
    *,
    workspace_id: str,
    gmail_connection_id: str,
) -> List[Dict[str, Any]]:
    """Return every label row for a connection, including is_selected."""
    if not workspace_id or not gmail_connection_id:
        return []
    try:
        client = get_supabase()
        resp = (
            client.table("gmail_labels")
            .select("label_id, name, type, is_selected")
            .eq("workspace_id", workspace_id)
            .eq("gmail_connection_id", gmail_connection_id)
            .order("name")
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_gmail_labels_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return []
    return getattr(resp, "data", None) or []


def set_selected_gmail_labels(
    *,
    workspace_id: str,
    gmail_connection_id: str,
    selected_label_ids: List[str],
) -> bool:
    """
    Replace the selected-set for a Gmail connection. Mirrors
    set_selected_channels.

    Two updates (one for True, one for False) so the database stays
    consistent with the new selection without us having to read first.
    """
    if not workspace_id or not gmail_connection_id:
        return False
    ids = [lid.strip() for lid in (selected_label_ids or []) if lid.strip()]

    try:
        client = get_supabase()
        if ids:
            client.table("gmail_labels").update(
                {"is_selected": True},
            ).eq("workspace_id", workspace_id).eq(
                "gmail_connection_id",
                gmail_connection_id,
            ).in_("label_id", ids).execute()
            client.table("gmail_labels").update(
                {"is_selected": False},
            ).eq("workspace_id", workspace_id).eq(
                "gmail_connection_id",
                gmail_connection_id,
            ).not_.in_("label_id", ids).execute()
        else:
            client.table("gmail_labels").update(
                {"is_selected": False},
            ).eq("workspace_id", workspace_id).eq(
                "gmail_connection_id",
                gmail_connection_id,
            ).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_set_selected_gmail_labels_failed',
            extra={
                'workspace_id': workspace_id,
                'error': type(e).__name__,
                'selected_count': len(ids),
            },
        )
        return False
    return True


def list_selected_gmail_label_ids(
    *,
    workspace_id: str,
    gmail_connection_id: str,
) -> List[str]:
    """Return the currently-selected label IDs for a Gmail connection."""
    if not workspace_id or not gmail_connection_id:
        return []
    try:
        client = get_supabase()
        resp = (
            client.table("gmail_labels")
            .select("label_id")
            .eq("workspace_id", workspace_id)
            .eq("gmail_connection_id", gmail_connection_id)
            .eq("is_selected", True)
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_list_selected_gmail_labels_failed',
            extra={'workspace_id': workspace_id, 'error': type(e).__name__},
        )
        return []
    rows = getattr(resp, "data", None) or []
    return [(r.get("label_id") or "").strip() for r in rows if r.get("label_id")]


# ---------- Gmail ingestion state ----------------------------------------


def upsert_gmail_ingestion_state(
    *,
    workspace_id: str,
    gmail_connection_id: str,
    label_id: str,
    last_history_id: Optional[str] = None,
) -> bool:
    """
    Stamp ingestion progress for a (connection, label). last_synced_at
    is set to now() server-side via the trigger; last_history_id is
    only written when explicitly passed (None preserves the prior value).
    """
    if not workspace_id or not gmail_connection_id or not label_id:
        return False
    payload: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "gmail_connection_id": gmail_connection_id,
        "label_id": label_id,
        "last_synced_at": "now()",
    }
    if last_history_id:
        payload["last_history_id"] = last_history_id
    try:
        client = get_supabase()
        client.table("gmail_ingestion_state").upsert(
            payload,
            on_conflict="gmail_connection_id,label_id",
        ).execute()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            'supabase_upsert_gmail_ingestion_state_failed',
            extra={
                'workspace_id': workspace_id,
                'connection_id': gmail_connection_id,
                'label_id': label_id,
                'error': type(e).__name__,
            },
        )
        return False
    return True
