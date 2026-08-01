"""
Gmail Connect (Phase 8) — OAuth state, code exchange, refresh, label /
message fetch, document builder, and the per-workspace ingestion
runner.

Single-module-per-connector mirrors the Slack module (slack_oauth.py)
on purpose:
    - same OAuth state pattern (HMAC-signed, nonce + expiry)
    - same "build_connect_url / exchange_code / run_*_ingest" surface
    - same callable signatures so the routes look symmetric

We deliberately use plain `requests` calls against the Gmail REST API
rather than google-api-python-client. That keeps the dependency
footprint minimal (no pyopenssl / grpc / oauthlib churn) and makes
tests trivial to mock — patch `requests.post` / `requests.get`.

Token security:
    - access_token + refresh_token live ONLY in gmail_connections
      (RLS denies all client reads; only the service-role backend
      can pull them).
    - Tokens are NEVER logged. Helpers redact them everywhere.
    - The frontend gets only the public projection (see
      supabase_client.get_gmail_connection_public).

Email-body privacy:
    - We log message counts, label IDs, and connection IDs.
    - We DO NOT log subjects, addresses, or body text. The dead-letter
      logger receives only counts + IDs.
"""

from __future__ import annotations

import base64
import html
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

from logging_config import get_logger
from oauth_common import make_oauth_state as _core_make_state
from oauth_common import verify_oauth_state as _core_verify_state
from observability import emit_dead_letter
from retry import retry_with_backoff

logger = get_logger(__name__)

# Minimal read-only Gmail scopes. We DO NOT request gmail.modify or
# gmail.send -- Phase 8 is read-only. openid + email + profile give us
# enough identity info to remember which Google account this is.
GMAIL_SCOPES = (
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
)


# Cap how many messages a single ingest run can pull. Defends against
# accidental whole-mailbox ingests. Operators can raise it via env.
def _max_messages_per_run() -> int:
    try:
        return max(1, int(os.getenv("GMAIL_MAX_MESSAGES_PER_RUN", "100")))
    except ValueError:
        return 100


# ---------------------------------------------------------------------- #
# Attachment ingestion caps (Phase C). All env-tunable. These bound the
# extra Gmail API calls, memory, and per-document size that attachment
# extraction can introduce, so the scheduler stays stable.
# ---------------------------------------------------------------------- #
def _max_attachments_per_email() -> int:
    """Max attachments we extract per email (excess are skipped)."""
    try:
        return max(0, int(os.getenv("GMAIL_MAX_ATTACHMENTS_PER_EMAIL", "10")))
    except ValueError:
        return 10


def _attachment_max_chars() -> int:
    """Per-attachment cap on extracted characters."""
    try:
        return max(1, int(os.getenv("GMAIL_ATTACHMENT_MAX_CHARS", "20000")))
    except ValueError:
        return 20_000


def _attachment_total_max_chars() -> int:
    """Cap on combined extracted attachment text per email document."""
    try:
        return max(1, int(os.getenv("GMAIL_ATTACHMENT_TOTAL_MAX_CHARS", "64000")))
    except ValueError:
        return 64_000


# ---------------------------------------------------------------------- #
# Env access (helpers wrapped so tests can monkeypatch fresh values)
# ---------------------------------------------------------------------- #


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _client_id() -> str:
    return _env("GMAIL_CLIENT_ID")


def _client_secret() -> str:
    return _env("GMAIL_CLIENT_SECRET")


def _redirect_uri() -> str:
    return _env("GMAIL_REDIRECT_URI")


def _state_secret() -> str:
    """
    HMAC key for OAuth state. Separate from SUPABASE_JWT_SECRET and
    SLACK_OAUTH_STATE_SECRET on purpose -- a leak of one doesn't
    compromise the others.
    """
    return _env("GMAIL_OAUTH_STATE_SECRET")


def gmail_oauth_configured() -> bool:
    """True iff all three Google OAuth env values are present."""
    return bool(_client_id() and _client_secret() and _redirect_uri())


# ---------------------------------------------------------------------- #
# OAuth state — HMAC-signed token binding workspace_id + user_id + nonce
# ---------------------------------------------------------------------- #
# Thin wrappers around oauth_common. The shared crypto lives there so
# a single fix applies to both Slack and Gmail; the connector-specific
# secret lookup and fail-closed guard stay here.


def make_oauth_state(workspace_id: str, user_id: str) -> str:
    """
    Build a tamper-evident state token for Google OAuth.

    Format: base64url(payload) "." base64url(signature)
    """
    secret = _state_secret()
    if not secret:
        raise RuntimeError("GMAIL_OAUTH_STATE_SECRET is not set.")
    return _core_make_state(secret, workspace_id, user_id)


def verify_oauth_state(state: str) -> Optional[Dict[str, Any]]:
    """
    Validate a state returned by Google. Returns the payload dict on
    success, None on any failure. Never raises -- callers branch on None.
    """
    return _core_verify_state(_state_secret(), state)


# ---------------------------------------------------------------------- #
# Connect-Gmail URL
# ---------------------------------------------------------------------- #


def build_connect_url(*, workspace_id: str, user_id: str) -> str:
    """
    Build the Google OAuth 2.0 authorize URL.

    Notes on params:
      - access_type=offline -> Google issues a refresh_token.
      - prompt=consent      -> Forces the consent screen so Google
                               re-issues the refresh_token on every
                               connect (otherwise re-connecting an
                               account returns NO refresh_token,
                               leaving us with a dead connection).
      - include_granted_scopes=true -> incremental auth, future-proof.
    """
    state = make_oauth_state(workspace_id, user_id)
    qs = urlencode(
        {
            "client_id": _client_id(),
            "redirect_uri": _redirect_uri(),
            "response_type": "code",
            "scope": " ".join(GMAIL_SCOPES),
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
            "state": state,
        }
    )
    return f"https://accounts.google.com/o/oauth2/v2/auth?{qs}"


# ---------------------------------------------------------------------- #
# OAuth code exchange + token refresh
# ---------------------------------------------------------------------- #


def exchange_code(code: str) -> Optional[Dict[str, Any]]:
    """
    Exchange an authorization code for an access/refresh token pair.

    Returns the parsed token response dict on success, None on failure.
    Never raises, never logs tokens.
    """
    try:
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": _client_id(),
                "client_secret": _client_secret(),
                "redirect_uri": _redirect_uri(),
                "grant_type": "authorization_code",
            },
            timeout=15,
        )
    except requests.RequestException as e:
        logger.warning(
            "gmail_oauth_exchange_request_failed",
            extra={"error": type(e).__name__},
        )
        return None

    if not resp.ok:
        logger.warning(
            "gmail_oauth_exchange_http_error",
            extra={"status": resp.status_code},
        )
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict) or "access_token" not in data:
        logger.warning("gmail_oauth_exchange_missing_token")
        return None
    return data


def _refresh_access_token_detailed(
    refresh_token: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Exchange a refresh_token for a fresh access_token AND classify the
    outcome so callers can react to a revoked grant vs. a transient
    failure.

    Returns a (data, outcome) tuple where:
        outcome == "ok"        -> data is the parsed token response
        outcome == "revoked"   -> the grant is permanently invalid
                                  (Google `invalid_grant`, 401/403, or a
                                  missing refresh_token). The user must
                                  reconnect.
        outcome == "transient" -> a temporary failure (network, 429, 5xx,
                                  non-JSON, or an unexpected 400). A later
                                  run may succeed.

    Never raises, never logs tokens.
    """
    if not refresh_token:
        # Nothing to refresh with -> the only fix is a reconnect.
        return None, "revoked"
    try:
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": _client_id(),
                "client_secret": _client_secret(),
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=15,
        )
    except requests.RequestException as e:
        logger.warning(
            "gmail_oauth_refresh_request_failed",
            extra={"error": type(e).__name__},
        )
        return None, "transient"

    status = resp.status_code
    if status == 400:
        # Google signals a revoked / expired / mismatched grant with a
        # 400 + {"error": "invalid_grant"}. Other 400s (e.g.
        # invalid_request) are treated as transient so a config blip
        # doesn't permanently mark a connection dead.
        error_code = ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                error_code = (body.get("error") or "").strip().lower()
        except ValueError:
            error_code = ""
        if error_code == "invalid_grant":
            logger.warning("gmail_oauth_refresh_invalid_grant")
            return None, "revoked"
        logger.warning(
            "gmail_oauth_refresh_http_error",
            extra={"status": status},
        )
        return None, "transient"
    if status in (401, 403):
        # An unauthorized refresh almost always means the grant or the
        # client authorization is gone. Treat as revoked.
        logger.warning(
            "gmail_oauth_refresh_unauthorized",
            extra={"status": status},
        )
        return None, "revoked"
    if status == 429 or 500 <= status < 600:
        logger.warning(
            "gmail_oauth_refresh_http_error",
            extra={"status": status},
        )
        return None, "transient"
    if not resp.ok:
        logger.warning(
            "gmail_oauth_refresh_http_error",
            extra={"status": status},
        )
        return None, "transient"
    try:
        data = resp.json()
    except ValueError:
        return None, "transient"
    if not isinstance(data, dict) or "access_token" not in data:
        logger.warning("gmail_oauth_refresh_missing_token")
        return None, "transient"
    return data, "ok"


def refresh_access_token(refresh_token: str) -> Optional[Dict[str, Any]]:
    """
    Exchange a refresh_token for a fresh access_token.

    Returns the parsed response (which contains a new `access_token`
    and an `expires_in`) or None on failure. Google does NOT re-issue
    a refresh_token here -- the caller keeps the existing one.

    Thin, backwards-compatible wrapper over
    `_refresh_access_token_detailed`. Callers that need to react to a
    revoked grant (e.g. _authed_request) use the detailed helper via
    `_refresh_and_mark`.
    """
    data, _outcome = _refresh_access_token_detailed(refresh_token)
    return data


def revoke_token(token: str) -> bool:
    """
    Best-effort revocation of a Google OAuth token (access or refresh).

    Called when a user disconnects a Gmail account so the stored grant
    can't be reused. Revoking a refresh_token also invalidates the
    access tokens derived from it.

    Never raises. Returns True when Google confirms the token is no
    longer valid (HTTP 200), or when Google reports it was already
    invalid (HTTP 400) -- both mean "the grant is gone", which is the
    caller's goal. Returns False on network errors or unexpected status.
    """
    if not token:
        return False
    try:
        resp = requests.post(
            "https://oauth2.googleapis.com/revoke",
            data={"token": token},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
    except requests.RequestException as e:
        logger.warning(
            "gmail_token_revoke_request_failed",
            extra={"error": type(e).__name__},
        )
        return False
    if resp.status_code in (200, 400):
        return True
    logger.warning(
        "gmail_token_revoke_http_error",
        extra={"status": resp.status_code},
    )
    return False


def fetch_user_info(access_token: str) -> Optional[Dict[str, Any]]:
    """
    Resolve the Google user's id + email using the userinfo endpoint.
    Required at callback time so we know which gmail_connections row
    to upsert into.
    """
    if not access_token:
        return None
    try:
        resp = requests.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except requests.RequestException as e:
        logger.warning(
            "gmail_userinfo_request_failed",
            extra={"error": type(e).__name__},
        )
        return None
    if not resp.ok:
        logger.warning(
            "gmail_userinfo_http_error",
            extra={"status": resp.status_code},
        )
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def installation_from_token_response(
    token_resp: Dict[str, Any],
    user_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Project a Google token-exchange response + userinfo response into
    the row shape gmail_connections expects. Missing fields collapse
    to safe defaults so the upsert can still proceed.

    expiry_iso is set when `expires_in` is present, in UTC.
    """
    expires_in = token_resp.get("expires_in")
    expiry_iso: Optional[str] = None
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        expiry = datetime.now(timezone.utc).timestamp() + int(expires_in)
        expiry_iso = datetime.fromtimestamp(
            expiry,
            tz=timezone.utc,
        ).isoformat()

    return {
        "google_user_id": (user_info.get("sub") or "").strip(),
        "email": (user_info.get("email") or "").strip(),
        "access_token": (token_resp.get("access_token") or "").strip(),
        "refresh_token": (token_resp.get("refresh_token") or "").strip(),
        "scopes": (token_resp.get("scope") or "").strip(),
        "token_expiry": expiry_iso,
    }


# ---------------------------------------------------------------------- #
# Authenticated-Gmail calls — auto-refresh on 401
# ---------------------------------------------------------------------- #
# Every helper below routes through _authed_request so a single place
# handles "access token expired -> refresh -> retry". The refresh
# updates the in-memory `access_token` on the passed connection dict
# AND returns the new value so callers can persist it back.


class GmailApiError(Exception):
    """Raised by helpers when Gmail returns a permanent error."""


def _mark_connection_status(connection: Dict[str, Any], status: str) -> None:
    """
    Persist gmail_connections.status when it changes, and mirror the new
    value onto the in-memory connection dict.

    Best-effort and idempotent:
      - If the in-memory status already equals `status`, this is a no-op
        (no DB write) -- so a healthy connection that refreshes its
        access token never incurs a status write.
      - Workspace-scoped: we pass BOTH the connection id and workspace_id
        to the writer so a status flip can never touch another
        workspace's row.
      - NEVER raises. A status-write failure must not mask the Gmail
        error that triggered it; we only update the in-memory value when
        the write actually succeeded, so a failed write is retried on the
        next call within the same run.
    """
    try:
        if (connection.get("status") or "") == status:
            return
        connection_id = (connection.get("id") or "").strip()
        workspace_id = (connection.get("workspace_id") or "").strip()
        if not connection_id or not workspace_id:
            # No identifiers to scope a write (e.g. a bare test stub).
            # Keep the in-memory value consistent for callers.
            connection["status"] = status
            return
        from supabase_client import set_gmail_connection_status  # noqa: PLC0415

        ok = set_gmail_connection_status(
            connection_id=connection_id,
            workspace_id=workspace_id,
            status=status,
        )
        if ok:
            connection["status"] = status
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "gmail_status_mark_failed",
            extra={"status": status, "error": type(e).__name__},
        )


def _refresh_and_mark(connection: Dict[str, Any]) -> str:
    """
    Refresh the access token for `connection`, updating the in-memory
    dict and the persisted connection status, then return the new
    access token.

    On success: stamps `_token_refreshed=True` (so the ingest runner
    persists the token once at end-of-run) and restores status to
    'active' if it had drifted (recovery-on-refresh).

    On failure: persists status 'revoked' (permanent / invalid_grant)
    or 'error' (transient) and raises GmailApiError so the existing
    control flow (retry layer + dead-letter) is unchanged.
    """
    data, outcome = _refresh_access_token_detailed(connection.get("refresh_token") or "")
    if outcome == "ok" and data and "access_token" in data:
        connection["access_token"] = data["access_token"]
        connection["_token_refreshed"] = True
        _mark_connection_status(connection, "active")
        return data["access_token"]
    if outcome == "revoked":
        _mark_connection_status(connection, "revoked")
        raise GmailApiError("Gmail authorization revoked (refresh failed).")
    _mark_connection_status(connection, "error")
    raise GmailApiError("Gmail token refresh failed (transient).")


def _authed_request(
    method: str,
    url: str,
    connection: Dict[str, Any],
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 15,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Make a request to Gmail. Refresh the access_token once on 401 and
    retry. Returns (json, connection) where `connection` has been
    updated with the new access_token if a refresh occurred.

    Phase 11: when a refresh happens we ALSO stamp the connection
    dict with a sentinel `_token_refreshed=True`. The ingest runner
    reads this at end-of-run and persists the new access_token back
    to gmail_connections in ONE call regardless of how many requests
    triggered refreshes. (Persisting per-request would cost a write
    on every 401, and there can be many in a row right after an
    access token expires.)

    Raises GmailApiError on persistent failure (so the ingest runner
    can dead-letter the job).
    """
    access_token = (connection.get("access_token") or "").strip()
    if not access_token:
        # No usable access token in memory; mint one before the call.
        # _refresh_and_mark classifies a revoked grant vs. a transient
        # failure, persists the connection status accordingly, and
        # raises GmailApiError on failure.
        access_token = _refresh_and_mark(connection)

    headers = {"Authorization": f"Bearer {access_token}"}
    try:
        resp = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise GmailApiError(f"Gmail HTTP failed: {type(e).__name__}")

    if resp.status_code == 401:
        # Refresh and retry exactly once. _refresh_and_mark persists the
        # revoked/error status and raises GmailApiError on failure.
        access_token = _refresh_and_mark(connection)
        headers = {"Authorization": f"Bearer {access_token}"}
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )
        except requests.RequestException as e:
            raise GmailApiError(f"Gmail HTTP retry failed: {type(e).__name__}")

    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        # Transient: the retry layer above us can re-call.
        raise GmailApiError(f"Gmail transient HTTP {resp.status_code}")
    if not resp.ok:
        raise GmailApiError(f"Gmail HTTP {resp.status_code}")

    try:
        return resp.json(), connection
    except ValueError:
        raise GmailApiError("Gmail returned non-JSON response.")


def list_labels(connection: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return every label visible to this Gmail account.

    Shape:
        [{"label_id": "Label_1", "name": "Updates", "type": "user"}, ...]
    """
    data, _conn = _authed_request(
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/labels",
        connection,
    )
    out: List[Dict[str, Any]] = []
    for row in (data or {}).get("labels") or []:
        lid = (row.get("id") or "").strip()
        if not lid:
            continue
        out.append(
            {
                "label_id": lid,
                "name": (row.get("name") or "").strip(),
                "type": (row.get("type") or "user").strip(),
            }
        )
    return out


def list_message_ids_for_label(
    connection: Dict[str, Any],
    label_id: str,
    *,
    max_results: int = 100,
) -> List[str]:
    """
    Return the most recent message IDs for a label. `max_results` is
    capped at GMAIL_MAX_MESSAGES_PER_RUN by the runner; we honor whatever
    the caller passes here so unit tests can use small numbers.
    """
    ids: List[str] = []
    page_token: Optional[str] = None
    while len(ids) < max_results:
        params: Dict[str, Any] = {
            "labelIds": label_id,
            "maxResults": min(100, max_results - len(ids)),
        }
        if page_token:
            params["pageToken"] = page_token
        data, _conn = _authed_request(
            "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            connection,
            params=params,
        )
        for row in (data or {}).get("messages") or []:
            mid = (row.get("id") or "").strip()
            if mid:
                ids.append(mid)
        page_token = (data or {}).get("nextPageToken")
        if not page_token:
            break
    return ids[:max_results]


def list_thread_ids_for_label(
    connection: Dict[str, Any],
    label_id: str,
    *,
    max_results: int = 100,
) -> List[str]:
    """
    Return the most recent THREAD ids for a label (Phase D full-sync path).

    messages.list returns {id, threadId} per message; many messages share
    a thread, so we collect threadIds in recency order and de-duplicate.
    `max_results` caps the number of distinct threads (the runner treats
    its per-run budget as a THREAD budget). Honors whatever the caller
    passes so unit tests can use small numbers.
    """
    thread_ids: List[str] = []
    seen: set = set()
    page_token: Optional[str] = None
    while len(thread_ids) < max_results:
        params: Dict[str, Any] = {
            "labelIds": label_id,
            "maxResults": 100,
        }
        if page_token:
            params["pageToken"] = page_token
        data, _conn = _authed_request(
            "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            connection,
            params=params,
        )
        for row in (data or {}).get("messages") or []:
            tid = (row.get("threadId") or "").strip()
            if tid and tid not in seen:
                seen.add(tid)
                thread_ids.append(tid)
                if len(thread_ids) >= max_results:
                    break
        page_token = (data or {}).get("nextPageToken")
        if not page_token:
            break
    return thread_ids[:max_results]


# ---------------------------------------------------------------------- #
# Incremental sync (Phase 11)
# ---------------------------------------------------------------------- #
# Two helpers wrap the Gmail history API so the ingest runner can pull
# only what changed since the last sync:
#
#   get_mailbox_profile -> users.getProfile
#       Used on the very first sync (no last_history_id yet) to seed
#       the watermark with the current high-water mark. After that we
#       just call list_history_message_ids on each subsequent run.
#
#   list_history_message_ids -> users.history.list?historyTypes=messageAdded
#       Returns the message ids that were ADDED (or labelAdded for the
#       label we're tracking) since start_history_id. Limited to the
#       given label so the deltas stay narrow. Returns a sentinel
#       {"invalidated": True} on the 404 case Google emits when the
#       watermark is older than ~7 days -- the runner then falls back
#       to a full sync and clears the watermark.
#
# Both call _authed_request and inherit the 401-refresh + 429-retry
# behavior. Neither requires new OAuth scopes -- gmail.readonly already
# covers history.list.


class GmailHistoryInvalidated(Exception):
    """Raised internally when Gmail returns 404 for a history.list call.
    The runner catches this, falls back to the listing path, and
    clears the affected label's last_history_id."""


def get_mailbox_profile(connection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch the mailbox profile. The interesting field is `historyId`,
    used to seed last_history_id for incremental sync.

    Shape: {"emailAddress": ..., "messagesTotal": int, "threadsTotal": int, "historyId": str}
    """
    data, _conn = _authed_request(
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/profile",
        connection,
    )
    return data or {}


def list_history_message_ids(
    connection: Dict[str, Any],
    *,
    start_history_id: str,
    label_id: Optional[str] = None,
    max_results: int = 100,
) -> Dict[str, Any]:
    """
    Pull the delta since `start_history_id` and return:

        {
          "message_ids":         List[str],  # added, deduped, capped at max_results
          "deleted_message_ids": List[str],  # permanently-deleted, deduped (uncapped)
          "next_history_id":     str|None,   # the new high-water mark
          "invalidated":         bool,       # True iff Gmail returned 404
        }

    `label_id` narrows the delta to one label so the runner can
    process labels independently.

    We request BOTH messageAdded and messageDeleted history types so the
    runner can ingest new mail AND remove permanently-deleted mail from
    HydraDB in the same incremental pass (`requests` encodes a list value
    as repeated `historyTypes=` query params, which Gmail accepts). A
    message id that appears in BOTH within the same window is treated as
    deleted (it is removed from `message_ids`) so we never ingest
    something the user just deleted.

    `invalidated`: Gmail garbage-collects history records after about a
    week. A `last_history_id` older than that returns 404; we surface
    that via the `invalidated` flag so the runner can fall back to a full
    sync and reset the watermark.
    """
    if not start_history_id:
        return {
            "message_ids": [],
            "deleted_message_ids": [],
            "added_thread_ids": [],
            "deleted_thread_ids": [],
            "next_history_id": None,
            "invalidated": False,
        }

    out_ids: List[str] = []
    deleted_ids: List[str] = []
    added_thread_ids: List[str] = []
    deleted_thread_ids: List[str] = []
    last_seen_history_id: Optional[str] = None
    page_token: Optional[str] = None
    seen: set = set()
    deleted_seen: set = set()
    added_thread_seen: set = set()
    deleted_thread_seen: set = set()

    while len(out_ids) < max_results:
        params: Dict[str, Any] = {
            "startHistoryId": str(start_history_id),
            "historyTypes": ["messageAdded", "messageDeleted"],
            "maxResults": min(500, max_results - len(out_ids)),
        }
        if label_id:
            params["labelId"] = label_id
        if page_token:
            params["pageToken"] = page_token
        try:
            data, _conn = _authed_request(
                "GET",
                "https://gmail.googleapis.com/gmail/v1/users/me/history",
                connection,
                params=params,
            )
        except GmailApiError as e:
            # 404 -> watermark too old. Surface invalidation so the
            # caller can fall back. We detect 404 via the GmailApiError
            # message which carries the status code; any non-404 error
            # re-raises so the retry layer can deal with it.
            msg = str(e)
            if "HTTP 404" in msg:
                return {
                    "message_ids": [],
                    "deleted_message_ids": [],
                    "added_thread_ids": [],
                    "deleted_thread_ids": [],
                    "next_history_id": None,
                    "invalidated": True,
                }
            raise

        # Track the high-water mark even when no messages came back so
        # we can advance the watermark and avoid re-scanning empty
        # ranges next time.
        new_history_id = (data or {}).get("historyId")
        if new_history_id:
            last_seen_history_id = str(new_history_id)

        for entry in (data or {}).get("history") or []:
            for added in entry.get("messagesAdded") or []:
                m = added.get("message") or {}
                mid = (m.get("id") or "").strip()
                if mid and mid not in seen:
                    seen.add(mid)
                    out_ids.append(mid)
                tid = (m.get("threadId") or "").strip()
                if tid and tid not in added_thread_seen:
                    added_thread_seen.add(tid)
                    added_thread_ids.append(tid)
            # Deletions are collected separately and are NOT bounded by
            # the added-message cap -- cleanup should be complete for the
            # pages we scan.
            for removed in entry.get("messagesDeleted") or []:
                m = removed.get("message") or {}
                mid = (m.get("id") or "").strip()
                if mid and mid not in deleted_seen:
                    deleted_seen.add(mid)
                    deleted_ids.append(mid)
                # Gmail history carries threadId on deleted messages too,
                # so the affected thread is resolvable WITHOUT re-fetching
                # the (gone) message -- this is what makes thread-keyed
                # deletion sync possible.
                tid = (m.get("threadId") or "").strip()
                if tid and tid not in deleted_thread_seen:
                    deleted_thread_seen.add(tid)
                    deleted_thread_ids.append(tid)
            if len(out_ids) >= max_results:
                break

        page_token = (data or {}).get("nextPageToken")
        if not page_token:
            break

    # If a message was added and then deleted within the same window,
    # the deletion wins -- don't ingest something the user removed.
    if deleted_seen:
        out_ids = [mid for mid in out_ids if mid not in deleted_seen]

    return {
        "message_ids": out_ids[:max_results],
        "deleted_message_ids": deleted_ids,
        "added_thread_ids": added_thread_ids,
        "deleted_thread_ids": deleted_thread_ids,
        "next_history_id": last_seen_history_id,
        "invalidated": False,
    }


def fetch_message(
    connection: Dict[str, Any],
    message_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch a single message with `format=full` so we get headers + body.
    Returns the Gmail message dict, or None on permanent failure.
    """
    try:
        data, _conn = _authed_request(
            "GET",
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            connection,
            params={"format": "full"},
        )
    except GmailApiError as e:
        logger.warning(
            "gmail_fetch_message_failed",
            extra={"message_id": message_id, "error": str(e)},
        )
        return None
    return data


def fetch_attachment_data(
    connection: Dict[str, Any],
    message_id: str,
    attachment_id: str,
) -> Optional[str]:
    """
    Fetch a single attachment's bytes (as the base64url `data` string
    Gmail returns) via users.messages.attachments.get. Used only for
    large attachments whose bytes don't arrive inline on the full
    message. Returns None on any permanent failure -- the caller treats
    that as "skip this attachment".

    Inherits the 401-refresh + 429-retry behavior of _authed_request and
    requires no new OAuth scope (gmail.readonly covers attachments.get).
    """
    if not message_id or not attachment_id:
        return None
    try:
        data, _conn = _authed_request(
            "GET",
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}/attachments/{attachment_id}",
            connection,
        )
    except GmailApiError as e:
        logger.warning(
            "gmail_fetch_attachment_failed",
            extra={"message_id": message_id, "error": str(e)},
        )
        return None
    if not isinstance(data, dict):
        return None
    return data.get("data") or None


def fetch_thread(connection: Dict[str, Any], thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a whole Gmail thread with `format=full` (users.threads.get), so a
    single call returns every message in the conversation WITH headers +
    bodies + part structure. Returns the thread dict (with `messages`), or
    None when the thread no longer exists (HTTP 404 -> fully deleted).

    A 404 is reported as None (a normal "gone" signal the materializer
    uses to delete the consolidated doc); any other error re-raises as
    GmailApiError so the retry layer / caller can handle it. Inherits the
    401-refresh + 429-retry behavior of _authed_request and needs no new
    OAuth scope (gmail.readonly covers threads.get).
    """
    if not thread_id:
        return None
    try:
        data, _conn = _authed_request(
            "GET",
            f"https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}",
            connection,
            params={"format": "full"},
        )
    except GmailApiError as e:
        if "HTTP 404" in str(e):
            return None
        raise
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------- #
# Message -> Markdown
# ---------------------------------------------------------------------- #


def _decode_b64url(s: str) -> str:
    if not s:
        return ""
    padding = "=" * (-len(s) % 4)
    try:
        return base64.urlsafe_b64decode(s + padding).decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _extract_text_from_payload(payload: Dict[str, Any]) -> str:
    """
    Walk a Gmail message payload tree and return readable body text.

    A text/plain part wins over a text/html sibling; when only HTML is
    present we strip it to text. In both cases the result is passed
    through _clean_email_text, which trims signatures, mobile footers,
    unsubscribe blocks, and legal disclaimers while preserving the real
    message body.

    Two passes through the tree so a text/plain part wins over a
    text/html sibling regardless of which appears first. Real
    multipart/alternative payloads list HTML before plain, and we
    don't want a leading html part to short-circuit the search.
    """
    if not isinstance(payload, dict):
        return ""

    plain = _find_part_text(payload, "text/plain")
    if plain:
        return _clean_email_text(plain)
    html_text = _find_part_text(payload, "text/html")
    if html_text:
        return _clean_email_text(_strip_html(html_text))
    return ""


def _find_part_text(payload: Dict[str, Any], wanted_mime: str) -> str:
    """
    Depth-first search for the first body data with mimeType == `wanted_mime`.
    Returns the decoded text, or "" if no such part exists.

    Parts that carry a non-empty `filename` are ATTACHMENTS, not the
    message body -- they're handled by the attachment pipeline, so we
    never let a (e.g.) text/plain .txt attachment masquerade as the body.
    """
    if not isinstance(payload, dict):
        return ""

    mime = (payload.get("mimeType") or "").lower()
    body = payload.get("body") or {}
    data = body.get("data") or ""
    is_attachment = bool((payload.get("filename") or "").strip())

    if mime == wanted_mime and data and not is_attachment:
        return _decode_b64url(data)

    for child in payload.get("parts") or []:
        text = _find_part_text(child, wanted_mime)
        if text:
            return text
    return ""


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_TAG_RE = re.compile(
    r"</?\s*(?:br|p|div|tr|li|ul|ol|h[1-6]|table|blockquote|hr|header|footer|section|article)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_INLINE_WS_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")


def _strip_html(raw_html: str) -> str:
    """
    Convert an HTML email body to readable plain text.

    Improved over the original tags->space collapse: we drop
    script/style blocks entirely, turn structural tags (br, p, div, tr,
    li, headings, hr, ...) into newlines so paragraph and list
    boundaries survive, strip the remaining inline tags, decode HTML
    entities, then collapse runs of intra-line whitespace WITHOUT
    destroying the line breaks. Preserving newlines is what lets
    _clean_email_text detect signature delimiters and trailing
    boilerplate.
    """
    if not raw_html:
        return ""
    s = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    s = _BLOCK_TAG_RE.sub("\n", s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = html.unescape(s)
    # Collapse spaces/tabs within lines but keep newlines.
    s = _INLINE_WS_RE.sub(" ", s)
    # Trim each line, then collapse runs of blank lines.
    lines = [ln.strip() for ln in s.split("\n")]
    s = "\n".join(lines)
    s = _MULTI_NL_RE.sub("\n", s)
    return s.strip()


# --- Email body cleanup: signatures, mobile footers, unsubscribe & legal ---
# Conservative by design: we PRESERVE real content. Cleanup either cuts
# from a recognized trailer to the end (only when the trailer sits in the
# latter half of the message, so a newsletter's top-of-body unsubscribe
# link doesn't nuke everything) or removes individual boilerplate lines.
# A final safety net returns the original text if cleanup somehow emptied
# it.
_SIG_DELIM_RE = re.compile(r"^--\s*$")

_MOBILE_FOOTER_RES = (
    re.compile(r"^sent from my\b.*", re.IGNORECASE),
    re.compile(r"^sent from (?:mail|outlook|yahoo|proton ?mail|gmail|samsung)\b.*", re.IGNORECASE),
    re.compile(r"^get outlook for\b.*", re.IGNORECASE),
    re.compile(r"^sent (?:via|from) .+ (?:app|for (?:ios|android))\b.*", re.IGNORECASE),
    re.compile(r"^download .+ (?:app|for (?:ios|android))\b.*", re.IGNORECASE),
)

_TRAILER_CUT_RES = (
    re.compile(r"\bunsubscribe\b", re.IGNORECASE),
    re.compile(r"you (?:are )?receiv(?:e|ed|ing) this (?:email|message)", re.IGNORECASE),
    re.compile(r"to (?:stop receiving|opt[\s-]?out)", re.IGNORECASE),
    re.compile(r"manage your (?:email )?(?:preferences|subscription|notifications)", re.IGNORECASE),
    re.compile(r"update your (?:email )?(?:preferences|subscription)", re.IGNORECASE),
    re.compile(
        r"this (?:e-?mail|message)(?: and any attachments)? (?:is|are|may be) (?:confidential|intended)",
        re.IGNORECASE,
    ),
    re.compile(r"confidentiality notice", re.IGNORECASE),
    re.compile(r"if you are not the intended recipient", re.IGNORECASE),
    re.compile(r"this (?:e-?mail|message) is intended (?:only |solely )?for", re.IGNORECASE),
)

# Phase B retrieval-noise patterns. Conservative: each targets lines that
# are almost always boilerplate/tracking rather than sentence content, so
# the real message survives.
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_VIEW_IN_BROWSER_RES = (
    re.compile(
        r"^view (?:this )?(?:e-?mail|message|newsletter)?\s*(?:in (?:your )?browser|online)\b.*",
        re.IGNORECASE,
    ),
    re.compile(r"^having trouble (?:reading|viewing|seeing) this(?: e-?mail| message)?\b.*", re.IGNORECASE),
    re.compile(r"^can'?t see (?:this|the) (?:e-?mail|message|images)\b.*", re.IGNORECASE),
)
# A line made up only of decorative separators (===, ---, ***, • • •, ...).
# Requires >=3 chars so the RFC-3676 "--" signature delimiter (handled
# separately) is never caught here.
_SEPARATOR_LINE_RE = re.compile(r"^[\s\-_=*~•·—–.]{3,}$")
_IMAGE_PLACEHOLDER_RE = re.compile(r"^\[(?:image|cid|logo)\b[^\]]*\]$", re.IGNORECASE)
# A line that is essentially nothing but a single URL (optionally
# angle-bracketed). Only dropped when it ALSO carries tracking markers, so
# plain content links survive.
_URL_ONLY_LINE_RE = re.compile(r"^<?https?://\S+>?$", re.IGNORECASE)
_TRACKING_MARKERS_RE = re.compile(
    r"(?:utm_[a-z]+=|/wf/click|/wf/open|list-manage\.com|mailchi(?:mp)?\.|sendgrid\.|sparkpostmail|mailgun|"
    r"cmail\d|exct\.net|/CL0/|click\.|/track(?:ing)?[/?]|/redirect[/?]|beacon|/open\.aspx|/pixel)",
    re.IGNORECASE,
)


def _clean_email_text(text: str) -> str:
    """
    Trim trailing boilerplate AND inline retrieval noise from an email
    body while preserving the real content.

    Removed, in order:
      0. Zero-width / soft-hyphen characters (globally) -- invisible
         artifacts that fragment tokens and hurt recall.
      1. RFC 3676 signature blocks (a line that is exactly "--"), cut to
         the end. Never cuts at the very first line.
      2. Unsubscribe / legal-disclaimer trailers, cut to the end -- but
         only when the marker appears in the latter half of the message,
         so a newsletter whose header carries an unsubscribe link isn't
         wiped out.
      3. Line-by-line noise: mobile footers, stray unsubscribe/legal
         lines, "view in browser" / "having trouble viewing" lines,
         decorative separator rules, "[image: ...]" placeholders, and
         standalone URL lines carrying tracking markers (utm_, click
         trackers, list-manage, ...). Plain content URLs are KEPT.

    Conservative throughout: only lines that are almost always boilerplate
    are dropped; sentence-bearing lines are preserved. A safety net
    returns the original text if cleanup would empty a non-empty body.
    Never raises.

    The email document HEADER block (Subject/From/Date/Labels/...) is built
    separately by build_email_document and is NOT touched here, so recall
    header-harvesting and memory extraction see the same contract as before.
    """
    if not text or not text.strip():
        return text or ""

    # 0. Drop invisible characters before any line analysis.
    text = _ZERO_WIDTH_RE.sub("", text)
    original_stripped = text.strip()
    if not original_stripped:
        return ""

    lines = text.split("\n")
    n = len(lines)

    # 1. Signature delimiter -> cut to end (never at line 0).
    for i in range(1, n):
        if _SIG_DELIM_RE.match(lines[i].strip()):
            lines = lines[:i]
            break
    n = len(lines)

    # 2. Trailer block (unsubscribe / legal) -> cut to end, but only when
    #    the marker is in the latter half so top-of-body links survive.
    if n >= 4:
        for i in range(1, n):
            if any(rx.search(lines[i]) for rx in _TRAILER_CUT_RES):
                if i >= (n // 2):
                    lines = lines[:i]
                    break
                # Too early to be a trailer; line-removal (below) handles it.

    # 3. Line-level removal of boilerplate + tracking noise.
    kept: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            kept.append(ln)
            continue
        if any(rx.match(s) for rx in _MOBILE_FOOTER_RES):
            continue
        if any(rx.search(s) for rx in _TRAILER_CUT_RES):
            continue
        if any(rx.match(s) for rx in _VIEW_IN_BROWSER_RES):
            continue
        if _SEPARATOR_LINE_RE.match(s):
            continue
        if _IMAGE_PLACEHOLDER_RE.match(s):
            continue
        if _URL_ONLY_LINE_RE.match(s) and _TRACKING_MARKERS_RE.search(s):
            continue
        kept.append(ln)

    cleaned = "\n".join(kept)
    cleaned = _MULTI_NL_RE.sub("\n", cleaned).strip()

    if not cleaned:
        return original_stripped
    return cleaned


def _header_value(headers: List[Dict[str, str]], name: str) -> str:
    name_lower = name.lower()
    for h in headers or []:
        if (h.get("name") or "").lower() == name_lower:
            return (h.get("value") or "").strip()
    return ""


def _gmail_ts_to_iso(value: Any) -> Optional[str]:
    """
    Phase 12: convert a Gmail email timestamp (stored as unix seconds
    by build_email_document) into an ISO 8601 string. Returns None on
    any parse failure so the persistence layer falls back to null.
    """
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(
                float(value),
                tz=timezone.utc,
            ).isoformat()
        if isinstance(value, str) and value.strip():
            return datetime.fromtimestamp(
                float(value.strip()),
                tz=timezone.utc,
            ).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None
    return None


def stable_key_for_gmail_message(message_id: str) -> str:
    """
    Stable, unique key for HydraDB dedupe. Gmail's `id` is globally
    unique across mailboxes, so we don't need to include workspace_id.

    NOTE: Phase D moved Gmail's primary unit of ingestion to the THREAD
    (stable_key_for_gmail_thread). This per-message key is retained for
    (a) the lazy migration that deletes legacy per-message docs when a
    thread is first materialized, and (b) legacy deletion cleanup.
    """
    return f"gmail:msg:{message_id}"


def stable_key_for_gmail_thread(thread_id: str) -> str:
    """
    Stable, unique key for a consolidated Gmail THREAD document (Phase D).

    Gmail's threadId is globally unique, so (like the per-message key) we
    don't fold in workspace_id. The `gmail:` prefix keeps source-kind
    classification working and `document_type` stays "email", so recall,
    ranking, recency, and the source-card harvester treat a thread doc
    exactly like a single-email doc.
    """
    return f"gmail:thread:{thread_id}"


def _gmail_internal_date_seconds(message: Dict[str, Any]) -> Optional[float]:
    """Gmail `internalDate` (ms since epoch, as a string) -> unix seconds."""
    v = message.get("internalDate") if isinstance(message, dict) else None
    try:
        return int(v) / 1000.0 if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _thread_max_chars() -> int:
    """Cap on the combined rendered text of a consolidated thread doc."""
    try:
        return max(1, int(os.getenv("GMAIL_THREAD_MAX_CHARS", "48000")))
    except ValueError:
        return 48_000


def _safe_filename_part(s: str, max_len: int = 40) -> str:
    """Filename-safe slug (matches the Slack ingestion approach)."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s or "").strip("_")
    return s[:max_len] or "x"


def _truncate(s: str, max_len: int) -> str:
    s = s or ""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _collect_attachment_parts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Walk a Gmail message payload tree and return the SUPPORTED attachment
    parts as a list of:

        {"filename", "mime_type", "size", "attachment_id"|None, "inline_data"|None}

    A part is an attachment iff it carries a non-empty `filename`.
    Unsupported types (by is_supported_attachment) are skipped here so
    the caller never even fetches their bytes.
    """
    from gmail_attachments import is_supported_attachment  # noqa: PLC0415

    out: List[Dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        filename = (node.get("filename") or "").strip()
        if filename:
            mime = (node.get("mimeType") or "").strip()
            if is_supported_attachment(filename, mime):
                body = node.get("body") or {}
                try:
                    size = int(body.get("size") or 0)
                except (TypeError, ValueError):
                    size = 0
                out.append(
                    {
                        "filename": filename,
                        "mime_type": mime,
                        "size": size,
                        "attachment_id": (body.get("attachmentId") or "").strip() or None,
                        "inline_data": body.get("data") or None,
                    }
                )
        for child in node.get("parts") or []:
            _walk(child)

    _walk(payload)
    return out


def gather_attachment_sections(
    connection: Dict[str, Any],
    message: Dict[str, Any],
    summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Extract text from a message's supported attachments and return a list
    of section dicts:

        [{"filename", "mime_type", "size", "chars", "text"}, ...]

    Best-effort and bounded:
      - Skips unsupported types and anything over the per-attachment byte
        cap (avoids loading huge blobs into memory).
      - Caps the number of attachments per email and the total extracted
        characters per email document.
      - A failure on ONE attachment (fetch error, corrupt file, empty
        extraction) is logged + counted and never aborts the others or
        the message.

    Updates `summary` counters: attachments_processed / attachments_failed
    / attachments_skipped. Privacy: logs only mime type, size, and the
    message id -- never the filename or extracted text.
    """
    from gmail_attachments import extract_text_from_attachment, max_attachment_bytes  # noqa: PLC0415

    sections: List[Dict[str, Any]] = []
    if not isinstance(message, dict):
        return sections
    message_id = (message.get("id") or "").strip()
    payload = message.get("payload") or {}
    parts = _collect_attachment_parts(payload)
    if not parts:
        return sections

    max_count = _max_attachments_per_email()
    if max_count <= 0:
        return sections
    max_bytes = max_attachment_bytes()
    per_cap = _attachment_max_chars()
    total_cap = _attachment_total_max_chars()
    used_chars = 0
    seen_keys: set = set()

    for part in parts:
        if len(sections) >= max_count:
            summary["attachments_skipped"] += 1
            continue

        # Dedupe within the email (same attachment can appear twice in
        # oddly-structured multiparts).
        dedupe_key = part.get("attachment_id") or part.get("filename")
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)

        size = part.get("size") or 0
        if size and size > max_bytes:
            summary["attachments_skipped"] += 1
            logger.info(
                "gmail_attachment_too_large",
                extra={"message_id": message_id, "mime_type": part.get("mime_type"), "size": size},
            )
            continue

        # Obtain bytes: inline data if present (no extra API call), else
        # fetch by attachment id.
        try:
            b64 = part.get("inline_data")
            if not b64 and part.get("attachment_id"):
                b64 = fetch_attachment_data(connection, message_id, part["attachment_id"])
            if not b64:
                summary["attachments_failed"] += 1
                continue
            raw = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4))
        except Exception:  # noqa: BLE001
            summary["attachments_failed"] += 1
            logger.warning(
                "gmail_attachment_fetch_failed",
                extra={"message_id": message_id, "mime_type": part.get("mime_type"), "size": size},
            )
            continue

        if len(raw) > max_bytes:
            summary["attachments_skipped"] += 1
            logger.info(
                "gmail_attachment_too_large",
                extra={"message_id": message_id, "mime_type": part.get("mime_type"), "size": len(raw)},
            )
            continue

        try:
            text = extract_text_from_attachment(part["filename"], part["mime_type"], raw)
        except Exception:  # noqa: BLE001
            text = None
        if not text or not text.strip():
            summary["attachments_failed"] += 1
            logger.warning(
                "gmail_attachment_extract_failed",
                extra={"message_id": message_id, "mime_type": part.get("mime_type"), "size": size},
            )
            continue

        text = text.strip()
        if len(text) > per_cap:
            text = text[:per_cap]
        # Enforce the per-email total budget.
        remaining_budget = total_cap - used_chars
        if remaining_budget <= 0:
            summary["attachments_skipped"] += 1
            continue
        if len(text) > remaining_budget:
            text = text[:remaining_budget]
        used_chars += len(text)

        sections.append(
            {
                "filename": part["filename"],
                "mime_type": part.get("mime_type") or "",
                "size": size,
                "chars": len(text),
                "text": text,
            }
        )
        summary["attachments_processed"] += 1

    return sections


def build_email_document(
    message: Dict[str, Any],
    connection_email: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convert a Gmail message dict into the {filename, content, stable_key,
    ...} shape HydraDBClient.upload_knowledge expects. Returns None when
    the message has no usable text (no body, no snippet, AND no
    extracted attachment text).

    Phase C: extracted attachment text (from `attachments`, produced by
    gather_attachment_sections) is appended to the SAME document under an
    "## Attachments" section, after the email body. This keeps one
    HydraDB document per email -- so document_type stays "email", the
    stable_key stays gmail:msg:<id>, recall/ranking/source-card/recency
    are unchanged, deletion sync removes attachments with their parent,
    and memory extraction sees attachment text associated with the email.

    `attachments` is an optional list of section dicts:
        {"filename", "mime_type", "size", "chars", "text"}

    DOES NOT log any header values, body text, or attachment text.
    """
    if not isinstance(message, dict):
        return None
    message_id = (message.get("id") or "").strip()
    if not message_id:
        return None

    attachments = attachments or []
    payload = message.get("payload") or {}
    headers = payload.get("headers") or []
    subject = _header_value(headers, "Subject") or "(no subject)"
    sender = _header_value(headers, "From")
    to = _header_value(headers, "To")
    cc = _header_value(headers, "Cc")
    date = _header_value(headers, "Date")
    snippet = (message.get("snippet") or "").strip()
    label_ids = message.get("labelIds") or []
    body_text = _extract_text_from_payload(payload).strip()

    has_attachment_text = any((a.get("text") or "").strip() for a in attachments)
    if not body_text and not snippet and not has_attachment_text:
        # Nothing to index; skip silently.
        return None

    stable_key = stable_key_for_gmail_message(message_id)
    # Gmail web client deep link. Always works for the mailbox owner.
    permalink = f"https://mail.google.com/mail/u/0/#all/{message_id}" if message_id else None

    # Build the header block. `Cc:` is only emitted when present so we
    # don't pollute every email doc with a blank line.
    header_lines = [
        "# Email",
        f"Source Key: {stable_key}",
        f"Message-Id: {message_id}",
        f"Mailbox: {connection_email}",
        f"Subject: {_truncate(subject, 200)}",
        f"From: {_truncate(sender, 200)}",
        f"To: {_truncate(to, 200)}",
    ]
    if cc:
        header_lines.append(f"Cc: {_truncate(cc, 200)}")
    header_lines.extend(
        [
            f"Date: {date}",
            f"Labels: {', '.join(label_ids)}",
            f"Snippet: {_truncate(snippet, 280)}",
        ]
    )
    if permalink:
        header_lines.append(f"Permalink: {permalink}")

    # Cap the body at 32k chars. Real emails rarely exceed this; if one
    # does we'd rather index a meaningful prefix than refuse the doc.
    body_for_doc = _truncate(body_text or snippet, 32_000)
    content_lines = header_lines + ["", body_for_doc]

    # Append extracted attachment text. The section markers use "##"/"###"
    # and an "Attachment:" label that deliberately does NOT collide with
    # the recall header field names (Subject:/From:/Date:/Labels:/
    # Permalink:), and "## Attachments" is not the "# Email" doc header --
    # so header harvesting (which matches the FIRST occurrence at the top)
    # is unaffected.
    attachment_meta: List[Dict[str, Any]] = []
    if attachments:
        rendered = []
        for a in attachments:
            text = (a.get("text") or "").strip()
            if not text:
                continue
            name = a.get("filename") or "attachment"
            mime = a.get("mime_type") or ""
            size = a.get("size") or 0
            rendered.append(f"### Attachment: {name} ({mime}, {size} bytes)")
            rendered.append(text)
            rendered.append("")
            attachment_meta.append(
                {
                    "filename": name,
                    "mime_type": mime,
                    "size": size,
                    "chars": len(text),
                }
            )
        if rendered:
            content_lines.append("")
            content_lines.append("## Attachments")
            content_lines.append("")
            content_lines.extend(rendered)

    content = "\n".join(content_lines)

    filename = f"gmail_{_safe_filename_part(message_id)}.md"
    return {
        "filename": filename,
        "content": content,
        "stable_key": stable_key,
        # Extra metadata that HydraDB / state.mark_uploaded carry forward.
        # We intentionally do NOT include the subject or body here --
        # only IDs, so a state.json leak doesn't expose mail content.
        "message_id": message_id,
        "document_type": "email",
        "snippet": _truncate(snippet, 280),
        "permalink": permalink,
        # Attachment provenance (counts + types + sizes only; no text).
        # Observability metadata -- recall ignores unknown doc keys.
        "attachments": attachment_meta,
    }


# ---------------------------------------------------------------------- #
# Thread-aware document construction (Phase D)
# ---------------------------------------------------------------------- #


def build_email_thread_document(
    thread_messages: List[Dict[str, Any]],
    connection_email: str,
    message_attachments: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Consolidate ALL messages of a Gmail thread into ONE document, mirroring
    the Slack thread model. Returns the {filename, content, stable_key,
    ...} dict upload_knowledge expects, or None when the thread has no
    indexable content.

    The document keeps document_type="email", a "# Email" header, and the
    Subject/From/Date/Labels/Permalink header lines (from the LATEST
    message; labels unioned across the thread), plus the gmail: stable-key
    prefix -- so recall/ranking/source-card/recency treat it exactly like a
    single-email doc and need NO changes. Each message becomes a
    "## Message from <sender> on <date>" block in chronological order, with
    its Phase C attachment sections embedded. The block markers and the
    additive "Thread:"/"Messages:" header lines deliberately avoid the
    harvester's field names, and the real header sits FIRST, so header
    harvesting (first-match-wins) is unaffected.

    `message_attachments` maps message id -> the Phase C attachment section
    list for that message.

    DOES NOT log any header values, body text, or attachment text.
    """
    message_attachments = message_attachments or {}
    msgs = [m for m in (thread_messages or []) if isinstance(m, dict) and (m.get("id") or "").strip()]
    if not msgs:
        return None

    # Chronological order by internalDate (fallback: input order via stable sort).
    msgs.sort(key=lambda m: (_gmail_internal_date_seconds(m) or 0.0))
    latest = msgs[-1]
    thread_id = (latest.get("threadId") or msgs[0].get("threadId") or "").strip()
    if not thread_id:
        return None

    latest_payload = latest.get("payload") or {}
    latest_headers = latest_payload.get("headers") or []
    subject = _header_value(latest_headers, "Subject") or "(no subject)"
    sender = _header_value(latest_headers, "From")
    to = _header_value(latest_headers, "To")
    date = _header_value(latest_headers, "Date")
    snippet = (latest.get("snippet") or "").strip()

    # Union labels across the thread (sorted for stable output) so label-
    # aware ranking (Phase B) matches if ANY message carried the label.
    label_set: set = set()
    for m in msgs:
        for lid in m.get("labelIds") or []:
            if isinstance(lid, str) and lid.strip():
                label_set.add(lid.strip())
    label_ids = sorted(label_set)

    stable_key = stable_key_for_gmail_thread(thread_id)
    permalink = f"https://mail.google.com/mail/u/0/#all/{thread_id}"
    latest_message_id = (latest.get("id") or "").strip()

    header_lines = [
        "# Email",
        f"Source Key: {stable_key}",
        f"Thread: {thread_id}",
        f"Messages: {len(msgs)}",
        f"Mailbox: {connection_email}",
        f"Subject: {_truncate(subject, 200)}",
        f"From: {_truncate(sender, 200)}",
        f"To: {_truncate(to, 200)}",
        f"Date: {date}",
        f"Labels: {', '.join(label_ids)}",
        f"Snippet: {_truncate(snippet, 280)}",
        f"Permalink: {permalink}",
    ]

    total_cap = _thread_max_chars()
    # Per-message budget with a sensible floor so short threads aren't
    # over-trimmed; the running total is still hard-capped at total_cap.
    per_msg_cap = max(2_000, total_cap // max(1, len(msgs)))
    used = 0
    body_lines: List[str] = []
    attachment_meta: List[Dict[str, Any]] = []
    indexable = False

    for m in msgs:
        if used >= total_cap:
            break
        mp = m.get("payload") or {}
        mh = mp.get("headers") or []
        m_from = _header_value(mh, "From") or "(unknown sender)"
        m_date = _header_value(mh, "Date") or ""
        m_body = _extract_text_from_payload(mp).strip()
        m_snippet = (m.get("snippet") or "").strip()
        block_text = m_body or m_snippet
        if len(block_text) > per_msg_cap:
            block_text = _truncate(block_text, per_msg_cap)

        body_lines.append("")
        body_lines.append(f"## Message from {_truncate(m_from, 200)} on {m_date}".rstrip())
        if block_text:
            remaining_total = total_cap - used
            if len(block_text) > remaining_total:
                block_text = block_text[:remaining_total]
            if block_text:
                body_lines.append(block_text)
                used += len(block_text)
                indexable = True

        # Embedded attachments for this message (Phase C format/markers).
        for a in message_attachments.get((m.get("id") or "").strip()) or []:
            if used >= total_cap:
                break
            text = (a.get("text") or "").strip()
            if not text:
                continue
            name = a.get("filename") or "attachment"
            mime = a.get("mime_type") or ""
            size = a.get("size") or 0
            remaining_total = total_cap - used
            if len(text) > remaining_total:
                text = text[:remaining_total]
            if not text:
                break
            body_lines.append(f"### Attachment: {name} ({mime}, {size} bytes)")
            body_lines.append(text)
            used += len(text)
            indexable = True
            attachment_meta.append({"filename": name, "mime_type": mime, "size": size, "chars": len(text)})

    if not indexable and not snippet:
        return None

    content = "\n".join(header_lines + body_lines)
    filename = f"gmail_thread_{_safe_filename_part(thread_id)}.md"

    return {
        "filename": filename,
        "content": content,
        "stable_key": stable_key,
        # Latest message id (for permalink/debug); the doc is keyed by
        # thread, not message.
        "message_id": latest_message_id,
        "thread_id": thread_id,
        "message_count": len(msgs),
        "document_type": "email",
        "snippet": _truncate(snippet, 280),
        "permalink": permalink,
        # Recency anchor = latest message time (unix seconds).
        "timestamp": _gmail_internal_date_seconds(latest),
        "attachments": attachment_meta,
        # Default memory owner = the latest sender.
        "from_name": sender or None,
    }


def _delete_gmail_thread(
    *,
    hydra: Any,
    workspace_id: str,
    connection_id: Optional[str],
    thread_id: str,
    summary: Dict[str, Any],
) -> None:
    """
    Remove a consolidated thread document + its derived memories when the
    whole thread is gone. Best-effort; never raises.
    """
    key = stable_key_for_gmail_thread(thread_id)
    try:
        hydra.delete_knowledge([key])
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "gmail_thread_delete_failed",
            extra={
                "workspace_id": workspace_id,
                "connection_id": connection_id,
                "thread_id": thread_id,
                "error": type(e).__name__,
            },
        )
    try:
        from memory_store import delete_memories_by_source  # noqa: PLC0415

        delete_memories_by_source(workspace_id=workspace_id, source_stable_key=key)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "gmail_thread_memory_delete_failed",
            extra={
                "workspace_id": workspace_id,
                "connection_id": connection_id,
                "thread_id": thread_id,
                "error": type(e).__name__,
            },
        )
    summary["threads_deleted"] += 1
    logger.info(
        "gmail_thread_deleted",
        extra={"workspace_id": workspace_id, "connection_id": connection_id, "thread_id": thread_id},
    )


def _materialize_gmail_thread(
    connection: Dict[str, Any],
    thread_id: str,
    *,
    hydra: Any,
    workspace_id: str,
    connection_id: Optional[str],
    connection_email: str,
    summary: Dict[str, Any],
) -> None:
    """
    (Re)build the consolidated document for ONE Gmail thread and reconcile
    HydraDB + memories. This single operation is the unit of work for adds,
    replies, re-syncs, deletions, AND lazy migration off legacy per-message
    docs:

      - Thread gone (threads.get 404 / no messages) -> delete the
        consolidated `gmail:thread:` doc + its memories.
      - Thread present -> rebuild + upload the consolidated doc, extract
        memory under the thread key, then delete any legacy
        `gmail:msg:<id>` docs (+ memories) for the thread's messages so a
        migrated thread never coexists with its old per-message docs.

    Fully defensive: never raises into the runner; every HydraDB / memory
    step is best-effort. One threads.get call returns all messages with
    payloads, so the message body path needs no per-message fetch.
    """
    thread_id = (thread_id or "").strip()
    if not thread_id:
        return

    try:
        thread = retry_with_backoff(
            fetch_thread,
            connection,
            thread_id,
            attempts=2,
            initial_delay=0.5,
            max_delay=2.0,
            retry_on=(GmailApiError,),
            op_name="gmail_fetch_thread",
        )
    except GmailApiError:
        summary["threads_failed"] += 1
        return

    messages = [m for m in ((thread or {}).get("messages") or []) if isinstance(m, dict)]
    if not thread or not messages:
        # Whole thread is gone -> remove the consolidated doc + memories.
        _delete_gmail_thread(
            hydra=hydra,
            workspace_id=workspace_id,
            connection_id=connection_id,
            thread_id=thread_id,
            summary=summary,
        )
        return

    summary["messages_fetched"] += len(messages)

    # Gather attachments per message (Phase C), keyed by message id.
    message_attachments: Dict[str, List[Dict[str, Any]]] = {}
    for m in messages:
        mid = (m.get("id") or "").strip()
        if not mid:
            continue
        try:
            message_attachments[mid] = gather_attachment_sections(connection, m, summary)
        except Exception as e:  # noqa: BLE001
            message_attachments[mid] = []
            logger.warning(
                "gmail_attachments_gather_failed",
                extra={
                    "workspace_id": workspace_id,
                    "connection_id": connection_id,
                    "thread_id": thread_id,
                    "error": type(e).__name__,
                },
            )

    doc = build_email_thread_document(messages, connection_email, message_attachments=message_attachments)
    if doc is None:
        summary["threads_skipped"] += 1
        return

    # Upload the consolidated thread document.
    from hydradb_client import summarize_upload_response  # noqa: PLC0415

    try:
        response = hydra.upload_knowledge([doc])
    except Exception as e:  # noqa: BLE001
        emit_dead_letter(
            kind="gmail_ingest_upload",
            workspace_id=workspace_id,
            error=e,
            context={"connection_id": connection_id, "thread_id": thread_id, "file_count": 1},
        )
        summary["threads_failed"] += 1
        summary["messages_failed"] += len(messages)
        return

    ok, _bad = summarize_upload_response(response if isinstance(response, dict) else {}, batch_size=1)
    # DIAGNOSTIC (instrumentation only): the per-thread upload outcome.
    # ok=0 with non-empty raw_keys => HydraDB rejected the write (4xx/5xx
    # body); ok=0 with raw_keys=[] => empty {} from a network/RetryExhausted
    # failure (see hydradb_client.upload_knowledge). ok>=1 => write accepted.
    logger.info(
        "gmail_thread_upload_result",
        extra={
            "thread_id": thread_id,
            "ok": ok,
            "bad": _bad,
            "raw_keys": list(response.keys()) if isinstance(response, dict) else None,
        },
    )
    if not ok:
        summary["threads_failed"] += 1
        summary["messages_failed"] += len(messages)
        return

    summary["threads_uploaded"] += 1
    summary["threads_processed"] += 1
    summary["messages_uploaded"] += len(messages)

    # Memory extraction under the THREAD key (the whole conversation), so a
    # question like "what did we decide in the pricing thread?" extracts
    # from one coherent unit. Defensive: failure never blocks ingest.
    try:
        from memory_store import extract_and_persist  # noqa: PLC0415

        extract_and_persist(
            workspace_id=workspace_id,
            source_kind="gmail",
            source_stable_key=doc["stable_key"],
            source_timestamp=_gmail_ts_to_iso(doc.get("timestamp")),
            text=doc.get("content") or "",
            default_owner=(doc.get("from_name") or None),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "gmail_memory_extract_failed",
            extra={
                "workspace_id": workspace_id,
                "connection_id": connection_id,
                "thread_id": thread_id,
                "error": type(e).__name__,
            },
        )

    # Lazy migration / dedup: delete legacy per-message docs (+ memories)
    # for the messages now consolidated here, so a touched thread never
    # duplicates content with its pre-Phase-D per-message documents.
    legacy_keys = [
        stable_key_for_gmail_message((m.get("id") or "").strip()) for m in messages if (m.get("id") or "").strip()
    ]
    if legacy_keys:
        try:
            hydra.delete_knowledge(legacy_keys)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "gmail_legacy_msg_delete_failed",
                extra={
                    "workspace_id": workspace_id,
                    "connection_id": connection_id,
                    "thread_id": thread_id,
                    "error": type(e).__name__,
                },
            )
        try:
            from memory_store import delete_memories_by_source  # noqa: PLC0415

            for key in legacy_keys:
                try:
                    delete_memories_by_source(workspace_id=workspace_id, source_stable_key=key)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------- #
# Per-workspace ingestion runner
# ---------------------------------------------------------------------- #
# Synchronous on purpose: the caller wires this into a FastAPI
# BackgroundTask so the HTTP request returns immediately and the heavy
# lifting happens in the worker. Mirrors slack_oauth.run_workspace_ingest.


def _process_gmail_deletions(
    *,
    hydra: Any,
    workspace_id: str,
    connection_id: Optional[str],
    label_id: str,
    deleted_message_ids: List[str],
    summary: Dict[str, Any],
) -> None:
    """
    Remove permanently-deleted Gmail messages from HydraDB and clear
    their derived memories.

    Both steps are best-effort: a failure here logs and continues so a
    cleanup problem never blocks ingestion of the (separately handled)
    added messages, and never raises into the runner.

    Deletion is keyed by the per-message stable key
    (`gmail:msg:<id>`) -- the same `Source Key:` we stamp into every
    email document's markdown header.
    """
    keys: List[str] = []
    for mid in deleted_message_ids or []:
        mid = (mid or "").strip()
        if mid:
            keys.append(stable_key_for_gmail_message(mid))
    if not keys:
        return

    # 1. Remove the documents from HydraDB (batched, best-effort).
    try:
        hydra.delete_knowledge(keys)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "gmail_hydra_delete_failed",
            extra={
                "workspace_id": workspace_id,
                "connection_id": connection_id,
                "label_id": label_id,
                "key_count": len(keys),
                "error": type(e).__name__,
            },
        )

    # 2. Clear derived memories for each deleted source (best-effort).
    try:
        from memory_store import delete_memories_by_source  # noqa: PLC0415

        for key in keys:
            try:
                delete_memories_by_source(
                    workspace_id=workspace_id,
                    source_stable_key=key,
                )
            except Exception:  # noqa: BLE001
                # Per-key failure must not stop the rest of the cleanup.
                pass
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "gmail_memory_delete_failed",
            extra={
                "workspace_id": workspace_id,
                "connection_id": connection_id,
                "label_id": label_id,
                "error": type(e).__name__,
            },
        )

    summary["messages_deleted"] += len(keys)
    logger.info(
        "gmail_messages_deleted",
        extra={
            "workspace_id": workspace_id,
            "connection_id": connection_id,
            "label_id": label_id,
            "count": len(keys),
        },
    )


def run_workspace_gmail_ingest(
    *,
    workspace_id: str,
    connection: Dict[str, Any],
    label_ids: List[str],
    hydradb_sub_tenant_id: Optional[str] = None,
    max_messages: Optional[int] = None,
    sync_mode: str = "auto",
) -> Dict[str, Any]:
    """
    Ingest the most recent messages from each selected label into the
    workspace's HydraDB sub-tenant. Returns a stats dict.

    Phase 11 additions (incremental sync + observability):
      - `sync_mode`:
            "auto"        -> per label: incremental if a last_history_id
                              exists, else full. (default; scheduler uses this)
            "incremental" -> force history.list per label; if no
                              last_history_id exists, behaves like "full"
                              for that label and seeds the watermark.
            "full"        -> force the legacy listing path. Used by the
                              manual /api/gmail/ingest route so a user
                              who clicked "Run ingest" always gets the
                              most-recent N messages even if a recent
                              run already advanced the watermark.

      - If a Gmail history watermark is invalidated by Google (>= 7 days
        old, returns 404), we log + clear the watermark + fall back to
        the listing path for that label.

      - A refreshed access_token is persisted back to gmail_connections
        exactly once at end-of-run (whichever request triggered the
        refresh stamps `connection["_token_refreshed"] = True`).

      - Returns sync metadata: sync_mode_effective per label, total
        duration_ms, refresh_token_used, incremental_label_count,
        full_label_count, invalidations.

    Behavior unchanged from Phase 8 / Phase 4 fail-closed tenancy:
      - `hydradb_sub_tenant_id` is REQUIRED. Missing/blank values refuse
        the run (dead-letter + error summary) — never fall back to an
        env / hard-coded HydraDB tenant.
      - Per-run cap (GMAIL_MAX_MESSAGES_PER_RUN) is shared across labels.
      - SPAM/TRASH labels skipped unless GMAIL_ALLOW_SPAM_TRASH=true.
      - Per-label permanent errors emit dead_letter and continue.
      - gmail_ingestion_state.last_synced_at is stamped for every
        label we successfully processed.

    The function returns a stats dict; it never raises to the caller.
    """
    import time as _time  # noqa: PLC0415

    from hydradb_client import HydraDBClient, require_sub_tenant_id  # noqa: PLC0415
    from errors import WorkspaceTenantError  # noqa: PLC0415
    from supabase_client import (  # noqa: PLC0415
        get_gmail_ingestion_state_map,
        update_gmail_connection_tokens,
        upsert_gmail_ingestion_state,
    )

    started_at = datetime.now(timezone.utc)
    started_perf = _time.perf_counter()
    summary: Dict[str, Any] = {
        "labels_processed": 0,
        "labels_skipped": 0,
        "labels_failed": 0,
        "messages_fetched": 0,
        "messages_uploaded": 0,
        "messages_failed": 0,
        "messages_skipped": 0,
        "messages_deleted": 0,
        # Phase D thread-aware counters.
        "threads_processed": 0,
        "threads_uploaded": 0,
        "threads_failed": 0,
        "threads_skipped": 0,
        "threads_deleted": 0,
        "attachments_processed": 0,
        "attachments_failed": 0,
        "attachments_skipped": 0,
        # Phase 11 observability fields.
        "sync_mode_requested": sync_mode,
        "sync_started_at": started_at.isoformat(),
        "sync_finished_at": None,
        "duration_ms": 0,
        "refresh_token_used": False,
        "incremental_label_count": 0,
        "full_label_count": 0,
        "invalidations": 0,
        "per_label": [],  # one entry per label processed
    }
    if not label_ids:
        summary["sync_finished_at"] = datetime.now(timezone.utc).isoformat()
        return summary

    refresh_token = (connection.get("refresh_token") or "").strip()
    if not refresh_token:
        emit_dead_letter(
            kind="gmail_ingest",
            workspace_id=workspace_id,
            error=RuntimeError("missing_refresh_token"),
            context={"connection_id": connection.get("id")},
        )
        summary["sync_finished_at"] = datetime.now(timezone.utc).isoformat()
        return summary

    cap_total = max_messages if max_messages is not None else _max_messages_per_run()
    cap_total = max(1, int(cap_total))
    allow_spam_trash = os.getenv("GMAIL_ALLOW_SPAM_TRASH", "").strip().lower() in ("1", "true", "yes", "on")

    # Fail closed: sub-tenant is REQUIRED. Never fall back to the env /
    # hard-coded HydraDB tenant (cross-workspace leak).
    try:
        sub_tenant = require_sub_tenant_id(
            hydradb_sub_tenant_id,
            context=f"run_workspace_gmail_ingest workspace_id={workspace_id}",
        )
    except WorkspaceTenantError:
        logger.error(
            "gmail_ingest_no_sub_tenant",
            extra={"workspace_id": workspace_id},
        )
        emit_dead_letter(
            kind="gmail_ingest",
            workspace_id=workspace_id,
            error=RuntimeError("missing_sub_tenant"),
            context={"connection_id": connection.get("id")},
        )
        summary["labels_failed"] = len(label_ids)
        summary["sync_finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["error"] = "missing_sub_tenant"
        return summary

    hydra = HydraDBClient(sub_tenant_id=sub_tenant)

    connection_id = connection.get("id")
    connection_email = (connection.get("email") or "").strip()

    # Snapshot the access token we started with so we can detect a
    # mid-run refresh and persist exactly once. _authed_request stamps
    # connection["_token_refreshed"] = True when it refreshes; we also
    # cross-check by comparing the access_token value so a stale
    # sentinel can't lie.
    initial_access_token = (connection.get("access_token") or "").strip()
    connection.pop("_token_refreshed", None)

    # Pull watermarks per label in ONE call. Empty dict for a fresh
    # connection that has never been synced.
    try:
        state_map = get_gmail_ingestion_state_map(
            workspace_id=workspace_id,
            gmail_connection_id=connection_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "gmail_state_map_failed",
            extra={
                "workspace_id": workspace_id,
                "connection_id": connection_id,
                "error": type(e).__name__,
            },
        )
        state_map = {}

    # Seen-this-run set: dedupes threads across labels in one sweep
    # (a thread can be reachable under more than one label).
    seen_thread_ids_this_run: set = set()

    logger.info(
        "gmail_ingest_start",
        extra={
            "workspace_id": workspace_id,
            "connection_id": connection_id,
            "label_count": len(label_ids),
            "cap_total": cap_total,
            "sync_mode_requested": sync_mode,
            "labels_with_watermark": sum(1 for v in state_map.values() if v.get("last_history_id")),
        },
    )

    remaining = cap_total
    for label_id in label_ids:
        if remaining <= 0:
            summary["labels_skipped"] += 1
            continue

        # Spam/trash safety guard.
        if not allow_spam_trash and label_id in ("SPAM", "TRASH"):
            logger.info(
                "gmail_ingest_label_blocked",
                extra={
                    "workspace_id": workspace_id,
                    "connection_id": connection_id,
                    "label_id": label_id,
                    "reason": "spam_or_trash",
                },
            )
            summary["labels_skipped"] += 1
            continue

        # Decide between incremental and full FOR THIS LABEL.
        label_state = state_map.get(label_id) or {}
        last_history_id = (label_state.get("last_history_id") or "").strip()

        use_incremental = False
        if sync_mode == "incremental":
            use_incremental = bool(last_history_id)
        elif sync_mode == "auto":
            use_incremental = bool(last_history_id)
        # sync_mode == "full" -> never use incremental.

        affected_thread_ids: List[str] = []
        new_history_id: Optional[str] = None
        invalidated_this_label = False
        effective_label_mode = "full"

        if use_incremental:
            try:
                hist_result = retry_with_backoff(
                    list_history_message_ids,
                    connection,
                    start_history_id=last_history_id,
                    label_id=label_id,
                    max_results=min(remaining, 100),
                    attempts=3,
                    initial_delay=0.5,
                    max_delay=4.0,
                    retry_on=(GmailApiError,),
                    op_name="gmail_list_history",
                )
            except GmailApiError as e:
                summary["labels_failed"] += 1
                emit_dead_letter(
                    kind="gmail_ingest_label",
                    workspace_id=workspace_id,
                    error=e,
                    context={
                        "connection_id": connection_id,
                        "label_id": label_id,
                        "stage": "list_history",
                    },
                )
                continue

            if hist_result.get("invalidated"):
                # Watermark too old -- fall back to full listing and
                # clear the stored last_history_id (we'll seed a fresh
                # one below from the listing's high-water mark via
                # getProfile).
                invalidated_this_label = True
                summary["invalidations"] += 1
                logger.info(
                    "gmail_history_invalidated",
                    extra={
                        "workspace_id": workspace_id,
                        "connection_id": connection_id,
                        "label_id": label_id,
                    },
                )
                # Fall through to the full-listing path below.
                use_incremental = False
            else:
                new_history_id = hist_result.get("next_history_id")
                effective_label_mode = "incremental"

                # Legacy deletion cleanup (Phase A): remove any pre-Phase-D
                # per-message `gmail:msg:` docs + their memories for
                # permanently-deleted messages. Safe no-op once a thread
                # has been migrated to a consolidated thread doc. Never
                # blocks the thread (re)materialization below.
                deleted_ids = hist_result.get("deleted_message_ids") or []
                if deleted_ids:
                    _process_gmail_deletions(
                        hydra=hydra,
                        workspace_id=workspace_id,
                        connection_id=connection_id,
                        label_id=label_id,
                        deleted_message_ids=deleted_ids,
                        summary=summary,
                    )

                # Thread-aware sync: ANY added OR deleted message marks its
                # thread for (re)materialization. Gmail history carries
                # threadId on both, so a deleted message's thread resolves
                # without re-fetching the gone message. The materializer
                # re-fetches each thread and either rebuilds it (still has
                # messages) or deletes the consolidated doc (fully gone).
                for tid in (hist_result.get("added_thread_ids") or []) + (hist_result.get("deleted_thread_ids") or []):
                    tid = (tid or "").strip()
                    if tid and tid not in affected_thread_ids:
                        affected_thread_ids.append(tid)

        if not use_incremental:
            # Full listing path -> the set of recent THREADS for the label.
            try:
                affected_thread_ids = retry_with_backoff(
                    list_thread_ids_for_label,
                    connection,
                    label_id,
                    max_results=min(remaining, 100),
                    attempts=3,
                    initial_delay=0.5,
                    max_delay=4.0,
                    retry_on=(GmailApiError,),
                    op_name="gmail_list_threads",
                )
            except GmailApiError as e:
                summary["labels_failed"] += 1
                emit_dead_letter(
                    kind="gmail_ingest_label",
                    workspace_id=workspace_id,
                    error=e,
                    context={
                        "connection_id": connection_id,
                        "label_id": label_id,
                        "stage": "list_threads",
                    },
                )
                continue
            effective_label_mode = "full"
            # Seed a new high-water mark from the mailbox profile so
            # the NEXT run can go incremental. Best-effort -- if this
            # fails the next run just runs full again.
            if new_history_id is None:
                try:
                    profile = retry_with_backoff(
                        get_mailbox_profile,
                        connection,
                        attempts=2,
                        initial_delay=0.5,
                        max_delay=2.0,
                        retry_on=(GmailApiError,),
                        op_name="gmail_get_profile",
                    )
                    if profile and profile.get("historyId"):
                        new_history_id = str(profile["historyId"])
                except GmailApiError:
                    new_history_id = None

        # Dedupe threads across labels in this same run (a thread can be
        # reachable under more than one label).
        affected_thread_ids = [t for t in affected_thread_ids if t and t not in seen_thread_ids_this_run]

        # DIAGNOSTIC (instrumentation only): how many threads this label
        # resolved to, and the sub-tenant we will write them into. Lets us
        # distinguish "zero threads materialized" from "threads found but
        # uploads failing", and compare the ingest sub-tenant against the
        # recall sub-tenant (recall_corpus_probe).
        logger.info(
            "gmail_ingest_label_threads",
            extra={
                "workspace_id": workspace_id,
                "connection_id": connection_id,
                "label_id": label_id,
                "sub_tenant": hydradb_sub_tenant_id,
                "mode": effective_label_mode,
                "thread_count": len(affected_thread_ids),
            },
        )

        # Materialize each affected thread. The per-run budget (`remaining`)
        # is now a THREAD budget: one consolidated document per thread.
        materialized_this_label = 0
        for tid in affected_thread_ids:
            if remaining <= 0:
                break
            seen_thread_ids_this_run.add(tid)
            _materialize_gmail_thread(
                connection,
                tid,
                hydra=hydra,
                workspace_id=workspace_id,
                connection_id=connection_id,
                connection_email=connection_email,
                summary=summary,
            )
            materialized_this_label += 1
            remaining -= 1

        # Persist the ingestion-state row. Always stamp last_synced_at;
        # advance last_history_id only when we have a fresh one.
        try:
            upsert_gmail_ingestion_state(
                workspace_id=workspace_id,
                gmail_connection_id=connection_id,
                label_id=label_id,
                last_history_id=new_history_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "gmail_ingestion_state_update_failed",
                extra={
                    "workspace_id": workspace_id,
                    "connection_id": connection_id,
                    "label_id": label_id,
                    "error": type(e).__name__,
                },
            )

        summary["labels_processed"] += 1
        if effective_label_mode == "incremental":
            summary["incremental_label_count"] += 1
        else:
            summary["full_label_count"] += 1
        summary["per_label"].append(
            {
                "label_id": label_id,
                "mode": effective_label_mode,
                "invalidated": invalidated_this_label,
                "threads": materialized_this_label,
                "new_history_id": new_history_id,
            }
        )

    # Persist refreshed access token, if a refresh happened.
    current_access_token = (connection.get("access_token") or "").strip()
    refreshed = (
        connection.get("_token_refreshed") is True
        and current_access_token
        and current_access_token != initial_access_token
    )
    if refreshed:
        # Cross-workspace defense in depth: pass workspace_id explicitly.
        try:
            ok = update_gmail_connection_tokens(
                connection_id=connection_id,
                workspace_id=workspace_id,
                access_token=current_access_token,
            )
        except Exception as e:  # noqa: BLE001
            ok = False
            logger.warning(
                "gmail_token_persist_failed",
                extra={
                    "workspace_id": workspace_id,
                    "connection_id": connection_id,
                    "error": type(e).__name__,
                },
            )
        summary["refresh_token_used"] = bool(ok)
    # Always clear the sentinel so a future caller that reuses the
    # same connection dict starts clean.
    connection.pop("_token_refreshed", None)

    finished_at = datetime.now(timezone.utc)
    summary["sync_finished_at"] = finished_at.isoformat()
    summary["duration_ms"] = int((_time.perf_counter() - started_perf) * 1000)

    logger.info(
        "gmail_ingest_complete",
        extra={
            "workspace_id": workspace_id,
            "connection_id": connection_id,
            "duration_ms": summary["duration_ms"],
            "labels_processed": summary["labels_processed"],
            "labels_skipped": summary["labels_skipped"],
            "labels_failed": summary["labels_failed"],
            "messages_uploaded": summary["messages_uploaded"],
            "messages_failed": summary["messages_failed"],
            "messages_deleted": summary["messages_deleted"],
            "threads_processed": summary["threads_processed"],
            "threads_deleted": summary["threads_deleted"],
            "attachments_processed": summary["attachments_processed"],
            "incremental_label_count": summary["incremental_label_count"],
            "full_label_count": summary["full_label_count"],
            "invalidations": summary["invalidations"],
            "refresh_token_used": summary["refresh_token_used"],
        },
    )

    # Phase 15: emit analytics. Defensive -- analytics failure must
    # NOT affect the ingest summary.
    try:
        from analytics_store import emit_event  # noqa: PLC0415

        emit_event(
            workspace_id=workspace_id,
            kind="ingest_completed",
            source_kind="gmail",
            latency_ms=summary["duration_ms"],
            success=summary["labels_failed"] == 0,
            payload={
                "connection_id": connection_id,
                "labels_processed": summary["labels_processed"],
                "labels_failed": summary["labels_failed"],
                "messages_uploaded": summary["messages_uploaded"],
                "messages_failed": summary["messages_failed"],
                "messages_deleted": summary["messages_deleted"],
                "threads_processed": summary["threads_processed"],
                "threads_deleted": summary["threads_deleted"],
                "attachments_processed": summary["attachments_processed"],
                "attachments_failed": summary["attachments_failed"],
                "incremental_label_count": summary["incremental_label_count"],
                "full_label_count": summary["full_label_count"],
                "invalidations": summary["invalidations"],
                "sync_mode_requested": summary["sync_mode_requested"],
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return summary
