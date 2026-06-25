"""
FastAPI app for the Second Brain MVP.

Endpoints:
    GET    /                                              -> service info card             (public)
    GET    /api/health/live                               -> liveness probe               (public)
    GET    /api/health/ready                              -> readiness probe              (public)
    GET    /api/health                                    -> {"status": "ok", ...}         (public)
    POST   /api/query                                     -> {"answer", "sources", ...}    (Supabase JWT + Workspace)
    POST   /api/query/stream                              -> Server-Sent Events            (Supabase JWT + Workspace)
    GET    /api/me                                        -> {"id", "email"}               (Supabase JWT)
    GET    /api/me/workspaces                             -> [{"id","name","slug","role"}] (Supabase JWT)
    GET    /api/chat/sessions                             -> [{...sessions...}]            (Supabase JWT + Workspace)
    POST   /api/chat/sessions                             -> created session row           (Supabase JWT + Workspace)
    GET    /api/chat/sessions/{id}/messages               -> [{...messages...}]            (Supabase JWT + Workspace)
    POST   /api/chat/sessions/{id}/messages               -> created message row           (Supabase JWT + Workspace)
    GET    /api/saved-answers                             -> [{...saved answers...}]       (Supabase JWT + Workspace)
    POST   /api/saved-answers                             -> created saved answer row      (Supabase JWT + Workspace)
    DELETE /api/saved-answers/{id}                        -> {"id","deleted":true}         (Supabase JWT + Workspace)
    GET    /api/slack/connect-url                         -> {"url": "..."}                (Supabase JWT + Workspace)
    GET    /api/slack/oauth/callback                      -> redirect to frontend          (signed state token)
    GET    /api/slack/channels                            -> {"connected", "channels"}     (Supabase JWT + Workspace)
    POST   /api/slack/channels                            -> {"selected_count"}            (Supabase JWT + Workspace)
    POST   /api/slack/ingest                              -> {"status": "started"}         (Supabase JWT + Workspace)
    DELETE /api/slack/disconnect                          -> {"disconnected": true}         (Supabase JWT + Workspace)
    GET    /api/admin/status                              -> ingestion status snapshot     (X-API-Key, legacy)
    POST   /slack/events                                  -> Slack Events API webhook      (Slack signature)

User-facing routes require:
    Authorization: Bearer <supabase_jwt>
    X-Workspace-Id: <uuid of an active workspace the user is a member of>

The admin/internal `/api/admin/status` route is still gated by the
shared APP_API_KEY (X-API-Key) — moving it to Supabase admin-role is
deferred.

The /slack/oauth/callback route is browser-redirected by Slack itself,
so it cannot require an Authorization header. Instead the state
parameter is a signed token binding the OAuth attempt to a specific
workspace + user; the handler verifies that signature before storing
the installation.

The /slack/events endpoint is public but authenticated via Slack's
HMAC-SHA256 request signature — see slack_signature.py.
"""

import json
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env vars at runtime.
load_dotenv()

from fastapi import (  # noqa: E402
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from auth import require_api_key  # noqa: E402
from auth_supabase import (  # noqa: E402
    SupabaseUser,
    WorkspaceContext,
    require_user,
    require_workspace,
)
from date_utils import parse_date_query  # noqa: E402
from errors import AppError, app_error_handler  # noqa: E402

# Phase 8: Gmail connector. Aliased on import so the symbol names don't
# clash with the Slack helpers above (both modules expose
# build_connect_url, exchange_code, verify_oauth_state, etc.).
from gmail_oauth import build_connect_url as gmail_build_connect_url  # noqa: E402
from gmail_oauth import exchange_code as gmail_exchange_code  # noqa: E402
from gmail_oauth import fetch_user_info as gmail_fetch_user_info  # noqa: E402
from gmail_oauth import (  # noqa: E402
    gmail_oauth_configured,
)
from gmail_oauth import installation_from_token_response as gmail_installation_from_token_response  # noqa: E402
from gmail_oauth import list_labels as list_gmail_labels_from_api  # noqa: E402
from gmail_oauth import (  # noqa: E402
    run_workspace_gmail_ingest,
)
from gmail_oauth import verify_oauth_state as verify_gmail_oauth_state  # noqa: E402
from health import router as health_router  # noqa: E402
from llm import stream_grounded_answer  # noqa: E402
from logging_config import configure_logging, get_logger  # noqa: E402
from prompts import INSUFFICIENT_CONTEXT_ANSWER  # noqa: E402
from query_cache import build_cache_key, get_cached  # noqa: E402
from query_cache import put as cache_put  # noqa: E402
from query_rewriter import rewrite_query  # noqa: E402
from rate_limit import (  # noqa: E402
    make_rate_limit_dependency,
    rate_limit_dependency,
)
from realtime_ingest import (  # noqa: E402
    _event_already_seen,
    admin_status_snapshot,
    process_slack_event,
)
from recall import (  # noqa: E402
    answer_question,
    finalize_answer,
    prepare_recall_context,
)
from request_context import RequestContextMiddleware  # noqa: E402
from scheduler import auto_ingest_enabled, start_scheduler, stop_scheduler  # noqa: E402
from slack_oauth import (  # noqa: E402
    build_connect_url,
    exchange_code,
    installation_from_oauth_response,
    list_slack_channels,
    revoke_bot_token,
    run_workspace_ingest,
    slack_oauth_configured,
    verify_oauth_state,
)
from slack_signature import verify_slack_signature  # noqa: E402
from startup import validate_required_env  # noqa: E402
from supabase_client import (  # noqa: E402
    create_chat_message,
    create_chat_session,
    create_saved_answer,
    create_share_link,
    delete_gmail_connection,
    delete_saved_answer,
    delete_slack_installation,
    ensure_workspace_sub_tenant,
    get_gmail_connection,
    get_gmail_connection_public,
    get_gmail_connection_sync_summary,
    get_saved_answer,
    get_share_link_by_token,
    get_slack_installation,
    list_chat_messages,
    list_chat_sessions,
    list_gmail_connections_public,
    list_gmail_labels,
    list_saved_answers,
    list_selected_channel_ids,
    list_selected_channel_settings,
    list_selected_gmail_label_ids,
    list_share_links_for_workspace,
    list_user_workspaces,
    list_workspace_channels,
    revoke_share_link,
    set_selected_channels,
    set_selected_gmail_labels,
    upsert_gmail_connection,
    upsert_gmail_labels,
    upsert_slack_channels,
    upsert_slack_installation,
)

configure_logging(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = get_logger(__name__)


# ---------- CORS configuration ---------- #
DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://localhost:5173"


def _parse_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip() or DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


ALLOWED_ORIGINS = _parse_cors_origins()


# ---------- Phase 7: per-route rate-limit dependencies ---------- #
# Each one targets a named bucket so a flood in one area can't starve
# the others. The factory reads its limit from env at request time, so
# operators can retune limits without a restart.
auth_rate_limit = make_rate_limit_dependency("auth")
slack_webhook_rate_limit = make_rate_limit_dependency("slack_webhook")
slack_ingest_rate_limit = make_rate_limit_dependency("ingest")
# Phase 13: public share-link reads have no auth, so they get their
# own rate-limit bucket. The bucket name is unknown to rate_limit.py's
# defaults table -- which just means it falls back to the global
# default limit. That's the right behavior; we don't want to be more
# generous to public anonymous traffic than to authenticated users.
public_share_rate_limit = make_rate_limit_dependency("public_share")


# ---------- Lifespan: startup checks + scheduler ---------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_required_env()
    # Phase 7: opt-in Sentry init (no-op when SENTRY_DSN unset).
    # Initialized AFTER env validation so a startup config error
    # surfaces in stdout logs, not Sentry.
    from observability import init_sentry  # noqa: PLC0415

    init_sentry()
    start_scheduler()
    try:
        yield
    finally:
        stop_scheduler()


app = FastAPI(title="Second Brain (Slack MVP)", lifespan=lifespan)

# Translate every typed AppError into a normalized JSON 4xx/5xx response.
app.add_exception_handler(AppError, app_error_handler)

# Health endpoints: /api/health/live, /api/health/ready, /api/health
app.include_router(health_router)


# ---------- CORS ---------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    # DELETE is needed for /api/saved-answers/{id}. PATCH is included for
    # forward-compat; PUT and HEAD are too. OPTIONS is what CORS uses
    # for the preflight itself.
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
        "Authorization",
        "X-Workspace-Id",
    ],
)
app.add_middleware(RequestContextMiddleware)


# ---------- Request validation ---------- #
QueryMode = Literal[
    "default",
    "summary",
    "decisions",
    "action_items",
    "who_said",
    "exact",
    "hybrid",
]
DocumentType = Literal["message", "thread"]
HistoryRole = Literal["user", "assistant"]
# Sources the user can restrict a query to. None / empty = all sources.
# Currently the backend writes only Slack ("message" / "thread"
# documents) and Gmail ("email" documents); the literal is closed so a
# typo in the JSON body becomes a 422 instead of a silent no-op.
SourceKind = Literal["slack", "gmail"]


# Cap on conversation_history entries from the request. Frontend keeps the
# last 6; we enforce the same on the server as defense in depth — a buggy
# or malicious client can't blow up the prompt with thousands of turns.
MAX_CONVERSATION_HISTORY = 6


class ConversationMessage(BaseModel):
    """One turn of recent chat history sent by the frontend."""

    model_config = ConfigDict(str_strip_whitespace=True)

    role: HistoryRole = Field(
        ...,
        description="Speaker of this turn: 'user' or 'assistant'.",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The turn's text. Long assistant answers are truncated "
        "downstream when formatted into the prompt.",
    )


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's natural-language question (3–2000 chars, " "leading/trailing whitespace is stripped).",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=10,
        description="How many context chunks to retrieve from HydraDB (1–10).",
    )
    mode: QueryMode = Field(
        "default",
        description="Answer style + retrieval strategy. 'default' is a "
        "concise grounded answer; 'exact' prefers literal "
        "keyword matches; 'hybrid' combines semantic + keyword.",
    )
    channel: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional channel-name filter (e.g. 'all-second-brain').",
    )
    user: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional user-name filter.",
    )
    document_type: Optional[DocumentType] = Field(
        None,
        description="Optional filter: 'message' or 'thread'.",
    )
    # Optional restriction on which connector source a candidate must
    # come from. None / omitted / empty list means "all sources" (the
    # default and the pre-Phase-9 behavior). Each list item is one of
    # SourceKind ("slack", "gmail"). Pydantic enforces the literal so
    # an unknown source name in the body returns 422.
    allowed_sources: Optional[List[SourceKind]] = Field(
        None,
        description="Optional restriction on connector source(s). "
        "Omit or pass null for all sources; pass ['slack'] "
        "or ['gmail'] to filter to one connector; pass "
        "['slack','gmail'] to allow both explicitly.",
    )
    # Date range — accept either a Slack ts string ("1778775842.876209")
    # or a unix-timestamp number. recall.py normalizes both. Explicit
    # values override anything derived from date_query.
    start_timestamp: Optional[Union[str, float]] = Field(
        None,
        description="Inclusive lower bound on source timestamp (Slack ts string or unix seconds).",
    )
    end_timestamp: Optional[Union[str, float]] = Field(
        None,
        description="Inclusive upper bound on source timestamp.",
    )
    # Natural-language date phrase (e.g. "last week", "yesterday",
    # "after May 10"). Parsed server-side; explicit start/end_timestamp
    # win where they overlap.
    date_query: Optional[str] = Field(
        None,
        max_length=200,
        description="Natural-language date phrase. Examples: 'today', "
        "'last week', 'last 7 days', 'after May 10', "
        "'from May 1 to May 7'.",
    )
    # Recent chat turns (oldest first). Used by the LLM to resolve
    # references like 'he' / 'that' / 'the earlier discussion'. Does NOT
    # affect HydraDB retrieval — only the latest `question` does.
    # Server caps at MAX_CONVERSATION_HISTORY turns regardless of what
    # the client sends.
    conversation_history: Optional[List[ConversationMessage]] = Field(
        None,
        description="Recent chat turns (oldest first), up to "
        f"{MAX_CONVERSATION_HISTORY}. Anything beyond that is "
        "truncated server-side to the most recent turns.",
    )


def _resolve_date_filters(req: QueryRequest) -> Dict[str, Any]:
    """
    Resolve the request's date inputs into a single effective range
    plus a debug record. Explicit start/end_timestamp always win.

    Returns:
        {
            "start_timestamp": float | str | None,    # effective start
            "end_timestamp":   float | str | None,    # effective end
            "date_query_debug": {                     # surfaced in /api debug
                "phrase":   str,
                "matched":  bool,
                "note":     str,
                "applied_start": float | None,
                "applied_end":   float | None,
            } | None,
        }
    """
    explicit_start = req.start_timestamp
    explicit_end = req.end_timestamp

    parsed = parse_date_query(req.date_query)
    if not parsed["phrase"]:
        # User didn't send date_query at all. No debug info needed.
        return {
            "start_timestamp": explicit_start,
            "end_timestamp": explicit_end,
            "date_query_debug": None,
        }

    # Explicit values win. We only apply parsed bounds where the explicit
    # value is None, so a request that sets both keeps full control.
    effective_start = explicit_start if explicit_start is not None else parsed["start_timestamp"]
    effective_end = explicit_end if explicit_end is not None else parsed["end_timestamp"]

    return {
        "start_timestamp": effective_start,
        "end_timestamp": effective_end,
        "date_query_debug": {
            "phrase": parsed["phrase"],
            "matched": parsed["matched"],
            "note": parsed["note"],
            "applied_start": parsed["start_timestamp"] if explicit_start is None else None,
            "applied_end": parsed["end_timestamp"] if explicit_end is None else None,
        },
    }


def _cache_key_from_request(
    req: QueryRequest,
    resolved: Dict[str, Any],
    rewrite: Dict[str, Any],
) -> str:
    """
    Build a cache key from the request + resolved date range + resolved
    query-rewrite filters.

    Including the resolved date range (not the raw date_query phrase) means
    two requests with different phrasings of the same window — "last week"
    vs "from <last Monday> to <last Sunday>" — both share a cache slot.

    Including the effective_channel/effective_user/metadata_bias from the
    query rewriter means two queries that look textually similar but
    resolve to different inferred filters get different cache slots. The
    `metadata_bias` is also keyed because weak inference still changes
    the ranking order.
    """
    return build_cache_key(
        {
            "question": req.question,
            "top_k": req.top_k,
            "mode": req.mode,
            # Use the EFFECTIVE filters (explicit OR strong-inferred) so an
            # inferred filter cache-isolates from an unfiltered query.
            "channel": rewrite["effective_channel"],
            "user": rewrite["effective_user"],
            "document_type": req.document_type,
            "start_timestamp": resolved["start_timestamp"],
            "end_timestamp": resolved["end_timestamp"],
            # Weak inference: metadata_bias is a dict or None.
            "metadata_bias": rewrite["metadata_bias"],
        }
    )


def _normalize_history(req: QueryRequest) -> List[Dict[str, str]]:
    """
    Convert the request's conversation_history (Pydantic models) into a
    plain list-of-dicts capped at MAX_CONVERSATION_HISTORY, oldest-first.

    Defensive on the server side:
      - If the client sends more than the cap, we keep only the most
        recent turns.
      - Empty/whitespace content is dropped (the Pydantic min_length=1
        catches truly empty strings, but a whitespace-only string would
        slip through and we'd rather skip it than show "User: " to the LLM).

    Returns an empty list when no usable history was provided, which the
    downstream "history_used" flag treats as "no history" — same as if
    the client had omitted the field entirely.
    """
    raw = req.conversation_history or []
    cleaned: List[Dict[str, str]] = []
    for msg in raw:
        content = (msg.content or "").strip()
        if not content:
            continue
        cleaned.append({"role": msg.role, "content": content})
    # Keep the most recent N if the client over-sent.
    if len(cleaned) > MAX_CONVERSATION_HISTORY:
        cleaned = cleaned[-MAX_CONVERSATION_HISTORY:]
    return cleaned


def _resolve_query_rewrite(req: QueryRequest) -> Dict[str, Any]:
    """
    Run person/channel inference on the question and split the results
    into "effective filters" and "metadata bias":

      strong inference -> becomes the effective filter (hard filter)
      weak inference   -> becomes the metadata bias (ranking-only)

    Explicit request filters always win — if the user sent `channel: ...`,
    inference for channel is suppressed entirely. Same for `user`.

    Returns:
        {
            "effective_channel": str | None,   # what prepare_recall sees
            "effective_user":    str | None,
            "metadata_bias":     {channel?, user?} | None,
            "rewrite_debug":     {                  # surfaced to UI
                "inferred_person":   str | None,
                "inferred_channel":  str | None,
                "person_confidence": "strong" | "weak" | None,
                "channel_confidence":"strong" | "weak" | None,
                "retrieval_biases_applied": list[str],
            } | None,
        }
    """
    rewrite = rewrite_query(
        req.question,
        explicit_channel=req.channel,
        explicit_user=req.user,
    )

    effective_channel = req.channel
    effective_user = req.user
    bias: Dict[str, str] = {}

    # Strong inference → hard filter (only when caller didn't set one).
    if (
        rewrite["inferred_channel"]
        and rewrite["channel_confidence"] == "strong"
        and not (req.channel and req.channel.strip())
    ):
        effective_channel = rewrite["inferred_channel"]
    elif (
        rewrite["inferred_channel"]
        and rewrite["channel_confidence"] == "weak"
        and not (req.channel and req.channel.strip())
    ):
        bias["channel"] = rewrite["inferred_channel"]

    if rewrite["inferred_person"] and rewrite["person_confidence"] == "strong" and not (req.user and req.user.strip()):
        effective_user = rewrite["inferred_person"]
    elif rewrite["inferred_person"] and rewrite["person_confidence"] == "weak" and not (req.user and req.user.strip()):
        bias["user"] = rewrite["inferred_person"]

    rewrite_debug = None
    if rewrite["inferred_person"] or rewrite["inferred_channel"]:
        # Only attach the debug record when we actually inferred something —
        # otherwise the UI would have to special-case empty rewrite blobs.
        rewrite_debug = {
            "inferred_person": rewrite["inferred_person"],
            "inferred_channel": rewrite["inferred_channel"],
            "person_confidence": rewrite["person_confidence"],
            "channel_confidence": rewrite["channel_confidence"],
            "retrieval_biases_applied": rewrite["retrieval_biases_applied"],
        }

    return {
        "effective_channel": effective_channel,
        "effective_user": effective_user,
        "metadata_bias": bias or None,
        "rewrite_debug": rewrite_debug,
    }


# ---------- Public routes ---------- #
@app.get("/")
def root() -> Dict[str, str]:
    return {
        "name": "Second Brain HydraDB MVP",
        "status": "ok",
        "docs": "/docs",
        "health": "/api/health",
        "liveness": "/api/health/live",
        "readiness": "/api/health/ready",
    }


@app.get("/api/health")
def health() -> Dict[str, str]:
    """
    Liveness probe for load balancers (Render, Railway) + uptime
    monitors. Intentionally cheap: no DB calls, no third-party
    reach-out. Returns 200 as long as the process can serve HTTP.

    Production hosts can poll this every few seconds without budget
    concerns. Use /api/admin/status (X-API-Key gated) for richer
    ingestion telemetry.
    """
    return {
        "status": "ok",
        "service": "second-brain-api",
        # ENVIRONMENT lets dashboards distinguish prod vs preview vs
        # local. Blank when unset rather than guessing.
        "environment": (os.getenv("ENVIRONMENT") or "").strip(),
        # APP_VERSION is set by the host's build step (e.g. Render's
        # RENDER_GIT_COMMIT, Railway's RAILWAY_GIT_COMMIT_SHA). Falling
        # back to "dev" makes local responses obvious in logs.
        "version": (
            os.getenv("APP_VERSION") or os.getenv("RENDER_GIT_COMMIT") or os.getenv("RAILWAY_GIT_COMMIT_SHA") or "dev"
        ).strip(),
    }


@app.get("/api/ready")
def ready(response: Response) -> Dict[str, Any]:
    """
    Readiness probe. Confirms the backend's CRITICAL upstream
    dependencies (Supabase, HydraDB, OpenAI/compatible) are reachable.

    Distinct from /api/health (liveness):
      - /api/health  -> "the process can serve HTTP"  (cheap, always 200 once boot completes)
      - /api/ready   -> "the process can serve TRAFFIC" (200 only when all deps OK)

    Returns 200 with check details when all deps are healthy, or 503
    with the same shape when any dep fails. The body always includes a
    per-check breakdown so a probe failure is debuggable from logs.
    """
    from observability import check_dependencies  # noqa: PLC0415

    result = check_dependencies()
    if not result.get("ok"):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


# ---------- Protected routes ---------- #
@app.post(
    "/api/query",
    dependencies=[Depends(rate_limit_dependency)],
)
def query(
    req: QueryRequest,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Retrieve Slack context from HydraDB, ask the cloud LLM for a grounded
    answer, and return it along with the source list.

    Cached responses are flagged via debug.cache_hit=true.

    Requires headers:
        Authorization: Bearer <supabase_jwt>
        X-Workspace-Id: <uuid>
    """
    resolved = _resolve_date_filters(req)
    history = _normalize_history(req)
    rewrite = _resolve_query_rewrite(req)

    # Cache behavior when conversation_history is present:
    #   We BYPASS the cache entirely. Reasons:
    #     1. Two follow-ups in different conversations may share the same
    #        question text ("what did he say?") but mean different things.
    #     2. A long history makes for a wide cache key with near-zero hit
    #        rate.
    #     3. Stateless queries (no history) keep working through the cache
    #        exactly as before — no regression.
    use_cache = not history
    cache_key = _cache_key_from_request(req, resolved, rewrite) if use_cache else None

    if use_cache:
        cached = get_cached(cache_key)
        if cached is not None:
            return cached

    # Phase 16: intelligence query routing. A narrow regex classifier
    # detects ownership / status-blocker / decision-history / timeline
    # questions and answers them from structured memory with citations
    # to source_stable_keys. The contract is strict: when the question
    # matches no intent (zero I/O in that case), or matches but has no
    # supporting memory data, or anything fails, the router returns
    # None and this route proceeds byte-identically to the existing
    # retrieval pipeline below. Stateless queries only -- follow-ups
    # need the conversation history the LLM path provides. Routed
    # answers are never cached (they're cheap to recompute and track
    # live memory state).
    if not history:
        intel_result: Optional[Dict[str, Any]] = None
        try:
            from memory_intelligence import route_intelligence_query  # noqa: PLC0415

            intel_result = route_intelligence_query(
                workspace_id=workspace.workspace_id,
                question=req.question,
            )
        except Exception as _e:  # noqa: BLE001
            logger.debug(
                "intelligence_route_skipped",
                extra={"error": type(_e).__name__},
            )
            intel_result = None
        if intel_result is not None:
            debug = dict(intel_result.get("debug") or {})
            debug["cache_hit"] = False
            if resolved["date_query_debug"] is not None:
                debug["date_query"] = resolved["date_query_debug"]
            if rewrite["rewrite_debug"] is not None:
                debug["query_rewrite"] = rewrite["rewrite_debug"]
            return {**intel_result, "debug": debug}

    result = answer_question(
        question=req.question,
        top_k=req.top_k,
        mode=req.mode,
        # Effective filters: explicit > strong-inferred > None.
        channel=rewrite["effective_channel"],
        user=rewrite["effective_user"],
        document_type=req.document_type,
        start_timestamp=resolved["start_timestamp"],
        end_timestamp=resolved["end_timestamp"],
        conversation_history=history,
        # Weak inference becomes a ranking bias (matching chunks float up
        # but non-matching chunks aren't dropped).
        metadata_bias=rewrite["metadata_bias"],
        # Phase 9: source filter (Slack / Gmail). None = all sources,
        # matching pre-Phase-9 behavior.
        allowed_sources=req.allowed_sources,
        # Phase 12: workspace_id reaches the recall layer so it can
        # pull matching structured memories (action items / decisions
        # / summaries / entities) alongside the HydraDB chunks.
        workspace_id=workspace.workspace_id,
        # Phase 4: route this query to the workspace's HydraDB
        # sub-tenant. ensure_workspace_sub_tenant materializes the row
        # value on the fly for any pre-Phase-4 workspace that missed
        # the migration backfill.
        hydradb_sub_tenant_id=ensure_workspace_sub_tenant(
            workspace_id=workspace.workspace_id,
        ),
    )

    # Mark non-cached on the way out so the UI can render a "fresh" badge
    # when it wants to, mirroring the cache-hit case. Attach the
    # date_query parsing record too so the UI can show "matched 'last week'".
    debug = dict(result.get("debug") or {})
    debug["cache_hit"] = False
    if resolved["date_query_debug"] is not None:
        debug["date_query"] = resolved["date_query_debug"]
    if rewrite["rewrite_debug"] is not None:
        debug["query_rewrite"] = rewrite["rewrite_debug"]
    # Surface cache bypass reason so the UI can show why a follow-up
    # didn't hit the cache.
    if history:
        debug["cache_bypassed"] = "conversation_history present"
    result = {**result, "debug": debug}

    # Only cache when we actually produced a useful answer (i.e. context
    # was found) AND there's no conversation history. Don't cache the
    # fallback string either — letting it retry next time is harmless
    # and the answer might change after fresh ingestion.
    if use_cache and result.get("answer") and result["answer"] != INSUFFICIENT_CONTEXT_ANSWER:
        cache_put(cache_key, result)
    return result


# ---------- Streaming variant ---------- #
def _sse_event(event: str, data: Any) -> bytes:
    """
    Format one SSE event. Both `event:` and `data:` lines are required for
    named events. The blank line ends the message. We always JSON-encode
    `data` so the frontend has a uniform parse path.
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


@app.post(
    "/api/query/stream",
    dependencies=[Depends(rate_limit_dependency)],
)
def query_stream(
    req: QueryRequest,
    request: Request,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> StreamingResponse:
    """
    Same request schema as POST /api/query, but the answer is streamed
    via Server-Sent Events. Event types:

      event: token   data: { "text": "<delta>" }
      event: done    data: { "answer": str, "sources": [...], "debug": {...} }
      event: error   data: { "detail": "...", "error_type": "..." }

    The frontend assembles tokens into the running answer and uses the
    final `done` event for source metadata. Cached responses are served
    by emitting a single `done` event immediately.
    """
    # ----- Resolve date inputs, history, query rewrite, cache key ----
    resolved = _resolve_date_filters(req)
    history = _normalize_history(req)
    rewrite = _resolve_query_rewrite(req)

    # Same bypass rule as /api/query: when conversation_history is set,
    # don't use or write the cache. Stateless streams still cache.
    use_cache = not history
    cache_key = _cache_key_from_request(req, resolved, rewrite) if use_cache else None
    cached = get_cached(cache_key) if use_cache else None

    # Snapshot the date_query + rewrite debug records once; both the
    # cached and the fresh paths attach them to their `done` event.
    date_query_debug = resolved["date_query_debug"]
    rewrite_debug = rewrite["rewrite_debug"]

    def _generator():
        # If we have a cached result, hand the client the whole answer in
        # one `token` event followed by `done`. The UI renders it the same
        # way as a streamed answer.
        if cached is not None:
            yield _sse_event("token", {"text": cached.get("answer", "")})
            cached_debug = dict(cached.get("debug", {}) or {})
            if date_query_debug is not None:
                cached_debug["date_query"] = date_query_debug
            if rewrite_debug is not None:
                cached_debug["query_rewrite"] = rewrite_debug
            yield _sse_event(
                "done",
                {
                    "answer": cached.get("answer", ""),
                    "sources": cached.get("sources", []),
                    "debug": cached_debug,
                },
            )
            return

        # ----- Prepare context first (HydraDB recall + source build) ---
        # NOTE: retrieval uses only the current question, never the
        # history — same invariant as /api/query.
        try:
            prepared = prepare_recall_context(
                question=req.question,
                top_k=req.top_k,
                mode=req.mode,
                channel=rewrite["effective_channel"],
                user=rewrite["effective_user"],
                document_type=req.document_type,
                start_timestamp=resolved["start_timestamp"],
                end_timestamp=resolved["end_timestamp"],
                metadata_bias=rewrite["metadata_bias"],
                # Phase 9: source filter (Slack / Gmail).
                allowed_sources=req.allowed_sources,
                # Phase 12: workspace_id for structured-memory lookup.
                workspace_id=workspace.workspace_id,
                # Phase 4: workspace-isolated recall.
                hydradb_sub_tenant_id=ensure_workspace_sub_tenant(
                    workspace_id=workspace.workspace_id,
                ),
            )
        except AppError as e:
            yield _sse_event(
                "error",
                {
                    "detail": e.detail,
                    "error_type": e.error_type,
                },
            )
            return
        except Exception as e:  # noqa: BLE001
            yield _sse_event(
                "error",
                {
                    "detail": "Unexpected server error.",
                    "error_type": "internal_error",
                },
            )
            logger.error('query_stream_unexpected_error', extra={'error': type(e).__name__})
            return

        if not prepared["ready"]:
            # No context found — emit the canonical fallback as one token,
            # then done. This keeps the frontend's state machine simple.
            yield _sse_event("token", {"text": INSUFFICIENT_CONTEXT_ANSWER})
            fallback_debug = {**prepared["fallback_debug"], "cache_hit": False}
            if date_query_debug is not None:
                fallback_debug["date_query"] = date_query_debug
            if rewrite_debug is not None:
                fallback_debug["query_rewrite"] = rewrite_debug
            if history:
                fallback_debug["cache_bypassed"] = "conversation_history present"
            yield _sse_event(
                "done",
                {
                    "answer": INSUFFICIENT_CONTEXT_ANSWER,
                    "sources": [],
                    "debug": fallback_debug,
                },
            )
            return

        # ----- Stream tokens from the LLM ------------------------------
        # `conversation_history` is passed to the LLM so it can resolve
        # references in the latest question. Retrieval already finished
        # using only the question.
        accumulated_parts: List[str] = []
        try:
            for piece in stream_grounded_answer(
                question=req.question,
                context=prepared["context_text"],
                mode=req.mode,
                conversation_history=history,
            ):
                accumulated_parts.append(piece)
                yield _sse_event("token", {"text": piece})
        except AppError as e:
            yield _sse_event(
                "error",
                {
                    "detail": e.detail,
                    "error_type": e.error_type,
                },
            )
            return
        except Exception as e:  # noqa: BLE001
            yield _sse_event(
                "error",
                {
                    "detail": "Unexpected LLM error.",
                    "error_type": "internal_error",
                },
            )
            logger.error('query_stream_llm_error', extra={'error': type(e).__name__})
            return

        # ----- Finalize and emit done ---------------------------------
        raw_answer = "".join(accumulated_parts).strip()
        finalized = finalize_answer(
            raw_answer=raw_answer,
            sources=prepared["sources"],
            top_k=req.top_k,
        )
        debug = {
            "chunks_returned": prepared["chunks_count"],
            "chunks_used": len(prepared["sources"]),
            "chunks_filtered_out": prepared["filtered_out"],
            "sources_before_clean": finalized["sources_before"],
            "sources_after_clean": finalized["sources_after"],
            "mode": req.mode,
            "retrieval_mode": prepared.get("retrieval_mode", req.mode),
            "exact_matches_found": prepared.get("exact_matches", 0),
            "query_terms": prepared.get("query_terms", []),
            "top_k": req.top_k,
            "cache_hit": False,
            "history_used": bool(history),
            "history_turns": len(history),
        }
        if date_query_debug is not None:
            debug["date_query"] = date_query_debug
        if rewrite_debug is not None:
            debug["query_rewrite"] = rewrite_debug
        if history:
            debug["cache_bypassed"] = "conversation_history present"
        full_payload = {
            "answer": finalized["answer"],
            "sources": finalized["cleaned_sources"],
            "debug": debug,
        }
        yield _sse_event("done", full_payload)

        # Cache the finalized result for next identical request, same
        # rules as /api/query: only when stateless AND not the no-context
        # fallback.
        if use_cache and full_payload["answer"] and full_payload["answer"] != INSUFFICIENT_CONTEXT_ANSWER:
            cache_put(cache_key, full_payload)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            # Disable buffering on proxies/nginx so events arrive immediately.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------- Slack Events API webhook ---------- #
# Public route — authenticated by Slack's HMAC signature, not X-API-Key.
# Slack expects us to ACK within 3 seconds, so we verify, queue the
# ingestion as a BackgroundTask, and return 200 immediately.
@app.post(
    "/slack/events",
    dependencies=[Depends(slack_webhook_rate_limit)],
)
async def slack_events(
    request: Request,
    background_tasks: BackgroundTasks,
    x_slack_signature: Optional[str] = Header(default=None, alias="X-Slack-Signature"),
    x_slack_request_timestamp: Optional[str] = Header(default=None, alias="X-Slack-Request-Timestamp"),
):
    # Read body BEFORE parsing so we can verify the signature over the
    # exact bytes Slack sent. Reparsing the JSON to bytes would change
    # whitespace and break the HMAC.
    body = await request.body()

    if not verify_slack_signature(body, x_slack_request_timestamp, x_slack_signature):
        # Slack will keep retrying briefly. The 401 makes the failure
        # visible in Slack's event dashboard so misconfigurations show up
        # there instead of disappearing into the logs.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Slack signature.",
        )

    # Parse the body now that we trust it. If the body isn't valid JSON
    # we return 400 — Slack should never send anything else, but it's the
    # right response.
    try:
        payload = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Body is not valid JSON.",
        )

    # ----- URL verification handshake -----
    # When you add or edit the Events Subscription URL in Slack, Slack
    # POSTs {"type": "url_verification", "challenge": "..."} and expects
    # us to echo the challenge back as plain text (or as {"challenge": "..."}).
    if payload.get("type") == "url_verification":
        challenge = payload.get("challenge", "")
        return PlainTextResponse(content=challenge, status_code=200)

    # ----- Event callback -----
    if payload.get("type") == "event_callback":
        # Idempotency: Slack retries failed deliveries up to 3 times within
        # an hour, so we dedupe by event_id. The webhook still ACKs 200 on
        # duplicates so Slack stops retrying.
        event_id = payload.get("event_id") or ""
        if _event_already_seen(event_id):
            logger.debug('slack_events_duplicate', extra={'event_id': event_id})
            return JSONResponse(content={"ok": True})

        # Phase 5: pass the FULL payload (not just the inner event) so
        # the realtime handler can read `team_id` / `authorizations`
        # and route to the right workspace. The handler then resolves
        # the workspace's bot_token + HydraDB sub_tenant_id internally
        # — we never need workspace context on the webhook itself.
        background_tasks.add_task(process_slack_event, payload)
        return JSONResponse(content={"ok": True})

    # Unknown top-level type. Ack 200 so Slack stops retrying.
    logger.debug('slack_events_unknown_type', extra={'payload_type': payload.get('type')})
    return JSONResponse(content={"ok": True})


# ---------- Phase 2 request models (chat history + saved answers) ---------- #
class ChatSessionCreate(BaseModel):
    """Body for POST /api/chat/sessions."""

    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(default=None, max_length=200)


class ChatMessageCreate(BaseModel):
    """Body for POST /api/chat/sessions/{session_id}/messages."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(default="", max_length=200_000)
    sources: Optional[List[Dict[str, Any]]] = None


class SavedAnswerCreate(BaseModel):
    """Body for POST /api/saved-answers."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(default="", max_length=5000)
    answer: str = Field(default="", max_length=200_000)
    sources: Optional[List[Dict[str, Any]]] = None
    mode: Optional[str] = Field(default=None, max_length=64)
    filters: Optional[Dict[str, Any]] = None
    debug: Optional[Dict[str, Any]] = None


# ---------- Phase 3 request models (Slack Connect) ---------- #
class SlackChannelSelection(BaseModel):
    """Body for POST /api/slack/channels."""

    model_config = ConfigDict(extra="forbid")
    selected_channel_ids: List[str] = Field(default_factory=list)
    bot_message_channel_ids: List[str] = Field(default_factory=list)


# ---------- Phase 8: Gmail request models ---------- #
class GmailLabelSelection(BaseModel):
    """Body for POST /api/gmail/labels."""

    model_config = ConfigDict(extra="forbid")
    connection_id: str
    selected_label_ids: List[str] = Field(default_factory=list)


class GmailIngestRequest(BaseModel):
    """Body for POST /api/gmail/ingest."""

    model_config = ConfigDict(extra="forbid")
    connection_id: str


# ---------- Admin status (light, read-only) ---------- #
@app.get(
    "/api/admin/status",
    dependencies=[Depends(require_api_key)],
)
def admin_status() -> Dict[str, Any]:
    """
    Lightweight ingestion-status snapshot for the frontend admin card.
    Does NOT expose any per-document content — only counters and flags.
    """
    return admin_status_snapshot(scheduler_enabled=auto_ingest_enabled())


# ---------- User profile + workspaces (Phase 1 multi-user) ---------- #
@app.get("/api/me", dependencies=[Depends(auth_rate_limit)])
def me(user: SupabaseUser = Depends(require_user)) -> Dict[str, Any]:
    """
    Return the JWT-verified caller's identity. Useful for the frontend
    to confirm auth without fetching the workspaces list.
    """
    return {"id": user.id, "email": user.email}


@app.get("/api/me/workspaces", dependencies=[Depends(auth_rate_limit)])
def my_workspaces(
    user: SupabaseUser = Depends(require_user),
) -> List[Dict[str, Any]]:
    """
    Return the workspaces the caller is a member of, with their role.
    The frontend uses this to populate the workspace switcher and to
    decide which X-Workspace-Id to send on subsequent /api/query calls.
    """
    return list_user_workspaces(user_id=user.id)


# ---------- Chat sessions (Phase 2) ---------- #
# Sessions are personal-within-workspace: shared organization but
# private threads. Messages live under sessions and are read-scoped by
# the same rule. All routes require Supabase JWT + X-Workspace-Id —
# enforced once by the require_workspace dependency.


@app.get("/api/chat/sessions")
def chat_sessions_list(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> List[Dict[str, Any]]:
    """Return the caller's chat sessions in this workspace, newest first."""
    return list_chat_sessions(
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
    )


@app.post("/api/chat/sessions", status_code=status.HTTP_201_CREATED)
def chat_sessions_create(
    req: ChatSessionCreate,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """Create a new chat session in the caller's workspace."""
    row = create_chat_session(
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
        title=req.title or "New chat",
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not create chat session.",
        )
    return row


@app.get("/api/chat/sessions/{session_id}/messages")
def chat_session_messages_list(
    session_id: str,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> List[Dict[str, Any]]:
    """
    Return the messages in a session, oldest first.

    Scoped to the caller's own session inside the active workspace; an
    unknown / forbidden session_id returns an empty list rather than a
    404 so it can't be used for existence-probing.
    """
    return list_chat_messages(
        session_id=session_id,
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
    )


@app.post(
    "/api/chat/sessions/{session_id}/messages",
    status_code=status.HTTP_201_CREATED,
)
def chat_session_messages_create(
    session_id: str,
    req: ChatMessageCreate,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """Append a user / assistant message to a session the caller owns."""
    row = create_chat_message(
        session_id=session_id,
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
        role=req.role,
        content=req.content,
        sources=req.sources,
    )
    if row is None:
        # Distinguish "session doesn't belong to caller" from real DB
        # failure isn't worth a probe-able 404 — both map to the same
        # client-visible refusal.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not append message to session.",
        )
    return row


# ---------- Saved answers (Phase 2) ---------- #


@app.get("/api/saved-answers")
def saved_answers_list(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> List[Dict[str, Any]]:
    """Return the caller's saved answers in this workspace, newest first."""
    return list_saved_answers(
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
    )


@app.post("/api/saved-answers", status_code=status.HTTP_201_CREATED)
def saved_answers_create(
    req: SavedAnswerCreate,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """Save an assistant answer to the caller's bookmarks."""
    row = create_saved_answer(
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
        question=req.question,
        answer=req.answer,
        sources=req.sources,
        mode=req.mode,
        filters=req.filters,
        debug=req.debug,
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not save answer.",
        )
    return row


@app.delete("/api/saved-answers/{saved_id}")
def saved_answers_delete(
    saved_id: str,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """Delete a saved answer the caller owns."""
    removed = delete_saved_answer(
        saved_id=saved_id,
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
    )
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved answer not found.",
        )
    return {"id": saved_id, "deleted": True}


# ---------- Phase 13: share links + workspace status ---------- #
# Three authenticated routes (create/list/revoke) and one PUBLIC route
# (read by token). The public route is the only place in the API
# surface that doesn't require auth -- its security comes entirely
# from the share_token's entropy (256 bits via secrets.token_urlsafe(32)).
#
# Public-route projection rules:
#   - NEVER include workspace_id, created_by, user_id, debug
#   - NEVER include any cross-record data
#   - Revoked or expired -> 404 (collapsed, never distinguishable)


# Built once so the share-link URL builder doesn't re-read env each
# call. It still respects FRONTEND_BASE_URL override via the helper.
def _share_url_for(token: str) -> str:
    """Build the user-visible share URL the frontend renders the
    standalone read view at."""
    base = _frontend_base_url()
    return f"{base}/shared/{token}"


@app.post(
    "/api/saved-answers/{saved_id}/share",
    status_code=status.HTTP_201_CREATED,
)
def saved_answers_share(
    saved_id: str,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Mint a share link for a saved answer the caller owns. The token
    is opaque (URL-safe base64 of 32 random bytes -> 43 chars) and
    serves as the public read credential.

    Ownership check: we require the saved answer to live in this
    workspace. We do NOT require the caller to be the same user_id
    that originally saved it -- workspace members can share each
    other's bookmarks. (Revoke is still tied to created_by.)
    """
    saved = get_saved_answer(
        saved_id=saved_id,
        workspace_id=workspace.workspace_id,
    )
    if not saved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Saved answer not found.",
        )
    token = secrets.token_urlsafe(32)
    row = create_share_link(
        workspace_id=workspace.workspace_id,
        saved_answer_id=saved_id,
        created_by=workspace.user.id,
        share_token=token,
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not create share link.",
        )
    return {
        "id": row.get("id"),
        "share_token": token,
        "url": _share_url_for(token),
        "saved_answer_id": saved_id,
        "created_at": row.get("created_at"),
    }


@app.get("/api/saved-answers/{saved_id}/shares")
def saved_answers_shares_list(
    saved_id: str,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    List active (non-revoked) shares for one saved answer. The
    frontend uses this to render a "this answer is shared" badge +
    revoke button on saved answers.
    """
    rows = list_share_links_for_workspace(
        workspace_id=workspace.workspace_id,
        saved_answer_id=saved_id,
    )
    # Hide revoked shares from the listing -- once revoked, a share
    # link is effectively gone from the user's perspective.
    active = [
        {
            "id": r.get("id"),
            "share_token": r.get("share_token"),
            "url": _share_url_for(r.get("share_token") or ""),
            "created_at": r.get("created_at"),
            "expires_at": r.get("expires_at"),
            "created_by": r.get("created_by"),
        }
        for r in rows
        if not r.get("revoked_at")
    ]
    return {"shares": active}


@app.delete("/api/saved-answers/share/{share_token}")
def saved_answers_share_revoke(
    share_token: str,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Revoke a share link. Scoped to (workspace_id, created_by) so one
    user can't revoke another user's share. Returns 404 if the token
    doesn't exist in this workspace or wasn't created by the caller --
    we don't distinguish those cases to avoid leaking existence.
    """
    ok = revoke_share_link(
        share_token=share_token,
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Share link not found.",
        )
    return {"revoked": True}


@app.get(
    "/api/shared/{share_token}",
    dependencies=[Depends(public_share_rate_limit)],
)
def shared_answer_public(
    share_token: str,
) -> Dict[str, Any]:
    """
    PUBLIC read-only fetch. No auth -- the share_token IS the
    credential. The response NEVER includes workspace_id, created_by,
    debug payload, or any cross-record data.

    Returns 404 for: missing / revoked / expired tokens. Collapsed
    so an attacker probing tokens can't distinguish the three cases.
    """
    link = get_share_link_by_token(share_token=share_token)
    if not link:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shared link not found.",
        )
    saved = get_saved_answer(
        saved_id=link.get("saved_answer_id") or "",
        workspace_id=link.get("workspace_id") or "",
    )
    if not saved:
        # The saved answer was deleted but the share link row outlived
        # it (FK cascade should prevent this in practice; defensive).
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Shared link not found.",
        )
    # PUBLIC projection allowlist -- nothing here ties back to a
    # workspace or user.
    return {
        "question": saved.get("question") or "",
        "answer": saved.get("answer") or "",
        "sources": saved.get("sources") or [],
        "mode": saved.get("mode"),
        "created_at": saved.get("created_at"),
    }


@app.get("/api/workspace/status")
def workspace_status(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Lightweight snapshot the frontend uses to render the
    "connectors + sync health" hint bar. Workspace-scoped, no
    secrets, no internal details.

    Shape:
      {
        "slack": {
            "connected":         bool,
            "channels_selected": int,
            "scheduler_enabled": bool,
        },
        "gmail": {
            "connection_count": int,
            "labels_selected":  int,
            "last_synced_at":   ISO str | null,
        },
      }
    """
    # ---- Slack ----
    slack_install = get_slack_installation(workspace_id=workspace.workspace_id)
    slack_channels = list_selected_channel_ids(
        workspace_id=workspace.workspace_id,
    )
    slack_status = {
        "connected": bool(slack_install),
        "channels_selected": len(slack_channels),
        "scheduler_enabled": auto_ingest_enabled(),
    }

    # ---- Gmail ----
    # Aggregate per-connection summary into a single workspace-level
    # snapshot: max last_synced_at, summed labels_selected. This is
    # the smallest payload the hint bar can render without further
    # round-trips.
    gmail_conns = list_gmail_connections_public(
        workspace_id=workspace.workspace_id,
    )
    last_synced: Optional[str] = None
    labels_total = 0
    for c in gmail_conns:
        cid = c.get("id") or ""
        if not cid:
            continue
        try:
            summary = get_gmail_connection_sync_summary(
                workspace_id=workspace.workspace_id,
                gmail_connection_id=cid,
            )
        except Exception as _e:  # noqa: BLE001
            logger.warning(
                "workspace_status_gmail_summary_failed",
                extra={"connection_id": cid, "error": type(_e).__name__},
            )
            summary = {"last_synced_at": None, "labels_synced": 0}
        ts = summary.get("last_synced_at")
        if ts and (last_synced is None or str(ts) > str(last_synced)):
            last_synced = ts
        try:
            label_ids = list_selected_gmail_label_ids(
                workspace_id=workspace.workspace_id,
                gmail_connection_id=cid,
            )
            labels_total += len(label_ids or [])
        except Exception as _e:  # noqa: BLE001
            logger.warning(
                "workspace_status_gmail_labels_failed",
                extra={"connection_id": cid, "error": type(_e).__name__},
            )
    gmail_status = {
        "connection_count": len(gmail_conns),
        "labels_selected": labels_total,
        "last_synced_at": last_synced,
    }

    return {"slack": slack_status, "gmail": gmail_status}


# ---------- Phase 15: analytics ---------- #
# Four read-only endpoints under /api/analytics. Each one delegates
# to the analytics_store / analytics_intelligence layer, which is
# workspace-scoped and degrades gracefully on Supabase failure.
# None of these endpoints write -- they're pure projections over
# data already on disk (analytics_events + extracted_memories).


def _clamp_days(value: Any, default: int = 7, max_days: int = 90) -> int:
    """Defensive parse of the ?days= query param."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, max_days))


@app.get("/api/analytics/overview")
def analytics_overview(
    days: int = 7,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Combined query + ingest + retrieval-failure stats over the
    configured window. The UI renders this as the top of the
    analytics panel.
    """
    from analytics_store import (  # local import keeps boot fast
        aggregate_ingest_stats,
        aggregate_query_stats,
        aggregate_retrieval_failure_stats,
    )

    safe_days = _clamp_days(days)
    return {
        "window_days": safe_days,
        "query": aggregate_query_stats(
            workspace_id=workspace.workspace_id,
            days=safe_days,
        ),
        "ingest": aggregate_ingest_stats(
            workspace_id=workspace.workspace_id,
            days=safe_days,
        ),
        "retrieval_failures": aggregate_retrieval_failure_stats(
            workspace_id=workspace.workspace_id,
            days=safe_days,
        ),
    }


@app.get("/api/analytics/topics")
def analytics_topics(
    days: int = 30,
    top_n: int = 10,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Top entities + their co-occurring entities (memory graph) plus
    a cluster count. Drives the "Top Topics" card.
    """
    from analytics_intelligence import topic_overview

    safe_days = _clamp_days(days, default=30)
    safe_top = max(1, min(int(top_n or 10), 50))
    return topic_overview(
        workspace_id=workspace.workspace_id,
        days=safe_days,
        top_n=safe_top,
    )


@app.get("/api/analytics/timeline")
def analytics_timeline(
    entity: Optional[str] = None,
    kind: Optional[str] = None,
    days: int = 90,
    limit: int = 50,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Chronological memory rows for one entity / kind. Used by the
    timeline view when the user clicks a topic from the topics card.

    `kind` accepts the same memory kinds as the /api/memories filter:
    "action_item" | "decision" | "summary" | "entity". Multiple kinds
    can be passed comma-separated.
    """
    from analytics_intelligence import reconstruct_timeline

    kinds = None
    if kind:
        kinds = [k.strip() for k in kind.split(",") if k.strip()]
    safe_days = _clamp_days(days, default=90, max_days=365)
    safe_limit = max(1, min(int(limit or 50), 200))
    rows = reconstruct_timeline(
        workspace_id=workspace.workspace_id,
        entity=entity,
        kinds=kinds,
        days=safe_days,
        limit=safe_limit,
    )
    return {"rows": rows}


@app.get("/api/analytics/insights")
def analytics_insights(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Combined proactive-intelligence payload:
      stale action items, dormant projects, surging entities,
      plus the recurring-pattern list. The UI renders these as
      separate cards in the analytics panel.
    """
    from analytics_intelligence import (
        proactive_insights,
        recurring_patterns,
    )

    insights = proactive_insights(workspace_id=workspace.workspace_id)
    recurring = recurring_patterns(
        workspace_id=workspace.workspace_id,
        days=7,
        min_mentions=3,
    )
    return {**insights, "recurring": recurring}


# ---------- Phase 16: memory intelligence ---------- #
# Three read-only endpoints under /api/intelligence. Each one
# delegates to memory_intelligence, which derives everything on read
# from extracted_memories -- workspace-scoped, defensive, no writes.


@app.get("/api/intelligence/graph")
def intelligence_graph(
    days: int = 90,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Canonical entity relationship graph: people / projects / services
    / channels / decisions / action items as nodes, recurrence-x-
    recency-weighted co-occurrence edges, cross-source flags, and the
    full alias map for traceability.
    """
    from memory_intelligence import relationship_graph  # noqa: PLC0415

    return relationship_graph(
        workspace_id=workspace.workspace_id,
        days=_clamp_days(days, default=90, max_days=365),
    )


@app.get("/api/intelligence/projects")
def intelligence_projects(
    days: int = 90,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Inferred project intelligence: active/dormant status, owners,
    timeline, linked decisions, blockers, unresolved tasks.
    """
    from memory_intelligence import project_intelligence  # noqa: PLC0415

    return {
        "projects": project_intelligence(
            workspace_id=workspace.workspace_id,
            days=_clamp_days(days, default=90, max_days=365),
        )
    }


@app.get("/api/intelligence/conversation")
def intelligence_conversation(
    decision: str,
    days: int = 180,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Conversation reconstruction: walk backward from the decision
    matching `decision` through the pre-dating co-occurring memories.
    """
    from memory_intelligence import reconstruct_conversation  # noqa: PLC0415

    return reconstruct_conversation(
        workspace_id=workspace.workspace_id,
        decision=decision,
        days=_clamp_days(days, default=180, max_days=365),
    )


# ---------- Slack Connect (Phase 3) ---------- #
# Per-workspace Slack OAuth. The bot token Slack returns is stored in
# Supabase (slack_installations.bot_token) and never echoed back to
# the frontend. The frontend only needs to know that Slack is connected
# (team name + when), then can list channels and toggle which ones to
# ingest.


def _frontend_base_url() -> str:
    """
    Where the OAuth callback should redirect after success/failure.
    Falls back to the first allowed CORS origin (which is where the
    frontend dev server runs) so a missing FRONTEND_BASE_URL doesn't
    bounce the user to localhost:8000.
    """
    explicit = (os.getenv("FRONTEND_BASE_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    if ALLOWED_ORIGINS:
        return ALLOWED_ORIGINS[0].rstrip("/")
    return "http://localhost:5173"


@app.get("/api/slack/connect-url")
def slack_connect_url(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, str]:
    """
    Return a one-shot Slack OAuth authorize URL the frontend should
    redirect the user to. The URL embeds a signed state token binding
    this attempt to the caller's workspace + user.

    503 if Slack OAuth isn't configured in the env — the frontend uses
    this status to render a "Slack integration is disabled" message
    rather than a broken Connect button.
    """
    if not slack_oauth_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Slack OAuth is not configured on the server.",
        )
    try:
        url = build_connect_url(
            workspace_id=workspace.workspace_id,
            user_id=workspace.user.id,
        )
    except RuntimeError as e:
        # SLACK_OAUTH_STATE_SECRET missing — 503, not 500, so the
        # client message is consistent with "OAuth disabled".
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    return {"url": url}


@app.get("/api/slack/oauth/callback")
def slack_oauth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    """
    Slack redirects the user's browser here after they approve (or
    deny) the install. We cannot require an Authorization header on
    this route — Slack's redirect doesn't carry one. Instead we:

      1. Verify the state token's HMAC signature + expiry. The token
         was minted by /api/slack/connect-url, which DID require
         workspace auth, so a valid state binds this callback to a
         specific (workspace_id, user_id).
      2. Exchange the code for a bot token via Slack's oauth.v2.access.
      3. Upsert the installation into slack_installations.
      4. Redirect the user back to the frontend with a short status
         query string so the UI can show a toast.

    All failure modes redirect with `?slack_connect=error&reason=...`
    so the frontend can surface a clean message instead of a JSON
    blob in the address bar.
    """
    frontend = _frontend_base_url()

    if error:
        # User canceled or Slack returned an error.
        return _redirect_with_status(frontend, "error", error)
    if not code or not state:
        return _redirect_with_status(frontend, "error", "missing_params")

    payload = verify_oauth_state(state)
    if not payload:
        return _redirect_with_status(frontend, "error", "bad_state")

    data = exchange_code(code)
    if not data:
        return _redirect_with_status(frontend, "error", "exchange_failed")

    row = installation_from_oauth_response(data)
    if not row.get("slack_team_id") or not row.get("bot_token"):
        return _redirect_with_status(frontend, "error", "incomplete_install")

    saved = upsert_slack_installation(
        workspace_id=payload["workspace_id"],
        slack_team_id=row["slack_team_id"],
        slack_team_name=row["slack_team_name"],
        bot_user_id=row["bot_user_id"],
        bot_token=row["bot_token"],
        scopes=row["scopes"],
        # Phase 6+ audit: the verified OAuth state binds the caller's
        # Supabase user_id. Forwarding it populates installed_by in
        # the production schema -- a nullable column, so callers that
        # don't pass it (older tests) continue to work.
        installed_by=payload.get("user_id") or None,
    )
    if not saved:
        return _redirect_with_status(frontend, "error", "persist_failed")

    return _redirect_with_status(frontend, "ok", row["slack_team_name"] or "")


def _oauth_redirect(frontend: str, connector: str, result: str, reason: str):
    """
    Build the post-callback redirect URL the user's browser lands on.
    `connector` is the query-param key (e.g. "slack_connect", "gmail_connect")
    so the frontend can dispatch by connector type.
    `reason` is never untrusted: on success it's a team name or email
    (validated upstream); on failure it's a short fixed code we choose.
    """
    from fastapi.responses import RedirectResponse  # noqa: PLC0415

    qs = urlencode_safely({connector: result, "reason": reason})
    return RedirectResponse(url=f"{frontend}/?{qs}", status_code=302)


def _redirect_with_status(frontend: str, result: str, reason: str):
    return _oauth_redirect(frontend, "slack_connect", result, reason)


def urlencode_safely(d: Dict[str, str]) -> str:
    """
    Tiny urlencode wrapper that drops empty/None values so the final
    URL stays tidy. We don't want `&reason=` cluttering the query
    string on a successful connect where reason happens to be blank.
    """
    from urllib.parse import urlencode  # noqa: PLC0415

    cleaned = {k: v for k, v in d.items() if v not in (None, "")}
    return urlencode(cleaned)


@app.get("/api/slack/channels")
def slack_channels_list(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Return the current Slack-channel picker state for this workspace.

    If Slack isn't connected yet, returns `connected: false` and an
    empty channel list — the frontend uses that to show the Connect
    button instead of a picker.

    If Slack IS connected, we refresh the channel list from Slack's
    API (upserting into slack_channels) BEFORE reading it back so the
    user always sees the current state of their Slack workspace.
    Selection state is preserved across the refresh — only `name` and
    `is_archived` get updated.
    """
    install = get_slack_installation(workspace_id=workspace.workspace_id)
    if not install:
        return {
            "connected": False,
            "team_name": "",
            "channels": [],
        }

    bot_token = (install.get("bot_token") or "").strip()
    if bot_token:
        try:
            fresh = list_slack_channels(bot_token)
        except Exception:  # noqa: BLE001
            fresh = []
        if fresh:
            # Production schema's slack_channels.installation_id FK
            # is non-null. Forward the installation row's id so the
            # upsert payload satisfies it. The kwarg is optional
            # (older dev databases without the column still work).
            upsert_slack_channels(
                workspace_id=workspace.workspace_id,
                channels=fresh,
                installation_id=install.get("id"),
            )

    rows = list_workspace_channels(workspace_id=workspace.workspace_id)
    return {
        "connected": True,
        "team_name": install.get("slack_team_name") or "",
        "channels": rows,
    }


@app.post("/api/slack/channels")
def slack_channels_save(
    req: SlackChannelSelection,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Replace the workspace's selected-channel set. Channels not present
    in the request are flipped to is_selected=false; channels listed
    are flipped to is_selected=true. We do NOT allow inserting brand
    new channel rows here — the rows must already exist (created by
    the previous GET, which refreshes from Slack).
    """
    ok = set_selected_channels(
        workspace_id=workspace.workspace_id,
        selected_ids=req.selected_channel_ids,
        bot_message_ids=req.bot_message_channel_ids,
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not update channel selection.",
        )
    return {"selected_count": len(req.selected_channel_ids)}


@app.post(
    "/api/slack/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(slack_ingest_rate_limit)],
)
def slack_ingest(
    background_tasks: BackgroundTasks,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Kick off an ingestion run for this workspace's selected channels.
    Returns immediately with 202; the actual work happens in a
    BackgroundTask so the request doesn't block on Slack/HydraDB I/O.

    The runner re-uses the existing ingestion primitives — same chunk
    layout, same dedupe state file. Moving state into Supabase is
    explicitly out of scope for Phase 3.
    """
    install = get_slack_installation(workspace_id=workspace.workspace_id)
    if not install:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Slack is not connected for this workspace.",
        )
    bot_token = (install.get("bot_token") or "").strip()
    if not bot_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stored Slack installation is missing a bot token.",
        )

    channel_settings = list_selected_channel_settings(
        workspace_id=workspace.workspace_id,
    )
    if not channel_settings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No channels selected for ingestion.",
        )
    channel_ids = [c["slack_channel_id"] for c in channel_settings]
    channel_bot_messages = {c["slack_channel_id"]: c["include_bot_messages"] for c in channel_settings}

    # Phase 4: resolve (or lazy-create) the workspace's HydraDB
    # sub-tenant before scheduling the background task. A blank value
    # here means a DB error occurred — we refuse rather than fall back
    # to the global sub-tenant, which would leak this workspace's
    # Slack content into the shared bucket.
    sub_tenant = ensure_workspace_sub_tenant(
        workspace_id=workspace.workspace_id,
    )
    if not sub_tenant:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not resolve workspace HydraDB tenant.",
        )

    background_tasks.add_task(
        run_workspace_ingest,
        workspace_id=workspace.workspace_id,
        bot_token=bot_token,
        channel_ids=channel_ids,
        channel_bot_messages=channel_bot_messages,
        hydradb_sub_tenant_id=sub_tenant,
    )
    return {
        "status": "started",
        "channels_queued": len(channel_ids),
    }


@app.delete("/api/slack/disconnect")
def slack_disconnect(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, bool]:
    """
    Disconnect the Slack workspace: revoke the bot token with Slack,
    then delete the installation row (cascades to channel records).

    Token revocation is fire-and-forget — local cleanup proceeds even
    if Slack's auth.revoke returns an error (e.g. token already expired).
    """
    install = get_slack_installation(workspace_id=workspace.workspace_id)
    if not install:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Slack connection found for this workspace.",
        )

    # Revoke with Slack first so the token can't be reused after we
    # delete our local record.  Failure here is non-fatal.
    revoke_bot_token(install.get("bot_token") or "")

    deleted = delete_slack_installation(workspace_id=workspace.workspace_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not remove the Slack installation.",
        )
    return {"disconnected": True}


# =====================================================================
# Phase 8: Gmail connector routes
# =====================================================================
# Mirrors the Slack route surface so the frontend can plug in with the
# same mental model. Differences:
#   - One workspace can host MULTIPLE Gmail connections (a personal +
#     a shared mailbox, for instance). The Slack model is one
#     installation per workspace.
#   - Labels (Gmail's equivalent of channels) belong to a CONNECTION,
#     not directly to the workspace. So the labels / ingest routes
#     take a connection_id query/body param.
#
# All routes except the OAuth callback require auth + workspace.


@app.get(
    "/api/gmail/connect-url",
    dependencies=[Depends(auth_rate_limit)],
)
def gmail_connect_url(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, str]:
    """
    Build the Google OAuth URL the frontend should redirect the user
    to. Returns 503 when Gmail OAuth env vars aren't configured -- the
    connector is opt-in per deployment.
    """
    if not gmail_oauth_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gmail OAuth is not configured on this server.",
        )
    url = gmail_build_connect_url(
        workspace_id=workspace.workspace_id,
        user_id=workspace.user.id,
    )
    return {"url": url}


@app.get("/api/gmail/oauth/callback")
def gmail_oauth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
):
    """
    Google redirects the user's browser here after they approve or
    deny consent. Cannot require an Authorization header -- Google's
    redirect doesn't carry one. Instead we verify the HMAC-signed
    state, which binds this callback to a specific (workspace_id,
    user_id) minted by /api/gmail/connect-url.

    All failures redirect with `?gmail_connect=error&reason=...` so
    the frontend can surface a clean toast.
    """
    frontend = _frontend_base_url()

    if error:
        return _gmail_redirect_with_status(frontend, "error", error)
    if not code or not state:
        return _gmail_redirect_with_status(frontend, "error", "missing_params")

    payload = verify_gmail_oauth_state(state)
    if not payload:
        return _gmail_redirect_with_status(frontend, "error", "bad_state")

    token_resp = gmail_exchange_code(code)
    if not token_resp:
        return _gmail_redirect_with_status(frontend, "error", "exchange_failed")

    user_info = gmail_fetch_user_info(token_resp.get("access_token") or "")
    if not user_info:
        return _gmail_redirect_with_status(frontend, "error", "userinfo_failed")

    install = gmail_installation_from_token_response(token_resp, user_info)
    if not install.get("google_user_id") or not install.get("email"):
        return _gmail_redirect_with_status(frontend, "error", "incomplete_install")

    saved = upsert_gmail_connection(
        workspace_id=payload["workspace_id"],
        google_user_id=install["google_user_id"],
        email=install["email"],
        access_token=install["access_token"],
        refresh_token=install["refresh_token"],
        token_expiry=install["token_expiry"],
        scopes=install["scopes"],
    )
    if not saved:
        return _gmail_redirect_with_status(frontend, "error", "persist_failed")

    return _gmail_redirect_with_status(frontend, "ok", install["email"])


def _gmail_redirect_with_status(frontend: str, result: str, reason: str):
    return _oauth_redirect(frontend, "gmail_connect", result, reason)


@app.get("/api/gmail/connections")
def gmail_connections_list(
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    List every Gmail connection in this workspace. The PUBLIC
    projection is used -- tokens are NEVER returned.

    Phase 11: each connection is enriched with a `sync_summary`
    sub-object carrying `{last_synced_at, labels_synced}`. The
    timestamp is the most-recent `last_synced_at` across the
    connection's per-label rows in gmail_ingestion_state; the
    counter is how many labels have a recorded sync. The frontend
    uses this to render "Last synced 5 min ago" without a second
    round-trip. No tokens, no message ids, no PII.
    """
    connections = list_gmail_connections_public(
        workspace_id=workspace.workspace_id,
    )
    enriched: List[Dict[str, Any]] = []
    for conn in connections:
        cid = conn.get("id") or ""
        sync_summary: Dict[str, Any] = {
            "last_synced_at": None,
            "labels_synced": 0,
        }
        if cid:
            try:
                sync_summary = get_gmail_connection_sync_summary(
                    workspace_id=workspace.workspace_id,
                    gmail_connection_id=cid,
                )
            except Exception:  # noqa: BLE001 -- never block listing
                pass
        enriched.append({**conn, "sync_summary": sync_summary})
    return {"connections": enriched}


@app.delete("/api/gmail/connections/{connection_id}")
def gmail_connection_delete(
    connection_id: str,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, bool]:
    """Delete a Gmail connection (cascades to labels + ingestion state)."""
    ok = delete_gmail_connection(
        connection_id=connection_id,
        workspace_id=workspace.workspace_id,
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gmail connection not found.",
        )
    return {"deleted": True}


@app.get("/api/gmail/labels")
def gmail_labels_list(
    connection_id: Optional[str] = None,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Return the label picker state for one Gmail connection.

    Mirrors /api/slack/channels: refresh from Gmail (so newly-created
    labels show up), upsert into Supabase preserving is_selected, then
    return the stored rows. The refresh is best-effort -- a Gmail API
    blip falls through to the previously-stored set.
    """
    if not connection_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing connection_id query parameter.",
        )

    connection = get_gmail_connection(
        connection_id=connection_id,
        workspace_id=workspace.workspace_id,
    )
    if not connection:
        # Defensive: return an empty set rather than 404 so a deleted
        # connection on the picker side renders as "no labels".
        return {"connected": False, "labels": []}

    try:
        live_labels = list_gmail_labels_from_api(connection)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "gmail_labels_refresh_failed",
            extra={
                "workspace_id": workspace.workspace_id,
                "connection_id": connection_id,
                "error": type(e).__name__,
            },
        )
        live_labels = []

    if live_labels:
        upsert_gmail_labels(
            workspace_id=workspace.workspace_id,
            gmail_connection_id=connection_id,
            labels=live_labels,
        )

    stored = list_gmail_labels(
        workspace_id=workspace.workspace_id,
        gmail_connection_id=connection_id,
    )
    return {"connected": True, "labels": stored}


@app.post("/api/gmail/labels")
def gmail_labels_save(
    req: GmailLabelSelection,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """Replace the selected-label set for a Gmail connection."""
    # Phase 8 multi-tenant safety: verify the connection actually
    # belongs to THIS workspace before writing. set_selected_gmail_labels
    # already filters by workspace_id, so a foreign connection_id would
    # silently no-op -- but that's confusing UX AND it lets one
    # workspace probe for the existence of another's connection IDs.
    # 404 closes both leaks.
    existing = get_gmail_connection_public(
        connection_id=req.connection_id,
        workspace_id=workspace.workspace_id,
    )
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gmail connection not found.",
        )

    ok = set_selected_gmail_labels(
        workspace_id=workspace.workspace_id,
        gmail_connection_id=req.connection_id,
        selected_label_ids=req.selected_label_ids,
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not persist Gmail label selection.",
        )
    return {"selected_count": len(req.selected_label_ids)}


@app.post(
    "/api/gmail/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(slack_ingest_rate_limit)],  # reuse "ingest" bucket
)
def gmail_ingest(
    req: GmailIngestRequest,
    background_tasks: BackgroundTasks,
    workspace: WorkspaceContext = Depends(require_workspace),
) -> Dict[str, Any]:
    """
    Kick off a Gmail ingestion run for one connection's selected labels.
    Returns immediately with 202; the actual work happens in a
    BackgroundTask so the request doesn't block on Gmail / HydraDB I/O.
    """
    connection = get_gmail_connection(
        connection_id=req.connection_id,
        workspace_id=workspace.workspace_id,
    )
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gmail connection not found.",
        )

    label_ids = list_selected_gmail_label_ids(
        workspace_id=workspace.workspace_id,
        gmail_connection_id=req.connection_id,
    )
    if not label_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No labels selected for ingestion.",
        )

    # Phase 4-style: resolve (or lazy-create) the workspace's HydraDB
    # sub-tenant before kicking off. A blank value means a DB error
    # occurred -- we refuse rather than fall back to the global tenant,
    # which would leak emails into the shared HydraDB bucket.
    sub_tenant = ensure_workspace_sub_tenant(
        workspace_id=workspace.workspace_id,
    )
    if not sub_tenant:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not resolve workspace HydraDB tenant.",
        )

    background_tasks.add_task(
        run_workspace_gmail_ingest,
        workspace_id=workspace.workspace_id,
        connection=connection,
        label_ids=label_ids,
        hydradb_sub_tenant_id=sub_tenant,
    )
    return {
        "status": "started",
        "labels_queued": len(label_ids),
    }
