"""
Shared pytest fixtures for the Second Brain test suite.

Sets up required env vars BEFORE any app imports so modules that read env
at import time (e.g. ingest_slack.py constants) see the test values.
All external systems (HydraDB, OpenAI, Slack, Supabase) are mocked — no
real network calls are made and no real credentials are needed.
"""

import os
import sys
from pathlib import Path
from typing import Generator, Optional
from unittest.mock import MagicMock, patch

import pytest

# ── Bootstrap sys.path so `import main` works from the tests/ directory ──
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ── Set required env vars BEFORE any app module is imported ──────────────
# These must be set early because some modules read env vars at module-load
# time (e.g. ingest_slack.py's MESSAGES_PER_CHANNEL / UPLOAD_BATCH_SIZE).
os.environ.setdefault("APP_API_KEY", "test-secret-key")
os.environ.setdefault("HYDRADB_API_KEY", "test-hydradb-key")
os.environ.setdefault("HYDRADB_TENANT_ID", "test-tenant-id")
os.environ.setdefault("HYDRADB_SUB_TENANT_ID", "test-sub-tenant")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test-token")
os.environ.setdefault("SLACK_SIGNING_SECRET", "test-slack-signing-secret")
os.environ.setdefault("SLACK_CHANNEL_IDS", "C123456,C789012")
# Supabase (Phase 1) — placeholder values so module imports don't trip.
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
# Slack Connect (Phase 3) — placeholder OAuth credentials.
os.environ.setdefault("SLACK_CLIENT_ID", "test-slack-client-id")
os.environ.setdefault("SLACK_CLIENT_SECRET", "test-slack-client-secret")
os.environ.setdefault(
    "SLACK_REDIRECT_URI",
    "http://127.0.0.1:8000/api/slack/oauth/callback",
)
os.environ.setdefault(
    "SLACK_OAUTH_STATE_SECRET",
    "test-state-secret-do-not-use-in-prod",
)
# Gmail Connect (Phase 8) — placeholder OAuth credentials. Separate
# state secret so tests can prove the Gmail flow doesn't accept tokens
# signed by the Slack secret (and vice versa).
os.environ.setdefault("GMAIL_CLIENT_ID", "test-gmail-client-id")
os.environ.setdefault("GMAIL_CLIENT_SECRET", "test-gmail-client-secret")
os.environ.setdefault(
    "GMAIL_REDIRECT_URI",
    "http://127.0.0.1:8000/api/gmail/oauth/callback",
)
os.environ.setdefault(
    "GMAIL_OAUTH_STATE_SECRET",
    "test-gmail-state-secret-do-not-use-in-prod",
)
os.environ.setdefault("FRONTEND_BASE_URL", "http://localhost:5173")
# Disable the query cache in tests so each test sees live responses.
os.environ["QUERY_CACHE_ENABLED"] = "false"
# Keep the scheduler off so tests don't spin up APScheduler threads.
os.environ["AUTO_INGEST"] = "false"
# Raise the rate limit so ordinary tests don't trip it.
os.environ["RATE_LIMIT_PER_5_MIN"] = "10000"


# ── Test identities used by the dep overrides below ───────────────────────
# Used by the bridge that lets existing tests keep their `X-API-Key`-style
# headers while the production routes now require a Supabase JWT.
TEST_USER_ID = "00000000-0000-0000-0000-000000000001"
TEST_USER_EMAIL = "test-user@example.com"
TEST_WORKSPACE_ID = "00000000-0000-0000-0000-00000000aaaa"
TEST_WORKSPACE_ROLE = "owner"


# ── Lazy app import (after env is set) ───────────────────────────────────
# We patch the lifespan-called functions so TestClient startup doesn't try
# to talk to real services.
@pytest.fixture(scope="session")
def _patched_app():
    """
    Import the FastAPI app with lifespan hooks patched out AND with the
    user-route auth dependencies overridden so existing tests that send
    only `X-API-Key: test-secret-key` still authenticate.

    The override accepts EITHER:
        - a non-empty Authorization header (real JWT path), OR
        - a non-empty X-API-Key header (legacy test-suite shim).
    Neither present -> 401, preserving `test_no_key_returns_401`-style
    assertions in the existing suite.

    Tests that need to exercise the *real* `require_user` /
    `require_workspace` dependencies (see test_supabase_auth.py and
    test_workspace_resolution.py) call them directly as plain functions
    rather than going through the FastAPI route — that path is uneffected
    by these overrides.
    """
    with (
        patch("startup.validate_required_env"),
        patch("scheduler.start_scheduler"),
        patch("scheduler.stop_scheduler"),
    ):
        from fastapi import Header, HTTPException, status  # noqa: PLC0415

        import main as _main  # noqa: PLC0415
        from auth_supabase import (  # noqa: PLC0415
            SupabaseUser,
            WorkspaceContext,
            require_user,
            require_workspace,
        )

        _TEST_USER = SupabaseUser(id=TEST_USER_ID, email=TEST_USER_EMAIL)

        def _override_require_user(
            authorization: Optional[str] = Header(default=None, alias="Authorization"),
            x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
        ) -> SupabaseUser:
            if (authorization and authorization.strip()) or (x_api_key and x_api_key.strip()):
                return _TEST_USER
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )

        def _override_require_workspace(
            authorization: Optional[str] = Header(default=None, alias="Authorization"),
            x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
            x_workspace_id: Optional[str] = Header(default=None, alias="X-Workspace-Id"),
        ) -> WorkspaceContext:
            user = _override_require_user(
                authorization=authorization,
                x_api_key=x_api_key,
            )
            workspace_id = (x_workspace_id or "").strip() or TEST_WORKSPACE_ID
            return WorkspaceContext(
                user=user,
                workspace_id=workspace_id,
                role=TEST_WORKSPACE_ROLE,
            )

        _main.app.dependency_overrides[require_user] = _override_require_user
        _main.app.dependency_overrides[require_workspace] = _override_require_workspace

        return _main.app


@pytest.fixture()
def client(_patched_app):
    """Per-test TestClient that triggers the (mocked) lifespan."""
    from fastapi.testclient import TestClient

    with (
        patch("main.validate_required_env"),
        patch("main.start_scheduler"),
        patch("main.stop_scheduler"),
    ):
        with TestClient(_patched_app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture(autouse=True)
def stub_workspace_sub_tenant():
    """
    Fail-closed tenancy means every /api/query and /api/*/ingest call
    resolves a HydraDB sub-tenant via require_workspace_sub_tenant.
    Stub it so the suite never hits live Supabase for that lookup.

    Tests that assert missing-tenant behavior can still override this
    by nesting their own patch(..., side_effect=WorkspaceTenantError()).
    """
    with patch(
        "main.require_workspace_sub_tenant",
        return_value="ws_testtenant",
    ):
        yield


# ── Common header helpers ─────────────────────────────────────────────────
TEST_API_KEY = "test-secret-key"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture()
def auth_headers():
    """
    Existing fixture — kept on X-API-Key so legacy tests that hit
    /api/admin/status (still gated by APP_API_KEY) continue to work
    unchanged, and user-route tests still authenticate via the conftest
    override above.
    """
    return dict(AUTH_HEADERS)


@pytest.fixture()
def jwt_auth_headers():
    """
    Header set the production frontend will send: Authorization bearer
    plus an X-Workspace-Id. The conftest dep override treats any non-empty
    Authorization as the test user. Used by new tests that want to send
    'real' headers without forging a JWT.
    """
    return {
        "Authorization": "Bearer test-jwt",
        "X-Workspace-Id": TEST_WORKSPACE_ID,
    }


# ── HydraDB mock ──────────────────────────────────────────────────────────
def _make_hydra_chunk(
    text: str = "This is a test chunk from Slack.",
    source_id: str = "doc-abc",
    filename: str = "slack_general_1000000000.md",
    score: float = 0.95,
    channel: str = "general",
    stable_key: str = "slack:C123:1000000000.000000",
) -> dict:
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": filename,
        "metadata": {
            "channel": channel,
            "stable_key": stable_key,
        },
    }


@pytest.fixture()
def mock_hydra_response():
    """A minimal but valid HydraDB recall response."""
    return {
        "chunks": [
            _make_hydra_chunk(
                text="Alice said the sprint deadline is Friday.",
                source_id="doc-001",
                stable_key="slack:C123:1000000001.000000",
            ),
            _make_hydra_chunk(
                text="Bob confirmed the API design in the meeting.",
                source_id="doc-002",
                stable_key="slack:C123:1000000002.000000",
            ),
        ]
    }


@pytest.fixture()
def mock_hydra_empty():
    """HydraDB response with no usable chunks."""
    return {"chunks": []}


# ── LLM mock ─────────────────────────────────────────────────────────────
@pytest.fixture()
def mock_llm_answer():
    return "Alice mentioned the sprint deadline is Friday [1]."


# ── Ingestion state helpers ───────────────────────────────────────────────
@pytest.fixture()
def tmp_state_path(tmp_path):
    """A temporary ingestion_state.json path (file does NOT exist yet)."""
    return tmp_path / "ingestion_state.json"


@pytest.fixture()
def populated_state_path(tmp_path):
    """A pre-populated ingestion state file for lookup tests."""
    import json

    state = {
        "version": 2,
        "entries": {
            "slack:C123:1000000001.000000": {
                "stable_key": "slack:C123:1000000001.000000",
                "filename": "slack_general_1000000001.md",
                "source_id": "doc-001",
                "channel_id": "C123",
                "channel_name": "general",
                "ts": "1000000001.000000",
                "thread_ts": None,
                "uploaded_at": "2026-01-01T00:00:00+00:00",
                "user_name": "Alice",
                "timestamp": "1000000001.000000",
                "snippet": "Sprint deadline is Friday",
                "permalink": "https://slack.com/archives/C123/p1000000001000000",
                "document_type": "message",
            },
            "slack_thread:C123:2000000001.000000": {
                "stable_key": "slack_thread:C123:2000000001.000000",
                "filename": "slack_general_2000000001.md",
                "source_id": "doc-002",
                "channel_id": "C123",
                "channel_name": "general",
                "ts": None,
                "thread_ts": "2000000001.000000",
                "uploaded_at": "2026-01-02T00:00:00+00:00",
                "user_name": "Bob",
                "timestamp": "2000000001.000000",
                "snippet": "API design finalized",
                "permalink": "https://slack.com/archives/C123/p2000000001000000",
                "document_type": "thread",
            },
        },
        "channels": {
            "C123": {"last_synced_ts": "2000000001.000000"},
            "_meta": {"last_ingested_at": "2026-01-02T00:00:00+00:00"},
        },
    }
    p = tmp_path / "ingestion_state.json"
    p.write_text(json.dumps(state))
    return p
