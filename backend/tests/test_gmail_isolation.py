"""
Phase 8 multi-tenant isolation tests.

The Phase 8 spec calls out a strict isolation guarantee:

    Workspace A:
      User A -> Gmail A -> HydraDB sub-tenant A
    Workspace B:
      User B -> Gmail B -> HydraDB sub-tenant B

These tests pin that contract explicitly, on top of the Slack-side
Phase 4 isolation tests. They verify:

  - One workspace cannot read another workspace's connections.
  - One workspace cannot delete another workspace's connection.
  - One workspace cannot save labels for another workspace's connection.
  - One workspace cannot kick off ingest against another workspace's
    connection.
  - Even when ingest IS legal, it MUST construct HydraDBClient with
    THIS workspace's sub-tenant id (never the global default).
  - OAuth state from workspace A cannot be replayed to land a
    connection under workspace B.
  - Tokens never appear in any API response.
  - The ingest runner refuses to start when no sub-tenant is available
    (it would otherwise leak emails into the shared HydraDB bucket).
"""

import time
from unittest.mock import MagicMock, patch

import pytest

# The conftest's jwt_auth_headers fixture authenticates as
# TEST_WS_ID below ("workspace A"). The tests then probe what happens
# when the requested connection / labels belong to a DIFFERENT
# workspace ("workspace B").
WORKSPACE_A = "00000000-0000-0000-0000-00000000aaaa"
WORKSPACE_B = "00000000-0000-0000-0000-00000000bbbb"


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    """Clear bucketed rate-limit state per-test so the ingest tests in
    this file don't trip the 5/5min limit on each other."""
    from rate_limit import _limiter

    with _limiter._lock:
        _limiter._buckets.clear()
    yield
    with _limiter._lock:
        _limiter._buckets.clear()


# =====================================================================
# Connection isolation
# =====================================================================
class TestConnectionIsolation:
    def test_list_connections_only_returns_caller_workspace(
        self,
        client,
        jwt_auth_headers,
    ):
        """list_gmail_connections_public is called with the CALLER's
        workspace_id -- it cannot leak rows from another workspace."""
        with patch(
            "main.list_gmail_connections_public",
            return_value=[],
        ) as mock_list:
            r = client.get(
                "/api/gmail/connections",
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        kwargs = mock_list.call_args.kwargs
        # The route MUST pass workspace_id; the helper enforces the
        # WHERE clause on its side.
        assert kwargs["workspace_id"] == WORKSPACE_A

    def test_delete_foreign_connection_returns_404(
        self,
        client,
        jwt_auth_headers,
    ):
        """If the connection lives in workspace B, the workspace-A
        scoped DELETE returns 0 rows -- the helper returns False ->
        the route returns 404. No leak; no UI confusion."""
        with patch(
            "main.delete_gmail_connection",
            return_value=False,
        ) as mock_delete:
            r = client.delete(
                "/api/gmail/connections/foreign-conn",
                headers=jwt_auth_headers,
            )
        assert r.status_code == 404
        # Confirm we DID scope the delete by the caller's workspace.
        kwargs = mock_delete.call_args.kwargs
        assert kwargs["workspace_id"] == WORKSPACE_A

    def test_get_foreign_connection_for_ingest_returns_400(
        self,
        client,
        jwt_auth_headers,
    ):
        """A bogus or foreign connection_id passed to /api/gmail/ingest
        returns 400 -- get_gmail_connection's workspace_id filter
        means the lookup returns None."""
        with patch(
            "main.get_gmail_connection",
            return_value=None,
        ), patch(
            "main.run_workspace_gmail_ingest",
        ) as mock_runner:
            r = client.post(
                "/api/gmail/ingest",
                headers=jwt_auth_headers,
                json={"connection_id": "foreign-or-unknown"},
            )
        assert r.status_code == 400
        mock_runner.assert_not_called()

    def test_save_labels_for_foreign_connection_returns_404(
        self,
        client,
        jwt_auth_headers,
    ):
        """Save-labels guards against silently no-op'ing on a foreign
        connection. The 404 also prevents probing for the existence of
        another workspace's connection IDs."""
        with patch(
            "main.get_gmail_connection_public",
            return_value=None,
        ), patch(
            "main.set_selected_gmail_labels",
        ) as mock_save:
            r = client.post(
                "/api/gmail/labels",
                headers=jwt_auth_headers,
                json={
                    "connection_id": "foreign-conn",
                    "selected_label_ids": ["INBOX"],
                },
            )
        assert r.status_code == 404
        mock_save.assert_not_called()


# =====================================================================
# HydraDB sub-tenant routing (the headline Phase 4/8 guarantee)
# =====================================================================
class TestSubTenantRouting:
    def test_ingest_uses_workspace_sub_tenant(
        self,
        client,
        jwt_auth_headers,
    ):
        """The route MUST resolve THIS workspace's sub-tenant id and
        forward it to the ingest runner. A global default would mix
        workspaces in HydraDB."""
        connection = {
            "id": "conn-1",
            "workspace_id": WORKSPACE_A,
            "email": "owner@example.com",
            "access_token": "at",
            "refresh_token": "rt",
        }
        with patch(
            "main.get_gmail_connection",
            return_value=connection,
        ), patch(
            "main.list_selected_gmail_label_ids",
            return_value=["INBOX"],
        ), patch(
            "main.require_workspace_sub_tenant",
            return_value="ws_aaaaaaaaaaaa",
        ) as mock_ensure, patch(
            "main.run_workspace_gmail_ingest",
            return_value={},
        ) as mock_runner:
            r = client.post(
                "/api/gmail/ingest",
                headers=jwt_auth_headers,
                json={"connection_id": "conn-1"},
            )
        assert r.status_code == 202
        # The sub-tenant resolver is asked about THIS workspace, not the
        # connection's workspace_id field (defensive: even if the row
        # somehow had a wrong workspace_id, we'd route to the JWT
        # workspace).
        assert mock_ensure.call_args.kwargs["workspace_id"] == WORKSPACE_A
        # The runner receives that workspace's sub-tenant.
        assert mock_runner.call_args.kwargs["hydradb_sub_tenant_id"] == ("ws_aaaaaaaaaaaa")

    def test_missing_sub_tenant_502_blocks_ingest(
        self,
        client,
        jwt_auth_headers,
    ):
        """If we can't resolve the workspace's sub-tenant, we refuse
        the ingest with 502 -- we MUST NOT fall back to the env-default
        sub-tenant, which would leak emails into the shared bucket."""
        from errors import WorkspaceTenantError

        with patch(
            "main.get_gmail_connection",
            return_value={"id": "conn-1", "workspace_id": WORKSPACE_A, "refresh_token": "rt"},
        ), patch(
            "main.list_selected_gmail_label_ids",
            return_value=["INBOX"],
        ), patch(
            "main.require_workspace_sub_tenant",
            side_effect=WorkspaceTenantError(),
        ), patch(
            "main.run_workspace_gmail_ingest",
        ) as mock_runner:
            r = client.post(
                "/api/gmail/ingest",
                headers=jwt_auth_headers,
                json={"connection_id": "conn-1"},
            )
        assert r.status_code == 502
        mock_runner.assert_not_called()

    def test_runner_constructs_hydradb_with_sub_tenant(self):
        """Direct test against the ingest runner: HydraDBClient must
        be constructed with the workspace's sub_tenant_id (not the
        env default) whenever one is passed."""
        from gmail_oauth import run_workspace_gmail_ingest

        connection = {
            "id": "conn-1",
            "email": "u@example.com",
            "access_token": "at",
            "refresh_token": "rt",
        }
        with patch(
            "gmail_oauth.list_message_ids_for_label",
            return_value=[],
        ), patch(
            "hydradb_client.HydraDBClient",
        ) as mock_cls, patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            run_workspace_gmail_ingest(
                workspace_id=WORKSPACE_A,
                connection=connection,
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_aaaaaaaaaaaa",
            )
        # The HydraDB client is constructed exactly once with the
        # passed sub-tenant id.
        mock_cls.assert_called_with(sub_tenant_id="ws_aaaaaaaaaaaa")


# =====================================================================
# OAuth state isolation
# =====================================================================
class TestOAuthStateIsolation:
    def test_state_carries_signed_workspace_id(self):
        """State minted for workspace A must verify with workspace A's
        workspace_id and no other."""
        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state(WORKSPACE_A, "user-a")
        payload = verify_oauth_state(token)
        assert payload is not None
        assert payload["workspace_id"] == WORKSPACE_A
        assert payload["user_id"] == "user-a"

    def test_swapping_workspace_id_in_payload_invalidates_signature(self):
        """If an attacker decodes the state, swaps the workspace_id,
        and re-encodes, the HMAC signature fails -- the callback
        rejects the tampered state."""
        import base64
        import json

        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state(WORKSPACE_A, "user-a")
        payload_b64, sig_b64 = token.split(".", 1)
        padding = "=" * (-len(payload_b64) % 4)
        raw = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(raw.decode("utf-8"))
        # Swap A -> B but keep nonce + expiry + signature.
        payload["workspace_id"] = WORKSPACE_B
        new_raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        new_payload_b64 = base64.urlsafe_b64encode(new_raw).rstrip(b"=").decode()
        forged = f"{new_payload_b64}.{sig_b64}"
        # HMAC over the new payload won't match the old signature.
        assert verify_oauth_state(forged) is None

    def test_expired_state_rejected(self):
        """A stolen state that survives past its 5-minute window must
        not be replayable, regardless of which workspace it claims."""
        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state(WORKSPACE_A, "user-a")
        with patch("oauth_common.time.time", return_value=time.time() + 10_000):
            assert verify_oauth_state(token) is None

    def test_callback_uses_state_workspace_not_caller_workspace(
        self,
        client,
    ):
        """Critical safety: the OAuth callback has NO Authorization
        header (Google's redirect can't send one). It MUST infer the
        target workspace from the HMAC-signed state -- never from a
        header an attacker could spoof. We verify the upsert receives
        the state's workspace_id."""
        from gmail_oauth import make_oauth_state

        state = make_oauth_state(WORKSPACE_B, "user-b")  # state for B
        with patch(
            "main.gmail_exchange_code",
            return_value={"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
        ), patch(
            "main.gmail_fetch_user_info",
            return_value={"sub": "google-1", "email": "u@example.com"},
        ), patch(
            "main.upsert_gmail_connection",
            return_value={"id": "conn-x", "email": "u@example.com", "google_user_id": "google-1"},
        ) as mock_upsert:
            r = client.get(
                "/api/gmail/oauth/callback",
                params={"code": "abc", "state": state},
                # Even if the caller's JWT happened to be in another
                # browser session for workspace A, the OAuth callback
                # accepts no Authorization header at all.
                follow_redirects=False,
            )
        assert r.status_code == 302
        # The upsert MUST persist the connection under WORKSPACE_B
        # (from the signed state), regardless of any other context.
        kwargs = mock_upsert.call_args.kwargs
        assert kwargs["workspace_id"] == WORKSPACE_B


# =====================================================================
# Token secrecy
# =====================================================================
class TestTokenSecrecy:
    def test_connections_response_carries_no_token_fields(
        self,
        client,
        jwt_auth_headers,
    ):
        """The public-projection helper strips tokens. We additionally
        scan the rendered response body for the substrings, so a future
        change that accidentally widens the projection still fails the
        test."""
        with patch(
            "main.list_gmail_connections_public",
            return_value=[
                {
                    "id": "conn-1",
                    "workspace_id": WORKSPACE_A,
                    "google_user_id": "google-1",
                    "email": "u@example.com",
                    "scopes": "openid email",
                    "status": "active",
                    "connected_at": "2025-01-01T00:00:00Z",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                    "token_expiry": None,
                },
            ],
        ):
            r = client.get(
                "/api/gmail/connections",
                headers=jwt_auth_headers,
            )
        body_text = r.text
        # Even the FIELD NAMES "access_token" / "refresh_token" must
        # never appear in the response body.
        assert "access_token" not in body_text
        assert "refresh_token" not in body_text

    def test_get_public_helper_strips_token_fields(self):
        """Unit-level: the public projection helper omits token fields
        EVEN IF a buggy caller upstream forgets to."""
        from supabase_client import _gmail_public_projection

        row = {
            "id": "conn-1",
            "workspace_id": WORKSPACE_A,
            "google_user_id": "google-1",
            "email": "u@example.com",
            "access_token": "should-NEVER-leave",
            "refresh_token": "should-NEVER-leave",
            "token_expiry": "2025-01-01T00:00:00Z",
            "scopes": "openid email",
            "status": "active",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        }
        public = _gmail_public_projection(row)
        assert "access_token" not in public
        assert "refresh_token" not in public
        # Sanity: the public field IS there.
        assert public["email"] == "u@example.com"

    def test_update_tokens_requires_workspace_id(self):
        """Defense-in-depth: token updates must be workspace-scoped so
        a buggy caller can't accidentally overwrite another
        workspace's tokens."""
        from supabase_client import update_gmail_connection_tokens

        # Blank workspace_id -> refuse the write entirely, return False.
        assert (
            update_gmail_connection_tokens(
                connection_id="conn-1",
                workspace_id="",
                access_token="new-at",
            )
            is False
        )


# =====================================================================
# Spam/Trash protection
# =====================================================================
class TestSpamTrashProtection:
    def test_spam_and_trash_blocked_by_default(self):
        """Even if a user manages to mark SPAM/TRASH as selected, the
        ingest runner refuses to pull them unless GMAIL_ALLOW_SPAM_TRASH
        is explicitly set."""
        from gmail_oauth import run_workspace_gmail_ingest

        connection = {
            "id": "c",
            "email": "u@example.com",
            "refresh_token": "rt",
            "access_token": "at",
        }
        with patch(
            "gmail_oauth.list_message_ids_for_label",
        ) as mock_list, patch(
            "hydradb_client.HydraDBClient",
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            stats = run_workspace_gmail_ingest(
                workspace_id=WORKSPACE_A,
                connection=connection,
                label_ids=["SPAM", "TRASH"],
                hydradb_sub_tenant_id="ws_aaaaaaaaaaaa",
            )
        mock_list.assert_not_called()
        assert stats["labels_skipped"] == 2
