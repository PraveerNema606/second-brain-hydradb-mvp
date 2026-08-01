"""
Tests for the Phase 8 Gmail HTTP routes:
    GET    /api/gmail/connect-url
    GET    /api/gmail/oauth/callback
    GET    /api/gmail/connections
    DELETE /api/gmail/connections/{connection_id}
    GET    /api/gmail/labels
    POST   /api/gmail/labels
    POST   /api/gmail/ingest

We patch the supabase_client + gmail_oauth helpers AT THE main module's
namespace (main.upsert_gmail_connection, main.gmail_exchange_code, etc.)
because the routes call them by the names main imported.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

TEST_WS_ID = "00000000-0000-0000-0000-00000000aaaa"


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    """The new per-bucket rate limits accumulate across tests in the
    same process. Without resetting them, later tests hit the
    /api/gmail/ingest 5/5min limit and get 429 instead of 4xx."""
    from rate_limit import _limiter

    with _limiter._lock:
        _limiter._buckets.clear()
    yield
    with _limiter._lock:
        _limiter._buckets.clear()


# ──────────────────────────────────────────────────────────────────────
# GET /api/gmail/connect-url
# ──────────────────────────────────────────────────────────────────────
class TestConnectUrl:
    def test_returns_google_url(self, client, jwt_auth_headers):
        r = client.get("/api/gmail/connect-url", headers=jwt_auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert "accounts.google.com" in body["url"]
        assert "client_id=test-gmail-client-id" in body["url"]

    def test_returns_503_when_oauth_disabled(
        self,
        client,
        jwt_auth_headers,
        monkeypatch,
    ):
        monkeypatch.setenv("GMAIL_CLIENT_ID", "")
        r = client.get("/api/gmail/connect-url", headers=jwt_auth_headers)
        assert r.status_code == 503

    def test_requires_auth(self, client):
        r = client.get("/api/gmail/connect-url")
        assert r.status_code == 401


# ──────────────────────────────────────────────────────────────────────
# GET /api/gmail/oauth/callback
# ──────────────────────────────────────────────────────────────────────
class TestOauthCallback:
    def _good_state(self) -> str:
        from gmail_oauth import make_oauth_state

        return make_oauth_state(TEST_WS_ID, "test-user-id")

    def test_happy_path_redirects_ok(self, client):
        state = self._good_state()
        with patch(
            "main.gmail_exchange_code",
            return_value={
                "access_token": "at-1",
                "refresh_token": "rt-1",
                "expires_in": 3600,
                "scope": "openid email profile",
            },
        ), patch(
            "main.gmail_fetch_user_info",
            return_value={"sub": "google-id-1", "email": "u@example.com"},
        ), patch(
            "main.upsert_gmail_connection",
            return_value={"id": "conn-1", "email": "u@example.com", "google_user_id": "google-id-1"},
        ):
            r = client.get(
                "/api/gmail/oauth/callback",
                params={"code": "abc", "state": state},
                follow_redirects=False,
            )
        assert r.status_code == 302
        assert "gmail_connect=ok" in r.headers["location"]

    def test_user_denied_redirects_error(self, client):
        r = client.get(
            "/api/gmail/oauth/callback",
            params={"error": "access_denied"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "gmail_connect=error" in r.headers["location"]
        assert "access_denied" in r.headers["location"]

    def test_missing_code_redirects_error(self, client):
        r = client.get(
            "/api/gmail/oauth/callback",
            params={"state": self._good_state()},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "missing_params" in r.headers["location"]

    def test_invalid_state_redirects_error(self, client):
        r = client.get(
            "/api/gmail/oauth/callback",
            params={"code": "abc", "state": "bogus.state"},
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "bad_state" in r.headers["location"]

    def test_exchange_failure_redirects_error(self, client):
        with patch("main.gmail_exchange_code", return_value=None):
            r = client.get(
                "/api/gmail/oauth/callback",
                params={"code": "abc", "state": self._good_state()},
                follow_redirects=False,
            )
        assert r.status_code == 302
        assert "exchange_failed" in r.headers["location"]

    def test_userinfo_failure_redirects_error(self, client):
        with patch(
            "main.gmail_exchange_code",
            return_value={"access_token": "at", "expires_in": 3600},
        ), patch(
            "main.gmail_fetch_user_info",
            return_value=None,
        ):
            r = client.get(
                "/api/gmail/oauth/callback",
                params={"code": "abc", "state": self._good_state()},
                follow_redirects=False,
            )
        assert r.status_code == 302
        assert "userinfo_failed" in r.headers["location"]

    def test_persist_failure_redirects_error(self, client):
        with patch(
            "main.gmail_exchange_code",
            return_value={"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
        ), patch(
            "main.gmail_fetch_user_info",
            return_value={"sub": "g-1", "email": "u@x.com"},
        ), patch(
            "main.upsert_gmail_connection",
            return_value=None,
        ):
            r = client.get(
                "/api/gmail/oauth/callback",
                params={"code": "abc", "state": self._good_state()},
                follow_redirects=False,
            )
        assert r.status_code == 302
        assert "persist_failed" in r.headers["location"]

    def test_incomplete_install_redirects_error(self, client):
        # User info missing sub -> incomplete_install branch.
        with patch(
            "main.gmail_exchange_code",
            return_value={"access_token": "at", "expires_in": 3600},
        ), patch(
            "main.gmail_fetch_user_info",
            return_value={"email": "u@x.com"},  # no "sub"
        ):
            r = client.get(
                "/api/gmail/oauth/callback",
                params={"code": "abc", "state": self._good_state()},
                follow_redirects=False,
            )
        assert r.status_code == 302
        assert "incomplete_install" in r.headers["location"]


# ──────────────────────────────────────────────────────────────────────
# GET /api/gmail/connections
# ──────────────────────────────────────────────────────────────────────
class TestListConnections:
    def test_returns_public_projection_no_tokens(
        self,
        client,
        jwt_auth_headers,
    ):
        with patch(
            "main.list_gmail_connections_public",
            return_value=[
                {
                    "id": "conn-1",
                    "workspace_id": TEST_WS_ID,
                    "google_user_id": "google-id-1",
                    "email": "u@example.com",
                    "scopes": "openid email",
                    "status": "active",
                    "connected_at": "2025-01-01T00:00:00Z",
                    "created_at": "2025-01-01T00:00:00Z",
                    "updated_at": "2025-01-01T00:00:00Z",
                    "token_expiry": None,
                },
            ],
        ), patch(
            "main.get_gmail_connection_sync_summary",
            return_value={
                "last_synced_at": "2025-02-01T12:00:00Z",
                "labels_synced": 2,
            },
        ):
            r = client.get(
                "/api/gmail/connections",
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        body = r.json()
        # Defensive: tokens MUST NOT appear in the response body
        # anywhere, even as keys with null values.
        rendered = repr(body)
        assert "access_token" not in rendered
        assert "refresh_token" not in rendered
        assert body["connections"][0]["email"] == "u@example.com"
        # Phase 11: sync_summary enrichment is present and shape-stable.
        sync = body["connections"][0]["sync_summary"]
        assert sync["last_synced_at"] == "2025-02-01T12:00:00Z"
        assert sync["labels_synced"] == 2

    def test_empty_list_when_none(self, client, jwt_auth_headers):
        with patch(
            "main.list_gmail_connections_public",
            return_value=[],
        ):
            r = client.get(
                "/api/gmail/connections",
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        assert r.json()["connections"] == []

    def test_requires_auth(self, client):
        r = client.get("/api/gmail/connections")
        assert r.status_code == 401


# ──────────────────────────────────────────────────────────────────────
# DELETE /api/gmail/connections/{connection_id}
# ──────────────────────────────────────────────────────────────────────
class TestDeleteConnection:
    def test_deletes_existing(self, client, jwt_auth_headers):
        with patch("main.delete_gmail_connection", return_value=True):
            r = client.delete(
                "/api/gmail/connections/conn-1",
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_unknown_returns_404(self, client, jwt_auth_headers):
        with patch("main.delete_gmail_connection", return_value=False):
            r = client.delete(
                "/api/gmail/connections/unknown",
                headers=jwt_auth_headers,
            )
        assert r.status_code == 404

    def test_requires_auth(self, client):
        r = client.delete("/api/gmail/connections/x")
        assert r.status_code == 401


# ──────────────────────────────────────────────────────────────────────
# GET /api/gmail/labels
# ──────────────────────────────────────────────────────────────────────
class TestListLabels:
    def test_missing_connection_id_returns_400(self, client, jwt_auth_headers):
        r = client.get("/api/gmail/labels", headers=jwt_auth_headers)
        assert r.status_code == 400

    def test_unknown_connection_returns_empty(self, client, jwt_auth_headers):
        with patch("main.get_gmail_connection", return_value=None):
            r = client.get(
                "/api/gmail/labels",
                params={"connection_id": "missing"},
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        assert r.json() == {"connected": False, "labels": []}

    def test_refreshes_then_returns_stored(self, client, jwt_auth_headers):
        with patch(
            "main.get_gmail_connection",
            return_value={
                "id": "conn-1",
                "workspace_id": TEST_WS_ID,
                "email": "u@example.com",
                "access_token": "at",
                "refresh_token": "rt",
            },
        ), patch(
            "main.list_gmail_labels_from_api",
            return_value=[
                {"label_id": "INBOX", "name": "Inbox", "type": "system"},
                {"label_id": "Label_5", "name": "News", "type": "user"},
            ],
        ), patch(
            "main.upsert_gmail_labels",
            return_value=2,
        ), patch(
            "main.list_gmail_labels",
            return_value=[
                {"label_id": "INBOX", "name": "Inbox", "type": "system", "is_selected": False},
                {"label_id": "Label_5", "name": "News", "type": "user", "is_selected": True},
            ],
        ):
            r = client.get(
                "/api/gmail/labels",
                params={"connection_id": "conn-1"},
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is True
        assert len(body["labels"]) == 2

    def test_requires_auth(self, client):
        r = client.get("/api/gmail/labels")
        assert r.status_code == 401


# ──────────────────────────────────────────────────────────────────────
# POST /api/gmail/labels
# ──────────────────────────────────────────────────────────────────────
class TestSaveLabels:
    def test_saves_selected_set(self, client, jwt_auth_headers):
        with patch(
            # The save route now validates the connection belongs to
            # this workspace before writing -- mock both calls.
            "main.get_gmail_connection_public",
            return_value={"id": "conn-1", "email": "u@x.com"},
        ), patch(
            "main.set_selected_gmail_labels",
            return_value=True,
        ) as mock_save:
            r = client.post(
                "/api/gmail/labels",
                headers=jwt_auth_headers,
                json={
                    "connection_id": "conn-1",
                    "selected_label_ids": ["INBOX", "Label_5"],
                },
            )
        assert r.status_code == 200
        assert r.json()["selected_count"] == 2
        kwargs = mock_save.call_args.kwargs
        assert kwargs["workspace_id"] == TEST_WS_ID
        assert kwargs["gmail_connection_id"] == "conn-1"
        assert kwargs["selected_label_ids"] == ["INBOX", "Label_5"]

    def test_saves_empty_set(self, client, jwt_auth_headers):
        with patch(
            "main.get_gmail_connection_public",
            return_value={"id": "conn-1", "email": "u@x.com"},
        ), patch(
            "main.set_selected_gmail_labels",
            return_value=True,
        ):
            r = client.post(
                "/api/gmail/labels",
                headers=jwt_auth_headers,
                json={"connection_id": "conn-1", "selected_label_ids": []},
            )
        assert r.status_code == 200

    def test_unknown_connection_returns_404(self, client, jwt_auth_headers):
        # Phase 8 multi-tenant safety: cannot write labels for a
        # connection that doesn't belong to this workspace.
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
                    "connection_id": "foreign-or-unknown",
                    "selected_label_ids": ["INBOX"],
                },
            )
        assert r.status_code == 404
        mock_save.assert_not_called()

    def test_db_failure_returns_502(self, client, jwt_auth_headers):
        with patch(
            "main.get_gmail_connection_public",
            return_value={"id": "conn-1", "email": "u@x.com"},
        ), patch(
            "main.set_selected_gmail_labels",
            return_value=False,
        ):
            r = client.post(
                "/api/gmail/labels",
                headers=jwt_auth_headers,
                json={"connection_id": "conn-1", "selected_label_ids": []},
            )
        assert r.status_code == 502

    def test_rejects_extra_field(self, client, jwt_auth_headers):
        r = client.post(
            "/api/gmail/labels",
            headers=jwt_auth_headers,
            json={
                "connection_id": "conn-1",
                "selected_label_ids": [],
                "rogue_field": "x",
            },
        )
        assert r.status_code == 422

    def test_requires_auth(self, client):
        r = client.post(
            "/api/gmail/labels",
            json={"connection_id": "x", "selected_label_ids": []},
        )
        assert r.status_code == 401


# ──────────────────────────────────────────────────────────────────────
# POST /api/gmail/ingest
# ──────────────────────────────────────────────────────────────────────
class TestRunIngest:
    def test_kicks_off_background_ingest(self, client, jwt_auth_headers):
        connection = {
            "id": "conn-1",
            "workspace_id": TEST_WS_ID,
            "email": "u@example.com",
            "access_token": "at",
            "refresh_token": "rt",
        }
        with patch(
            "main.get_gmail_connection",
            return_value=connection,
        ), patch(
            "main.list_selected_gmail_label_ids",
            return_value=["INBOX", "Label_5"],
        ), patch(
            "main.require_workspace_sub_tenant",
            return_value="ws_test_abc",
        ), patch(
            "main.run_workspace_gmail_ingest",
            return_value={},
        ) as mock_runner:
            r = client.post(
                "/api/gmail/ingest",
                headers=jwt_auth_headers,
                json={"connection_id": "conn-1"},
            )
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "started"
        assert body["labels_queued"] == 2
        mock_runner.assert_called_once()
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["workspace_id"] == TEST_WS_ID
        assert kwargs["connection"]["id"] == "conn-1"
        assert kwargs["label_ids"] == ["INBOX", "Label_5"]
        assert kwargs["hydradb_sub_tenant_id"] == "ws_test_abc"

    def test_no_connection_returns_400(self, client, jwt_auth_headers):
        with patch("main.get_gmail_connection", return_value=None):
            r = client.post(
                "/api/gmail/ingest",
                headers=jwt_auth_headers,
                json={"connection_id": "missing"},
            )
        assert r.status_code == 400

    def test_no_labels_returns_400(self, client, jwt_auth_headers):
        with patch(
            "main.get_gmail_connection",
            return_value={"id": "conn-1"},
        ), patch(
            "main.list_selected_gmail_label_ids",
            return_value=[],
        ):
            r = client.post(
                "/api/gmail/ingest",
                headers=jwt_auth_headers,
                json={"connection_id": "conn-1"},
            )
        assert r.status_code == 400

    def test_sub_tenant_lookup_failure_returns_502(
        self,
        client,
        jwt_auth_headers,
    ):
        from errors import WorkspaceTenantError

        with patch(
            "main.get_gmail_connection",
            return_value={"id": "conn-1"},
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

    def test_rejects_extra_field(self, client, jwt_auth_headers):
        r = client.post(
            "/api/gmail/ingest",
            headers=jwt_auth_headers,
            json={"connection_id": "conn-1", "extra": True},
        )
        assert r.status_code == 422

    def test_requires_auth(self, client):
        r = client.post(
            "/api/gmail/ingest",
            json={"connection_id": "x"},
        )
        assert r.status_code == 401
