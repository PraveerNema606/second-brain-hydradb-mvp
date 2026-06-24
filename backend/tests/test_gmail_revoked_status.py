"""
Gmail revoked-access detection (audit item 1).

Covers:
  - _refresh_access_token_detailed: classifies refresh outcomes as
    "ok" / "revoked" / "transient".
  - _mark_connection_status: idempotent, workspace-scoped, never raises,
    only mirrors in-memory state on a successful write.
  - _authed_request: persists 'revoked' on an invalid_grant refresh and
    'error' on a transient refresh (then raises GmailApiError), and
    restores 'active' on a successful refresh (recovery).

All external calls are mocked — no network, no real Supabase.
"""

from unittest.mock import MagicMock, patch

import pytest


# =====================================================================
# Refresh outcome classification
# =====================================================================
class TestRefreshClassification:
    def test_invalid_grant_is_revoked(self):
        from gmail_oauth import _refresh_access_token_detailed

        resp = MagicMock(status_code=400, ok=False)
        resp.json.return_value = {"error": "invalid_grant"}
        with patch("gmail_oauth.requests.post", return_value=resp):
            data, outcome = _refresh_access_token_detailed("rt")
        assert data is None
        assert outcome == "revoked"

    def test_other_400_is_transient(self):
        from gmail_oauth import _refresh_access_token_detailed

        resp = MagicMock(status_code=400, ok=False)
        resp.json.return_value = {"error": "invalid_request"}
        with patch("gmail_oauth.requests.post", return_value=resp):
            data, outcome = _refresh_access_token_detailed("rt")
        assert data is None
        assert outcome == "transient"

    def test_401_is_revoked(self):
        from gmail_oauth import _refresh_access_token_detailed

        resp = MagicMock(status_code=401, ok=False)
        with patch("gmail_oauth.requests.post", return_value=resp):
            _data, outcome = _refresh_access_token_detailed("rt")
        assert outcome == "revoked"

    def test_403_is_revoked(self):
        from gmail_oauth import _refresh_access_token_detailed

        resp = MagicMock(status_code=403, ok=False)
        with patch("gmail_oauth.requests.post", return_value=resp):
            _data, outcome = _refresh_access_token_detailed("rt")
        assert outcome == "revoked"

    def test_500_is_transient(self):
        from gmail_oauth import _refresh_access_token_detailed

        resp = MagicMock(status_code=503, ok=False)
        with patch("gmail_oauth.requests.post", return_value=resp):
            _data, outcome = _refresh_access_token_detailed("rt")
        assert outcome == "transient"

    def test_429_is_transient(self):
        from gmail_oauth import _refresh_access_token_detailed

        resp = MagicMock(status_code=429, ok=False)
        with patch("gmail_oauth.requests.post", return_value=resp):
            _data, outcome = _refresh_access_token_detailed("rt")
        assert outcome == "transient"

    def test_network_error_is_transient(self):
        import requests

        from gmail_oauth import _refresh_access_token_detailed

        with patch(
            "gmail_oauth.requests.post",
            side_effect=requests.ConnectionError("dns"),
        ):
            data, outcome = _refresh_access_token_detailed("rt")
        assert data is None
        assert outcome == "transient"

    def test_empty_token_is_revoked_without_call(self):
        from gmail_oauth import _refresh_access_token_detailed

        with patch("gmail_oauth.requests.post") as mock_post:
            data, outcome = _refresh_access_token_detailed("")
        assert data is None
        assert outcome == "revoked"
        mock_post.assert_not_called()

    def test_success_is_ok(self):
        from gmail_oauth import _refresh_access_token_detailed

        resp = MagicMock(status_code=200, ok=True)
        resp.json.return_value = {"access_token": "new-at", "expires_in": 3600}
        with patch("gmail_oauth.requests.post", return_value=resp):
            data, outcome = _refresh_access_token_detailed("rt")
        assert outcome == "ok"
        assert data["access_token"] == "new-at"

    def test_public_wrapper_preserves_none_contract(self):
        # refresh_access_token must still return None on failure so
        # legacy callers / tests are unaffected.
        from gmail_oauth import refresh_access_token

        resp = MagicMock(status_code=400, ok=False)
        resp.json.return_value = {"error": "invalid_grant"}
        with patch("gmail_oauth.requests.post", return_value=resp):
            assert refresh_access_token("rt") is None


# =====================================================================
# _mark_connection_status
# =====================================================================
class TestMarkConnectionStatus:
    def test_writes_when_status_changes(self):
        from gmail_oauth import _mark_connection_status

        conn = {"id": "c1", "workspace_id": "w1", "status": "active"}
        with patch(
            "supabase_client.set_gmail_connection_status",
            return_value=True,
        ) as mock_set:
            _mark_connection_status(conn, "revoked")
        mock_set.assert_called_once_with(
            connection_id="c1",
            workspace_id="w1",
            status="revoked",
        )
        assert conn["status"] == "revoked"

    def test_noop_when_already_at_status(self):
        from gmail_oauth import _mark_connection_status

        conn = {"id": "c1", "workspace_id": "w1", "status": "active"}
        with patch("supabase_client.set_gmail_connection_status") as mock_set:
            _mark_connection_status(conn, "active")
        mock_set.assert_not_called()

    def test_missing_ids_updates_memory_only(self):
        from gmail_oauth import _mark_connection_status

        conn = {"status": "active"}  # no id / workspace_id
        with patch("supabase_client.set_gmail_connection_status") as mock_set:
            _mark_connection_status(conn, "revoked")
        mock_set.assert_not_called()
        assert conn["status"] == "revoked"

    def test_failed_write_leaves_memory_unchanged(self):
        from gmail_oauth import _mark_connection_status

        conn = {"id": "c1", "workspace_id": "w1", "status": "active"}
        with patch(
            "supabase_client.set_gmail_connection_status",
            return_value=False,
        ):
            _mark_connection_status(conn, "revoked")
        # Write failed -> keep the old in-memory value so a later call retries.
        assert conn["status"] == "active"

    def test_writer_exception_is_swallowed(self):
        from gmail_oauth import _mark_connection_status

        conn = {"id": "c1", "workspace_id": "w1", "status": "active"}
        with patch(
            "supabase_client.set_gmail_connection_status",
            side_effect=RuntimeError("boom"),
        ):
            # Must not raise.
            _mark_connection_status(conn, "revoked")


# =====================================================================
# _authed_request integration: detect + persist + recover
# =====================================================================
class TestAuthedRequestStatus:
    def _gmail_resp(self, status_code, ok=True, json_body=None):
        r = MagicMock(status_code=status_code, ok=ok)
        r.json.return_value = json_body if json_body is not None else {"ok": 1}
        return r

    def test_401_then_invalid_grant_marks_revoked_and_raises(self):
        from gmail_oauth import GmailApiError, _authed_request

        conn = {
            "id": "c1",
            "workspace_id": "w1",
            "status": "active",
            "access_token": "at",
            "refresh_token": "rt",
        }
        gmail_401 = self._gmail_resp(401, ok=False)
        refresh_bad = MagicMock(status_code=400, ok=False)
        refresh_bad.json.return_value = {"error": "invalid_grant"}

        with patch(
            "gmail_oauth.requests.request",
            return_value=gmail_401,
        ), patch(
            "gmail_oauth.requests.post",
            return_value=refresh_bad,
        ), patch(
            "supabase_client.set_gmail_connection_status",
            return_value=True,
        ) as mock_set:
            with pytest.raises(GmailApiError):
                _authed_request("GET", "https://gmail/x", conn)

        mock_set.assert_called_once_with(
            connection_id="c1",
            workspace_id="w1",
            status="revoked",
        )

    def test_401_then_transient_marks_error_and_raises(self):
        from gmail_oauth import GmailApiError, _authed_request

        conn = {
            "id": "c1",
            "workspace_id": "w1",
            "status": "active",
            "access_token": "at",
            "refresh_token": "rt",
        }
        gmail_401 = self._gmail_resp(401, ok=False)
        refresh_5xx = MagicMock(status_code=503, ok=False)

        with patch(
            "gmail_oauth.requests.request",
            return_value=gmail_401,
        ), patch(
            "gmail_oauth.requests.post",
            return_value=refresh_5xx,
        ), patch(
            "supabase_client.set_gmail_connection_status",
            return_value=True,
        ) as mock_set:
            with pytest.raises(GmailApiError):
                _authed_request("GET", "https://gmail/x", conn)

        mock_set.assert_called_once_with(
            connection_id="c1",
            workspace_id="w1",
            status="error",
        )

    def test_successful_refresh_restores_active(self):
        from gmail_oauth import _authed_request

        # Connection sitting in 'error', no access token in memory ->
        # forces a refresh. A successful refresh restores 'active'.
        conn = {
            "id": "c1",
            "workspace_id": "w1",
            "status": "error",
            "access_token": "",
            "refresh_token": "rt",
        }
        refresh_ok = MagicMock(status_code=200, ok=True)
        refresh_ok.json.return_value = {"access_token": "fresh", "expires_in": 3600}
        gmail_ok = self._gmail_resp(200, ok=True, json_body={"value": 42})

        with patch(
            "gmail_oauth.requests.post",
            return_value=refresh_ok,
        ), patch(
            "gmail_oauth.requests.request",
            return_value=gmail_ok,
        ), patch(
            "supabase_client.set_gmail_connection_status",
            return_value=True,
        ) as mock_set:
            data, out_conn = _authed_request("GET", "https://gmail/x", conn)

        assert data == {"value": 42}
        assert out_conn["_token_refreshed"] is True
        mock_set.assert_called_once_with(
            connection_id="c1",
            workspace_id="w1",
            status="active",
        )