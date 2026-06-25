"""
Tests for backend/gmail_oauth.py — OAuth state, code exchange, refresh,
label/message fetch, document builder, and the per-workspace ingest
runner.

These mirror the Slack OAuth test layout (test_slack_oauth.py) so the
two connectors test the same things in the same shape.
"""

import os
import time
from unittest.mock import MagicMock, patch

import pytest


# =====================================================================
# OAuth state
# =====================================================================
class TestOAuthState:
    def test_round_trip_valid(self):
        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state("ws-1", "user-1")
        payload = verify_oauth_state(token)
        assert payload is not None
        assert payload["workspace_id"] == "ws-1"
        assert payload["user_id"] == "user-1"

    def test_two_tokens_differ_due_to_nonce(self):
        from gmail_oauth import make_oauth_state

        a = make_oauth_state("ws-1", "u-1")
        b = make_oauth_state("ws-1", "u-1")
        assert a != b

    def test_empty_string_rejected(self):
        from gmail_oauth import verify_oauth_state

        assert verify_oauth_state("") is None

    def test_garbage_rejected(self):
        from gmail_oauth import verify_oauth_state

        assert verify_oauth_state("not.a.real.state") is None

    def test_missing_dot_rejected(self):
        from gmail_oauth import verify_oauth_state

        assert verify_oauth_state("no-separator-here") is None

    def test_tampered_signature_rejected(self):
        import base64

        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state("ws-1", "u-1")
        payload_b64, sig_b64 = token.split(".", 1)
        # Decode -> mutate a real byte -> re-encode. Flipping just the
        # LAST base64 char isn't reliable: base64url's tail has up to
        # 4 don't-care bits, so two different b64 chars can decode to
        # IDENTICAL bytes -- the original tweak flaked ~1 run in 4.
        padding = "=" * (-len(sig_b64) % 4)
        raw_sig = bytearray(base64.urlsafe_b64decode(sig_b64 + padding))
        raw_sig[0] ^= 0x01
        bad_sig = base64.urlsafe_b64encode(bytes(raw_sig)).rstrip(b"=").decode("ascii")
        tampered = f"{payload_b64}.{bad_sig}"
        assert verify_oauth_state(tampered) is None

    def test_tampered_payload_rejected(self):
        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state("ws-1", "u-1")
        payload_b64, sig_b64 = token.split(".", 1)
        # Replace the payload with something different but well-formed.
        bad_payload = "eyJ3b3Jrc3BhY2VfaWQiOiJ3cy0yIn0"
        assert verify_oauth_state(f"{bad_payload}.{sig_b64}") is None

    def test_wrong_secret_rejects_token(self, monkeypatch):
        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state("ws-1", "u-1")
        monkeypatch.setenv("GMAIL_OAUTH_STATE_SECRET", "completely-different-key")
        assert verify_oauth_state(token) is None

    def test_missing_secret_refuses_to_mint(self, monkeypatch):
        from gmail_oauth import make_oauth_state

        monkeypatch.setenv("GMAIL_OAUTH_STATE_SECRET", "")
        with pytest.raises(RuntimeError):
            make_oauth_state("ws-1", "u-1")

    def test_missing_secret_rejects_verify(self, monkeypatch):
        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state("ws-1", "u-1")
        monkeypatch.setenv("GMAIL_OAUTH_STATE_SECRET", "")
        assert verify_oauth_state(token) is None

    def test_expired_token_rejected(self, monkeypatch):
        from gmail_oauth import make_oauth_state, verify_oauth_state

        token = make_oauth_state("ws-1", "u-1")
        # time.time() is called inside oauth_common now, so patch it there.
        with patch("oauth_common.time.time", return_value=time.time() + 10_000):
            assert verify_oauth_state(token) is None


# =====================================================================
# Build connect URL
# =====================================================================
class TestBuildConnectUrl:
    def test_url_contains_required_params(self):
        from gmail_oauth import build_connect_url

        url = build_connect_url(workspace_id="ws-1", user_id="user-1")
        assert "accounts.google.com" in url
        assert "/o/oauth2/v2/auth" in url
        # The required Google OAuth params must all appear.
        for needle in (
            "client_id=test-gmail-client-id",
            "redirect_uri=",
            "scope=",
            "access_type=offline",
            "prompt=consent",
            "state=",
            "response_type=code",
        ):
            assert needle in url

    def test_state_in_url_is_verifiable(self):
        from urllib.parse import parse_qs, urlparse

        from gmail_oauth import build_connect_url, verify_oauth_state

        url = build_connect_url(workspace_id="ws-7", user_id="u-7")
        qs = parse_qs(urlparse(url).query)
        state = qs["state"][0]
        payload = verify_oauth_state(state)
        assert payload is not None
        assert payload["workspace_id"] == "ws-7"
        assert payload["user_id"] == "u-7"


# =====================================================================
# Token exchange
# =====================================================================
class TestExchangeCode:
    def _build_mock_response(self, *, ok=True, status_code=200, body=None):
        resp = MagicMock()
        resp.ok = ok
        resp.status_code = status_code
        resp.json.return_value = body if body is not None else {}
        return resp

    def test_happy_path(self):
        from gmail_oauth import exchange_code

        body = {
            "access_token": "at-1",
            "refresh_token": "rt-1",
            "expires_in": 3600,
            "scope": "openid email profile",
            "token_type": "Bearer",
        }
        with patch(
            "gmail_oauth.requests.post",
            return_value=self._build_mock_response(body=body),
        ):
            data = exchange_code("auth-code-123")
        assert data is not None
        assert data["access_token"] == "at-1"
        assert data["refresh_token"] == "rt-1"

    def test_http_error_returns_none(self):
        from gmail_oauth import exchange_code

        with patch(
            "gmail_oauth.requests.post",
            return_value=self._build_mock_response(ok=False, status_code=400),
        ):
            assert exchange_code("bad") is None

    def test_non_json_returns_none(self):
        from gmail_oauth import exchange_code

        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        with patch("gmail_oauth.requests.post", return_value=resp):
            assert exchange_code("x") is None

    def test_missing_access_token_returns_none(self):
        from gmail_oauth import exchange_code

        with patch(
            "gmail_oauth.requests.post",
            return_value=self._build_mock_response(body={"foo": "bar"}),
        ):
            assert exchange_code("x") is None

    def test_request_exception_returns_none(self):
        import requests

        from gmail_oauth import exchange_code

        with patch(
            "gmail_oauth.requests.post",
            side_effect=requests.ConnectionError("dns"),
        ):
            assert exchange_code("x") is None


# =====================================================================
# Refresh access token
# =====================================================================
class TestRefreshAccessToken:
    def test_success(self):
        from gmail_oauth import refresh_access_token

        resp = MagicMock(ok=True, status_code=200)
        resp.json.return_value = {"access_token": "new-at", "expires_in": 3600}
        with patch("gmail_oauth.requests.post", return_value=resp):
            data = refresh_access_token("rt-1")
        assert data["access_token"] == "new-at"

    def test_blank_returns_none(self):
        from gmail_oauth import refresh_access_token

        with patch("gmail_oauth.requests.post") as mock_post:
            assert refresh_access_token("") is None
        mock_post.assert_not_called()

    def test_http_error_returns_none(self):
        from gmail_oauth import refresh_access_token

        resp = MagicMock(ok=False, status_code=400)
        with patch("gmail_oauth.requests.post", return_value=resp):
            assert refresh_access_token("rt") is None


# =====================================================================
# Fetch user info
# =====================================================================
class TestFetchUserInfo:
    def test_success(self):
        from gmail_oauth import fetch_user_info

        resp = MagicMock(ok=True, status_code=200)
        resp.json.return_value = {"sub": "google-123", "email": "u@example.com"}
        with patch("gmail_oauth.requests.get", return_value=resp):
            info = fetch_user_info("at-1")
        assert info["sub"] == "google-123"
        assert info["email"] == "u@example.com"

    def test_http_error_returns_none(self):
        from gmail_oauth import fetch_user_info

        resp = MagicMock(ok=False, status_code=401)
        with patch("gmail_oauth.requests.get", return_value=resp):
            assert fetch_user_info("bad") is None

    def test_blank_returns_none(self):
        from gmail_oauth import fetch_user_info

        with patch("gmail_oauth.requests.get") as mock_get:
            assert fetch_user_info("") is None
        mock_get.assert_not_called()


# =====================================================================
# Installation projection
# =====================================================================
class TestInstallationProjection:
    def test_happy_path_projects_fields(self):
        from gmail_oauth import installation_from_token_response

        token = {
            "access_token": "at-1",
            "refresh_token": "rt-1",
            "expires_in": 3600,
            "scope": "openid email profile https://www.googleapis.com/auth/gmail.readonly",
        }
        info = {"sub": "google-123", "email": "u@example.com"}
        out = installation_from_token_response(token, info)
        assert out["google_user_id"] == "google-123"
        assert out["email"] == "u@example.com"
        assert out["access_token"] == "at-1"
        assert out["refresh_token"] == "rt-1"
        assert "gmail.readonly" in out["scopes"]
        assert out["token_expiry"] is not None

    def test_missing_fields_collapse_to_empty(self):
        from gmail_oauth import installation_from_token_response

        out = installation_from_token_response({}, {})
        assert out["google_user_id"] == ""
        assert out["email"] == ""
        assert out["access_token"] == ""
        assert out["refresh_token"] == ""
        assert out["scopes"] == ""
        assert out["token_expiry"] is None


# =====================================================================
# Email -> markdown document
# =====================================================================
class TestBuildEmailDocument:
    def _sample_message(self, **overrides):
        # text/plain payload, base64url-encoded.
        import base64

        plain_body = (
            base64.urlsafe_b64encode(b"Hello team,\nQuarterly review on Friday.\n--Alice").rstrip(b"=").decode("ascii")
        )
        msg = {
            "id": "msg-abc-123",
            "snippet": "Hello team, Quarterly review on Friday...",
            "labelIds": ["INBOX", "Label_5"],
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": plain_body, "size": 50},
                "headers": [
                    {"name": "Subject", "value": "Quarterly review"},
                    {"name": "From", "value": "Alice <alice@example.com>"},
                    {"name": "To", "value": "team@example.com"},
                    {"name": "Date", "value": "Mon, 5 May 2025 10:00:00 +0000"},
                ],
            },
        }
        msg.update(overrides)
        return msg

    def test_extracts_text_plain_body(self):
        from gmail_oauth import build_email_document

        doc = build_email_document(self._sample_message(), "owner@example.com")
        assert doc is not None
        assert "Quarterly review on Friday." in doc["content"]
        assert "Subject: Quarterly review" in doc["content"]
        assert "From: Alice <alice@example.com>" in doc["content"]
        assert "Labels: INBOX, Label_5" in doc["content"]
        # Phase 8 spec: every doc must carry the gmail message id and
        # the snippet line as part of the header block.
        assert "Message-Id: msg-abc-123" in doc["content"]
        assert "Snippet:" in doc["content"]

    def test_cc_header_included_when_present(self):
        import base64

        from gmail_oauth import build_email_document

        # Construct a message that includes a Cc header.
        body = base64.urlsafe_b64encode(b"body text").rstrip(b"=").decode("ascii")
        msg = {
            "id": "msg-with-cc",
            "snippet": "x",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": body},
                "headers": [
                    {"name": "Subject", "value": "team update"},
                    {"name": "From", "value": "a@example.com"},
                    {"name": "To", "value": "b@example.com"},
                    {"name": "Cc", "value": "manager@example.com, eng@example.com"},
                    {"name": "Date", "value": "Mon, 5 May 2025 10:00:00 +0000"},
                ],
            },
        }
        doc = build_email_document(msg, "owner@example.com")
        assert doc is not None
        # Cc explicitly required by the Phase 8 spec.
        assert "Cc: manager@example.com, eng@example.com" in doc["content"]

    def test_cc_header_omitted_when_missing(self):
        # When Cc is absent we don't emit an empty "Cc: " line.
        from gmail_oauth import build_email_document

        doc = build_email_document(self._sample_message(), "owner@example.com")
        # _sample_message has no Cc; resulting doc should not contain
        # a "Cc:" line at all.
        assert "Cc:" not in doc["content"]

    def test_falls_back_to_html_when_no_text_plain(self):
        import base64

        from gmail_oauth import build_email_document

        html_body = (
            base64.urlsafe_b64encode(b"<p>Hello <b>team</b></p><p>See you <i>Friday</i>.</p>")
            .rstrip(b"=")
            .decode("ascii")
        )
        msg = {
            "id": "msg-html",
            "snippet": "Hello team",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "text/html",
                "body": {"data": html_body},
                "headers": [{"name": "Subject", "value": "hi"}],
            },
        }
        doc = build_email_document(msg, "owner@example.com")
        assert doc is not None
        # HTML tags are stripped.
        assert "<p>" not in doc["content"]
        assert "Hello team" in doc["content"]

    def test_walks_multipart_for_text_plain(self):
        import base64

        from gmail_oauth import build_email_document

        text_body = base64.urlsafe_b64encode(b"Plain part wins.").rstrip(b"=").decode("ascii")
        msg = {
            "id": "msg-multi",
            "snippet": "",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "multipart/alternative",
                "body": {"size": 0},
                "headers": [{"name": "Subject", "value": "mixed"}],
                "parts": [
                    {
                        "mimeType": "text/html",
                        "body": {"data": "PHA+SFRNTDwvcD4="},
                    },
                    {
                        "mimeType": "text/plain",
                        "body": {"data": text_body},
                    },
                ],
            },
        }
        doc = build_email_document(msg, "owner@example.com")
        assert doc is not None
        assert "Plain part wins." in doc["content"]

    def test_stable_key_uses_message_id(self):
        from gmail_oauth import build_email_document

        doc = build_email_document(self._sample_message(), "owner@example.com")
        assert doc["stable_key"] == "gmail:msg:msg-abc-123"

    def test_permalink_present(self):
        from gmail_oauth import build_email_document

        doc = build_email_document(self._sample_message(), "owner@example.com")
        assert doc["permalink"] == "https://mail.google.com/mail/u/0/#all/msg-abc-123"

    def test_body_capped_at_32k(self):
        import base64

        from gmail_oauth import build_email_document

        huge = ("x" * 40_000).encode()
        body = base64.urlsafe_b64encode(huge).rstrip(b"=").decode("ascii")
        msg = {
            "id": "msg-huge",
            "snippet": "",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": body},
                "headers": [{"name": "Subject", "value": "big"}],
            },
        }
        doc = build_email_document(msg, "owner@example.com")
        assert doc is not None
        # Header + a 32k-capped body. Should be well under 33k total.
        assert len(doc["content"]) < 33_000

    def test_empty_body_and_no_snippet_returns_none(self):
        from gmail_oauth import build_email_document

        msg = {
            "id": "msg-empty",
            "snippet": "",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": ""},
                "headers": [{"name": "Subject", "value": "blank"}],
            },
        }
        assert build_email_document(msg, "owner@example.com") is None

    def test_no_subject_or_metadata_stored_in_doc_dict(self):
        """The returned doc dict itself only carries IDs, snippet, and
        permalink as metadata -- the subject and body live ONLY inside
        `content`. A leak of the state.json shouldn't expose subjects."""
        from gmail_oauth import build_email_document

        doc = build_email_document(self._sample_message(), "owner@example.com")
        assert "subject" not in doc
        assert "body" not in doc
        # message_id is fine -- it's a stable opaque identifier.
        assert doc["message_id"] == "msg-abc-123"


# =====================================================================
# run_workspace_gmail_ingest
# =====================================================================
class TestRunWorkspaceGmailIngest:
    def _connection(self, **overrides):
        c = {
            "id": "conn-1",
            "workspace_id": "ws-1",
            "email": "owner@example.com",
            "access_token": "at-1",
            "refresh_token": "rt-1",
        }
        c.update(overrides)
        return c

    def test_constructs_hydradb_client_with_sub_tenant(self):
        from gmail_oauth import run_workspace_gmail_ingest

        with patch(
            "gmail_oauth.list_thread_ids_for_label",
            return_value=[],
        ), patch(
            "hydradb_client.HydraDBClient",
        ) as mock_hydra, patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_test_abc",
            )
        mock_hydra.assert_called_once_with(sub_tenant_id="ws_test_abc")

    def test_missing_sub_tenant_falls_back_to_default(self):
        from gmail_oauth import run_workspace_gmail_ingest

        with patch(
            "gmail_oauth.list_thread_ids_for_label",
            return_value=[],
        ), patch(
            "hydradb_client.HydraDBClient",
        ) as mock_hydra, patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["INBOX"],
            )
        mock_hydra.assert_called_once_with()

    def test_no_labels_returns_zero_counts(self):
        from gmail_oauth import run_workspace_gmail_ingest

        with patch("hydradb_client.HydraDBClient") as mock_hydra:
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=[],
                hydradb_sub_tenant_id="ws_x",
            )
        assert stats["messages_uploaded"] == 0
        mock_hydra.assert_not_called()

    def test_missing_refresh_token_dead_letters(self):
        from gmail_oauth import run_workspace_gmail_ingest

        with patch(
            "gmail_oauth.emit_dead_letter",
        ) as mock_dl, patch(
            "hydradb_client.HydraDBClient",
        ) as mock_hydra:
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(refresh_token=""),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
            )
        mock_dl.assert_called_once()
        # We bail BEFORE constructing the HydraDB client.
        mock_hydra.assert_not_called()
        assert stats["messages_uploaded"] == 0

    def test_spam_and_trash_blocked_by_default(self):
        from gmail_oauth import run_workspace_gmail_ingest

        with patch(
            "gmail_oauth.list_thread_ids_for_label",
        ) as mock_list, patch(
            "hydradb_client.HydraDBClient",
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["SPAM", "TRASH"],
                hydradb_sub_tenant_id="ws_x",
            )
        # Both skipped -> no Gmail API calls.
        mock_list.assert_not_called()
        assert stats["labels_skipped"] == 2

    def test_spam_trash_allowed_via_env(self, monkeypatch):
        from gmail_oauth import run_workspace_gmail_ingest

        monkeypatch.setenv("GMAIL_ALLOW_SPAM_TRASH", "true")
        with patch(
            "gmail_oauth.list_thread_ids_for_label",
            return_value=[],
        ) as mock_list, patch(
            "hydradb_client.HydraDBClient",
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["SPAM"],
                hydradb_sub_tenant_id="ws_x",
            )
        # Env flag flipped -> we DO call the Gmail API for SPAM.
        mock_list.assert_called_once()
        assert stats["labels_processed"] == 1

    def test_max_messages_cap_respected(self):
        # Cap at 2 even if Gmail returns 5 thread IDs. The per-run budget
        # is now a THREAD budget, so at most 2 threads are fetched.
        from gmail_oauth import run_workspace_gmail_ingest

        def _thread(tid):
            return {
                "messages": [
                    {
                        "id": f"{tid}-m1",
                        "threadId": tid,
                        "snippet": "hi",
                        "internalDate": "1700000000000",
                        "labelIds": ["INBOX"],
                        "payload": {
                            "mimeType": "text/plain",
                            "body": {"data": ""},
                            "headers": [{"name": "Subject", "value": "x"}],
                        },
                    }
                ]
            }

        mock_hydra_instance = MagicMock()
        mock_hydra_instance.upload_knowledge.return_value = {
            "success": True,
            "success_count": 1,
            "failed_count": 0,
        }
        with patch(
            "gmail_oauth.list_thread_ids_for_label",
            return_value=["t1", "t2", "t3", "t4", "t5"],
        ), patch(
            "gmail_oauth.fetch_thread",
            side_effect=lambda conn, tid: _thread(tid),
        ) as mock_fetch, patch(
            "hydradb_client.HydraDBClient",
            return_value=mock_hydra_instance,
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
                max_messages=2,
            )
        # fetch_thread is called at most max_messages (= thread budget) times.
        assert mock_fetch.call_count <= 2
        assert stats["threads_processed"] <= 2
        assert stats["threads_uploaded"] <= 2

    def test_returns_stats_shape(self):
        from gmail_oauth import run_workspace_gmail_ingest

        with patch(
            "gmail_oauth.list_thread_ids_for_label",
            return_value=[],
        ), patch(
            "hydradb_client.HydraDBClient",
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
            )
        for key in (
            "labels_processed",
            "labels_skipped",
            "labels_failed",
            "messages_fetched",
            "messages_uploaded",
            "messages_failed",
            "messages_skipped",
        ):
            assert key in stats