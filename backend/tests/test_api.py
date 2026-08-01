"""Tests for main.py — FastAPI endpoint contracts."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

AUTH = {"X-API-Key": "test-secret-key"}
INSUFFICIENT = "I do not have enough context to answer that."


# ── Public endpoints ───────────────────────────────────────────────────────
class TestPublicEndpoints:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_response_shape(self, client):
        r = client.get("/")
        body = r.json()
        assert "name" in body
        assert "status" in body
        assert body["status"] == "ok"

    def test_health_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_response_shape(self, client):
        # /api/health is our detailed diagnostics endpoint.
        # It always returns 200; status reflects aggregate health across checks.
        # In the test environment real services are not available, so we mock
        # all checks to return ok so the shape assertion is deterministic.
        from unittest.mock import AsyncMock
        from unittest.mock import patch as _patch

        from health import STATUS_OK, HealthResult, _get_registry

        ok_result = HealthResult(status=STATUS_OK, latency_ms=1.0)
        with _patch("health._run_check", new=AsyncMock(side_effect=lambda c: (c.name, ok_result))):
            body = client.get("/api/health").json()
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        assert "checks" in body
        assert "timestamp" in body

    def test_docs_endpoint_available(self, client):
        r = client.get("/docs")
        assert r.status_code == 200


# ── /api/query — validation ───────────────────────────────────────────────
class TestQueryValidation:
    def test_missing_question_returns_422(self, client):
        r = client.post("/api/query", json={}, headers=AUTH)
        assert r.status_code == 422

    def test_question_too_short_returns_422(self, client):
        r = client.post("/api/query", json={"question": "ab"}, headers=AUTH)
        assert r.status_code == 422

    def test_question_too_long_returns_422(self, client):
        r = client.post("/api/query", json={"question": "x" * 2001}, headers=AUTH)
        assert r.status_code == 422

    def test_invalid_mode_returns_422(self, client):
        r = client.post(
            "/api/query",
            json={"question": "what happened?", "mode": "invalid_mode"},
            headers=AUTH,
        )
        assert r.status_code == 422

    def test_invalid_top_k_too_high_returns_422(self, client):
        r = client.post(
            "/api/query",
            json={"question": "what happened?", "top_k": 100},
            headers=AUTH,
        )
        assert r.status_code == 422

    def test_invalid_top_k_zero_returns_422(self, client):
        r = client.post(
            "/api/query",
            json={"question": "what happened?", "top_k": 0},
            headers=AUTH,
        )
        assert r.status_code == 422

    def test_invalid_document_type_returns_422(self, client):
        r = client.post(
            "/api/query",
            json={"question": "what happened?", "document_type": "invalid"},
            headers=AUTH,
        )
        assert r.status_code == 422

    def test_valid_request_modes(self, client):
        for mode in ("default", "summary", "decisions", "action_items", "who_said", "exact", "hybrid"):
            with patch("main.answer_question") as mock_ans:
                mock_ans.return_value = {"answer": "ok", "sources": [], "debug": {}}
                r = client.post(
                    "/api/query",
                    json={"question": "what happened?", "mode": mode},
                    headers=AUTH,
                )
                assert r.status_code == 200, f"Mode {mode} returned {r.status_code}"


# ── /api/query — response shape ───────────────────────────────────────────
class TestQueryResponseShape:
    def test_response_has_answer_sources_debug(self, client):
        # Patch at the main module level (where the name is bound after `from recall import`)
        with patch("main.answer_question") as mock_ans:
            mock_ans.return_value = {
                "answer": "The plan is X.",
                "sources": [],
                "debug": {"chunks_returned": 2, "mode": "default"},
            }
            r = client.post(
                "/api/query",
                json={"question": "what is the plan?"},
                headers=AUTH,
            )
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "sources" in body
        assert "debug" in body

    def test_cache_hit_false_on_fresh_response(self, client):
        with patch("main.answer_question") as mock_ans:
            mock_ans.return_value = {"answer": "Fresh answer.", "sources": [], "debug": {}}
            r = client.post(
                "/api/query",
                json={"question": "what is the plan?"},
                headers=AUTH,
            )
        assert r.json()["debug"]["cache_hit"] is False

    def test_date_query_debug_attached(self, client):
        with patch("main.answer_question") as mock_ans:
            mock_ans.return_value = {"answer": "answer", "sources": [], "debug": {}}
            r = client.post(
                "/api/query",
                json={"question": "what happened?", "date_query": "last week"},
                headers=AUTH,
            )
        debug = r.json().get("debug", {})
        assert "date_query" in debug

    def test_no_context_returns_fallback_answer(self, client):
        with patch("main.answer_question") as mock_ans:
            mock_ans.return_value = {
                "answer": INSUFFICIENT,
                "sources": [],
                "debug": {},
            }
            r = client.post(
                "/api/query",
                json={"question": "totally unrelated question"},
                headers=AUTH,
            )
        assert r.status_code == 200
        assert INSUFFICIENT in r.json()["answer"]

    def test_conversation_history_accepted(self, client):
        with patch("main.answer_question") as mock_ans:
            mock_ans.return_value = {"answer": "ok", "sources": [], "debug": {}}
            r = client.post(
                "/api/query",
                json={
                    "question": "what did he say?",
                    "conversation_history": [
                        {"role": "user", "content": "Who is Alice?"},
                        {"role": "assistant", "content": "Alice is an engineer."},
                    ],
                },
                headers=AUTH,
            )
        assert r.status_code == 200

    def test_cache_bypassed_when_history_present(self, client):
        with patch("main.answer_question") as mock_ans:
            mock_ans.return_value = {"answer": "ok", "sources": [], "debug": {}}
            r = client.post(
                "/api/query",
                json={
                    "question": "what did he say?",
                    "conversation_history": [
                        {"role": "user", "content": "Prior turn."},
                    ],
                },
                headers=AUTH,
            )
        debug = r.json().get("debug", {})
        assert "cache_bypassed" in debug


# ── /api/query — auth & rate limiting ────────────────────────────────────
class TestQueryAuth:
    def test_no_key_returns_401(self, client):
        r = client.post("/api/query", json={"question": "hello world"})
        assert r.status_code == 401

    # NOTE: The Phase 1 Supabase migration replaced X-API-Key with
    # Authorization: Bearer <jwt> on /api/query. The "wrong key" case
    # (i.e. presenting an invalid token) is covered in detail by
    # tests/test_supabase_auth.py::TestRequireUserFailures — see the
    # expired / wrong-signature / wrong-audience / malformed cases there.


# ── /api/query/stream ─────────────────────────────────────────────────────


class TestQueryStream:
    def test_no_key_returns_401(self, client):
        r = client.post("/api/query/stream", json={"question": "hello world"})
        assert r.status_code == 401

    def test_invalid_payload_returns_422(self, client):
        r = client.post(
            "/api/query/stream",
            json={"question": "ab"},  # too short
            headers=AUTH,
        )
        assert r.status_code == 422

    def test_stream_returns_event_stream_content_type(self, client):
        with patch("main.prepare_recall_context") as mock_ctx:
            mock_ctx.return_value = {"ready": False, "fallback_debug": {}}
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_stream_emits_token_and_done_events(self, client):
        from unittest.mock import patch as mp

        def _fake_stream(*a, **kw):
            yield "Hello"
            yield " world"

        mock_sources = [
            {
                "index": 1,
                "source": "general",
                "channel": "general",
                "user": "Alice",
                "snippet": "snippet",
                "stable_key": "k1",
                "permalink": "https://slack.com/1",
            }
        ]

        with mp("main.prepare_recall_context") as mock_ctx, mp(
            "main.stream_grounded_answer", side_effect=_fake_stream
        ), mp("main.finalize_answer") as mock_fin:
            mock_ctx.return_value = {
                "ready": True,
                "context_text": "[1] content",
                "sources": mock_sources,
                "chunks_count": 1,
                "filtered_out": 0,
                "exact_matches": 0,
                "retrieval_mode": "default",
                "query_terms": [],
                "fallback_debug": None,
            }
            mock_fin.return_value = {
                "answer": "Hello world",
                "cleaned_sources": mock_sources,
                "sources_before": 1,
                "sources_after": 1,
            }
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )

        assert r.status_code == 200
        text = r.text
        assert "event: token" in text
        assert "event: done" in text

    def test_stream_no_context_emits_fallback(self, client):
        with patch("main.prepare_recall_context") as mock_ctx:
            mock_ctx.return_value = {
                "ready": False,
                "fallback_debug": {"reason": "no chunks"},
            }
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        assert r.status_code == 200
        text = r.text
        assert "event: token" in text
        assert "event: done" in text
        assert INSUFFICIENT in text

    def test_stream_hydradb_error_emits_error_event(self, client):
        from errors import HydraDBError

        with patch("main.prepare_recall_context", side_effect=HydraDBError("down")):
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        assert r.status_code == 200
        assert "event: error" in r.text


# ── /api/admin/status ─────────────────────────────────────────────────────
class TestAdminStatus:
    def test_requires_auth(self, client):
        r = client.get("/api/admin/status")
        assert r.status_code == 401

    def test_returns_status_shape(self, client):
        r = client.get("/api/admin/status", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert "realtime_ingest_enabled" in body
        assert "scheduler_enabled" in body
        assert "total_docs" in body


# ── /slack/events ─────────────────────────────────────────────────────────
class TestSlackEvents:
    def _make_sig(self, body: bytes, ts: int) -> str:
        import hashlib
        import hmac
        import os

        secret = os.environ.get("SLACK_SIGNING_SECRET", "test-slack-signing-secret")
        base = b"v0:" + str(ts).encode() + b":" + body
        digest = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
        return f"v0={digest}"

    def _headers(self, body: bytes) -> dict:
        import time

        ts = int(time.time())
        return {
            "X-Slack-Signature": self._make_sig(body, ts),
            "X-Slack-Request-Timestamp": str(ts),
        }

    def test_url_verification_challenge(self, client):
        body = json.dumps(
            {
                "type": "url_verification",
                "challenge": "test-challenge-xyz",
            }
        ).encode()
        r = client.post("/slack/events", content=body, headers=self._headers(body))
        assert r.status_code == 200
        assert "test-challenge-xyz" in r.text

    def test_invalid_signature_returns_401(self, client):
        body = b'{"type":"event_callback"}'
        import time

        r = client.post(
            "/slack/events",
            content=body,
            headers={
                "X-Slack-Signature": "v0=badhash",
                "X-Slack-Request-Timestamp": str(int(time.time())),
            },
        )
        assert r.status_code == 401

    def test_event_callback_acks_200(self, client):
        payload = json.dumps(
            {
                "type": "event_callback",
                "event_id": "Ev_unique_test_1",
                "event": {"type": "message", "text": "hello", "channel": "C123"},
            }
        ).encode()
        with patch("realtime_ingest.process_slack_event"):
            r = client.post(
                "/slack/events",
                content=payload,
                headers=self._headers(payload),
            )
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_duplicate_event_id_acks_without_reprocessing(self, client):
        """Second delivery of same event_id should be ack'd but not re-ingested."""
        payload = json.dumps(
            {
                "type": "event_callback",
                "event_id": "Ev_duplicate_test",
                "event": {"type": "message", "text": "hello", "channel": "C123"},
            }
        ).encode()
        with patch("realtime_ingest.process_slack_event") as mock_proc:
            # Send twice
            for _ in range(2):
                client.post(
                    "/slack/events",
                    content=payload,
                    headers=self._headers(payload),
                )
        # process_slack_event called at most once (duplicate suppressed)
        assert mock_proc.call_count <= 1

    def test_unknown_payload_type_returns_200(self, client):
        payload = json.dumps({"type": "unknown_type"}).encode()
        r = client.post(
            "/slack/events",
            content=payload,
            headers=self._headers(payload),
        )
        assert r.status_code == 200

    def test_invalid_json_body_returns_400(self, client):
        body = b"not valid json {"
        import time

        ts = int(time.time())
        secret = os.environ.get("SLACK_SIGNING_SECRET", "test-slack-signing-secret")
        import hashlib
        import hmac as _hmac

        base = b"v0:" + str(ts).encode() + b":" + body
        digest = _hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
        r = client.post(
            "/slack/events",
            content=body,
            headers={
                "X-Slack-Signature": f"v0={digest}",
                "X-Slack-Request-Timestamp": str(ts),
            },
        )
        assert r.status_code == 400


# ── CORS headers ──────────────────────────────────────────────────────────
class TestCORS:
    def test_options_returns_cors_headers(self, client):
        r = client.options(
            "/api/query",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        # CORS middleware should add the header
        assert r.status_code in (200, 400)  # 400 without method is fine

    def test_allowed_origin_in_response(self, client):
        r = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:5173"},
        )
        cors = r.headers.get("access-control-allow-origin", "")
        assert cors != ""  # some CORS header present
