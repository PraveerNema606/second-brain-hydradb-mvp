"""
Integration tests: streaming response lifecycle (/api/query/stream).

Flow 2: SSE token stream — verifies event format, ordering, and error paths.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

AUTH = {"X-API-Key": "test-secret-key"}
INSUFFICIENT = "I do not have enough context to answer that."


def _parse_sse(text: str):
    """Parse raw SSE body into list of (event_type, data_dict)."""
    events = []
    current_event = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("event:"):
            current_event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            raw = line.split(":", 1)[1].strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = raw
            if current_event is not None:
                events.append((current_event, data))
                current_event = None
    return events


def _sources():
    return [
        {
            "index": 1,
            "source": "general",
            "channel": "general",
            "user": "Alice",
            "snippet": "snippet",
            "stable_key": "slack:C1:1.0",
            "permalink": "https://slack.com/1",
        }
    ]


class TestStreamingSSEFormat:
    def test_content_type_is_event_stream(self, client):
        with patch("main.prepare_recall_context") as mock_ctx:
            mock_ctx.return_value = {"ready": False, "fallback_debug": {}}
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_cache_control_no_cache(self, client):
        with patch("main.prepare_recall_context") as mock_ctx:
            mock_ctx.return_value = {"ready": False, "fallback_debug": {}}
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        assert "no-cache" in r.headers.get("cache-control", "").lower()

    def test_no_context_emits_token_then_done(self, client):
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
        events = _parse_sse(r.text)
        event_types = [e[0] for e in events]
        assert "token" in event_types
        assert "done" in event_types
        assert event_types.index("token") < event_types.index("done")

    def test_no_context_token_is_fallback_string(self, client):
        with patch("main.prepare_recall_context") as mock_ctx:
            mock_ctx.return_value = {"ready": False, "fallback_debug": {}}
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        events = _parse_sse(r.text)
        token_events = [d for t, d in events if t == "token"]
        assert any(INSUFFICIENT in (d.get("text") or "") for d in token_events)

    def test_done_event_has_sources_and_debug(self, client):
        def _fake_stream(*a, **kw):
            yield "Hello"
            yield " world"

        with patch("main.prepare_recall_context") as mock_ctx, patch(
            "main.stream_grounded_answer", side_effect=_fake_stream
        ), patch("main.finalize_answer") as mock_fin:
            mock_ctx.return_value = {
                "ready": True,
                "context_text": "[1] content",
                "sources": _sources(),
                "chunks_count": 1,
                "filtered_out": 0,
                "exact_matches": 0,
                "retrieval_mode": "default",
                "query_terms": [],
                "fallback_debug": None,
            }
            mock_fin.return_value = {
                "answer": "Hello world",
                "cleaned_sources": _sources(),
                "sources_before": 1,
                "sources_after": 1,
            }
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )

        events = _parse_sse(r.text)
        done_events = [d for t, d in events if t == "done"]
        assert len(done_events) == 1
        done = done_events[0]
        assert "answer" in done
        assert "sources" in done
        assert "debug" in done

    def test_tokens_accumulate_to_full_answer(self, client):
        tokens = ["The ", "sprint ", "deadline ", "is ", "Friday."]

        def _fake_stream(*a, **kw):
            yield from tokens

        with patch("main.prepare_recall_context") as mock_ctx, patch(
            "main.stream_grounded_answer", side_effect=_fake_stream
        ), patch("main.finalize_answer") as mock_fin:
            mock_ctx.return_value = {
                "ready": True,
                "context_text": "[1] context",
                "sources": _sources(),
                "chunks_count": 1,
                "filtered_out": 0,
                "exact_matches": 0,
                "retrieval_mode": "default",
                "query_terms": [],
                "fallback_debug": None,
            }
            mock_fin.return_value = {
                "answer": "".join(tokens).strip(),
                "cleaned_sources": _sources(),
                "sources_before": 1,
                "sources_after": 1,
            }
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )

        events = _parse_sse(r.text)
        token_texts = [d.get("text", "") for t, d in events if t == "token"]
        combined = "".join(token_texts)
        assert "sprint" in combined
        assert "Friday" in combined

    def test_llm_error_emits_error_event(self, client):
        from errors import LLMError

        def _bad_stream(*a, **kw):
            raise LLMError("LLM is down")
            yield  # make it a generator

        with patch("main.prepare_recall_context") as mock_ctx, patch(
            "main.stream_grounded_answer", side_effect=_bad_stream
        ):
            mock_ctx.return_value = {
                "ready": True,
                "context_text": "[1] content",
                "sources": _sources(),
                "chunks_count": 1,
                "filtered_out": 0,
                "exact_matches": 0,
                "retrieval_mode": "default",
                "query_terms": [],
                "fallback_debug": None,
            }
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )

        events = _parse_sse(r.text)
        error_events = [d for t, d in events if t == "error"]
        assert len(error_events) >= 1
        assert "llm_error" in error_events[0].get("error_type", "")

    def test_hydradb_error_emits_error_event(self, client):
        from errors import HydraDBError

        with patch("recall.prepare_recall_context", side_effect=HydraDBError("HydraDB down")):
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )

        events = _parse_sse(r.text)
        error_events = [d for t, d in events if t == "error"]
        assert len(error_events) >= 1

    def test_debug_has_cache_hit_false(self, client):
        def _fake_stream(*a, **kw):
            yield "answer"

        with patch("main.prepare_recall_context") as mock_ctx, patch(
            "main.stream_grounded_answer", side_effect=_fake_stream
        ), patch("main.finalize_answer") as mock_fin:
            mock_ctx.return_value = {
                "ready": True,
                "context_text": "[1] ctx",
                "sources": _sources(),
                "chunks_count": 1,
                "filtered_out": 0,
                "exact_matches": 0,
                "retrieval_mode": "default",
                "query_terms": [],
                "fallback_debug": None,
            }
            mock_fin.return_value = {
                "answer": "answer",
                "cleaned_sources": _sources(),
                "sources_before": 1,
                "sources_after": 1,
            }
            r = client.post(
                "/api/query/stream",
                json={"question": "what happened?"},
                headers=AUTH,
            )

        events = _parse_sse(r.text)
        done_events = [d for t, d in events if t == "done"]
        assert done_events[0]["debug"]["cache_hit"] is False

    def test_no_auth_returns_401(self, client):
        r = client.post("/api/query/stream", json={"question": "what happened?"})
        assert r.status_code == 401

    def test_history_present_sets_cache_bypassed(self, client):
        with patch("main.prepare_recall_context") as mock_ctx:
            mock_ctx.return_value = {"ready": False, "fallback_debug": {}}
            r = client.post(
                "/api/query/stream",
                json={
                    "question": "what happened?",
                    "conversation_history": [
                        {"role": "user", "content": "prior turn"},
                    ],
                },
                headers=AUTH,
            )
        events = _parse_sse(r.text)
        done_events = [d for t, d in events if t == "done"]
        if done_events:
            debug = done_events[0].get("debug", {})
            assert "cache_bypassed" in debug
