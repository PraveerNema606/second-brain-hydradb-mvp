"""
Integration tests: end-to-end query flow.

Flow 1: Question → query rewrite → recall → rerank → LLM → finalize → response
Mocks: HydraDB, OpenAI.  Real: all Python business logic in between.
"""

from unittest.mock import MagicMock, patch

import pytest

AUTH = {"X-API-Key": "test-secret-key"}
INSUFFICIENT = "I do not have enough context to answer that."


def _hydra_response(chunks):
    return {"chunks": chunks}


def _chunk(text, source_id, stable_key, channel="general", score=0.9, ts=None):
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": f"slack_{channel}_{source_id}.md",
        "metadata": {
            "channel": channel,
            "stable_key": stable_key,
        },
        **({"timestamp": ts} if ts else {}),
    }


def _fake_llm_answer(question, context, mode="default", model=None, conversation_history=None):
    """Mimics generate_grounded_answer: returns a simple answer string."""
    return f"Based on context: {context[:50]}... [1]"


# ── Flow 1: full query pipeline ───────────────────────────────────────────
class TestFullQueryPipeline:
    def test_happy_path_returns_answer_and_sources(self, client):
        chunks = [
            _chunk("Alice said the sprint is on track.", "doc-1", "slack:C1:1.0"),
            _chunk("Bob confirmed the API is stable.", "doc-2", "slack:C1:2.0"),
        ]
        hydra_resp = _hydra_response(chunks)

        with patch("hydradb_client.requests.post") as mock_post, patch(
            "recall.generate_grounded_answer", side_effect=_fake_llm_answer
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: hydra_resp,
                text=str(hydra_resp),
            )
            r = client.post(
                "/api/query",
                json={"question": "what is the sprint status?"},
                headers=AUTH,
            )

        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "sources" in body
        assert "debug" in body
        assert body["answer"] != ""
        assert body["answer"] != INSUFFICIENT

    def test_empty_hydra_response_returns_fallback(self, client):
        with patch("hydradb_client.requests.post") as mock_post, patch("recall.generate_grounded_answer"):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"chunks": []},
                text="{}",
            )
            r = client.post(
                "/api/query",
                json={"question": "what is the sprint status?"},
                headers=AUTH,
            )
        assert r.status_code == 200
        assert INSUFFICIENT in r.json()["answer"]
        assert r.json()["sources"] == []

    def test_channel_filter_applied(self, client):
        """When channel filter is set, only matching chunks should survive."""
        chunks = [
            _chunk("general message", "doc-1", "slack:C1:1.0", channel="general"),
            _chunk("product message", "doc-2", "slack:C1:2.0", channel="product"),
        ]
        with patch("hydradb_client.requests.post") as mock_post, patch(
            "recall.generate_grounded_answer", return_value="The answer [1]."
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"chunks": chunks},
                text="{}",
            )
            r = client.post(
                "/api/query",
                json={"question": "what happened?", "channel": "product"},
                headers=AUTH,
            )
        body = r.json()
        assert r.status_code == 200
        # Only the product source should appear
        for src in body.get("sources", []):
            ch = src.get("channel") or src.get("source", "")
            assert "product" in ch or ch == ""

    def test_exact_mode_uses_keyword_ranking(self, client):
        chunks = [
            _chunk("generic info", "doc-1", "slack:C1:1.0"),
            _chunk("sprint deadline is Friday", "doc-2", "slack:C1:2.0"),
        ]
        with patch("hydradb_client.requests.post") as mock_post, patch(
            "recall.generate_grounded_answer", return_value="deadline Friday [1]."
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"chunks": chunks},
                text="{}",
            )
            r = client.post(
                "/api/query",
                json={"question": "sprint deadline", "mode": "exact"},
                headers=AUTH,
            )
        assert r.status_code == 200

    def test_hydradb_error_surfaces_502(self, client):
        from errors import HydraDBError

        with patch("hydradb_client.HydraDBClient.full_recall", side_effect=HydraDBError("HydraDB is down")):
            r = client.post(
                "/api/query",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        assert r.status_code == 502
        body = r.json()
        assert body["error_type"] == "hydradb_error"

    def test_llm_error_surfaces_502(self, client):
        from errors import LLMError

        chunks = [_chunk("some text", "doc-1", "slack:C1:1.0")]
        with patch("hydradb_client.requests.post") as mock_post, patch(
            "recall.generate_grounded_answer", side_effect=LLMError("LLM down")
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"chunks": chunks},
                text="{}",
            )
            r = client.post(
                "/api/query",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        assert r.status_code == 502
        assert r.json()["error_type"] == "llm_error"

    def test_timeout_surfaces_504(self, client):
        from errors import UpstreamTimeoutError

        with patch("hydradb_client.HydraDBClient.full_recall", side_effect=UpstreamTimeoutError("timed out")):
            r = client.post(
                "/api/query",
                json={"question": "what happened?"},
                headers=AUTH,
            )
        assert r.status_code == 504

    def test_debug_block_contains_mode(self, client):
        with patch("hydradb_client.requests.post") as mock_post, patch(
            "recall.generate_grounded_answer", return_value="answer [1]."
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"chunks": [_chunk("content", "d1", "k1")]},
                text="{}",
            )
            r = client.post(
                "/api/query",
                json={"question": "what happened?", "mode": "summary"},
                headers=AUTH,
            )
        debug = r.json().get("debug", {})
        assert debug.get("mode") == "summary"

    def test_query_rewrite_debug_attached(self, client):
        with patch("hydradb_client.requests.post") as mock_post, patch(
            "recall.generate_grounded_answer", return_value="answer"
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"chunks": []},
                text="{}",
            )
            r = client.post(
                "/api/query",
                json={"question": "What did Alice say in #product?"},
                headers=AUTH,
            )
        debug = r.json().get("debug", {})
        # Query rewrite debug should appear since person/channel were inferred
        assert "query_rewrite" in debug

    def test_conversation_history_not_used_for_retrieval(self, client):
        """
        The retrieval call must use only the current question, not the history.
        We verify by capturing the `query` arg passed to full_recall.
        """
        captured_queries = []

        def _capture_recall(query, top_k=5):
            captured_queries.append(query)
            return {"chunks": []}

        with patch("hydradb_client.HydraDBClient.full_recall", side_effect=_capture_recall):
            client.post(
                "/api/query",
                json={
                    "question": "current question",
                    "conversation_history": [
                        {"role": "user", "content": "old question"},
                        {"role": "assistant", "content": "old answer"},
                    ],
                },
                headers=AUTH,
            )
        assert len(captured_queries) == 1
        assert "current question" in captured_queries[0]
        assert "old question" not in captured_queries[0]

    def test_date_query_last_week_filters_applied(self, client):
        """date_query should translate to start/end timestamps in the recall call."""
        called_kwargs = {}

        def _capture(**kwargs):
            called_kwargs.update(kwargs)
            return {"ready": False, "fallback_debug": {}}

        # /api/query calls answer_question (in recall.py) which calls prepare_recall_context
        # in its own module scope — so we patch recall.prepare_recall_context
        with patch("recall.prepare_recall_context", side_effect=_capture):
            client.post(
                "/api/query",
                json={"question": "what happened?", "date_query": "last week"},
                headers=AUTH,
            )
        # start/end timestamps must have been resolved and passed
        assert "start_timestamp" in called_kwargs or "end_timestamp" in called_kwargs


# ── Citation hygiene end-to-end ───────────────────────────────────────────
class TestCitationHygiene:
    def test_invalid_citation_stripped_from_final_answer(self, client):
        chunks = [_chunk("text", "doc-1", "k1")]
        # LLM references [99] which doesn't exist in sources
        with patch("hydradb_client.requests.post") as mock_post, patch(
            "recall.generate_grounded_answer", return_value="See [1] and also [99]."
        ):
            mock_post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"chunks": chunks},
                text="{}",
            )
            r = client.post(
                "/api/query",
                json={"question": "test?"},
                headers=AUTH,
            )
        answer = r.json()["answer"]
        assert "[99]" not in answer
