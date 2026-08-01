"""
Phase 2 — retrieval outcomes, filter-relax retry, connector-agnostic prompts.
"""

from unittest.mock import patch

import pytest

from prompts import (
    INSUFFICIENT_CONTEXT_ANSWER,
    LEGACY_INSUFFICIENT_CONTEXT_ANSWER,
    OUTCOME_EMPTY_CORPUS,
    OUTCOME_FILTERED_OUT,
    OUTCOME_LLM_REFUSAL,
    OUTCOME_NO_USABLE_TEXT,
    OUTCOME_OK,
    answer_for_retrieval_outcome,
    is_insufficient_context_answer,
    system_prompt_for_mode,
)


def _slack_chunk(*, text, source_id, channel="general", ts="1700000000.0", score=0.9):
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": {
            "channel": channel,
            "stable_key": f"slack:msg:{source_id}:{ts}",
            "timestamp": ts,
            "document_type": "message",
        },
    }


def _call(question="what happened?", top_k=5, **kwargs):
    from recall import prepare_recall_context

    kwargs.setdefault("hydradb_sub_tenant_id", "ws_testtenant")
    return prepare_recall_context(question, top_k, **kwargs)


# ---------------------------------------------------------------------- #
# Prompts
# ---------------------------------------------------------------------- #
class TestConnectorAgnosticPrompts:
    def test_insufficient_string_is_not_slack_specific(self):
        assert "Slack" not in INSUFFICIENT_CONTEXT_ANSWER

    def test_system_prompt_mentions_gmail_and_slack(self):
        prompt = system_prompt_for_mode("default")
        assert "Slack" in prompt
        assert "Gmail" in prompt
        assert INSUFFICIENT_CONTEXT_ANSWER in prompt

    def test_is_insufficient_recognizes_legacy_and_new(self):
        assert is_insufficient_context_answer(INSUFFICIENT_CONTEXT_ANSWER)
        assert is_insufficient_context_answer(LEGACY_INSUFFICIENT_CONTEXT_ANSWER)
        assert is_insufficient_context_answer(answer_for_retrieval_outcome(OUTCOME_EMPTY_CORPUS))
        assert not is_insufficient_context_answer("The plan is X.")


# ---------------------------------------------------------------------- #
# Retrieval outcomes
# ---------------------------------------------------------------------- #
class TestRetrievalOutcomes:
    def test_empty_corpus_outcome(self):
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ):
            result = _call("anything")
        assert result["ready"] is False
        debug = result["fallback_debug"]
        assert debug["retrieval_outcome"] == OUTCOME_EMPTY_CORPUS
        assert debug["retrieval"]["outcome"] == OUTCOME_EMPTY_CORPUS
        assert "indexing" in debug["answer"].lower() or "ingested" in debug["answer"].lower()

    def test_no_usable_text_outcome(self):
        # Chunk present but no extractable body text.
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": [{"score": 0.9, "id": "x", "metadata": {"channel": "eng"}}]},
        ):
            result = _call("anything")
        assert result["ready"] is False
        assert result["fallback_debug"]["retrieval_outcome"] == OUTCOME_NO_USABLE_TEXT

    def test_allowed_sources_filter_empty_is_filtered_out(self):
        # Only Slack chunks, but user asked for Gmail-only — not a
        # relaxable filter, so we do NOT retry; outcome is filtered_out.
        chunks = [
            _slack_chunk(text="hello from slack", source_id="s1", channel="general"),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("hello", allowed_sources=["gmail"])
        assert result["ready"] is False
        assert result["fallback_debug"]["retrieval_outcome"] == OUTCOME_FILTERED_OUT
        assert result["fallback_debug"]["filters_relaxed"] is False


# ---------------------------------------------------------------------- #
# Filter-relax retry
# ---------------------------------------------------------------------- #
class TestFilterRelaxRetry:
    def test_wrong_channel_filter_retries_and_returns_chunks(self):
        chunks = [
            _slack_chunk(text="deploy friday", source_id="s1", channel="engineering"),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call(
                "what about deploy?",
                channel="product",  # would wipe all candidates
            )
        assert result["ready"] is True
        assert result["filters_relaxed"] is True
        assert result["retrieval"]["outcome"] == OUTCOME_OK
        assert result["retrieval"]["filters_relaxed"] is True
        assert len(result["sources"]) >= 1

    def test_matching_channel_does_not_need_retry(self):
        chunks = [
            _slack_chunk(text="deploy friday", source_id="s1", channel="engineering"),
            _slack_chunk(text="random chatter", source_id="s2", channel="random"),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("deploy?", channel="engineering")
        assert result["ready"] is True
        assert result["filters_relaxed"] is False
        assert all(s.get("channel") == "engineering" for s in result["sources"])


# ---------------------------------------------------------------------- #
# LLM refusal tagging
# ---------------------------------------------------------------------- #
class TestLlmRefusalOutcome:
    def test_answer_question_tags_llm_refusal(self):
        from recall import answer_question

        prepared = {
            "ready": True,
            "context_text": "[1] (source: eng)\nhello",
            "sources": [{"index": 1, "source": "eng", "channel": "eng"}],
            "chunks_count": 1,
            "filtered_out": 0,
            "exact_matches": 0,
            "retrieval_mode": "default",
            "query_terms": [],
            "filters_relaxed": False,
            "retrieval": {
                "outcome": OUTCOME_OK,
                "reason": "context ready",
                "chunks_returned": 1,
                "chunks_extractable": 1,
                "chunks_filtered_out": 0,
                "filters_relaxed": False,
                "filters_applied": {},
            },
        }
        with patch("recall.prepare_recall_context", return_value=prepared), patch(
            "recall.generate_grounded_answer",
            return_value=INSUFFICIENT_CONTEXT_ANSWER,
        ):
            result = answer_question(
                question="what happened?",
                hydradb_sub_tenant_id="ws_testtenant",
            )
        assert result["debug"]["retrieval_outcome"] == OUTCOME_LLM_REFUSAL
        assert result["debug"]["retrieval"]["outcome"] == OUTCOME_LLM_REFUSAL

    def test_answer_question_uses_outcome_specific_empty_answer(self):
        from recall import answer_question

        prepared = {
            "ready": False,
            "fallback_debug": {
                "reason": "empty",
                "retrieval_outcome": OUTCOME_EMPTY_CORPUS,
                "answer": answer_for_retrieval_outcome(OUTCOME_EMPTY_CORPUS),
                "retrieval": {"outcome": OUTCOME_EMPTY_CORPUS},
            },
        }
        with patch("recall.prepare_recall_context", return_value=prepared):
            result = answer_question(
                question="what happened?",
                hydradb_sub_tenant_id="ws_testtenant",
            )
        assert result["answer"] == answer_for_retrieval_outcome(OUTCOME_EMPTY_CORPUS)
        assert result["debug"]["retrieval_outcome"] == OUTCOME_EMPTY_CORPUS
