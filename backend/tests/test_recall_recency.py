"""
Tests for the "latest message" / recency-intent path in recall.py.

Production symptom that motivated this code:
    "what is the latest message" returned an older Slack message
    instead of the newest one.

These tests pin the contract:

  1. Recency intent fires for Slack-message queries with cues
     "latest" / "newest" / "most recent" / "last" + a message-noun.
  2. The reranker sorts surviving Slack chunks by Slack ts DESC.
  3. Channel filter is respected when the user says "in engineering"
     or "in #engineering".
  4. Semantic ranking is preserved for non-recency questions.
  5. Recency overrides semantic ranking when a newer-but-less-relevant
     message is in the candidate set.
  6. Boundary case "latest news" -> no recency intent (no Slack noun).
"""

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------- #
# Fixture helpers
# ---------------------------------------------------------------------- #
def _slack_chunk(
    *,
    text,
    source_id,
    stable_key,
    ts,
    channel="general",
    score=0.9,
):
    """
    Build a HydraDB-shaped chunk that represents an INGESTED SLACK
    MESSAGE. The recency reranker looks at document_type AND the
    stable_key prefix; we set both so the test isn't sensitive to
    which path the heuristic uses.
    """
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": {
            "channel": channel,
            "stable_key": stable_key,  # also surfaced as "slack:..."
            "timestamp": ts,
            "document_type": "message",
        },
    }


def _non_slack_chunk(*, text, source_id, ts=None):
    """A doc that should NOT participate in the recency rerank
    (e.g. a Gmail email). No 'slack:' stable_key prefix; no
    document_type=='message'."""
    md = {"stable_key": f"gmail:msg:{source_id}", "document_type": "email"}
    if ts is not None:
        md["timestamp"] = ts
    return {
        "text": text,
        "score": 0.95,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": md,
    }


def _call(question, top_k=5, **kwargs):
    from recall import prepare_recall_context

    kwargs.setdefault("hydradb_sub_tenant_id", "ws_testtenant")
    return prepare_recall_context(question, top_k, **kwargs)


# ---------------------------------------------------------------------- #
# Recency intent detector
# ---------------------------------------------------------------------- #
class TestRecencyIntentDetector:
    @pytest.mark.parametrize(
        "q",
        [
            "what is the latest message",
            "what is the latest message in engineering",
            "latest message in #engineering",
            "newest message",
            "most recent message",
            "show me the latest slack message",
            "most recent post in product",
            "last ping in announcements",
        ],
    )
    def test_fires_for_slack_message_queries(self, q):
        from recall import _detect_recency_intent

        assert _detect_recency_intent(q) is True

    @pytest.mark.parametrize(
        "q",
        [
            # Existing test_query_rewriter boundary: "latest news" must
            # NOT trigger -- it's a generic semantic query.
            "what is the latest news?",
            # "Recent decision" is a content question, not "show me the
            # newest log line".
            "what's the most recent decision",
            "summarize the latest update",
            # No recency cue at all.
            "what did Alice say about the migration",
            "tell me about the engineering channel",
            # "lastly" looks like "last" but isn't a recency cue.
            "lastly, what happened in engineering",
            "",
        ],
    )
    def test_does_not_fire_for_non_recency_queries(self, q):
        from recall import _detect_recency_intent

        assert _detect_recency_intent(q) is False


# ---------------------------------------------------------------------- #
# Global "latest message"
# ---------------------------------------------------------------------- #
class TestLatestGlobal:
    def test_latest_message_returns_newest_slack_timestamp(self):
        """Three Slack messages. The newest must come back first even
        if its semantic score is lower than an older message's."""
        chunks = [
            _slack_chunk(
                text="Morning Reema. Quick status on the Kafka event migration",
                source_id="d-old",
                stable_key="slack:msg:C1:1700000000",
                ts="1700000000.000",  # oldest
                channel="engineering",
                score=0.99,  # high semantic relevance
            ),
            _slack_chunk(
                text="REALTIME TEST 999",
                source_id="d-mid",
                stable_key="slack:msg:C1:1735000000",
                ts="1735000000.000",
                channel="engineering",
                score=0.50,
            ),
            _slack_chunk(
                text="PROD REALTIME FINAL TEST 12345 23:15",
                source_id="d-new",
                stable_key="slack:msg:C1:1740000000",
                ts="1740000000.000",  # newest
                channel="engineering",
                score=0.30,  # lowest semantic relevance
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("what is the latest message", top_k=1)
        assert result["ready"] is True
        # The newest message is the first (and only) source.
        assert len(result["sources"]) == 1
        first = result["sources"][0]
        # Confirm we got the newest one, not the older-but-higher-score.
        assert first["stable_key"] == "slack:msg:C1:1740000000"
        # Mode is reported as "recency" so the UI can show it if needed.
        assert result["retrieval_mode"] == "recency"

    def test_recency_widens_candidate_pool(self):
        """top_k=1 by itself would only ask HydraDB for 1 result. The
        recency path must widen the candidate pool to ~50 so the
        newest message has a chance of appearing."""
        from recall import _RECENCY_CANDIDATE_POOL

        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ) as mock_recall:
            _call("what is the latest message", top_k=1)
        # full_recall was called with the WIDENED top_k.
        called_top_k = mock_recall.call_args.kwargs.get("top_k")
        assert called_top_k == _RECENCY_CANDIDATE_POOL


# ---------------------------------------------------------------------- #
# Channel-specific "latest message in #engineering"
# ---------------------------------------------------------------------- #
class TestLatestInChannel:
    def _two_channel_corpus(self):
        return [
            # Older "engineering" message.
            _slack_chunk(
                text="engineering older msg",
                source_id="e-old",
                stable_key="slack:msg:E1:1700000000",
                ts="1700000000.0",
                channel="engineering",
            ),
            # NEWER message in a DIFFERENT channel -- this must NOT win
            # when the user asks for "latest in engineering".
            _slack_chunk(
                text="random newer msg",
                source_id="r-new",
                stable_key="slack:msg:R1:1740000000",
                ts="1740000000.0",
                channel="random",
            ),
            # Newest "engineering" message -- this is the expected answer.
            _slack_chunk(
                text="engineering newest msg",
                source_id="e-new",
                stable_key="slack:msg:E1:1739000000",
                ts="1739000000.0",
                channel="engineering",
            ),
        ]

    def test_bare_channel_name(self):
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": self._two_channel_corpus()},
        ):
            result = _call(
                "what is the latest message in engineering",
                top_k=1,
                channel="engineering",
            )
        assert result["sources"][0]["stable_key"] == "slack:msg:E1:1739000000"

    def test_hash_prefixed_channel_name(self):
        # When the caller passes channel="#engineering" the filter must
        # still match the stored "engineering" channel.
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": self._two_channel_corpus()},
        ):
            result = _call(
                "latest message in #engineering",
                top_k=1,
                channel="#engineering",
            )
        assert result["sources"][0]["stable_key"] == "slack:msg:E1:1739000000"

    def test_other_channels_filtered_out(self):
        # With a tight top_k the only surviving row should be from the
        # requested channel, even though a newer row exists elsewhere.
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": self._two_channel_corpus()},
        ):
            result = _call(
                "latest message in engineering",
                top_k=5,
                channel="engineering",
            )
        for src in result["sources"]:
            assert src["channel"] == "engineering"


# ---------------------------------------------------------------------- #
# Semantic mode is preserved for non-recency questions
# ---------------------------------------------------------------------- #
class TestSemanticPreserved:
    def test_non_recency_question_does_not_get_recency_rerank(self):
        """For "what did Alice say about Kafka", the higher-semantic
        score (HydraDB's default order) wins, NOT the newest timestamp."""
        chunks = [
            _slack_chunk(
                text="Alice on Kafka: the migration is going well",
                source_id="alice-1",
                stable_key="slack:msg:E1:1700000000",
                ts="1700000000.0",
                channel="engineering",
                score=0.95,
            ),
            _slack_chunk(
                text="unrelated newest message",
                source_id="newest-1",
                stable_key="slack:msg:E1:1740000000",
                ts="1740000000.0",
                channel="engineering",
                score=0.20,
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call(
                "what did Alice say about Kafka?",
                top_k=1,
            )
        # The HIGH-semantic-score chunk wins, regardless of timestamp.
        assert result["sources"][0]["stable_key"] == "slack:msg:E1:1700000000"
        # Mode should NOT have been overridden to "recency".
        assert result["retrieval_mode"] != "recency"


# ---------------------------------------------------------------------- #
# Recency reranker beats semantic when query asks for "latest"
# ---------------------------------------------------------------------- #
class TestRealtimeBeatsSemantic:
    def test_realtime_new_message_beats_older_semantic_match(self):
        """The headline production case: a newly-ingested short
        message ("PROD REALTIME FINAL TEST 12345") must outrank an
        older but semantically lush message ("Morning Reema...Kafka")
        when the user asks for the latest message."""
        chunks = [
            _slack_chunk(
                text="Morning Reema. Quick status on the Kafka event migration",
                source_id="kafka-1",
                stable_key="slack:msg:E1:1700000000",
                ts="1700000000.0",
                channel="engineering",
                score=0.99,
            ),
            _slack_chunk(
                text="PROD REALTIME FINAL TEST 12345 23:15",
                source_id="prod-1",
                stable_key="slack:msg:E1:1740000000",
                ts="1740000000.0",
                channel="engineering",
                score=0.10,
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("what is the latest message in engineering", top_k=1, channel="engineering")
        first_text = result["context_text"].split("\n", 1)[-1]
        assert "PROD REALTIME FINAL TEST" in result["context_text"]
        # And the older-but-relevant message is NOT the chosen answer.
        assert first_text.startswith(result["context_text"].splitlines()[1])  # first source IS the prod one


# ---------------------------------------------------------------------- #
# Fallback behavior when recency intent fires but no Slack chunks exist
# ---------------------------------------------------------------------- #
class TestRecencyFallback:
    def test_gmail_with_timestamp_participates_in_recency(self):
        """Phase 10: a Gmail email WITH a parseable timestamp is a
        valid recency candidate. "Latest message" + Gmail-only corpus
        now returns the newest Gmail email via the recency path
        (instead of silently falling back to semantic).

        This is a deliberate behavior change from Phase 9 where the
        rerank was Slack-only. See test_ranking_v2.py for the
        Gmail-first tests that motivated this."""
        chunks = [
            _non_slack_chunk(
                text="email subject: status update",
                source_id="email-1",
                ts="1740000000.0",
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("what is the latest message", top_k=1)
        assert result["ready"] is True
        # NEW: Gmail email participates in recency rerank.
        assert result["retrieval_mode"] == "recency"

    def test_no_recency_eligible_chunks_falls_back_to_semantic(self):
        """The "no eligible chunks" fallback is unchanged from Phase 9
        in spirit -- it just covers a smaller set of cases now that
        Gmail counts as eligible. Here we pass an unclassifiable chunk
        (no document_type, no recognizable stable_key prefix) so
        nothing qualifies for the recency rerank and semantic mode
        wins."""
        chunks = [
            {
                "text": "weird chunk with no source signals",
                "score": 0.9,
                "source_id": "weird-1",
                "filename": "weird-1.md",
                "metadata": {},  # no document_type, no stable_key
            }
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("what is the latest message", top_k=1)
        assert result["ready"] is True
        assert result["retrieval_mode"] != "recency"

    def test_slack_chunk_without_timestamp_falls_back(self):
        """A Slack chunk that somehow has no parseable timestamp must
        not crash the recency reranker -- it falls back to semantic."""
        chunks = [
            _slack_chunk(
                text="weird old chunk with no ts",
                source_id="ts-less",
                stable_key="slack:msg:E1:notnumeric",
                ts="not-a-timestamp",
                channel="engineering",
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("latest message in engineering", top_k=1, channel="engineering")
        assert result["ready"] is True
        # No usable recency chunk -> falls back to semantic mode.
        assert result["retrieval_mode"] != "recency"
