"""
Tests for the `allowed_sources` filter on `/api/query`.

The feature: a user can restrict a query to a subset of connector
sources by passing `allowed_sources: ["slack"]` (or `["gmail"]`, or
both). Omitting the field / passing null / passing an empty list
preserves the pre-Phase-9 behavior — all sources allowed.

These tests pin five contracts:

  1. Default (no filter / null / []) returns chunks from every
     source, matching the pre-Phase-9 behavior exactly.
  2. ["slack"] excludes Gmail chunks; ["gmail"] excludes Slack chunks.
  3. ["slack","gmail"] is the explicit "allow both" form. Result is
     identical to "no filter".
  4. The source filter composes correctly with the existing channel
     filter (intersection — both must pass).
  5. The recency rerank (Slack-message-aware) keeps working when
     allowed_sources=["slack"] — i.e. the filter doesn't accidentally
     break the recency code path.
"""

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------- #
# Fixture helpers
# ---------------------------------------------------------------------- #
# Same chunk-shape conventions as test_recall_recency.py so the recency
# co-tests below stay coherent with the existing fixtures.


def _slack_chunk(
    *,
    text,
    source_id,
    channel="general",
    ts=None,
    score=0.9,
    document_type="message",
):
    """A HydraDB chunk that represents an ingested Slack message.

    `ts` defaults to a value derived from source_id so each test
    fixture gets a distinct stable_key (dedupe_by_stable_key would
    otherwise collapse two chunks that share a key)."""
    if ts is None:
        # Deterministic, unique-per-source-id. Hash the source_id and
        # bias into a recent-Slack-timestamp range so anything that
        # parses it as unix seconds still gets a sane value.
        ts = f"1700{abs(hash(source_id)) % 1_000_000:06d}.0"
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": {
            "channel": channel,
            "stable_key": f"slack:msg:C1:{ts}:{source_id}",
            "timestamp": ts,
            "document_type": document_type,
        },
    }


def _gmail_chunk(*, text, source_id, score=0.9):
    """A HydraDB chunk that represents an ingested Gmail email."""
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": {
            "stable_key": f"gmail:msg:{source_id}",
            "document_type": "email",
        },
    }


def _unknown_chunk(*, text, source_id, score=0.9):
    """A chunk we can't classify — no document_type, no recognizable
    stable_key prefix. Should pass through every source filter so we
    never silently drop a legitimate match (current policy)."""
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": {},
    }


def _call(question, top_k=5, **kwargs):
    from recall import prepare_recall_context

    kwargs.setdefault("hydradb_sub_tenant_id", "ws_testtenant")
    return prepare_recall_context(question, top_k, **kwargs)


def _source_ids(result):
    """Helper: pull source_id-equivalent fingerprints out of the
    rendered context. The source cards' `source` field carries
    minimal_source which is the chunk's source_id or filename — both
    of which we set in the fixtures above."""
    sources = result.get("sources") or []
    return [s.get("source") for s in sources]


# ---------------------------------------------------------------------- #
# Helper-level: _extract_source_kind
# ---------------------------------------------------------------------- #
class TestExtractSourceKind:
    def test_slack_via_document_type(self):
        from recall import _extract_source_kind

        assert _extract_source_kind({"document_type": "message"}) == "slack"
        assert _extract_source_kind({"document_type": "thread"}) == "slack"

    def test_gmail_via_document_type(self):
        from recall import _extract_source_kind

        assert _extract_source_kind({"document_type": "email"}) == "gmail"

    def test_slack_via_stable_key_fallback(self):
        from recall import _extract_source_kind

        # document_type missing; only the prefix tells us.
        assert _extract_source_kind({"stable_key": "slack:msg:C1:123"}) == "slack"
        assert _extract_source_kind({"stable_key": "slack:thread:C1:123"}) == "slack"

    def test_gmail_via_stable_key_fallback(self):
        from recall import _extract_source_kind

        assert _extract_source_kind({"stable_key": "gmail:msg:abc"}) == "gmail"

    def test_unknown_returns_none(self):
        from recall import _extract_source_kind

        # No document_type, no recognizable prefix -> None (unknown).
        assert _extract_source_kind({}) is None
        assert _extract_source_kind({"stable_key": "weird:thing"}) is None
        assert _extract_source_kind({"document_type": "totally_made_up"}) is None

    def test_document_type_wins_over_stable_key(self):
        # If both are present and they disagree, document_type wins
        # (it's the more authoritative signal — set by the ingest
        # builder directly).
        from recall import _extract_source_kind

        card = {
            "document_type": "email",
            "stable_key": "slack:msg:C1:123",  # contradictory
        }
        assert _extract_source_kind(card) == "gmail"

    def test_non_dict_returns_none(self):
        from recall import _extract_source_kind

        assert _extract_source_kind(None) is None
        assert _extract_source_kind("not-a-card") is None


# ---------------------------------------------------------------------- #
# Filter behavior end-to-end via prepare_recall_context
# ---------------------------------------------------------------------- #
class TestSourceFilterEndToEnd:
    def _mixed_corpus(self):
        return [
            _slack_chunk(text="slack-A", source_id="sa", channel="engineering"),
            _slack_chunk(text="slack-B", source_id="sb", channel="random"),
            _gmail_chunk(text="gmail-X", source_id="gx"),
            _gmail_chunk(text="gmail-Y", source_id="gy"),
        ]

    def _run(self, question="anything I want?", top_k=10, **kwargs):
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": self._mixed_corpus()},
        ):
            return _call(question, top_k=top_k, **kwargs)

    # ── Default / null / [] — all sources ─────────────────────────────
    def test_default_no_filter_returns_all_sources(self):
        result = self._run()
        ids = _source_ids(result)
        # All 4 mixed chunks survived (semantic q, no filter).
        assert "sa" in ids and "sb" in ids
        assert "gx" in ids and "gy" in ids

    def test_null_filter_returns_all_sources(self):
        result = self._run(allowed_sources=None)
        ids = _source_ids(result)
        assert {"sa", "sb", "gx", "gy"}.issubset(set(ids))

    def test_empty_list_returns_all_sources(self):
        # Empty list collapses to "no filter" (default-behavior
        # preservation requirement #1 from the task spec).
        result = self._run(allowed_sources=[])
        ids = _source_ids(result)
        assert {"sa", "sb", "gx", "gy"}.issubset(set(ids))

    # ── ["slack"] — slack-only ────────────────────────────────────────
    def test_slack_only_excludes_gmail(self):
        result = self._run(allowed_sources=["slack"])
        ids = _source_ids(result)
        assert "sa" in ids and "sb" in ids
        assert "gx" not in ids and "gy" not in ids
        # filtered_out is a per-chunk counter, must reflect the gmails.
        assert result["filtered_out"] >= 2

    # ── ["gmail"] — gmail-only ────────────────────────────────────────
    def test_gmail_only_excludes_slack(self):
        result = self._run(allowed_sources=["gmail"])
        ids = _source_ids(result)
        assert "gx" in ids and "gy" in ids
        assert "sa" not in ids and "sb" not in ids
        assert result["filtered_out"] >= 2

    # ── ["slack","gmail"] — explicit both ─────────────────────────────
    def test_both_explicit_returns_all_sources(self):
        # Same observable behavior as "no filter"; included so the UI
        # can render an explicit "Slack + Gmail" choice without that
        # being a no-op vs "All".
        result = self._run(allowed_sources=["slack", "gmail"])
        ids = _source_ids(result)
        assert {"sa", "sb", "gx", "gy"}.issubset(set(ids))

    # ── Normalization: trimmed, lowercased, deduped ───────────────────
    def test_normalization_trims_lowercases_dedupes(self):
        # Garbage in -> still parses correctly. Mirrors what a sloppy
        # frontend might send.
        result = self._run(
            allowed_sources=[" Slack ", "slack", "", "  ", "SLACK"],
        )
        ids = _source_ids(result)
        # Effectively ["slack"] -> Gmail excluded.
        assert "sa" in ids and "sb" in ids
        assert "gx" not in ids and "gy" not in ids

    # ── Unknown-source cards pass through ─────────────────────────────
    def test_unknown_source_chunks_pass_through(self):
        # If a chunk has no document_type AND no recognizable stable_key
        # prefix, the filter SHOULD let it through (current "don't
        # silently drop" policy mirrors how channel/user filters behave).
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={
                "chunks": [
                    _slack_chunk(text="slack-A", source_id="sa"),
                    _unknown_chunk(text="weird", source_id="weird"),
                ]
            },
        ):
            result = _call("anything?", top_k=10, allowed_sources=["slack"])
        ids = _source_ids(result)
        # Both pass: the slack one matches, the unknown one is let
        # through under the don't-drop rule.
        assert "sa" in ids
        assert "weird" in ids


# ---------------------------------------------------------------------- #
# Composition with the channel filter
# ---------------------------------------------------------------------- #
class TestSourceFilterComposesWithChannel:
    def _two_channel_corpus(self):
        return [
            _slack_chunk(text="eng-1", source_id="e1", channel="engineering"),
            _slack_chunk(text="rand-1", source_id="r1", channel="random"),
            _gmail_chunk(text="email-1", source_id="g1"),
        ]

    def test_slack_only_plus_channel(self):
        # Source filter AND channel filter must both pass (intersection).
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": self._two_channel_corpus()},
        ):
            result = _call(
                "what did the team discuss?",
                top_k=10,
                channel="engineering",
                allowed_sources=["slack"],
            )
        ids = _source_ids(result)
        # Only the engineering Slack chunk survives. The other Slack
        # chunk is dropped by the channel filter, and the Gmail chunk
        # is dropped by the source filter.
        assert "e1" in ids
        assert "r1" not in ids
        assert "g1" not in ids

    def test_gmail_only_plus_channel(self):
        # Gmail doesn't have channels; the channel filter only applies
        # when the card carries a `channel` (Gmail cards don't), so the
        # gmail chunk should pass the channel filter (don't-drop rule)
        # AND pass the source filter.
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": self._two_channel_corpus()},
        ):
            result = _call(
                "any emails?",
                top_k=10,
                channel="engineering",
                allowed_sources=["gmail"],
            )
        ids = _source_ids(result)
        assert "g1" in ids
        assert "e1" not in ids and "r1" not in ids


# ---------------------------------------------------------------------- #
# Recency rerank still works under allowed_sources=["slack"]
# ---------------------------------------------------------------------- #
class TestRecencyStillWorksUnderSlackFilter:
    def test_latest_message_with_slack_only_filter(self):
        # The recency reranker sorts surviving Slack chunks by ts DESC.
        # Adding a Slack-only filter MUST NOT break that behavior.
        chunks = [
            _slack_chunk(
                text="older slack",
                source_id="old",
                channel="engineering",
                ts="1700000000.0",
                score=0.99,
            ),
            _slack_chunk(
                text="NEWEST slack",
                source_id="new",
                channel="engineering",
                ts="1740000000.0",
                score=0.30,
            ),
            _gmail_chunk(text="some email", source_id="email-1"),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call(
                "what is the latest message in engineering",
                top_k=1,
                channel="engineering",
                allowed_sources=["slack"],
            )
        # Newest Slack message wins; Gmail excluded; recency mode active.
        assert result["sources"][0]["source"] in ("new",)
        assert result["retrieval_mode"] == "recency"

    def test_gmail_only_under_recency_falls_back_to_semantic(self):
        # If the user asks for "latest message" but restricts to Gmail
        # only and the Gmail chunks have NO parseable timestamp (no
        # Date header harvested, no `timestamp` in metadata), there
        # are no recency-eligible candidates and the pipeline falls
        # back to semantic mode. Phase 10 made Gmail eligible for
        # recency in principle; the eligibility still depends on a
        # timestamp being present.
        chunks = [
            _gmail_chunk(text="email A", source_id="ea", score=0.95),
            _gmail_chunk(text="email B", source_id="eb", score=0.30),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call(
                "what is the latest message",
                top_k=2,
                allowed_sources=["gmail"],
            )
        assert result["ready"] is True
        # No Slack chunks survived (correctly) -> NOT recency mode.
        assert result["retrieval_mode"] != "recency"
        ids = _source_ids(result)
        # Both Gmail chunks should be available.
        assert "ea" in ids


# ---------------------------------------------------------------------- #
# Pydantic-level: the request model rejects unknown source names
# ---------------------------------------------------------------------- #
class TestRequestModelValidation:
    def test_known_sources_accepted(self):
        from main import QueryRequest

        for s in ([], ["slack"], ["gmail"], ["slack", "gmail"]):
            req = QueryRequest(question="hello world", allowed_sources=s)
            assert req.allowed_sources == s

    def test_unknown_source_is_422(self):
        # Pydantic's Literal enforcement turns a typo into a 422 at the
        # API boundary instead of silently no-op'ing in the filter.
        from pydantic import ValidationError

        from main import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(question="hello world", allowed_sources=["notion"])

    def test_null_accepted(self):
        from main import QueryRequest

        req = QueryRequest(question="hello world", allowed_sources=None)
        assert req.allowed_sources is None

    def test_field_omitted_defaults_to_none(self):
        from main import QueryRequest

        req = QueryRequest(question="hello world")
        assert req.allowed_sources is None
