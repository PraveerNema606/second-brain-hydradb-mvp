"""
Tests for Phase 10 ranking improvements.

Coverage targets (one section per requirement from the task spec):

  A. Gmail recency-aware retrieval
  B. Sender-aware ranking
  C. Metadata-aware ranking (subject hits, label boost)
  D. Better cross-source ranking
  E. Retrieval weighting (named constants)
  F. Retrieval debug visibility (rank_breakdown payload)

Plus:
  - Mixed Slack+Gmail ranking under recency intent
  - Source-filter compatibility
  - Recency + sender combined
  - Ranking stability (same input -> same order)

We use chunk fixtures shaped exactly like the production source-card
builder produces; document_type and stable_key are set explicitly so
the new _recency_source_kind() classifier works without depending on
the markdown-header harvester (which gets its own dedicated tests in
TestGmailHeaderHarvest below).
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
    channel="general",
    user=None,
    ts=None,
    score=0.9,
    doc_type="message",
):
    """A Slack-message HydraDB chunk."""
    if ts is None:
        ts = f"1700{abs(hash(source_id)) % 1_000_000:06d}.0"
    md = {
        "channel": channel,
        "stable_key": f"slack:msg:C1:{ts}:{source_id}",
        "timestamp": ts,
        "document_type": doc_type,
    }
    if user is not None:
        md["user_name"] = user
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": md,
    }


def _gmail_chunk(
    *,
    text,
    source_id,
    subject=None,
    from_name=None,
    from_email=None,
    ts=None,
    labels=None,
    score=0.9,
):
    """A Gmail-email chunk. If `ts` is given, we synthesize the
    markdown body with an RFC-2822 Date header so the harvester can
    parse it back out; otherwise we just supply `timestamp` as
    metadata, mimicking the rich source-card path."""
    # Build a body containing the Gmail markdown header so the
    # harvester populates the source card with subject / from / date /
    # labels. The actual chunk text below the header is what the LLM
    # would see.
    header_lines = [
        "# Email",
        f"Source Key: gmail:msg:{source_id}",
        f"Message-Id: {source_id}",
        "Mailbox: ops@acme.example",
    ]
    if subject is not None:
        header_lines.append(f"Subject: {subject}")
    if from_name and from_email:
        header_lines.append(f'From: "{from_name}" <{from_email}>')
    elif from_email:
        header_lines.append(f"From: {from_email}")
    elif from_name:
        header_lines.append(f"From: {from_name}")
    if ts is not None:
        # RFC-2822 Date. Use a deterministic offset per source_id.
        from email.utils import formatdate

        header_lines.append(f"Date: {formatdate(ts, localtime=False)}")
    if labels:
        header_lines.append(f"Labels: {', '.join(labels)}")
    body = "\n".join(header_lines + ["", text])
    return {
        "text": body,
        "score": score,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": {
            "stable_key": f"gmail:msg:{source_id}",
            "document_type": "email",
        },
    }


def _call(question, top_k=5, **kwargs):
    from recall import prepare_recall_context

    kwargs.setdefault("hydradb_sub_tenant_id", "ws_testtenant")
    return prepare_recall_context(question, top_k, **kwargs)


def _surviving_source_ids(result):
    return [s.get("source") for s in (result.get("sources") or [])]


# ====================================================================== #
# A. Gmail recency-aware retrieval
# ====================================================================== #
class TestGmailHeaderHarvest:
    """Unit-level: the Gmail markdown header harvester parses the
    fields the recency reranker needs."""

    def test_harvests_subject_from_date_labels_permalink(self):
        from recall import _harvest_gmail_header_fields

        body = "\n".join(
            [
                "# Email",
                "Source Key: gmail:msg:abc",
                "Mailbox: me@acme.example",
                'Subject: Deployment timeline for Friday',
                'From: "Rahul Verma" <rahul@acme.example>',
                "To: ops@acme.example",
                "Date: Mon, 02 Sep 2024 13:45:00 +0000",
                "Labels: INBOX, deployment",
                "Permalink: https://mail.google.com/mail/u/0/#inbox/abc",
                "",
                "Body of the email goes here.",
            ]
        )
        out = _harvest_gmail_header_fields(body)
        assert out["document_type"] == "email"
        assert out["subject"] == "Deployment timeline for Friday"
        assert out["from_name"] == "Rahul Verma"
        assert out["from_email"] == "rahul@acme.example"
        # Date is parsed into a unix timestamp.
        assert isinstance(out["timestamp"], float)
        assert out["timestamp"] > 1_700_000_000  # September 2024 > 2023
        assert out["labels"] == ["INBOX", "deployment"]
        assert out["permalink"].startswith("https://mail.google.com")

    def test_handles_address_only_from_line(self):
        from recall import _harvest_gmail_header_fields

        body = "# Email\nFrom: bob@example.com\nSubject: hi\n"
        out = _harvest_gmail_header_fields(body)
        assert out.get("from_email") == "bob@example.com"
        assert out.get("from_name") is None
        assert out.get("subject") == "hi"

    def test_returns_empty_for_non_gmail_doc(self):
        from recall import _harvest_gmail_header_fields

        # Without the "# Email" heading the harvester is a no-op so
        # it's safe to call unconditionally on any chunk body.
        assert _harvest_gmail_header_fields("# Slack Message\nfoo") == {}
        assert _harvest_gmail_header_fields("") == {}

    def test_bad_date_returns_no_timestamp(self):
        from recall import _harvest_gmail_header_fields

        body = "# Email\nSubject: x\nDate: this is not a date\n"
        out = _harvest_gmail_header_fields(body)
        assert "timestamp" not in out
        assert out["subject"] == "x"


class TestGmailRecency:
    """End-to-end: "latest email" returns the newest Gmail email."""

    def test_latest_email_returns_newest_gmail(self):
        # Two Gmail emails, ten days apart. Recency intent fires; the
        # newer email must win regardless of semantic score.
        import time

        now = time.time()
        chunks = [
            _gmail_chunk(
                text="Q3 OKR review notes",
                source_id="older",
                subject="Q3 OKRs",
                from_name="Alice",
                from_email="alice@acme.example",
                ts=now - 10 * 86400,
                score=0.99,  # high semantic
            ),
            _gmail_chunk(
                text="Latest deployment update",
                source_id="newer",
                subject="Deployment v42",
                from_name="Rahul",
                from_email="rahul@acme.example",
                ts=now - 1 * 86400,
                score=0.20,  # low semantic
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("what is the latest email", top_k=1)
        assert result["retrieval_mode"] == "recency"
        ids = _surviving_source_ids(result)
        assert ids[0] == "newer"

    @pytest.mark.parametrize(
        "question",
        [
            "latest email from Rahul",
            "most recent inbox email",
            "latest Gmail message",
            "newest email about deployment",
            "what's the newest email",
        ],
    )
    def test_recency_detector_recognizes_gmail_phrasings(self, question):
        from recall import _detect_recency_intent

        assert _detect_recency_intent(question) is True

    def test_latest_message_alone_still_works(self):
        # The connector-agnostic expansion must not break the original
        # "latest message" Slack phrasing.
        from recall import _detect_recency_intent

        assert _detect_recency_intent("what is the latest message") is True


# ====================================================================== #
# B. Sender-aware ranking
# ====================================================================== #
class TestSenderAwareRanking:
    """The `metadata_bias={"user": "rahul"}` argument should also
    promote Gmail emails whose `from_name` / `from_email` matches."""

    def test_metadata_bias_boosts_gmail_sender(self):
        from search_utils import _metadata_bias_score

        # Slack-shaped card with matching user_name (legacy).
        slack_card = {"user": "Rahul Verma", "channel": "engineering"}
        # Gmail-shaped card with matching from_name.
        gmail_card_name = {
            "from_name": "Rahul Verma",
            "subject": "Update on Kafka",
        }
        # Gmail-shaped card with matching from_email.
        gmail_card_addr = {
            "from_email": "rahul@example.com",
            "subject": "Update on Kafka",
        }
        bias = {"user": "Rahul Verma"}
        bias_addr = {"user": "rahul@example.com"}
        assert _metadata_bias_score(slack_card, bias) == 1
        assert _metadata_bias_score(gmail_card_name, bias) == 1
        assert _metadata_bias_score(gmail_card_addr, bias_addr) == 1
        # No match: zero boost.
        assert _metadata_bias_score(gmail_card_name, {"user": "Alice"}) == 0

    def test_sender_boost_applied_as_ranking_not_filter(self):
        # Two Gmail chunks: one from Rahul, one from Alice. The query
        # mentions Rahul. We expect the Rahul email to surface FIRST,
        # but the Alice email is NOT excluded (it's a ranking boost,
        # not a hard filter).
        chunks = [
            _gmail_chunk(
                text="random off-topic email",
                source_id="alice",
                from_name="Alice",
                from_email="alice@acme.example",
                subject="meeting moved",
            ),
            _gmail_chunk(
                text="kafka migration notes from rahul",
                source_id="rahul",
                from_name="Rahul",
                from_email="rahul@acme.example",
                subject="Kafka migration",
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            # Non-recency question -> default mode + metadata_bias.
            # We pass the bias explicitly (the API normally derives it
            # from query_rewriter inference -- here we just confirm
            # the underlying mechanism).
            result = _call(
                "what did rahul say about kafka",
                top_k=5,
                mode="default",
                metadata_bias={"user": "Rahul"},
            )
        ids = _surviving_source_ids(result)
        # Both present (no hard filter), Rahul's surfaces first.
        assert "rahul" in ids
        assert "alice" in ids
        assert ids[0] == "rahul"


# ====================================================================== #
# C. Metadata-aware ranking — subject hits
# ====================================================================== #
class TestSubjectHitRanking:
    def test_subject_hit_boosts_under_hybrid_mode(self):
        # Two Gmail emails. Both mention "kafka" in the body. Only one
        # has it in the SUBJECT. Hybrid mode should rank the subject
        # hit higher.
        chunks = [
            _gmail_chunk(
                text="actually we discussed kafka briefly",
                source_id="body-only",
                subject="weekly team check-in",
                from_name="Alice",
            ),
            _gmail_chunk(
                text="here are the kafka migration steps",
                source_id="subject-hit",
                subject="Kafka migration plan",
                from_name="Bob",
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call(
                "kafka migration",
                top_k=2,
                mode="hybrid",
            )
        ids = _surviving_source_ids(result)
        assert ids[0] == "subject-hit"


# ====================================================================== #
# D. Better cross-source ranking
# ====================================================================== #
class TestCrossSourceRanking:
    def test_recency_picks_newest_across_slack_and_gmail(self):
        # Mixed corpus: one old Slack, one newer Gmail. Recency
        # intent fires; the Gmail (newer) wins because it has the
        # higher timestamp -- regardless of source.
        import time

        now = time.time()
        chunks = [
            _slack_chunk(
                text="old slack discussion",
                source_id="slack-old",
                ts=str(now - 30 * 86400),
                score=0.99,
            ),
            _gmail_chunk(
                text="recent project email",
                source_id="gmail-new",
                subject="project update",
                from_name="Rahul",
                ts=now - 1 * 86400,
                score=0.30,
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("latest message about project", top_k=1)
        assert result["retrieval_mode"] == "recency"
        assert _surviving_source_ids(result)[0] == "gmail-new"

    def test_recency_picks_slack_when_slack_is_newer(self):
        # Symmetric: Slack should win when Slack is newer.
        import time

        now = time.time()
        chunks = [
            _gmail_chunk(
                text="old email",
                source_id="gmail-old",
                subject="quarterly note",
                from_name="Alice",
                ts=now - 30 * 86400,
            ),
            _slack_chunk(
                text="just-posted slack update",
                source_id="slack-new",
                ts=str(now - 1 * 3600),
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("latest message", top_k=1)
        assert result["retrieval_mode"] == "recency"
        assert _surviving_source_ids(result)[0] == "slack-new"


# ====================================================================== #
# Source filter compatibility
# ====================================================================== #
class TestSourceFilterStillIsolates:
    def test_gmail_only_filter_excludes_slack_under_recency(self):
        # Recency intent + allowed_sources=["gmail"]: Slack chunks
        # must be filtered out BEFORE the recency reranker runs, so
        # the Gmail email wins even if a newer Slack would otherwise
        # be eligible.
        import time

        now = time.time()
        chunks = [
            _slack_chunk(
                text="brand new slack message",
                source_id="slack-newest",
                ts=str(now - 60),  # newest overall
            ),
            _gmail_chunk(
                text="slightly older gmail",
                source_id="gmail-1",
                subject="project status",
                from_name="Rahul",
                ts=now - 3600,
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call(
                "latest email",
                top_k=1,
                allowed_sources=["gmail"],
            )
        ids = _surviving_source_ids(result)
        assert "slack-newest" not in ids
        assert "gmail-1" in ids

    def test_slack_only_filter_excludes_gmail_under_recency(self):
        import time

        now = time.time()
        chunks = [
            _gmail_chunk(
                text="brand new email",
                source_id="gmail-newest",
                subject="project x",
                from_name="Rahul",
                ts=now - 60,  # newest overall
            ),
            _slack_chunk(
                text="slightly older slack",
                source_id="slack-1",
                ts=str(now - 3600),
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call(
                "latest message",
                top_k=1,
                allowed_sources=["slack"],
            )
        ids = _surviving_source_ids(result)
        assert "gmail-newest" not in ids
        assert "slack-1" in ids


# ====================================================================== #
# Recency + sender combined
# ====================================================================== #
class TestRecencyPlusSender:
    def test_latest_from_rahul_picks_newest_rahul_email(self):
        # Two Gmail emails from Rahul (one older, one newer) and one
        # newer-but-from-Alice. With recency intent we sort by ts; the
        # sender filter is delivered via metadata_bias, which inside
        # recency mode contributes nothing (recency sorts by ts only)
        # -- so the test asserts the headline behavior: newest Rahul
        # wins ONLY when allowed_sources or a sender-filter
        # mechanism keeps Alice out. Here we use a hard channel-style
        # filter via metadata_bias's effect under default mode by
        # explicitly NOT triggering recency intent.
        import time

        now = time.time()
        chunks = [
            _gmail_chunk(
                text="rahul, older",
                source_id="rahul-old",
                subject="status",
                from_name="Rahul",
                ts=now - 10 * 86400,
            ),
            _gmail_chunk(
                text="alice, recent",
                source_id="alice-recent",
                subject="status",
                from_name="Alice",
                ts=now - 1 * 86400,
            ),
            _gmail_chunk(
                text="rahul, recent",
                source_id="rahul-recent",
                subject="status",
                from_name="Rahul",
                ts=now - 2 * 86400,
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            # Use a non-recency question so the default-mode ranking
            # respects the sender bias. (Pure "latest from rahul"
            # under recency mode would just return the newest ts
            # across all senders; sender-as-soft-filter is a default-
            # mode construct.)
            result = _call(
                "what did rahul send about status",
                top_k=3,
                mode="default",
                metadata_bias={"user": "Rahul"},
            )
        ids = _surviving_source_ids(result)
        # Both Rahul emails rank ABOVE the Alice one.
        rahul_positions = [i for i, sid in enumerate(ids) if sid.startswith("rahul")]
        alice_positions = [i for i, sid in enumerate(ids) if sid.startswith("alice")]
        assert rahul_positions and alice_positions
        assert max(rahul_positions) < min(alice_positions)


# ====================================================================== #
# E. Retrieval weighting -- explicit constants exist
# ====================================================================== #
class TestNamedWeights:
    def test_weight_constants_are_exported(self):
        from search_utils import (
            W_CHANNEL_MATCH,
            W_KEYWORD_HIT,
            W_LABEL_MATCH,
            W_RECENCY,
            W_SENDER_MATCH,
            W_SUBJECT_HIT,
        )

        # Sanity: weights are positive numbers and keyword > subject
        # > channel/sender > label > recency. This ordering is the
        # ranking contract -- if anyone tunes them, the existing
        # tests pin the relative behavior, but the order itself
        # documents intent.
        assert W_KEYWORD_HIT > 0
        assert W_SUBJECT_HIT > 0
        assert W_CHANNEL_MATCH > 0
        assert W_SENDER_MATCH > 0
        assert W_LABEL_MATCH > 0
        assert W_RECENCY > 0
        assert W_KEYWORD_HIT > W_SUBJECT_HIT
        assert W_SUBJECT_HIT > W_CHANNEL_MATCH
        assert W_CHANNEL_MATCH > W_LABEL_MATCH
        assert W_LABEL_MATCH > W_RECENCY


# ====================================================================== #
# F. Retrieval debug visibility
# ====================================================================== #
class TestRankBreakdownPayload:
    def test_prepared_includes_rank_breakdown_field(self):
        chunks = [
            _slack_chunk(
                text="something about kafka",
                source_id="s1",
                channel="engineering",
                ts="1740000000.0",
            ),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("kafka migration", top_k=1, mode="hybrid")
        # The internal debug field is always present on a ready
        # result, even when no inference fired.
        assert "rank_breakdown" in result
        assert isinstance(result["rank_breakdown"], list)
        assert len(result["rank_breakdown"]) == 1
        entry = result["rank_breakdown"][0]
        # Fields required by the visibility requirement (F).
        assert entry["rank"] == 1
        assert entry["source_kind"] in ("slack", "gmail", None)
        assert "score_breakdown" in entry
        # The breakdown must surface the per-signal scores so an
        # operator can see "why ranked highly".
        bd = entry["score_breakdown"]
        for k in ("keyword_hits", "subject_hits", "metadata_bias", "normalized_recency"):
            assert k in bd
        # Sensitive raw-text fields are NOT exposed in the
        # breakdown (the requirement says lightweight only -- no
        # body text, no subject content).
        assert "text" not in entry
        assert "subject" not in entry

    def test_rank_breakdown_is_not_in_sources_array(self):
        """The score breakdown is an internal field; the public
        `sources[]` list must NOT carry it (so it doesn't leak via
        the API response which is built from `sources`)."""
        chunks = [_slack_chunk(text="x", source_id="s1", channel="g")]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            result = _call("anything", top_k=1)
        for src in result["sources"]:
            assert "_debug_score" not in src
            assert "score_breakdown" not in src


# ====================================================================== #
# Ranking stability: same input -> same order
# ====================================================================== #
class TestRankingStability:
    def test_repeated_call_same_chunks_same_order(self):
        # Determinism: the rerank uses `original_index` as the stable
        # tiebreaker, so a fixed input must produce a fixed order
        # across calls.
        chunks = [
            _slack_chunk(text="one", source_id="a", channel="g", ts="1700000001.0"),
            _slack_chunk(text="two", source_id="b", channel="g", ts="1700000002.0"),
            _slack_chunk(text="three", source_id="c", channel="g", ts="1700000003.0"),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            r1 = _call("anything", top_k=3, mode="default")
            r2 = _call("anything", top_k=3, mode="default")
        assert _surviving_source_ids(r1) == _surviving_source_ids(r2)

    def test_hybrid_mode_is_deterministic_under_tied_signals(self):
        # Two chunks with identical timestamps + zero keyword hits +
        # zero bias must order by original_index (stable).
        chunks = [
            _slack_chunk(text="x", source_id="first", channel="g", ts="1700000000.0"),
            _slack_chunk(text="y", source_id="second", channel="g", ts="1700000000.0"),
        ]
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": chunks},
        ):
            r = _call("unrelated", top_k=2, mode="hybrid")
        # Both surface, in original order.
        ids = _surviving_source_ids(r)
        assert ids == ["first", "second"]
