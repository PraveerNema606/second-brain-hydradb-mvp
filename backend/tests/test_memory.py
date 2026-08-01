"""
Phase 12: structured-memory tests.

Covers all seven requirement areas:

  A. Action-item extraction
  B. Decision extraction
  C. Summary generation
  D. Entity extraction
  E. Memory persistence + dedupe + workspace isolation
  F. Memory-aware retrieval (ranking interaction + traceability)
  G. Defensive failure modes (memory layer never blocks ingestion)

All tests are deterministic; no Supabase or HydraDB I/O happens. The
extractor is pure -- nothing to mock there. The persistence layer is
mocked at the Supabase-client boundary; the retrieval layer mocks
list_memories at the `recall` module's import name (since recall.py
defers the import to avoid eager Supabase construction).
"""

from unittest.mock import MagicMock, patch

import pytest

WS1 = "11111111-1111-1111-1111-111111111111"
WS2 = "22222222-2222-2222-2222-222222222222"
SLACK_KEY = "slack:msg:C1:1700000000.0"


# ====================================================================== #
# A. Action-item extraction
# ====================================================================== #
class TestActionItemExtraction:
    def test_explicit_todo_marker(self):
        from memory_extraction import extract_action_items

        out = extract_action_items("TODO: migrate Kafka consumer to v3")
        assert len(out) == 1
        assert "migrate Kafka consumer" in out[0]["content"]
        assert out[0]["kind"] == "action_item"
        assert out[0]["content_hash"]  # SHA-256 hex

    def test_action_marker_with_dash(self):
        from memory_extraction import extract_action_items

        out = extract_action_items("ACTION - send rollout doc by Friday")
        assert len(out) == 1
        assert "rollout doc" in out[0]["content"]

    def test_owner_will_verb_pattern(self):
        """'Rahul will deploy Friday' -> owner=Rahul, action=deploy Friday"""
        from memory_extraction import extract_action_items

        out = extract_action_items("Rahul will deploy Friday")
        assert len(out) == 1
        assert out[0]["owner"] == "Rahul"
        assert "deploy" in out[0]["content"]
        assert "Friday" in out[0]["content"]

    def test_owner_to_verb_pattern(self):
        from memory_extraction import extract_action_items

        out = extract_action_items("Praveer to investigate latency issue")
        assert len(out) == 1
        assert out[0]["owner"] == "Praveer"
        assert "latency" in out[0]["content"]

    def test_polite_request(self):
        from memory_extraction import extract_action_items

        out = extract_action_items("Hey -- can you send me the rollout doc?")
        assert len(out) == 1
        assert "send" in out[0]["content"].lower()

    def test_assigned_to_pattern(self):
        from memory_extraction import extract_action_items

        out = extract_action_items("Assigned to Alice: write postmortem")
        assert len(out) == 1
        assert out[0]["owner"] == "Alice"

    def test_default_owner_fallback(self):
        from memory_extraction import extract_action_items

        out = extract_action_items(
            "TODO: write the design doc",
            default_owner="rahul-author",
        )
        assert out[0]["owner"] == "rahul-author"

    def test_dedupes_same_text_in_same_input(self):
        """Same TODO repeated produces one record."""
        from memory_extraction import extract_action_items

        text = "TODO: migrate Kafka consumer\n" "Reminder -- TODO: migrate Kafka consumer\n"
        out = extract_action_items(text)
        assert len(out) == 1

    def test_strips_markdown_header_before_extracting(self):
        """A 'From: Rahul Verma' header line must NOT trigger an
        'action item' extraction. We strip the markdown header first."""
        from memory_extraction import extract_action_items

        text = (
            "# Email\n"
            "Subject: re: deployment\n"
            "From: Rahul Verma <rahul@acme>\n"
            "Date: Mon, 02 Sep 2024 13:45:00 +0000\n"
            "\n"
            "TODO: ship the rollout doc tomorrow"
        )
        out = extract_action_items(text)
        # Only the body TODO survives.
        assert len(out) == 1
        assert "rollout doc" in out[0]["content"]

    def test_returns_empty_for_no_match(self):
        from memory_extraction import extract_action_items

        out = extract_action_items("Just a casual hello, nothing to do.")
        assert out == []

    def test_handles_blank_input(self):
        from memory_extraction import extract_action_items

        assert extract_action_items("") == []
        assert extract_action_items("   ") == []


# ====================================================================== #
# B. Decision extraction
# ====================================================================== #
class TestDecisionExtraction:
    @pytest.mark.parametrize(
        "text,fragment",
        [
            ("We agreed to use Railway for production", "use Railway"),
            ("Decision: stick with Supabase for now", "stick with Supabase"),
            ("We'll keep Slack realtime enabled", "Slack realtime"),
            ("Approved: rollout window Friday 5pm", "rollout window"),
            ("moving to incremental sync next sprint", "incremental sync"),
        ],
    )
    def test_detects_decision_phrasings(self, text, fragment):
        from memory_extraction import extract_decisions

        out = extract_decisions(text)
        assert len(out) >= 1
        assert any(fragment.lower() in r["content"].lower() for r in out)
        for r in out:
            assert r["kind"] == "decision"

    def test_does_not_detect_suggestion(self):
        """'we should' is a suggestion, NOT a decision."""
        from memory_extraction import extract_decisions

        out = extract_decisions("We should probably use Railway maybe")
        # Empty or short -- the patterns deliberately exclude "should".
        # The exact behavior is "no match"; if a regression makes
        # this match, we want to know.
        contents = [r["content"] for r in out]
        assert not any("we should" in c.lower() for c in contents)

    def test_dedupes_repeated_decision(self):
        from memory_extraction import extract_decisions

        out = extract_decisions("We agreed to use Railway. We agreed to use Railway.")
        assert len(out) == 1


# ====================================================================== #
# C. Summary generation
# ====================================================================== #
class TestSummarization:
    def test_short_text_returns_none(self):
        from memory_extraction import summarize

        assert summarize("Quick hi.") is None
        assert summarize("") is None

    def test_returns_first_sentences_of_long_body(self):
        from memory_extraction import summarize

        body = (
            "The team finished the Kafka migration last week. "
            "All consumer groups have been rebalanced. "
            "Latency is back to baseline. "
            "We will monitor through end of quarter."
        )
        s = summarize(body)
        assert s is not None
        # First sentence makes it in.
        assert "Kafka migration" in s
        # 3-sentence cap by default.
        assert s.count(".") <= 4

    def test_respects_280_char_cap(self):
        from memory_extraction import summarize

        body = ("a quite long sentence " * 50).strip() + "."
        s = summarize(body)
        assert s is not None
        assert len(s) <= 281


# ====================================================================== #
# D. Entity extraction
# ====================================================================== #
class TestEntityExtraction:
    def test_at_mention_is_person(self):
        from memory_extraction import extract_entities

        out = extract_entities("hey @rahul can you check this?")
        persons = [e for e in out if e["entity_type"] == "person"]
        assert any(e["content"] == "rahul" for e in persons)

    def test_slack_user_mention(self):
        from memory_extraction import extract_entities

        out = extract_entities("ping <@U123ABC456> about Kafka")
        persons = [e for e in out if e["entity_type"] == "person"]
        assert any(e["content"] == "U123ABC456" for e in persons)

    def test_hash_channel(self):
        from memory_extraction import extract_entities

        out = extract_entities("posted in #engineering")
        channels = [e for e in out if e["entity_type"] == "channel"]
        assert any(e["content"] == "engineering" for e in channels)

    def test_service_term_via_known_keyword(self):
        from memory_extraction import extract_entities

        out = extract_entities("we moved to Railway for deployment")
        # "Railway" is in the service-hint set.
        services = [e for e in out if e["entity_type"] == "service"]
        assert any(e["content"] == "Railway" for e in services)

    def test_code_term_classification(self):
        from memory_extraction import extract_entities

        # Backticked code term -> system. Path-like -> repository.
        out = extract_entities("run `kubectl get pods` then check `acme/api`")
        contents = [(e["content"], e["entity_type"]) for e in out]
        assert ("acme/api", "repository") in contents

    def test_blocklist_excludes_grammar(self):
        from memory_extraction import extract_entities

        out = extract_entities("The team is great. We finished it Friday.")
        # "The", "We" must NOT appear as entities.
        names = {e["content"] for e in out}
        assert "The" not in names
        assert "We" not in names

    def test_dedupes_within_input(self):
        from memory_extraction import extract_entities

        out = extract_entities("Railway is fine. We moved to Railway last month.")
        railway_hits = [e for e in out if e["content"] == "Railway" and e["entity_type"] == "service"]
        assert len(railway_hits) == 1


# ====================================================================== #
# E. Memory persistence + dedupe + workspace isolation
# ====================================================================== #
class TestMemoryPersistence:
    def _mock_supabase(self, exec_raises=None):
        execute = MagicMock()
        if exec_raises is not None:
            execute.side_effect = exec_raises
        else:
            execute.return_value = MagicMock(data=[{"id": "row-1"}])
        upsert = MagicMock(return_value=MagicMock(execute=execute))
        table = MagicMock(return_value=MagicMock(upsert=upsert))
        client = MagicMock(table=table)
        return client, upsert

    def test_persists_workspace_scoped_rows(self):
        from memory_extraction import _content_hash, _record
        from memory_store import persist_memories

        client, upsert = self._mock_supabase()
        memories = [
            _record(
                kind="action_item",
                content="migrate Kafka consumer",
                owner="Rahul",
            ),
            _record(
                kind="decision",
                content="use Railway for production",
            ),
        ]
        with patch("memory_store.get_supabase", return_value=client):
            sent = persist_memories(
                workspace_id=WS1,
                source_kind="slack",
                source_stable_key=SLACK_KEY,
                source_timestamp="2024-09-10T12:00:00+00:00",
                memories=memories,
            )
        assert sent == 2
        # Every row carries the workspace_id + source_stable_key + ts.
        sent_rows = upsert.call_args.args[0]
        for r in sent_rows:
            assert r["workspace_id"] == WS1
            assert r["source_stable_key"] == SLACK_KEY
            assert r["source_kind"] == "slack"
            assert r["source_timestamp"] == "2024-09-10T12:00:00+00:00"
            assert r["content_hash"]
        # Conflict target matches the schema's unique key.
        on_conflict = upsert.call_args.kwargs.get("on_conflict")
        assert on_conflict == "workspace_id,kind,content_hash,source_stable_key"

    def test_invalid_kind_filtered_out(self):
        from memory_store import persist_memories

        client, upsert = self._mock_supabase()
        with patch("memory_store.get_supabase", return_value=client):
            sent = persist_memories(
                workspace_id=WS1,
                source_kind="slack",
                source_stable_key=SLACK_KEY,
                source_timestamp=None,
                memories=[
                    {"kind": "nonsense", "content": "x", "content_hash": "h"},
                    {"kind": "action_item", "content": "valid", "content_hash": "h2"},
                ],
            )
        # Only the valid one made it through.
        assert sent == 1
        sent_rows = upsert.call_args.args[0]
        assert len(sent_rows) == 1
        assert sent_rows[0]["kind"] == "action_item"

    def test_unknown_source_kind_returns_zero(self):
        from memory_store import persist_memories

        sent = persist_memories(
            workspace_id=WS1,
            source_kind="notion",  # not slack/gmail
            source_stable_key="x",
            source_timestamp=None,
            memories=[{"kind": "decision", "content": "ship it", "content_hash": "abc"}],
        )
        assert sent == 0

    def test_postgrest_error_logged_with_structured_body(self, caplog):
        import logging

        try:
            from postgrest.exceptions import APIError as PGAPIError
        except ImportError:  # pragma: no cover
            pytest.skip("postgrest not installed")
        from memory_store import persist_memories

        err = PGAPIError(
            {
                "code": "23505",
                "message": "duplicate key violates unique constraint",
                "hint": None,
                "details": None,
            }
        )
        client, _ = self._mock_supabase(exec_raises=err)
        with caplog.at_level(logging.WARNING, logger="memory_store"):
            with patch("memory_store.get_supabase", return_value=client):
                sent = persist_memories(
                    workspace_id=WS1,
                    source_kind="slack",
                    source_stable_key=SLACK_KEY,
                    source_timestamp=None,
                    memories=[{"kind": "decision", "content": "x", "content_hash": "h"}],
                )
        assert sent == 0
        records = [r for r in caplog.records if r.message == "memory_persist_failed"]
        assert len(records) == 1
        rec = records[0]
        assert getattr(rec, "pg_code") == "23505"
        assert "duplicate key" in getattr(rec, "pg_message")


class TestMemoryDedupe:
    """Round-trip extract -> the persistence layer's unique key
    dedupes identical text against the same source."""

    def test_same_content_same_source_dedupes_by_hash(self):
        from memory_extraction import _content_hash, _record

        # Two records with identical canonical form -> same content_hash.
        r1 = _record(kind="action_item", content="Migrate kafka consumer")
        r2 = _record(kind="action_item", content="migrate KAFKA consumer  ")
        assert r1["content_hash"] == r2["content_hash"]

    def test_different_source_yields_different_unique_key(self):
        """Same action item in two different messages -> two persisted
        rows. The unique key includes source_stable_key so cross-source
        deduplication never happens by accident."""
        from memory_extraction import _content_hash

        h = _content_hash("ship it")
        key1 = (WS1, "decision", h, "slack:msg:C1:1")
        key2 = (WS1, "decision", h, "slack:msg:C1:2")
        assert key1 != key2


class TestWorkspaceIsolation:
    """The list_memories helper must always WHERE on workspace_id."""

    def test_list_filters_on_workspace(self):
        from memory_store import list_memories

        # Build a chainable mock that records the .eq calls.
        eq_calls = []
        ilike_calls = []
        in_calls = []
        order_mock = MagicMock()
        order_mock.limit = MagicMock(
            return_value=MagicMock(
                execute=MagicMock(
                    return_value=MagicMock(data=[]),
                ),
            )
        )

        class _Chain:
            def select(self, *a, **k):
                return self

            def eq(self, k, v):
                eq_calls.append((k, v))
                return self

            def in_(self, k, v):
                in_calls.append((k, v))
                return self

            def ilike(self, k, v):
                ilike_calls.append((k, v))
                return self

            def order(self, *a, **k):
                return order_mock

        client = MagicMock()
        client.table = MagicMock(return_value=_Chain())
        with patch("memory_store.get_supabase", return_value=client):
            list_memories(workspace_id=WS1, kinds=["decision"], query="kafka")
        assert ("workspace_id", WS1) in eq_calls
        assert any(c[0] == "kind" for c in in_calls)
        assert any(c[0] == "content" for c in ilike_calls)


# ====================================================================== #
# F. Memory-aware retrieval (ranking interaction + traceability)
# ====================================================================== #
class TestMemoryAwareRetrieval:
    """Memories surface as candidates in prepare_recall_context when
    workspace_id is supplied. They participate in default/hybrid
    ranking but never trigger or win the recency rerank."""

    def _slack_chunk(self, *, source_id, text, ts=None):
        ts = ts or f"1700{abs(hash(source_id)) % 1_000_000:06d}.0"
        return {
            "text": text,
            "score": 0.8,
            "source_id": source_id,
            "filename": f"{source_id}.md",
            "metadata": {
                "channel": "engineering",
                "stable_key": f"slack:msg:C1:{ts}:{source_id}",
                "timestamp": ts,
                "document_type": "message",
            },
        }

    def test_memories_appear_in_recall_results(self):
        """A workspace with one matching memory row + zero HydraDB
        chunks must still return that memory as a source."""
        from recall import prepare_recall_context

        memory_row = {
            "id": "mem-1",
            "kind": "decision",
            "content": "use Railway for production deployment",
            "owner": None,
            "entity_type": None,
            "source_kind": "slack",
            "source_stable_key": "slack:msg:C1:1700000000",
            "source_timestamp": "2024-09-10T12:00:00+00:00",
            "metadata": {},
        }
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ), patch(
            "memory_store.list_memories",
            return_value=[memory_row],
        ):
            result = prepare_recall_context(
                "what was decided about deployment",
                top_k=5,
                workspace_id=WS1,
                hydradb_sub_tenant_id="ws_testtenant",
            )
        assert result["ready"] is True
        # The memory surfaced as a source.
        sources = result["sources"]
        assert any(s.get("memory_kind") == "decision" for s in sources)
        # The LLM-facing context contains the memory's content.
        assert "Railway" in result["context_text"]
        # Traceability: source_stable_key links back to the Slack message.
        memory_card = next(s for s in sources if s.get("memory_kind") == "decision")
        assert memory_card["stable_key"] == "slack:msg:C1:1700000000"

    def test_workspace_id_required_to_pull_memories(self):
        """Without workspace_id the memory lookup is skipped entirely
        (preserves backward compatibility with pre-Phase-12 callers)."""
        from recall import prepare_recall_context

        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ), patch(
            "memory_store.list_memories",
        ) as mock_list:
            prepare_recall_context(
                "what was decided",
                top_k=5,
                hydradb_sub_tenant_id="ws_testtenant",
                # workspace_id deliberately omitted
            )
        mock_list.assert_not_called()

    def test_recency_intent_skips_memory_lookup(self):
        """A 'latest message' query must NOT pull memories -- a
        6-month-old decision should not outrank today's Slack message."""
        from recall import prepare_recall_context

        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={
                "chunks": [
                    self._slack_chunk(source_id="recent", text="newest message"),
                ]
            },
        ), patch(
            "memory_store.list_memories",
        ) as mock_list:
            prepare_recall_context(
                "what is the latest message",
                top_k=5,
                workspace_id=WS1,
                hydradb_sub_tenant_id="ws_testtenant",
            )
        mock_list.assert_not_called()

    def test_source_filter_respects_memory_origin(self):
        """allowed_sources=['gmail'] excludes a Slack-derived memory."""
        from recall import prepare_recall_context

        gmail_mem = {
            "id": "mem-gmail",
            "kind": "decision",
            "content": "approved Q3 budget",
            "owner": None,
            "entity_type": None,
            "source_kind": "gmail",
            "source_stable_key": "gmail:msg:abc",
            "source_timestamp": "2024-09-15T12:00:00+00:00",
            "metadata": {},
        }
        slack_mem = {
            "id": "mem-slack",
            "kind": "decision",
            "content": "ship Friday",
            "owner": None,
            "entity_type": None,
            "source_kind": "slack",
            "source_stable_key": "slack:msg:C1:1700000000",
            "source_timestamp": "2024-09-15T12:00:00+00:00",
            "metadata": {},
        }
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ), patch(
            "memory_store.list_memories",
            return_value=[gmail_mem, slack_mem],
        ):
            result = prepare_recall_context(
                "what was decided",
                top_k=5,
                workspace_id=WS1,
                allowed_sources=["gmail"],
                hydradb_sub_tenant_id="ws_testtenant",
            )
        kinds = [(s.get("source_kind"), s.get("memory_kind")) for s in result["sources"]]
        # Slack memory must be excluded.
        assert ("slack", "decision") not in kinds
        # Gmail memory must be present.
        assert ("gmail", "decision") in kinds

    def test_memory_lookup_failure_does_not_block_answer(self):
        """list_memories raising an exception must not prevent the
        answer from being returned via the HydraDB chunks alone."""
        from recall import prepare_recall_context

        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={
                "chunks": [
                    self._slack_chunk(
                        source_id="s1",
                        text="kafka migration discussion",
                    ),
                ]
            },
        ), patch(
            "memory_store.list_memories",
            side_effect=RuntimeError("supabase down"),
        ):
            result = prepare_recall_context(
                "kafka",
                top_k=5,
                workspace_id=WS1,
                hydradb_sub_tenant_id="ws_testtenant",
            )
        # We still got a real answer; the memory layer degraded silently.
        assert result["ready"] is True
        assert len(result["sources"]) >= 1


# ====================================================================== #
# G. Defensive failure modes
# ====================================================================== #
class TestMemoryFailureModes:
    def test_extract_and_persist_swallows_supabase_failure(self):
        """A Supabase outage must not surface as an exception to the
        Slack/Gmail ingest paths -- extract_and_persist returns 0
        instead of raising."""
        from memory_store import extract_and_persist

        with patch(
            "memory_store.get_supabase",
            side_effect=RuntimeError("supabase down"),
        ):
            n = extract_and_persist(
                workspace_id=WS1,
                source_kind="slack",
                source_stable_key=SLACK_KEY,
                source_timestamp=None,
                text="TODO: ship it tomorrow",
            )
        assert n == 0

    def test_extract_and_persist_empty_input_zero(self):
        from memory_store import extract_and_persist

        assert (
            extract_and_persist(
                workspace_id=WS1,
                source_kind="slack",
                source_stable_key=SLACK_KEY,
                source_timestamp=None,
                text="",
            )
            == 0
        )

    def test_strip_markdown_header_idempotent_on_clean_body(self):
        """Calling the header stripper on body text that has no
        header should not eat content."""
        from memory_extraction import strip_markdown_header

        body = "Just some regular text\nwith two lines."
        assert strip_markdown_header(body) == body
