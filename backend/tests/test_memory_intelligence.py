"""
Phase 16: memory intelligence tests.

Coverage areas:

  A. Entity resolution: merging across all four person identifier
     forms (Slack ID / @mention / display name / email), the
     no-false-merge guarantees, and project variant canonicalization.
  B. Relationship graph: recurrence x recency edge weighting,
     surfaced relations, cross-source flags, defensive failure modes.
  C. Memory importance: signal behavior, [0, 1] bounds, ranking
     stability, and the recall.py wiring (0.3 + 0.7 * importance with
     a preserved 0.5 fallback).
  D. Project intelligence: status, owner inference through merged
     identities, decisions / blockers / unresolved tasks, timeline.
  E. Conversation reconstruction: backward walk + citations.
  F. Intent classification + routing: every intent, subject
     extraction, and the clean fall-through contract.
  G. Routes: /api/intelligence/{graph,projects,conversation} and the
     /api/query routing hook.

Pure-core functions (build_*) take rows directly; workspace-scoped
wrappers are exercised by patching memory_intelligence._fetch_memories
or memory_intelligence.get_supabase, mirroring test_analystics.py.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

TEST_WS = "00000000-0000-0000-0000-00000000aaaa"


def _iso(days_ago: float = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


_ROW_SEQ = {"n": 0}


def _row(
    *,
    kind: str,
    content: str,
    source_stable_key: str,
    days_ago: float = 1,
    owner=None,
    entity_type=None,
    source_kind: str = "slack",
    rid=None,
):
    _ROW_SEQ["n"] += 1
    return {
        "id": rid or f"mem-{_ROW_SEQ['n']:04d}",
        "kind": kind,
        "content": content,
        "owner": owner,
        "entity_type": entity_type,
        "source_kind": source_kind,
        "source_stable_key": source_stable_key,
        "source_timestamp": _iso(days_ago),
        "created_at": _iso(days_ago),
        "metadata": {},
    }


def _person(content, sk, **kw):
    return _row(kind="entity", content=content, entity_type="person", source_stable_key=sk, **kw)


def _project(content, sk, **kw):
    return _row(kind="entity", content=content, entity_type="project", source_stable_key=sk, **kw)


# ====================================================================== #
# A. Entity resolution
# ====================================================================== #
class TestEntityResolution:
    def test_merges_all_four_identifier_forms(self):
        """Slack ID + @mention + display name + email collapse to ONE
        canonical person when each merge rule has its evidence."""
        from memory_intelligence import build_alias_map, resolve_alias

        rows = [
            # Slack ID person-entity, with adjacency evidence pairing
            # the ID to the display name inside a summary row.
            _person("U0123ABCD", "slack:msg:C1:1"),
            _row(
                kind="summary",
                content="Rahul Verma (<@U0123ABCD>) will lead the rollout.",
                source_stable_key="slack:msg:C1:1",
            ),
            # @mention form (extractor stores it without the @).
            _person("rahul.verma", "slack:msg:C1:2"),
            # display-name form.
            _person("Rahul Verma", "slack:msg:C1:3"),
            # email form arrives via the owner field on a Gmail task.
            _row(
                kind="action_item",
                content="send the Q3 deck",
                owner="rahul.verma@acme.com",
                source_stable_key="gmail:msg:abc",
                source_kind="gmail",
            ),
        ]
        alias_map = build_alias_map(rows)
        people = {k: v for k, v in alias_map["entities"].items() if v["entity_type"] == "person"}
        assert len(people) == 1, f"expected one merged person, got {list(people)}"
        ((canonical_id, person),) = people.items()
        assert person["canonical"] == "Rahul Verma"
        assert "U0123ABCD" in person["aliases"]
        assert "rahul.verma" in person["aliases"]
        assert "Rahul Verma" in person["aliases"]
        assert "rahul.verma@acme.com" in person["aliases"]
        # Every form resolves to the same canonical id.
        for form in ("<@U0123ABCD>", "U0123ABCD", "@rahul.verma", "Rahul Verma", "rahul.verma@acme.com", "RAHUL_VERMA"):
            assert resolve_alias(alias_map, form, entity_type="person") == canonical_id
        # Traceability: the ID merge is labelled with its rule.
        rules = {(r["alias"], r["rule"]) for r in person["merge_rules"]}
        assert ("U0123ABCD", "slack_id_adjacency") in rules

    def test_ambiguous_first_name_does_not_merge(self):
        """'Rahul' must NOT merge when two multi-token Rahuls exist."""
        from memory_intelligence import build_alias_map

        rows = [
            _person("Rahul Verma", "slack:msg:C1:1"),
            _person("Rahul Sharma", "slack:msg:C1:2"),
            _person("Rahul", "slack:msg:C1:3"),
        ]
        people = {v["canonical"] for v in build_alias_map(rows)["entities"].values() if v["entity_type"] == "person"}
        assert people == {"Rahul Verma", "Rahul Sharma", "Rahul"}

    def test_unambiguous_first_name_merges(self):
        from memory_intelligence import build_alias_map

        rows = [
            _person("Rahul Verma", "slack:msg:C1:1"),
            _person("rahul", "slack:msg:C1:2"),
        ]
        people = [v for v in build_alias_map(rows)["entities"].values() if v["entity_type"] == "person"]
        assert len(people) == 1
        assert people[0]["canonical"] == "Rahul Verma"
        assert {("rahul", "first_name_unambiguous")} <= {(r["alias"], r["rule"]) for r in people[0]["merge_rules"]}

    def test_distinct_emails_never_merge(self):
        from memory_intelligence import build_alias_map

        rows = [
            _row(kind="action_item", content="do thing one ok", owner="rahul.verma@acme.com", source_stable_key="g:1"),
            _row(kind="action_item", content="do thing two ok", owner="priya.shah@acme.com", source_stable_key="g:2"),
        ]
        people = [v for v in build_alias_map(rows)["entities"].values() if v["entity_type"] == "person"]
        assert len(people) == 2

    def test_slack_id_without_evidence_stays_separate(self):
        """Plain co-occurrence is never enough to merge an opaque ID."""
        from memory_intelligence import build_alias_map

        rows = [
            _person("U0999ZZZZ", "slack:msg:C1:1"),
            _person("Rahul Verma", "slack:msg:C1:1"),  # same source!
        ]
        people = [v for v in build_alias_map(rows)["entities"].values() if v["entity_type"] == "person"]
        assert len(people) == 2

    def test_project_case_punctuation_variants_merge(self):
        from memory_intelligence import build_alias_map, resolve_alias

        rows = [
            _project("Project Apollo", "s:1"),
            _project("project-apollo", "s:2"),
            _project("ProjectApollo", "s:3"),
            _project("Project Apollo", "s:4"),
        ]
        alias_map = build_alias_map(rows)
        projects = [v for v in alias_map["entities"].values() if v["entity_type"] == "project"]
        assert len(projects) == 1
        # Most-mentioned original casing wins the display form.
        assert projects[0]["canonical"] == "Project Apollo"
        assert projects[0]["mentions"] == 4
        assert resolve_alias(alias_map, "PROJECT_APOLLO", entity_type="project") == "project::project apollo"

    def test_resolve_alias_falls_back_for_unknowns(self):
        from memory_intelligence import build_alias_map, resolve_alias

        alias_map = build_alias_map([])
        assert resolve_alias(alias_map, "Mystery Person", entity_type="person") == "person::mystery person"
        assert alias_map["entities"] == {}


# ====================================================================== #
# B. Relationship graph
# ====================================================================== #
class TestRelationshipGraph:
    def _edge(self, graph, a_label, b_label):
        labels = {n["id"]: n["label"] for n in graph["nodes"]}
        for e in graph["edges"]:
            pair = {labels[e["source"]], labels[e["target"]]}
            if pair == {a_label, b_label}:
                return e
        return None

    def test_recurrence_and_recency_weighting(self):
        """2 recent co-occurrences must outweigh 1 stale one."""
        from memory_intelligence import build_relationship_graph

        rows = [
            # Apollo + Rahul in two fresh sources.
            _project("Apollo", "s:1", days_ago=1),
            _person("Rahul Verma", "s:1", days_ago=1),
            _project("Apollo", "s:2", days_ago=2),
            _person("Rahul Verma", "s:2", days_ago=2),
            # Beta + Rahul in one 90-day-old source.
            _project("Beta", "s:3", days_ago=90),
            _person("Rahul Verma", "s:3", days_ago=90),
        ]
        graph = build_relationship_graph(rows)
        apollo_edge = self._edge(graph, "Apollo", "Rahul Verma")
        beta_edge = self._edge(graph, "Beta", "Rahul Verma")
        assert apollo_edge and beta_edge
        assert apollo_edge["recurrence"] == 2
        assert beta_edge["recurrence"] == 1
        assert apollo_edge["weight"] > beta_edge["weight"]
        assert apollo_edge["relation"] == "person-project"
        assert set(apollo_edge["sources"]) == {"s:1", "s:2"}

    def test_cross_source_flag(self):
        from memory_intelligence import build_relationship_graph

        rows = [
            _project("Apollo", "slack:1", source_kind="slack"),
            _person("Rahul Verma", "slack:1", source_kind="slack"),
            _project("Apollo", "gmail:1", source_kind="gmail"),
            _person("Rahul Verma", "gmail:1", source_kind="gmail"),
            _project("Beta", "slack:2", source_kind="slack"),
            _person("Priya Shah", "slack:2", source_kind="slack"),
        ]
        graph = build_relationship_graph(rows)
        assert self._edge(graph, "Apollo", "Rahul Verma")["cross_source"] is True
        assert self._edge(graph, "Beta", "Priya Shah")["cross_source"] is False

    def test_surfaces_decision_and_action_item_relations(self):
        from memory_intelligence import build_relationship_graph

        rows = [
            _project("Apollo", "s:1"),
            _row(kind="decision", content="ship Apollo on Railway", source_stable_key="s:1"),
            _row(
                kind="entity",
                content="deploys",
                entity_type="channel",
                source_stable_key="s:1",
            ),
            # owner links the person even without a person-entity row.
            _row(
                kind="action_item",
                content="migrate the consumer",
                owner="Rahul Verma",
                source_stable_key="s:2",
            ),
        ]
        graph = build_relationship_graph(rows)
        relations = {e["relation"] for e in graph["edges"]}
        assert "decision-project" in relations
        assert "channel-project" in relations
        assert "action_item-person" in relations

    def test_alias_resolution_applied_inside_aggregation(self):
        """'project-apollo' and 'Project Apollo' co-occurrences merge
        into ONE node so the edge counts both sources."""
        from memory_intelligence import build_relationship_graph

        rows = [
            _project("Project Apollo", "s:1"),
            _person("Rahul Verma", "s:1"),
            _project("project-apollo", "s:2"),
            _person("rahul.verma@x.com", "s:2"),  # email-form person entity
        ]
        graph = build_relationship_graph(rows)
        project_nodes = [n for n in graph["nodes"] if n["type"] == "project"]
        person_nodes = [n for n in graph["nodes"] if n["type"] == "person"]
        assert len(project_nodes) == 1
        assert len(person_nodes) == 1
        edge = self._edge(graph, "Project Apollo", "Rahul Verma")
        assert edge["recurrence"] == 2

    def test_blank_workspace_short_circuits(self):
        from memory_intelligence import relationship_graph

        with patch("memory_intelligence.get_supabase") as mock_get:
            graph = relationship_graph(workspace_id="")
        mock_get.assert_not_called()
        assert graph["nodes"] == [] and graph["edges"] == []

    def test_supabase_failure_degrades_gracefully(self):
        from memory_intelligence import relationship_graph

        with patch("memory_intelligence.get_supabase", side_effect=RuntimeError("down")):
            graph = relationship_graph(workspace_id=TEST_WS)
        assert graph == {
            "nodes": [],
            "edges": [],
            "alias_map": {"entities": {}, "lookup": {}},
            "window_days": 90,
        }


# ====================================================================== #
# C. Memory importance + recall wiring
# ====================================================================== #
class TestMemoryImportance:
    def test_scores_bounded_and_stable(self):
        from memory_intelligence import compute_memory_importance

        rows = [
            _row(kind="decision", content="use Railway", source_stable_key="s:1", days_ago=3),
            _row(kind="action_item", content="ship the deck", owner="Rahul", source_stable_key="s:1", days_ago=3),
            _row(kind="summary", content="weekly digest", source_stable_key="s:2", days_ago=200),
        ]
        first = compute_memory_importance(rows)
        second = compute_memory_importance(rows)
        assert first == second, "identical input must produce identical scores (ranking stability)"
        assert set(first) == {r["id"] for r in rows}
        assert all(0.0 <= v <= 1.0 for v in first.values())

    def test_recurrence_reinforces(self):
        from memory_intelligence import compute_memory_importance

        repeated = [
            _row(kind="decision", content="use Railway", source_stable_key=f"s:{i}", days_ago=5, rid=f"rep-{i}")
            for i in range(4)
        ]
        single = [_row(kind="decision", content="use Render", source_stable_key="s:x", days_ago=5, rid="solo")]
        scores = compute_memory_importance(repeated + single)
        assert scores["rep-0"] > scores["solo"]

    def test_recency_decays_and_owner_boosts(self):
        from memory_intelligence import compute_memory_importance

        fresh = _row(kind="action_item", content="task aa", source_stable_key="s:1", days_ago=0, rid="fresh")
        stale = _row(kind="action_item", content="task bb", source_stable_key="s:2", days_ago=120, rid="stale")
        owned = _row(
            kind="action_item", content="task cc", owner="Rahul", source_stable_key="s:3", days_ago=0, rid="owned"
        )
        scores = compute_memory_importance([fresh, stale, owned])
        assert scores["fresh"] > scores["stale"]
        assert scores["owned"] > scores["fresh"]

    def test_empty_and_idless_rows_handled(self):
        from memory_intelligence import compute_memory_importance

        assert compute_memory_importance([]) == {}
        assert compute_memory_importance([{"content": "no id"}]) == {}


class TestRecallImportanceWiring:
    _MEMORY_ROW = {
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

    def _prepare(self):
        from recall import prepare_recall_context

        return prepare_recall_context(
            "what was decided about deployment",
            top_k=5,
            workspace_id=TEST_WS,
        )

    def test_score_is_importance_scaled_when_scoring_succeeds(self):
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ), patch(
            "memory_store.list_memories",
            return_value=[dict(self._MEMORY_ROW)],
        ), patch(
            "memory_intelligence.compute_memory_importance",
            return_value={"mem-1": 1.0},
        ):
            result = self._prepare()
        card = next(s for s in result["sources"] if s.get("memory_kind") == "decision")
        assert card["score"] == pytest.approx(0.3 + 0.7 * 1.0)

    def test_score_uses_formula_midrange(self):
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ), patch(
            "memory_store.list_memories",
            return_value=[dict(self._MEMORY_ROW)],
        ), patch(
            "memory_intelligence.compute_memory_importance",
            return_value={"mem-1": 0.5},
        ):
            result = self._prepare()
        card = next(s for s in result["sources"] if s.get("memory_kind") == "decision")
        assert card["score"] == pytest.approx(0.3 + 0.7 * 0.5)

    def test_fallback_to_half_when_scoring_fails(self):
        """The legacy 0.5 baseline survives any importance failure."""
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ), patch(
            "memory_store.list_memories",
            return_value=[dict(self._MEMORY_ROW)],
        ), patch(
            "memory_intelligence.compute_memory_importance",
            side_effect=RuntimeError("boom"),
        ):
            result = self._prepare()
        card = next(s for s in result["sources"] if s.get("memory_kind") == "decision")
        assert card["score"] == 0.5
        assert result["ready"] is True

    def test_fallback_when_importance_missing_or_out_of_range(self):
        with patch(
            "hydradb_client.HydraDBClient.full_recall",
            return_value={"chunks": []},
        ), patch(
            "memory_store.list_memories",
            return_value=[dict(self._MEMORY_ROW)],
        ), patch(
            "memory_intelligence.compute_memory_importance",
            return_value={"someone-else": 0.9, "mem-1": 7.0},
        ):
            result = self._prepare()
        card = next(s for s in result["sources"] if s.get("memory_kind") == "decision")
        assert card["score"] == 0.5


# ====================================================================== #
# D. Project intelligence
# ====================================================================== #
class TestProjectIntelligence:
    def _rows(self):
        return [
            # Apollo: fresh activity, owner evidence under two identity
            # forms, a decision, a blocker, an unresolved task.
            _project("Project Apollo", "s:1", days_ago=2),
            _person("Rahul Verma", "s:1", days_ago=2),
            _row(kind="decision", content="we will use Railway for Apollo", source_stable_key="s:1", days_ago=2),
            _project("project-apollo", "s:2", days_ago=1),
            _row(
                kind="action_item",
                content="blocked on the security review",
                owner="rahul.verma@acme.com",
                source_stable_key="s:2",
                days_ago=1,
                source_kind="gmail",
            ),
            _row(
                kind="action_item",
                content="finish the migration runbook",
                owner="Rahul Verma",
                source_stable_key="s:2",
                days_ago=1,
                source_kind="gmail",
            ),
            # Zeus: one stale mention only.
            _project("Zeus", "s:3", days_ago=60),
        ]

    def test_status_owner_and_linked_records(self):
        from memory_intelligence import build_project_intelligence

        projects = {p["project"]: p for p in build_project_intelligence(self._rows())}
        apollo = projects["Project Apollo"]
        zeus = projects["Zeus"]

        assert apollo["status"] == "active"
        assert zeus["status"] == "dormant"

        # Owner inference goes through entity resolution: the email
        # owner and the display-name entity merged into one person.
        assert apollo["owners"][0]["person"] == "Rahul Verma"
        assert len([o for o in apollo["owners"] if o["person"] == "Rahul Verma"]) == 1

        assert len(apollo["decisions"]) == 1
        assert apollo["decisions"][0]["source_stable_key"] == "s:1"
        assert [b["content"] for b in apollo["blockers"]] == ["blocked on the security review"]

        # Both s:2 action items postdate the s:1 decision -> unresolved.
        unresolved = {t["content"] for t in apollo["unresolved_tasks"]}
        assert unresolved == {"blocked on the security review", "finish the migration runbook"}

        # Timeline is chronological and carries citations.
        timeline_ts = [t["timestamp"] for t in apollo["timeline"]]
        assert timeline_ts == sorted(timeline_ts)
        assert all(t["source_stable_key"] for t in apollo["timeline"])
        assert apollo["first_seen"] <= apollo["last_seen"]
        assert set(apollo["sources"]) == {"s:1", "s:2"}

    def test_task_predating_latest_decision_is_resolved(self):
        from memory_intelligence import build_project_intelligence

        rows = [
            _project("Apollo", "s:1", days_ago=10),
            _row(kind="action_item", content="draft the proposal", source_stable_key="s:1", days_ago=10),
            _project("Apollo", "s:2", days_ago=2),
            _row(kind="decision", content="proposal approved, moving on", source_stable_key="s:2", days_ago=2),
        ]
        (apollo,) = build_project_intelligence(rows)
        assert apollo["unresolved_tasks"] == []

    def test_workspace_scoped_and_defensive(self):
        from memory_intelligence import project_intelligence

        with patch("memory_intelligence.get_supabase") as mock_get:
            assert project_intelligence(workspace_id="") == []
        mock_get.assert_not_called()
        with patch("memory_intelligence.get_supabase", side_effect=RuntimeError("down")):
            assert project_intelligence(workspace_id=TEST_WS) == []


# ====================================================================== #
# E. Conversation reconstruction
# ====================================================================== #
class TestConversationReconstruction:
    def _rows(self):
        return [
            # Older discussion in another source sharing the Apollo entity.
            _project("Apollo", "s:old", days_ago=10),
            _row(kind="summary", content="Apollo latency is hurting us", source_stable_key="s:old", days_ago=10),
            _row(kind="action_item", content="benchmark Railway and Render", source_stable_key="s:old", days_ago=10),
            # The decision source.
            _project("Apollo", "s:dec", days_ago=2),
            _row(
                kind="decision",
                content="we will use Railway for Apollo",
                source_stable_key="s:dec",
                days_ago=2,
                rid="dec-1",
            ),
            # A LATER memory that must be excluded from the backward walk.
            _project("Apollo", "s:new", days_ago=0.5),
            _row(kind="summary", content="Railway rollout going well", source_stable_key="s:new", days_ago=0.5),
            # An unrelated source with no shared entities.
            _project("Zeus", "s:other", days_ago=5),
            _row(kind="summary", content="Zeus kickoff notes here", source_stable_key="s:other", days_ago=5),
        ]

    def test_walks_backward_through_cooccurring_memories(self):
        from memory_intelligence import build_conversation_reconstruction

        recon = build_conversation_reconstruction(self._rows(), decision="use Railway")
        assert recon["decision"]["id"] == "dec-1"
        assert recon["decision"]["source_stable_key"] == "s:dec"
        contents = [s["content"] for s in recon["steps"]]
        assert "Apollo latency is hurting us" in contents
        assert "benchmark Railway and Render" in contents
        assert "Railway rollout going well" not in contents, "post-decision rows must not appear"
        assert "Zeus kickoff notes here" not in contents, "entity-disjoint sources must not appear"
        ts = [s["timestamp"] for s in recon["steps"]]
        assert ts == sorted(ts)
        assert "Apollo" in recon["entities"]

    def test_no_matching_decision_returns_empty_shape(self):
        from memory_intelligence import build_conversation_reconstruction

        recon = build_conversation_reconstruction(self._rows(), decision="adopt Kubernetes")
        assert recon == {"decision": None, "steps": [], "entities": []}

    def test_workspace_scoped_and_defensive(self):
        from memory_intelligence import reconstruct_conversation

        with patch("memory_intelligence.get_supabase") as mock_get:
            recon = reconstruct_conversation(workspace_id="", decision="anything")
        mock_get.assert_not_called()
        assert recon["decision"] is None
        with patch("memory_intelligence.get_supabase", side_effect=RuntimeError("down")):
            recon = reconstruct_conversation(workspace_id=TEST_WS, decision="use Railway")
        assert recon == {"decision": None, "steps": [], "entities": []}


# ====================================================================== #
# F. Intent classification + routing
# ====================================================================== #
class TestIntentClassifier:
    @pytest.mark.parametrize(
        "question,intent,subject",
        [
            ("Who owns Project Apollo?", "ownership", "Project Apollo"),
            ("who is responsible for the Apollo rollout", "ownership", "Apollo rollout"),
            ("Who's working on Apollo?", "ownership", "Apollo"),
            ("What is the status of Apollo?", "status_blocker", "Apollo"),
            ("what's blocking Project Apollo right now", "status_blocker", "Project Apollo"),
            ("any blockers for Apollo?", "status_blocker", "Apollo"),
            ("is Apollo blocked?", "status_blocker", "Apollo"),
            ("Why did we decide to use Railway?", "decision_history", "use Railway"),
            ("why did we go with Railway", "decision_history", "Railway"),
            ("how was the Railway migration decided?", "decision_history", "Railway migration"),
            ("timeline of Project Apollo", "timeline", "Project Apollo"),
            ("show me the chronology of Apollo please", "timeline", "Apollo please"),
        ],
    )
    def test_intents_detected_with_subject(self, question, intent, subject):
        from memory_intelligence import classify_intelligence_intent

        result = classify_intelligence_intent(question)
        assert result is not None, question
        assert result["intent"] == intent
        assert result["subject"] == subject

    @pytest.mark.parametrize(
        "question",
        [
            "what happened?",
            "what is the plan?",
            "what is the sprint status?",  # attributive form: no explicit subject
            "hello world",
            "summarize the engineering channel",
            "what did Alice say in #product?",
            "",
        ],
    )
    def test_non_intelligence_questions_fall_through(self, question):
        from memory_intelligence import classify_intelligence_intent

        assert classify_intelligence_intent(question) is None


class TestIntelligenceRouting:
    def _apollo_rows(self):
        return [
            _project("Project Apollo", "slack:msg:C1:100", days_ago=2),
            _person("Rahul Verma", "slack:msg:C1:100", days_ago=2),
            _row(
                kind="decision",
                content="we will use Railway for Apollo",
                source_stable_key="slack:msg:C1:100",
                days_ago=2,
            ),
            _row(
                kind="action_item",
                content="blocked on the security review",
                owner="Rahul Verma",
                source_stable_key="slack:msg:C1:101",
                days_ago=1,
            ),
            _project("Apollo", "slack:msg:C1:101", days_ago=1),
        ]

    def test_no_intent_means_no_io_and_none(self):
        from memory_intelligence import route_intelligence_query

        with patch("memory_intelligence._fetch_memories") as mock_fetch:
            result = route_intelligence_query(workspace_id=TEST_WS, question="what is the plan?")
        mock_fetch.assert_not_called()
        assert result is None

    def test_intent_with_no_rows_falls_through(self):
        from memory_intelligence import route_intelligence_query

        with patch("memory_intelligence._fetch_memories", return_value=[]):
            assert route_intelligence_query(workspace_id=TEST_WS, question="who owns Apollo?") is None

    def test_intent_with_unknown_subject_falls_through(self):
        from memory_intelligence import route_intelligence_query

        with patch("memory_intelligence._fetch_memories", return_value=self._apollo_rows()):
            result = route_intelligence_query(workspace_id=TEST_WS, question="who owns Zeus?")
        assert result is None

    def test_ownership_answer_with_citations(self):
        from memory_intelligence import route_intelligence_query

        with patch("memory_intelligence._fetch_memories", return_value=self._apollo_rows()):
            result = route_intelligence_query(workspace_id=TEST_WS, question="Who owns Project Apollo?")
        assert result is not None
        assert "Rahul Verma" in result["answer"]
        assert result["debug"]["intelligence_intent"] == "ownership"
        assert result["debug"]["routed"] == "memory_intelligence"
        stable_keys = {s["stable_key"] for s in result["sources"]}
        assert stable_keys <= {"slack:msg:C1:100", "slack:msg:C1:101"}
        assert stable_keys, "ownership answers must cite source_stable_keys"
        assert [s["index"] for s in result["sources"]] == list(range(1, len(result["sources"]) + 1))

    def test_status_blocker_answer(self):
        from memory_intelligence import route_intelligence_query

        with patch("memory_intelligence._fetch_memories", return_value=self._apollo_rows()):
            result = route_intelligence_query(workspace_id=TEST_WS, question="What is the status of Apollo?")
        assert result is not None
        assert "active" in result["answer"]
        assert "security review" in result["answer"]
        assert result["debug"]["intelligence_intent"] == "status_blocker"

    def test_timeline_answer(self):
        from memory_intelligence import route_intelligence_query

        with patch("memory_intelligence._fetch_memories", return_value=self._apollo_rows()):
            result = route_intelligence_query(workspace_id=TEST_WS, question="timeline of Project Apollo")
        assert result is not None
        assert result["debug"]["intelligence_intent"] == "timeline"
        assert "Timeline for Project Apollo" in result["answer"]

    def test_decision_history_answer(self):
        from memory_intelligence import route_intelligence_query

        with patch("memory_intelligence._fetch_memories", return_value=self._apollo_rows()):
            result = route_intelligence_query(
                workspace_id=TEST_WS,
                question="Why did we decide to use Railway?",
            )
        assert result is not None
        assert result["debug"]["intelligence_intent"] == "decision_history"
        assert "use Railway" in result["answer"]
        assert any(s["stable_key"] == "slack:msg:C1:100" for s in result["sources"])

    def test_routing_is_defensive(self):
        from memory_intelligence import route_intelligence_query

        with patch("memory_intelligence._fetch_memories", side_effect=RuntimeError("boom")):
            assert route_intelligence_query(workspace_id=TEST_WS, question="who owns Apollo?") is None
        assert route_intelligence_query(workspace_id="", question="who owns Apollo?") is None


# ====================================================================== #
# G. Routes
# ====================================================================== #
class TestIntelligenceRoutes:
    def test_graph_route(self, client, jwt_auth_headers):
        with patch(
            "memory_intelligence.relationship_graph",
            return_value={"nodes": [], "edges": [], "alias_map": {"entities": {}, "lookup": {}}, "window_days": 90},
        ) as mock_fn:
            r = client.get("/api/intelligence/graph?days=30", headers=jwt_auth_headers)
        assert r.status_code == 200
        assert r.json()["nodes"] == []
        assert mock_fn.call_args.kwargs["days"] == 30

    def test_projects_route(self, client, jwt_auth_headers):
        with patch(
            "memory_intelligence.project_intelligence",
            return_value=[{"project": "Apollo"}],
        ):
            r = client.get("/api/intelligence/projects", headers=jwt_auth_headers)
        assert r.status_code == 200
        assert r.json()["projects"] == [{"project": "Apollo"}]

    def test_conversation_route(self, client, jwt_auth_headers):
        with patch(
            "memory_intelligence.reconstruct_conversation",
            return_value={"decision": None, "steps": [], "entities": []},
        ) as mock_fn:
            r = client.get(
                "/api/intelligence/conversation?decision=use%20Railway",
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        assert mock_fn.call_args.kwargs["decision"] == "use Railway"
        assert r.json()["steps"] == []

    def test_conversation_route_requires_decision_param(self, client, jwt_auth_headers):
        r = client.get("/api/intelligence/conversation", headers=jwt_auth_headers)
        assert r.status_code == 422

    def test_routes_require_auth(self, client):
        for path in (
            "/api/intelligence/graph",
            "/api/intelligence/projects",
            "/api/intelligence/conversation?decision=x",
        ):
            assert client.get(path).status_code == 401, path


class TestQueryRouteIntegration:
    """The /api/query hook: routed answers come back as-is with the
    standard debug extras; non-matches reach the legacy pipeline
    untouched."""

    def test_routed_answer_short_circuits_pipeline(self, client, jwt_auth_headers):
        routed = {
            "answer": "Apollo is owned by Rahul Verma.",
            "sources": [{"index": 1, "stable_key": "slack:msg:C1:100"}],
            "debug": {"routed": "memory_intelligence", "intelligence_intent": "ownership", "subject": "Apollo"},
        }
        with patch(
            "memory_intelligence.route_intelligence_query",
            return_value=routed,
        ) as mock_route, patch("main.answer_question") as mock_answer:
            r = client.post(
                "/api/query",
                json={"question": "Who owns Project Apollo?"},
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Apollo is owned by Rahul Verma."
        assert body["debug"]["intelligence_intent"] == "ownership"
        assert body["debug"]["cache_hit"] is False
        mock_route.assert_called_once()
        mock_answer.assert_not_called()

    def test_none_falls_through_to_existing_pipeline(self, client, jwt_auth_headers):
        normal = {"answer": "The plan is X.", "sources": [], "debug": {"mode": "default"}}
        with patch(
            "memory_intelligence.route_intelligence_query",
            return_value=None,
        ), patch("main.answer_question", return_value=normal) as mock_answer:
            r = client.post(
                "/api/query",
                json={"question": "Who owns Project Apollo?"},
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        assert r.json()["answer"] == "The plan is X."
        mock_answer.assert_called_once()

    def test_router_exception_falls_through(self, client, jwt_auth_headers):
        normal = {"answer": "The plan is X.", "sources": [], "debug": {"mode": "default"}}
        with patch(
            "memory_intelligence.route_intelligence_query",
            side_effect=RuntimeError("boom"),
        ), patch("main.answer_question", return_value=normal):
            r = client.post(
                "/api/query",
                json={"question": "Who owns Project Apollo?"},
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        assert r.json()["answer"] == "The plan is X."

    def test_history_bypasses_routing(self, client, jwt_auth_headers):
        normal = {"answer": "Follow-up answer.", "sources": [], "debug": {"mode": "default"}}
        with patch("memory_intelligence.route_intelligence_query") as mock_route, patch(
            "main.answer_question", return_value=normal
        ):
            r = client.post(
                "/api/query",
                json={
                    "question": "Who owns Project Apollo?",
                    "conversation_history": [
                        {"role": "user", "content": "tell me about apollo"},
                        {"role": "assistant", "content": "Apollo is a project."},
                    ],
                },
                headers=jwt_auth_headers,
            )
        assert r.status_code == 200
        mock_route.assert_not_called()
