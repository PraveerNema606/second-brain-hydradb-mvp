"""
Phase 4 -- HydraDB workspace isolation.

These tests verify the new contract: every recall + every ingest run
must be routed to the workspace's own HydraDB sub_tenant_id, never
the global env default. The tests don't hit real HydraDB or real
Supabase -- they patch at module boundaries to inspect the kwargs
passed to the lower-level helpers.
"""

from unittest.mock import MagicMock, patch

import pytest

TEST_WS_ID = "00000000-0000-0000-0000-00000000aaaa"


# =====================================================================
# supabase_client helpers
# =====================================================================
class TestDeriveSubTenantId:
    """The Python-side derive function must match the SQL function."""

    def test_format_matches_sql_helper(self):
        from supabase_client import _derived_sub_tenant_id

        # Same input as smoke-test in phase4_hydradb_workspace_isolation.sql
        result = _derived_sub_tenant_id("11111111-aaaa-bbbb-cccc-000000000001")
        assert result == "ws_11111111aaaa"

    def test_blank_workspace_id_returns_empty_string(self):
        from supabase_client import _derived_sub_tenant_id

        assert _derived_sub_tenant_id("") == ""

    def test_strips_dashes_and_takes_first_12_hex(self):
        from supabase_client import _derived_sub_tenant_id

        # All dashes stripped, only the first 12 hex chars used.
        result = _derived_sub_tenant_id("abcdef01-2345-6789-0000-000000000000")
        assert result == "ws_abcdef012345"


class TestEnsureWorkspaceSubTenant:
    def test_returns_existing_value_when_present(self):
        from supabase_client import ensure_workspace_sub_tenant

        with patch(
            "supabase_client.get_workspace_sub_tenant_id",
            return_value="ws_existing1234",
        ):
            result = ensure_workspace_sub_tenant(workspace_id=TEST_WS_ID)
        assert result == "ws_existing1234"

    def test_writes_derived_value_when_missing(self):
        from supabase_client import ensure_workspace_sub_tenant

        mock_client = MagicMock()
        with patch(
            "supabase_client.get_workspace_sub_tenant_id",
            return_value=None,
        ), patch(
            "supabase_client.get_supabase",
            return_value=mock_client,
        ):
            result = ensure_workspace_sub_tenant(workspace_id=TEST_WS_ID)
        # Derived from TEST_WS_ID (first 12 hex chars after dashes
        # stripped: 00000000-0000 -> 000000000000).
        assert result == "ws_000000000000"
        # Confirm we issued an UPDATE on workspaces.
        mock_client.table.assert_called_with("workspaces")

    def test_returns_none_when_db_write_fails(self):
        from supabase_client import ensure_workspace_sub_tenant

        mock_client = MagicMock()
        # Make the .update().eq().execute() chain raise.
        mock_client.table.return_value.update.return_value.eq.return_value.execute.side_effect = RuntimeError("db down")
        with patch(
            "supabase_client.get_workspace_sub_tenant_id",
            return_value=None,
        ), patch(
            "supabase_client.get_supabase",
            return_value=mock_client,
        ):
            result = ensure_workspace_sub_tenant(workspace_id=TEST_WS_ID)
        assert result is None


# =====================================================================
# recall.py: HydraDBClient must be constructed with the passed sub-tenant
# =====================================================================
class TestRecallUsesWorkspaceSubTenant:
    def test_prepare_recall_context_constructs_client_with_sub_tenant(self):
        """
        When hydradb_sub_tenant_id is provided, HydraDBClient must be
        instantiated with that sub_tenant_id (not the env default).
        """
        from recall import prepare_recall_context

        # Patch HydraDBClient to a MagicMock so we can inspect ctor args.
        with patch("recall.HydraDBClient") as mock_client_cls:
            instance = mock_client_cls.return_value
            instance.full_recall.return_value = {"results": []}
            # We don't care what comes out -- just that the client was
            # built with the right kwargs.
            prepare_recall_context(
                question="anything",
                top_k=3,
                hydradb_sub_tenant_id="ws_abcdef012345",
            )
        mock_client_cls.assert_called_once_with(
            sub_tenant_id="ws_abcdef012345",
        )

    def test_prepare_recall_context_refuses_when_sub_tenant_omitted(self):
        # Fail closed: omitting hydradb_sub_tenant_id must NOT fall back
        # to an env / hard-coded HydraDB tenant.
        from errors import WorkspaceTenantError
        from recall import prepare_recall_context

        with patch("recall.HydraDBClient") as mock_client_cls:
            with pytest.raises(WorkspaceTenantError):
                prepare_recall_context(question="anything", top_k=3)
        mock_client_cls.assert_not_called()

    def test_answer_question_forwards_sub_tenant(self):
        """answer_question must thread its hydradb_sub_tenant_id arg
        into prepare_recall_context."""
        from recall import answer_question

        captured = {}

        def fake_prepare(**kwargs):
            captured.update(kwargs)
            return {"ready": False, "fallback_debug": {"reason": "ok"}}

        with patch("recall.prepare_recall_context", side_effect=fake_prepare):
            answer_question(
                question="q",
                top_k=5,
                hydradb_sub_tenant_id="ws_workspace_xx",
            )
        assert captured["hydradb_sub_tenant_id"] == "ws_workspace_xx"


# =====================================================================
# /api/query routes thread the workspace sub-tenant end-to-end
# =====================================================================
class TestApiQueryUsesWorkspaceSubTenant:
    def test_api_query_calls_answer_question_with_sub_tenant(
        self,
        client,
        jwt_auth_headers,
    ):
        """End-to-end: POST /api/query must call answer_question with
        the workspace's resolved sub_tenant_id, NOT the env default."""
        # Bypass query rewrite + caching to keep the test focused.
        with patch(
            "main.require_workspace_sub_tenant",
            return_value="ws_resolved_abc",
        ) as mock_ensure, patch(
            "main.answer_question",
            return_value={"answer": "x", "sources": [], "debug": {}},
        ) as mock_answer:
            r = client.post(
                "/api/query",
                headers=jwt_auth_headers,
                json={"question": "anything", "top_k": 3, "mode": "default"},
            )
        assert r.status_code == 200
        mock_ensure.assert_called_once_with(workspace_id=TEST_WS_ID)
        _, kwargs = mock_answer.call_args
        assert kwargs["hydradb_sub_tenant_id"] == "ws_resolved_abc"

    def test_api_query_stream_calls_prepare_recall_with_sub_tenant(
        self,
        client,
        jwt_auth_headers,
    ):
        """Same contract for the SSE streaming route."""
        # The streaming route exits early when prepare_recall_context
        # returns ready=False, so we can stub it minimally.
        with patch(
            "main.require_workspace_sub_tenant",
            return_value="ws_resolved_abc",
        ), patch(
            "main.prepare_recall_context",
            return_value={"ready": False, "fallback_debug": {"reason": "x"}},
        ) as mock_prepare:
            r = client.post(
                "/api/query/stream",
                headers=jwt_auth_headers,
                json={"question": "anything", "top_k": 3, "mode": "default"},
            )
        assert r.status_code == 200
        _, kwargs = mock_prepare.call_args
        assert kwargs["hydradb_sub_tenant_id"] == "ws_resolved_abc"


# =====================================================================
# slack_oauth.run_workspace_ingest must route uploads correctly
# =====================================================================
class TestRunWorkspaceIngestUsesSubTenant:
    def test_constructs_hydradb_client_with_passed_sub_tenant(self):
        """run_workspace_ingest must build HydraDBClient(sub_tenant_id=X)
        when X is provided."""
        from slack_oauth import run_workspace_ingest

        # Patch every primitive the runner reaches for; we only care
        # about the HydraDBClient construction kwargs.
        with patch("slack_oauth.WebClient"), patch("hydradb_client.HydraDBClient") as mock_hydra_cls, patch(
            "ingestion.slack_client.SlackClientWrapper"
        ), patch(
            "ingestion.ingest_slack.process_channel",
            return_value={"files": [], "newest_ts_seen": None, "channel_id": "C1", "skipped_count": 0},
        ), patch(
            "ingestion.ingest_slack.upload_in_batches", return_value={"successes": 0, "failures": 0}
        ), patch(
            "ingestion.ingestion_state.IngestionState"
        ):
            run_workspace_ingest(
                workspace_id=TEST_WS_ID,
                bot_token="xoxb-test",
                channel_ids=["C1"],
                hydradb_sub_tenant_id="ws_workspace_xx",
            )
        mock_hydra_cls.assert_called_once_with(
            sub_tenant_id="ws_workspace_xx",
        )

    def test_refuses_when_sub_tenant_missing(self):
        """When the caller doesn't pass a sub-tenant, the runner must
        refuse — never fall back to an env / hard-coded HydraDB tenant."""
        from slack_oauth import run_workspace_ingest

        with patch("hydradb_client.HydraDBClient") as mock_hydra_cls:
            result = run_workspace_ingest(
                workspace_id=TEST_WS_ID,
                bot_token="xoxb-test",
                channel_ids=["C1"],
                # hydradb_sub_tenant_id deliberately omitted
            )
        mock_hydra_cls.assert_not_called()
        assert result.get("error") == "missing_sub_tenant"
        assert result.get("failures", 0) >= 1

    def test_short_circuits_when_no_channels(self):
        """No channel_ids -> no HydraDB client construction at all."""
        from slack_oauth import run_workspace_ingest

        with patch("hydradb_client.HydraDBClient") as mock_hydra_cls:
            result = run_workspace_ingest(
                workspace_id=TEST_WS_ID,
                bot_token="xoxb-test",
                channel_ids=[],
                hydradb_sub_tenant_id="ws_workspace_xx",
            )
        assert result["channels_processed"] == 0
        mock_hydra_cls.assert_not_called()


# =====================================================================
# Cross-workspace isolation: two workspaces -> two distinct sub-tenants
# =====================================================================
class TestCrossWorkspaceIsolation:
    def test_two_workspaces_route_to_their_own_sub_tenants(self):
        """
        Simulate two consecutive API calls from two different
        workspaces. Each call must end up using a DIFFERENT
        hydradb_sub_tenant_id. This is the headline isolation
        guarantee Phase 4 ships.
        """
        from supabase_client import ensure_workspace_sub_tenant

        ws_a = "aaaaaaaa-1111-1111-1111-000000000001"
        ws_b = "bbbbbbbb-2222-2222-2222-000000000002"

        # Patch get_workspace_sub_tenant_id to simulate a fresh DB
        # where neither workspace has a sub-tenant yet -- both must
        # lazy-create distinct values.
        mock_client = MagicMock()
        with patch(
            "supabase_client.get_workspace_sub_tenant_id",
            return_value=None,
        ), patch(
            "supabase_client.get_supabase",
            return_value=mock_client,
        ):
            tenant_a = ensure_workspace_sub_tenant(workspace_id=ws_a)
            tenant_b = ensure_workspace_sub_tenant(workspace_id=ws_b)

        assert tenant_a == "ws_aaaaaaaa1111"
        assert tenant_b == "ws_bbbbbbbb2222"
        assert tenant_a != tenant_b


# =====================================================================
# list_active_workspaces_with_slack shape
# =====================================================================
class TestListActiveWorkspacesWithSlack:
    def test_returns_only_workspaces_with_install_and_sub_tenant(self):
        from supabase_client import list_active_workspaces_with_slack

        mock_client = MagicMock()

        def fake_table(name):
            tbl = MagicMock()
            if name == "workspaces":
                tbl.select.return_value.eq.return_value.execute.return_value.data = [
                    {"id": "ws-1", "hydradb_sub_tenant_id": "ws_aaaaaa1", "hydradb_status": "active"},
                    {"id": "ws-2", "hydradb_sub_tenant_id": "ws_bbbbbb2", "hydradb_status": "active"},
                    {"id": "ws-3", "hydradb_sub_tenant_id": "ws_cccccc3", "hydradb_status": "active"},
                ]
            elif name == "slack_installations":
                tbl.select.return_value.execute.return_value.data = [
                    {"workspace_id": "ws-1", "bot_token": "xoxb-1"},
                    # ws-2 has no installation -> filtered out.
                    {"workspace_id": "ws-3", "bot_token": "xoxb-3"},
                ]
            elif name == "slack_channels":
                tbl.select.return_value.eq.return_value.execute.return_value.data = [
                    {"workspace_id": "ws-1", "slack_channel_id": "C1"},
                    {"workspace_id": "ws-1", "slack_channel_id": "C2"},
                    {"workspace_id": "ws-3", "slack_channel_id": "C9"},
                ]
            return tbl

        mock_client.table.side_effect = fake_table

        with patch("supabase_client.get_supabase", return_value=mock_client):
            result = list_active_workspaces_with_slack()

        ids = [row["workspace_id"] for row in result]
        # ws-2 is dropped (no installation); ws-1 + ws-3 remain.
        assert sorted(ids) == ["ws-1", "ws-3"]
        # Each row carries the full payload the scheduler needs.
        by_id = {row["workspace_id"]: row for row in result}
        assert by_id["ws-1"]["hydradb_sub_tenant_id"] == "ws_aaaaaa1"
        assert by_id["ws-1"]["bot_token"] == "xoxb-1"
        assert by_id["ws-1"]["channel_ids"] == ["C1", "C2"]
        assert by_id["ws-3"]["channel_ids"] == ["C9"]

    def test_swallows_supabase_errors(self):
        from supabase_client import list_active_workspaces_with_slack

        mock_client = MagicMock()
        mock_client.table.side_effect = RuntimeError("supabase down")
        with patch("supabase_client.get_supabase", return_value=mock_client):
            result = list_active_workspaces_with_slack()
        assert result == []
