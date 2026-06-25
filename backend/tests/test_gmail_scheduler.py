"""
Tests for Phase 11 Gmail automation:

  A. Scheduled Gmail sync via scheduler._gmail_sweep / run_all_workspaces_once
  B. Incremental Gmail history sync via gmail_oauth.list_history_message_ids
  C. Token refresh persistence (gmail_oauth.run_workspace_gmail_ingest)
  D. Sync-status metadata (per-run summary fields)
  E. Scheduler/debug visibility
  F. Workspace + connection isolation under failure

We mock at the boundaries the production code talks to:
  - scheduler.list_active_workspaces_with_gmail
  - scheduler.run_workspace_gmail_ingest
  - hydradb_client.HydraDBClient.upload_knowledge
  - gmail_oauth._authed_request           (for history/profile shape tests)
  - supabase_client.get_gmail_ingestion_state_map / upsert_gmail_ingestion_state
  - supabase_client.update_gmail_connection_tokens
"""

from unittest.mock import MagicMock, patch

import pytest

WS1 = "00000000-0000-0000-0000-000000000001"
WS2 = "00000000-0000-0000-0000-000000000002"
CONN_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
CONN_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


# ====================================================================== #
# A. Scheduled Gmail sync
# ====================================================================== #
class TestScheduledGmailSweep:
    def test_no_gmail_connections_returns_zero(self):
        from scheduler import _gmail_sweep

        with patch(
            "scheduler.list_active_workspaces_with_gmail",
            return_value=[],
        ):
            summary = _gmail_sweep()
        assert summary["connections_total"] == 0
        assert summary["connections_run"] == 0
        assert summary["connections_failed"] == 0

    def test_routes_each_connection_to_its_own_sub_tenant(self):
        """Two workspaces, two connections (one each). The runner
        must be called with the right (workspace, connection, sub_tenant,
        labels) tuple for each."""
        from scheduler import _gmail_sweep

        rows = [
            {
                "workspace_id": WS1,
                "hydradb_sub_tenant_id": "ws_aaaaaaaaaaaa",
                "connection": {"id": CONN_A, "email": "a@example.com", "refresh_token": "rt-a"},
                "selected_label_ids": ["INBOX"],
            },
            {
                "workspace_id": WS2,
                "hydradb_sub_tenant_id": "ws_bbbbbbbbbbbb",
                "connection": {"id": CONN_B, "email": "b@example.com", "refresh_token": "rt-b"},
                "selected_label_ids": ["INBOX", "Label_1"],
            },
        ]
        with patch(
            "scheduler.list_active_workspaces_with_gmail",
            return_value=rows,
        ), patch(
            "scheduler.run_workspace_gmail_ingest",
            return_value={
                "labels_processed": 1,
                "labels_skipped": 0,
                "labels_failed": 0,
                "messages_uploaded": 0,
                "messages_failed": 0,
                "incremental_label_count": 0,
                "full_label_count": 1,
                "invalidations": 0,
                "refresh_token_used": False,
                "sync_mode_requested": "auto",
                "duration_ms": 50,
            },
        ) as mock_run:
            summary = _gmail_sweep()

        assert summary["connections_total"] == 2
        assert summary["connections_run"] == 2
        assert mock_run.call_count == 2
        first_kw = mock_run.call_args_list[0].kwargs
        second_kw = mock_run.call_args_list[1].kwargs
        # Verify each call carried the correct sub_tenant_id + connection.
        assert first_kw["workspace_id"] == WS1
        assert first_kw["hydradb_sub_tenant_id"] == "ws_aaaaaaaaaaaa"
        assert first_kw["connection"]["id"] == CONN_A
        assert first_kw["label_ids"] == ["INBOX"]
        assert first_kw["sync_mode"] == "auto"
        assert second_kw["workspace_id"] == WS2
        assert second_kw["hydradb_sub_tenant_id"] == "ws_bbbbbbbbbbbb"
        assert second_kw["connection"]["id"] == CONN_B
        assert second_kw["label_ids"] == ["INBOX", "Label_1"]

    def test_one_connection_failure_does_not_block_others(self):
        from scheduler import _gmail_sweep

        rows = [
            {
                "workspace_id": WS1,
                "hydradb_sub_tenant_id": "ws_aaaaaaaaaaaa",
                "connection": {"id": CONN_A, "refresh_token": "x"},
                "selected_label_ids": ["INBOX"],
            },
            {
                "workspace_id": WS2,
                "hydradb_sub_tenant_id": "ws_bbbbbbbbbbbb",
                "connection": {"id": CONN_B, "refresh_token": "x"},
                "selected_label_ids": ["INBOX"],
            },
        ]

        def side_effect(**kw):
            if kw["connection"]["id"] == CONN_A:
                raise RuntimeError("transient API outage")
            return {
                "labels_processed": 1,
                "labels_skipped": 0,
                "labels_failed": 0,
                "messages_uploaded": 5,
                "messages_failed": 0,
                "incremental_label_count": 0,
                "full_label_count": 1,
                "invalidations": 0,
                "refresh_token_used": False,
                "sync_mode_requested": "auto",
                "duration_ms": 50,
            }

        with patch(
            "scheduler.list_active_workspaces_with_gmail",
            return_value=rows,
        ), patch(
            "scheduler.run_workspace_gmail_ingest",
            side_effect=side_effect,
        ):
            summary = _gmail_sweep()
        # Bad CONN_A counted as failed; CONN_B still ran cleanly.
        assert summary["connections_failed"] == 1
        assert summary["connections_run"] == 1
        assert summary["messages_uploaded"] == 5

    def test_combined_run_all_namespaces_gmail_summary(self):
        """run_all_workspaces_once must return Slack keys at top
        level (unchanged from Phase 4) and Gmail under "gmail"."""
        from scheduler import run_all_workspaces_once

        with patch(
            "scheduler.list_active_workspaces_with_slack",
            return_value=[],
        ), patch(
            "scheduler.list_active_workspaces_with_gmail",
            return_value=[],
        ):
            summary = run_all_workspaces_once()
        assert "workspaces_total" in summary
        assert summary["gmail"]["connections_total"] == 0


# ====================================================================== #
# B. Incremental Gmail history sync
# ====================================================================== #
class TestListHistoryMessageIds:
    """Test the history.list helper directly via _authed_request mock."""

    def test_returns_message_ids_and_advances_high_water_mark(self):
        from gmail_oauth import list_history_message_ids

        fake_response = {
            "historyId": "987654",
            "history": [
                {"messagesAdded": [{"message": {"id": "m-new-1"}}]},
                {
                    "messagesAdded": [
                        {"message": {"id": "m-new-2"}},
                        {"message": {"id": "m-new-3"}},
                    ]
                },
            ],
        }
        with patch(
            "gmail_oauth._authed_request",
            return_value=(fake_response, {"access_token": "tok"}),
        ):
            out = list_history_message_ids(
                {"access_token": "tok", "refresh_token": "x"},
                start_history_id="123456",
                label_id="INBOX",
                max_results=10,
            )
        assert out["invalidated"] is False
        assert out["message_ids"] == ["m-new-1", "m-new-2", "m-new-3"]
        assert out["next_history_id"] == "987654"

    def test_history_404_returns_invalidated_sentinel(self):
        from gmail_oauth import GmailApiError, list_history_message_ids

        with patch(
            "gmail_oauth._authed_request",
            side_effect=GmailApiError("Gmail HTTP 404"),
        ):
            out = list_history_message_ids(
                {"access_token": "tok", "refresh_token": "x"},
                start_history_id="too-old",
                label_id="INBOX",
            )
        assert out["invalidated"] is True
        assert out["message_ids"] == []
        assert out["next_history_id"] is None

    def test_non_404_error_propagates(self):
        from gmail_oauth import GmailApiError, list_history_message_ids

        with patch(
            "gmail_oauth._authed_request",
            side_effect=GmailApiError("Gmail HTTP 500"),
        ):
            with pytest.raises(GmailApiError):
                list_history_message_ids(
                    {"access_token": "tok", "refresh_token": "x"},
                    start_history_id="x",
                    label_id="INBOX",
                )

    def test_blank_start_history_id_short_circuits(self):
        from gmail_oauth import list_history_message_ids

        # No HTTP call should happen.
        with patch("gmail_oauth._authed_request") as mock:
            out = list_history_message_ids(
                {"access_token": "tok", "refresh_token": "x"},
                start_history_id="",
                label_id="INBOX",
            )
        mock.assert_not_called()
        assert out["message_ids"] == []
        assert out["invalidated"] is False


class TestIncrementalSyncMode:
    """End-to-end: run_workspace_gmail_ingest in auto / incremental / full."""

    @staticmethod
    def _patches(*, state_map=None, history_result=None, listing_ids=None, profile=None):
        """Bundle of patches that lets each test focus on one signal."""
        ctx = []
        ctx.append(
            patch(
                "supabase_client.get_gmail_ingestion_state_map",
                return_value=state_map or {},
            )
        )
        ctx.append(
            patch(
                "supabase_client.upsert_gmail_ingestion_state",
                return_value=True,
            )
        )
        if history_result is not None:
            ctx.append(
                patch(
                    "gmail_oauth.list_history_message_ids",
                    return_value=history_result,
                )
            )
        if listing_ids is not None:
            ctx.append(
                patch(
                    "gmail_oauth.list_thread_ids_for_label",
                    return_value=listing_ids,
                )
            )
        if profile is not None:
            ctx.append(
                patch(
                    "gmail_oauth.get_mailbox_profile",
                    return_value=profile,
                )
            )
        # No-op thread fetch (we don't care about message bodies in these
        # tests; the listings above return no threads).
        ctx.append(
            patch(
                "gmail_oauth.fetch_thread",
                return_value=None,
            )
        )
        return ctx

    def _run(self, **kwargs):
        from gmail_oauth import run_workspace_gmail_ingest

        defaults = {
            "workspace_id": WS1,
            "connection": {
                "id": CONN_A,
                "email": "a@example.com",
                "refresh_token": "rt",
                "access_token": "initial-token",
            },
            "label_ids": ["INBOX"],
            "hydradb_sub_tenant_id": "ws_xxxxxxxxxxxx",
            "max_messages": 10,
        }
        defaults.update(kwargs)
        return run_workspace_gmail_ingest(**defaults)

    def test_first_sync_no_watermark_uses_full(self):
        """No last_history_id -> full listing path. Reports
        full_label_count=1 and seeds a fresh watermark via profile."""
        patches = self._patches(
            state_map={},  # no watermark
            listing_ids=[],  # empty -> nothing fetched
            profile={"historyId": "100"},
        )
        for p in patches:
            p.start()
        try:
            summary = self._run()
        finally:
            for p in patches:
                p.stop()
        assert summary["full_label_count"] == 1
        assert summary["incremental_label_count"] == 0
        # per_label record captures the fresh history id.
        assert summary["per_label"][0]["mode"] == "full"
        assert summary["per_label"][0]["new_history_id"] == "100"

    def test_subsequent_sync_with_watermark_uses_incremental(self):
        """With a stored last_history_id, auto mode goes incremental."""
        patches = self._patches(
            state_map={"INBOX": {"last_history_id": "999", "last_synced_at": "2024-09-01T00:00:00Z"}},
            history_result={
                "message_ids": [],
                "next_history_id": "1000",
                "invalidated": False,
            },
        )
        for p in patches:
            p.start()
        try:
            summary = self._run()
        finally:
            for p in patches:
                p.stop()
        assert summary["incremental_label_count"] == 1
        assert summary["full_label_count"] == 0
        assert summary["per_label"][0]["mode"] == "incremental"
        assert summary["per_label"][0]["new_history_id"] == "1000"

    def test_history_invalidated_falls_back_to_full(self):
        """When Gmail returns 404 (watermark too old), the runner
        falls back to full listing for that label and records the
        invalidation."""
        patches = self._patches(
            state_map={"INBOX": {"last_history_id": "ancient", "last_synced_at": "old"}},
            history_result={
                "message_ids": [],
                "next_history_id": None,
                "invalidated": True,
            },
            listing_ids=[],
            profile={"historyId": "new-200"},
        )
        for p in patches:
            p.start()
        try:
            summary = self._run()
        finally:
            for p in patches:
                p.stop()
        assert summary["invalidations"] == 1
        # Fell back -> counted under full, not incremental.
        assert summary["full_label_count"] == 1
        assert summary["incremental_label_count"] == 0
        # Watermark advanced to the fresh value from profile.
        assert summary["per_label"][0]["mode"] == "full"
        assert summary["per_label"][0]["invalidated"] is True
        assert summary["per_label"][0]["new_history_id"] == "new-200"

    def test_full_mode_skips_history_call(self):
        """sync_mode="full" must not call list_history_message_ids
        even if a watermark exists."""
        patches = self._patches(
            state_map={"INBOX": {"last_history_id": "999", "last_synced_at": "x"}},
            listing_ids=[],
            profile={"historyId": "111"},
        )
        # Add an extra patch to assert the history helper is NEVER called.
        history_mock = patch("gmail_oauth.list_history_message_ids")
        history_obj = history_mock.start()
        for p in patches:
            p.start()
        try:
            summary = self._run(sync_mode="full")
        finally:
            for p in patches:
                p.stop()
            history_mock.stop()
        assert summary["full_label_count"] == 1
        assert summary["incremental_label_count"] == 0
        history_obj.assert_not_called()


# ====================================================================== #
# C. Token refresh persistence
# ====================================================================== #
class TestTokenRefreshPersistence:
    def test_refreshed_access_token_persisted_once(self):
        """When _authed_request stamps `_token_refreshed=True` mid-run,
        the runner calls update_gmail_connection_tokens exactly once
        at end-of-run with the new access_token."""
        from gmail_oauth import run_workspace_gmail_ingest

        # The connection dict is mutable -- patch the listing call to
        # mutate it (mirrors what _authed_request does in production).
        def list_with_refresh(connection, label_id, *, max_results):
            connection["access_token"] = "REFRESHED-TOKEN"
            connection["_token_refreshed"] = True
            return []

        with patch(
            "gmail_oauth.list_thread_ids_for_label",
            side_effect=list_with_refresh,
        ), patch(
            "supabase_client.get_gmail_ingestion_state_map",
            return_value={},
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ), patch(
            "gmail_oauth.get_mailbox_profile",
            return_value={"historyId": "1"},
        ), patch(
            "supabase_client.update_gmail_connection_tokens",
            return_value=True,
        ) as mock_persist:
            summary = run_workspace_gmail_ingest(
                workspace_id=WS1,
                connection={
                    "id": CONN_A,
                    "refresh_token": "rt",
                    "access_token": "OLD-TOKEN",
                },
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_xxxxxxxxxxxx",
                max_messages=10,
            )

        # Persisted exactly once with the new token + workspace check.
        mock_persist.assert_called_once()
        kw = mock_persist.call_args.kwargs
        assert kw["workspace_id"] == WS1
        assert kw["connection_id"] == CONN_A
        assert kw["access_token"] == "REFRESHED-TOKEN"
        # Summary reflects it.
        assert summary["refresh_token_used"] is True

    def test_no_refresh_means_no_persist_call(self):
        """If the access token never changes, we don't make a
        useless write."""
        from gmail_oauth import run_workspace_gmail_ingest

        with patch(
            "gmail_oauth.list_thread_ids_for_label",
            return_value=[],
        ), patch(
            "supabase_client.get_gmail_ingestion_state_map",
            return_value={},
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ), patch(
            "gmail_oauth.get_mailbox_profile",
            return_value={"historyId": "1"},
        ), patch(
            "supabase_client.update_gmail_connection_tokens",
            return_value=True,
        ) as mock_persist:
            summary = run_workspace_gmail_ingest(
                workspace_id=WS1,
                connection={
                    "id": CONN_A,
                    "refresh_token": "rt",
                    "access_token": "OLD-TOKEN",
                },
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_xxxxxxxxxxxx",
                max_messages=10,
            )
        mock_persist.assert_not_called()
        assert summary["refresh_token_used"] is False


# ====================================================================== #
# D. Sync-status metadata visibility
# ====================================================================== #
class TestSyncStatusMetadata:
    def test_summary_includes_phase11_observability_fields(self):
        from gmail_oauth import run_workspace_gmail_ingest

        with patch(
            "supabase_client.get_gmail_ingestion_state_map",
            return_value={},
        ), patch(
            "gmail_oauth.list_thread_ids_for_label",
            return_value=[],
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ), patch(
            "gmail_oauth.get_mailbox_profile",
            return_value={"historyId": "1"},
        ):
            summary = run_workspace_gmail_ingest(
                workspace_id=WS1,
                connection={
                    "id": CONN_A,
                    "refresh_token": "rt",
                    "access_token": "tok",
                },
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_xxxxxxxxxxxx",
                max_messages=10,
                sync_mode="auto",
            )
        # Every required observability field is present.
        for key in (
            "sync_mode_requested",
            "sync_started_at",
            "sync_finished_at",
            "duration_ms",
            "refresh_token_used",
            "incremental_label_count",
            "full_label_count",
            "invalidations",
            "per_label",
        ):
            assert key in summary, f"missing summary key: {key}"
        # Per-label record carries enough to debug.
        assert summary["per_label"][0]["label_id"] == "INBOX"
        assert summary["per_label"][0]["mode"] in ("full", "incremental")

    def test_connection_sync_summary_projection(self):
        """The lightweight projection get_gmail_connection_sync_summary
        gives the frontend "last synced X ago" without leaking PII."""
        from supabase_client import get_gmail_connection_sync_summary

        # Build a fake state map response by patching the underlying
        # helper.
        state = {
            "INBOX": {"last_history_id": "1", "last_synced_at": "2024-09-10T00:00:00Z"},
            "Label_1": {"last_history_id": "2", "last_synced_at": "2024-09-15T00:00:00Z"},
            "Label_old": {"last_history_id": None, "last_synced_at": None},
        }
        with patch(
            "supabase_client.get_gmail_ingestion_state_map",
            return_value=state,
        ):
            summary = get_gmail_connection_sync_summary(
                workspace_id=WS1,
                gmail_connection_id=CONN_A,
            )
        # Most-recent timestamp wins.
        assert summary["last_synced_at"] == "2024-09-15T00:00:00Z"
        # Only labels with a timestamp count.
        assert summary["labels_synced"] == 2


# ====================================================================== #
# E. Workspace isolation under failure (extra defense-in-depth)
# ====================================================================== #
class TestSchedulerIsolation:
    def test_workspace_a_failure_does_not_affect_workspace_b(self):
        """Sister test to test_one_connection_failure_does_not_block_others
        but specifically about WORKSPACE isolation -- two separate
        workspaces, A blows up, B must still run cleanly with B's
        own sub-tenant."""
        from scheduler import _gmail_sweep

        rows = [
            {
                "workspace_id": WS1,
                "hydradb_sub_tenant_id": "ws_aaaaaaaaaaaa",
                "connection": {"id": CONN_A, "refresh_token": "x"},
                "selected_label_ids": ["INBOX"],
            },
            {
                "workspace_id": WS2,
                "hydradb_sub_tenant_id": "ws_bbbbbbbbbbbb",
                "connection": {"id": CONN_B, "refresh_token": "x"},
                "selected_label_ids": ["INBOX"],
            },
        ]
        captured_kwargs = []

        def side_effect(**kw):
            captured_kwargs.append(kw)
            if kw["workspace_id"] == WS1:
                raise Exception("workspace A is on fire")
            return {
                "labels_processed": 1,
                "labels_skipped": 0,
                "labels_failed": 0,
                "messages_uploaded": 3,
                "messages_failed": 0,
                "incremental_label_count": 1,
                "full_label_count": 0,
                "invalidations": 0,
                "refresh_token_used": False,
                "sync_mode_requested": "auto",
                "duration_ms": 30,
            }

        with patch(
            "scheduler.list_active_workspaces_with_gmail",
            return_value=rows,
        ), patch(
            "scheduler.run_workspace_gmail_ingest",
            side_effect=side_effect,
        ):
            summary = _gmail_sweep()
        # Both attempted.
        assert len(captured_kwargs) == 2
        # Each got its OWN sub-tenant.
        ws_to_st = {kw["workspace_id"]: kw["hydradb_sub_tenant_id"] for kw in captured_kwargs}
        assert ws_to_st[WS1] == "ws_aaaaaaaaaaaa"
        assert ws_to_st[WS2] == "ws_bbbbbbbbbbbb"
        # WS1 failed, WS2 ran cleanly.
        assert summary["connections_failed"] == 1
        assert summary["connections_run"] == 1
        assert summary["messages_uploaded"] == 3
