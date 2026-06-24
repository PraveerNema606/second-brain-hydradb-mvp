"""
Gmail deleted-email synchronization (audit item 2).

Covers:
  - list_history_message_ids surfaces messageDeleted ids in a new
    `deleted_message_ids` key, and nets an add+delete-in-window id out
    of `message_ids`.
  - run_workspace_gmail_ingest, on the incremental path, removes the
    deleted messages from HydraDB (HydraDBClient.delete_knowledge) and
    clears their derived memories (memory_store.delete_memories_by_source),
    counting them in summary["messages_deleted"] without blocking the
    rest of the run.

All external calls are mocked.
"""

from unittest.mock import MagicMock, patch


# =====================================================================
# list_history_message_ids: deleted-id extraction + netting
# =====================================================================
class TestHistoryDeletedIds:
    def test_collects_deleted_ids(self):
        from gmail_oauth import list_history_message_ids

        fake = {
            "historyId": "555",
            "history": [
                {"messagesAdded": [{"message": {"id": "a1"}}]},
                {"messagesDeleted": [{"message": {"id": "d1"}}, {"message": {"id": "d2"}}]},
            ],
        }
        with patch(
            "gmail_oauth._authed_request",
            return_value=(fake, {"access_token": "tok"}),
        ):
            out = list_history_message_ids(
                {"access_token": "tok", "refresh_token": "x"},
                start_history_id="100",
                label_id="INBOX",
                max_results=10,
            )
        assert out["message_ids"] == ["a1"]
        assert out["deleted_message_ids"] == ["d1", "d2"]
        assert out["next_history_id"] == "555"
        assert out["invalidated"] is False

    def test_add_then_delete_in_window_nets_out(self):
        from gmail_oauth import list_history_message_ids

        fake = {
            "historyId": "777",
            "history": [
                {"messagesAdded": [{"message": {"id": "m1"}}, {"message": {"id": "m2"}}]},
                {"messagesDeleted": [{"message": {"id": "m2"}}]},
            ],
        }
        with patch(
            "gmail_oauth._authed_request",
            return_value=(fake, {"access_token": "tok"}),
        ):
            out = list_history_message_ids(
                {"access_token": "tok", "refresh_token": "x"},
                start_history_id="100",
                label_id="INBOX",
            )
        # m2 was added then deleted -> excluded from ingest, present in deletes.
        assert out["message_ids"] == ["m1"]
        assert out["deleted_message_ids"] == ["m2"]

    def test_blank_start_includes_deleted_key(self):
        from gmail_oauth import list_history_message_ids

        with patch("gmail_oauth._authed_request") as mock:
            out = list_history_message_ids(
                {"access_token": "tok", "refresh_token": "x"},
                start_history_id="",
            )
        mock.assert_not_called()
        assert out["deleted_message_ids"] == []

    def test_invalidated_includes_deleted_key(self):
        from gmail_oauth import GmailApiError, list_history_message_ids

        with patch(
            "gmail_oauth._authed_request",
            side_effect=GmailApiError("Gmail HTTP 404"),
        ):
            out = list_history_message_ids(
                {"access_token": "tok", "refresh_token": "x"},
                start_history_id="old",
            )
        assert out["invalidated"] is True
        assert out["deleted_message_ids"] == []


# =====================================================================
# run_workspace_gmail_ingest: deletion processing on incremental path
# =====================================================================
class TestRunnerDeletionProcessing:
    def _connection(self):
        return {
            "id": "conn-1",
            "workspace_id": "ws-1",
            "email": "owner@example.com",
            "access_token": "at-1",
            "refresh_token": "rt-1",
            "status": "active",
        }

    def test_incremental_deletes_from_hydra_and_memory(self):
        from gmail_oauth import run_workspace_gmail_ingest

        hist_result = {
            "message_ids": [],
            "deleted_message_ids": ["d1", "d2"],
            "next_history_id": "200",
            "invalidated": False,
        }
        mock_hydra_instance = MagicMock()

        with patch(
            "supabase_client.get_gmail_ingestion_state_map",
            return_value={"INBOX": {"last_history_id": "100"}},
        ), patch(
            "gmail_oauth.list_history_message_ids",
            return_value=hist_result,
        ), patch(
            "hydradb_client.HydraDBClient",
            return_value=mock_hydra_instance,
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ), patch(
            "memory_store.delete_memories_by_source",
            return_value=True,
        ) as mock_mem_del:
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
                sync_mode="incremental",
            )

        # HydraDB deletion: one batched call with both stable keys.
        mock_hydra_instance.delete_knowledge.assert_called_once_with(
            ["gmail:msg:d1", "gmail:msg:d2"]
        )
        # Memory cleanup: one call per deleted source key.
        assert mock_mem_del.call_count == 2
        mock_mem_del.assert_any_call(
            workspace_id="ws-1", source_stable_key="gmail:msg:d1"
        )
        mock_mem_del.assert_any_call(
            workspace_id="ws-1", source_stable_key="gmail:msg:d2"
        )
        assert stats["messages_deleted"] == 2

    def test_hydra_delete_failure_does_not_break_run(self):
        from gmail_oauth import run_workspace_gmail_ingest

        hist_result = {
            "message_ids": [],
            "deleted_message_ids": ["d1"],
            "next_history_id": "200",
            "invalidated": False,
        }
        mock_hydra_instance = MagicMock()
        mock_hydra_instance.delete_knowledge.side_effect = RuntimeError("hydra down")

        with patch(
            "supabase_client.get_gmail_ingestion_state_map",
            return_value={"INBOX": {"last_history_id": "100"}},
        ), patch(
            "gmail_oauth.list_history_message_ids",
            return_value=hist_result,
        ), patch(
            "hydradb_client.HydraDBClient",
            return_value=mock_hydra_instance,
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ), patch(
            "memory_store.delete_memories_by_source",
            return_value=True,
        ):
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
                sync_mode="incremental",
            )

        # Even though HydraDB delete raised, the run completes and counts
        # the attempted deletions; the label is still processed.
        assert stats["messages_deleted"] == 1
        assert stats["labels_processed"] == 1

    def test_full_sync_does_not_attempt_deletions(self):
        from gmail_oauth import run_workspace_gmail_ingest

        mock_hydra_instance = MagicMock()
        with patch(
            "gmail_oauth.list_message_ids_for_label",
            return_value=[],
        ), patch(
            "hydradb_client.HydraDBClient",
            return_value=mock_hydra_instance,
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
                sync_mode="full",
            )
        mock_hydra_instance.delete_knowledge.assert_not_called()
        assert stats["messages_deleted"] == 0

    def test_messages_deleted_in_stats_shape(self):
        from gmail_oauth import run_workspace_gmail_ingest

        with patch(
            "gmail_oauth.list_message_ids_for_label",
            return_value=[],
        ), patch(
            "hydradb_client.HydraDBClient",
        ), patch(
            "supabase_client.upsert_gmail_ingestion_state",
            return_value=True,
        ):
            stats = run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._connection(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
            )
        assert "messages_deleted" in stats