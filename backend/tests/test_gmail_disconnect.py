"""
Gmail disconnect flow (audit item 4) — backend layer.

The existing DELETE /api/gmail/connections/{id} route (workspace-scoped,
cascade-deletes labels + ingestion state) is unchanged; future syncs
stop the moment the row is gone. This test set covers the added
disconnect hygiene: delete_gmail_connection best-effort revokes the
stored Google OAuth grant after a successful, workspace-scoped delete,
and a revocation failure never changes the delete result.
"""

from unittest.mock import MagicMock, patch


def _client_with(select_data, delete_data):
    """
    Build a fake Supabase client whose select-chain and delete-chain
    return the given `.data` payloads.
    """
    client = MagicMock()
    table = client.table.return_value

    select_exec = (
        table.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute
    )
    select_exec.return_value = MagicMock(data=select_data)

    delete_exec = table.delete.return_value.eq.return_value.eq.return_value.execute
    delete_exec.return_value = MagicMock(data=delete_data)
    return client


class TestDeleteGmailConnectionRevocation:
    def test_revokes_grant_on_successful_delete(self):
        from supabase_client import delete_gmail_connection

        client = _client_with(
            select_data=[{"refresh_token": "rt-xyz", "access_token": "at-xyz"}],
            delete_data=[{"id": "conn-1"}],
        )
        with patch("supabase_client.get_supabase", return_value=client), patch(
            "gmail_oauth.revoke_token",
            return_value=True,
        ) as mock_revoke:
            ok = delete_gmail_connection(connection_id="conn-1", workspace_id="ws-1")

        assert ok is True
        mock_revoke.assert_called_once_with("rt-xyz")

    def test_no_revoke_when_nothing_deleted(self):
        from supabase_client import delete_gmail_connection

        # Foreign / unknown connection: delete matches no row.
        client = _client_with(
            select_data=[],  # workspace-scoped read finds nothing
            delete_data=[],  # workspace-scoped delete removes nothing
        )
        with patch("supabase_client.get_supabase", return_value=client), patch(
            "gmail_oauth.revoke_token",
            return_value=True,
        ) as mock_revoke:
            ok = delete_gmail_connection(connection_id="foreign", workspace_id="ws-1")

        assert ok is False
        mock_revoke.assert_not_called()

    def test_revoke_failure_does_not_block_delete(self):
        from supabase_client import delete_gmail_connection

        client = _client_with(
            select_data=[{"refresh_token": "rt-xyz"}],
            delete_data=[{"id": "conn-1"}],
        )
        with patch("supabase_client.get_supabase", return_value=client), patch(
            "gmail_oauth.revoke_token",
            side_effect=RuntimeError("google down"),
        ):
            ok = delete_gmail_connection(connection_id="conn-1", workspace_id="ws-1")

        # Delete is authoritative; revocation is best-effort hygiene.
        assert ok is True

    def test_read_failure_still_deletes(self):
        from supabase_client import delete_gmail_connection

        client = MagicMock()
        table = client.table.return_value
        # select chain raises -> no token to revoke, delete still proceeds.
        table.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.side_effect = RuntimeError(
            "read boom"
        )
        table.delete.return_value.eq.return_value.eq.return_value.execute.return_value = MagicMock(
            data=[{"id": "conn-1"}]
        )
        with patch("supabase_client.get_supabase", return_value=client), patch(
            "gmail_oauth.revoke_token",
            return_value=True,
        ) as mock_revoke:
            ok = delete_gmail_connection(connection_id="conn-1", workspace_id="ws-1")

        assert ok is True
        # No token was read, so revocation is skipped.
        mock_revoke.assert_not_called()

    def test_missing_ids_returns_false_without_db(self):
        from supabase_client import delete_gmail_connection

        with patch("supabase_client.get_supabase") as mock_get:
            ok = delete_gmail_connection(connection_id="", workspace_id="ws-1")
        assert ok is False
        mock_get.assert_not_called()


class TestSetGmailConnectionStatus:
    def test_writes_status_workspace_scoped(self):
        from supabase_client import set_gmail_connection_status

        client = MagicMock()
        with patch("supabase_client.get_supabase", return_value=client):
            ok = set_gmail_connection_status(
                connection_id="conn-1",
                workspace_id="ws-1",
                status="revoked",
            )
        assert ok is True
        client.table.assert_called_with("gmail_connections")
        client.table.return_value.update.assert_called_once_with({"status": "revoked"})

    def test_missing_args_returns_false(self):
        from supabase_client import set_gmail_connection_status

        with patch("supabase_client.get_supabase") as mock_get:
            assert set_gmail_connection_status(
                connection_id="", workspace_id="ws-1", status="revoked"
            ) is False
        mock_get.assert_not_called()

    def test_db_error_returns_false(self):
        from supabase_client import set_gmail_connection_status

        client = MagicMock()
        client.table.return_value.update.return_value.eq.return_value.eq.return_value.execute.side_effect = RuntimeError(
            "boom"
        )
        with patch("supabase_client.get_supabase", return_value=client):
            ok = set_gmail_connection_status(
                connection_id="conn-1",
                workspace_id="ws-1",
                status="error",
            )
        assert ok is False