"""
Tests for the Phase 3 Slack channel + ingest routes:
    GET  /api/slack/channels
    POST /api/slack/channels
    POST /api/slack/ingest

We patch the supabase_client helpers and the Slack-API enumeration at
the main module's namespace (main.list_slack_channels, etc.) — the
routes call them by name from main, so the patch sits exactly there.
"""

from unittest.mock import patch

import pytest

TEST_WS_ID = "00000000-0000-0000-0000-00000000aaaa"


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    """
    Phase 7: the new per-bucket rate limits (slack_ingest=5/5min,
    auth=30/5min, etc.) accumulate across tests in the same process.
    Without resetting them, the 6th test in TestRunIngest hits the
    /api/slack/ingest limit and gets 429 -- shadowing the 401/400/etc.
    the test was actually checking for. Clear the buckets per-test
    so each assertion sees the correct status code.
    """
    from rate_limit import _limiter

    with _limiter._lock:
        _limiter._buckets.clear()
    yield
    with _limiter._lock:
        _limiter._buckets.clear()


# ── GET /api/slack/channels ──────────────────────────────────────────────
class TestListChannels:
    def test_not_connected_returns_empty_state(self, client, jwt_auth_headers):
        with patch("main.get_slack_installation", return_value=None):
            r = client.get("/api/slack/channels", headers=jwt_auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is False
        assert body["channels"] == []

    def test_connected_returns_channels_and_refreshes(
        self,
        client,
        jwt_auth_headers,
    ):
        install = {
            "workspace_id": TEST_WS_ID,
            "slack_team_name": "Acme",
            "bot_token": "xoxb-test",
        }
        fresh_from_slack = [
            {"slack_channel_id": "C1", "name": "general", "is_archived": False},
            {"slack_channel_id": "C2", "name": "random", "is_archived": False},
        ]
        stored_rows = [
            {
                "slack_channel_id": "C1",
                "name": "general",
                "is_selected": True,
                "is_archived": False,
                "updated_at": "2026-01-01T10:00:00+00:00",
            },
            {
                "slack_channel_id": "C2",
                "name": "random",
                "is_selected": False,
                "is_archived": False,
                "updated_at": "2026-01-01T10:00:00+00:00",
            },
        ]
        with patch("main.get_slack_installation", return_value=install), patch(
            "main.list_slack_channels", return_value=fresh_from_slack
        ) as mock_fresh, patch("main.upsert_slack_channels", return_value=2) as mock_upsert, patch(
            "main.list_workspace_channels", return_value=stored_rows
        ) as mock_list:
            r = client.get("/api/slack/channels", headers=jwt_auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is True
        assert body["team_name"] == "Acme"
        assert len(body["channels"]) == 2

        # Verify the refresh-then-read pattern actually fired.
        mock_fresh.assert_called_once()
        mock_upsert.assert_called_once()
        _, kwargs = mock_list.call_args
        assert kwargs["workspace_id"] == TEST_WS_ID

    def test_connected_but_slack_refresh_fails_still_returns_stored(
        self,
        client,
        jwt_auth_headers,
    ):
        # Slack API hiccup shouldn't break the picker — we just return
        # what's already in the DB.
        install = {"slack_team_name": "Acme", "bot_token": "xoxb-test"}
        stored_ch = [
            {
                "slack_channel_id": "C1",
                "name": "general",
                "is_selected": True,
                "is_archived": False,
                "updated_at": "2026-01-01T10:00:00+00:00",
            }
        ]
        with patch("main.get_slack_installation", return_value=install), patch(
            "main.list_slack_channels", return_value=[]
        ), patch("main.upsert_slack_channels", return_value=0) as mock_upsert, patch(
            "main.list_workspace_channels", return_value=stored_ch
        ):
            r = client.get("/api/slack/channels", headers=jwt_auth_headers)
        assert r.status_code == 200
        body = r.json()
        assert body["connected"] is True
        # No fresh channels -> no upsert call.
        mock_upsert.assert_not_called()
        assert len(body["channels"]) == 1

    def test_requires_auth(self, client):
        r = client.get("/api/slack/channels")
        assert r.status_code == 401


# ── POST /api/slack/channels ─────────────────────────────────────────────
class TestSaveChannels:
    def test_saves_selected_set(self, client, jwt_auth_headers):
        with patch(
            "main.set_selected_channels",
            return_value=True,
        ) as mock_fn:
            r = client.post(
                "/api/slack/channels",
                headers=jwt_auth_headers,
                json={"selected_channel_ids": ["C1", "C3"]},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["selected_count"] == 2
        _, kwargs = mock_fn.call_args
        assert kwargs["workspace_id"] == TEST_WS_ID
        assert kwargs["selected_ids"] == ["C1", "C3"]

    def test_saves_empty_set(self, client, jwt_auth_headers):
        with patch(
            "main.set_selected_channels",
            return_value=True,
        ) as mock_fn:
            r = client.post(
                "/api/slack/channels",
                headers=jwt_auth_headers,
                json={"selected_channel_ids": []},
            )
        assert r.status_code == 200
        assert r.json()["selected_count"] == 0
        _, kwargs = mock_fn.call_args
        assert kwargs["selected_ids"] == []

    def test_db_failure_returns_502(self, client, jwt_auth_headers):
        with patch("main.set_selected_channels", return_value=False):
            r = client.post(
                "/api/slack/channels",
                headers=jwt_auth_headers,
                json={"selected_channel_ids": ["C1"]},
            )
        assert r.status_code == 502

    def test_rejects_extra_field(self, client, jwt_auth_headers):
        r = client.post(
            "/api/slack/channels",
            headers=jwt_auth_headers,
            json={"selected_channel_ids": ["C1"], "random": "no"},
        )
        assert r.status_code == 422

    def test_requires_auth(self, client):
        r = client.post(
            "/api/slack/channels",
            json={"selected_channel_ids": ["C1"]},
        )
        assert r.status_code == 401


# ── POST /api/slack/ingest ───────────────────────────────────────────────
class TestRunIngest:
    def test_kicks_off_background_ingest(self, client, jwt_auth_headers):
        install = {"bot_token": "xoxb-test"}
        channel_settings = [
            {"slack_channel_id": "C1", "include_bot_messages": False},
            {"slack_channel_id": "C2", "include_bot_messages": True},
        ]
        with patch(
            "main.get_slack_installation",
            return_value=install,
        ), patch(
            "main.list_selected_channel_settings",
            return_value=channel_settings,
        ), patch(
            # Phase 4: the route resolves the workspace's HydraDB
            # sub-tenant before scheduling the runner. Patch this so
            # the test doesn't hit Supabase.
            "main.ensure_workspace_sub_tenant",
            return_value="ws_test_abc",
        ), patch(
            "main.run_workspace_ingest",
            return_value={},
        ) as mock_runner:
            r = client.post("/api/slack/ingest", headers=jwt_auth_headers)
        assert r.status_code == 202
        body = r.json()
        assert body["status"] == "started"
        assert body["channels_queued"] == 2
        # BackgroundTask executes synchronously when TestClient drains
        # its lifespan, so the runner WILL have been called.
        mock_runner.assert_called_once()
        _, kwargs = mock_runner.call_args
        assert kwargs["workspace_id"] == TEST_WS_ID
        assert kwargs["bot_token"] == "xoxb-test"
        assert kwargs["channel_ids"] == ["C1", "C2"]
        assert kwargs["channel_bot_messages"] == {"C1": False, "C2": True}
        # Phase 4: the resolved sub_tenant_id must be forwarded.
        assert kwargs["hydradb_sub_tenant_id"] == "ws_test_abc"

    def test_no_installation_returns_400(self, client, jwt_auth_headers):
        with patch("main.get_slack_installation", return_value=None):
            r = client.post("/api/slack/ingest", headers=jwt_auth_headers)
        assert r.status_code == 400

    def test_blank_bot_token_returns_400(self, client, jwt_auth_headers):
        with patch(
            "main.get_slack_installation",
            return_value={"bot_token": "   "},
        ):
            r = client.post("/api/slack/ingest", headers=jwt_auth_headers)
        assert r.status_code == 400

    def test_no_channels_selected_returns_400(self, client, jwt_auth_headers):
        with patch(
            "main.get_slack_installation",
            return_value={"bot_token": "xoxb-test"},
        ), patch(
            "main.list_selected_channel_ids",
            return_value=[],
        ):
            r = client.post("/api/slack/ingest", headers=jwt_auth_headers)
        assert r.status_code == 400

    def test_sub_tenant_lookup_failure_returns_502(
        self,
        client,
        jwt_auth_headers,
    ):
        # Phase 4: if ensure_workspace_sub_tenant returns None we
        # refuse rather than fall back to the env default -- that
        # would leak data into the shared HydraDB bucket.
        with patch(
            "main.get_slack_installation",
            return_value={"bot_token": "xoxb-test"},
        ), patch(
            "main.list_selected_channel_settings",
            return_value=[{"slack_channel_id": "C1", "include_bot_messages": False}],
        ), patch(
            "main.ensure_workspace_sub_tenant",
            return_value=None,
        ), patch(
            "main.run_workspace_ingest",
        ) as mock_runner:
            r = client.post("/api/slack/ingest", headers=jwt_auth_headers)
        assert r.status_code == 502
        mock_runner.assert_not_called()

    def test_requires_auth(self, client):
        r = client.post("/api/slack/ingest")
        assert r.status_code == 401


# ── upsert_slack_channels — production schema alignment ──────────────────
class TestUpsertChannelsProductionPayload:
    """
    Pins the wire-level shape of the Slack-channels upsert. Production
    schema has slack_channels.installation_id NOT NULL and several
    other columns the original code wasn't populating.
    """

    def _mock_supabase(self, exec_result=None, exec_raises=None):
        from unittest.mock import MagicMock

        execute = MagicMock()
        if exec_raises is not None:
            execute.side_effect = exec_raises
        else:
            execute.return_value = MagicMock(data=exec_result or [{"id": "r-1"}])
        upsert = MagicMock(return_value=MagicMock(execute=execute))
        table = MagicMock(return_value=MagicMock(upsert=upsert))
        client = MagicMock(table=table)
        return client, upsert

    def test_includes_installation_id_when_provided(self):
        """Production's slack_channels.installation_id FK is required.
        When the route passes it, every row in the upsert payload must
        carry it."""
        from unittest.mock import patch

        from supabase_client import upsert_slack_channels

        client, upsert = self._mock_supabase(
            exec_result=[{"id": "r-1"}, {"id": "r-2"}],
        )
        with patch("supabase_client.get_supabase", return_value=client):
            upsert_slack_channels(
                workspace_id=TEST_WS_ID,
                installation_id="inst-uuid-abc",
                channels=[
                    {"slack_channel_id": "C1", "name": "general"},
                    {"slack_channel_id": "C2", "name": "random"},
                ],
            )
        rows = upsert.call_args.args[0]
        assert len(rows) == 2
        for r in rows:
            assert r["installation_id"] == "inst-uuid-abc"

    def test_omits_installation_id_when_not_provided(self):
        """Backwards compat: older callers / dev DBs without the FK
        column must still work. The kwarg is optional and the field is
        excluded from the payload when blank."""
        from unittest.mock import patch

        from supabase_client import upsert_slack_channels

        client, upsert = self._mock_supabase()
        with patch("supabase_client.get_supabase", return_value=client):
            upsert_slack_channels(
                workspace_id=TEST_WS_ID,
                channels=[{"slack_channel_id": "C1", "name": "g"}],
            )
        row = upsert.call_args.args[0][0]
        assert "installation_id" not in row

    def test_populates_all_production_columns(self):
        """Every production schema column must end up in the payload
        with a value of the right type."""
        from unittest.mock import patch

        from supabase_client import upsert_slack_channels

        client, upsert = self._mock_supabase()
        with patch("supabase_client.get_supabase", return_value=client):
            upsert_slack_channels(
                workspace_id=TEST_WS_ID,
                installation_id="inst-1",
                channels=[
                    {
                        "slack_channel_id": "C1",
                        "name": "engineering",
                        "is_private": True,
                        "is_archived": False,
                        "member_count": 42,
                        "topic": "ship it",
                        "purpose": "where we ship",
                    }
                ],
            )
        row = upsert.call_args.args[0][0]
        assert row["workspace_id"] == TEST_WS_ID
        assert row["installation_id"] == "inst-1"
        assert row["slack_channel_id"] == "C1"
        assert row["name"] == "engineering"
        assert row["is_private"] is True
        assert row["is_archived"] is False
        assert row["member_count"] == 42
        assert row["topic"] == "ship it"
        assert row["purpose"] == "where we ship"
        assert "last_seen_at" in row  # ISO timestamp string
        # is_selected MUST NOT appear — we'd clobber user selections.
        assert "is_selected" not in row

    def test_safe_defaults_when_slack_omits_fields(self):
        """Slack omits num_members for private channels the bot isn't
        in. The payload must not insert NULL into a NOT NULL column."""
        from unittest.mock import patch

        from supabase_client import upsert_slack_channels

        client, upsert = self._mock_supabase()
        with patch("supabase_client.get_supabase", return_value=client):
            upsert_slack_channels(
                workspace_id=TEST_WS_ID,
                installation_id="inst-1",
                channels=[{"slack_channel_id": "C1", "name": "g"}],
            )
        row = upsert.call_args.args[0][0]
        assert row["member_count"] == 0
        assert row["topic"] == ""
        assert row["purpose"] == ""
        assert row["is_private"] is False
        assert row["is_archived"] is False

    def test_logs_real_postgrest_error_body(self, caplog):
        """The previous code only logged `type(e).__name__`. Now we
        capture the structured PostgREST body so an operator can see
        WHY the upsert failed."""
        import logging
        from unittest.mock import patch

        try:
            from postgrest.exceptions import APIError as PGAPIError
        except ImportError:  # pragma: no cover
            pytest.skip("postgrest not installed")

        err = PGAPIError(
            {
                "code": "23502",
                "message": "null value in column \"installation_id\" violates NOT NULL",
                "hint": None,
                "details": "Failing row contains...",
            }
        )
        from supabase_client import upsert_slack_channels

        client, _ = self._mock_supabase(exec_raises=err)

        with caplog.at_level(logging.WARNING, logger="supabase_client"):
            with patch("supabase_client.get_supabase", return_value=client):
                result = upsert_slack_channels(
                    workspace_id=TEST_WS_ID,
                    channels=[{"slack_channel_id": "C1", "name": "g"}],
                )
        assert result == 0
        records = [r for r in caplog.records if r.message == "supabase_upsert_channels_failed"]
        assert len(records) == 1
        rec = records[0]
        assert getattr(rec, "pg_code") == "23502"
        assert "installation_id" in getattr(rec, "pg_message")
        # error_repr only appears on the non-APIError fallback path.
        assert not hasattr(rec, "error_repr")


# ── list_slack_channels — Slack response mapping ─────────────────────────
class TestListSlackChannelsMapping:
    def test_extracts_production_fields(self):
        """The Slack-side helper must extract every field the production
        DB has a column for: is_private, num_members → member_count,
        topic.value, purpose.value."""
        from unittest.mock import MagicMock, patch

        # WebClient is constructed inside the function; intercept it.
        fake_resp = {
            "channels": [
                {
                    "id": "C1",
                    "name": "engineering",
                    "is_archived": False,
                    "is_private": True,
                    "num_members": 12,
                    "topic": {"value": "ship it", "creator": "U1", "last_set": 0},
                    "purpose": {"value": "where we ship", "creator": "U1", "last_set": 0},
                }
            ],
            "response_metadata": {"next_cursor": ""},
        }
        mock_client = MagicMock()
        mock_client.conversations_list.return_value = fake_resp
        from slack_oauth import list_slack_channels

        with patch("slack_oauth.WebClient", return_value=mock_client):
            out = list_slack_channels("xoxb-test")
        assert len(out) == 1
        ch = out[0]
        assert ch["slack_channel_id"] == "C1"
        assert ch["name"] == "engineering"
        assert ch["is_private"] is True
        assert ch["is_archived"] is False
        assert ch["member_count"] == 12
        assert ch["topic"] == "ship it"
        assert ch["purpose"] == "where we ship"

    def test_handles_missing_num_members_safely(self):
        """Slack omits num_members for private channels the bot isn't
        a member of. Must default to 0, not raise."""
        from unittest.mock import MagicMock, patch

        fake_resp = {
            "channels": [
                {
                    "id": "C1",
                    "name": "private-thing",
                    "is_archived": False,
                    "is_private": True,
                    # num_members deliberately absent.
                }
            ],
            "response_metadata": {"next_cursor": ""},
        }
        mock_client = MagicMock()
        mock_client.conversations_list.return_value = fake_resp
        from slack_oauth import list_slack_channels

        with patch("slack_oauth.WebClient", return_value=mock_client):
            out = list_slack_channels("xoxb-test")
        assert out[0]["member_count"] == 0

    def test_handles_missing_topic_and_purpose(self):
        from unittest.mock import MagicMock, patch

        fake_resp = {
            "channels": [
                {
                    "id": "C1",
                    "name": "g",
                    "is_archived": False,
                    "is_private": False,
                    # topic and purpose absent entirely.
                }
            ],
            "response_metadata": {"next_cursor": ""},
        }
        mock_client = MagicMock()
        mock_client.conversations_list.return_value = fake_resp
        from slack_oauth import list_slack_channels

        with patch("slack_oauth.WebClient", return_value=mock_client):
            out = list_slack_channels("xoxb-test")
        assert out[0]["topic"] == ""
        assert out[0]["purpose"] == ""


# ── /api/slack/channels route forwards installation_id ────────────────────
class TestRouteForwardsInstallationId:
    def test_get_channels_passes_installation_id_to_upsert(
        self,
        client,
        jwt_auth_headers,
    ):
        install = {
            "id": "inst-uuid-real",
            "slack_team_name": "Acme",
            "bot_token": "xoxb-test",
        }
        fresh_from_slack = [
            {
                "slack_channel_id": "C1",
                "name": "g",
                "is_archived": False,
                "is_private": False,
                "member_count": 0,
                "topic": "",
                "purpose": "",
            }
        ]
        with patch("main.get_slack_installation", return_value=install), patch(
            "main.list_slack_channels", return_value=fresh_from_slack
        ), patch("main.upsert_slack_channels", return_value=1) as mock_upsert, patch(
            "main.list_workspace_channels", return_value=[]
        ):
            r = client.get("/api/slack/channels", headers=jwt_auth_headers)
        assert r.status_code == 200
        kwargs = mock_upsert.call_args.kwargs
        # The headline production fix: installation_id MUST be forwarded.
        assert kwargs["installation_id"] == "inst-uuid-real"
        assert kwargs["workspace_id"] == TEST_WS_ID
