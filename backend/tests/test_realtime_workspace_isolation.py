"""
Phase 5 -- workspace-aware realtime Slack Events ingestion.

These tests verify the new contract:

  /slack/events
    -> signature check (preserved from Phase 1)
    -> url_verification handshake (preserved)
    -> event_callback:
         dedupe by event_id (preserved)
         -> realtime_ingest.process_slack_event(payload)  (now takes FULL payload)
              -> map team_id -> slack_installation -> workspace_id
              -> if channel not is_selected for that workspace: drop
              -> ingest with workspace's bot_token + sub_tenant_id

We patch at module boundaries so the tests don't hit Slack or HydraDB.
"""

import hashlib
import hmac
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------- #
# Helpers: build signed /slack/events requests
# ---------------------------------------------------------------------- #
def _sign(body: bytes) -> dict:
    ts = int(time.time())
    secret = os.environ.get(
        "SLACK_SIGNING_SECRET",
        "test-slack-signing-secret",
    )
    base = b"v0:" + str(ts).encode() + b":" + body
    digest = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return {
        "X-Slack-Signature": f"v0={digest}",
        "X-Slack-Request-Timestamp": str(ts),
    }


def _post(client, payload: dict):
    body = json.dumps(payload).encode()
    return client.post("/slack/events", content=body, headers=_sign(body))


# ---------------------------------------------------------------------- #
# Webhook-shell behavior (preserved from Phase 1)
# ---------------------------------------------------------------------- #
class TestWebhookShellPreserved:
    def test_url_verification_still_works(self, client):
        r = _post(
            client,
            {
                "type": "url_verification",
                "challenge": "phase5-challenge",
            },
        )
        assert r.status_code == 200
        assert "phase5-challenge" in r.text

    def test_invalid_signature_still_returns_401(self, client):
        body = b'{"type":"event_callback"}'
        r = client.post(
            "/slack/events",
            content=body,
            headers={
                "X-Slack-Signature": "v0=badhash",
                "X-Slack-Request-Timestamp": str(int(time.time())),
            },
        )
        assert r.status_code == 401

    def test_event_callback_acks_200_even_when_dropped(self, client):
        """Slack must always see 200 -- otherwise it retries forever."""
        payload = {
            "type": "event_callback",
            "event_id": "Ev_phase5_ack_1",
            "team_id": "T_UNKNOWN",
            "event": {"type": "message", "channel": "C1", "text": "hi"},
        }
        with patch(
            "realtime_ingest.get_slack_installation_by_team_id",
            return_value=None,
        ):
            r = _post(client, payload)
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_duplicate_event_id_suppressed(self, client):
        payload = {
            "type": "event_callback",
            "event_id": "Ev_phase5_dup_1",
            "team_id": "T_TEAM",
            "event": {"type": "message", "channel": "C1", "text": "hi"},
        }
        with patch(
            "main.process_slack_event",
        ) as mock_proc:
            for _ in range(3):
                _post(client, payload)
        # process_slack_event is the symbol our route schedules. The
        # dedupe gate sits BEFORE the background_tasks.add_task call,
        # so duplicates never get scheduled.
        assert mock_proc.call_count == 1


# ---------------------------------------------------------------------- #
# Route now passes the FULL payload to process_slack_event (not just event)
# ---------------------------------------------------------------------- #
class TestRoutePassesFullPayload:
    def test_route_forwards_full_payload(self, client):
        """The handler needs `team_id` from the envelope to route."""
        payload = {
            "type": "event_callback",
            "event_id": "Ev_phase5_passthrough",
            "team_id": "T_PASS",
            "event": {"type": "message", "channel": "C1", "text": "hi"},
        }
        with patch(
            "main.process_slack_event",
        ) as mock_proc:
            r = _post(client, payload)
        assert r.status_code == 200
        mock_proc.assert_called_once()
        # The first positional arg must be the full envelope, not the
        # inner event -- otherwise the new workspace-routing logic
        # has no team_id to work with.
        forwarded = mock_proc.call_args.args[0]
        assert forwarded.get("type") == "event_callback"
        assert forwarded.get("team_id") == "T_PASS"
        assert forwarded.get("event_id") == "Ev_phase5_passthrough"


# ---------------------------------------------------------------------- #
# Workspace routing: team_id -> installation -> workspace
# ---------------------------------------------------------------------- #
class TestProcessSlackEventRouting:
    """
    Direct unit tests against realtime_ingest.process_slack_event so we
    can inspect what gets called (or not) inside.
    """

    @pytest.fixture(autouse=True)
    def _reset_seen_cache(self, monkeypatch):
        """Each test starts with a fresh in-memory dedupe cache."""
        import realtime_ingest as r

        # Ensure realtime is enabled regardless of what the .env file says
        # (the TestClient tests load .env which may have it disabled).
        monkeypatch.setenv("REALTIME_INGEST_ENABLED", "true")
        r._seen_event_ids.clear()
        r._bot_user_id_by_token.clear()
        r._in_flight.clear()
        yield
        r._seen_event_ids.clear()
        r._bot_user_id_by_token.clear()
        r._in_flight.clear()

    def _build_payload(
        self,
        *,
        team_id="T_TEAM",
        channel="C1",
        text="hello world",
        ts="1700000000.000100",
        thread_ts=None,
        subtype=None,
        user="U_HUMAN",
    ):
        event = {
            "type": "message",
            "channel": channel,
            "text": text,
            "ts": ts,
            "user": user,
        }
        if thread_ts:
            event["thread_ts"] = thread_ts
        if subtype:
            event["subtype"] = subtype
        return {
            "type": "event_callback",
            "event_id": "Ev_test",
            "team_id": team_id,
            "event": event,
        }

    def test_unknown_team_id_is_silently_dropped(self):
        """Slack apps can be installed in multiple teams; we ignore
        events from teams we have no installation row for."""
        from realtime_ingest import process_slack_event

        payload = self._build_payload(team_id="T_UNKNOWN")
        with patch(
            "realtime_ingest.get_slack_installation_by_team_id",
            return_value=None,
        ) as mock_lookup, patch(
            "realtime_ingest.SlackClientWrapper",
        ) as mock_slack, patch(
            "realtime_ingest.HydraDBClient",
        ) as mock_hydra:
            process_slack_event(payload)
        mock_lookup.assert_called_once_with(slack_team_id="T_UNKNOWN")
        mock_slack.assert_not_called()
        mock_hydra.assert_not_called()

    def test_unselected_channel_is_dropped_before_slack_or_hydradb(self):
        """If the user hasn't opted this channel in, we don't even
        construct a SlackClientWrapper -- saving an unnecessary
        auth.test round-trip."""
        from realtime_ingest import process_slack_event

        payload = self._build_payload(team_id="T_TEAM", channel="C_NOT_PICKED")
        installation = {
            "workspace_id": "ws-1",
            "bot_token": "xoxb-team-1",
            "slack_team_id": "T_TEAM",
        }
        with patch(
            "realtime_ingest.get_slack_installation_by_team_id",
            return_value=installation,
        ), patch(
            "realtime_ingest.get_channel_ingest_settings",
            return_value=None,
        ) as mock_sel, patch(
            "realtime_ingest.SlackClientWrapper",
        ) as mock_slack, patch(
            "realtime_ingest.HydraDBClient",
        ) as mock_hydra:
            process_slack_event(payload)
        mock_sel.assert_called_once_with(
            workspace_id="ws-1",
            slack_channel_id="C_NOT_PICKED",
        )
        mock_slack.assert_not_called()
        mock_hydra.assert_not_called()

    def test_selected_channel_ingests_with_workspace_bot_token(self):
        """Happy path: channel is selected -> we build a SlackClientWrapper
        with the workspace's bot_token and a HydraDBClient with the
        workspace's sub_tenant_id."""
        from realtime_ingest import process_slack_event

        payload = self._build_payload(team_id="T_TEAM", channel="C_PICKED")
        installation = {
            "workspace_id": "ws-1",
            "bot_token": "xoxb-team-1",
            "slack_team_id": "T_TEAM",
        }

        mock_slack_instance = MagicMock()
        mock_slack_instance.client.auth_test.return_value = {
            "user_id": "U_BOT_1",
        }
        mock_slack_instance.fetch_thread_replies.return_value = []

        mock_hydra_instance = MagicMock()
        mock_hydra_instance.upload_knowledge.return_value = {
            "success": True,
            "success_count": 1,
            "failed_count": 0,
        }

        with patch(
            "realtime_ingest.get_slack_installation_by_team_id",
            return_value=installation,
        ), patch(
            "realtime_ingest.get_channel_ingest_settings",
            return_value={"is_selected": True, "include_bot_messages": False},
        ), patch(
            "realtime_ingest.ensure_workspace_sub_tenant",
            return_value="ws_workspace_1",
        ), patch(
            "realtime_ingest.SlackClientWrapper",
            return_value=mock_slack_instance,
        ) as mock_slack_cls, patch(
            "realtime_ingest.HydraDBClient",
            return_value=mock_hydra_instance,
        ) as mock_hydra_cls, patch(
            "realtime_ingest.fetch_channel_name",
            return_value="general",
        ), patch(
            "realtime_ingest.build_message_file",
            return_value={
                "filename": "slack_general_1700000000_000100.md",
                "content": "# msg",
                "stable_key": "slack:msg:C_PICKED:1700000000.000100",
                "doc_type": "message",
            },
        ), patch(
            "realtime_ingest.IngestionState",
        ) as mock_state_cls:
            mock_state = MagicMock()
            mock_state.has.return_value = False
            mock_state_cls.return_value = mock_state
            mock_state_cls.locked.return_value.__enter__.return_value = mock_state

            process_slack_event(payload)

        # Slack client built with the workspace's bot_token. The retry
        # wrapper may invoke the underlying handler multiple times if
        # an inner mock raises -- assert "at least one call" with the
        # expected kwargs, since retry behavior is exercised separately
        # in test_phase7_hardening.py.
        mock_slack_cls.assert_any_call(token="xoxb-team-1")
        # HydraDB client built with the workspace's sub_tenant_id.
        mock_hydra_cls.assert_any_call(sub_tenant_id="ws_workspace_1")
        # Upload actually fired.
        assert mock_hydra_instance.upload_knowledge.called


# ---------------------------------------------------------------------- #
# Subtype + bot-loop filtering
# ---------------------------------------------------------------------- #
class TestEventFiltering:
    @pytest.fixture(autouse=True)
    def _reset(self):
        import realtime_ingest as r

        r._seen_event_ids.clear()
        r._bot_user_id_by_token.clear()
        r._in_flight.clear()
        yield

    @pytest.mark.parametrize(
        "subtype",
        [
            "channel_join",
            "channel_leave",
        ],
    )
    def test_ignored_subtypes_short_circuit_before_lookup(self, subtype):
        """channel_join/channel_leave are noise and dropped BEFORE we hit
        Supabase. bot_message is handled later (after channel selection) so
        the per-channel include_bot_messages flag can be checked."""
        from realtime_ingest import process_slack_event

        payload = {
            "type": "event_callback",
            "event_id": f"Ev_{subtype}",
            "team_id": "T_TEAM",
            "event": {
                "type": "message",
                "subtype": subtype,
                "channel": "C1",
                "text": "x",
                "ts": "1700000000.000100",
            },
        }
        with patch(
            "realtime_ingest.get_slack_installation_by_team_id",
        ) as mock_lookup:
            process_slack_event(payload)
        mock_lookup.assert_not_called()

    def test_non_message_event_ignored(self):
        from realtime_ingest import process_slack_event

        payload = {
            "type": "event_callback",
            "event_id": "Ev_reaction",
            "team_id": "T_TEAM",
            "event": {"type": "reaction_added", "channel": "C1"},
        }
        with patch(
            "realtime_ingest.get_slack_installation_by_team_id",
        ) as mock_lookup:
            process_slack_event(payload)
        mock_lookup.assert_not_called()


# ---------------------------------------------------------------------- #
# team_id resolution variants
# ---------------------------------------------------------------------- #
class TestTeamIdResolution:
    def test_prefers_authorizations_team_id(self):
        """Slack-recommended path: read team_id from authorizations[0]."""
        from realtime_ingest import _resolve_team_id

        payload = {
            "team_id": "T_OUTER",
            "authorizations": [{"team_id": "T_AUTHED"}],
            "event": {"team": "T_EVENT"},
        }
        assert _resolve_team_id(payload) == "T_AUTHED"

    def test_falls_back_to_top_level_team_id(self):
        from realtime_ingest import _resolve_team_id

        payload = {
            "team_id": "T_OUTER",
            "event": {"team": "T_EVENT"},
        }
        assert _resolve_team_id(payload) == "T_OUTER"

    def test_falls_back_to_event_team(self):
        from realtime_ingest import _resolve_team_id

        payload = {"event": {"team": "T_EVENT"}}
        assert _resolve_team_id(payload) == "T_EVENT"

    def test_returns_empty_when_nothing_present(self):
        from realtime_ingest import _resolve_team_id

        assert _resolve_team_id({}) == ""
        assert _resolve_team_id({"event": {}}) == ""


# ---------------------------------------------------------------------- #
# Realtime disabled
# ---------------------------------------------------------------------- #
class TestRealtimeDisabled:
    def test_short_circuits_when_disabled(self, monkeypatch):
        from realtime_ingest import process_slack_event

        monkeypatch.setenv("REALTIME_INGEST_ENABLED", "false")
        with patch(
            "realtime_ingest.get_slack_installation_by_team_id",
        ) as mock_lookup:
            process_slack_event(
                {
                    "type": "event_callback",
                    "team_id": "T_TEAM",
                    "event": {"type": "message", "channel": "C1", "text": "x"},
                }
            )
        mock_lookup.assert_not_called()


# ---------------------------------------------------------------------- #
# supabase_client helpers
# ---------------------------------------------------------------------- #
class TestSupabaseHelpers:
    def test_get_slack_installation_by_team_id_returns_row(self):
        from supabase_client import get_slack_installation_by_team_id

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {
                "workspace_id": "ws-1",
                "slack_team_id": "T1",
                "bot_token": "xoxb-x",
            }
        ]
        with patch("supabase_client.get_supabase", return_value=mock_client):
            row = get_slack_installation_by_team_id(slack_team_id="T1")
        assert row is not None
        assert row["workspace_id"] == "ws-1"

    def test_get_slack_installation_by_team_id_returns_none_when_missing(self):
        from supabase_client import get_slack_installation_by_team_id

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = (
            []
        )
        with patch("supabase_client.get_supabase", return_value=mock_client):
            row = get_slack_installation_by_team_id(slack_team_id="T_UNKNOWN")
        assert row is None

    def test_get_slack_installation_by_team_id_handles_blank(self):
        from supabase_client import get_slack_installation_by_team_id

        # No round-trip when the input is blank.
        with patch("supabase_client.get_supabase") as mock_get:
            assert get_slack_installation_by_team_id(slack_team_id="") is None
        mock_get.assert_not_called()

    def test_is_channel_selected_true_when_selected(self):
        from supabase_client import is_channel_selected_for_workspace

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {"is_selected": True}
        ]
        with patch("supabase_client.get_supabase", return_value=mock_client):
            ok = is_channel_selected_for_workspace(
                workspace_id="ws-1",
                slack_channel_id="C1",
            )
        assert ok is True

    def test_is_channel_selected_false_when_unselected(self):
        from supabase_client import is_channel_selected_for_workspace

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {"is_selected": False}
        ]
        with patch("supabase_client.get_supabase", return_value=mock_client):
            ok = is_channel_selected_for_workspace(
                workspace_id="ws-1",
                slack_channel_id="C1",
            )
        assert ok is False

    def test_is_channel_selected_false_when_no_row(self):
        from supabase_client import is_channel_selected_for_workspace

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = (
            []
        )
        with patch("supabase_client.get_supabase", return_value=mock_client):
            ok = is_channel_selected_for_workspace(
                workspace_id="ws-1",
                slack_channel_id="C_NEW",
            )
        assert ok is False
