"""
HydraDBClient.delete_knowledge wrapper (supports audit item 2).

The method POSTs to /ingestion/delete_knowledge with the tenant /
sub-tenant and a list of stable source keys, and is best-effort: it
returns {} on any failure and never raises, so a cleanup pass can't
crash an ingest run.
"""

from unittest.mock import MagicMock, patch

import requests


def _client():
    from hydradb_client import HydraDBClient

    # conftest sets HYDRADB_API_KEY / HYDRADB_TENANT_ID.
    return HydraDBClient(sub_tenant_id="ws_test")


class TestDeleteKnowledge:
    def test_empty_keys_skips_call(self):
        from hydradb_client import HydraDBClient  # noqa: F401

        with patch("hydradb_client._post_delete") as mock_post:
            out = _client().delete_knowledge([])
        assert out == {}
        mock_post.assert_not_called()

    def test_blank_keys_skipped(self):
        with patch("hydradb_client._post_delete") as mock_post:
            out = _client().delete_knowledge(["", "   "])
        assert out == {}
        mock_post.assert_not_called()

    def test_success_sends_expected_payload(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"success": True, "deleted_count": 2}
        with patch("hydradb_client._post_delete", return_value=resp) as mock_post:
            out = _client().delete_knowledge(["gmail:msg:a", "gmail:msg:b"])

        assert out == {"success": True, "deleted_count": 2}
        mock_post.assert_called_once()
        _args, kwargs = mock_post.call_args
        # _post_delete(url, headers, payload) — positional.
        url = mock_post.call_args.args[0]
        payload = mock_post.call_args.args[2]
        assert url.endswith("/ingestion/delete_knowledge")
        assert payload["tenant_id"]  # populated from env
        assert payload["sub_tenant_id"] == "ws_test"
        assert payload["source_keys"] == ["gmail:msg:a", "gmail:msg:b"]

    def test_network_error_returns_empty(self):
        with patch(
            "hydradb_client._post_delete",
            side_effect=requests.ConnectionError("dns"),
        ):
            out = _client().delete_knowledge(["gmail:msg:a"])
        assert out == {}

    def test_4xx_returns_empty(self):
        resp = MagicMock(status_code=404)
        with patch("hydradb_client._post_delete", return_value=resp):
            out = _client().delete_knowledge(["gmail:msg:a"])
        assert out == {}

    def test_non_json_returns_empty(self):
        resp = MagicMock(status_code=200)
        resp.json.side_effect = ValueError("not json")
        with patch("hydradb_client._post_delete", return_value=resp):
            out = _client().delete_knowledge(["gmail:msg:a"])
        assert out == {}
