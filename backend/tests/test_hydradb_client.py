"""
Unit tests for hydradb_client.py — HydraDBClient.

All HTTP calls (requests.post) are mocked; no real HydraDB credentials needed.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests as req_lib

from errors import HydraDBError, UpstreamTimeoutError

# ── Fixtures / helpers ─────────────────────────────────────────────────────


def _client():
    from hydradb_client import HydraDBClient

    return HydraDBClient(
        base_url="https://hydra.test",
        api_key="test-key",
        tenant_id="test-tenant",
        sub_tenant_id="test-sub",
    )


def _mock_response(status_code=200, json_data=None, text="", raises=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or str(json_data or "")
    if raises:
        resp.json.side_effect = raises
    else:
        resp.json.return_value = json_data or {}
    return resp


# ── upload_knowledge ───────────────────────────────────────────────────────


class TestUploadKnowledge:
    def _file(self, name="f.md", content="hello"):
        return {"filename": name, "content": content}

    def test_success_returns_payload(self):
        c = _client()
        payload = {
            "success": True,
            "success_count": 1,
            "failed_count": 0,
            "results": [{"filename": "f.md", "status": "queued"}],
        }
        with patch("hydradb_client.requests.post", return_value=_mock_response(200, payload)):
            result = c.upload_knowledge([self._file()])

        assert result["success"] is True
        assert result["success_count"] == 1

    def test_empty_file_list_returns_empty_dict(self):
        c = _client()
        with patch("hydradb_client.requests.post") as mock_post:
            result = c.upload_knowledge([])

        mock_post.assert_not_called()
        assert result == {}

    def test_http_4xx_returns_empty_dict(self):
        c = _client()
        with patch("hydradb_client.requests.post", return_value=_mock_response(400, text="bad request")):
            result = c.upload_knowledge([self._file()])

        assert result == {}

    def test_network_error_returns_empty_dict(self):
        c = _client()
        with patch("hydradb_client.requests.post", side_effect=req_lib.RequestException("conn reset")):
            result = c.upload_knowledge([self._file()])

        assert result == {}

    def test_non_json_response_returns_empty_dict(self):
        c = _client()
        with patch(
            "hydradb_client.requests.post",
            return_value=_mock_response(200, raises=ValueError("not json"), text="not-json"),
        ):
            result = c.upload_knowledge([self._file()])

        assert result == {}

    def test_partial_success_counts_reflected(self):
        c = _client()
        payload = {
            "success": True,
            "results": [
                {"filename": "a.md", "status": "queued"},
                {"filename": "b.md", "status": "failed", "error": "bad doc"},
            ],
        }
        with patch("hydradb_client.requests.post", return_value=_mock_response(200, payload)):
            result = c.upload_knowledge([self._file("a.md"), self._file("b.md")])

        from hydradb_client import summarize_upload_response

        ok, bad = summarize_upload_response(result, batch_size=2)
        assert ok == 1
        assert bad == 1

    def test_bytes_content_uploaded_correctly(self):
        c = _client()
        captured = []

        def _capture(url, headers, data, files, timeout):
            captured.extend(files)
            return _mock_response(200, {"success": True})

        with patch("hydradb_client.requests.post", side_effect=_capture):
            c.upload_knowledge([{"filename": "b.md", "content": b"bytes content"}])

        assert len(captured) == 1
        _, (_, content_bytes, _) = captured[0]
        assert content_bytes == b"bytes content"


# ── full_recall ────────────────────────────────────────────────────────────


class TestFullRecall:
    def test_success_returns_chunks(self):
        c = _client()
        payload = {"chunks": [{"text": "hello", "score": 0.9}]}
        with patch("hydradb_client.requests.post", return_value=_mock_response(200, payload)):
            result = c.full_recall("what happened?")

        assert "chunks" in result
        assert len(result["chunks"]) == 1

    def test_empty_query_raises_hydradb_error(self):
        c = _client()
        with pytest.raises(HydraDBError):
            c.full_recall("")

    def test_whitespace_only_query_raises(self):
        c = _client()
        with pytest.raises(HydraDBError):
            c.full_recall("   ")

    def test_timeout_raises_upstream_timeout_error(self):
        c = _client()
        with patch("hydradb_client.requests.post", side_effect=req_lib.Timeout("timed out")):
            with pytest.raises(UpstreamTimeoutError):
                c.full_recall("anything")

    def test_network_error_raises_hydradb_error(self):
        c = _client()
        with patch("hydradb_client.requests.post", side_effect=req_lib.RequestException("connection refused")):
            with pytest.raises(HydraDBError):
                c.full_recall("anything")

    def test_http_4xx_raises_hydradb_error(self):
        c = _client()
        with patch("hydradb_client.requests.post", return_value=_mock_response(403, text='{"detail":"forbidden"}')):
            with pytest.raises(HydraDBError):
                c.full_recall("anything")

    def test_http_403_error_carries_status(self):
        c = _client()
        with patch("hydradb_client.requests.post", return_value=_mock_response(403, text="forbidden")):
            with pytest.raises(HydraDBError) as exc_info:
                c.full_recall("anything")
        assert "403" in str(exc_info.value.detail)

    def test_non_json_response_raises_hydradb_error(self):
        c = _client()
        with patch(
            "hydradb_client.requests.post",
            return_value=_mock_response(200, raises=ValueError("bad json"), text="not-json"),
        ):
            with pytest.raises(HydraDBError):
                c.full_recall("anything")

    def test_top_k_forwarded(self):
        c = _client()
        captured_payload = {}

        def _capture(url, headers, json, timeout):
            captured_payload.update(json)
            return _mock_response(200, {"chunks": []})

        with patch("hydradb_client.requests.post", side_effect=_capture):
            c.full_recall("test query", top_k=7)

        from hydradb_client import RECALL_TOP_K_FIELD

        assert captured_payload.get(RECALL_TOP_K_FIELD) == 7


# ── summarize_upload_response ──────────────────────────────────────────────


class TestSummarizeUploadResponse:
    def test_empty_payload_all_fail(self):
        from hydradb_client import summarize_upload_response

        ok, bad = summarize_upload_response({}, batch_size=3)
        assert ok == 0
        assert bad == 3

    def test_success_false_all_fail(self):
        from hydradb_client import summarize_upload_response

        ok, bad = summarize_upload_response({"success": False}, batch_size=2)
        assert ok == 0
        assert bad == 2

    def test_uses_success_count_fields(self):
        from hydradb_client import summarize_upload_response

        ok, bad = summarize_upload_response({"success_count": 3, "failed_count": 1}, batch_size=4)
        assert ok == 3
        assert bad == 1

    def test_counts_from_results_list(self):
        from hydradb_client import summarize_upload_response

        payload = {
            "results": [
                {"status": "queued"},
                {"status": "failed"},
                {"status": "error", "error": "oops"},
                {"status": "queued"},
            ]
        }
        ok, bad = summarize_upload_response(payload, batch_size=4)
        assert ok == 2
        assert bad == 2

    def test_no_counts_assumes_all_ok(self):
        from hydradb_client import summarize_upload_response

        ok, bad = summarize_upload_response({"something": "else"}, batch_size=5)
        assert ok == 5
        assert bad == 0

    def test_result_with_non_null_error_field_counts_as_failure(self):
        """Line 73: _result_is_failed returns True when error field is non-empty."""
        from hydradb_client import summarize_upload_response

        payload = {
            "results": [
                {"status": "queued", "error": "ingestion failed: bad encoding"},
            ]
        }
        ok, bad = summarize_upload_response(payload, batch_size=1)
        assert ok == 0
        assert bad == 1


# ── HydraDBClient constructor guards ──────────────────────────────────────


class TestHydraDBClientInit:
    def test_raises_when_api_key_missing(self, monkeypatch):
        """Line 131: ValueError raised when HYDRADB_API_KEY is absent."""
        monkeypatch.delenv("HYDRADB_API_KEY", raising=False)
        from hydradb_client import HydraDBClient

        with pytest.raises(ValueError, match="HYDRADB_API_KEY"):
            HydraDBClient(
                base_url="https://hydra.test",
                api_key=None,
                tenant_id="tenant",
                sub_tenant_id="test-sub",
            )

    def test_raises_when_tenant_id_missing(self, monkeypatch):
        """Line 133: ValueError raised when HYDRADB_TENANT_ID is absent."""
        monkeypatch.delenv("HYDRADB_TENANT_ID", raising=False)
        from hydradb_client import HydraDBClient

        with pytest.raises(ValueError, match="HYDRADB_TENANT_ID"):
            HydraDBClient(
                base_url="https://hydra.test",
                api_key="some-key",
                tenant_id=None,
                sub_tenant_id="test-sub",
            )

    def test_raises_when_sub_tenant_missing(self, monkeypatch):
        """Fail closed: no env / hard-coded default sub-tenant."""
        monkeypatch.delenv("HYDRADB_SUB_TENANT_ID", raising=False)
        from errors import WorkspaceTenantError
        from hydradb_client import HydraDBClient

        with pytest.raises(WorkspaceTenantError):
            HydraDBClient(
                base_url="https://hydra.test",
                api_key="some-key",
                tenant_id="tenant",
                sub_tenant_id=None,
            )
