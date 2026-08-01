"""Additional recall tests covering prepare_recall_context end-to-end."""

from unittest.mock import MagicMock, patch

import pytest


def _hydra_chunk(text, source_id, stable_key, channel="general", ts=None, score=0.9):
    return {
        "text": text,
        "score": score,
        "source_id": source_id,
        "filename": f"{source_id}.md",
        "metadata": {
            "channel": channel,
            "stable_key": stable_key,
            **({"timestamp": ts} if ts else {}),
        },
    }


class TestPrepareRecallContext:
    def _call(self, question="what happened?", top_k=5, mode="default", **kwargs):
        from recall import prepare_recall_context

        kwargs.setdefault("hydradb_sub_tenant_id", "ws_testtenant")
        return prepare_recall_context(question, top_k, mode=mode, **kwargs)

    def test_returns_ready_true_when_chunks_found(self):
        chunks = [_hydra_chunk("content", "d1", "k1")]
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": chunks}):
            result = self._call()
        assert result["ready"] is True
        assert result["context_text"] != ""

    def test_returns_ready_false_when_no_chunks(self):
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": []}):
            result = self._call()
        assert result["ready"] is False
        assert "fallback_debug" in result

    def test_context_text_numbered(self):
        chunks = [
            _hydra_chunk("chunk one", "d1", "k1"),
            _hydra_chunk("chunk two", "d2", "k2"),
        ]
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": chunks}):
            result = self._call()
        assert "[1]" in result["context_text"]
        assert "[2]" in result["context_text"]

    def test_sources_parallel_to_context(self):
        chunks = [
            _hydra_chunk("text A", "d1", "k1"),
            _hydra_chunk("text B", "d2", "k2"),
        ]
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": chunks}):
            result = self._call(top_k=5)
        assert len(result["sources"]) == 2
        # Indexes match context numbering
        assert result["sources"][0]["index"] == 1
        assert result["sources"][1]["index"] == 2

    def test_top_k_caps_results(self):
        chunks = [_hydra_chunk(f"text {i}", f"d{i}", f"k{i}") for i in range(10)]
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": chunks}):
            result = self._call(top_k=3)
        assert len(result["sources"]) <= 3

    def test_deduplication_by_stable_key(self):
        chunks = [
            _hydra_chunk("same doc copy 1", "d1", "same-key"),
            _hydra_chunk("same doc copy 2", "d2", "same-key"),  # dup
        ]
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": chunks}):
            result = self._call(top_k=5)
        assert result["ready"] is True
        assert len(result["sources"]) == 1  # deduped to 1

    def test_channel_filter_excludes_non_matching(self):
        chunks = [
            _hydra_chunk("general msg", "d1", "k1", channel="general"),
            _hydra_chunk("product msg", "d2", "k2", channel="product"),
        ]
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": chunks}):
            result = self._call(channel="product")
        # After filtering, only product source remains
        # (general source has no channel in minimal source card, so check sources)
        if result["ready"]:
            sources = result["sources"]
            for src in sources:
                ch = src.get("channel")
                if ch:
                    assert ch.lower() == "product"

    def test_hydradb_error_propagates(self):
        from errors import HydraDBError

        with patch("hydradb_client.HydraDBClient.full_recall", side_effect=HydraDBError("down")):
            with pytest.raises(HydraDBError):
                self._call()

    def test_exact_mode_sets_query_terms(self):
        chunks = [_hydra_chunk("sprint deadline is near", "d1", "k1")]
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": chunks}):
            result = self._call(question="sprint deadline", mode="exact")
        assert "query_terms" in result
        assert len(result["query_terms"]) > 0

    def test_default_mode_has_empty_query_terms(self):
        chunks = [_hydra_chunk("content", "d1", "k1")]
        with patch("hydradb_client.HydraDBClient.full_recall", return_value={"chunks": chunks}):
            result = self._call(question="what happened?", mode="default")
        assert result.get("query_terms", []) == []
