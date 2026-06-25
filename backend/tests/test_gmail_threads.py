"""
Gmail thread-aware ingestion (Phase D).

Covers:
  - stable_key_for_gmail_thread / build_email_thread_document (single +
    multi-message, ordering, label union, latest-message header,
    attachment embedding, caps, empty -> None).
  - list_history_message_ids surfacing added/deleted THREAD ids.
  - _materialize_gmail_thread: present thread -> consolidated upload +
    memory under the thread key + legacy per-message migration; gone
    thread -> consolidated doc + memories deleted.
  - Runner full + incremental paths group by thread and dedupe.
  - Deletion scenarios (whole thread gone, one reply removed).
  - Ranking / source-card compatibility (recall classifies the thread
    doc as a gmail "email" with unioned labels).
  - Memory extraction compatibility (one coherent unit per thread).
"""

import base64
from unittest.mock import MagicMock, patch

import gmail_oauth as go


def _b64(s: bytes) -> str:
    return base64.urlsafe_b64encode(s).rstrip(b"=").decode("ascii")


def _msg(mid, thread_id, *, subject="Pricing", sender="A <a@x.com>", body="hello", ts="1700000000000", labels=None):
    return {
        "id": mid,
        "threadId": thread_id,
        "internalDate": ts,
        "snippet": body[:50],
        "labelIds": labels if labels is not None else ["INBOX"],
        "payload": {
            "mimeType": "text/plain",
            "body": {"data": _b64(body.encode())},
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
                {"name": "Date", "value": "Mon, 1 Jan 2024 00:00:00 +0000"},
            ],
        },
    }


# =====================================================================
# stable key
# =====================================================================
def test_stable_key_for_gmail_thread():
    assert go.stable_key_for_gmail_thread("abc123") == "gmail:thread:abc123"


# =====================================================================
# build_email_thread_document
# =====================================================================
class TestBuildThreadDocument:
    def test_single_message_thread(self):
        doc = go.build_email_thread_document([_msg("m1", "t1")], "owner@x.com")
        assert doc is not None
        assert doc["stable_key"] == "gmail:thread:t1"
        assert doc["document_type"] == "email"
        assert doc["content"].splitlines()[0] == "# Email"
        assert doc["message_count"] == 1
        assert "## Message from" in doc["content"]
        assert "hello" in doc["content"]

    def test_multi_message_ordering_and_header_from_latest(self):
        early = _msg("m1", "t1", subject="Pricing", sender="A <a@x.com>", body="opening offer", ts="1700000000000")
        late = _msg("m2", "t1", subject="Re: Pricing", sender="B <b@x.com>", body="final agreement", ts="1700000900000")
        # Pass out of order; builder sorts chronologically.
        doc = go.build_email_thread_document([late, early], "owner@x.com")
        assert doc["message_count"] == 2
        body = doc["content"]
        # Header (Subject/From) reflects the LATEST message.
        assert "Subject: Re: Pricing" in body
        assert "From: B <b@x.com>" in body
        # Both messages present; the earlier sender's block precedes the
        # later one (compare the block headers, since the latest snippet
        # also appears in the Snippet: header line).
        assert body.index("## Message from A <a@x.com>") < body.index("## Message from B <b@x.com>")
        assert "opening offer" in body and "final agreement" in body
        # Latest message id is recorded; recency timestamp = latest.
        assert doc["message_id"] == "m2"
        assert doc["timestamp"] == 1700000900.0

    def test_label_union_across_thread(self):
        m1 = _msg("m1", "t1", labels=["INBOX"])
        m2 = _msg("m2", "t1", labels=["IMPORTANT", "INBOX"])
        doc = go.build_email_thread_document([m1, m2], "owner@x.com")
        # Labels header is the sorted union.
        labels_line = [ln for ln in doc["content"].splitlines() if ln.startswith("Labels:")][0]
        assert "IMPORTANT" in labels_line
        assert "INBOX" in labels_line

    def test_attachment_embedding(self):
        m1 = _msg("m1", "t1", body="see attached")
        atts = {"m1": [{"filename": "deck.pdf", "mime_type": "application/pdf", "size": 10, "chars": 9, "text": "SLIDE TEXT"}]}
        doc = go.build_email_thread_document([m1], "owner@x.com", message_attachments=atts)
        assert "### Attachment: deck.pdf" in doc["content"]
        assert "SLIDE TEXT" in doc["content"]
        assert doc["attachments"][0]["filename"] == "deck.pdf"
        assert "text" not in doc["attachments"][0]

    def test_empty_thread_returns_none(self):
        assert go.build_email_thread_document([], "owner@x.com") is None
        assert go.build_email_thread_document([{"foo": "bar"}], "owner@x.com") is None

    def test_total_cap_enforced(self, monkeypatch):
        monkeypatch.setenv("GMAIL_THREAD_MAX_CHARS", "100")
        big = "x" * 5000
        msgs = [_msg(f"m{i}", "t1", body=big, ts=str(1700000000000 + i)) for i in range(5)]
        doc = go.build_email_thread_document(msgs, "owner@x.com")
        # Body text (excluding header) is bounded by the cap.
        body_only = doc["content"].split("Permalink:", 1)[1]
        assert len(body_only) < 100 + 2000  # cap + one floor block, generous


# =====================================================================
# list_history_message_ids -> thread ids
# =====================================================================
class TestHistoryThreadIds:
    def test_added_and_deleted_thread_ids(self):
        history_page = {
            "historyId": "200",
            "history": [
                {"messagesAdded": [{"message": {"id": "m1", "threadId": "tA"}}]},
                {"messagesDeleted": [{"message": {"id": "m2", "threadId": "tB"}}]},
            ],
        }
        with patch.object(go, "_authed_request", return_value=(history_page, {})):
            out = go.list_history_message_ids({"access_token": "x"}, start_history_id="100", label_id="INBOX")
        assert out["added_thread_ids"] == ["tA"]
        assert out["deleted_thread_ids"] == ["tB"]
        assert out["message_ids"] == ["m1"]
        assert out["deleted_message_ids"] == ["m2"]


# =====================================================================
# _materialize_gmail_thread
# =====================================================================
class TestMaterializeThread:
    def _hydra(self):
        h = MagicMock()
        h.upload_knowledge.return_value = {"success": True, "success_count": 1, "failed_count": 0}
        return h

    def test_present_thread_uploads_and_migrates(self):
        hydra = self._hydra()
        summary = _fresh_summary()
        thread = {"messages": [_msg("m1", "t1"), _msg("m2", "t1", ts="1700000900000")]}
        with patch.object(go, "fetch_thread", return_value=thread), patch(
            "memory_store.extract_and_persist"
        ) as mock_extract, patch("memory_store.delete_memories_by_source") as mock_del_mem:
            go._materialize_gmail_thread(
                {"access_token": "x"},
                "t1",
                hydra=hydra,
                workspace_id="ws-1",
                connection_id="c1",
                connection_email="owner@x.com",
                summary=summary,
            )
        # Consolidated doc uploaded once.
        hydra.upload_knowledge.assert_called_once()
        uploaded = hydra.upload_knowledge.call_args.args[0][0]
        assert uploaded["stable_key"] == "gmail:thread:t1"
        # Memory extracted under the THREAD key.
        assert mock_extract.call_args.kwargs["source_stable_key"] == "gmail:thread:t1"
        # Lazy migration: legacy per-message docs deleted.
        deleted_keys = hydra.delete_knowledge.call_args.args[0]
        assert "gmail:msg:m1" in deleted_keys and "gmail:msg:m2" in deleted_keys
        assert summary["threads_uploaded"] == 1
        assert summary["messages_uploaded"] == 2

    def test_gone_thread_deletes_consolidated_doc(self):
        hydra = MagicMock()
        summary = _fresh_summary()
        with patch.object(go, "fetch_thread", return_value=None), patch(
            "memory_store.delete_memories_by_source"
        ) as mock_del_mem:
            go._materialize_gmail_thread(
                {"access_token": "x"},
                "t1",
                hydra=hydra,
                workspace_id="ws-1",
                connection_id="c1",
                connection_email="owner@x.com",
                summary=summary,
            )
        hydra.delete_knowledge.assert_called_once_with(["gmail:thread:t1"])
        mock_del_mem.assert_called_once()
        assert summary["threads_deleted"] == 1
        hydra.upload_knowledge.assert_not_called()


# =====================================================================
# Runner: full + incremental thread flows
# =====================================================================
class TestRunnerThreads:
    def _conn(self):
        return {"id": "c1", "workspace_id": "ws-1", "email": "owner@x.com", "access_token": "at", "refresh_token": "rt"}

    def test_full_sync_groups_into_threads(self):
        hydra = MagicMock()
        hydra.upload_knowledge.return_value = {"success": True, "success_count": 1, "failed_count": 0}
        threads = {
            "t1": {"messages": [_msg("m1", "t1"), _msg("m2", "t1", ts="1700000900000")]},
            "t2": {"messages": [_msg("m3", "t2")]},
        }
        with patch("gmail_oauth.list_thread_ids_for_label", return_value=["t1", "t2"]), patch(
            "gmail_oauth.fetch_thread", side_effect=lambda conn, tid: threads[tid]
        ), patch("hydradb_client.HydraDBClient", return_value=hydra), patch(
            "supabase_client.upsert_gmail_ingestion_state", return_value=True
        ), patch("memory_store.extract_and_persist"), patch("memory_store.delete_memories_by_source"):
            stats = go.run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._conn(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
                sync_mode="full",
            )
        assert stats["threads_uploaded"] == 2
        # 2 threads, 3 messages total.
        assert stats["messages_uploaded"] == 3
        assert hydra.upload_knowledge.call_count == 2

    def test_incremental_added_and_deleted_threads(self):
        hydra = MagicMock()
        hydra.upload_knowledge.return_value = {"success": True, "success_count": 1, "failed_count": 0}
        hist = {
            "message_ids": ["m1"],
            "deleted_message_ids": ["m9"],
            "added_thread_ids": ["tA"],
            "deleted_thread_ids": ["tGONE"],
            "next_history_id": "1000",
            "invalidated": False,
        }

        def _fetch(conn, tid):
            if tid == "tA":
                return {"messages": [_msg("m1", "tA")]}
            return None  # tGONE is gone

        with patch("gmail_oauth.list_history_message_ids", return_value=hist), patch(
            "gmail_oauth.fetch_thread", side_effect=_fetch
        ), patch("hydradb_client.HydraDBClient", return_value=hydra), patch(
            "supabase_client.get_gmail_ingestion_state_map",
            return_value={"INBOX": {"last_history_id": "999"}},
        ), patch("supabase_client.upsert_gmail_ingestion_state", return_value=True), patch(
            "memory_store.extract_and_persist"
        ), patch("memory_store.delete_memories_by_source"):
            stats = go.run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._conn(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
                sync_mode="auto",
            )
        # tA materialized (uploaded), tGONE deleted.
        assert stats["threads_uploaded"] == 1
        assert stats["threads_deleted"] == 1
        # Legacy per-message deletion cleanup still ran for m9.
        assert stats["messages_deleted"] >= 1

    def test_thread_update_rebuilds_whole_thread(self):
        """A reply (new message in an existing thread) re-fetches and
        rebuilds the full thread doc -- not a per-message append."""
        hydra = MagicMock()
        hydra.upload_knowledge.return_value = {"success": True, "success_count": 1, "failed_count": 0}
        hist = {
            "message_ids": ["m2"],
            "deleted_message_ids": [],
            "added_thread_ids": ["t1"],
            "deleted_thread_ids": [],
            "next_history_id": "1000",
            "invalidated": False,
        }
        full_thread = {"messages": [_msg("m1", "t1", body="original"), _msg("m2", "t1", body="the reply", ts="1700000900000")]}
        with patch("gmail_oauth.list_history_message_ids", return_value=hist), patch(
            "gmail_oauth.fetch_thread", return_value=full_thread
        ), patch("hydradb_client.HydraDBClient", return_value=hydra), patch(
            "supabase_client.get_gmail_ingestion_state_map",
            return_value={"INBOX": {"last_history_id": "999"}},
        ), patch("supabase_client.upsert_gmail_ingestion_state", return_value=True), patch(
            "memory_store.extract_and_persist"
        ), patch("memory_store.delete_memories_by_source"):
            go.run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._conn(),
                label_ids=["INBOX"],
                hydradb_sub_tenant_id="ws_x",
                sync_mode="auto",
            )
        doc = hydra.upload_knowledge.call_args.args[0][0]
        # Rebuilt doc contains BOTH the original and the reply.
        assert "original" in doc["content"]
        assert "the reply" in doc["content"]
        assert doc["message_count"] == 2

    def test_thread_deduped_across_run(self):
        """A thread reachable under two labels is materialized once."""
        hydra = MagicMock()
        hydra.upload_knowledge.return_value = {"success": True, "success_count": 1, "failed_count": 0}
        with patch("gmail_oauth.list_thread_ids_for_label", return_value=["t1"]), patch(
            "gmail_oauth.fetch_thread", return_value={"messages": [_msg("m1", "t1")]}
        ), patch("hydradb_client.HydraDBClient", return_value=hydra), patch(
            "supabase_client.upsert_gmail_ingestion_state", return_value=True
        ), patch("memory_store.extract_and_persist"), patch("memory_store.delete_memories_by_source"):
            stats = go.run_workspace_gmail_ingest(
                workspace_id="ws-1",
                connection=self._conn(),
                label_ids=["INBOX", "IMPORTANT"],
                hydradb_sub_tenant_id="ws_x",
                sync_mode="full",
            )
        # Only one upload despite two labels listing the same thread.
        assert hydra.upload_knowledge.call_count == 1
        assert stats["threads_uploaded"] == 1


# =====================================================================
# Recall / ranking / source-card compatibility
# =====================================================================
class TestRecallCompatibility:
    def test_thread_doc_harvests_as_gmail_email(self):
        import recall

        m1 = _msg("m1", "t1", subject="Pricing discussion", labels=["INBOX"])
        m2 = _msg("m2", "t1", subject="Re: Pricing discussion", labels=["IMPORTANT"], ts="1700000900000")
        doc = go.build_email_thread_document([m1, m2], "owner@x.com")

        harvested = recall._harvest_gmail_header_fields(doc["content"])
        assert harvested["document_type"] == "email"
        assert harvested["subject"] == "Re: Pricing discussion"
        # Labels parsed into a list; union present.
        assert "IMPORTANT" in harvested["labels"]
        assert "INBOX" in harvested["labels"]
        assert harvested["permalink"].endswith("#all/t1")

    def test_thread_card_classified_as_gmail(self):
        import recall

        card = {"document_type": "email", "stable_key": "gmail:thread:t1", "labels": ["INBOX"]}
        assert recall._extract_source_kind(card) == "gmail"


def _fresh_summary():
    return {
        "messages_fetched": 0,
        "messages_uploaded": 0,
        "messages_failed": 0,
        "messages_skipped": 0,
        "messages_deleted": 0,
        "threads_processed": 0,
        "threads_uploaded": 0,
        "threads_failed": 0,
        "threads_skipped": 0,
        "threads_deleted": 0,
        "attachments_processed": 0,
        "attachments_failed": 0,
        "attachments_skipped": 0,
    }