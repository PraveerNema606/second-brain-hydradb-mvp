"""
Gmail attachment ingestion (Phase C).

Covers:
  - Pure extraction for TXT / CSV / DOCX / PDF (gmail_attachments).
  - Unsupported types and corrupt/failed extraction -> None (skipped).
  - _find_part_text ignores attachment parts (a .txt attachment is NOT
    mistaken for the email body).
  - build_email_document embeds attachment sections into the SAME email
    document, preserving document_type / stable_key / header contracts.
  - The runner gathers attachments (inline + fetched), counts them, keeps
    the message on extraction failure, and respects the per-email cap.

All network + heavy deps are mocked where needed; DOCX uses a real
in-memory zip; PDF success is exercised via a patched pypdf reader and
PDF failure via garbage bytes.
"""

import base64
import io
import zipfile
from unittest.mock import MagicMock, patch

import gmail_attachments as ga


# =====================================================================
# Type classification
# =====================================================================
class TestSupportedTypes:
    def test_kind_by_extension(self):
        assert ga.attachment_kind("report.pdf", "application/octet-stream") == "pdf"
        assert ga.attachment_kind("doc.DOCX", None) == "docx"
        assert ga.attachment_kind("data.csv", None) == "csv"
        assert ga.attachment_kind("notes.txt", None) == "txt"

    def test_kind_by_mime_fallback(self):
        assert ga.attachment_kind("noext", "application/pdf") == "pdf"
        assert ga.attachment_kind("noext", "text/csv") == "csv"

    def test_unsupported(self):
        assert ga.attachment_kind("image.png", "image/png") is None
        assert ga.attachment_kind("archive.zip", "application/zip") is None
        assert ga.is_supported_attachment("logo.png", "image/png") is False
        assert ga.is_supported_attachment("a.pdf", None) is True


# =====================================================================
# Pure extraction
# =====================================================================
class TestExtraction:
    def test_txt(self):
        raw = b"Hello team,\nThe Q3 plan is attached."
        out = ga.extract_text_from_attachment("notes.txt", "text/plain", raw)
        assert "Q3 plan" in out

    def test_csv(self):
        raw = b"region,revenue\nAPAC,120\nEMEA,90\n"
        out = ga.extract_text_from_attachment("report.csv", "text/csv", raw)
        assert "region,revenue" in out
        assert "APAC,120" in out

    def test_docx_real_zip(self):
        # Build a minimal but valid .docx in memory.
        document_xml = (
            '<?xml version="1.0"?>'
            "<w:document xmlns:w=\"x\"><w:body>"
            "<w:p><w:r><w:t>Quarterly proposal for Acme.</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Budget is 50k.</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", document_xml)
        out = ga.extract_text_from_attachment("proposal.docx", None, buf.getvalue())
        assert "Quarterly proposal for Acme." in out
        assert "Budget is 50k." in out

    def test_pdf_success_via_patched_reader(self):
        class _Page:
            def __init__(self, t):
                self._t = t

            def extract_text(self):
                return self._t

        class _Reader:
            is_encrypted = False

            def __init__(self, *_a, **_k):
                self.pages = [_Page("Invoice total: $1,200"), _Page("Due on receipt")]

        fake_pypdf = MagicMock()
        fake_pypdf.PdfReader = _Reader
        with patch.dict("sys.modules", {"pypdf": fake_pypdf}):
            out = ga.extract_text_from_attachment("invoice.pdf", "application/pdf", b"%PDF-1.4 fake")
        assert "Invoice total: $1,200" in out
        assert "Due on receipt" in out

    def test_pdf_corrupt_returns_none(self):
        # Real pypdf on garbage -> exception -> None (never raises).
        out = ga.extract_text_from_attachment("broken.pdf", "application/pdf", b"not a real pdf")
        assert out is None

    def test_docx_corrupt_returns_none(self):
        out = ga.extract_text_from_attachment("broken.docx", None, b"not a zip")
        assert out is None

    def test_unsupported_returns_none(self):
        assert ga.extract_text_from_attachment("p.png", "image/png", b"\x89PNG") is None

    def test_empty_bytes_returns_none(self):
        assert ga.extract_text_from_attachment("a.txt", "text/plain", b"") is None

    def test_pdf_lib_missing_returns_none(self):
        with patch.dict("sys.modules", {"pypdf": None}):
            out = ga.extract_text_from_attachment("x.pdf", "application/pdf", b"%PDF-1.4")
        assert out is None


# =====================================================================
# _find_part_text ignores attachment parts
# =====================================================================
class TestBodyVsAttachment:
    def _b64(self, s):
        return base64.urlsafe_b64encode(s.encode()).rstrip(b"=").decode("ascii")

    def test_text_plain_attachment_not_used_as_body(self):
        from gmail_oauth import _extract_text_from_payload

        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {"mimeType": "text/plain", "body": {"data": self._b64("real body text")}},
                {
                    "mimeType": "text/plain",
                    "filename": "attached.txt",
                    "body": {"data": self._b64("attachment body should not win")},
                },
            ],
        }
        out = _extract_text_from_payload(payload)
        assert "real body text" in out
        assert "should not win" not in out

    def test_only_attachment_yields_empty_body(self):
        from gmail_oauth import _extract_text_from_payload

        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "text/plain",
                    "filename": "attached.txt",
                    "body": {"data": self._b64("attachment only")},
                },
            ],
        }
        assert _extract_text_from_payload(payload) == ""


# =====================================================================
# build_email_document embedding
# =====================================================================
class TestBuildEmailDocumentAttachments:
    def _msg(self):
        return {
            "id": "msg-att-1",
            "snippet": "see attached",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": base64.urlsafe_b64encode(b"Body here").rstrip(b"=").decode("ascii")},
                "headers": [{"name": "Subject", "value": "Proposal"}],
            },
        }

    def test_attachments_embedded_in_same_doc(self):
        from gmail_oauth import build_email_document

        sections = [
            {"filename": "proposal.pdf", "mime_type": "application/pdf", "size": 1200, "chars": 20, "text": "PDF PROPOSAL TEXT"},
            {"filename": "data.csv", "mime_type": "text/csv", "size": 50, "chars": 10, "text": "a,b\n1,2"},
        ]
        doc = build_email_document(self._msg(), "owner@example.com", attachments=sections)
        assert doc is not None
        # Contracts preserved.
        assert doc["document_type"] == "email"
        assert doc["stable_key"] == "gmail:msg:msg-att-1"
        assert doc["content"].splitlines()[0] == "# Email"
        assert "Subject: Proposal" in doc["content"]
        # Attachment text embedded.
        assert "## Attachments" in doc["content"]
        assert "PDF PROPOSAL TEXT" in doc["content"]
        assert "proposal.pdf" in doc["content"]
        assert "data.csv" in doc["content"]
        # Provenance metadata present (no text leaked into metadata).
        assert len(doc["attachments"]) == 2
        assert doc["attachments"][0]["filename"] == "proposal.pdf"
        assert "text" not in doc["attachments"][0]

    def test_attachment_only_email_still_indexed(self):
        from gmail_oauth import build_email_document

        msg = {
            "id": "msg-att-2",
            "snippet": "",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "text/plain",
                "body": {"data": ""},
                "headers": [{"name": "Subject", "value": "(no body)"}],
            },
        }
        sections = [{"filename": "r.txt", "mime_type": "text/plain", "size": 5, "chars": 11, "text": "only in file"}]
        doc = build_email_document(msg, "owner@example.com", attachments=sections)
        assert doc is not None
        assert "only in file" in doc["content"]

    def test_no_attachments_arg_unchanged(self):
        from gmail_oauth import build_email_document

        doc = build_email_document(self._msg(), "owner@example.com")
        assert doc is not None
        assert "## Attachments" not in doc["content"]
        assert doc["attachments"] == []


# =====================================================================
# Runner integration
# =====================================================================
class TestRunnerAttachmentIngest:
    def _connection(self):
        return {
            "id": "conn-1",
            "workspace_id": "ws-1",
            "email": "owner@example.com",
            "access_token": "at-1",
            "refresh_token": "rt-1",
            "status": "active",
        }

    def _msg_with_attachment(self, mid="m1"):
        # text/plain body + a PDF attachment delivered by attachmentId.
        body = base64.urlsafe_b64encode(b"Email body").rstrip(b"=").decode("ascii")
        return {
            "id": mid,
            "snippet": "hi",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [{"name": "Subject", "value": "With attachment"}],
                "body": {"size": 0},
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": body}},
                    {
                        "mimeType": "application/pdf",
                        "filename": "report.pdf",
                        "body": {"size": 1024, "attachmentId": "att-xyz"},
                    },
                ],
            },
        }

    def test_runner_ingests_attachment_text(self):
        from gmail_oauth import run_workspace_gmail_ingest

        captured = {}
        mock_hydra = MagicMock()

        def _capture_upload(prepared):
            captured["docs"] = prepared
            return {"success": True, "success_count": len(prepared), "failed_count": 0}

        mock_hydra.upload_knowledge.side_effect = _capture_upload

        with patch(
            "gmail_oauth.list_message_ids_for_label",
            return_value=["m1"],
        ), patch(
            "gmail_oauth.fetch_message",
            return_value=self._msg_with_attachment("m1"),
        ), patch(
            "gmail_oauth.fetch_attachment_data",
            return_value=base64.urlsafe_b64encode(b"%PDF").rstrip(b"=").decode("ascii"),
        ), patch(
            "gmail_attachments.extract_text_from_attachment",
            return_value="EXTRACTED PDF CONTENT",
        ), patch(
            "hydradb_client.HydraDBClient",
            return_value=mock_hydra,
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

        assert stats["attachments_processed"] == 1
        assert captured["docs"], "a document should have been uploaded"
        assert "EXTRACTED PDF CONTENT" in captured["docs"][0]["content"]
        assert "## Attachments" in captured["docs"][0]["content"]

    def test_runner_survives_attachment_extraction_failure(self):
        from gmail_oauth import run_workspace_gmail_ingest

        mock_hydra = MagicMock()
        mock_hydra.upload_knowledge.return_value = {"success": True, "success_count": 1, "failed_count": 0}

        with patch(
            "gmail_oauth.list_message_ids_for_label",
            return_value=["m1"],
        ), patch(
            "gmail_oauth.fetch_message",
            return_value=self._msg_with_attachment("m1"),
        ), patch(
            "gmail_oauth.fetch_attachment_data",
            return_value=base64.urlsafe_b64encode(b"garbage").rstrip(b"=").decode("ascii"),
        ), patch(
            "gmail_attachments.extract_text_from_attachment",
            return_value=None,  # extraction fails
        ), patch(
            "hydradb_client.HydraDBClient",
            return_value=mock_hydra,
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

        # Extraction failed but the email itself is still ingested.
        assert stats["attachments_failed"] == 1
        assert stats["attachments_processed"] == 0
        assert stats["messages_uploaded"] == 1

    def test_runner_respects_per_email_cap(self, monkeypatch):
        from gmail_oauth import run_workspace_gmail_ingest

        monkeypatch.setenv("GMAIL_MAX_ATTACHMENTS_PER_EMAIL", "1")

        # Build a message with TWO inline txt attachments.
        def _b64(s):
            return base64.urlsafe_b64encode(s).rstrip(b"=").decode("ascii")

        msg = {
            "id": "m1",
            "snippet": "hi",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [{"name": "Subject", "value": "two files"}],
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64(b"body")}},
                    {"mimeType": "text/plain", "filename": "a.txt", "body": {"data": _b64(b"first file")}},
                    {"mimeType": "text/plain", "filename": "b.txt", "body": {"data": _b64(b"second file")}},
                ],
            },
        }
        mock_hydra = MagicMock()
        mock_hydra.upload_knowledge.return_value = {"success": True, "success_count": 1, "failed_count": 0}

        with patch(
            "gmail_oauth.list_message_ids_for_label",
            return_value=["m1"],
        ), patch(
            "gmail_oauth.fetch_message",
            return_value=msg,
        ), patch(
            "hydradb_client.HydraDBClient",
            return_value=mock_hydra,
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

        assert stats["attachments_processed"] == 1
        assert stats["attachments_skipped"] == 1

    def test_attachment_counters_in_stats_shape(self):
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
        for key in ("attachments_processed", "attachments_failed", "attachments_skipped"):
            assert key in stats