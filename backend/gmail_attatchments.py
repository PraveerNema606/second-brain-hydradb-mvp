"""
Gmail attachment text extraction (Phase C).

Pure, network-free, dependency-isolated text extraction for the
attachment types Second Brain indexes:

    .pdf   -> pypdf (optional dependency; gracefully skipped if absent)
    .docx  -> stdlib zipfile + XML text-run extraction (no dependency)
    .csv   -> UTF-8 decode (kept as readable rows)
    .txt   -> UTF-8 decode

Design notes:
  - This module knows NOTHING about Gmail, HTTP, or HydraDB. It takes
    bytes + a filename/mime and returns text (or None). That keeps it
    trivially unit-testable and free of import cycles -- gmail_oauth.py
    handles the network (fetching attachment bytes) and calls in here
    for the pure extraction step.
  - EVERY extractor is defensive: a corrupt/encrypted/unsupported input
    returns None instead of raising, so a single bad attachment can
    never fail the surrounding Gmail ingest run.
  - Privacy: nothing here logs filenames, attachment bytes, or extracted
    text. Callers log only counts / mime types / sizes.
"""

from __future__ import annotations

import html
import io
import os
import re
import zipfile
from typing import Optional

from logging_config import get_logger

logger = get_logger(__name__)


# Supported types -- by file extension first (most reliable for Gmail,
# which often sends generic "application/octet-stream" for attachments),
# with a mime-type fallback.
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".csv", ".txt")

_MIME_TO_KIND = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/csv": "csv",
    "text/comma-separated-values": "csv",
    "application/csv": "csv",
    "text/plain": "txt",
}

_EXT_TO_KIND = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".csv": "csv",
    ".txt": "txt",
}


def _ext(filename: str) -> str:
    """Lower-cased extension including the dot, or '' if none."""
    if not filename:
        return ""
    _, dot, tail = filename.rpartition(".")
    return ("." + tail.lower()) if dot else ""


def attachment_kind(filename: str, mime_type: Optional[str]) -> Optional[str]:
    """
    Classify an attachment as one of "pdf" / "docx" / "csv" / "txt", or
    None when it isn't a supported type.

    Extension wins (Gmail frequently mislabels attachment mime types);
    mime type is the fallback when the filename has no useful extension.
    """
    kind = _EXT_TO_KIND.get(_ext(filename or ""))
    if kind:
        return kind
    if mime_type:
        return _MIME_TO_KIND.get(mime_type.strip().lower())
    return None


def is_supported_attachment(filename: str, mime_type: Optional[str]) -> bool:
    """True iff this attachment is a type we can extract text from."""
    return attachment_kind(filename, mime_type) is not None


# ---------------------------------------------------------------------- #
# Per-type extractors. Each returns text or None; never raises.
# ---------------------------------------------------------------------- #
_WS_RUN_RE = re.compile(r"[ \t\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def _normalize(text: str) -> str:
    """Collapse intra-line whitespace, trim lines, cap blank-line runs."""
    if not text:
        return ""
    lines = [_WS_RUN_RE.sub(" ", ln).strip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out = "\n".join(lines)
    out = _MULTI_NL_RE.sub("\n\n", out)
    return out.strip()


def _extract_txt(raw: bytes) -> Optional[str]:
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    text = _normalize(text)
    return text or None


def _extract_csv(raw: bytes) -> Optional[str]:
    # CSV is already human-readable text; decode and normalize lightly so
    # rows stay on their own lines for retrieval. We intentionally do NOT
    # reflow commas -- the raw rows carry the most signal.
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    text = _normalize(text)
    return text or None


_DOCX_PARA_RE = re.compile(r"<w:p\b[^>]*>", re.IGNORECASE)
_DOCX_BR_RE = re.compile(r"<w:br\b[^>]*/?>", re.IGNORECASE)
_DOCX_TAB_RE = re.compile(r"<w:tab\b[^>]*/?>", re.IGNORECASE)
_XML_TAG_RE = re.compile(r"<[^>]+>")


def _extract_docx(raw: bytes) -> Optional[str]:
    """
    Extract text from a .docx (Office Open XML) using only the stdlib.

    A .docx is a zip; the body lives in word/document.xml. We convert
    paragraph / line-break / tab elements to whitespace, drop every
    remaining tag, and unescape XML entities. The text content of a docx
    lives in <w:t> nodes, so stripping all tags leaves exactly the words
    (with the paragraph newlines we injected first).
    """
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            if "word/document.xml" not in zf.namelist():
                return None
            xml = zf.read("word/document.xml").decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, KeyError, OSError, ValueError):
        return None
    except Exception:  # noqa: BLE001
        return None

    xml = _DOCX_PARA_RE.sub("\n", xml)
    xml = _DOCX_BR_RE.sub("\n", xml)
    xml = _DOCX_TAB_RE.sub("\t", xml)
    xml = _XML_TAG_RE.sub("", xml)
    text = html.unescape(xml)
    text = _normalize(text)
    return text or None


def _extract_pdf(raw: bytes) -> Optional[str]:
    """
    Extract text from a PDF using pypdf if available.

    pypdf is an OPTIONAL dependency: if it isn't installed the function
    returns None (the caller skips the attachment and logs it) rather
    than crashing the ingest run. Encrypted or malformed PDFs also
    return None.
    """
    try:
        import pypdf  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        logger.warning("gmail_attachment_pdf_lib_missing")
        return None

    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        if getattr(reader, "is_encrypted", False):
            # Try an empty-password decrypt; bail quietly if it fails.
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                return None
        pages_out = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                page_text = ""
            if page_text:
                pages_out.append(page_text)
        text = _normalize("\n".join(pages_out))
        return text or None
    except Exception:  # noqa: BLE001
        return None


_EXTRACTORS = {
    "pdf": _extract_pdf,
    "docx": _extract_docx,
    "csv": _extract_csv,
    "txt": _extract_txt,
}


def extract_text_from_attachment(
    filename: str,
    mime_type: Optional[str],
    raw_bytes: bytes,
) -> Optional[str]:
    """
    Extract readable text from an attachment's raw bytes.

    Returns the extracted text, or None when the type is unsupported,
    the bytes are empty/corrupt, or extraction yields nothing. NEVER
    raises -- the caller treats None as "skip this attachment".
    """
    if not raw_bytes:
        return None
    kind = attachment_kind(filename, mime_type)
    if kind is None:
        return None
    extractor = _EXTRACTORS.get(kind)
    if extractor is None:
        return None
    try:
        return extractor(raw_bytes)
    except Exception:  # noqa: BLE001
        # Belt-and-suspenders: individual extractors are already
        # defensive, but a single bad attachment must never bubble up.
        return None


# Env-tunable wrapper kept here so the size policy lives next to the
# extractors. The caller (gmail_oauth) enforces it before fetching bytes.
def max_attachment_bytes() -> int:
    """Hard cap on attachment size we will download + extract (bytes)."""
    try:
        return max(1, int(os.getenv("GMAIL_ATTACHMENT_MAX_BYTES", "5000000")))
    except ValueError:
        return 5_000_000