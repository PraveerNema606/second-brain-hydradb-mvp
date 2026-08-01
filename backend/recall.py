"""
Recall + answer orchestration for the Second Brain MVP.

Pipeline:
    user question
        -> HydraDB /recall/full_recall   (semantic retrieval)
        -> extract & number context chunks
        -> cloud LLM with strict grounding prompt
        -> { "answer": ..., "sources": [...], "debug": {...} }

Notes on extraction strategy:
HydraDB's recall response shape varies. We probe an ordered list of
"shallow" text fields first (so we don't accidentally return JSON metadata
when a clean `text` field exists somewhere deeper), and only fall back to a
recursive dict-walk if every documented path comes up empty.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from hydradb_client import HydraDBClient, require_sub_tenant_id
from ingestion.ingestion_state import IngestionState
from llm import generate_grounded_answer
from logging_config import get_logger
from prompts import INSUFFICIENT_CONTEXT_ANSWER

logger = get_logger(__name__)


# Path to the ingestion state file (written by ingest_slack.py). Loaded once
# per request inside answer_question, never re-read between chunks.
_BACKEND_DIR = Path(__file__).resolve().parent
STATE_PATH = _BACKEND_DIR / "data" / "ingestion_state.json"


def _debug_recall_enabled() -> bool:
    """
    Verbose recall debugging is OFF by default to keep demo logs clean and
    avoid leaking Slack content through stdout. Flip DEBUG_RECALL=true in
    the environment to see the raw HydraDB response and first-chunk preview.
    """
    return os.getenv("DEBUG_RECALL", "").strip().lower() in ("1", "true", "yes", "on")


# Where to look for the list of chunks at the top level of the recall response.
CHUNKS_KEYS = ("chunks", "results", "documents", "matches", "items", "data")

# Direct text fields on a chunk object, tried in order. First non-empty wins.
DIRECT_TEXT_KEYS = (
    "text",
    "content",
    "chunk",
    "body",
    "page_content",
    "document",
    "memory",
    "value",
    "data",
    "raw_text",
    "chunk_text",
)

# Dotted paths to try when the direct keys above all miss.
NESTED_TEXT_PATHS = (
    ("payload", "text"),
    ("payload", "content"),
    ("metadata", "text"),
    ("metadata", "content"),
    ("metadata", "chunk_text"),
    ("metadata", "document"),
    ("metadata", "raw_text"),
    ("source", "text"),
    ("source", "content"),
)

# Source-identifier fields on a chunk, tried in order.
DIRECT_SOURCE_KEYS = (
    "source_id",
    "filename",
    "source",
    "doc_id",
    "document_id",
    "id",
    "name",
)
NESTED_SOURCE_PATHS = (
    ("metadata", "filename"),
    ("metadata", "source_id"),
    ("metadata", "source"),
    ("metadata", "channel"),
    ("metadata", "channel_id"),
)

# Score-ish fields.
SCORE_KEYS = ("score", "similarity", "distance", "relevance")

# Field names that look like metadata, not body text. We avoid these when
# falling back to a recursive search.
NON_TEXT_FIELD_NAMES = {
    "id",
    "doc_id",
    "document_id",
    "source_id",
    "filename",
    "source",
    "channel",
    "channel_id",
    "thread_ts",
    "ts",
    "user_id",
    "user",
    "url",
    "permalink",
    "score",
    "similarity",
    "distance",
    "relevance",
    "type",
    "doc_type",
    "name",
    "tenant_id",
    "sub_tenant_id",
    "mime_type",
}

MIN_REAL_TEXT_LEN = 20  # below this, a string probably isn't body content


# ---------------------------------------------------------------------- #
# Generic helpers
# ---------------------------------------------------------------------- #
def _get_path(obj: Any, path) -> Any:
    """Walk a tuple of keys into nested dicts. Return None if any step misses."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _coerce_to_text(value: Any) -> str:
    """
    Turn a candidate value into clean text without dropping legitimate content.

    - str  -> the string itself (stripped)
    - list -> if all items look stringy, join with newlines; otherwise ""
    - dict -> "" (callers handle dicts via recursive search)
    - other -> ""
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
        if parts and len(parts) == sum(1 for x in value if isinstance(x, str)):
            return "\n".join(parts)
        return ""
    return ""


def _recursive_find_text(obj: Any, depth: int = 0) -> str:
    """
    Last-resort walk: descend through a dict/list looking for the longest
    plausible body-text string. Skips fields that are obviously metadata.

    Capped at depth 4 to avoid pathological structures.
    """
    if depth > 4:
        return ""

    best = ""

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key.lower() in NON_TEXT_FIELD_NAMES:
                continue
            if isinstance(value, str):
                candidate = value.strip()
                if len(candidate) >= MIN_REAL_TEXT_LEN and len(candidate) > len(best):
                    best = candidate
            elif isinstance(value, (dict, list)):
                deeper = _recursive_find_text(value, depth + 1)
                if len(deeper) > len(best):
                    best = deeper

    elif isinstance(obj, list):
        for item in obj:
            deeper = _recursive_find_text(item, depth + 1)
            if len(deeper) > len(best):
                best = deeper

    return best


# ---------------------------------------------------------------------- #
# Chunk-level extractors
# ---------------------------------------------------------------------- #
def _extract_chunks(payload: Dict[str, Any]) -> List[Any]:
    """Pull the chunks list out of whatever shape HydraDB returned."""
    if not isinstance(payload, dict):
        return []

    for key in CHUNKS_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return value

    data = payload.get("data")
    if isinstance(data, dict):
        for key in CHUNKS_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return value

    return []


def _chunk_text(chunk: Any) -> str:
    """
    Pull readable body text out of a single chunk.

    Order of preference:
      1. The chunk is itself a string.
      2. A direct top-level text field (text/content/chunk/body/...).
         If that field is a dict, recursively search inside it.
      3. A documented nested path (payload.text, metadata.content, etc.).
      4. Last resort: recursive walk of the whole chunk, picking the
         longest non-metadata string.
    """
    if isinstance(chunk, str):
        return chunk.strip()
    if not isinstance(chunk, dict):
        return ""

    # 1. Direct top-level text-like fields.
    for key in DIRECT_TEXT_KEYS:
        if key not in chunk:
            continue
        value = chunk[key]
        text = _coerce_to_text(value)
        if text:
            return text
        # If the value is a dict, dig into it before moving on.
        if isinstance(value, dict):
            deeper = _recursive_find_text(value)
            if deeper:
                return deeper

    # 2. Documented nested paths.
    for path in NESTED_TEXT_PATHS:
        value = _get_path(chunk, path)
        text = _coerce_to_text(value)
        if text:
            return text

    # 3. Recursive fallback across the whole chunk.
    deeper = _recursive_find_text(chunk)
    if deeper:
        return deeper

    return ""


def _chunk_source(chunk: Any, index: int) -> str:
    """Pull a short source identifier; fall back to 'chunk_N'."""
    if not isinstance(chunk, dict):
        return f"chunk_{index}"

    for key in DIRECT_SOURCE_KEYS:
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for path in NESTED_SOURCE_PATHS:
        value = _get_path(chunk, path)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return f"chunk_{index}"


def _chunk_score(chunk: Any) -> Optional[Any]:
    """Pull a numeric score if available, else None."""
    if not isinstance(chunk, dict):
        return None
    for key in SCORE_KEYS:
        if key in chunk:
            return chunk.get(key)
    meta = chunk.get("metadata")
    if isinstance(meta, dict):
        for key in SCORE_KEYS:
            if key in meta:
                return meta.get(key)
    return None


# ---------------------------------------------------------------------- #
# Debug helpers
# ---------------------------------------------------------------------- #
def _safe_json(obj: Any, limit: int = 8000) -> str:
    """Pretty-print as JSON, truncating to `limit` characters."""
    try:
        text = json.dumps(obj, indent=2, default=str)
    except Exception:
        text = repr(obj)
    if len(text) > limit:
        text = text[:limit] + f"\n... [truncated, {len(text)} chars total]"
    return text


def _first_chunk_preview(chunk: Any, limit: int = 2000) -> str:
    """Compact JSON preview of the first chunk, for the debug payload."""
    return _safe_json(chunk, limit=limit)


# ---------------------------------------------------------------------- #
# Linking recall chunks back to ingestion state
# ---------------------------------------------------------------------- #
# Fields on a chunk that might carry the HydraDB source_id, the stable_key
# we wrote into the markdown, or the filename — any one of these is enough
# to find the rich metadata row in ingestion_state.json.
SOURCE_ID_KEYS = ("source_id", "doc_id", "document_id", "id")
STABLE_KEY_KEYS = ("stable_key", "source_key")
FILENAME_KEYS = ("filename", "file_name", "name")


def _candidate_string(chunk: Dict[str, Any], keys) -> Optional[str]:
    """First non-empty string value found at chunk[k] or chunk['metadata'][k]."""
    for key in keys:
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = chunk.get("metadata")
    if isinstance(meta, dict):
        for key in keys:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _build_source_card(
    chunk: Any,
    index: int,
    score: Any,
    state: Optional[IngestionState],
) -> Dict[str, Any]:
    """
    Build the UI-friendly source object for one recall chunk.

    Resolution order against ingestion state:
      1. by source_id     (set by HydraDB on upload)
      2. by stable_key    (written into the markdown header)
      3. by filename      (the .md filename we sent)

    If state lookup fails, fall back to the previous minimal shape:
        {"index", "source", "score"}
    """
    minimal_source = _chunk_source(chunk, index)

    candidate_source_id = _candidate_string(chunk, SOURCE_ID_KEYS) if isinstance(chunk, dict) else None
    candidate_stable_key = _candidate_string(chunk, STABLE_KEY_KEYS) if isinstance(chunk, dict) else None
    candidate_filename = _candidate_string(chunk, FILENAME_KEYS) if isinstance(chunk, dict) else None

    entry: Optional[Dict[str, Any]] = None
    if state is not None:
        if candidate_source_id:
            entry = state.find_by_source_id(candidate_source_id)
        if entry is None and candidate_stable_key:
            entry = state.get(candidate_stable_key)
        if entry is None and candidate_filename:
            entry = state.find_by_filename(candidate_filename)
        # `minimal_source` might itself be the source_id (current behavior)
        # or the filename. Try it as a last resort before giving up.
        if entry is None and minimal_source:
            entry = state.find_by_source_id(minimal_source) or state.find_by_filename(minimal_source)

    if entry is None:
        # Graceful fallback: keep the old minimal shape so callers can still
        # render something even if state is missing or doesn't know the doc.
        # Also promote stable_key and channel from the raw chunk so that
        # dedupe_by_stable_key and _source_passes_filters work correctly even
        # without a state-file entry.
        #
        # Recency fix: ALSO promote `timestamp` and `document_type` so
        # the recency reranker can rank Slack messages by Slack ts even
        # when no IngestionState entry exists yet. Realtime-ingested
        # messages frequently hit this path because state.json may not
        # have been re-read since the upload finished. We harvest from
        # chunk metadata first (cheap, structured), and fall back to
        # parsing the markdown header that the builder always writes
        # ("Timestamp: <ts>" / "# Slack Message" / "# Slack Thread").
        chunk_channel = _get_path(chunk, ("metadata", "channel")) if isinstance(chunk, dict) else None
        chunk_timestamp = _get_path(chunk, ("metadata", "timestamp")) if isinstance(chunk, dict) else None
        chunk_doc_type = _get_path(chunk, ("metadata", "document_type")) if isinstance(chunk, dict) else None
        # Last-resort: parse the markdown header body if metadata
        # didn't carry the fields. We try Slack first (cheap regex
        # against `# Slack Message` / `# Slack Thread`); if it's
        # actually a Gmail doc, the Slack harvester returns {} and we
        # try Gmail. Each harvester is a no-op on the other format.
        gmail_fields: Dict[str, Any] = {}
        if chunk_timestamp is None or chunk_doc_type is None or not chunk_channel:
            body_text = _chunk_text(chunk) if isinstance(chunk, dict) else ""
            slack_harvested = _harvest_slack_header_fields(body_text)
            if chunk_timestamp is None:
                chunk_timestamp = slack_harvested.get("timestamp")
            if chunk_doc_type is None:
                chunk_doc_type = slack_harvested.get("document_type")
            if not chunk_channel:
                chunk_channel = slack_harvested.get("channel")
            # Gmail harvesting: only meaningful when the doc actually
            # looks like an email. The harvester guards on `# Email`
            # internally so this stays a no-op for Slack chunks.
            if not slack_harvested:
                gmail_fields = _harvest_gmail_header_fields(body_text)
                if chunk_doc_type is None:
                    chunk_doc_type = gmail_fields.get("document_type")
                if chunk_timestamp is None and "timestamp" in gmail_fields:
                    chunk_timestamp = gmail_fields["timestamp"]
        card: Dict[str, Any] = {
            "index": index,
            "source": minimal_source,
            "score": score,
            "stable_key": candidate_stable_key,
            "channel": chunk_channel,
            "timestamp": chunk_timestamp,
            "document_type": chunk_doc_type,
        }
        # Attach Gmail-specific enrichments (sender, subject, labels,
        # permalink) so the rankers can use them as boost signals.
        # Stays on the card only when this chunk really is Gmail.
        for k in ("subject", "from_name", "from_email", "labels", "permalink"):
            if k in gmail_fields:
                card[k] = gmail_fields[k]
        return card

    # Rich source card backed by ingestion state.
    return {
        "index": index,
        "source": entry.get("channel_name") or minimal_source,
        "channel": entry.get("channel_name"),
        "channel_id": entry.get("channel_id"),
        "user": entry.get("user_name"),
        "timestamp": entry.get("timestamp"),
        "snippet": entry.get("snippet"),
        "permalink": entry.get("permalink"),
        "stable_key": entry.get("stable_key"),
        "document_type": entry.get("document_type"),
        "score": score,
    }


# Compiled once at import: the markdown header lines build_message_file
# / build_thread_file always emit. We use these to harvest Slack
# metadata back out of the doc body when chunk metadata is missing.
_SLACK_HEADER_TIMESTAMP_RE = re.compile(
    r"^Timestamp:\s*(\S+)",
    re.MULTILINE | re.IGNORECASE,
)
_SLACK_HEADER_CHANNEL_RE = re.compile(
    r"^Channel:\s*(\S+)",
    re.MULTILINE | re.IGNORECASE,
)
_SLACK_DOC_HEADER_RE = re.compile(
    r"^#\s*Slack\s+(Message|Thread)\b",
    re.MULTILINE | re.IGNORECASE,
)


def _harvest_slack_header_fields(body_text: str) -> Dict[str, Any]:
    """
    Parse the Slack-doc markdown header (always written by the
    ingestion builder) to recover channel + timestamp + document_type
    when chunk metadata didn't carry them. Returns a dict with any
    fields that were found.
    """
    out: Dict[str, Any] = {}
    if not body_text:
        return out
    m = _SLACK_HEADER_TIMESTAMP_RE.search(body_text)
    if m:
        out["timestamp"] = m.group(1)
    m = _SLACK_HEADER_CHANNEL_RE.search(body_text)
    if m:
        out["channel"] = m.group(1)
    m = _SLACK_DOC_HEADER_RE.search(body_text)
    if m:
        out["document_type"] = m.group(1).lower()  # "message" or "thread"
    return out


# ---------------------------------------------------------------------- #
# Gmail markdown-header harvesting (Phase 10)
# ---------------------------------------------------------------------- #
# gmail_oauth.build_email_file emits a fixed header block for every
# ingested email:
#
#   # Email
#   Source Key: gmail:msg:<id>
#   Message-Id: <id>
#   Mailbox: <connection_email>
#   Subject: <subject>
#   From: <sender header>
#   To: <to>
#   Cc: <cc>                  (optional)
#   Date: <RFC 2822 date>
#   Labels: <comma-separated label ids>
#   Snippet: <snippet>
#   Permalink: <url>          (optional)
#
# We mirror the Slack harvester pattern: parse these lines back out so
# the source card always has from_email / from_name / subject / labels
# / timestamp / document_type populated, even when the ingestion-state
# lookup misses. This is what powers the Gmail recency rerank below.
_GMAIL_DOC_HEADER_RE = re.compile(r"^#\s*Email\b", re.MULTILINE | re.IGNORECASE)
_GMAIL_SUBJECT_RE = re.compile(r"^Subject:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_GMAIL_FROM_RE = re.compile(r"^From:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_GMAIL_DATE_RE = re.compile(r"^Date:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_GMAIL_LABELS_RE = re.compile(r"^Labels:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_GMAIL_PERMALINK_RE = re.compile(r"^Permalink:\s*(\S+)", re.MULTILINE | re.IGNORECASE)

# Split "Display Name <user@example.com>" into the parts. The address
# part is parsed without the angle brackets; if no `<...>` is present
# we treat the whole string as the address and leave the name empty.
_FROM_ADDR_RE = re.compile(r"<([^<>]+)>")


def _parse_gmail_from(raw_from: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a "Name <addr>" / "addr" / "Name" header into (name, addr).
    Either part may be None; callers handle None gracefully. Names are
    stripped of surrounding quotes and whitespace.
    """
    if not raw_from:
        return None, None
    s = raw_from.strip()
    m = _FROM_ADDR_RE.search(s)
    if m:
        addr = m.group(1).strip() or None
        name_part = (s[: m.start()] + s[m.end() :]).strip().strip('"').strip()
        name = name_part or None
        return name, addr
    # No angle brackets. Heuristic: if it looks like an email, treat as
    # addr-only; otherwise treat the whole thing as a name.
    if "@" in s and " " not in s:
        return None, s
    return s.strip('"').strip() or None, None


def _parse_rfc2822_date_to_unix(date_str: str) -> Optional[float]:
    """
    Parse an RFC 2822 Date header ("Mon, 02 Sep 2024 13:45:00 +0000")
    into unix seconds. Returns None on any failure -- the caller
    treats a missing timestamp as "don't recency-sort this card".
    """
    if not date_str:
        return None
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(date_str.strip())
        if dt is None:
            return None
        return dt.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _harvest_gmail_header_fields(body_text: str) -> Dict[str, Any]:
    """
    Parse the Gmail-doc markdown header (always written by
    gmail_oauth.build_email_file) and return the fields that the
    source-card builder needs:

      - document_type: "email"
      - subject:       string
      - from_name:     parsed display name (or None)
      - from_email:    parsed addr (or None)
      - timestamp:     unix seconds from the Date header (float, or None)
      - labels:        list[str] of label ids
      - permalink:     string (or None)

    Fields that aren't present in the header are simply omitted from
    the returned dict.
    """
    out: Dict[str, Any] = {}
    if not body_text:
        return out
    if not _GMAIL_DOC_HEADER_RE.search(body_text):
        # Not a Gmail doc — bail early. This keeps the function safe
        # to call unconditionally from the source-card builder.
        return out
    out["document_type"] = "email"

    m = _GMAIL_SUBJECT_RE.search(body_text)
    if m:
        out["subject"] = m.group(1).strip()
    m = _GMAIL_FROM_RE.search(body_text)
    if m:
        name, addr = _parse_gmail_from(m.group(1))
        if name:
            out["from_name"] = name
        if addr:
            out["from_email"] = addr
    m = _GMAIL_DATE_RE.search(body_text)
    if m:
        ts = _parse_rfc2822_date_to_unix(m.group(1))
        if ts is not None:
            out["timestamp"] = ts
    m = _GMAIL_LABELS_RE.search(body_text)
    if m:
        labels = [s.strip() for s in m.group(1).split(",") if s.strip()]
        if labels:
            out["labels"] = labels
    m = _GMAIL_PERMALINK_RE.search(body_text)
    if m:
        out["permalink"] = m.group(1).strip()
    return out


def _load_state_safely() -> Optional[IngestionState]:
    """Load ingestion state once per request. Return None on any failure."""
    try:
        return IngestionState(STATE_PATH)
    except Exception as e:  # noqa: BLE001 -- we never want recall to crash on this
        logger.warning('recall_state_load_failed', extra={'error': type(e).__name__})
        return None


# ---------------------------------------------------------------------- #
# Source-list cleaning for the UI
# ---------------------------------------------------------------------- #
# A "rich" source has at least one Slack-derived field. Anything else is a
# minimal fallback (just an opaque HydraDB id / chunk_N placeholder).
RICH_SOURCE_FIELDS = ("channel", "user", "snippet", "permalink", "stable_key")


def _is_rich_source(source: Dict[str, Any]) -> bool:
    """True if the source card carries any UI-useful Slack metadata."""
    return any(source.get(field) for field in RICH_SOURCE_FIELDS)


def _dedupe_key(source: Dict[str, Any]) -> Optional[str]:
    """Identifier used to dedupe sources: stable_key -> permalink -> source."""
    for field in ("stable_key", "permalink", "source"):
        value = source.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _clean_sources_for_ui(
    sources: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Trim the sources array for the API response WITHOUT touching the LLM
    context. Rules:
      1. If at least one rich source exists, drop the minimal ones.
      2. Dedupe by stable_key -> permalink -> source (first occurrence wins,
         which keeps the earliest matching index so citations stay aligned).
      3. Cap the result at top_k.
      4. If there are no rich sources at all, fall back to deduped minimal.
    """
    if not sources:
        return []

    rich = [s for s in sources if _is_rich_source(s)]
    pool = rich if rich else sources

    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for source in pool:
        key = _dedupe_key(source)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        # If we have no key to dedupe on, let the source through — we can't
        # safely call it a duplicate of anything.
        deduped.append(source)

    return deduped[:top_k]


# ---------------------------------------------------------------------- #
# Citation hygiene
# ---------------------------------------------------------------------- #
# The LLM emits citation markers that look like:
#     [1]
#     【1】
#     【1†source: slack_all-second-brain_1778775842.md】
#
# Cleaning the sources list can drop indexes (e.g. dedupe collapses [4] into
# [1]). If the answer still references those gone indexes the UI shows
# dangling citations. We strip exactly those markers — and only those.
#
# Pattern explanation:
#   - ASCII form:  '[', optional spaces, digits, optional spaces, ']'
#   - CJK form:    '【', optional spaces, digits, optional spaces,
#                  optional '†<anything but the closing bracket>',
#                  '】'
# Two alternates, two capture groups; whichever matched holds the integer.
_CITATION_PATTERN = re.compile(r"\[\s*(\d+)\s*\]" r"|" r"【\s*(\d+)\s*(?:†[^】]*)?】")


def _strip_invalid_citations(answer: str, allowed_indexes: Set[int]) -> str:
    """
    Remove citation markers from `answer` whose index is NOT in
    `allowed_indexes`. Valid citation markers are left exactly as they are.

    Per spec, we do not touch any other character — surrounding whitespace
    or punctuation around a removed marker is left untouched. Some UI
    renderers may show a small extra space; that's fine.
    """
    if not answer or not allowed_indexes:
        # No allowed indexes means everything is invalid -> strip them all.
        # No answer -> nothing to do.
        if not answer:
            return answer

    def _replace(match: re.Match) -> str:
        idx_str = match.group(1) or match.group(2)
        try:
            idx = int(idx_str)
        except (TypeError, ValueError):
            return match.group(0)  # unparseable -> leave alone (safer)
        return match.group(0) if idx in allowed_indexes else ""

    return _CITATION_PATTERN.sub(_replace, answer)


def _coerce_to_unix_seconds(value: Any) -> Optional[float]:
    """
    Accept either a Slack ts string ('1778775842.876209') or a numeric
    unix timestamp (int or float) and return seconds as a float.
    Returns None if we can't parse.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------- #
# Source-kind detection for the allowed_sources filter
# ---------------------------------------------------------------------- #
# A small helper that classifies a source card as coming from Slack or
# Gmail. Detection order:
#   1. document_type — set by the Slack ingest builder ("message" /
#      "thread") and the Gmail ingest builder ("email"). This is the
#      cheapest, most reliable signal.
#   2. stable_key prefix — "slack:..." for Slack, "gmail:..." for Gmail.
#      Used as a fallback when document_type didn't make it onto the
#      card (rare: ingestion-state lookup miss).
# Cards we can't classify return None, and the caller decides how to
# handle them. The current policy (see _source_passes_filters) treats
# unknown-source cards as "pass through" — better to over-include than
# to silently drop legitimate matches.

# Document types that count as a Slack source. Kept narrow so a future
# new doc_type doesn't accidentally classify as Slack.
_SLACK_DOC_TYPES = {"message", "thread"}
# Document types that count as a Gmail source.
_GMAIL_DOC_TYPES = {"email"}


def _extract_source_kind(card: Dict[str, Any]) -> Optional[str]:
    """
    Classify a source card as "slack" or "gmail" (or None if unknown).

    The classification is intentionally conservative — we'd rather
    return None and let an unknown card pass than mislabel one. The
    `allowed_sources` filter relies on this exactly: when the result
    is None, we don't filter the card out, we let it through.
    """
    if not isinstance(card, dict):
        return None
    doc_type = card.get("document_type")
    if isinstance(doc_type, str):
        if doc_type in _SLACK_DOC_TYPES:
            return "slack"
        if doc_type in _GMAIL_DOC_TYPES:
            return "gmail"
    # Fallback: stable_key prefix. The Slack message builder uses
    # "slack:msg:..." / "slack:thread:..."; the Gmail builder uses
    # "gmail:msg:...". Both prefixes are stable parts of the wire
    # format the backend writes -- safe to rely on.
    sk = card.get("stable_key")
    if isinstance(sk, str):
        if sk.startswith("slack:"):
            return "slack"
        if sk.startswith("gmail:"):
            return "gmail"
    return None


def _source_passes_filters(
    source_card: Dict[str, Any],
    channel: Optional[str],
    user: Optional[str],
    document_type: Optional[str],
    start_unix: Optional[float] = None,
    end_unix: Optional[float] = None,
    allowed_sources: Optional[List[str]] = None,
) -> bool:
    """
    Return True if this source card should be kept given the filters.

    Filters only apply when the source card carries the corresponding
    metadata. If a card was built from a chunk that didn't join to state
    (no `channel` / `user` / `document_type` / `timestamp` fields), we
    let it through for the metadata-style filters — better to over-
    include than to silently drop legitimate matches.

    `allowed_sources` is a small whitelist like ["slack"] or
    ["slack", "gmail"]. None / empty list means "all sources" and is
    the default behavior. Unknown-source cards (no document_type AND
    no recognizable stable_key prefix) pass through even when the
    filter is set — same "don't silently drop" principle.

    Date filters are slightly stricter: a card with no parseable
    timestamp is also let through, on the same "don't drop" principle.
    """
    if channel:
        # Strip a leading "#" so "#engineering" and "engineering" both
        # match the stored channel name. The query rewriter normalizes
        # this for inferred channels, but a user may also pass an
        # explicit `channel` argument carrying the prefix.
        wanted = channel.lstrip("#").lower()
        card_channel = source_card.get("channel")
        if isinstance(card_channel, str) and card_channel.lower() != wanted:
            return False
    if user:
        card_user = source_card.get("user")
        if isinstance(card_user, str) and card_user.lower() != user.lower():
            return False
    if document_type:
        card_type = source_card.get("document_type")
        if isinstance(card_type, str) and card_type != document_type:
            return False
    if allowed_sources:
        # None/empty -> "all sources allowed" (default behavior).
        # Otherwise we filter, but only when we can actually classify
        # the card. Unknown-source cards pass.
        kind = _extract_source_kind(source_card)
        if kind is not None and kind not in allowed_sources:
            return False
    if start_unix is not None or end_unix is not None:
        ts_value = _coerce_to_unix_seconds(source_card.get("timestamp"))
        if ts_value is not None:
            if start_unix is not None and ts_value < start_unix:
                return False
            if end_unix is not None and ts_value > end_unix:
                return False
    return True


# ---------------------------------------------------------------------- #
# Recency intent: "latest message" / "most recent message in #engineering"
# ---------------------------------------------------------------------- #
# When the user asks for the latest / newest / most recent Slack message
# we override semantic ranking and sort candidates by Slack timestamp
# descending. Pure semantic recall (top_k=5 from HydraDB) often misses
# the newest message because a brand-new "REALTIME TEST" line is short,
# semantically thin, and easily out-ranked by older but lexically richer
# messages. The fix: widen the candidate set, then sort by timestamp.
#
# Detection is deliberately conservative -- we only fire when both a
# recency word AND a Slack-message noun appear, so phrases like "latest
# news" or "what's the most recent decision" continue to use semantic
# recall. Boundary case "latest news" is pinned by an existing test.

# Recency target nouns. Both Slack-message and Gmail-email nouns count
# (Phase 10): the recency reranker is connector-agnostic and ranks any
# candidate chunk that carries a parseable timestamp. We DO NOT treat
# "decision" / "news" / "update" as recency targets -- those are
# semantic-recall use cases.
_RECENCY_TARGET_TOKENS = (
    # Slack
    "message",
    "messages",
    "post",
    "posts",
    "chat",
    "chats",
    "ping",
    "pings",
    "slack",  # "latest in slack", "newest slack message"
    # Gmail (Phase 10)
    "email",
    "emails",
    "mail",
    "mails",
    "inbox",
    "gmail",  # "latest in gmail", "newest gmail email"
)

# Recency cue words.
_RECENCY_CUE_TOKENS = (
    "latest",
    "newest",
    "recent",
    "last",
)

# Multi-word cues that don't fit the single-token loop above.
_RECENCY_CUE_PHRASES = (
    "most recent",
    "what's new",
    "whats new",
)

# Widened HydraDB candidate set when the recency intent fires. The
# default user-facing top_k is small (5) for LLM context-budget
# reasons; for recency reranking we want a much larger pool to pick
# from, since the newest message may sit far down the semantic ranking.
_RECENCY_CANDIDATE_POOL = 50


def _detect_recency_intent(question: str) -> bool:
    """
    True iff the question is asking for the latest Slack message (or
    similar). Conservative: requires a recency cue AND a Slack-message
    target noun. "latest news" alone returns False (no chat-like noun).
    """
    if not question:
        return False
    q = question.lower()

    has_target = any(tok in q for tok in _RECENCY_TARGET_TOKENS)
    if not has_target:
        return False

    # Token-level cue match using word boundaries so "lastly" doesn't
    # trip "last".
    for cue in _RECENCY_CUE_TOKENS:
        if re.search(rf"\b{re.escape(cue)}\b", q):
            return True
    for phrase in _RECENCY_CUE_PHRASES:
        if phrase in q:
            return True
    return False


def _is_slack_message_card(card: Dict[str, Any]) -> bool:
    """
    Backwards-compatible: True iff the card came from a Slack
    message / thread ingest. Used by tests + the Phase 1–9 surface
    area. Recency reranking itself now uses _recency_source_kind()
    below, which also returns "gmail" so emails participate.
    """
    return _recency_source_kind(card) == "slack"


def _recency_source_kind(card: Dict[str, Any]) -> Optional[str]:
    """
    Classify a source card for the recency reranker:

      - "slack"  -> a Slack message/thread doc (timestamps == Slack ts)
      - "gmail"  -> a Gmail email doc          (timestamps == Date header)
      - None     -> unknown shape, exclude from the recency rerank

    Heuristic order: document_type first (cheapest + most reliable),
    stable_key prefix as a fallback. Mirrors _extract_source_kind in
    spirit but is scoped to recency-eligible documents only (we want
    `email` here, not just any Gmail doc that might exist in the
    future).
    """
    if not isinstance(card, dict):
        return None
    doc_type = card.get("document_type")
    if isinstance(doc_type, str):
        if doc_type in ("message", "thread"):
            return "slack"
        if doc_type == "email":
            return "gmail"
    sk = card.get("stable_key")
    if isinstance(sk, str):
        if sk.startswith("slack:"):
            return "slack"
        if sk.startswith("gmail:"):
            return "gmail"
    return None


def _rerank_by_recency(
    chunks_with_meta: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Sort surviving recency-eligible chunks by timestamp DESC and cap
    at top_k.

    Phase 10: both Slack-message and Gmail-email chunks participate.
    Timestamp source per kind:
      - Slack: chunk timestamp_float (Slack ts already in unix seconds)
      - Gmail: timestamp_float harvested from the email's Date header

    Chunks WITHOUT a parseable timestamp are dropped from the recency
    result entirely; if no recency-eligible chunks survive, the
    caller falls back to semantic ranking. This matches the original
    "no Slack chunks -> fall back" contract; Gmail just expands the
    set of eligible candidates.

    Returns a list with the same shape rerank_chunks would return.
    """
    eligible = [
        c
        for c in chunks_with_meta
        if _recency_source_kind(c["source_card"]) is not None and c.get("timestamp_float") is not None
    ]
    if not eligible:
        return []
    eligible.sort(
        key=lambda c: c["timestamp_float"],
        reverse=True,
    )
    return eligible[:top_k]


def prepare_recall_context(
    question: str,
    top_k: int,
    mode: str = "default",
    channel: Optional[str] = None,
    user: Optional[str] = None,
    document_type: Optional[str] = None,
    start_timestamp: Any = None,
    end_timestamp: Any = None,
    metadata_bias: Optional[Dict[str, str]] = None,
    hydradb_sub_tenant_id: Optional[str] = None,
    allowed_sources: Optional[List[str]] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run HydraDB recall and build the LLM-ready context + parallel sources.

    The `mode` parameter changes how we rank the chunks BEFORE they're
    handed to the LLM:
      - "exact":  prefer chunks containing the user's keywords verbatim.
                  Fall back to semantic order if no exact matches exist.
      - "hybrid": combine semantic order with a keyword-hit bonus.
      - all others: preserve HydraDB's semantic order.

    `metadata_bias` (e.g. `{"channel": "product", "user": "rahul"}`) is
    forwarded to the reranker. Matching chunks get a ranking boost in
    every mode — useful for weak person/channel inference where we don't
    want to hard-filter but DO want matching chunks to surface first.
    Strong inference should already have been collapsed into the
    `channel` / `user` arguments before this function is called.

    Returns a dict with one of two shapes:

    Success:
        {
            "ready":            True,
            "context_text":     str,                  # numbered context for LLM
            "sources":          List[Dict[str, Any]], # parallel cards (re-numbered)
            "chunks_count":     int,
            "filtered_out":     int,
            "exact_matches":    int,                  # >=1 keyword hit
            "retrieval_mode":   str,                  # what we actually did
            "query_terms":      List[str],
            "fallback_debug":   None,
        }

    No usable context:
        {
            "ready":          False,
            "fallback_debug": Dict[str, Any],
        }
    """
    # Local imports keep recall.py's top-of-file import block stable.
    from search_utils import (
        dedupe_by_stable_key,
        extract_query_terms,
        rerank_chunks,
    )

    # Phase 4: workspace-isolated HydraDB. Fail closed when no
    # sub-tenant is provided — never fall back to HYDRADB_SUB_TENANT_ID
    # / a hard-coded default (that would search the wrong corpus or
    # leak across workspaces). User-facing routes must call
    # ensure_workspace_sub_tenant first and pass the result here.
    sub_tenant = require_sub_tenant_id(
        hydradb_sub_tenant_id,
        context="prepare_recall_context",
    )
    hydra = HydraDBClient(sub_tenant_id=sub_tenant)

    # Normalize the allowed_sources whitelist. Input shapes we accept:
    #   None                      -> all sources allowed (default)
    #   []                        -> all sources allowed (same as None)
    #   ["slack"]                 -> Slack only
    #   ["gmail"]                 -> Gmail only
    #   ["slack","gmail"]         -> both (= all currently known)
    #   ["Slack", " slack ", ""]  -> ["slack"] (trimmed, lowercased, deduped)
    # Anything that normalizes to an empty set is collapsed to None so
    # the filter becomes a no-op downstream.
    normalized_sources: Optional[List[str]] = None
    if allowed_sources:
        cleaned = {s.strip().lower() for s in allowed_sources if isinstance(s, str) and s.strip()}
        normalized_sources = sorted(cleaned) if cleaned else None

    # Recency intent: widen the candidate pool BEFORE the HydraDB call.
    # Pure semantic top_k=5 frequently misses the newest message
    # because brand-new short messages are lexically thin and lose to
    # older but more-keyword-dense docs. We pull a much larger pool
    # and re-sort by Slack timestamp below.
    recency_intent = _detect_recency_intent(question)
    effective_top_k = max(top_k, _RECENCY_CANDIDATE_POOL) if recency_intent else top_k
    raw_response = hydra.full_recall(query=question, top_k=effective_top_k)
    chunks = _extract_chunks(raw_response)
    # DIAGNOSTIC (instrumentation only): the sub-tenant recall actually
    # queried and how many chunks HydraDB returned. Compare `sub_tenant`
    # against the ingest-side gmail_ingest_label_threads log to catch a
    # write/read sub-tenant mismatch; chunks_returned=0 confirms an empty
    # corpus for this query.
    logger.info(
        "recall_corpus_probe",
        extra={
            "workspace_id": workspace_id,
            "sub_tenant": sub_tenant,
            "chunks_returned": len(chunks),
        },
    )
    debug_on = _debug_recall_enabled()

    if debug_on:
        logger.debug(
            'recall_raw_response',
            extra={
                'chunks_count': len(chunks),
                'raw_response_preview': _safe_json(raw_response, limit=500),
            },
        )
        if chunks:
            logger.debug(
                'recall_first_chunk_preview',
                extra={
                    'first_chunk_keys': list(chunks[0].keys()) if isinstance(chunks[0], dict) else None,
                },
            )

    state = _load_state_safely()
    start_unix = _coerce_to_unix_seconds(start_timestamp)
    end_unix = _coerce_to_unix_seconds(end_timestamp)
    query_terms = extract_query_terms(question) if mode in ("exact", "hybrid") else []

    # ---- Step 1: gather per-chunk metadata for ranking ----
    # We build a parallel list of {text, source_card, original_index,
    # timestamp_float, hits(_)}. Filters drop chunks before ranking so
    # the rerank operates only on the candidate set the user actually
    # wants.
    chunks_with_meta: List[Dict[str, Any]] = []
    filtered_out = 0
    for original_index, chunk in enumerate(chunks, start=1):
        text = _chunk_text(chunk).strip()
        if not text:
            continue
        score = _chunk_score(chunk)
        # The 'index' we initially put on the card is the HydraDB order.
        # We'll renumber after reranking so the cards line up with the
        # [N] labels the LLM sees.
        source_card = _build_source_card(chunk, original_index, score, state)

        if not _source_passes_filters(
            source_card,
            channel,
            user,
            document_type,
            start_unix,
            end_unix,
            allowed_sources=normalized_sources,
        ):
            filtered_out += 1
            continue

        chunks_with_meta.append(
            {
                "text": text,
                "source_card": source_card,
                "original_index": original_index,
                "timestamp_float": _coerce_to_unix_seconds(source_card.get("timestamp")),
            }
        )

    # ---- Step 1.5: fold structured memory candidates into the pool ----
    # Phase 12: extracted memories (action items, decisions, summaries,
    # entities) live in `extracted_memories`. We pull a small batch
    # matching the question and add them as candidates with the SAME
    # shape as HydraDB chunks so the existing reranker treats them
    # uniformly. Memories are AUGMENTATION -- they share the keyword-
    # hit + metadata-bias signals with HydraDB chunks but never
    # participate in the recency rerank (we don't want a 6-month-old
    # decision outranking today's Slack message when the user asks
    # "latest message"). The memory layer fails gracefully: a Supabase
    # outage degrades to no memories but never blocks the answer.
    memory_candidates: List[Dict[str, Any]] = []
    if workspace_id and not recency_intent:
        try:
            # Defer the import so the test suite can mock at the
            # memory_store boundary without forcing this module to
            # eagerly construct a Supabase client at import time.
            from memory_store import list_memories  # noqa: PLC0415

            memory_rows = list_memories(
                workspace_id=workspace_id,
                # Pull from every kind by default; the question-text
                # filter narrows down to relevant content.
                query=question.strip() or None,
                limit=10,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "memory_lookup_skipped",
                extra={
                    "workspace_id": workspace_id,
                    "error": type(e).__name__,
                },
            )
            memory_rows = []
        # Phase 16: derive an importance score per memory row
        # (recurrence + recency + owner-presence + cluster-size) so
        # important memories carry a higher card score than the old
        # flat 0.5. Defensive: any failure inside the importance
        # computation degrades to the legacy 0.5 baseline so memory
        # injection itself can never break.
        importance_by_id: Dict[Any, float] = {}
        if memory_rows:
            try:
                # Deferred import for the same reason as memory_store
                # above: tests mock at this module boundary.
                from memory_intelligence import compute_memory_importance  # noqa: PLC0415

                importance_by_id = compute_memory_importance(memory_rows) or {}
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "memory_importance_skipped",
                    extra={
                        "workspace_id": workspace_id,
                        "error": type(e).__name__,
                    },
                )
                importance_by_id = {}
        memory_start_index = len(chunks_with_meta) + 1
        for offset, row in enumerate(memory_rows):
            content = (row.get("content") or "").strip()
            if not content:
                continue
            kind = (row.get("kind") or "").strip()
            source_kind = row.get("source_kind")
            source_stable_key = row.get("source_stable_key") or ""
            source_ts = row.get("source_timestamp")
            ts_float = _coerce_to_unix_seconds(source_ts) if source_ts else None
            # Phase 16: importance-aware score. When the importance
            # computation succeeded for this row, scale it into
            # [0.3, 1.0]; otherwise keep the legacy neutral 0.5 so
            # pre-Phase-16 behavior (and its tests) is preserved.
            _imp = importance_by_id.get(row.get("id"))
            if isinstance(_imp, (int, float)) and 0.0 <= float(_imp) <= 1.0:
                memory_score = round(0.3 + 0.7 * float(_imp), 6)
            else:
                memory_score = 0.5  # neutral semantic baseline
            # Build a memory-flavored source card. The `source_kind`
            # field carries the ORIGINAL connector ("slack"/"gmail")
            # so existing source-filter logic keeps working --
            # asking for slack-only correctly includes Slack-derived
            # memories. We also stamp `memory_kind` for downstream
            # consumers / log lines.
            card: Dict[str, Any] = {
                "index": memory_start_index + offset,
                "source": source_stable_key or f"memory:{kind}",
                "score": memory_score,
                "stable_key": source_stable_key,
                "source_kind": source_kind,
                "document_type": f"memory_{kind}" if kind else "memory",
                "memory_kind": kind,
                "memory_id": row.get("id"),
                "timestamp": source_ts,
                "owner": row.get("owner") or None,
                "entity_type": row.get("entity_type") or None,
            }
            # Friendly LLM-facing text: prepend a small kind tag so
            # the model knows it's reading structured memory and not
            # raw conversation.
            prefix = {
                "action_item": "Action item",
                "decision": "Decision",
                "summary": "Summary",
                "entity": "Entity",
            }.get(kind, "Memory")
            owner = card.get("owner")
            text_block = f"[{prefix}{' (owner: ' + owner + ')' if owner else ''}] " f"{content}"
            memory_candidates.append(
                {
                    "text": text_block,
                    "source_card": card,
                    "original_index": card["index"],
                    "timestamp_float": ts_float,
                }
            )

    if memory_candidates:
        # Apply the same workspace-scoped filters (channel/user/etc.)
        # to memory candidates. Most memories carry no `channel` so
        # the don't-silently-drop rule lets them through any channel
        # filter; the source filter does honor them (e.g. slack-only
        # gets memory rows where source_kind=="slack").
        kept: List[Dict[str, Any]] = []
        for mc in memory_candidates:
            if _source_passes_filters(
                mc["source_card"],
                channel,
                user,
                document_type,
                start_unix,
                end_unix,
                allowed_sources=normalized_sources,
            ):
                kept.append(mc)
            else:
                filtered_out += 1
        chunks_with_meta.extend(kept)

    # Same Slack message that resurfaces twice in recall shouldn't get
    # twice the ranking boost.
    chunks_with_meta = dedupe_by_stable_key(chunks_with_meta)

    # ---- Step 3: rerank if the mode asks for it; otherwise just cap ----
    # Recency intent overrides the normal modes. If the recency
    # reranker returns chunks, we use them; if it returns nothing
    # (no surviving Slack-message chunks with timestamps -- e.g. the
    # workspace has only Gmail docs), we fall back to the normal
    # semantic rerank so the query still gets answered.
    if recency_intent:
        recency_ranked = _rerank_by_recency(chunks_with_meta, top_k)
    else:
        recency_ranked = []

    if recency_ranked:
        ranked = recency_ranked
        exact_matches_found = 0
        retrieval_mode_effective = "recency"
    else:
        ranked, exact_matches_found = rerank_chunks(
            chunks_with_meta,
            query_terms,
            mode,
            top_k,
            metadata_bias=metadata_bias,
        )
        retrieval_mode_effective = mode

    logger.debug(
        'recall_context_ready',
        extra={
            'chunks_count': len(chunks),
            'filtered_out': filtered_out,
            'mode': retrieval_mode_effective,
            'recency_intent': recency_intent,
            'top_k': top_k,
        },
    )

    # Build a compact per-chunk ranking breakdown so test cases (and
    # `DEBUG_RECALL=true` runs) can see *why* each surviving chunk
    # ranked where it did. This is INTERNAL only -- main.py never
    # forwards this field to the public API response, and tests read
    # it directly off the prepare_recall_context return value. We
    # never include row-level body text or any user-identifying free
    # text here; only the source kind, the score breakdown, and the
    # stable_key (already a public identifier).
    rank_breakdown: List[Dict[str, Any]] = []
    for i, chunk in enumerate(ranked, start=1):
        card = chunk.get("source_card") or {}
        entry: Dict[str, Any] = {
            "rank": i,
            "source_kind": _recency_source_kind(card),
            "stable_key": card.get("stable_key"),
            "original_index": chunk.get("original_index"),
            "timestamp": chunk.get("timestamp_float"),
            "score_breakdown": dict(chunk.get("_debug_score") or {}),
            "retrieval_mode": retrieval_mode_effective,
        }
        rank_breakdown.append(entry)
    if rank_breakdown:
        logger.debug(
            'recall_rank_breakdown',
            extra={
                'mode': retrieval_mode_effective,
                'top_chunks': rank_breakdown[:5],  # cap to 5 to keep log volume sane
            },
        )

    if not ranked:
        first_chunk = chunks[0] if chunks else None
        first_chunk_keys = list(first_chunk.keys()) if isinstance(first_chunk, dict) else None
        debug_payload: Dict[str, Any] = {
            "reason": "no usable text found in HydraDB chunks",
            "raw_response_keys": (list(raw_response.keys()) if isinstance(raw_response, dict) else None),
            "chunks_returned": len(chunks),
            "chunks_filtered_out": filtered_out,
            "first_chunk_keys": first_chunk_keys,
            "retrieval_mode": mode,
            "exact_matches_found": 0,
            "query_terms": query_terms,
            "top_k": top_k,
        }
        if debug_on and first_chunk is not None:
            debug_payload["first_chunk_preview"] = _first_chunk_preview(first_chunk)
        return {"ready": False, "fallback_debug": debug_payload}

    # ---- Step 4: renumber 1..N so citations align with surviving cards ----
    context_blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    for i, chunk in enumerate(ranked, start=1):
        text = chunk["text"]
        card = dict(chunk["source_card"])
        card["index"] = i  # overwrite the original-order index
        context_label = card.get("channel") or card.get("source")
        context_blocks.append(f"[{i}] (source: {context_label})\n{text}")
        sources.append(card)

    return {
        "ready": True,
        "context_text": "\n\n".join(context_blocks),
        "sources": sources,
        "chunks_count": len(chunks),
        "filtered_out": filtered_out,
        "exact_matches": exact_matches_found,
        "retrieval_mode": retrieval_mode_effective,
        "query_terms": query_terms,
        "fallback_debug": None,
        # Phase 10: internal-only ranking breakdown for logs and
        # tests. Never forwarded to the public API response (main.py
        # builds its debug shape field-by-field rather than spreading
        # this dict).
        "rank_breakdown": rank_breakdown,
    }


def finalize_answer(
    raw_answer: str,
    sources: List[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    """
    Apply source-cleaning + citation-stripping to a raw LLM answer.

    Used by BOTH /api/query and /api/query/stream so the output post-
    processing is identical regardless of how the answer text arrived.
    """
    cleaned_sources = _clean_sources_for_ui(sources, top_k=top_k)
    allowed_indexes: Set[int] = {s["index"] for s in cleaned_sources if isinstance(s.get("index"), int)}
    final_answer = _strip_invalid_citations(raw_answer, allowed_indexes)
    return {
        "answer": final_answer,
        "cleaned_sources": cleaned_sources,
        "sources_before": len(sources),
        "sources_after": len(cleaned_sources),
    }


# ---------------------------------------------------------------------- #
# Public entry point (non-streaming)
# ---------------------------------------------------------------------- #
def answer_question(
    question: str,
    top_k: int = 5,
    mode: str = "default",
    channel: Optional[str] = None,
    user: Optional[str] = None,
    document_type: Optional[str] = None,
    start_timestamp: Any = None,
    end_timestamp: Any = None,
    conversation_history: Optional[List[Any]] = None,
    metadata_bias: Optional[Dict[str, str]] = None,
    hydradb_sub_tenant_id: Optional[str] = None,
    allowed_sources: Optional[List[str]] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end recall + grounded answer.

    Args:
        question:              the natural-language question.
        top_k:                 how many chunks to ask HydraDB for.
        mode:                  "default" | "summary" | "decisions" | "action_items"
                               | "who_said" | "exact" | "hybrid" — selects the
                               system prompt AND the retrieval/ranking strategy.
        channel:               optional channel-name filter (e.g. "general").
        user:                  optional user-name filter (e.g. "Praveer Nema").
        document_type:         optional "message" or "thread".
        start_timestamp:       optional inclusive lower bound on source timestamps
                               (Slack ts string or unix seconds).
        end_timestamp:         optional inclusive upper bound.
        conversation_history:  optional recent {role, content} turns. Passed
                               to the LLM for reference resolution ("he",
                               "that decision", etc). DOES NOT affect
                               retrieval — only the latest question goes
                               into HydraDB's similarity search.
        metadata_bias:         optional dict like {"channel": "product",
                               "user": "rahul"} — weak inferred filters
                               applied as a ranking bias rather than a
                               hard filter. Strong inference should be
                               passed via `channel`/`user` directly.

    Returns:
        {
            "answer":  str,
            "sources": [...],
            "debug":   { ... }
        }
    """
    if not question or not question.strip():
        return {
            "answer": "Please ask a question.",
            "sources": [],
            "debug": {"reason": "empty question"},
        }

    # Phase 15: capture a coarse start timestamp so the analytics
    # emit at the end of this function can report latency without
    # needing the caller to time us.
    import time as _t_init  # noqa: PLC0415

    _start_ms = int(_t_init.perf_counter() * 1000)

    # IMPORTANT: retrieval uses only the current question. Conversation
    # history is intentionally NOT concatenated into the search query.
    prepared = prepare_recall_context(
        question=question,
        top_k=top_k,
        mode=mode,
        channel=channel,
        user=user,
        document_type=document_type,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        metadata_bias=metadata_bias,
        hydradb_sub_tenant_id=hydradb_sub_tenant_id,
        allowed_sources=allowed_sources,
        workspace_id=workspace_id,
    )

    if not prepared["ready"]:
        # Phase 15: emit retrieval_failure so the analytics view can
        # surface empty-result patterns over time.
        if workspace_id:
            try:
                from analytics_store import emit_event  # noqa: PLC0415

                fallback_debug = prepared.get("fallback_debug") or {}
                emit_event(
                    workspace_id=workspace_id,
                    kind="retrieval_failure",
                    success=False,
                    payload={
                        "reason": str(fallback_debug.get("reason") or "no_chunks")[:200],
                        "mode": mode,
                        "top_k": top_k,
                    },
                )
            except Exception:  # noqa: BLE001
                pass
        return {
            "answer": INSUFFICIENT_CONTEXT_ANSWER,
            "sources": [],
            "debug": prepared["fallback_debug"],
        }

    raw_answer = generate_grounded_answer(
        question=question,
        context=prepared["context_text"],
        mode=mode,
        conversation_history=conversation_history,
    )

    finalized = finalize_answer(
        raw_answer=raw_answer,
        sources=prepared["sources"],
        top_k=top_k,
    )

    history_used = bool(conversation_history)
    result = {
        "answer": finalized["answer"],
        "sources": finalized["cleaned_sources"],
        "debug": {
            "chunks_returned": prepared["chunks_count"],
            "chunks_used": len(prepared["sources"]),
            "chunks_filtered_out": prepared["filtered_out"],
            "sources_before_clean": finalized["sources_before"],
            "sources_after_clean": finalized["sources_after"],
            "mode": mode,
            "retrieval_mode": prepared.get("retrieval_mode", mode),
            "exact_matches_found": prepared.get("exact_matches", 0),
            "query_terms": prepared.get("query_terms", []),
            "top_k": top_k,
            "history_used": history_used,
            "history_turns": len(conversation_history) if history_used else 0,
        },
    }

    # Phase 15: emit a query_completed analytics event. Defensive:
    # any analytics failure must NOT affect the answer (analytics is
    # a fire-and-forget signal). We compute the per-event payload
    # from data we already have here.
    if workspace_id:
        try:
            import time as _t  # noqa: PLC0415

            from analytics_store import emit_event  # noqa: PLC0415

            # The caller measured no latency; we approximate using the
            # debug fields. For a more precise number we'd time the
            # whole answer_question call, but the analytics consumer
            # just needs an order-of-magnitude figure for stat trends.
            sources_out = result["sources"] or []
            source_kinds = sorted({(s.get("source_kind") or _recency_source_kind(s) or "unknown") for s in sources_out})
            memory_hit = any((s.get("memory_kind") or "") for s in sources_out)
            emit_event(
                workspace_id=workspace_id,
                kind="query_completed",
                latency_ms=int(_t.perf_counter() * 1000) - _start_ms,
                payload={
                    "mode": mode,
                    "retrieval_mode": prepared.get("retrieval_mode", mode),
                    "top_k": top_k,
                    "sources_count": len(sources_out),
                    "source_kinds": source_kinds,
                    "memory_hit": bool(memory_hit),
                    "history_used": history_used,
                },
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "analytics_query_emit_skipped",
                extra={"error": type(e).__name__},
            )
    return result
