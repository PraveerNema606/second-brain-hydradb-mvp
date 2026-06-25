"""
Phase 12: deterministic structured-memory extraction.

This module is pure: no I/O, no Supabase, no HydraDB, no LLM. Given a
text body it returns lightweight structured records that the
memory_store module persists to the `extracted_memories` table.

Design choices and trade-offs
-----------------------------

1.  **Regex, not ML.** The task spec says "no heavy ML infrastructure".
    Action-item / decision phrasing is linguistically narrow and
    follows predictable patterns ("we agreed to X", "Rahul will Y",
    "TODO: Z", "Can you send me Q"). A deterministic extractor handles
    the strong-signal majority with zero runtime cost. False negatives
    are acceptable; false positives are the real risk and we lean
    toward narrow patterns.

2.  **Recall over precision on entities.** Entities are cheap to store
    and serve as soft retrieval anchors -- a noisy extra "entity"
    almost never hurts recall. We bias toward catching candidates.

3.  **No header-line extraction.** We strip the Slack/Gmail markdown
    headers BEFORE running extraction. A header line like
    "From: Rahul Verma <rahul@acme>" must not produce an action item.

4.  **Content hashing for dedupe.** Every extracted record carries a
    `content_hash` (SHA-256 of canonical lowercased + whitespace-
    collapsed text). The persistence layer uses this as part of the
    UNIQUE key; the same action item said twice in the same source
    deduplicates to one row.

5.  **Lightweight summarization.** "Summary" here is the first 1-3
    salient sentences of the body, NOT an LLM rewrite. The LLM at
    query time still gets to compose a polished answer; this layer's
    job is to surface the right material.

Returns
-------
Each extractor returns a list of dicts. The persistence layer expects
the following minimum shape:

    {
        "kind":           "action_item" | "decision" | "summary" | "entity",
        "content":        str,                       # the canonical text
        "content_hash":   str,                       # SHA-256 hex
        "owner":          Optional[str],             # action_items only
        "entity_type":    Optional[str],             # entities only
        "metadata":       Dict[str, Any],            # always present, may be {}
    }
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #


def extract_all(
    text: str,
    *,
    default_owner: Optional[str] = None,
    source_kind: str = "slack",
) -> List[Dict[str, Any]]:
    """
    Convenience: run every extractor over `text` and return one
    flat list. The persistence layer dedupes within (kind,
    content_hash, source_stable_key) so it's safe to call any number
    of extractors per source.

    `default_owner` is used as the action-item owner when the text
    doesn't supply one explicitly (e.g. Slack messages where the
    speaker IS the owner of any TODO they typed about themselves).
    """
    body = strip_markdown_header(text)
    if not body:
        return []
    out: List[Dict[str, Any]] = []
    out.extend(extract_action_items(body, default_owner=default_owner))
    out.extend(extract_decisions(body))
    out.extend(extract_entities(body, source_kind=source_kind))
    summary = summarize(body)
    if summary:
        out.append(
            _record(
                kind="summary",
                content=summary,
                metadata={"sentence_count": summary.count(". ") + 1},
            )
        )
    return out


# ---------------------------------------------------------------------- #
# Header stripping
# ---------------------------------------------------------------------- #
# Slack and Gmail ingestion both write a markdown header at the top
# of every doc. We strip it before extracting so a "From:" / "Subject:"
# line never produces a false action item.

_SLACK_HEADING_RE = re.compile(r"^#\s*Slack\s+(?:Message|Thread)\b", re.IGNORECASE | re.MULTILINE)
_GMAIL_HEADING_RE = re.compile(r"^#\s*Email\b", re.IGNORECASE | re.MULTILINE)

# Header-style lines (`Key: value` immediately under the heading)
# that we always drop.
_HEADER_LINE_RE = re.compile(
    r"^(Source Key|Channel(?:\s+ID)?|Timestamp|User|Permalink|"
    r"Message-Id|Mailbox|Subject|From|To|Cc|Date|Labels|Snippet):\s*.*$",
    re.IGNORECASE,
)


def strip_markdown_header(text: str) -> str:
    """
    Drop the standard Slack/Gmail markdown header so the rest of the
    extraction operates on body text only. Idempotent.

    Returns body text with leading whitespace trimmed; empty string
    if `text` was None or whitespace-only.
    """
    if not text or not text.strip():
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    out: List[str] = []
    in_header = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Heading line opens the header section.
        if _SLACK_HEADING_RE.match(stripped) or _GMAIL_HEADING_RE.match(stripped):
            in_header = True
            continue
        if in_header:
            if not stripped:
                # First blank line ends the header.
                in_header = False
                continue
            if _HEADER_LINE_RE.match(stripped):
                continue
            # Non-header content on a line without a blank break: we
            # treat the header as ended and keep this line.
            in_header = False
            out.append(line)
            continue
        out.append(line)
    return "\n".join(out).strip()


# ---------------------------------------------------------------------- #
# Action-item extraction
# ---------------------------------------------------------------------- #
# Strong, narrow patterns. Each capture must include enough context
# that the resulting text is meaningful on its own ("deploy Friday"
# is fine; "do it" is not).

# Pattern 1: explicit TODO/ACTION/TASK marker.
_ACTION_MARKER_RE = re.compile(
    r"(?:^|\s)(?:TODO|ACTION(?:\s+ITEM)?|TASK|FOLLOW[\s\-]?UP)\s*[:\-]\s*(.{6,200})",
    re.IGNORECASE,
)

# Pattern 2: "PersonName will <verb>..." or "PersonName to <verb>..."
# Captures both name (owner) and the action. PersonName must look like
# a name: capitalized word, 2-30 chars, no `@` (so we don't match
# email addresses).
_ACTION_OWNER_VERB_RE = re.compile(
    r"\b([A-Z][a-zA-Z][a-zA-Z\-']{1,28})\s+"
    r"(?:will|is going to|plans? to|needs? to|should|"
    r"to (?:investigate|own|lead|drive|handle|fix|ship|deploy|"
    r"send|write|review|setup|set up|migrate|update|create|build|implement)) "
    r"\b(.{6,200})",
)

# Pattern 3: imperative ask -- "Can you send me X?" / "Please update Y"
_ACTION_REQUEST_RE = re.compile(
    r"(?:^|[\s.!?])(?:can you|could you|please|pls)\s+"
    r"((?:send|share|update|review|write|prepare|draft|fix|deploy|investigate|"
    r"check|verify|confirm|reply|respond|follow up|forward|attach)\s+.{4,200})",
    re.IGNORECASE,
)

# Pattern 4: assignment style -- "Assigned to X:" / "Owner: X"
_ACTION_ASSIGNED_RE = re.compile(
    r"\b(?:assigned to|owner|assignee)\s*[:\-]?\s*" r"([A-Z][a-zA-Z][a-zA-Z\-']{1,28})\s*[:\-]?\s*(.{6,200})",
    re.IGNORECASE,
)


def extract_action_items(
    text: str,
    *,
    default_owner: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Detect likely action items.

    `default_owner` is used when the pattern doesn't specify an owner --
    typically the Slack message author (a developer writing "TODO:
    migrate Kafka" on their own message implicitly owns it).

    Returns a list of `_record(kind="action_item", ...)`.
    """
    if not text:
        return []
    body = strip_markdown_header(text) or text
    seen: set = set()
    out: List[Dict[str, Any]] = []

    def _emit(task: str, owner: Optional[str], pattern: str) -> None:
        task = _clean_phrase(task)
        if not task or len(task) < 6:
            return
        key = _canon(task)
        if key in seen:
            return
        seen.add(key)
        out.append(
            _record(
                kind="action_item",
                content=task,
                owner=(owner or default_owner or None),
                metadata={"pattern": pattern},
            )
        )

    # 1. Explicit markers.
    for m in _ACTION_MARKER_RE.finditer(body):
        _emit(m.group(1), default_owner, "marker")
    # 2. "Name will do X" / "Name to do X"
    for m in _ACTION_OWNER_VERB_RE.finditer(body):
        owner_candidate = m.group(1)
        action_phrase = m.group(2)
        # Stitch the verb back into the action so "Rahul will deploy
        # Friday" becomes content "deploy Friday" instead of just
        # "Friday".
        verb_span = body[m.start() : m.start(2)]
        verb_only = verb_span.split(owner_candidate, 1)[-1].strip()
        # Drop the leading auxiliary ("will" / "to" / "should" / ...).
        verb_only = re.sub(
            r"^(?:will|is going to|plans? to|needs? to|should|to)\s+",
            "",
            verb_only.strip(),
            flags=re.IGNORECASE,
        )
        full = (verb_only + " " + action_phrase).strip()
        _emit(full, owner_candidate, "owner_verb")
    # 3. Polite requests.
    for m in _ACTION_REQUEST_RE.finditer(body):
        _emit(m.group(1), default_owner, "request")
    # 4. Explicit assignment.
    for m in _ACTION_ASSIGNED_RE.finditer(body):
        _emit(m.group(2), m.group(1), "assignment")

    return out


# ---------------------------------------------------------------------- #
# Decision extraction
# ---------------------------------------------------------------------- #
# Decisions surface in past- or present-tense agreements. The pattern
# bank is small and narrow on purpose -- a "we should X" is an action
# item / suggestion, not a settled decision.

_DECISION_PATTERNS = [
    re.compile(
        r"\b(?:we|the team|engineering)\s+(?:have\s+)?agreed\s+(?:to|on|that)\s+(.{6,200})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdecision\s*[:\-]\s*(.{6,200})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:we'?ll|we will)\s+(?:keep|use|adopt|move|switch|stick(?: with)?|go with|stop|disable|enable)\s+(.{4,200})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:approved|signed[\s\-]off|ratified)\s*[:\-]?\s*(.{4,200})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmoving\s+to\s+(.{4,200})",
        re.IGNORECASE,
    ),
]


def extract_decisions(text: str) -> List[Dict[str, Any]]:
    """
    Detect likely decisions. Returns `kind="decision"` records.
    """
    if not text:
        return []
    body = strip_markdown_header(text) or text
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for pat in _DECISION_PATTERNS:
        for m in pat.finditer(body):
            phrase = _clean_phrase(m.group(1))
            if not phrase or len(phrase) < 4:
                continue
            key = _canon(phrase)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                _record(
                    kind="decision",
                    content=phrase,
                    metadata={"pattern_id": _DECISION_PATTERNS.index(pat)},
                )
            )
    return out


# ---------------------------------------------------------------------- #
# Entity extraction
# ---------------------------------------------------------------------- #
# Lightweight: capitalized multi-word terms (likely proper nouns),
# `@mentions`, `#channels`, code-like terms in backticks. We classify
# loosely:
#   - @foo / U123 -> person
#   - #channel    -> channel
#   - PascalCase / known_service_keywords -> service
#   - everything else capitalized -> project

_AT_MENTION_RE = re.compile(r"<@([A-Z][A-Z0-9]+)>|@([A-Za-z][A-Za-z0-9_\-\.]{1,30})")
_HASH_CHANNEL_RE = re.compile(r"<#[A-Z][A-Z0-9]+\|([a-z0-9_\-]+)>|#([a-z][a-z0-9_\-]{1,40})")
_CODE_TERM_RE = re.compile(r"`([a-zA-Z][a-zA-Z0-9_/\-\.]{1,40})`")
# Two-or-more capitalized words in a row, OR a single PascalCase token.
_PROPER_NOUN_RE = re.compile(
    r"\b("
    r"(?:[A-Z][a-z]+(?:[\-\s][A-Z][a-z]+){0,3})"  # "Kafka", "Project Apollo"
    r"|[A-Z]{2,}(?:[A-Z][a-z]+)?"  # "AWS", "PostgreSQL"
    r")\b"
)

# Words to exclude from "project" classification because they're
# almost always grammatical noise -- the start of a sentence.
_NOUN_BLOCKLIST = frozenset(
    {
        "I",
        "We",
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Our",
        "Your",
        "It",
        "He",
        "She",
        "They",
        "But",
        "And",
        "Or",
        "If",
        "When",
        "Where",
        "Who",
        "What",
        "Why",
        "How",
        "Yes",
        "No",
        "OK",
        "Okay",
        "Hi",
        "Hey",
        "Hello",
        "Thanks",
        "Today",
        "Tomorrow",
        "Yesterday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "TODO",
        "ACTION",
        "TASK",
        "FYI",
        "BTW",
        "PR",
        "PRs",
    }
)

# Hints for "service" classification when we see PascalCase.
_SERVICE_HINT_TERMS = frozenset(
    {
        "kafka",
        "redis",
        "postgres",
        "postgresql",
        "mysql",
        "snowflake",
        "kubernetes",
        "k8s",
        "docker",
        "elasticsearch",
        "rabbitmq",
        "nginx",
        "stripe",
        "twilio",
        "sendgrid",
        "datadog",
        "sentry",
        "cloudflare",
        "vercel",
        "railway",
        "render",
        "fly",
        "aws",
        "gcp",
        "azure",
        "github",
        "gitlab",
        "supabase",
        "hydradb",
        "openai",
        "anthropic",
        "slack",
        "gmail",
        "notion",
        "linear",
        "asana",
        "jira",
    }
)


def extract_entities(
    text: str,
    *,
    source_kind: str = "slack",
) -> List[Dict[str, Any]]:
    """
    Extract lightweight entities. Bias toward recall: we'd rather
    surface a noisy candidate than miss a real signal.

    Returns `kind="entity"` records with `entity_type` set.
    """
    if not text:
        return []
    body = strip_markdown_header(text) or text
    seen: set = set()
    out: List[Dict[str, Any]] = []

    def _emit(content: str, etype: str) -> None:
        content = content.strip()
        if not content:
            return
        key = (etype, _canon(content))
        if key in seen:
            return
        seen.add(key)
        out.append(
            _record(
                kind="entity",
                content=content,
                entity_type=etype,
            )
        )

    # People: Slack U-id mentions and @name mentions.
    for m in _AT_MENTION_RE.finditer(body):
        person = m.group(1) or m.group(2)
        if person:
            _emit(person, "person")

    # Channels: <#C123|name> and bare #name.
    for m in _HASH_CHANNEL_RE.finditer(body):
        channel = m.group(1) or m.group(2)
        if channel:
            _emit(channel, "channel")

    # Code terms in backticks (likely services / repos / commands).
    for m in _CODE_TERM_RE.finditer(body):
        term = m.group(1)
        if "/" in term:
            _emit(term, "repository")
        elif term.lower() in _SERVICE_HINT_TERMS:
            _emit(term, "service")
        else:
            _emit(term, "system")

    # Proper-noun-style terms -> project, with "service" classification
    # when they match a known SaaS / infra word.
    for m in _PROPER_NOUN_RE.finditer(body):
        term = m.group(1).strip()
        if term in _NOUN_BLOCKLIST:
            continue
        if term.lower() in _SERVICE_HINT_TERMS:
            _emit(term, "service")
        else:
            _emit(term, "project")

    return out


# ---------------------------------------------------------------------- #
# Summarization
# ---------------------------------------------------------------------- #
# Heuristic: take the first 1-3 sentences of body text, capping at ~280
# characters total. This is intentionally NOT an LLM summary -- the
# extracted summary serves as a retrieval anchor, not as a polished
# answer. When the user asks "summarize X", the LLM at query time
# composes the polished response from this row + the underlying source.

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d])")
_SUMMARY_MAX_CHARS = 280
_SUMMARY_MAX_SENTENCES = 3


def summarize(text: str) -> Optional[str]:
    """
    Return a short summary, or None when the body is too short to
    bother. Short bodies (a one-liner Slack message) don't need a
    summary -- their full text already serves.
    """
    if not text:
        return None
    body = strip_markdown_header(text) or text
    body = body.strip()
    if len(body) < 80:
        # Single-line / very short messages -- no value in a summary.
        return None
    sentences = _SENTENCE_END_RE.split(body)
    pieces: List[str] = []
    total = 0
    for s in sentences[:_SUMMARY_MAX_SENTENCES]:
        s = " ".join(s.split())  # collapse internal whitespace
        if not s:
            continue
        if total + len(s) + 1 > _SUMMARY_MAX_CHARS:
            # Truncate the last sentence to fit.
            remaining = _SUMMARY_MAX_CHARS - total
            if remaining > 40:
                s = s[:remaining].rsplit(" ", 1)[0] + "…"
                pieces.append(s)
            break
        pieces.append(s)
        total += len(s) + 1
    summary = " ".join(pieces).strip()
    if not summary or len(summary) < 40:
        return None
    return summary


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #


def _clean_phrase(s: str) -> str:
    """
    Normalize a captured phrase: collapse whitespace, strip trailing
    punctuation noise + sentence-end punctuation, drop trailing
    parenthetical asides.
    """
    if not s:
        return ""
    s = " ".join(s.split())
    # Trim at the first sentence-ending mark so a captured group
    # doesn't extend into the next sentence.
    end = -1
    for ch in ".!?":
        idx = s.find(ch)
        if idx > 0 and (end == -1 or idx < end):
            end = idx
    if end > 0:
        s = s[:end]
    return s.strip(" \t\n,;:\"'`")


def _canon(s: str) -> str:
    """Canonical form used for dedupe + content_hash."""
    return " ".join(s.lower().split())


def _content_hash(content: str) -> str:
    """SHA-256 hex of the canonical form. Used by the persistence
    layer as part of the dedupe key."""
    return hashlib.sha256(_canon(content).encode("utf-8")).hexdigest()


def _record(
    *,
    kind: str,
    content: str,
    owner: Optional[str] = None,
    entity_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the persistence-layer-ready record shape."""
    return {
        "kind": kind,
        "content": content,
        "content_hash": _content_hash(content),
        "owner": owner,
        "entity_type": entity_type,
        "metadata": dict(metadata or {}),
    }
