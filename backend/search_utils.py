"""
Keyword extraction + chunk ranking utilities.

Used by recall.py to support:
  - mode="exact":  prefer chunks that match query terms verbatim; if no
                   exact matches exist, fall back to semantic order but
                   report exact_matches_found=0.
  - mode="hybrid": combine semantic order with an exact-match bonus.

All functions here are pure and side-effect-free so they're trivially
unit-testable.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Words too common to be useful as exact-match terms. Short list — we don't
# need NLP-grade stopword removal, just enough to stop noise like "the" or
# "what" from dominating the ranking.
STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "did",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "should",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "too",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "about",
    "any",
    "been",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "done",
    "go",
    "going",
}

# Tokens are alphanumeric + underscore + hyphen.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-]*")
# Phrases enclosed in double quotes: '"exact phrase"' -> "exact phrase".
_QUOTED_PHRASE_RE = re.compile(r'"([^"]+)"')


def extract_query_terms(question: str) -> List[str]:
    """
    Pull keyword terms (and quoted phrases) out of a question.

    Rules:
      - Quoted phrases are extracted whole.
      - Outside quotes, alphanumeric tokens >= 2 chars are kept, lowercased.
      - English stopwords are dropped.
      - Order is preserved; duplicates are removed (keeping first occurrence).
    """
    if not question:
        return []

    terms: List[str] = []
    seen: Set[str] = set()

    # 1. Quoted phrases first.
    remainder = question
    for match in _QUOTED_PHRASE_RE.finditer(question):
        phrase = match.group(1).strip().lower()
        if phrase and phrase not in seen:
            terms.append(phrase)
            seen.add(phrase)
    # Strip the quoted portions out of remainder so they aren't re-tokenized.
    remainder = _QUOTED_PHRASE_RE.sub(" ", remainder)

    # 2. Bare tokens.
    for tok_match in _TOKEN_RE.finditer(remainder):
        tok = tok_match.group(0).lower()
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        if tok in seen:
            continue
        terms.append(tok)
        seen.add(tok)

    return terms


def count_keyword_hits(text: str, terms: Iterable[str]) -> int:
    """
    Count how many distinct query terms appear in the text.

    - Quoted phrases (multi-word terms) are looked up as a substring.
    - Single-word terms are matched as whole words (so "ai" won't match
      "again").

    Returns a per-term hit count: each matching term contributes 1, even if
    it appears multiple times. We want diversity of matches, not raw freq.
    """
    if not text:
        return 0
    hay = text.lower()
    hits = 0
    for term in terms:
        if not term:
            continue
        if " " in term:
            # multi-word / quoted phrase -> substring match
            if term in hay:
                hits += 1
        else:
            # single token -> word-boundary match
            pattern = r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])"
            if re.search(pattern, hay):
                hits += 1
    return hits


def _ts_to_float(value: Any) -> Optional[float]:
    """Slack ts string or numeric -> float seconds, or None."""
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
# Ranking weights (Phase 10)
# ---------------------------------------------------------------------- #
# Explicit, named, tunable. All ranking decisions in this module use
# these constants -- no magic numbers in the sort keys. The defaults
# were picked so the relative ordering reproduces the Phase 9
# behavior for legacy callers (channel + user metadata bias as the
# only signal beyond keyword hits + recency).
#
# In the hybrid mode sort, the final score for a chunk is:
#
#   score = (keyword_hits      * W_KEYWORD_HIT)
#         + (subject_hits      * W_SUBJECT_HIT)
#         + (channel_match     * W_CHANNEL_MATCH)
#         + (sender_match      * W_SENDER_MATCH)
#         + (label_match       * W_LABEL_MATCH)
#         + (normalized_recency * W_RECENCY)
#
# Body-text hits stay the dominant signal (a verbatim keyword match
# in the doc body is the strongest evidence of relevance we have).
# Subject hits are a strong second because email subjects are
# semantically loaded. Sender/channel matches are weak filters --
# they refine ordering when the question implies a person or
# channel, but never override raw relevance. Recency is the gentlest
# signal: it acts as a tiebreaker, not a primary driver, outside
# the dedicated recency-rerank mode (which uses its own pure
# timestamp sort).
W_KEYWORD_HIT = 100
W_SUBJECT_HIT = 80
W_CHANNEL_MATCH = 50
W_SENDER_MATCH = 50
W_LABEL_MATCH = 30
W_RECENCY = 1.0


def _string_match_ci(a: Any, b: Any) -> bool:
    """Case-insensitive equality, robust to non-string inputs."""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a.strip().lower() == b.strip().lower()


def _metadata_bias_score(
    source_card: Dict[str, Any],
    bias: Optional[Dict[str, str]],
) -> int:
    """
    Count how many fields of `bias` (typically {"channel": ..., "user": ...})
    match the chunk's source card. Comparisons are case-insensitive on
    string values; missing card fields don't penalize, they just don't
    contribute.

    Phase 10: when `bias["user"]` is set, we ALSO check Gmail sender
    fields (`from_name`, `from_email`) so a question like "what did
    Rahul say about Kafka" promotes both Slack messages from Rahul
    AND Gmail emails from Rahul. The boost stays at the same
    per-field magnitude either way -- a sender match contributes
    exactly one point, same as a Slack user match -- so the existing
    Slack-only tests keep their relative ordering.

    Returns an integer 0..len(bias) -- used by the sort key in
    rerank_chunks to push matching chunks up.
    """
    if not bias:
        return 0
    score = 0
    for key, value in bias.items():
        if not value:
            continue
        if key == "label":
            # Gmail labels are scored independently by _label_match_score
            # and weighted by W_LABEL_MATCH -- they must NOT be folded
            # into the channel/user bias magnitude (which is weighted by
            # W_CHANNEL_MATCH in hybrid). Skip here.
            continue
        if key == "user":
            # Slack: card["user"] is the user_name. Gmail: the sender
            # lives in from_name OR from_email (we accept either). Any
            # one match counts.
            if (
                _string_match_ci(source_card.get("user"), value)
                or _string_match_ci(source_card.get("from_name"), value)
                or _string_match_ci(source_card.get("from_email"), value)
            ):
                score += 1
            continue
        card_value = source_card.get(key)
        if isinstance(card_value, str) and isinstance(value, str):
            if card_value.lower() == value.lower():
                score += 1
    return score


def _subject_keyword_hits(source_card: Dict[str, Any], terms: List[str]) -> int:
    """
    Count how many query terms appear in the card's subject (Gmail)
    or in a Slack equivalent slot if one ever gets attached. Slack
    docs don't carry an explicit subject today, so this is
    effectively a Gmail-only boost -- emails whose subject lines
    contain the query keywords surface higher than emails whose only
    matches are in the body. Returns 0 when nothing applies.
    """
    if not terms:
        return 0
    subject = source_card.get("subject")
    if not isinstance(subject, str) or not subject:
        return 0
    return count_keyword_hits(subject, terms)


# --- Gmail label vocabulary + inference ------------------------------------
# Maps natural-language label words to Gmail's canonical label IDs. Only
# Gmail SYSTEM labels are inferable from words, because the ingestion path
# stores label IDs (not user-label display names) on the document. User
# labels (opaque "Label_<n>" ids) therefore can't be matched by name -- a
# known limitation, but the system tabs below cover the common asks.
_GMAIL_LABEL_VOCAB: Tuple[Tuple[str, str], ...] = (
    ("inbox", "INBOX"),
    ("starred", "STARRED"),
    ("important", "IMPORTANT"),
    ("unread", "UNREAD"),
    ("sent", "SENT"),
    ("drafts", "DRAFT"),
    ("draft", "DRAFT"),
    ("promotions", "CATEGORY_PROMOTIONS"),
    ("promotion", "CATEGORY_PROMOTIONS"),
    ("updates", "CATEGORY_UPDATES"),
    ("social", "CATEGORY_SOCIAL"),
    ("forums", "CATEGORY_FORUMS"),
    ("forum", "CATEGORY_FORUMS"),
    ("spam", "SPAM"),
    ("trash", "TRASH"),
)

# A label is only inferred when the question is clearly email-scoped, so
# generic words ("an important decision", "social media") don't trigger a
# label bias. One of these tokens must be present.
_EMAIL_CONTEXT_RE = re.compile(r"\b(e-?mails?|mails?|inbox|gmail|messages?|msgs?)\b", re.IGNORECASE)


def infer_gmail_label(question: str) -> Optional[str]:
    """
    Infer a Gmail SYSTEM label id from a natural-language question, or
    None when no confident label is implied.

    Conservative by design: returns a label only when (a) the question
    carries an email-context token (email/mail/inbox/message/gmail) AND
    (b) it mentions a known label word. This keeps "important emails" ->
    IMPORTANT while leaving "an important question" untouched. The result
    feeds metadata_bias["label"], a ranking-only signal (W_LABEL_MATCH) --
    it never hard-filters, so a rare false positive only nudges order.

    Pure and side-effect-free, like the rest of this module.
    """
    if not question:
        return None
    q = question.lower()
    if not _EMAIL_CONTEXT_RE.search(q):
        return None
    for phrase, label_id in _GMAIL_LABEL_VOCAB:
        if re.search(r"\b" + re.escape(phrase) + r"\b", q):
            return label_id
    return None


def _label_match_score(source_card: Dict[str, Any], bias: Optional[Dict[str, str]]) -> int:
    """
    1 when the card's Gmail labels contain the biased label id, else 0.

    `bias["label"]` is a single Gmail label id (e.g. "INBOX",
    "CATEGORY_PROMOTIONS"). The card's `labels` is the list[str] of label
    ids harvested from the email document header. Comparison is
    case-insensitive. Slack cards (no `labels`) always score 0, so this
    signal is inert outside Gmail.
    """
    if not bias:
        return 0
    wanted = bias.get("label")
    if not wanted or not isinstance(wanted, str):
        return 0
    labels = source_card.get("labels")
    if not isinstance(labels, list):
        return 0
    target = wanted.strip().lower()
    for lid in labels:
        if isinstance(lid, str) and lid.strip().lower() == target:
            return 1
    return 0


def rerank_chunks(
    chunks_with_meta: List[Dict[str, Any]],
    terms: List[str],
    mode: str,
    top_k: int,
    metadata_bias: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Rerank a list of `{text, source_card, original_index, timestamp_float}`
    dicts according to `mode`, with an optional `metadata_bias`.

    Modes:
      - "exact":  Keep chunks with >=1 keyword hit, sorted by hit-count then
                  by metadata bias, then timestamp (newest first). If no
                  chunk has any hit, fall back to semantic order biased by
                  metadata.
      - "hybrid": Score each chunk with the configurable weights
                  (W_KEYWORD_HIT, W_SUBJECT_HIT, W_SENDER_MATCH, etc).
                  Even chunks with 0 hits stay, deprioritized.
      - anything else (incl. "default"): preserve semantic order, but push
                  chunks matching `metadata_bias` ahead of the rest. This
                  lets person/channel inference improve results in modes
                  that don't otherwise rerank.

    `metadata_bias` looks like `{"channel": "product", "user": "rahul"}`.
    Each matching key adds to a per-chunk bias score. Comparisons are
    case-insensitive; non-string card values are ignored. Phase 10:
    the `user` key also matches Gmail sender fields (`from_name`,
    `from_email`).

    Side effect: every chunk gets an internal `_debug_score` dict
    populated, capturing the per-signal breakdown the sort key uses.
    This is what powers the optional `rank_breakdown` debug payload
    in prepare_recall_context; it is NOT included in the public
    sources[] API response.

    Returns (ranked_chunks, exact_matches_found).
      `ranked_chunks` is capped at `top_k`. `exact_matches_found` is the
      count of chunks with >=1 keyword hit (whether kept or not).
    """
    if not chunks_with_meta:
        return [], 0

    # Annotate every chunk with its per-signal scores so the
    # mode-specific sort keys below can use them. Also stash a
    # per-chunk debug breakdown so prepare_recall_context can surface
    # the ranking rationale in logs and the (private) debug payload.
    max_ts = max((c.get("timestamp_float") or 0.0) for c in chunks_with_meta) or 1.0
    for chunk in chunks_with_meta:
        card = chunk.get("source_card", {}) or {}
        hits = count_keyword_hits(chunk.get("text", ""), terms)
        subj_hits = _subject_keyword_hits(card, terms)
        bias = _metadata_bias_score(card, metadata_bias)
        label_bias = _label_match_score(card, metadata_bias)
        ts = chunk.get("timestamp_float") or 0.0
        chunk["_hits"] = hits
        chunk["_subject_hits"] = subj_hits
        chunk["_bias"] = bias
        chunk["_label_bias"] = label_bias
        chunk["_debug_score"] = {
            "keyword_hits": hits,
            "subject_hits": subj_hits,
            "metadata_bias": bias,
            "label_match": label_bias,
            "timestamp": ts,
            "normalized_recency": (ts / max_ts) if max_ts else 0.0,
        }
    matched_count = sum(1 for c in chunks_with_meta if c["_hits"] > 0)

    if mode == "exact":
        matched = [c for c in chunks_with_meta if c["_hits"] > 0]
        if matched:
            # Sort key: (hits desc, subject-hits desc, bias desc, newer
            # first, stable index). Subject hits act as a secondary
            # signal -- two chunks with the same body-hit count, but
            # one with the keyword in the subject too, surfaces first.
            matched.sort(
                key=lambda c: (
                    -c["_hits"],
                    -c["_subject_hits"],
                    -c["_bias"],
                    -c["_label_bias"],
                    -(c.get("timestamp_float") or 0.0),
                    c["original_index"],
                ),
            )
            return matched[:top_k], matched_count
        # No exact matches: fall back to semantic order but still let
        # the metadata bias surface relevant chunks above unrelated ones.
        ranked = sorted(
            chunks_with_meta,
            key=lambda c: (
                -c["_bias"],
                -c["_label_bias"],
                c["original_index"],
            ),
        )
        return ranked[:top_k], 0

    if mode == "hybrid":
        # Composite score using the named weights. Stable: the
        # original_index disambiguator means two chunks with the same
        # composite score keep semantic order.
        def _hybrid_score(c: Dict[str, Any]) -> float:
            ts = c.get("timestamp_float") or 0.0
            return (
                c["_hits"] * W_KEYWORD_HIT
                + c["_subject_hits"] * W_SUBJECT_HIT
                + c["_bias"] * W_CHANNEL_MATCH  # generic bias-magnitude
                + c["_label_bias"] * W_LABEL_MATCH
                + (ts / max_ts) * W_RECENCY
            )

        for c in chunks_with_meta:
            c["_debug_score"]["hybrid_score"] = _hybrid_score(c)
        ranked = sorted(
            chunks_with_meta,
            key=lambda c: (-_hybrid_score(c), c["original_index"]),
        )
        return ranked[:top_k], matched_count

    # Default mode + summary/decisions/etc. — preserve semantic order, but
    # still let metadata bias push matching chunks up. When bias is zero
    # for everything (no inference), the sort collapses to original_index
    # which is identical to the input order.
    ranked = sorted(
        chunks_with_meta,
        key=lambda c: (
            -c["_bias"],
            -c["_label_bias"],
            c["original_index"],
        ),
    )
    return ranked[:top_k], matched_count


def dedupe_by_stable_key(chunks_with_meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop later chunks whose stable_key was already seen.
    First occurrence wins (best score-rank for the document).
    """
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for chunk in chunks_with_meta:
        key = chunk.get("source_card", {}).get("stable_key")
        if key:
            if key in seen:
                continue
            seen.add(key)
        out.append(chunk)
    return out