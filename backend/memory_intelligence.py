"""
Phase 16: the Second Brain intelligence core.

Everything in this module is DERIVED -- on read -- from the existing
`extracted_memories` table. No new schema, no graph database, no
background jobs, no ML dependencies, no LLM calls. The table rows are
small per workspace (memories dedupe aggressively at ingest time), so
single-fetch + Python aggregation is the right cost model, exactly as
in analytics_intelligence.py.

Capabilities
------------
  (a) Entity resolution: a deterministic, explainable aliasing layer
      that canonicalizes ONE person across their Slack user ID
      (`<@U…>` / bare `U…`), `@mention`, display name, and email
      address forms -- plus case/punctuation variants of projects,
      services, and channels. Every merge carries a `rule` label so
      the alias map is fully traceable. No fuzzy/ML matching: a merge
      happens only when a stated rule fires, and ambiguity always
      means "do not merge".

  (b) Weighted relationship graph: entity<->entity co-occurrence
      within the same `source_stable_key`, weighted by
      recurrence x recency, with people<->projects,
      projects<->channels, projects<->decisions and
      people<->action_items surfaced and cross-source (Slack+Gmail)
      edges flagged.

  (c) compute_memory_importance(rows) -> {id: score in [0, 1]} from
      recurrence + recency + owner-presence + cluster-size. Used by
      recall.py to replace the flat 0.5 memory-candidate score with
      0.3 + 0.7 * importance. Reinforcement (recurrence) and decay
      (recency) act as RANKING adjustments only -- nothing is ever
      written back.

  (d) Project intelligence: active/dormant status, inferred owners,
      timeline, linked decisions, blockers, unresolved tasks.

  (e) Conversation reconstruction: walk backward from a decision
      through the pre-dating memories that share its entities.

  (f) Intelligence query routing: a regex intent classifier (no LLM)
      that detects ownership / status-blocker / decision-history /
      timeline questions and answers them from the functions above,
      citing source_stable_keys. Anything that doesn't match an
      intent -- or matches but has no supporting memory data --
      returns None so the caller falls through to the existing
      retrieval pipeline byte-identically.

Failure mode
------------
Every public function is workspace-scoped and defensive: a Supabase
error returns an empty result with a structured warning log.
Intelligence NEVER raises into a request path.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from logging_config import get_logger
from supabase_client import get_supabase

logger = get_logger(__name__)


# Recency decay half-life (days). A co-occurrence observed 30 days ago
# contributes half the weight of one observed today. Shared by the
# graph weighting and the importance score so "recency" means the same
# thing everywhere in this module.
_HALF_LIFE_DAYS = 30.0

# A project entity not mentioned for this many days is "dormant".
_DORMANT_AFTER_DAYS = 21

# Blocker-ish phrasing inside action items / summaries. Narrow on
# purpose -- false "blocker" labels are worse than missed ones.
_BLOCKER_RE = re.compile(
    r"\b(?:blocked|blocker|blocking|waiting\s+on|stuck\s+on|held\s+up|can(?:no|')t\s+proceed)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------- #
# Low-level: fetch a window's worth of memory rows once and reuse
# ---------------------------------------------------------------------- #
# Mirrors analytics_intelligence._fetch_memories (each derived-read
# module owns its fetch so tests can patch get_supabase at exactly one
# module boundary), but also selects `metadata` because alias evidence
# may live there in future ingest phases.


def _fetch_memories(
    *,
    workspace_id: str,
    days: int = 90,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Pull recent memory rows for a workspace. Single defensive read;
    every higher-level function below operates on this in-memory list
    so one intelligence request never hits Supabase more than once.
    """
    if not workspace_id:
        return []
    safe_days = max(1, min(int(days or 90), 365))
    cutoff_dt = datetime.now(timezone.utc)
    cutoff = (cutoff_dt - _timedelta_days(safe_days)).isoformat()
    try:
        client = get_supabase()
        resp = (
            client.table("extracted_memories")
            .select(
                "id, kind, content, owner, entity_type, source_kind, "
                "source_stable_key, source_timestamp, created_at, metadata"
            )
            .eq("workspace_id", workspace_id)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(max(1, min(int(limit or 5000), 10000)))
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "intelligence_fetch_memories_failed",
            extra={
                "workspace_id": workspace_id,
                "error": type(e).__name__,
            },
        )
        return []
    return list(getattr(resp, "data", None) or [])


def _timedelta_days(days: int):
    from datetime import timedelta  # noqa: PLC0415

    return timedelta(days=days)


def _parse_iso(ts: str) -> datetime:
    """Tolerant ISO 8601 parser (mirrors analytics_intelligence)."""
    if not isinstance(ts, str) or not ts.strip():
        raise ValueError("empty timestamp")
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _row_ts(row: Dict[str, Any]) -> str:
    """The row's best timestamp string (source first, created_at fallback)."""
    return row.get("source_timestamp") or row.get("created_at") or ""


def _recency_factor(ts: str, *, now: Optional[datetime] = None) -> float:
    """
    Exponential decay in [0, 1]: 1.0 for "right now", 0.5 at one
    half-life, and so on. Unparseable / missing timestamps get a
    neutral-low 0.25 so an undated row neither dominates nor vanishes.
    """
    if not ts:
        return 0.25
    try:
        dt = _parse_iso(ts)
    except ValueError:
        return 0.25
    now = now or datetime.now(timezone.utc)
    age_days = max(0.0, (now - dt).total_seconds() / 86400.0)
    return 0.5 ** (age_days / _HALF_LIFE_DAYS)


# ====================================================================== #
# (a) Entity resolution
# ====================================================================== #
# Deterministic union of alias forms. Identifier forms recognized for a
# person:
#
#     Slack user ID   "<@U0123ABCD>" or bare "U0123ABCD"
#     @-mention       "@rahul.verma"  (the extractor stores it without @)
#     display name    "Rahul Verma"
#     email address   "rahul.verma@acme.com"
#
# Merge rules (each merge is labelled with its rule for traceability):
#
#   token_match              Aliases whose normalized token tuples are
#                            identical merge. Email local parts,
#                            @-mentions and display names all normalize
#                            to the same tokens ("rahul.verma@acme.com",
#                            "@rahul_verma" and "Rahul Verma" are all
#                            ("rahul", "verma")). Case and . _ - /
#                            punctuation variants collapse here too.
#   first_name_unambiguous   A single-token alias ("Rahul") merges into
#                            a multi-token person ("Rahul Verma") ONLY
#                            when exactly one such person exists in the
#                            window. Two candidates -> no merge.
#   slack_id_adjacency       A Slack user ID merges with a named person
#                            only on explicit textual evidence inside a
#                            memory row: "Rahul Verma (<@U0123ABCD>)" or
#                            "<@U0123ABCD> (Rahul Verma)". Plain
#                            co-occurrence is NEVER enough to merge an
#                            opaque ID -- that would be a fuzzy guess.
#
# Projects / services / channels canonicalize on the token rule alone
# ("Project-Apollo" == "project apollo" == "ProjectApollo").

_SLACK_ID_RE = re.compile(r"^<?@?([UW][A-Z0-9]{4,})>?$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_PUNCT_RE = re.compile(r"[._\-/]+")

# Explicit ID<->name pairing evidence inside row content.
_ID_NAME_ADJ_RE = re.compile(
    r"<@([UW][A-Z0-9]{4,})>\s*\(\s*([A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){0,3})\s*\)"
)
_NAME_ID_ADJ_RE = re.compile(
    r"([A-Za-z][A-Za-z.'\-]*(?:\s+[A-Za-z][A-Za-z.'\-]*){0,3})\s*\(\s*<@([UW][A-Z0-9]{4,})>\s*\)"
)


def _person_signature(raw: str) -> Tuple[str, Any]:
    """
    Classify one raw person alias into a deterministic signature key.

    Returns ("id", "U0123ABCD") for Slack IDs, or ("tok", tokens)
    where tokens is the normalized token tuple for every other form.
    """
    s = (raw or "").strip()
    m = _SLACK_ID_RE.match(s)
    if m:
        return ("id", m.group(1).upper())
    if _EMAIL_RE.match(s):
        s = s.split("@", 1)[0]
    s = s.lstrip("@")
    return ("tok", _norm_tokens(s))


def _norm_tokens(raw: str) -> Tuple[str, ...]:
    """Case/punctuation-insensitive token tuple ("Project-Apollo" ->
    ("project", "apollo")). CamelCase splits too ("ProjectApollo")."""
    s = (raw or "").strip()
    s = _CAMEL_SPLIT_RE.sub(" ", s)
    s = _PUNCT_RE.sub(" ", s)
    return tuple(t for t in s.lower().split() if t)


class _UnionFind:
    """Tiny deterministic union-find over hashable keys."""

    def __init__(self) -> None:
        self._parent: Dict[Any, Any] = {}

    def add(self, key: Any) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: Any) -> Any:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Deterministic root choice: lexicographically smaller repr wins.
        if repr(ra) <= repr(rb):
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb

    def keys(self) -> List[Any]:
        return sorted(self._parent.keys(), key=repr)


def build_alias_map(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the explainable alias map from a list of memory rows.

    Returns:
      {
        "entities": {
            canonical_id: {
                "canonical":   str,            # display form
                "entity_type": str,
                "aliases":     [str, ...],     # every raw form seen
                "merge_rules": [{"alias": str, "rule": str}, ...],
                "mentions":    int,
            },
            ...
        },
        "lookup": { "<type>::<normalized alias>": canonical_id, ... },
      }

    `lookup` keys are strings so the whole map is JSON-serializable and
    can be exposed verbatim for traceability.
    """
    rows = rows or []

    # ---- collect raw aliases per entity bucket -----------------------
    # person aliases come from person-entities AND from owner fields;
    # project/service/channel aliases come from their entity rows.
    person_aliases: Counter = Counter()
    other_aliases: Dict[str, Counter] = defaultdict(Counter)  # type -> alias counter
    for r in rows:
        kind = r.get("kind")
        if kind == "entity":
            etype = (r.get("entity_type") or "").strip().lower()
            content = (r.get("content") or "").strip()
            if not content or not etype:
                continue
            if etype == "person":
                person_aliases[content] += 1
            else:
                other_aliases[etype][content] += 1
        owner = (r.get("owner") or "").strip()
        if owner:
            person_aliases[owner] += 1

    # ---- person union-find -------------------------------------------
    uf = _UnionFind()
    sig_of: Dict[str, Tuple[str, Any]] = {}
    merge_rules: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for alias in sorted(person_aliases):
        sig = _person_signature(alias)
        if sig == ("tok", ()):
            continue
        sig_of[alias] = sig
        uf.add(sig)
        merge_rules_key = repr(sig)
        merge_rules[merge_rules_key].append({"alias": alias, "rule": "token_match"})

    # first_name_unambiguous: single-token group -> the unique
    # multi-token group starting with that token.
    tok_sigs = sorted({s for s in sig_of.values() if s[0] == "tok"}, key=repr)
    multi = [s for s in tok_sigs if len(s[1]) >= 2]
    for single in [s for s in tok_sigs if len(s[1]) == 1]:
        first = single[1][0]
        candidates = [m for m in multi if m[1][0] == first]
        if len(candidates) == 1:
            uf.union(single, candidates[0])
            merge_rules[repr(candidates[0])].append({"alias": " ".join(single[1]), "rule": "first_name_unambiguous"})

    # slack_id_adjacency: explicit "<@U…> (Name)" / "Name (<@U…>)"
    # evidence inside row content.
    for r in rows:
        content = r.get("content") or ""
        if "<@" not in content:
            continue
        pairs: List[Tuple[str, str]] = []
        for m in _ID_NAME_ADJ_RE.finditer(content):
            pairs.append((m.group(1).upper(), m.group(2)))
        for m in _NAME_ID_ADJ_RE.finditer(content):
            pairs.append((m.group(2).upper(), m.group(1)))
        for uid, name in pairs:
            name_tokens = _norm_tokens(name)
            if not name_tokens:
                continue
            id_sig = ("id", uid)
            name_sig = ("tok", name_tokens)
            uf.add(id_sig)
            uf.add(name_sig)
            uf.union(id_sig, name_sig)
            merge_rules[repr(name_sig)].append({"alias": uid, "rule": "slack_id_adjacency"})
            # Make sure the evidence forms are present as aliases even
            # if they never appeared as standalone entity rows.
            person_aliases.setdefault(uid, 0)
            person_aliases.setdefault(name, 0)
            sig_of.setdefault(uid, id_sig)
            sig_of.setdefault(name, name_sig)

    # ---- materialize person groups ------------------------------------
    groups: Dict[Any, List[str]] = defaultdict(list)
    for alias in sorted(person_aliases):
        sig = sig_of.get(alias) or _person_signature(alias)
        if sig == ("tok", ()):
            continue
        groups[uf.find(sig)].append(alias)

    entities: Dict[str, Dict[str, Any]] = {}
    lookup: Dict[str, str] = {}

    for root in sorted(groups, key=repr):
        aliases = sorted(set(groups[root]))
        canonical = _pick_person_display(aliases)
        canonical_id = f"person::{' '.join(_norm_tokens(canonical)) or canonical.lower()}"
        rules: List[Dict[str, str]] = []
        seen_rule_keys = set()
        for alias in aliases:
            sig = sig_of.get(alias) or _person_signature(alias)
            for entry in merge_rules.get(repr(sig), []):
                key = (entry["alias"], entry["rule"])
                if key in seen_rule_keys:
                    continue
                seen_rule_keys.add(key)
                rules.append(entry)
        entities[canonical_id] = {
            "canonical": canonical,
            "entity_type": "person",
            "aliases": aliases,
            "merge_rules": sorted(rules, key=lambda d: (d["alias"], d["rule"])),
            "mentions": sum(person_aliases[a] for a in aliases),
        }
        for alias in aliases:
            sig = sig_of.get(alias) or _person_signature(alias)
            if sig[0] == "id":
                lookup[f"person::{sig[1].lower()}"] = canonical_id
            else:
                lookup[f"person::{' '.join(sig[1])}"] = canonical_id

    # ---- projects / services / channels / everything else -------------
    for etype in sorted(other_aliases):
        by_tokens: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        for alias in sorted(other_aliases[etype]):
            tokens = _norm_tokens(alias)
            if tokens:
                by_tokens[tokens].append(alias)
        for tokens in sorted(by_tokens):
            aliases = sorted(set(by_tokens[tokens]))
            counts = other_aliases[etype]
            # Most-mentioned original casing wins; alphabetical tie-break.
            canonical = sorted(aliases, key=lambda a: (-counts[a], a))[0]
            canonical_id = f"{etype}::{' '.join(tokens)}"
            entities[canonical_id] = {
                "canonical": canonical,
                "entity_type": etype,
                "aliases": aliases,
                "merge_rules": [{"alias": a, "rule": "token_match"} for a in aliases],
                "mentions": sum(counts[a] for a in aliases),
            }
            lookup[canonical_id] = canonical_id

    return {"entities": entities, "lookup": lookup}


def _pick_person_display(aliases: List[str]) -> str:
    """
    Deterministic display form: prefer multi-token display names over
    mentions/emails/IDs, more tokens over fewer, then alphabetical.
    """

    def rank(a: str) -> Tuple[int, int, str]:
        sig = _person_signature(a)
        is_opaque = 1 if (sig[0] == "id" or _EMAIL_RE.match(a) or a.startswith("@")) else 0
        token_count = len(sig[1]) if sig[0] == "tok" else 0
        return (is_opaque, -token_count, a.lower())

    return sorted(aliases, key=rank)[0]


def resolve_alias(alias_map: Dict[str, Any], raw: str, *, entity_type: str) -> str:
    """
    Map one raw identifier to its canonical_id via the alias map.
    Unknown identifiers fall back to a normalized self-key so callers
    can aggregate consistently even for never-merged singletons.
    """
    etype = (entity_type or "").strip().lower()
    if etype == "person":
        sig = _person_signature(raw)
        if sig[0] == "id":
            key = f"person::{sig[1].lower()}"
        else:
            key = f"person::{' '.join(sig[1])}"
        return (alias_map.get("lookup") or {}).get(key) or key
    tokens = _norm_tokens(raw)
    key = f"{etype}::{' '.join(tokens)}"
    return (alias_map.get("lookup") or {}).get(key) or key


# ====================================================================== #
# (b) Weighted relationship graph
# ====================================================================== #


def relationship_graph(
    *,
    workspace_id: str,
    days: int = 90,
    max_edges: int = 200,
) -> Dict[str, Any]:
    """
    Build the canonical entity<->entity co-occurrence graph.

    Nodes: resolved people / projects / services / channels (and the
    other extractor types), plus decision and action_item memories.
    Edges: two nodes appearing in the same `source_stable_key`.

        weight     = sum over supporting sources of recency_factor(ts)
                     (i.e. recurrence x recency in one number: each
                     extra co-occurrence adds weight; newer ones add
                     more)
        recurrence = number of distinct supporting sources
        cross_source = supporting sources span Slack AND Gmail

    Returns {"nodes": [...], "edges": [...], "alias_map": {...},
    "window_days": int}. Empty-but-valid shape on any failure.
    """
    rows = _fetch_memories(workspace_id=workspace_id, days=days)
    return build_relationship_graph(rows, days=days, max_edges=max_edges)


def build_relationship_graph(
    rows: List[Dict[str, Any]],
    *,
    days: int = 90,
    max_edges: int = 200,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Pure core of relationship_graph -- operates on already-fetched
    rows so tests and other intelligence functions can reuse it."""
    rows = rows or []
    alias_map = build_alias_map(rows)
    entities = alias_map["entities"]
    now = now or datetime.now(timezone.utc)

    # node_id -> {type, label, ...}; per-source node sets for edges.
    node_meta: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, Dict[str, Any]] = {}  # stable_key -> {"nodes": set, "kinds": set, "ts": str}

    def _node_for_entity(canonical_id: str) -> str:
        ent = entities.get(canonical_id)
        node_meta.setdefault(
            canonical_id,
            {
                "id": canonical_id,
                "type": (ent or {}).get("entity_type") or canonical_id.split("::", 1)[0],
                "label": (ent or {}).get("canonical") or canonical_id.split("::", 1)[-1],
                "aliases": (ent or {}).get("aliases") or [],
                "mentions": (ent or {}).get("mentions") or 0,
            },
        )
        return canonical_id

    for r in rows:
        sk = r.get("source_stable_key") or ""
        if not sk:
            continue
        bucket = by_source.setdefault(sk, {"nodes": set(), "kinds": set(), "ts": ""})
        src_kind = (r.get("source_kind") or "").strip().lower()
        if src_kind:
            bucket["kinds"].add(src_kind)
        ts = _row_ts(r)
        if ts > bucket["ts"]:
            bucket["ts"] = ts

        kind = r.get("kind")
        if kind == "entity":
            etype = (r.get("entity_type") or "").strip().lower()
            content = (r.get("content") or "").strip()
            if etype and content:
                bucket["nodes"].add(_node_for_entity(resolve_alias(alias_map, content, entity_type=etype)))
        elif kind in ("decision", "action_item"):
            node_id = f"{kind}::{r.get('id')}"
            node_meta.setdefault(
                node_id,
                {
                    "id": node_id,
                    "type": kind,
                    "label": (r.get("content") or "")[:140],
                    "aliases": [],
                    "mentions": 1,
                    "source_stable_key": sk,
                },
            )
            bucket["nodes"].add(node_id)
            # people<->action_items: the owner is a first-class link
            # even when the owner string never appears in the body.
            owner = (r.get("owner") or "").strip()
            if owner:
                bucket["nodes"].add(_node_for_entity(resolve_alias(alias_map, owner, entity_type="person")))

    # Pairwise edges per source, accumulated across sources.
    edge_weight: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_sources: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    edge_kinds: Dict[Tuple[str, str], set] = defaultdict(set)
    for sk in sorted(by_source):
        bucket = by_source[sk]
        nodes = sorted(bucket["nodes"])
        if len(nodes) < 2:
            continue
        factor = _recency_factor(bucket["ts"], now=now)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pair = (nodes[i], nodes[j])
                edge_weight[pair] += factor
                edge_sources[pair].append(sk)
                edge_kinds[pair].update(bucket["kinds"])

    edges: List[Dict[str, Any]] = []
    for pair in sorted(edge_weight, key=lambda p: (-edge_weight[p], p)):
        a, b = pair
        type_a = node_meta[a]["type"]
        type_b = node_meta[b]["type"]
        relation = "-".join(sorted((type_a, type_b)))
        kinds = edge_kinds[pair]
        edges.append(
            {
                "source": a,
                "target": b,
                "relation": relation,
                "weight": round(edge_weight[pair], 4),
                "recurrence": len(edge_sources[pair]),
                "sources": sorted(set(edge_sources[pair])),
                "cross_source": ("slack" in kinds and "gmail" in kinds),
            }
        )
        if len(edges) >= max(1, int(max_edges or 200)):
            break

    nodes_out = [node_meta[k] for k in sorted(node_meta)]
    return {
        "nodes": nodes_out,
        "edges": edges,
        "alias_map": alias_map,
        "window_days": max(1, min(int(days or 90), 365)),
    }


# ====================================================================== #
# (c) Memory importance
# ====================================================================== #
# score = 0.40 * recency          (exponential decay, 30-day half-life)
#       + 0.30 * recurrence        (same canonical content across N
#                                   distinct sources; reinforcement)
#       + 0.15 * owner presence    (owned items are actionable)
#       + 0.15 * cluster size      (rows co-extracted from a dense
#                                   source carry more context)
#
# All four signals live in [0, 1] so the weighted sum does too.
# Pure function of `rows` -- no I/O, no persistence: reinforcement and
# decay influence RANKING only.

_W_RECENCY = 0.40
_W_RECURRENCE = 0.30
_W_OWNER = 0.15
_W_CLUSTER = 0.15


def compute_memory_importance(rows: List[Dict[str, Any]]) -> Dict[Any, float]:
    """
    Map each row id to an importance score in [0, 1]. Rows without an
    id are skipped. Deterministic: identical input rows always produce
    identical scores (the only time-dependence is wall-clock recency,
    which is shared across the batch).
    """
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    if not rows:
        return {}
    now = datetime.now(timezone.utc)

    # recurrence: distinct sources sharing the row's canonical content.
    sources_by_content: Dict[str, set] = defaultdict(set)
    cluster_size: Counter = Counter()
    for r in rows:
        canon = " ".join((r.get("content") or "").lower().split())
        sk = r.get("source_stable_key") or ""
        if canon and sk:
            sources_by_content[canon].add(sk)
        if sk:
            cluster_size[sk] += 1

    out: Dict[Any, float] = {}
    for r in rows:
        rid = r.get("id")
        if rid is None:
            continue
        canon = " ".join((r.get("content") or "").lower().split())
        sk = r.get("source_stable_key") or ""
        recurrence_n = len(sources_by_content.get(canon, set())) or 1
        recurrence = min(1.0, (recurrence_n - 1) / 3.0)
        recency = _recency_factor(_row_ts(r), now=now)
        owner = 1.0 if (r.get("owner") or "").strip() else 0.0
        cluster = min(1.0, max(0, cluster_size.get(sk, 1) - 1) / 5.0)
        score = _W_RECENCY * recency + _W_RECURRENCE * recurrence + _W_OWNER * owner + _W_CLUSTER * cluster
        out[rid] = max(0.0, min(1.0, round(score, 6)))
    return out


# ====================================================================== #
# (d) Project intelligence
# ====================================================================== #


def project_intelligence(
    *,
    workspace_id: str,
    days: int = 90,
    dormant_days: int = _DORMANT_AFTER_DAYS,
) -> List[Dict[str, Any]]:
    """
    Infer per-project intelligence from the memory rows:

      [
        {
          "project":          canonical display name,
          "canonical_id":     "project::…",
          "aliases":          [...],
          "status":           "active" | "dormant",
          "first_seen":       ISO | None,
          "last_seen":        ISO | None,
          "owners":           [{"person", "canonical_id", "weight"}],
          "decisions":        [{"id","content","source_stable_key","timestamp"}],
          "blockers":         [...same shape...],
          "unresolved_tasks": [...same shape + "owner"...],
          "timeline":         [{"timestamp","kind","content","source_stable_key"}],
          "sources":          [stable keys],
        },
        ...
      ]

    Sorted by mentions descending then name, deterministic.
    """
    rows = _fetch_memories(workspace_id=workspace_id, days=days)
    return build_project_intelligence(rows, dormant_days=dormant_days)


def build_project_intelligence(
    rows: List[Dict[str, Any]],
    *,
    dormant_days: int = _DORMANT_AFTER_DAYS,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Pure core of project_intelligence."""
    rows = rows or []
    alias_map = build_alias_map(rows)
    entities = alias_map["entities"]
    now = now or datetime.now(timezone.utc)

    rows_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sk = r.get("source_stable_key") or ""
        if sk:
            rows_by_source[sk].append(r)

    # project canonical_id -> set of supporting stable keys
    project_sources: Dict[str, set] = defaultdict(set)
    for r in rows:
        if r.get("kind") != "entity" or (r.get("entity_type") or "").lower() != "project":
            continue
        content = (r.get("content") or "").strip()
        sk = r.get("source_stable_key") or ""
        if not content or not sk:
            continue
        project_sources[resolve_alias(alias_map, content, entity_type="project")].add(sk)

    out: List[Dict[str, Any]] = []
    for canonical_id in sorted(project_sources):
        ent = entities.get(canonical_id) or {}
        sources = sorted(project_sources[canonical_id])
        related = [r for sk in sources for r in rows_by_source.get(sk, [])]

        timestamps = sorted(ts for ts in (_row_ts(r) for r in related) if ts)
        first_seen = timestamps[0] if timestamps else None
        last_seen = timestamps[-1] if timestamps else None
        status = "active"
        if last_seen:
            try:
                age = (now - _parse_iso(last_seen)).days
                status = "dormant" if age > max(1, int(dormant_days)) else "active"
            except ValueError:
                status = "active"
        else:
            status = "dormant"

        # Owners: people co-occurring in project sources, weighted by
        # recency-weighted co-occurrence; action-item owners get an
        # extra full vote per owned item (owning a task in the project
        # is a stronger ownership signal than being mentioned).
        owner_weight: Dict[str, float] = defaultdict(float)
        for r in related:
            factor = _recency_factor(_row_ts(r), now=now)
            if r.get("kind") == "entity" and (r.get("entity_type") or "").lower() == "person":
                cid = resolve_alias(alias_map, (r.get("content") or ""), entity_type="person")
                owner_weight[cid] += factor
            owner = (r.get("owner") or "").strip()
            if owner:
                cid = resolve_alias(alias_map, owner, entity_type="person")
                owner_weight[cid] += 2.0 * factor
        owners = [
            {
                "person": (entities.get(cid) or {}).get("canonical") or cid.split("::", 1)[-1],
                "canonical_id": cid,
                "weight": round(w, 4),
            }
            for cid, w in sorted(owner_weight.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        ]

        def _cite(r: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": r.get("id"),
                "content": r.get("content"),
                "source_stable_key": r.get("source_stable_key"),
                "timestamp": _row_ts(r) or None,
            }

        decisions = sorted(
            (_cite(r) for r in related if r.get("kind") == "decision"),
            key=lambda d: (d["timestamp"] or "", str(d["id"])),
        )
        blockers = sorted(
            (
                _cite(r)
                for r in related
                if r.get("kind") in ("action_item", "summary") and _BLOCKER_RE.search(r.get("content") or "")
            ),
            key=lambda d: (d["timestamp"] or "", str(d["id"])),
        )

        # Unresolved tasks: action items with no LATER decision inside
        # the same project's sources. Loose by design (same heuristic
        # family as analytics_intelligence's stale-action check) and
        # explainable: "no decision in this project postdates it".
        latest_decision_ts = max((d["timestamp"] or "" for d in decisions), default="")
        unresolved = []
        for r in related:
            if r.get("kind") != "action_item":
                continue
            ts = _row_ts(r)
            if latest_decision_ts and ts and latest_decision_ts > ts:
                continue
            unresolved.append({**_cite(r), "owner": r.get("owner") or None})
        unresolved.sort(key=lambda d: (d["timestamp"] or "", str(d["id"])))

        timeline = sorted(
            (
                {
                    "timestamp": _row_ts(r) or None,
                    "kind": r.get("kind"),
                    "content": r.get("content"),
                    "source_stable_key": r.get("source_stable_key"),
                }
                for r in related
                if r.get("kind") in ("decision", "action_item", "summary")
            ),
            key=lambda d: (d["timestamp"] or "", str(d["content"])),
        )[:50]

        out.append(
            {
                "project": ent.get("canonical") or canonical_id.split("::", 1)[-1],
                "canonical_id": canonical_id,
                "aliases": ent.get("aliases") or [],
                "status": status,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "owners": owners,
                "decisions": decisions,
                "blockers": blockers,
                "unresolved_tasks": unresolved,
                "timeline": timeline,
                "sources": sources,
            }
        )

    out.sort(key=lambda p: (-((entities.get(p["canonical_id"]) or {}).get("mentions") or 0), p["project"]))
    return out


# ====================================================================== #
# (e) Conversation reconstruction
# ====================================================================== #


def reconstruct_conversation(
    *,
    workspace_id: str,
    decision: str,
    days: int = 180,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Walk backward from a decision: find the best-matching decision row,
    then collect the pre-dating memories that share its source or its
    entities, in chronological order. Result:

      {
        "decision":  {...decision row cite...} | None,
        "steps":     [{"timestamp","kind","content","owner",
                       "source_stable_key"}, ...]  (oldest first),
        "entities":  [canonical labels anchoring the walk],
      }

    Empty-but-valid shape when nothing matches or on any failure.
    """
    rows = _fetch_memories(workspace_id=workspace_id, days=days)
    return build_conversation_reconstruction(rows, decision=decision, limit=limit)


def build_conversation_reconstruction(
    rows: List[Dict[str, Any]],
    *,
    decision: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """Pure core of reconstruct_conversation."""
    empty = {"decision": None, "steps": [], "entities": []}
    needle = (decision or "").strip().lower()
    rows = rows or []
    if not needle or not rows:
        return empty

    candidates = [r for r in rows if r.get("kind") == "decision" and needle in (r.get("content") or "").lower()]
    if not candidates:
        return empty
    # Most recent matching decision wins; deterministic tie-break on id.
    anchor = sorted(candidates, key=lambda r: (_row_ts(r), str(r.get("id"))))[-1]
    anchor_sk = anchor.get("source_stable_key") or ""
    anchor_ts = _row_ts(anchor)

    alias_map = build_alias_map(rows)
    entities = alias_map["entities"]

    def _source_entity_ids(sk: str) -> set:
        ids = set()
        for r in rows:
            if (r.get("source_stable_key") or "") != sk:
                continue
            if r.get("kind") == "entity":
                etype = (r.get("entity_type") or "").lower()
                content = (r.get("content") or "").strip()
                if etype and content:
                    ids.add(resolve_alias(alias_map, content, entity_type=etype))
            owner = (r.get("owner") or "").strip()
            if owner:
                ids.add(resolve_alias(alias_map, owner, entity_type="person"))
        return ids

    anchor_entities = _source_entity_ids(anchor_sk)

    steps: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("id") == anchor.get("id"):
            continue
        ts = _row_ts(r)
        if anchor_ts and ts and ts > anchor_ts:
            continue  # only walk BACKWARD from the decision
        sk = r.get("source_stable_key") or ""
        if sk != anchor_sk and not (anchor_entities and anchor_entities & _source_entity_ids(sk)):
            continue
        if r.get("kind") == "entity":
            continue  # entities anchor the walk; the steps are the substance
        steps.append(
            {
                "timestamp": ts or None,
                "kind": r.get("kind"),
                "content": r.get("content"),
                "owner": r.get("owner") or None,
                "source_stable_key": sk,
            }
        )
    steps.sort(key=lambda s: (s["timestamp"] or "", str(s["content"])))
    steps = steps[-max(1, min(int(limit or 20), 100)) :]

    return {
        "decision": {
            "id": anchor.get("id"),
            "content": anchor.get("content"),
            "source_stable_key": anchor_sk,
            "timestamp": anchor_ts or None,
        },
        "steps": steps,
        "entities": sorted((entities.get(e) or {}).get("canonical") or e.split("::", 1)[-1] for e in anchor_entities),
    }


# ====================================================================== #
# (f) Intelligence query routing
# ====================================================================== #
# Narrow, explicit-subject patterns only. The router must NEVER hijack
# a question that the semantic pipeline already serves well, so forms
# without a named subject ("what is the sprint status?", "what
# happened?") deliberately fall through. The fall-through contract is
# absolute: classify -> None means the caller's behavior is
# byte-identical to a build without this module.

_INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("ownership", re.compile(r"\bwho\s+owns\s+(?P<subject>.{2,80})", re.IGNORECASE)),
    (
        "ownership",
        re.compile(r"\bwho\s+is\s+(?:responsible\s+for|the\s+owner\s+of)\s+(?P<subject>.{2,80})", re.IGNORECASE),
    ),
    ("ownership", re.compile(r"\bwho(?:'s|\s+is)\s+(?:working|leading)\s+on\s+(?P<subject>.{2,80})", re.IGNORECASE)),
    (
        "status_blocker",
        re.compile(r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?status\s+of\s+(?P<subject>.{2,80})", re.IGNORECASE),
    ),
    ("status_blocker", re.compile(r"\bwhat(?:'s|\s+is)\s+blocking\s+(?P<subject>.{2,80})", re.IGNORECASE)),
    ("status_blocker", re.compile(r"\b(?:any\s+)?blockers?\s+(?:for|on)\s+(?P<subject>.{2,80})", re.IGNORECASE)),
    ("status_blocker", re.compile(r"\bis\s+(?P<subject>.{2,60}?)\s+blocked\b", re.IGNORECASE)),
    (
        "decision_history",
        re.compile(
            r"\bwhy\s+did\s+we\s+(?:decide(?:\s+(?:to|on))?|choose|pick|go\s+with)\s+(?P<subject>.{2,80})",
            re.IGNORECASE,
        ),
    ),
    (
        "decision_history",
        re.compile(
            r"\bwhat\s+led\s+to\s+(?:the\s+decision\s+(?:to|on|about)\s+)?(?P<subject>.{2,80})",
            re.IGNORECASE,
        ),
    ),
    (
        "decision_history",
        re.compile(r"\bhistory\s+of\s+the\s+decision\s+(?:to|on|about)\s+(?P<subject>.{2,80})", re.IGNORECASE),
    ),
    ("decision_history", re.compile(r"\bhow\s+was\s+(?P<subject>.{2,60}?)\s+decided\b", re.IGNORECASE)),
    ("timeline", re.compile(r"\b(?:timeline|chronology)\s+(?:of|for)\s+(?P<subject>.{2,80})", re.IGNORECASE)),
]


def classify_intelligence_intent(question: str) -> Optional[Dict[str, str]]:
    """
    Pure regex classifier. Returns {"intent": ..., "subject": ...} or
    None. No I/O, no LLM. First matching pattern wins (the pattern
    list order is part of the contract; tests pin it).
    """
    q = (question or "").strip()
    if not q:
        return None
    for intent, pattern in _INTENT_PATTERNS:
        m = pattern.search(q)
        if not m:
            continue
        subject = _clean_subject(m.group("subject"))
        if subject:
            return {"intent": intent, "subject": subject}
    return None


def _clean_subject(raw: str) -> str:
    """Trim question-mark/quote/article noise off a captured subject."""
    s = " ".join((raw or "").split())
    s = s.strip(" \t\"'?.!,;:")
    s = re.sub(r"^(?:the|our|a|an)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(?:project|right\s+now|currently|these\s+days)$", "", s, flags=re.IGNORECASE).strip(" ?.!,")
    return s


def route_intelligence_query(
    *,
    workspace_id: str,
    question: str,
    days: int = 90,
) -> Optional[Dict[str, Any]]:
    """
    Try to answer `question` from structured memory intelligence.

    Returns an /api/query-shaped dict:

        {"answer": str, "sources": [...], "debug": {...}}

    with every source citing its `source_stable_key` -- or None when:
      - the question matches no intelligence intent (zero I/O happens
        in that case), or
      - the intent matched but the workspace has no supporting memory
        rows for the subject, or
      - anything at all fails (defensive).

    None ALWAYS means "fall through to the existing retrieval pipeline
    unchanged".
    """
    try:
        classified = classify_intelligence_intent(question)
        if not classified or not workspace_id:
            return None
        # Only fetch once an intent matched -- non-matching questions
        # must cost nothing.
        rows = _fetch_memories(workspace_id=workspace_id, days=days)
        if not rows:
            return None
        intent = classified["intent"]
        subject = classified["subject"]
        if intent in ("ownership", "status_blocker", "timeline"):
            return _answer_from_projects(rows, intent=intent, subject=subject)
        if intent == "decision_history":
            return _answer_decision_history(rows, subject=subject)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "intelligence_routing_failed",
            extra={
                "workspace_id": workspace_id,
                "error": type(e).__name__,
            },
        )
        return None


def _match_project(projects: List[Dict[str, Any]], subject: str) -> Optional[Dict[str, Any]]:
    """Deterministic subject->project match on normalized tokens
    (exact tokens first, then subset containment)."""
    subject_tokens = set(_norm_tokens(subject))
    if not subject_tokens:
        return None
    for p in projects:
        if set(_norm_tokens(p["project"])) == subject_tokens:
            return p
    for p in projects:
        ptokens = set(_norm_tokens(p["project"]))
        if ptokens and (ptokens <= subject_tokens or subject_tokens <= ptokens):
            return p
    return None


def _cards_from_cites(cites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build /api/query-style source cards from cite dicts, numbered
    1..N and deduped on stable key."""
    cards: List[Dict[str, Any]] = []
    seen = set()
    for c in cites:
        sk = c.get("source_stable_key") or ""
        if not sk or sk in seen:
            continue
        seen.add(sk)
        cards.append(
            {
                "index": len(cards) + 1,
                "source": sk,
                "stable_key": sk,
                "source_kind": ("gmail" if sk.startswith("gmail") else "slack" if sk.startswith("slack") else None),
                "document_type": "memory_intelligence",
                "timestamp": c.get("timestamp"),
            }
        )
    return cards


def _result(answer: str, cites: List[Dict[str, Any]], *, intent: str, subject: str) -> Dict[str, Any]:
    return {
        "answer": answer,
        "sources": _cards_from_cites(cites),
        "debug": {
            "routed": "memory_intelligence",
            "intelligence_intent": intent,
            "subject": subject,
        },
    }


def _answer_from_projects(rows: List[Dict[str, Any]], *, intent: str, subject: str) -> Optional[Dict[str, Any]]:
    projects = build_project_intelligence(rows)
    project = _match_project(projects, subject)
    if not project:
        return None
    name = project["project"]

    if intent == "ownership":
        if not project["owners"]:
            return None
        names = [o["person"] for o in project["owners"]]
        lead, rest = names[0], names[1:]
        answer = f"{name} is owned by {lead}" + (f" (also involved: {', '.join(rest)})" if rest else "") + "."
        cites = project["decisions"] + project["unresolved_tasks"]
        cites = cites or [{"source_stable_key": sk, "timestamp": None} for sk in project["sources"]]
        return _result(answer, cites, intent=intent, subject=subject)

    if intent == "status_blocker":
        parts = [f"{name} is {project['status']}"]
        if project["last_seen"]:
            parts[0] += f" (last activity {project['last_seen'][:10]})"
        if project["blockers"]:
            parts.append("blockers: " + "; ".join((b["content"] or "")[:120] for b in project["blockers"][:3]))
        if project["unresolved_tasks"]:
            parts.append(
                "open tasks: " + "; ".join((t["content"] or "")[:120] for t in project["unresolved_tasks"][:3])
            )
        answer = ". ".join(parts) + "."
        cites = project["blockers"] + project["unresolved_tasks"] + project["decisions"]
        cites = cites or [{"source_stable_key": sk, "timestamp": None} for sk in project["sources"]]
        return _result(answer, cites, intent=intent, subject=subject)

    # timeline
    if not project["timeline"]:
        return None
    lines = [
        f"{(e['timestamp'] or '')[:10] or 'undated'} — {e['kind']}: {(e['content'] or '')[:120]}"
        for e in project["timeline"][:8]
    ]
    answer = f"Timeline for {name}:\n" + "\n".join(lines)
    return _result(answer, project["timeline"], intent=intent, subject=subject)


def _answer_decision_history(rows: List[Dict[str, Any]], *, subject: str) -> Optional[Dict[str, Any]]:
    recon = build_conversation_reconstruction(rows, decision=subject)
    if not recon["decision"]:
        return None
    decision = recon["decision"]
    lines = [f"Decision: {decision['content']}"]
    if recon["steps"]:
        lines.append("Leading up to it:")
        for s in recon["steps"][-6:]:
            lines.append(f"- {(s['timestamp'] or '')[:10] or 'undated'} {s['kind']}: {(s['content'] or '')[:120]}")
    answer = "\n".join(lines)
    cites = recon["steps"] + [decision]
    return _result(answer, cites, intent="decision_history", subject=subject)
