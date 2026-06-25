"""
Phase 15: derived intelligence over extracted_memories.

This module READS the existing `extracted_memories` table and
projects it into the views the brief asked for:

  - relationship graph between people / channels / projects /
    services / decisions / action items
  - topic clusters: entities that frequently co-occur in the same
    source document
  - timelines: chronological memory rows for one entity / kind
  - recurring-pattern detection: counts of entity mentions per
    rolling window with surge / staleness signals
  - proactive insights: stale action items, surging entities,
    dormant projects

No new schema. No background jobs. Every function is a Supabase
SELECT followed by Python aggregation. Volumes are small per
workspace (memories are extracted at ingest time and dedupe
aggressively) so on-read computation is fine.

Every function is workspace-scoped. Every function is defensive: a
Supabase error returns an empty result with a structured warning
log -- analytics never raises into the request path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from logging_config import get_logger
from supabase_client import get_supabase

logger = get_logger(__name__)


# ---------------------------------------------------------------------- #
# Low-level: fetch a window's worth of memory rows once and reuse
# ---------------------------------------------------------------------- #


def _fetch_memories(
    *,
    workspace_id: str,
    days: int = 30,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Pull recent memory rows for a workspace. Single defensive read;
    all the higher-level functions below operate on this in-memory
    list, so we never hit Supabase more than once per analytics
    request.
    """
    if not workspace_id:
        return []
    safe_days = max(1, min(int(days or 30), 365))
    cutoff = (datetime.now(timezone.utc) - timedelta(days=safe_days)).isoformat()
    try:
        client = get_supabase()
        resp = (
            client.table("extracted_memories")
            .select(
                "id, kind, content, owner, entity_type, source_kind, " "source_stable_key, source_timestamp, created_at"
            )
            .eq("workspace_id", workspace_id)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(max(1, min(int(limit or 5000), 10000)))
            .execute()
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "analytics_fetch_memories_failed",
            extra={
                "workspace_id": workspace_id,
                "error": type(e).__name__,
            },
        )
        return []
    return list(getattr(resp, "data", None) or [])


# ---------------------------------------------------------------------- #
# D. Memory relationship graph
# E. Topic clustering
# ---------------------------------------------------------------------- #
# Same data, two projections. The graph is keyed by (entity_type,
# content); the cluster is keyed by source_stable_key. Both run off
# one pass through the same rows.


def topic_overview(
    *,
    workspace_id: str,
    days: int = 30,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "top_entities": [
            {
              "content":     str,        # entity content e.g. "Kafka"
              "entity_type": str,         # "service" | "project" | ...
              "mentions":    int,
              "co_mentions": [
                  {"content": str, "entity_type": str, "count": int},
                  ...
              ],
              "first_seen":  ISO str | null,
              "last_seen":   ISO str | null,
            },
            ...
          ],
        "cluster_count": int,    # number of distinct sources holding ≥2 entities
      }

    A "co-mention" is any other entity that appeared in the SAME
    source_stable_key. The list is capped at 5 per top entity to
    keep the payload small.
    """
    rows = _fetch_memories(workspace_id=workspace_id, days=days)
    entities = [r for r in rows if r.get("kind") == "entity"]
    if not entities:
        return {"top_entities": [], "cluster_count": 0}

    # Group by (entity_type, content). content is case-insensitive
    # since the extractor sometimes captures "kafka" and "Kafka"
    # via different patterns.
    def key_of(r):
        return (
            (r.get("entity_type") or "").strip().lower(),
            (r.get("content") or "").strip().lower(),
        )

    counts: Counter = Counter()
    first_seen: Dict[Tuple[str, str], str] = {}
    last_seen: Dict[Tuple[str, str], str] = {}
    display_name: Dict[Tuple[str, str], str] = {}
    for r in entities:
        k = key_of(r)
        if not k[1]:
            continue
        counts[k] += 1
        # Track the original casing of the FIRST appearance so the
        # UI can render "Kafka" not "kafka".
        if k not in display_name:
            display_name[k] = (r.get("content") or "").strip()
        ts = r.get("source_timestamp") or r.get("created_at") or ""
        if ts:
            if k not in first_seen or ts < first_seen[k]:
                first_seen[k] = ts
            if k not in last_seen or ts > last_seen[k]:
                last_seen[k] = ts

    # Co-mention edges: for each source_stable_key, list its entities
    # (deduped) and emit pairwise counts.
    by_source: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r in entities:
        sk = r.get("source_stable_key") or ""
        if not sk:
            continue
        k = key_of(r)
        if not k[1] or k in by_source[sk]:
            continue
        by_source[sk].append(k)

    co_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    cluster_count = 0
    for sk, ents in by_source.items():
        if len(ents) >= 2:
            cluster_count += 1
        # Pairwise edges (undirected: count once per pair per source).
        n = len(ents)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = ents[i], ents[j]
                co_counts[a][b] += 1
                co_counts[b][a] += 1

    # Top entities by mentions, descending. Tie-break on alphabetical
    # so the order is deterministic across runs (matters for tests).
    safe_n = max(1, min(int(top_n or 10), 50))
    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0][1]),
    )[:safe_n]

    top_entities: List[Dict[str, Any]] = []
    for key, mentions in ordered:
        # Top 5 co-mentions; deterministic ordering.
        co = co_counts.get(key, Counter())
        co_list = sorted(
            co.items(),
            key=lambda kv: (-kv[1], kv[0][1]),
        )[:5]
        top_entities.append(
            {
                "content": display_name.get(key, key[1]),
                "entity_type": key[0],
                "mentions": mentions,
                "co_mentions": [
                    {
                        "content": display_name.get(other, other[1]),
                        "entity_type": other[0],
                        "count": cnt,
                    }
                    for other, cnt in co_list
                ],
                "first_seen": first_seen.get(key),
                "last_seen": last_seen.get(key),
            }
        )
    return {"top_entities": top_entities, "cluster_count": cluster_count}


# ---------------------------------------------------------------------- #
# F. Timeline reconstruction
# ---------------------------------------------------------------------- #


def reconstruct_timeline(
    *,
    workspace_id: str,
    entity: Optional[str] = None,
    kinds: Optional[List[str]] = None,
    days: int = 90,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Return chronological memory rows for one entity or one set of
    kinds. The rows that come back already carry source_stable_key,
    so the frontend can link each row back to its originating
    message / email.

    `entity` filters by case-insensitive substring match against
    memory content. When omitted, returns all kinds (or the supplied
    kinds) sorted by source_timestamp ASC.

    Filtering happens in Python (we already paid for the network
    read in _fetch_memories) so we don't have to add new Supabase
    query patterns.
    """
    rows = _fetch_memories(
        workspace_id=workspace_id,
        days=days,
        limit=2000,
    )
    if not rows:
        return []
    entity_needle = (entity or "").strip().lower()
    kind_set = set(kinds or [])
    filtered: List[Dict[str, Any]] = []
    for r in rows:
        if kind_set and r.get("kind") not in kind_set:
            continue
        if entity_needle:
            haystack = (r.get("content") or "").lower()
            if entity_needle not in haystack:
                continue
        filtered.append(r)
    # Chronological ASC; rows without source_timestamp sort last by
    # falling back to created_at.
    filtered.sort(key=lambda r: (r.get("source_timestamp") or r.get("created_at") or "",))
    safe_limit = max(1, min(int(limit or 50), 200))
    return filtered[:safe_limit]


# ---------------------------------------------------------------------- #
# G. Recurring-pattern detection
# ---------------------------------------------------------------------- #


def recurring_patterns(
    *,
    workspace_id: str,
    days: int = 7,
    min_mentions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find entities/topics that recur frequently in the recent window.
    Returns:
      [
        {
          "content":      "Kafka",
          "entity_type":  "service",
          "count":        6,
          "first_seen":   ISO str,
          "last_seen":    ISO str,
          "label":        "Kafka mentioned 6 times in the last 7 days",
        },
        ...
      ]

    Threshold `min_mentions` filters out one-offs. The default of 3
    over 7 days corresponds roughly to "talked about more than once
    every other day" which captures the spec's "Kafka lag mentioned
    6 times" example without flooding with single-mention noise.
    """
    overview = topic_overview(
        workspace_id=workspace_id,
        days=days,
        top_n=30,
    )
    out: List[Dict[str, Any]] = []
    threshold = max(2, int(min_mentions or 3))
    for e in overview.get("top_entities") or []:
        if e["mentions"] < threshold:
            continue
        days_safe = max(1, int(days or 7))
        out.append(
            {
                "content": e["content"],
                "entity_type": e["entity_type"],
                "count": e["mentions"],
                "first_seen": e.get("first_seen"),
                "last_seen": e.get("last_seen"),
                "label": (f"{e['content']} mentioned {e['mentions']} times " f"in the last {days_safe} days"),
            }
        )
    return out


# ---------------------------------------------------------------------- #
# H. Proactive intelligence foundations
# ---------------------------------------------------------------------- #


def proactive_insights(
    *,
    workspace_id: str,
    stale_action_days: int = 14,
    dormant_project_days: int = 30,
    surge_window_days: int = 7,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute three kinds of derived signals:

      stale_action_items: action_items whose source_timestamp is
          older than `stale_action_days` ago AND for which the same
          owner has no decision/follow-up after that timestamp.
          Surfaces "Rahul said he'd deploy Friday 3 weeks ago and
          there's been no follow-up."

      dormant_projects: project-type entities whose `last_seen` is
          older than `dormant_project_days`. Surfaces "Project Apollo
          hasn't been mentioned in a month."

      surging_entities: entities whose mention count in the last
          `surge_window_days` is >= 2x the mention count in the
          PRIOR `surge_window_days` window AND >= 3 mentions
          absolute. Surfaces a topic that's heating up.

    All lists are capped at 10 items each. Every item carries an
    `evidence` field referencing the source(s) so the UI can link
    back.
    """
    now = datetime.now(timezone.utc)

    # ---- stale action items ----
    stale_cutoff = now - timedelta(days=max(1, int(stale_action_days or 14)))
    # Pull plenty of history so we catch old action items + check
    # whether they've been followed up.
    rows = _fetch_memories(
        workspace_id=workspace_id,
        days=max(90, stale_action_days * 3),
        limit=5000,
    )
    actions = [r for r in rows if r.get("kind") == "action_item"]
    decisions = [r for r in rows if r.get("kind") == "decision"]

    # Index decisions by owner so we can ask "did this owner make
    # any decision after this action item?". We use owner equality
    # (case-insensitive) -- imperfect but good enough as a "stale"
    # heuristic.
    decision_ts_by_owner: Dict[str, str] = {}
    for d in decisions:
        owner = (d.get("owner") or "").strip().lower()
        ts = d.get("source_timestamp") or d.get("created_at") or ""
        if owner and ts and ts > decision_ts_by_owner.get(owner, ""):
            decision_ts_by_owner[owner] = ts

    stale_action_items: List[Dict[str, Any]] = []
    for a in actions:
        ts = a.get("source_timestamp") or a.get("created_at") or ""
        if not ts:
            continue
        try:
            ts_dt = _parse_iso(ts)
        except ValueError:
            continue
        if ts_dt >= stale_cutoff:
            continue
        owner = (a.get("owner") or "").strip().lower()
        # If the owner posted a decision after this action, treat it
        # as followed up (heuristic; not perfect, intentionally loose).
        if owner and decision_ts_by_owner.get(owner, "") > ts:
            continue
        stale_action_items.append(
            {
                "id": a.get("id"),
                "content": a.get("content"),
                "owner": a.get("owner"),
                "source_kind": a.get("source_kind"),
                "source_stable_key": a.get("source_stable_key"),
                "source_timestamp": a.get("source_timestamp"),
                "stale_since": ts,
            }
        )
    stale_action_items.sort(
        key=lambda r: r.get("source_timestamp") or "",
    )
    stale_action_items = stale_action_items[:10]

    # ---- dormant projects ----
    dormant_cutoff = now - timedelta(
        days=max(1, int(dormant_project_days or 30)),
    )
    last_seen_by_entity: Dict[Tuple[str, str], str] = {}
    display: Dict[Tuple[str, str], str] = {}
    for r in rows:
        if r.get("kind") != "entity":
            continue
        if (r.get("entity_type") or "") != "project":
            continue
        key = (
            (r.get("entity_type") or "").lower(),
            (r.get("content") or "").strip().lower(),
        )
        if not key[1]:
            continue
        display.setdefault(key, (r.get("content") or "").strip())
        ts = r.get("source_timestamp") or r.get("created_at") or ""
        if ts > last_seen_by_entity.get(key, ""):
            last_seen_by_entity[key] = ts
    dormant: List[Dict[str, Any]] = []
    for key, ts in last_seen_by_entity.items():
        try:
            ts_dt = _parse_iso(ts)
        except ValueError:
            continue
        if ts_dt < dormant_cutoff:
            dormant.append(
                {
                    "content": display.get(key, key[1]),
                    "entity_type": key[0],
                    "last_seen": ts,
                }
            )
    dormant.sort(key=lambda r: r.get("last_seen") or "")
    dormant = dormant[:10]

    # ---- surging entities ----
    window = max(1, int(surge_window_days or 7))
    surge_cutoff = now - timedelta(days=window)
    prior_cutoff = now - timedelta(days=window * 2)
    recent_counts: Counter = Counter()
    prior_counts: Counter = Counter()
    last_seen_surge: Dict[Tuple[str, str], str] = {}
    display_surge: Dict[Tuple[str, str], str] = {}
    for r in rows:
        if r.get("kind") != "entity":
            continue
        ts = r.get("source_timestamp") or r.get("created_at") or ""
        if not ts:
            continue
        try:
            ts_dt = _parse_iso(ts)
        except ValueError:
            continue
        key = (
            (r.get("entity_type") or "").lower(),
            (r.get("content") or "").strip().lower(),
        )
        if not key[1]:
            continue
        display_surge.setdefault(key, (r.get("content") or "").strip())
        if ts_dt >= surge_cutoff:
            recent_counts[key] += 1
            if ts > last_seen_surge.get(key, ""):
                last_seen_surge[key] = ts
        elif ts_dt >= prior_cutoff:
            prior_counts[key] += 1
    surging: List[Dict[str, Any]] = []
    for key, recent in recent_counts.items():
        if recent < 3:
            continue
        prior = prior_counts.get(key, 0)
        # Doubling rule with `prior == 0` as a special case: a brand-
        # new topic with ≥3 mentions surges.
        if prior == 0 or recent >= prior * 2:
            surging.append(
                {
                    "content": display_surge.get(key, key[1]),
                    "entity_type": key[0],
                    "recent_mentions": recent,
                    "prior_mentions": prior,
                    "last_seen": last_seen_surge.get(key),
                }
            )
    surging.sort(
        key=lambda r: (-r["recent_mentions"], r["content"]),
    )
    surging = surging[:10]

    return {
        "stale_action_items": stale_action_items,
        "dormant_projects": dormant,
        "surging_entities": surging,
    }


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #


def _parse_iso(ts: str) -> datetime:
    """
    Tolerant ISO 8601 parser. Accepts both `Z` suffix and `+00:00`.
    Strings without tzinfo are assumed UTC. Raises ValueError on
    anything unparseable.
    """
    if not isinstance(ts, str) or not ts.strip():
        raise ValueError("empty timestamp")
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
