"""
Tracks which Slack messages/threads have already been uploaded to HydraDB,
so re-running the ingestion script does not re-upload the same files.

State lives in a plain JSON file (no database) at:
    backend/data/ingestion_state.json

Schema:
    {
      "version": 2,
      "entries": {
        "<stable_key>": {
          "stable_key":   "slack:C123:1778775842.876209",
          "filename":     "slack_all-second-brain_1778775842.md",
          "source_id":    "aa54a1b1-...",       # from HydraDB results, may be null
          "channel_id":   "C123",
          "channel_name": "all-second-brain",
          "ts":           "1778775842.876209",  # for messages
          "thread_ts":    null,                 # for threads
          "uploaded_at":  "2026-05-15T12:34:56.789012+00:00",
          "user_name":    "Praveer Nema",
          "timestamp":    "1778775842.876209",
          "snippet":      "first ~200 chars ...",
          "permalink":    "https://...",
          "document_type":"message" | "thread"
        },
        ...
      },
      "channels": {
        "<channel_id>": {
          "last_synced_ts": "1778775842.876209"   # newest message ts we've ingested
        },
        ...
      }
    }

`channels` is added in version 2 for incremental Slack sync. Older files
(missing `channels` or `version: 1`) load without error — we just start
with an empty channels dict, which means the next run will fetch full
history (one-time cost) before incremental kicks in.
"""

try:
    import fcntl
except ImportError:
    fcntl = None
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


STATE_VERSION = 2


# ---------------------------------------------------------------------- #
# Stable keys
# ---------------------------------------------------------------------- #
def stable_key_for_message(channel_id: str, ts: str) -> str:
    """`slack:{channel_id}:{ts}` — never re-derived elsewhere; use this."""
    return f"slack:{channel_id}:{ts}"


def stable_key_for_thread(channel_id: str, thread_ts: str) -> str:
    """`slack_thread:{channel_id}:{thread_ts}` — never re-derived elsewhere; use this."""
    return f"slack_thread:{channel_id}:{thread_ts}"


# ---------------------------------------------------------------------- #
# State container
# ---------------------------------------------------------------------- #
class IngestionState:
    def __init__(self, path: Path):
        self.path = path
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.channels: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ----- disk I/O ---------------------------------------------------- #
    def _load(self) -> None:
        """Load the state file if it exists; otherwise start empty."""
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                'ingestion_state_load_failed',
                extra={
                    'path': str(self.path),
                    'error': type(e).__name__,
                },
            )
            return

        if not isinstance(raw, dict):
            return

        entries = raw.get("entries")
        if isinstance(entries, dict):
            self.entries = entries

        # `channels` was added in version 2. Older files don't have it; we
        # treat that as "empty" so the next run does a full sync once.
        channels = raw.get("channels")
        if isinstance(channels, dict):
            self.channels = channels

    def save(self) -> None:
        """Atomically write the state file (write to .tmp, then rename)."""
        self._save()

    def _save(self) -> None:
        """Atomically write the state file (write to .tmp, then rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {
            "version": STATE_VERSION,
            "entries": self.entries,
            "channels": self.channels,
        }
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.path)

    # ----- queries ----------------------------------------------------- #
    def has(self, stable_key: str) -> bool:
        return stable_key in self.entries

    # ----- mutations --------------------------------------------------- #
    def mark_uploaded(
        self,
        stable_key: str,
        filename: str,
        channel_id: str,
        channel_name: str,
        ts: Optional[str] = None,
        thread_ts: Optional[str] = None,
        source_id: Optional[str] = None,
        user_name: Optional[str] = None,
        timestamp: Optional[str] = None,
        snippet: Optional[str] = None,
        permalink: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> None:
        """Record a successful upload. Caller still needs to call save()."""
        self.entries[stable_key] = {
            "stable_key": stable_key,
            "filename": filename,
            "source_id": source_id,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "ts": ts,
            "thread_ts": thread_ts,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            # Newly added for UI-friendly source cards.
            "user_name": user_name,
            "timestamp": timestamp,
            "snippet": snippet,
            "permalink": permalink,
            "document_type": document_type,
        }

    def remove(self, stable_key: str) -> bool:
        """Remove a tracked entry. Returns True if it was present. Caller must save()."""
        if stable_key in self.entries:
            del self.entries[stable_key]
            return True
        return False

    # ----- lookups for recall ----------------------------------------- #
    def get(self, stable_key: str) -> Optional[Dict[str, Any]]:
        return self.entries.get(stable_key)

    def find_by_source_id(self, source_id: str) -> Optional[Dict[str, Any]]:
        if not source_id:
            return None
        for entry in self.entries.values():
            if entry.get("source_id") == source_id:
                return entry
        return None

    def find_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        if not filename:
            return None
        for entry in self.entries.values():
            if entry.get("filename") == filename:
                return entry
        return None

    # ----- per-channel sync timestamps -------------------------------- #
    def get_last_synced_ts(self, channel_id: str) -> Optional[str]:
        """The newest Slack ts we've successfully ingested for this channel."""
        info = self.channels.get(channel_id)
        if isinstance(info, dict):
            ts = info.get("last_synced_ts")
            if isinstance(ts, str) and ts:
                return ts
        return None

    def set_last_synced_ts(self, channel_id: str, ts: str) -> None:
        """
        Record the newest successfully-ingested ts for this channel.

        Caller should only invoke this AFTER the channel's batch upload
        succeeded, so a crash mid-run doesn't advance the watermark past
        unuploaded messages.

        We never move the watermark backward — if `ts` is older than the
        stored value, we keep the stored one. This guards against edge
        cases like a race where an old message arrives after a new one.
        """
        if not channel_id or not ts:
            return
        existing = self.get_last_synced_ts(channel_id)
        if existing and existing >= ts:
            # Slack ts strings sort lexicographically the same as numerically
            # because they are zero-padded "<seconds>.<micros>" forms with
            # equal-length seconds parts within any sensible time range.
            return
        self.channels[channel_id] = {"last_synced_ts": ts}

    # ----- last-ingestion timestamp (used by /api/admin/status) ------- #
    def get_last_ingested_at(self) -> Optional[str]:
        """ISO timestamp of the most recent successful upload, if any."""
        # We stash a top-level "_meta" dict via mark_uploaded.uploaded_at on
        # the newest entry, but explicit storage is cleaner for the admin
        # endpoint to read without scanning every entry.
        value = (
            self.channels.get("_meta", {}).get("last_ingested_at")
            if isinstance(self.channels.get("_meta"), dict)
            else None
        )
        return value if isinstance(value, str) else None

    def touch_last_ingested(self) -> None:
        """Mark "we just ingested something". Caller should still call save()."""
        # Stored alongside per-channel watermarks under a "_meta" key
        # (which can never collide with a real Slack channel ID since IDs
        # always start with a capital letter).
        self.channels["_meta"] = {
            "last_ingested_at": datetime.now(timezone.utc).isoformat(),
        }

    def total_docs(self) -> int:
        """How many docs we've successfully recorded."""
        return len(self.entries)

    # ----- cross-process locking ---------------------------------------- #
    def save_locked(self) -> None:
        """
        Cross-process-safe save for long-running ingest passes.

        Acquires an OS-level advisory lock on a ``<path>.lock`` sidecar file,
        reloads the state file from disk to incorporate any concurrent writes
        (e.g. the realtime webhook firing while a bulk ingest is running),
        merges this instance's in-memory mutations on top, then atomically
        writes and releases the lock.

        Merge rules:
          - ``entries``: this instance wins — new uploads take precedence.
          - Per-channel watermarks: keep whichever ts is **newer** so we
            never go backward on a channel.
          - ``_meta`` (last_ingested_at): this instance wins.

        Falls back to a plain :meth:`save` on systems where ``fcntl`` is
        unavailable (Windows). Single-process deployments are safe either way.
        """
        lock_path = self.path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lf:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                # Reload whatever is on disk right now (may include realtime
                # writes that happened after we loaded our in-memory copy).
                fresh = IngestionState(self.path)

                # Our newly-uploaded entries win.
                fresh.entries.update(self.entries)

                # Merge per-channel watermarks: keep the newer ts.
                for ch_id, ch_data in self.channels.items():
                    if ch_id == "_meta":
                        fresh.channels["_meta"] = ch_data
                    elif isinstance(ch_data, dict) and "last_synced_ts" in ch_data:
                        existing_ts = (fresh.channels.get(ch_id) or {}).get("last_synced_ts")
                        new_ts = ch_data["last_synced_ts"]
                        if not existing_ts or new_ts > existing_ts:
                            fresh.channels[ch_id] = {"last_synced_ts": new_ts}
                    else:
                        fresh.channels[ch_id] = ch_data

                fresh._save()  # atomic write-temp-rename
            finally:
                if fcntl:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    @classmethod
    @contextmanager
    def locked(cls, path: Path) -> Generator["IngestionState", None, None]:
        """
        Exclusive cross-process read-modify-save context manager.

        Acquires a POSIX advisory lock on a `<path>.lock` sidecar file,
        loads a fresh IngestionState from disk inside the lock, yields it
        for in-memory mutations, then saves and releases the lock.

        Usage::

            with IngestionState.locked(STATE_PATH) as state:
                state.mark_uploaded(...)
                # save() is called automatically on context exit

        Note: fcntl is POSIX-only (Linux / macOS). This is acceptable for
        a Docker/Linux deployment. On Windows, replace fcntl.flock with
        msvcrt.locking or use a third-party cross-platform lock library.
        """
        path = Path(path)
        lock_path = path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lf:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_EX)

            try:
                state = cls(path)  # fresh load while we hold the lock
                yield state
                state._save()
            finally:
                if fcntl:
                    fcntl.flock(lf, fcntl.LOCK_UN)
