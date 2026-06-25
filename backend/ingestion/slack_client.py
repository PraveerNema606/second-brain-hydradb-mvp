"""
Slack API client wrapper for the Second Brain MVP.

Wraps slack_sdk.WebClient with two simple methods:
    - fetch_channel_messages: pulls messages from a channel with pagination
    - fetch_thread_replies:   pulls replies for a single thread with pagination

Slack API errors are caught and printed so that one bad channel or thread
does not crash the whole ingestion run.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from retry import RetryExhausted, retry

logger = logging.getLogger(__name__)


# Slack's max page size for conversations.history / conversations.replies is 200.
SLACK_MAX_PAGE_SIZE = 200

# Maximum number of consecutive 429 responses to honour before giving up on a
# single page fetch.  Prevents an infinite loop if Slack persistently rejects
# a request (e.g. revoked bot scope or channel removed).
MAX_RATE_LIMIT_RETRIES = 5


class SlackClientWrapper:
    def __init__(self, token: Optional[str] = None):
        token = token or os.getenv("SLACK_BOT_TOKEN")
        if not token:
            raise ValueError("SLACK_BOT_TOKEN is not set in the environment.")
        self.client = WebClient(token=token)

        # In-memory caches. They live for the lifetime of this wrapper
        # instance (i.e. one ingestion run), which is all we need to avoid
        # calling Slack repeatedly for the same user / message.
        #   user_id -> display name (or None when lookup failed)
        self._user_name_cache: Dict[str, Optional[str]] = {}
        #   (channel_id, ts) -> permalink (or None when lookup failed)
        self._permalink_cache: Dict[tuple, Optional[str]] = {}
        # Channels where Slack returned not_in_channel or channel_not_found
        # during this run. Callers can inspect this after ingestion to update
        # the channel's bot_removed flag in the database.
        self.bot_removed_channels: set = set()

    # ------------------------------------------------------------------ #
    # Channel-level messages
    # ------------------------------------------------------------------ #
    def fetch_channel_messages(
        self,
        channel_id: str,
        limit_per_channel: int = 500,
        oldest: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch up to `limit_per_channel` messages from a single Slack channel.

        Uses conversations.history with cursor pagination. Stops as soon as we
        hit limit_per_channel or run out of messages.

        If `oldest` is provided (a Slack ts string like "1778775842.876209"),
        Slack returns only messages with ts STRICTLY greater than that
        value — i.e. only what's been posted since our last successful
        sync. This is the core of incremental ingestion.
        """
        collected: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        page_size = min(SLACK_MAX_PAGE_SIZE, limit_per_channel)

        # Build kwargs once; only include `oldest` if set so callers using
        # the default behavior aren't affected.
        base_kwargs: Dict[str, Any] = {"channel": channel_id, "limit": page_size}
        if oldest:
            base_kwargs["oldest"] = oldest

        rate_limit_retries = 0
        while True:
            try:
                response = self._call_conversations_history(
                    cursor=cursor,
                    **base_kwargs,
                )
            except SlackApiError as e:
                err = e.response.get("error", str(e)) if getattr(e, "response", None) else str(e)
                if self._is_rate_limited(e) and rate_limit_retries < MAX_RATE_LIMIT_RETRIES:
                    rate_limit_retries += 1
                    logger.info(
                        'slack_rate_limited',
                        extra={
                            'api': 'conversations_history',
                            'channel_id': channel_id,
                            'attempt': rate_limit_retries,
                            'max_attempts': MAX_RATE_LIMIT_RETRIES,
                        },
                    )
                    self._sleep_for_retry(e)
                    continue
                if err in ("not_in_channel", "channel_not_found"):
                    self.bot_removed_channels.add(channel_id)
                logger.warning(
                    'slack_api_error',
                    extra={
                        'api': 'conversations_history',
                        'channel_id': channel_id,
                        'error': err,
                    },
                )
                break

            messages = response.get("messages", []) or []
            collected.extend(messages)

            if len(collected) >= limit_per_channel:
                collected = collected[:limit_per_channel]
                break

            cursor = (response.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor:
                break

        return collected

    # ------------------------------------------------------------------ #
    # Thread replies
    # ------------------------------------------------------------------ #
    def fetch_thread_replies(
        self,
        channel_id: str,
        thread_ts: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch all replies for a thread (Slack returns the parent as the first
        message, followed by replies). Uses conversations.replies with cursor
        pagination.
        """
        collected: List[Dict[str, Any]] = []
        cursor: Optional[str] = None

        rate_limit_retries = 0
        while True:
            try:
                response = self._call_conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    limit=SLACK_MAX_PAGE_SIZE,
                    cursor=cursor,
                )
            except SlackApiError as e:
                err = e.response.get("error", str(e)) if getattr(e, "response", None) else str(e)
                if self._is_rate_limited(e) and rate_limit_retries < MAX_RATE_LIMIT_RETRIES:
                    rate_limit_retries += 1
                    logger.info(
                        'slack_rate_limited',
                        extra={
                            'api': 'conversations_replies',
                            'thread_ts': thread_ts,
                            'attempt': rate_limit_retries,
                            'max_attempts': MAX_RATE_LIMIT_RETRIES,
                        },
                    )
                    self._sleep_for_retry(e)
                    continue
                logger.warning(
                    'slack_api_error',
                    extra={
                        'api': 'conversations_replies',
                        'channel_id': channel_id,
                        'error': err,
                    },
                )
                break

            messages = response.get("messages", []) or []
            collected.extend(messages)

            cursor = (response.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor:
                break

        return collected

    # ------------------------------------------------------------------ #
    # User name resolution (cached)
    # ------------------------------------------------------------------ #
    def resolve_user_name(self, user_id: Optional[str]) -> Optional[str]:
        """
        Look up a Slack user's readable name from their U... ID.

        Preference: real_name -> display_name -> name -> None.
        Cached in-memory so we hit users.info at most once per user_id
        per ingestion run. Returns None for falsy ids or on API errors.
        """
        if not user_id:
            return None

        if user_id in self._user_name_cache:
            return self._user_name_cache[user_id]

        name: Optional[str] = None
        try:
            response = self._call_users_info(user_id=user_id)
            user = response.get("user") or {}
            profile = user.get("profile") or {}
            # Order matters: real_name is usually the best human label.
            name = (
                profile.get("real_name")
                or profile.get("display_name")
                or user.get("real_name")
                or user.get("name")
                or None
            )
            if isinstance(name, str):
                name = name.strip() or None
        except SlackApiError as e:
            err = e.response.get("error", str(e)) if getattr(e, "response", None) else str(e)
            logger.warning('slack_api_error', extra={'api': 'users_info', 'error': err})

        self._user_name_cache[user_id] = name
        return name

    # ------------------------------------------------------------------ #
    # Permalinks (cached)
    # ------------------------------------------------------------------ #
    def get_permalink(
        self,
        channel_id: str,
        message_ts: str,
    ) -> Optional[str]:
        """
        Return the Slack permalink for a (channel_id, message_ts) pair.

        Cached. Returns None on API error so ingestion never fails just
        because a permalink lookup failed.
        """
        if not channel_id or not message_ts:
            return None

        cache_key = (channel_id, message_ts)
        if cache_key in self._permalink_cache:
            return self._permalink_cache[cache_key]

        permalink: Optional[str] = None
        try:
            response = self._call_get_permalink(
                channel_id=channel_id,
                message_ts=message_ts,
            )
            permalink = response.get("permalink") or None
        except SlackApiError as e:
            err = e.response.get("error", str(e)) if getattr(e, "response", None) else str(e)
            logger.warning('slack_api_error', extra={'api': 'chat_getPermalink', 'error': err})

        self._permalink_cache[cache_key] = permalink
        return permalink

    # ------------------------------------------------------------------ #
    # Retry-wrapped API call helpers
    # ------------------------------------------------------------------ #
    # These wrap the network-level calls so transient connection errors are
    # retried automatically.  SlackApiError (including 429) is handled
    # separately by the pagination loops above (which honour Retry-After).

    # retryable_status_codes=() so that 429 (rate limit) is NOT retried here —
    # the outer pagination loops own rate-limit handling (sleep + counter).
    # Only network-level transients (connection reset, timeout) are retried.
    @retry(
        service="slack",
        max_attempts=3,
        initial_delay=1.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        retryable_status_codes=(),
    )
    def _call_conversations_history(self, **kwargs):
        """Retry-wrapped conversations.history — accepts any kwargs for forward compat."""
        return self.client.conversations_history(**kwargs)

    @retry(
        service="slack",
        max_attempts=3,
        initial_delay=1.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        retryable_status_codes=(),
    )
    def _call_conversations_replies(self, **kwargs):
        """Retry-wrapped conversations.replies — accepts any kwargs for forward compat."""
        return self.client.conversations_replies(**kwargs)

    @retry(
        service="slack",
        max_attempts=3,
        initial_delay=1.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    )
    def _call_users_info(self, user_id: str):
        return self.client.users_info(user=user_id)

    @retry(
        service="slack",
        max_attempts=3,
        initial_delay=1.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
    )
    def _call_get_permalink(self, channel_id: str, message_ts: str):
        return self.client.chat_getPermalink(channel=channel_id, message_ts=message_ts)

    # ------------------------------------------------------------------ #
    # File info (requires files:read scope)
    # ------------------------------------------------------------------ #
    def fetch_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the 'file' dict from files.info for the given file_id.

        Requires the files:read bot scope. Returns None on any error
        (missing scope, not found, network failure) so callers treat
        file preview as best-effort enrichment.
        """
        if not file_id:
            return None
        try:
            response = self._call_files_info(file_id=file_id)
            return response.get("file") or None
        except SlackApiError as e:
            err = e.response.get("error", str(e)) if getattr(e, "response", None) else str(e)
            if err == "missing_scope":
                logger.debug("slack_files_info_missing_scope", extra={"file_id": file_id})
            else:
                logger.warning(
                    "slack_api_error",
                    extra={"api": "files_info", "file_id": file_id, "error": err},
                )
            return None
        except RetryExhausted:
            logger.warning("slack_files_info_network_exhausted", extra={"file_id": file_id})
            return None

    @retry(
        service="slack",
        max_attempts=3,
        initial_delay=1.0,
        retryable_exceptions=(ConnectionError, TimeoutError, OSError),
        retryable_status_codes=(),
    )
    def _call_files_info(self, file_id: str):
        return self.client.files_info(file=file_id)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_rate_limited(e: SlackApiError) -> bool:
        try:
            return e.response.status_code == 429
        except Exception:
            return False

    @staticmethod
    def _sleep_for_retry(e: SlackApiError) -> None:
        retry_after = 1
        try:
            retry_after = int(e.response.headers.get("Retry-After", 1))
        except Exception:
            pass
        logger.info('slack_rate_limit_sleep', extra={'retry_after_seconds': retry_after})
        time.sleep(retry_after)
