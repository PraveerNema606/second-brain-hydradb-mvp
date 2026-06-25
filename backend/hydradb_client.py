"""
Minimal HydraDB client for the Second Brain MVP.

Endpoints used:
    POST {base_url}/ingestion/upload_knowledge   (ingestion — multipart)
    POST {base_url}/ingestion/delete_knowledge   (deletion — JSON)
    POST {base_url}/recall/full_recall           (retrieval — JSON)

Verified ingestion contract:
    - Content-Type: multipart/form-data  (set automatically by `requests`)
    - Form fields:
        tenant_id     = <HYDRADB_TENANT_ID>
        sub_tenant_id = <HYDRADB_SUB_TENANT_ID>
        files         = one or more uploaded .md / .txt files
    - Auth header:
        Authorization: Bearer <HYDRADB_API_KEY>

Recall contract (JSON body): tenant_id, sub_tenant_id, query, top_k.
If the deployment rejects `top_k`, flip RECALL_TOP_K_FIELD below.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from errors import HydraDBError, UpstreamTimeoutError
from logging_config import get_logger
from retry import RetryExhausted, retry

logger = get_logger(__name__)


# ---------------------------------------------------------------------- #
# If your HydraDB deployment uses "max_results" instead of "top_k" for the
# recall endpoint, change this one constant. The full_recall method will
# print a clear hint pointing here whenever the server returns a 4xx.
# ---------------------------------------------------------------------- #
RECALL_TOP_K_FIELD = "top_k"


# Retry-wrapped POST used by upload_knowledge.  Retries on network-level
# transients (timeout, connection reset) but not on auth or 4xx errors.
@retry(
    service="hydradb",
    max_attempts=3,
    initial_delay=1.0,
    retryable_exceptions=(requests.Timeout, requests.ConnectionError, OSError),
)
def _post_upload(url: str, headers: dict, data: dict, files: list) -> requests.Response:
    return requests.post(url, headers=headers, data=data, files=files, timeout=120)


# Retry-wrapped POST used by HydraDBClient.full_recall.  Operates at the raw
# requests level so exception translation in full_recall stays clean and tests
# always see HydraDBError / UpstreamTimeoutError (never RetryExhausted).
@retry(
    service="hydradb",
    max_attempts=3,
    initial_delay=1.0,
    retryable_exceptions=(requests.Timeout, requests.ConnectionError, OSError),
)
def _post_recall(url: str, headers: dict, payload: dict) -> requests.Response:
    return requests.post(url, headers=headers, json=payload, timeout=60)


# Retry-wrapped POST used by HydraDBClient.delete_knowledge.  Same retry
# policy as recall/upload: network transients retry, auth / 4xx do not.
@retry(
    service="hydradb",
    max_attempts=3,
    initial_delay=1.0,
    retryable_exceptions=(requests.Timeout, requests.ConnectionError, OSError),
)
def _post_delete(url: str, headers: dict, payload: dict) -> requests.Response:
    return requests.post(url, headers=headers, json=payload, timeout=60)


# ---------------------------------------------------------------------- #
# Counting helper (shared by HydraDBClient.upload_knowledge and by the
# CLI's upload_in_batches, so the per-batch print and the run-wide tally
# never drift apart).
#
# HydraDB returns HTTP 202 with a payload like:
#     {
#       "success":       true,
#       "status":        "queued",
#       "success_count": 2,
#       "failed_count":  0,
#       "results":       [ { "id": "...", "status": "queued", "error": null }, ... ]
#     }
#
# A document is treated as FAILED only when:
#   - the HTTP call itself failed  (payload arrives here as {}),
#   - top-level `success` is explicitly False,
#   - a result's `status` is explicitly "failed" or "error", OR
#   - a result's `error` field is non-null / non-empty.
#
# Everything else — "queued", "success", "ok", "indexed", "uploaded",
# missing status, etc. — counts as a successful upload state.
# ---------------------------------------------------------------------- #
FAILED_RESULT_STATUSES = {"failed", "error"}


def _result_is_failed(result: Dict[str, Any]) -> bool:
    """A single result item counts as failed only if explicitly bad."""
    status = (result.get("status") or "").lower()
    if status in FAILED_RESULT_STATUSES:
        return True
    if result.get("error"):  # non-null and non-empty
        return True
    return False


def summarize_upload_response(
    payload: Dict[str, Any],
    batch_size: int,
) -> Tuple[int, int]:
    """
    Return (success_count, failure_count) for one upload_knowledge response.

    Preference order:
      1. Empty payload -> whole batch failed (HTTP / network error upstream).
      2. Top-level `success: false` -> whole batch failed.
      3. Top-level `success_count` / `failed_count` -> trust the server.
      4. Per-item `results` -> count via _result_is_failed.
      5. Nothing usable -> assume the batch_size docs were accepted.
    """
    if not payload:
        return (0, batch_size)

    if payload.get("success") is False:
        return (0, batch_size)

    if "success_count" in payload or "failed_count" in payload:
        ok = int(payload.get("success_count") or 0)
        bad = int(payload.get("failed_count") or 0)
        return (ok, bad)

    results = payload.get("results")
    if isinstance(results, list):
        bad = sum(1 for r in results if _result_is_failed(r))
        ok = len(results) - bad
        return (ok, bad)

    return (batch_size, 0)


class HydraDBClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        sub_tenant_id: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("HYDRADB_BASE_URL", "https://api.hydradb.com")).rstrip("/")
        self.api_key = api_key or os.getenv("HYDRADB_API_KEY")
        self.tenant_id = tenant_id or os.getenv("HYDRADB_TENANT_ID")
        self.sub_tenant_id = sub_tenant_id or os.getenv("HYDRADB_SUB_TENANT_ID", "slack-second-brain")

        if not self.api_key:
            raise ValueError("HYDRADB_API_KEY is not set.")
        if not self.tenant_id:
            raise ValueError("HYDRADB_TENANT_ID is not set.")

    # ------------------------------------------------------------------ #
    def _auth_headers(self) -> Dict[str, str]:
        """
        Only the Authorization header — do NOT set Content-Type here.

        `requests` builds the multipart Content-Type with the correct
        boundary string when we pass `files=`. Setting it manually would
        break the multipart parsing on the server side.
        """
        return {"Authorization": f"Bearer {self.api_key}"}

    # ------------------------------------------------------------------ #
    def upload_knowledge(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upload a batch of knowledge files to HydraDB.

        Each item in `files` looks like:
            {"filename": "slack_all-second-brain_1778775842.md",
             "content":  "<markdown text>"}

        `content` may be str or bytes; str is encoded as UTF-8.

        Returns the parsed HydraDB response dict on success, or {} on failure.
        """
        if not files:
            logger.debug('hydradb_upload_skipped', extra={'reason': 'no_files'})
            return {}

        url = f"{self.base_url}/ingestion/upload_knowledge"

        # Form fields (everything that isn't a file goes in `data`).
        data = {
            "tenant_id": self.tenant_id,
            "sub_tenant_id": self.sub_tenant_id,
        }

        # Build the multipart file list. Reusing the same form key "files"
        # for every entry is what tells the server this is a multi-file
        # upload (FastAPI-style `List[UploadFile]`).
        multipart_files = []
        for item in files:
            filename = item["filename"]
            content = item["content"]
            content_bytes = content.encode("utf-8") if isinstance(content, str) else content
            multipart_files.append(("files", (filename, content_bytes, "text/markdown")))

        # ------------------------------------------------------------------
        try:
            response = _post_upload(
                url,
                headers=self._auth_headers(),
                data=data,
                files=multipart_files,
            )
        except (requests.RequestException, RetryExhausted) as e:
            logger.warning('hydradb_upload_network_error', extra={'error': type(e).__name__})
            return {}

        level = logging.DEBUG if response.status_code < 400 else logging.WARNING
        # DIAGNOSTIC (instrumentation only): surface the write sub-tenant and
        # a short response-body preview so a silent upload rejection (4xx/5xx
        # body) is visible and attributable to the correct sub-tenant. No
        # behavior change -- still returns {} on >=400 below.
        try:
            _resp_preview = (response.text or "")[:300]
        except Exception:  # noqa: BLE001
            _resp_preview = "<unavailable>"
        logger.log(
            level,
            'hydradb_upload_response',
            extra={
                'http_status': response.status_code,
                'status_code': response.status_code,
                'sub_tenant': self.sub_tenant_id,
                'file_count': len(files),
                'response_preview': _resp_preview,
            },
        )

        if response.status_code >= 400:
            return {}

        try:
            payload = response.json()
        except ValueError:
            logger.warning('hydradb_upload_non_json', extra={'http_status': response.status_code})
            return {}

        results = payload.get("results")
        if isinstance(results, list):
            for i, r in enumerate(results):
                status_str = (r.get("status") or "").lower() or "unknown"
                logger.debug(
                    'hydradb_upload_result',
                    extra={
                        'result_index': i,
                        'status': status_str,
                        'has_error': bool(r.get("error")),
                    },
                )

        ok, bad = summarize_upload_response(payload, batch_size=len(files))
        source = "server-reported" if ("success_count" in payload or "failed_count" in payload) else "derived"
        logger.info(
            'hydradb_upload_batch_summary',
            extra={
                'ok': ok,
                'failed': bad,
                'count_source': source,
            },
        )

        return payload

    # ------------------------------------------------------------------ #
    # Deletion
    # ------------------------------------------------------------------ #
    def delete_knowledge(self, source_keys: List[str]) -> Dict[str, Any]:
        """
        Delete previously-ingested documents from HydraDB by their stable
        source key(s).

        Used when an upstream source is removed (e.g. a Gmail message is
        permanently deleted) so recall stops returning stale content.

        `source_keys` are the stable keys we stamp into each document's
        markdown header as `Source Key:` (e.g. "gmail:msg:1899abcd").

        Wire contract (POST /ingestion/delete_knowledge, JSON):
            {
              "tenant_id":     <HYDRADB_TENANT_ID>,
              "sub_tenant_id": <this client's sub-tenant>,
              "source_keys":   ["gmail:msg:...", ...]
            }

        Returns the parsed response dict on success, or {} on any
        failure. Best-effort and non-raising by design (mirrors
        upload_knowledge, not full_recall): a cleanup pass must never
        crash the surrounding ingest run. Network transients are retried
        by _post_delete; auth / 4xx are not.

        NOTE: the exact `SourceDeleteRequest` field name/identity is the
        one part of this contract not verifiable from the consumer repo
        (HydraDB never receives the stable_key on upload -- only the
        filename + content). We key on the stable source key here; if a
        deployment keys deletion on a different field, this is the single
        place to adjust.
        """
        keys = [k.strip() for k in (source_keys or []) if k and str(k).strip()]
        if not keys:
            logger.debug('hydradb_delete_skipped', extra={'reason': 'no_keys'})
            return {}

        url = f"{self.base_url}/ingestion/delete_knowledge"
        payload = {
            "tenant_id": self.tenant_id,
            "sub_tenant_id": self.sub_tenant_id,
            "source_keys": keys,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = _post_delete(url, headers, payload)
        except (requests.RequestException, RetryExhausted) as e:
            logger.warning(
                'hydradb_delete_network_error',
                extra={'error': type(e).__name__, 'key_count': len(keys)},
            )
            return {}

        level = logging.DEBUG if response.status_code < 400 else logging.WARNING
        logger.log(
            level,
            'hydradb_delete_response',
            extra={'http_status': response.status_code, 'key_count': len(keys)},
        )

        if response.status_code >= 400:
            return {}

        try:
            return response.json()
        except ValueError:
            logger.warning('hydradb_delete_non_json', extra={'http_status': response.status_code})
            return {}

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def full_recall(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Call POST /recall/full_recall and return the parsed JSON response.

        Network retries (3 attempts, exponential backoff) are handled by the
        module-level _post_recall helper so exception translation here always
        yields stable types regardless of how many retries were attempted.

        Raises:
            HydraDBError          on non-2xx, empty query, bad JSON, or network
                                  failure (including exhausted retries).
            UpstreamTimeoutError  when all attempts time out.
        """
        if not query or not query.strip():
            raise HydraDBError(
                detail="Query was empty.",
                log_context="full_recall called with empty query",
            )

        url = f"{self.base_url}/recall/full_recall"
        payload = {
            "tenant_id": self.tenant_id,
            "sub_tenant_id": self.sub_tenant_id,
            "query": query,
            RECALL_TOP_K_FIELD: top_k,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = _post_recall(url, headers, payload)
        except requests.Timeout as e:
            logger.warning('hydradb_recall_timeout', extra={'error': type(e).__name__})
            raise UpstreamTimeoutError(
                log_context=f"HydraDB full_recall timed out: {e}",
            )
        except RetryExhausted as e:
            # All retry attempts failed — surface as appropriate typed error.
            cause = e.__cause__
            if isinstance(cause, requests.Timeout):
                logger.warning('hydradb_recall_timeout', extra={'error': 'RetryExhausted(Timeout)'})
                raise UpstreamTimeoutError(
                    log_context=f"HydraDB recall timed out after retries: {cause}",
                ) from e
            logger.warning('hydradb_recall_network_error', extra={'error': 'RetryExhausted'})
            raise HydraDBError(
                log_context=f"network error during full_recall after retries: {cause}",
            ) from e
        except requests.RequestException as e:
            logger.warning('hydradb_recall_network_error', extra={'error': type(e).__name__})
            raise HydraDBError(
                log_context=f"network error during full_recall: {e}",
            )

        logger.debug('hydradb_recall_response', extra={'http_status': response.status_code})

        if response.status_code >= 400:
            logger.warning(
                'hydradb_recall_error',
                extra={
                    'http_status': response.status_code,
                    'top_k_field': RECALL_TOP_K_FIELD,
                },
            )
            raise HydraDBError(
                detail=f"Knowledge backend returned HTTP {response.status_code}.",
                log_context=f"full_recall HTTP {response.status_code} body={response.text[:400]}",
                upstream_status=response.status_code,
            )

        try:
            return response.json()
        except ValueError:
            raise HydraDBError(
                log_context=f"non-JSON recall response: {response.text[:500]}",
            )