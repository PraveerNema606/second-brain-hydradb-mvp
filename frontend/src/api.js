// Thin client for the Second Brain backend.
//
// The frontend reads these Vite env vars at build time:
//   VITE_API_BASE_URL    e.g. http://127.0.0.1:8000
//   VITE_APP_API_KEY     legacy shared secret — ONLY used by getAdminStatus
//                        (the /api/admin/status route still gates on
//                        X-API-Key in Phase 1). Leave blank if you don't
//                        need the admin card.
//
// Two auth modes are supported on requests to this backend:
//
//   a) USER-ONLY (bearer only)
//      Authorization: Bearer <supabase_jwt>
//      Used by /api/me and /api/me/workspaces — routes that resolve the
//      caller's identity WITHOUT a workspace selection (you have to
//      list workspaces before you can pick one, after all).
//
//   b) WORKSPACE (bearer + workspace id)
//      Authorization: Bearer <supabase_jwt>
//      X-Workspace-Id: <uuid>
//      Used by /api/query, /api/saved-answers, /api/chat/sessions,
//      /api/slack/*, etc. — everything that operates within a specific
//      workspace.
//
// The Auth + Workspace providers push credentials in via setAuthContext()
// at module scope. accessToken can be either a string OR an async getter
// (a function returning Promise<string>); the getter form is preferred
// because it lets Supabase auto-refresh expired tokens transparently —
// each request asks the SDK for the current session right before the
// fetch fires, so a token that expired between page-load and the click
// is refreshed before being sent.

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const ADMIN_API_KEY = import.meta.env.VITE_APP_API_KEY || "";

// ---------------------------------------------------------------------
// Module-level auth context (set by WorkspaceProvider).
// ---------------------------------------------------------------------
// _accessTokenSource can be:
//   - "" (empty string)         -> not signed in
//   - a non-empty string        -> static snapshot (back-compat)
//   - an (async) function       -> getter, called at request time
let _accessTokenSource = "";
let _activeWorkspaceId = "";

/**
 * Wire the current Supabase access token and active workspace id into
 * this module. Called by WorkspaceProvider whenever either value
 * changes (sign-in/out, token refresh, workspace switch).
 *
 * Pass `accessToken` as a function (sync or async) returning the
 * current token to let Supabase auto-refresh transparently. A bare
 * string still works for backwards compatibility.
 */
export function setAuthContext({ accessToken, activeWorkspaceId }) {
  _accessTokenSource =
    typeof accessToken === "function" ? accessToken : (accessToken || "");
  _activeWorkspaceId = activeWorkspaceId || "";
}

/**
 * Resolve the access token at request time. Always returns a string
 * (possibly empty). Never throws — caller branches on the empty case.
 */
async function resolveAccessToken() {
  const src = _accessTokenSource;
  if (typeof src === "function") {
    try {
      const value = await src();
      return typeof value === "string" ? value : "";
    } catch {
      return "";
    }
  }
  return src || "";
}

/**
 * Build the headers for one request.
 *  - { withWorkspace: false } -> bearer only (for /api/me*)
 *  - { withWorkspace: true  } -> bearer + X-Workspace-Id (default)
 *
 * Returns null when the required credentials aren't available, which
 * the caller surfaces as a typed ApiError so the UI can react.
 */
async function authHeaders({ withWorkspace }) {
  const token = await resolveAccessToken();
  if (!token) return null;
  const headers = { Authorization: `Bearer ${token}` };
  if (withWorkspace) {
    if (!_activeWorkspaceId) return null;
    headers["X-Workspace-Id"] = _activeWorkspaceId;
  }
  return headers;
}

/**
 * Build a typed ApiError describing which credential was missing.
 * Used by both jsonFetch and the streaming/query helpers so the UI
 * sees a consistent error_type for the same root cause.
 */
function missingCredsError({ withWorkspace }) {
  // Resolve synchronously based on cached state. Good enough for the
 // error message — the actual fetch resolves the token async.
  const tokenLooksMissing =
    typeof _accessTokenSource === "function"
      ? false   // assume present; resolveAccessToken decides at call time
      : !_accessTokenSource;
  if (tokenLooksMissing) {
    return new ApiError("Not signed in.", {
      status: 0, errorType: "auth_missing",
    });
  }
  if (withWorkspace && !_activeWorkspaceId) {
    return new ApiError("No active workspace.", {
      status: 0, errorType: "workspace_missing",
    });
  }
  // Fallback (e.g. getter resolved to empty): treat as auth missing so
  // the UI prompts re-sign-in, which is the right action.
  return new ApiError("Not signed in.", {
    status: 0, errorType: "auth_missing",
  });
}

/**
 * Thrown for any non-2xx response or network failure.
 * The component switches on `status` to render different messages.
 */
export class ApiError extends Error {
  constructor(message, { status = 0, errorType = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errorType = errorType;
  }
}

/**
 * Convert the UI's `sources` sentinel into the array shape the
 * backend's `allowed_sources` field expects (or `null` to omit
 * the field entirely so the request preserves pre-Phase-9 defaults).
 *
 *   "all"   -> null            (omit -> backend default: all sources)
 *   "slack" -> ["slack"]
 *   "gmail" -> ["gmail"]
 *   "both"  -> ["slack","gmail"]
 *
 * Mirror of `sourcesToList` in App.jsx; duplicated here so api.js
 * stays standalone and importable in tests / non-React callers.
 */
function sourcesValueToArray(value) {
  switch (value) {
    case "slack": return ["slack"];
    case "gmail": return ["gmail"];
    case "both":  return ["slack", "gmail"];
    case "all":
    default:      return null;
  }
}

/**
 * Build the request body, omitting optional fields that are blank/null
 * so the backend's Pydantic model uses its defaults instead of receiving
 * "" for a filter the user didn't fill in.
 */
function buildRequestBody({
  question,
  topK,
  mode,
  channel,
  user,
  documentType,
  sources,
  startDate,
  endDate,
  dateQuery,
  conversationHistory,
}) {
  const body = { question, top_k: topK, mode };
  if (channel && channel.trim()) body.channel = channel.trim();
  if (user && user.trim()) body.user = user.trim();
  if (documentType) body.document_type = documentType;

  // Phase 9: source filter. The UI's "sources" state is a single
  // string sentinel ("all" | "slack" | "gmail" | "both"). The
  // backend wants either an array of source kinds or the field
  // omitted entirely (== all sources). We pass an explicit
  // ["slack","gmail"] for "both" so the request log shows the
  // user's intent even though the observable behavior matches
  // "all".
  const allowedSources = sourcesValueToArray(sources);
  if (allowedSources !== null) body.allowed_sources = allowedSources;

  // Natural-language date phrase. The backend parses it; explicit start/end
  // timestamps from the date pickers will still override the parsed range.
  if (dateQuery && dateQuery.trim()) body.date_query = dateQuery.trim();

  const startSec = dateInputToUnixSeconds(startDate, "start");
  const endSec = dateInputToUnixSeconds(endDate, "end");
  if (startSec !== null) body.start_timestamp = startSec;
  if (endSec !== null) body.end_timestamp = endSec;

  // Recent chat turns for follow-up reference resolution ("he", "that",
  // "the earlier discussion"). Backend caps at 6 server-side; we cap
  // here too so we don't send bytes the server is going to discard.
  // Sending an empty array would defeat the cache for stateless asks,
  // so we omit the field entirely when there's nothing to send.
  if (Array.isArray(conversationHistory) && conversationHistory.length > 0) {
    const trimmed = conversationHistory
      .filter((m) => m && (m.role === "user" || m.role === "assistant")
                       && typeof m.content === "string"
                       && m.content.trim().length > 0)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));
    if (trimmed.length > 0) body.conversation_history = trimmed;
  }

  return body;
}

function dateInputToUnixSeconds(value, bound) {
  if (!value) return null;
  const parts = value.split("-").map((p) => parseInt(p, 10));
  if (parts.length !== 3 || parts.some(Number.isNaN)) return null;
  const [y, m, d] = parts;
  const date =
    bound === "end"
      ? new Date(y, m - 1, d, 23, 59, 59, 999)
      : new Date(y, m - 1, d, 0, 0, 0, 0);
  if (Number.isNaN(date.getTime())) return null;
  return date.getTime() / 1000;
}

/**
 * Detect AbortError reliably across browsers.
 */
function isAbortError(err, signal) {
  if (err && err.name === "AbortError") return true;
  if (signal && signal.aborted) return true;
  return false;
}

// ====================================================================
// Shared JSON helper
// ====================================================================
// Used by every workspace-scoped helper and by the user-only
// /api/me/workspaces fetch. Pass `requireWorkspace: false` to skip the
// X-Workspace-Id header (used for /api/me*).
async function jsonFetch(
  path,
  { method = "GET", body, signal, requireWorkspace = true } = {},
) {
  const headers = await authHeaders({ withWorkspace: requireWorkspace });
  if (!headers) {
    throw missingCredsError({ withWorkspace: requireWorkspace });
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: {
        ...headers,
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (err) {
    if (isAbortError(err, signal)) throw err;
    throw new ApiError(
      `Could not reach the backend. Is it running on ${API_BASE_URL}?`,
      { status: 0, errorType: "network_error" }
    );
  }

  // DELETE returns a JSON body too in our backend; only worry about
  // 204-no-content for forward-compat with future endpoints.
  let data = null;
  if (response.status !== 204) {
    try { data = await response.json(); } catch { /* non-JSON */ }
  }

  if (!response.ok) {
    throw new ApiError(messageForStatus(response.status, data), {
      status: response.status,
      errorType: (data && data.error_type) || "",
    });
  }
  return data;
}

// ====================================================================
// User-only routes (bearer auth, NO X-Workspace-Id)
// ====================================================================
// These two power the workspace bootstrap. They MUST work before a
// workspace is selected — otherwise the user could never list their
// workspaces in the first place.

export function getMe(signal) {
  return jsonFetch("/api/me", { signal, requireWorkspace: false });
}

export function listUserWorkspaces(signal) {
  return jsonFetch("/api/me/workspaces", { signal, requireWorkspace: false });
}

// ====================================================================
// POST /api/query  (workspace-scoped, non-streaming)
// ====================================================================
export async function askQuery(params) {
  const headers = await authHeaders({ withWorkspace: true });
  if (!headers) throw missingCredsError({ withWorkspace: true });

  const url = `${API_BASE_URL}/api/query`;
  const body = buildRequestBody(params);

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...headers,
      },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new ApiError(
      `Could not reach the backend. Is it running on ${API_BASE_URL}?`,
      { status: 0, errorType: "network_error" }
    );
  }

  let data = null;
  try { data = await response.json(); } catch { /* non-JSON */ }

  if (!response.ok) {
    throw new ApiError(messageForStatus(response.status, data), {
      status: response.status,
      errorType: (data && data.error_type) || "",
    });
  }
  return data;
}

// ====================================================================
// GET /api/admin/status
// ====================================================================
/**
 * Fetch a small ingestion-status snapshot for the admin card.
 * Returns null if the request fails — the admin card is best-effort
 * UX and shouldn't break the rest of the app on failure.
 *
 * Note: this endpoint is still gated by the legacy X-API-Key in
 * Phase 1. Set VITE_APP_API_KEY in your .env.local to enable it.
 */
export async function getAdminStatus() {
  if (!ADMIN_API_KEY) return null;

  const url = `${API_BASE_URL}/api/admin/status`;
  let response;
  try {
    response = await fetch(url, {
      method: "GET",
      headers: { "X-API-Key": ADMIN_API_KEY },
    });
  } catch {
    return null;
  }
  if (!response.ok) return null;
  try {
    return await response.json();
  } catch {
    return null;
  }
}

// ====================================================================
// Phase 2: chat sessions, chat messages, saved answers (workspace-scoped)
// ====================================================================
// Pattern: each helper resolves to the parsed JSON on success, or
// throws ApiError on failure. Callers in App.jsx wrap the calls in
// try/catch and fall back to localStorage so the app keeps working
// even when the backend is down or the user is offline.

// ----- chat sessions -----
export function listChatSessions(signal) {
  return jsonFetch("/api/chat/sessions", { signal });
}

export function createChatSession(title, signal) {
  return jsonFetch("/api/chat/sessions", {
    method: "POST",
    body: { title: title || null },
    signal,
  });
}

export function listChatMessages(sessionId, signal) {
  return jsonFetch(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
    { signal },
  );
}

export function createChatMessage(sessionId, message, signal) {
  // message = {role: 'user'|'assistant', content: string, sources?: array}
  return jsonFetch(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/messages`,
    { method: "POST", body: message, signal },
  );
}

// ----- saved answers -----
export function listSavedAnswers(signal) {
  return jsonFetch("/api/saved-answers", { signal });
}

export function createSavedAnswer(item, signal) {
  // item = {question, answer, sources?, mode?, filters?, debug?}
  return jsonFetch("/api/saved-answers", {
    method: "POST",
    body: item,
    signal,
  });
}

export function deleteSavedAnswer(id, signal) {
  return jsonFetch(
    `/api/saved-answers/${encodeURIComponent(id)}`,
    { method: "DELETE", signal },
  );
}

// ----- Phase 13: share links + workspace status -----

/**
 * Mint a public share link for a saved answer. Returns
 * {id, share_token, url, saved_answer_id, created_at}.
 * The URL is suitable for direct rendering / copy-to-clipboard;
 * the token alone authenticates the public read.
 */
export function shareSavedAnswer(savedId, signal) {
  return jsonFetch(
    `/api/saved-answers/${encodeURIComponent(savedId)}/share`,
    { method: "POST", signal },
  );
}

/**
 * List active (non-revoked) shares for one saved answer. Returns
 * {shares: [{id, share_token, url, created_at, expires_at, created_by}]}.
 * The frontend uses this to badge already-shared answers and offer
 * a Revoke button.
 */
export function listSharesForAnswer(savedId, signal) {
  return jsonFetch(
    `/api/saved-answers/${encodeURIComponent(savedId)}/shares`,
    { signal },
  );
}

/**
 * Revoke a share link. Scoped server-side to the caller's user +
 * workspace.
 */
export function revokeShareLink(token, signal) {
  return jsonFetch(
    `/api/saved-answers/share/${encodeURIComponent(token)}`,
    { method: "DELETE", signal },
  );
}

/**
 * Fetch a publicly-shared answer by token. NO auth, NO workspace
 * header -- the token itself is the credential. Returns
 * {question, answer, sources, mode, created_at} or throws
 * ApiError (404 collapses missing/revoked/expired).
 *
 * Implemented as a raw fetch (not via jsonFetch) because jsonFetch
 * requires the auth + workspace headers we don't want to send here.
 */
export async function fetchPublicShare(token, signal) {
  let response;
  try {
    response = await fetch(
      `${API_BASE_URL}/api/shared/${encodeURIComponent(token)}`,
      { method: "GET", signal },
    );
  } catch (err) {
    if (isAbortError(err, signal)) throw err;
    throw new ApiError(
      `Could not reach the backend. Is it running on ${API_BASE_URL}?`,
      { status: 0, errorType: "network_error" },
    );
  }
  let data = null;
  try { data = await response.json(); } catch { /* non-JSON */ }
  if (!response.ok) {
    throw new ApiError(messageForStatus(response.status, data), {
      status: response.status,
      errorType: (data && data.error_type) || "",
    });
  }
  return data;
}

/**
 * Lightweight workspace-level connector + sync snapshot. The
 * frontend renders this in the connector hint bar; cheap enough
 * to fetch on initial load and after each saved-answer / ingest
 * action.
 *
 * Shape: {
 *   slack: {connected, channels_selected, scheduler_enabled},
 *   gmail: {connection_count, labels_selected, last_synced_at},
 * }
 */
export function getWorkspaceStatus(signal) {
  return jsonFetch("/api/workspace/status", { signal });
}


// ----- Phase 15: analytics -----

/**
 * Aggregated query + ingest + retrieval-failure stats. `days`
 * defaults to 7; backend clamps to [1, 90].
 */
export function getAnalyticsOverview(days, signal) {
  const qs = days ? `?days=${encodeURIComponent(days)}` : "";
  return jsonFetch(`/api/analytics/overview${qs}`, { signal });
}

/**
 * Top entities + co-mentions + cluster count. `days` defaults to
 * 30 (backend), `top_n` to 10.
 */
export function getAnalyticsTopics({ days, topN } = {}, signal) {
  const params = new URLSearchParams();
  if (days)  params.set("days",  String(days));
  if (topN)  params.set("top_n", String(topN));
  const qs = params.toString();
  return jsonFetch(`/api/analytics/topics${qs ? "?" + qs : ""}`, { signal });
}

/**
 * Chronological memory rows for an entity / kind. `kinds` can be
 * an array; gets joined comma-separated for the query string.
 */
export function getAnalyticsTimeline(
  { entity, kinds, days, limit } = {}, signal,
) {
  const params = new URLSearchParams();
  if (entity) params.set("entity", entity);
  if (kinds && kinds.length) params.set("kind", kinds.join(","));
  if (days)  params.set("days",  String(days));
  if (limit) params.set("limit", String(limit));
  const qs = params.toString();
  return jsonFetch(`/api/analytics/timeline${qs ? "?" + qs : ""}`, { signal });
}

/**
 * Proactive insights: stale action items, dormant projects,
 * surging entities, recurring patterns. All in one payload to
 * keep the panel render to a single round-trip.
 */
export function getAnalyticsInsights(signal) {
  return jsonFetch("/api/analytics/insights", { signal });
}

// ----- Slack Connect (Phase 3, workspace-scoped) -----

/**
 * Fetch the workspace's Slack OAuth authorize URL. The caller is
 * expected to navigate window.location to the returned URL — Slack
 * will redirect back to /api/slack/oauth/callback when done.
 */
export function getSlackConnectUrl(signal) {
  return jsonFetch("/api/slack/connect-url", { signal });
}

/**
 * Return the current Slack-channel picker state. Shape:
 *   { connected: bool, team_name: string, channels: [...] }
 * The backend refreshes the channel list from Slack on every call,
 * so this is also the "Refresh" button's RPC.
 */
export function getSlackChannels(signal) {
  return jsonFetch("/api/slack/channels", { signal });
}

/**
 * Replace the workspace's selected-channel set and per-channel bot-message
 * opt-in. `ids` is a string array of slack_channel_id values; passing an
 * empty array clears the selection. `botIds` is the subset of channels for
 * which include_bot_messages should be true.
 */
export function saveSlackChannels(ids, botIds, signal) {
  return jsonFetch("/api/slack/channels", {
    method: "POST",
    body: {
      selected_channel_ids: Array.isArray(ids) ? ids : [],
      bot_message_channel_ids: Array.isArray(botIds) ? botIds : [],
    },
    signal,
  });
}

/**
 * Kick off an ingestion run for the currently-selected channels.
 * Returns immediately; the actual work runs in a backend
 * BackgroundTask. Returns the parsed { status, channels_queued }
 * envelope so the UI can show "started for N channels".
 */
export function runSlackIngest(signal) {
  return jsonFetch("/api/slack/ingest", { method: "POST", signal });
}

/**
 * Revoke the Slack bot token and delete the workspace installation.
 * On success the backend returns { disconnected: true }.
 */
export function disconnectSlack(signal) {
  return jsonFetch("/api/slack/disconnect", { method: "DELETE", signal });
}

// ----- Gmail Connect (Phase 8, workspace-scoped) -----
// Mirrors the Slack helpers above. Same workspace-bound auth (bearer +
// X-Workspace-Id); the OAuth callback redirects back to the frontend
// with ?gmail_connect=ok|error&reason=... which GmailSettings reads
// once on mount.
//
// One workspace can hold MULTIPLE Gmail connections (personal +
// shared mailbox, etc.), so labels and ingest take a connection_id.

/**
 * Fetch the workspace's Google OAuth authorize URL. The caller is
 * expected to navigate window.location to the returned URL — Google
 * will redirect back to /api/gmail/oauth/callback when done.
 */
export function getGmailConnectUrl(signal) {
  return jsonFetch("/api/gmail/connect-url", { signal });
}

/**
 * List every Gmail connection in the current workspace. The backend
 * returns the PUBLIC projection — no access/refresh tokens. Shape:
 *   { connections: [{ id, email, status, connected_at, ... }] }
 */
export function listGmailConnections(signal) {
  return jsonFetch("/api/gmail/connections", { signal });
}

/**
 * Delete a Gmail connection. Cascades to its labels + ingestion-state
 * rows server-side. 404 if the connection doesn't belong to the
 * caller's workspace.
 */
export function deleteGmailConnection(connectionId, signal) {
  return jsonFetch(
    `/api/gmail/connections/${encodeURIComponent(connectionId)}`,
    { method: "DELETE", signal },
  );
}

/**
 * Return the label picker state for one Gmail connection. The backend
 * refreshes the labels from Gmail on every call (so newly-created
 * labels show up), then returns the stored rows including is_selected.
 * Shape:
 *   { connected: bool, labels: [{ label_id, name, type, is_selected }] }
 */
export function getGmailLabels(connectionId, signal) {
  const qs = new URLSearchParams({ connection_id: connectionId }).toString();
  return jsonFetch(`/api/gmail/labels?${qs}`, { signal });
}

/**
 * Replace the selected-label set for a Gmail connection. `ids` is a
 * string array of Gmail label IDs; passing an empty array clears the
 * selection. 404 if the connection doesn't belong to this workspace.
 */
export function saveGmailLabels(connectionId, ids, signal) {
  return jsonFetch("/api/gmail/labels", {
    method: "POST",
    body: {
      connection_id:      connectionId,
      selected_label_ids: Array.isArray(ids) ? ids : [],
    },
    signal,
  });
}

/**
 * Kick off a Gmail ingestion run for one connection's selected
 * labels. Returns immediately; the actual work runs in a backend
 * BackgroundTask. Shape: { status: "started", labels_queued }.
 */
export function runGmailIngest(connectionId, signal) {
  return jsonFetch("/api/gmail/ingest", {
    method: "POST",
    body: { connection_id: connectionId },
    signal,
  });
}

// ====================================================================
// POST /api/query/stream  (Server-Sent Events, workspace-scoped)
// ====================================================================
/**
 * Stream an answer from the backend.
 *
 * @param {Object} params       Same shape as askQuery().
 * @param {Object} handlers
 * @param {(text: string) => void} handlers.onToken
 *        Called for every streamed token chunk.
 * @param {(final: Object) => void} handlers.onDone
 *        Called once at the end with { answer, sources, debug }.
 * @param {(err: ApiError) => void} handlers.onError
 *        Called if the stream errors. After this fires no more callbacks.
 * @param {() => void} [handlers.onAbort]
 *        Called once if the request was aborted via `signal`. This is
 *        NOT an error — the user intentionally stopped the stream. After
 *        this fires no more callbacks. If omitted, abort is a silent
 *        no-op (back-compat with older callers).
 * @param {AbortSignal} [signal]
 *        Optional AbortSignal so the caller can cancel mid-stream.
 *
 * We can't use EventSource because it doesn't support custom request
 * headers (we need Authorization + X-Workspace-Id) or POST bodies. So we
 * fetch() with a text/event-stream body and parse SSE manually.
 */
export async function streamQuery(params, handlers, signal) {
  const { onToken, onDone, onError, onAbort } = handlers;

  // If the signal is already aborted before we start, short-circuit.
  if (signal && signal.aborted) {
    onAbort && onAbort();
    return;
  }

  const headers = await authHeaders({ withWorkspace: true });
  if (!headers) {
    onError(missingCredsError({ withWorkspace: true }));
    return;
  }

  const url = `${API_BASE_URL}/api/query/stream`;
  const body = buildRequestBody(params);

  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        ...headers,
      },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (isAbortError(err, signal)) {
      onAbort && onAbort();
      return;
    }
    onError(new ApiError(
      `Could not reach the backend. Is it running on ${API_BASE_URL}?`,
      { status: 0, errorType: "network_error" }
    ));
    return;
  }

  // Errors come back as JSON bodies, not SSE.
  if (!response.ok) {
    let data = null;
    try { data = await response.json(); } catch { /* non-JSON */ }
    onError(new ApiError(messageForStatus(response.status, data), {
      status: response.status,
      errorType: (data && data.error_type) || "",
    }));
    return;
  }

  if (!response.body) {
    onError(new ApiError("Streaming not supported by this browser.", {
      status: 0, errorType: "stream_unsupported",
    }));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      // Cheap inline check — `reader.read()` will also throw AbortError
      // when the signal aborts mid-read, but checking here means we exit
      // promptly even between two reads.
      if (signal && signal.aborted) {
        onAbort && onAbort();
        return;
      }

      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE messages are separated by a blank line.
      let sep;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const rawMessage = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const parsed = parseSseMessage(rawMessage);
        if (!parsed) continue;
        handleSseMessage(parsed, { onToken, onDone, onError });
      }
    }
  } catch (err) {
    if (isAbortError(err, signal)) {
      onAbort && onAbort();
      return;
    }
    onError(new ApiError("Stream interrupted.", {
      status: 0, errorType: "stream_error",
    }));
  } finally {
    // Best-effort cleanup. If the reader is already closed (done==true)
    // these are no-ops. If aborted mid-read they release resources.
    try { reader.releaseLock(); } catch { /* ignore */ }
  }
}

function parseSseMessage(raw) {
  let event = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (!line || line.startsWith(":")) continue; // blank or comment
    const idx = line.indexOf(":");
    if (idx < 0) continue;
    const field = line.slice(0, idx);
    const value = line.slice(idx + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  if (dataLines.length === 0) return null;
  const dataText = dataLines.join("\n");
  let data;
  try { data = JSON.parse(dataText); } catch { data = { text: dataText }; }
  return { event, data };
}

function handleSseMessage({ event, data }, { onToken, onDone, onError }) {
  if (event === "token") {
    if (typeof data.text === "string") onToken(data.text);
  } else if (event === "done") {
    onDone({
      answer:  typeof data.answer === "string" ? data.answer : "",
      sources: Array.isArray(data.sources) ? data.sources : [],
      debug:   data.debug || {},
    });
  } else if (event === "error") {
    onError(new ApiError(data.detail || "Stream error.", {
      status: 0,
      errorType: data.error_type || "stream_error",
    }));
  }
}

// ====================================================================
// Error message mapping
// ====================================================================
function messageForStatus(status, data) {
  if (status === 401) {
    return "Unauthorized — your session may have expired. Please sign in again.";
  }
  if (status === 403) {
    return (data && data.detail) || "No access to this workspace.";
  }
  if (status === 422 && data && Array.isArray(data.detail)) {
    return formatValidationErrors(data.detail);
  }
  if (status === 429) {
    return (data && data.detail) || "Too many requests. Please slow down.";
  }
  if (status === 504) {
    return "An upstream service timed out. Please try again.";
  }
  if (status >= 500) {
    return (data && data.detail) || "Server error. Please try again.";
  }
  return (data && data.detail) || `Request failed (HTTP ${status}).`;
}

function formatValidationErrors(detailArray) {
  const issues = detailArray
    .map((d) => {
      const field = Array.isArray(d.loc) ? d.loc.slice(1).join(".") : "field";
      return `${field}: ${d.msg}`;
    })
    .slice(0, 3);
  return "Invalid request — " + issues.join("; ");
}