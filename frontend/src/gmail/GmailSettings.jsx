// GmailSettings — Phase 8 Gmail Connect + label picker UI.
//
// Mirrors slack/SlackSettings.jsx in shape so the two connectors feel
// the same. The differences come from Gmail's model:
//
//   - One workspace can hold MULTIPLE Gmail connections (personal +
//     shared mailbox + ...). The Slack model is one installation per
//     workspace, so this component has a connection picker on top of
//     the label picker.
//   - Labels (Gmail's equivalent of channels) belong to a CONNECTION,
//     not directly to the workspace. The picker reloads whenever the
//     active connection changes.
//   - The OAuth callback uses ?gmail_connect=ok|error&reason=... so
//     it doesn't collide with the Slack callback's ?slack_connect=...
//
// Data shapes (matches backend/main.py contracts):
//
//   GET    /api/gmail/connections ->
//     { connections: [{ id, email, status, connected_at, ... }] }
//   DELETE /api/gmail/connections/{id} -> { deleted: true } | 404
//   GET    /api/gmail/labels?connection_id=... ->
//     { connected: bool, labels: [{ label_id, name, type, is_selected }] }
//   POST   /api/gmail/labels       body { connection_id, selected_label_ids }
//   POST   /api/gmail/ingest       body { connection_id }
//                                  -> { status: "started", labels_queued }
//   GET    /api/gmail/connect-url -> { url }  (503 when not configured)
//
// Failure modes surface inline; nothing here blocks the rest of the app.

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  deleteGmailConnection,
  getGmailConnectUrl,
  getGmailLabels,
  listGmailConnections,
  runGmailIngest,
  saveGmailLabels,
} from "../api.js";

export default function GmailSettings() {
  // Top-level state
  const [loadingConns, setLoadingConns] = useState(true);
  const [error, setError] = useState("");
  // OAuth-callback toast is read once on mount (and the query string
  // is scrubbed at the same time), so a useMemo gives us the same
  // behavior without an unused setter.
  const info = useMemo(() => getInitialCallbackInfo(), []);

  const [connections, setConnections] = useState([]);
  const [activeConnId, setActiveConnId] = useState("");

  // Per-connection label state. Re-fetched whenever activeConnId changes.
  const [loadingLabels, setLoadingLabels] = useState(false);
  const [labels, setLabels] = useState([]);
  const [selectedIds, setSelectedIds] = useState(() => new Set());

  // Action flags
  const [saving, setSaving]       = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [removing, setRemoving]   = useState(false);
  const [ingestResult, setIngestResult] = useState("");

  // ─── Connections ───────────────────────────────────────────────────
  const refreshConnections = useCallback(async () => {
    setLoadingConns(true);
    setError("");
    try {
      const data = await listGmailConnections();
      const list = Array.isArray(data?.connections) ? data.connections : [];
      setConnections(list);
      // If the currently-active connection vanished (e.g. user just
      // deleted it), pick the first remaining one; otherwise leave
      // the selection alone.
      setActiveConnId((prev) => {
        if (prev && list.some((c) => c.id === prev)) return prev;
        return list[0]?.id || "";
      });
    } catch (e) {
      setConnections([]);
      setActiveConnId("");
      setError(e?.message || "Could not load Gmail connections.");
    } finally {
      setLoadingConns(false);
    }
  }, []);

  useEffect(() => {
    refreshConnections();
  }, [refreshConnections]);

  // ─── Labels for the active connection ──────────────────────────────
  const refreshLabels = useCallback(async (connId) => {
    if (!connId) {
      setLabels([]);
      setSelectedIds(new Set());
      return;
    }
    setLoadingLabels(true);
    setError("");
    setIngestResult("");
    try {
      const data = await getGmailLabels(connId);
      const list = Array.isArray(data?.labels) ? data.labels : [];
      setLabels(list);
      const next = new Set();
      for (const l of list) {
        if (l?.is_selected && l?.label_id) {
          next.add(l.label_id);
        }
      }
      setSelectedIds(next);
    } catch (e) {
      setLabels([]);
      setSelectedIds(new Set());
      setError(e?.message || "Could not load Gmail labels.");
    } finally {
      setLoadingLabels(false);
    }
  }, []);

  useEffect(() => {
    refreshLabels(activeConnId);
  }, [activeConnId, refreshLabels]);

  function toggleLabel(id) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  // ─── Actions ───────────────────────────────────────────────────────
  async function handleConnect() {
    setError("");
    try {
      const { url } = await getGmailConnectUrl();
      if (url) {
        window.location.href = url;
      } else {
        // The backend returns 503 with detail "not configured" when
        // GMAIL_CLIENT_ID etc. aren't set; we should never reach this
        // branch in production, but keep a sensible message just in
        // case.
        setError("Gmail OAuth is not configured on the server.");
      }
    } catch (e) {
      setError(e?.message || "Could not start Gmail connection.");
    }
  }

  async function handleSave() {
    if (!activeConnId) return;
    setError("");
    setSaving(true);
    try {
      const ids = Array.from(selectedIds);
      await saveGmailLabels(activeConnId, ids);
      // Re-read so the UI mirrors persisted truth.
      await refreshLabels(activeConnId);
    } catch (e) {
      setError(e?.message || "Could not save label selection.");
    } finally {
      setSaving(false);
    }
  }

  async function handleIngest() {
    if (!activeConnId) return;
    setError("");
    setIngestResult("");
    setIngesting(true);
    try {
      const out = await runGmailIngest(activeConnId);
      const queued = out?.labels_queued ?? 0;
      setIngestResult(
        queued > 0
          ? `Ingestion started for ${queued} label${queued === 1 ? "" : "s"}.`
          : "Ingestion started."
      );
    } catch (e) {
      setError(e?.message || "Could not start Gmail ingestion.");
    } finally {
      setIngesting(false);
    }
  }

  async function handleRemove() {
    if (!activeConnId) return;
    setError("");
    setRemoving(true);
    try {
      await deleteGmailConnection(activeConnId);
      // After delete, refresh the connections list. The active
      // connection effect will null itself out (or fall back to the
      // first remaining one).
      await refreshConnections();
    } catch (e) {
      setError(e?.message || "Could not remove Gmail connection.");
    } finally {
      setRemoving(false);
    }
  }

  // Sort labels: system (INBOX, SENT, ...) first, then user labels by name.
  // We sort the union; the existing Slack pattern of splitting "archived"
  // doesn't apply (Gmail labels don't have an archived state).
  const sortedLabels = useMemo(() => {
    const list = [...labels];
    list.sort((a, b) => {
      const aSys = (a.type || "user") === "system" ? 0 : 1;
      const bSys = (b.type || "user") === "system" ? 0 : 1;
      if (aSys !== bSys) return aSys - bSys;
      return (a.name || a.label_id || "").localeCompare(
        b.name || b.label_id || "",
      );
    });
    return list;
  }, [labels]);

  const activeConn = useMemo(
    () => connections.find((c) => c.id === activeConnId) || null,
    [connections, activeConnId],
  );

  const connected = connections.length > 0;
  const busy = saving || ingesting || removing;

  // Phase: connection health. The backend sets gmail_connections.status
  // to 'revoked' (grant gone -- user must reconnect) or 'error'
  // (transient) when a token refresh fails at the _authed_request
  // chokepoint; 'active' otherwise. We surface it per-connection so one
  // broken mailbox never mislabels a healthy one.
  const activeStatus = (activeConn?.status || "active").toLowerCase();
  const statusBadge = useMemo(() => {
    switch (activeStatus) {
      case "revoked":
        return { label: "Access revoked", kind: "error" };
      case "error":
        return { label: "Connection error", kind: "error" };
      default:
        return null;
    }
  }, [activeStatus]);

  // Phase 11: render a "Last synced X ago" hint per active connection.
  // The data comes from the enriched /api/gmail/connections response;
  // we never make a separate fetch for this. Falls back to "Never
  // synced" when no label has been processed yet.
  const lastSyncedHint = useMemo(() => {
    const ts = activeConn?.sync_summary?.last_synced_at;
    if (!ts) return "Never synced";
    return `Last synced ${formatRelativeTime(ts)}`;
  }, [activeConn]);

  return (
    <section
      id="gmail-settings-panel"
      className="slack-settings"
      aria-label="Gmail settings"
    >
      <div className="slack-settings__header">
        <strong className="slack-settings__title">Gmail</strong>
        <span className="slack-settings__hint">
          {connected
            ? `Connected as ${activeConn?.email || "Gmail"}. Pick labels to ingest.`
            : "Connect a Gmail account to ingest selected labels."}
          {connected && (
            <>
              {" "}
              <span className="slack-settings__muted">
                ({lastSyncedHint})
              </span>
            </>
          )}
          {connected && statusBadge && (
            <>
              {" "}
              <span
                className={`slack-settings__tag slack-settings__tag--${statusBadge.kind}`}
              >
                {statusBadge.label}
              </span>
            </>
          )}
        </span>
      </div>

      {connected && statusBadge && (
        <p className="slack-settings__error" role="alert">
          {activeStatus === "revoked"
            ? "Google revoked access to this mailbox (or it expired). Syncing is paused until you reconnect."
            : "We hit an error refreshing this connection. We'll retry automatically; if it persists, reconnect the account."}
          {activeStatus === "revoked" && (
            <>
              {" "}
              <button
                type="button"
                className="btn btn--ghost"
                onClick={handleConnect}
                disabled={busy}
                title="Reconnect this Gmail account"
              >
                Reconnect
              </button>
            </>
          )}
        </p>
      )}

      {info && (
        <p
          className={`slack-settings__${
            info.kind === "error" ? "error" : "info"
          }`}
          role="alert"
        >
          {info.message}
        </p>
      )}

      {error && (
        <p className="slack-settings__error" role="alert">
          {error}
        </p>
      )}

      {loadingConns ? (
        <p className="slack-settings__muted">Loading…</p>
      ) : !connected ? (
        <div className="slack-settings__actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={handleConnect}
          >
            Connect Gmail
          </button>
        </div>
      ) : (
        <>
          {/* Connection picker — only shown when there's more than one,
              to keep the single-mailbox case clean. */}
          {connections.length > 1 && (
            <div className="slack-settings__actions">
              <label className="slack-settings__check">
                <span className="slack-settings__name">Account:</span>
                <select
                  value={activeConnId}
                  onChange={(e) => setActiveConnId(e.target.value)}
                  disabled={busy}
                >
                  {connections.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.email || c.id}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}

          <div className="slack-settings__actions">
            <button
              type="button"
              className="btn btn--primary"
              onClick={handleSave}
              disabled={busy || !activeConnId}
              title="Save the currently selected labels"
            >
              {saving ? "Saving…" : "Save labels"}
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={handleIngest}
              disabled={busy || selectedIds.size === 0 || !activeConnId}
              title={
                selectedIds.size === 0
                  ? "Select at least one label first"
                  : "Run ingestion for the selected labels"
              }
            >
              {ingesting ? "Starting…" : "Run ingest"}
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={() => refreshLabels(activeConnId)}
              disabled={busy || !activeConnId}
              title="Re-fetch labels from Gmail"
            >
              Refresh
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={handleConnect}
              disabled={busy}
              title="Connect an additional Gmail account"
            >
              Add account
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={handleRemove}
              disabled={busy || !activeConnId}
              title="Disconnect this Gmail account"
            >
              {removing ? "Removing…" : "Disconnect"}
            </button>
          </div>

          {ingestResult && (
            <p className="slack-settings__info">{ingestResult}</p>
          )}

          {loadingLabels ? (
            <p className="slack-settings__muted">Loading labels…</p>
          ) : sortedLabels.length === 0 ? (
            <p className="slack-settings__muted">
              No labels visible. Click Refresh after creating a label
              in Gmail, or reconnect this account.
            </p>
          ) : (
            <ul className="slack-settings__list">
              {sortedLabels.map((l) => (
                <li key={l.label_id} className="slack-settings__row">
                  <label className="slack-settings__check">
                    <input
                      type="checkbox"
                      checked={selectedIds.has(l.label_id)}
                      onChange={() => toggleLabel(l.label_id)}
                      disabled={busy}
                    />
                    <span className="slack-settings__name">
                      {l.name || l.label_id}
                    </span>
                    {l.type === "system" && (
                      <span className="slack-settings__tag">system</span>
                    )}
                  </label>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </section>
  );
}

/**
 * When the Gmail OAuth callback redirects back to the frontend, it
 * appends `?gmail_connect=ok&reason=...` (or `=error&reason=...`).
 * We read that here once, on first mount, so the panel can show a
 * post-connect toast. The query string is then cleared from the URL
 * so a reload doesn't re-show the toast. Same shape as the Slack
 * helper; we keep the two separate so the Gmail toast survives even
 * if both connectors finish OAuth in the same tab.
 */
function getInitialCallbackInfo() {
  if (typeof window === "undefined") return null;
  try {
    const params = new URLSearchParams(window.location.search);
    const status = params.get("gmail_connect");
    if (status !== "ok" && status !== "error") return null;
    const reason = params.get("reason") || "";

    params.delete("gmail_connect");
    params.delete("reason");
    const qs = params.toString();
    const newUrl =
      window.location.pathname + (qs ? `?${qs}` : "") + window.location.hash;
    window.history.replaceState({}, "", newUrl);

    if (status === "ok") {
      return {
        kind: "info",
        message: reason
          ? `Connected to ${reason}.`
          : "Gmail connected.",
      };
    }
    return {
      kind: "error",
      message: friendlyOauthError(reason),
    };
  } catch {
    return null;
  }
}

function friendlyOauthError(reason) {
  switch (reason) {
    case "bad_state":          return "OAuth state was invalid or expired. Please try again.";
    case "missing_params":     return "Google returned no code. Please try again.";
    case "exchange_failed":    return "Google rejected the OAuth exchange. Please try again.";
    case "userinfo_failed":    return "Could not read Google profile. Please try again.";
    case "persist_failed":     return "Could not save the Gmail connection. Please try again.";
    case "incomplete_install": return "Google returned an incomplete profile. Please try again.";
    case "access_denied":      return "Gmail connection was canceled.";
    default:                   return reason
      ? `Gmail connection failed (${reason}).`
      : "Gmail connection failed.";
  }
}
/**
 * Format an ISO 8601 timestamp as a short human-readable relative
 * time ("2 min ago", "5 hours ago", "3 days ago"). Used by the
 * Phase 11 "Last synced X ago" hint. Falls back to the raw value
 * when parsing fails so the UI never shows "NaN".
 */
function formatRelativeTime(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  const ms = Date.now() - date.getTime();
  if (Number.isNaN(ms)) return iso;
  if (ms < 0) return "just now";                // clock skew

  const SEC = 1000;
  const MIN = 60 * SEC;
  const HOUR = 60 * MIN;
  const DAY = 24 * HOUR;

  if (ms < 30 * SEC)  return "just now";
  if (ms < MIN)       return `${Math.floor(ms / SEC)} sec ago`;
  if (ms < HOUR)      {
    const n = Math.floor(ms / MIN);
    return `${n} min ago`;
  }
  if (ms < DAY) {
    const n = Math.floor(ms / HOUR);
    return `${n} hour${n === 1 ? "" : "s"} ago`;
  }
  const n = Math.floor(ms / DAY);
  return `${n} day${n === 1 ? "" : "s"} ago`;
}