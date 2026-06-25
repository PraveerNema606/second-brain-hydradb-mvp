// SlackSettings — minimal Phase 3 Slack Connect + channel picker UI.
//
// Mounted by App.jsx behind a "Slack" header button so the panel
// doesn't clutter the default view. The component is intentionally
// self-contained (its own state, its own fetches, its own error
// surface) so a future redesign of the surrounding shell doesn't
// disturb it.
//
// Data shape it cares about (matches backend/main.py contracts):
//
//   GET /api/slack/channels   ->
//     { connected: bool, team_name: string, channels: [
//         { slack_channel_id, name, is_selected, is_archived,
//           updated_at }
//     ] }
//
//   POST /api/slack/channels  body { selected_channel_ids: string[], bot_message_channel_ids: string[] }
//   POST /api/slack/ingest    -> { status: "started", channels_queued }
//   GET  /api/slack/connect-url -> { url }
//
// Failure modes are surfaced inline; nothing here blocks the rest of
// the app, and the panel can be closed at any time.

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  disconnectSlack,
  getSlackChannels,
  getSlackConnectUrl,
  runSlackIngest,
  saveSlackChannels,
} from "../api.js";

export default function SlackSettings() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  // The OAuth-callback toast is read once on mount (and the callback
  // query string is scrubbed from the URL at the same time). We never
  // re-set it after that, so a useState pair is overkill — useMemo
  // gives us the same behavior without an unused setter.
  const info = useMemo(() => getInitialCallbackInfo(), []);

  const [connected, setConnected] = useState(false);
  const [teamName, setTeamName] = useState("");
  const [channels, setChannels] = useState([]);
  // The set of currently-checked slack_channel_ids. Initialized from
  // server state on load, then mutated locally by the checkbox handler.
  // We only POST on Save so accidental clicks are recoverable via
  // Cancel (i.e. close-and-reopen).
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [botMessageIds, setBotMessageIds] = useState(() => new Set());
  const [saving, setSaving] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState("");
  const [disconnecting, setDisconnecting] = useState(false);
  const [confirmDisconnect, setConfirmDisconnect] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await getSlackChannels();
      const list = Array.isArray(data?.channels) ? data.channels : [];
      setConnected(Boolean(data?.connected));
      setTeamName(data?.team_name || "");
      setChannels(list);
      // Hydrate the local selected-set and bot-message-set from server
      // truth so the checkboxes start in the right state.
      const next = new Set();
      const botNext = new Set();
      for (const c of list) {
        if (c?.is_selected && c?.slack_channel_id) {
          next.add(c.slack_channel_id);
        }
        if (c?.include_bot_messages && c?.slack_channel_id) {
          botNext.add(c.slack_channel_id);
        }
      }
      setSelectedIds(next);
      setBotMessageIds(botNext);
    } catch (e) {
      // Authentication or workspace missing surfaces here too — we
      // show the message and let the user try again rather than
      // tearing down the panel.
      setConnected(false);
      setChannels([]);
      setSelectedIds(new Set());
      setBotMessageIds(new Set());
      setError(e?.message || "Could not load Slack settings.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  function toggleChannel(id) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    // When deselecting a channel, clear its bot-message opt-in so the
    // checkbox doesn't reappear stale if the channel is re-selected later.
    if (selectedIds.has(id)) {
      setBotMessageIds((prev) => {
        if (!prev.has(id)) return prev;
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }
  }

  function toggleBotMessages(id) {
    setBotMessageIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  async function handleConnect() {
    setError("");
    try {
      const { url } = await getSlackConnectUrl();
      if (url) {
        window.location.href = url;
      } else {
        setError("Slack OAuth is not configured on the server.");
      }
    } catch (e) {
      setError(e?.message || "Could not start Slack connection.");
    }
  }

  async function handleSave() {
    setError("");
    setSaving(true);
    try {
      const ids = Array.from(selectedIds);
      const botIds = Array.from(botMessageIds).filter((id) => selectedIds.has(id));
      await saveSlackChannels(ids, botIds);
      // Re-read from the server so the checkbox state matches what
      // was just persisted (the round-trip also surfaces any
      // transient errors).
      await refresh();
    } catch (e) {
      setError(e?.message || "Could not save channel selection.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDisconnect() {
    setDisconnecting(true);
    setError("");
    try {
      await disconnectSlack();
      // Reset all local state to the unconnected view.
      setConnected(false);
      setTeamName("");
      setChannels([]);
      setSelectedIds(new Set());
      setBotMessageIds(new Set());
      setIngestResult("");
      setConfirmDisconnect(false);
    } catch (e) {
      setError(e?.message || "Could not disconnect Slack.");
    } finally {
      setDisconnecting(false);
    }
  }

  async function handleIngest() {
    setError("");
    setIngestResult("");
    setIngesting(true);
    try {
      const out = await runSlackIngest();
      const queued = out?.channels_queued ?? 0;
      setIngestResult(
        queued > 0
          ? `Ingestion started for ${queued} channel${queued === 1 ? "" : "s"}.`
          : "Ingestion started."
      );
    } catch (e) {
      setError(e?.message || "Could not start ingestion.");
    } finally {
      setIngesting(false);
    }
  }

  // Split channels into selectable (not archived) vs archived so the
  // picker isn't cluttered with channels you probably don't want to
  // ingest. Memoized so the sort runs once per channels change.
  const { active, archived } = useMemo(() => {
    const a = [];
    const ar = [];
    for (const c of channels) {
      if (c?.is_archived) ar.push(c);
      else a.push(c);
    }
    a.sort((x, y) => (x.name || "").localeCompare(y.name || ""));
    ar.sort((x, y) => (x.name || "").localeCompare(y.name || ""));
    return { active: a, archived: ar };
  }, [channels]);

  return (
    <section
      id="slack-settings-panel"
      className="slack-settings"
      aria-label="Slack settings"
    >
      <div className="slack-settings__header">
        <strong className="slack-settings__title">Slack</strong>
        <span className="slack-settings__hint">
          {connected
            ? `Connected to ${teamName || "your Slack workspace"}. Pick channels to ingest.`
            : "Connect a Slack workspace to ingest its channels."}
        </span>
      </div>

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

      {loading ? (
        <p className="slack-settings__muted">Loading…</p>
      ) : !connected ? (
        <div className="slack-settings__actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={handleConnect}
          >
            Connect Slack
          </button>
        </div>
      ) : (
        <>
          <div className="slack-settings__actions">
            <button
              type="button"
              className="btn btn--primary"
              onClick={handleSave}
              disabled={saving || ingesting || disconnecting || confirmDisconnect}
              title="Save the currently selected channels"
            >
              {saving ? "Saving…" : "Save channels"}
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={handleIngest}
              disabled={
                saving || ingesting || disconnecting || confirmDisconnect ||
                selectedIds.size === 0
              }
              title={
                selectedIds.size === 0
                  ? "Select at least one channel first"
                  : "Run ingestion for the selected channels"
              }
            >
              {ingesting ? "Starting…" : "Run ingest"}
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={refresh}
              disabled={saving || ingesting || disconnecting || confirmDisconnect}
              title="Re-fetch channels from Slack"
            >
              Refresh
            </button>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={handleConnect}
              disabled={saving || ingesting || disconnecting || confirmDisconnect}
              title="Re-run the OAuth flow (e.g. after rotating the Slack app)"
            >
              Reconnect
            </button>
            {confirmDisconnect ? (
              <>
                <span className="slack-settings__muted">
                  Remove this workspace?
                </span>
                <button
                  type="button"
                  className="btn btn--armed"
                  onClick={handleDisconnect}
                  disabled={disconnecting}
                >
                  {disconnecting ? "Disconnecting…" : "Yes, disconnect"}
                </button>
                <button
                  type="button"
                  className="btn btn--ghost"
                  onClick={() => setConfirmDisconnect(false)}
                  disabled={disconnecting}
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => setConfirmDisconnect(true)}
                disabled={saving || ingesting || disconnecting}
                title="Revoke the Slack bot token and remove this workspace"
              >
                Disconnect
              </button>
            )}
          </div>

          {ingestResult && (
            <p className="slack-settings__info">{ingestResult}</p>
          )}

          {channels.length === 0 ? (
            <p className="slack-settings__muted">
              No channels visible. Make sure the Slack app has been
              invited to the channels you want to ingest, then click
              Refresh.
            </p>
          ) : (
            <>
              <ul className="slack-settings__list">
                {active.map((c) => (
                  <li key={c.slack_channel_id} className="slack-settings__row">
                    <label className="slack-settings__check">
                      <input
                        type="checkbox"
                        checked={selectedIds.has(c.slack_channel_id)}
                        onChange={() => toggleChannel(c.slack_channel_id)}
                        disabled={saving || ingesting}
                      />
                      <span className="slack-settings__name">
                        #{c.name || c.slack_channel_id}
                      </span>
                      {c.bot_removed && (
                        <span
                          className="slack-settings__tag slack-settings__tag--warning"
                          title="Bot removed — re-invite the app to this channel to resume ingestion"
                        >
                          bot removed
                        </span>
                      )}
                    </label>
                    {selectedIds.has(c.slack_channel_id) && (
                      <label
                        className="slack-settings__bot-toggle"
                        title="Also ingest messages from bots (CI, deployment alerts, etc.)"
                      >
                        <input
                          type="checkbox"
                          checked={botMessageIds.has(c.slack_channel_id)}
                          onChange={() => toggleBotMessages(c.slack_channel_id)}
                          disabled={saving || ingesting}
                        />
                        <span className="slack-settings__muted">include bots</span>
                      </label>
                    )}
                  </li>
                ))}
              </ul>

              {archived.length > 0 && (
                <details className="slack-settings__archived">
                  <summary>
                    Archived ({archived.length})
                  </summary>
                  <ul className="slack-settings__list">
                    {archived.map((c) => (
                      <li
                        key={c.slack_channel_id}
                        className="slack-settings__row slack-settings__row--archived"
                      >
                        <label className="slack-settings__check">
                          <input
                            type="checkbox"
                            checked={selectedIds.has(c.slack_channel_id)}
                            onChange={() => toggleChannel(c.slack_channel_id)}
                            disabled={saving || ingesting}
                          />
                          <span className="slack-settings__name">
                            #{c.name || c.slack_channel_id}
                          </span>
                          <span className="slack-settings__tag">archived</span>
                          {c.bot_removed && (
                            <span
                              className="slack-settings__tag slack-settings__tag--warning"
                              title="Bot removed — re-invite the app to this channel to resume ingestion"
                            >
                              bot removed
                            </span>
                          )}
                        </label>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </>
          )}
        </>
      )}
    </section>
  );
}

/**
 * When the Slack OAuth callback redirects back to the frontend, it
 * appends `?slack_connect=ok&reason=...` (or `=error&reason=...`).
 * We read that here once, on first mount, so the panel can show a
 * little post-connect toast. The query string is then cleared from
 * the URL so a reload doesn't re-show the toast.
 */
function getInitialCallbackInfo() {
  if (typeof window === "undefined") return null;
  try {
    const params = new URLSearchParams(window.location.search);
    const status = params.get("slack_connect");
    if (status !== "ok" && status !== "error") return null;
    const reason = params.get("reason") || "";

    // Strip the OAuth params from the URL so a refresh doesn't re-
    // trigger the toast (and so the address bar stays clean).
    params.delete("slack_connect");
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
          : "Slack connected.",
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
    case "missing_params":     return "Slack returned no code. Please try again.";
    case "exchange_failed":    return "Slack rejected the OAuth exchange. Please try again.";
    case "persist_failed":     return "Could not save the Slack installation. Please try again.";
    case "incomplete_install": return "Slack returned an incomplete installation. Please try again.";
    case "access_denied":      return "Slack connection was canceled.";
    default:                   return reason
      ? `Slack connection failed (${reason}).`
      : "Slack connection failed.";
  }
}