import { useCallback, useEffect, useState } from "react";
import { metaApi } from "./metaApi";
import type { AdsStatus, MetaStatus, AdsProposal, LogPage } from "./metaApi";
import "./MetaConnections.css";
import { MetaLogin } from "./Harness";

const platformNames = {
  whatsapp: "WhatsApp",
  messenger: "Messenger",
  instagram: "Instagram",
};
const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);

export default function MetaConnections() {
  const [meta, setMeta] = useState<MetaStatus | null>(null);
  const [ads, setAds] = useState<AdsStatus | null>(null);
  const [autoAct, setAutoAct] = useState(false);
  const [level, setLevel] = useState<"campaign" | "adset">("adset");
  const [since, setSince] = useState(yesterday);
  const [until, setUntil] = useState(yesterday);
  const [proposals, setProposals] = useState<AdsProposal[]>([]);
  const [log, setLog] = useState<LogPage>({ items: [], next_after: null });
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      const [messages, status] = await Promise.all([
        metaApi.status(),
        metaApi.adsStatus(),
      ]);
      setMeta(messages);
      setAds(status);
      setAutoAct(status.auto_act);
      const history = await metaApi
        .log()
        .catch(() => ({ items: [], next_after: null }));
      setLog(history);
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Connections unavailable.",
      );
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);
  async function perform(action: () => Promise<void>) {
    if (busy) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await action();
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Connection request failed. Try again.",
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <section
      className="meta-connections teach-card"
      aria-labelledby="meta-connections-title"
    >
      <div className="section-heading">
        <div>
          <p className="eyebrow">BUILT FOR YOUR META APPS</p>
          <h2 id="meta-connections-title">Connect your decisions.</h2>
        </div>
        <button
          className="connection-button"
          disabled={busy}
          onClick={() => void perform(load)}
        >
          Refresh connections
        </button>
      </div>
      <p className="section-description">
        Incoming messages become Ren decisions in your workspace inbox. Confirm
        or correct them to teach Ren your preferences.
      </p>
      {ads?.demo && (
        <p className="connector-demo">
          Synthetic demo · simulated Ren proposals and mocked Meta transport. No
          live calls or account changes.
        </p>
      )}
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      <p className="connector-notice" role="status">
        {notice}
      </p>
      <MetaLogin />
      <div className="message-connections">
        {(Object.keys(platformNames) as (keyof typeof platformNames)[]).map(
          (platform) => (
            <article key={platform} className="message-connection">
              <h3>{platformNames[platform]}</h3>
              <p>
                {meta?.accounts[platform].length
                  ? `${meta.accounts[platform].length} receiving account connected`
                  : "No receiving account mapped"}
              </p>
              <span>
                {meta?.model_ready &&
                meta?.signing_configured &&
                meta?.accounts[platform].length
                  ? "Ready for signed messages"
                  : "Waiting for connection setup"}
              </span>
              <small>
                {platform === "whatsapp" && meta?.whatsapp_send_enabled
                  ? "WhatsApp sends enabled by operator"
                  : platform !== "whatsapp" && meta?.message_send_enabled
                    ? "Message sends enabled by operator"
                    : "Outbound replies remain previews"}
              </small>
            </article>
          ),
        )}
      </div>
      <div className="ads-heading">
        <div>
          <h3>Ads · sandbox insights</h3>
          <p>
            Read campaign or ad set spend, CPA and CTR. Ren proposes pause,
            keep, raise budget or lower budget.
          </p>
        </div>
        <span className="connector-state">
          {ads?.server_send_enabled
            ? "Execution switch on"
            : "Execution switch off"}
        </span>
      </div>
      <p className="connector-detail">
        {ads?.sandbox_allowed
          ? `Mapped sandbox: ${ads.account}`
          : "No allowed sandbox Ads account mapped."}{" "}
        {!ads?.token_configured && "Read access needs setup."}
      </p>
      {!ads?.workspace_ready && (
        <p className="connector-detail">
          Use a workspace trained with these four labels: pause, keep, raise
          budget, lower budget.
        </p>
      )}
      <form
        className="ads-preference"
        onSubmit={(event) => {
          event.preventDefault();
          void perform(async () => {
            const status = await metaApi.preferences(autoAct);
            setAds(status);
            setNotice(
              status.auto_act
                ? "Auto-act preference saved. Certification and the server execution switch are still required."
                : "Auto-act off. Ads decisions remain proposals.",
            );
            setLog(await metaApi.log());
          });
        }}
      >
        <label>
          <input
            type="checkbox"
            checked={autoAct}
            disabled={busy || !ads?.workspace_ready}
            onChange={(event) => setAutoAct(event.target.checked)}
          />
          <span>
            Allow certified Ren decisions to act automatically in this workspace
          </span>
        </label>
        <p>
          Off by default. Execution also requires the server switch. Budget
          steps: {ads?.budget_step_percent ?? 10}%. A human answer cancels the
          original proposal.
        </p>
        <button
          className="connection-button"
          disabled={busy || !ads?.workspace_ready || autoAct === ads.auto_act}
        >
          Save auto-act preference
        </button>
      </form>
      <form
        className="ads-fetch"
        onSubmit={(event) => {
          event.preventDefault();
          void perform(async () => {
            const result = await metaApi.sync(level, since, until);
            setProposals(result);
            setLog(await metaApi.log());
            setNotice(
              `${result.length} Ren proposal${result.length === 1 ? "" : "s"} loaded. Uncertified decisions wait in your inbox.`,
            );
          });
        }}
      >
        <label>
          Insights level
          <select
            value={level}
            onChange={(event) =>
              setLevel(event.target.value as "campaign" | "adset")
            }
            disabled={busy}
          >
            <option value="adset">Ad set</option>
            <option value="campaign">Campaign</option>
          </select>
        </label>
        <label>
          From
          <input
            type="date"
            value={since}
            max={yesterday}
            onChange={(event) => setSince(event.target.value)}
            required
            disabled={busy}
          />
        </label>
        <label>
          Through
          <input
            type="date"
            value={until}
            min={since}
            max={yesterday}
            onChange={(event) => setUntil(event.target.value)}
            required
            disabled={busy}
          />
        </label>
        <button
          className="primary-button"
          disabled={
            busy ||
            !ads?.workspace_ready ||
            !ads?.sandbox_allowed ||
            !ads?.token_configured
          }
        >
          {busy ? "Working…" : "Fetch sandbox insights"}
        </button>
      </form>
      <div className="ads-proposals">
        {proposals.map((proposal) => (
          <article key={proposal.item_id} className="ads-proposal">
            <div>
              <span className="connector-state">
                Ads · {proposal.insight.level}
              </span>
              <span className="proposal-status">
                {proposal.status.replaceAll("_", " ")}
              </span>
            </div>
            <h4>{proposal.insight.name}</h4>
            <p className="proposal-dates">
              {proposal.insight.since} – {proposal.insight.until}
            </p>
            <dl>
              <div>
                <dt>Spend</dt>
                <dd>
                  {proposal.insight.currency} {proposal.insight.spend}
                </dd>
              </div>
              <div>
                <dt>CPA ({proposal.insight.cpa_action_type})</dt>
                <dd>
                  {proposal.insight.cpa == null
                    ? "Unavailable"
                    : `${proposal.insight.currency} ${proposal.insight.cpa}`}
                </dd>
              </div>
              <div>
                <dt>CTR</dt>
                <dd>
                  {proposal.insight.ctr == null
                    ? "Unavailable"
                    : `${proposal.insight.ctr}%`}
                </dd>
              </div>
            </dl>
            <p className="ren-proposal">
              Ren proposes <strong>{proposal.action}</strong>
            </p>
            <p>
              {proposal.decision.auto
                ? "Ren marked this decision automatic. Execution requires its current certification and both switches."
                : "Needs your judgment. Answer in the workspace inbox."}
            </p>
            {ads?.auto_act &&
              ads.server_send_enabled &&
              proposal.decision.auto &&
              proposal.status === "proposed" && (
                <button
                  className="connection-button"
                  disabled={busy}
                  onClick={() =>
                    void perform(async () => {
                      const updated = await metaApi.execute(proposal.item_id);
                      setProposals((old) =>
                        old.map((row) =>
                          row.item_id === updated.item_id ? updated : row,
                        ),
                      );
                      setLog(await metaApi.log());
                      setNotice(
                        `Proposal ${updated.status.replaceAll("_", " ")}.`,
                      );
                    })
                  }
                >
                  Execute certified proposal
                </button>
              )}
          </article>
        ))}
      </div>
      <details className="ads-log">
        <summary>Action log · proposed and executed</summary>
        {!log.items.length ? (
          <p>No Ads actions logged yet.</p>
        ) : (
          <ol>
            {log.items.map((row) => (
              <li key={row.id}>
                <strong>{row.event.replaceAll("_", " ")}</strong>
                <span>
                  {typeof row.details.action === "string"
                    ? row.details.action
                    : typeof row.details.auto_act === "boolean"
                      ? `Auto-act ${row.details.auto_act ? "on" : "off"}`
                      : ""}
                </span>
                <time dateTime={row.created_at}>
                  {new Date(row.created_at).toLocaleString()}
                </time>
              </li>
            ))}
          </ol>
        )}
        {log.next_after != null && (
          <button
            className="connection-button"
            disabled={busy}
            onClick={() =>
              void perform(async () => {
                const next = await metaApi.log(log.next_after!);
                setLog((old) => ({
                  ...next,
                  items: [...old.items, ...next.items],
                }));
              })
            }
          >
            More log entries
          </button>
        )}
      </details>
      <p className="connector-detail">
        Secnd is independently built for Meta’s platform, and is not an official
        Meta product. Ren makes each decision.
      </p>
    </section>
  );
}
