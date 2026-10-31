import { useCallback, useEffect, useRef, useState } from "react";
import { api, percent, workspaceId } from "./api";
import type { Decision, Page } from "./api";
import "./Companion.css";
import SourceIcon from "./SourceIcon";
import { decisionBasis } from "./decisionBasis";

type Tab = "Pending" | "Answered automatically" | "Overridden";
const tabNames = {
  Pending: "Needs you",
  "Answered automatically": "Done by Ren",
  Overridden: "Corrected",
};
type Undo = { key: string; label: string };
const empty: Page<Decision> = { items: [], next_cursor: null };
const sources = [
  "whatsapp",
  "messenger",
  "instagram",
  "ads",
  "workspace",
] as const;
const names = {
  whatsapp: "WhatsApp",
  messenger: "Messenger",
  instagram: "Instagram",
  ads: "Ads",
  workspace: "Business",
};
export default function Companion({ onAnswer }: { onAnswer: () => void }) {
  const [tab, setTab] = useState<Tab>("Pending");
  const [pages, setPages] = useState<Record<Tab, Page<Decision>>>({
    Pending: empty,
    "Answered automatically": empty,
    Overridden: empty,
  });
  const [activeId, setActiveId] = useState<string | null>(null);
  const cardNodes = useRef(new Map<string, HTMLElement>());
  const [fixture, setFixture] = useState(false);
  const [demo, setDemo] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [undo, setUndo] = useState<Undo[]>(() => {
    try {
      const saved: unknown = JSON.parse(
        sessionStorage.getItem(`inbox-undo-${workspaceId}`) ?? "[]",
      );
      return Array.isArray(saved)
        ? saved.filter(
            (row): row is Undo =>
              typeof row?.key === "string" && typeof row?.label === "string",
          )
        : [];
    } catch {
      return [];
    }
  });
  const [other, setOther] = useState<string | null>(null);
  const [otherLabel, setOtherLabel] = useState("");
  const [labels, setLabels] = useState<string[]>([]);
  const paged = useRef(false);
  const keys = useRef(new Map<string, string>());
  const revision = useRef(0);
  const mutating = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const tabButtons = useRef<(HTMLButtonElement | null)[]>([]);
  useEffect(() => {
    sessionStorage.setItem(`inbox-undo-${workspaceId}`, JSON.stringify(undo));
  }, [undo]);
  const fetchPage = (value: Tab, cursor?: string) =>
    value === "Pending"
      ? api.queue(cursor)
      : value === "Overridden"
        ? api.overridden(cursor)
        : api.feed(cursor);

  const load = useCallback(async () => {
    if (mutating.current) return;
    const request = ++revision.current;
    try {
      const [pending, automatic, overridden, labelList] = await Promise.all([
        api.queue(),
        api.feed(),
        api.overridden(),
        api.labels(),
      ]);
      if (request !== revision.current || mutating.current) return;
      paged.current = false;
      setLabels(labelList.data);
      setPages({
        Pending: pending.data,
        "Answered automatically": automatic.data,
        Overridden: overridden.data,
      });
      setActiveId(old => pending.data.items.some(item => item.item_id === old) ? old : null);
      setFixture(pending.fixture || automatic.fixture);
      setDemo(pending.demo);
    } catch (cause) {
      if (request === revision.current)
        setError(
          cause instanceof Error ? cause.message : "Could not load decisions.",
        );
    } finally {
      if (request === revision.current) setLoading(false);
    }
  }, []);
  useEffect(() => {
    void load();
    const timer = window.setInterval(() => {
      // Leave pagination and expanded forms stable while the person reviews them.
      if (document.visibilityState === "visible" && !other && !paged.current)
        void load();
    }, 10000);
    return () => {
      window.clearInterval(timer);
      revision.current++;
    };
  }, [load, other]);

  async function answer(item: Decision, label: string) {
    if (mutating.current) return;
    mutating.current = true;
    revision.current++;
    setBusy(true);
    setError("");
    setNotice("");
    const action = `${item.item_id}\0${label}`;
    const key = keys.current.get(action) ?? crypto.randomUUID();
    keys.current.set(action, key);
    try {
      const result = await api.answer(item.item_id, label, key);
      keys.current.delete(action);
      setNotice(
        result.fixture
          ? "Preview only. Answer has not been saved."
          : `Saved ${label} to Ren’s history. ${
              result.data.action_status === "message_sent"
                ? "Message sent. This app cannot retract it."
                : result.data.action_status === "ad_executed"
                  ? "Ad change applied. Undo is available in Activity."
                  : result.data.action_status?.endsWith("delivery_unknown")
                    ? "Delivery is unknown. Check Activity before trying another action."
                    : result.data.action_status
                      ? "Proposal saved. No live change was applied."
                      : ""
            }`,
      );
      if (!result.fixture) {
        setUndo((previous) => [...previous, { key, label }]);
        onAnswer();
      }
      setOther(null);
      setOtherLabel("");
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not save answer. Try again.",
      );
    } finally {
      mutating.current = false;
      setBusy(false);
      await load();
      requestAnimationFrame(() => heading.current?.focus());
    }
  }
  async function undoLast() {
    const last = undo.at(-1);
    if (!last || mutating.current) return;
    mutating.current = true;
    revision.current++;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await api.undo(last.key);
      setUndo((previous) => previous.slice(0, -1));
      setNotice(
        `Undid ${last.label}. Previous answer restored; Ren’s history updated.`,
      );
      onAnswer();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not undo answer.",
      );
    } finally {
      mutating.current = false;
      setBusy(false);
      await load();
      requestAnimationFrame(() => heading.current?.focus());
    }
  }
  async function more() {
    if (mutating.current || !pages[tab].next_cursor) return;
    revision.current++;
    mutating.current = true;
    setBusy(true);
    try {
      const next = await fetchPage(tab, pages[tab].next_cursor!);
      paged.current = true;
      setPages((previous) => ({
        ...previous,
        [tab]: {
          ...next.data,
          items: [
            ...previous[tab].items,
            ...next.data.items.filter(
              (row) =>
                !previous[tab].items.some((old) => old.item_id === row.item_id),
            ),
          ],
        },
      }));
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not load more decisions.",
      );
    } finally {
      mutating.current = false;
      setBusy(false);
    }
  }
  function changeTab(value: Tab) {
    setTab(value);
    setActiveId(null);
    setOther(null);
  }
  useEffect(() => {
    function keyboard(event: KeyboardEvent) {
      const target = event.target as HTMLElement;
      if (event.ctrlKey || event.metaKey || event.altKey || target.closest("input, textarea, select, [contenteditable=true]")) return;
      if (busy) return;
      const items = sources.flatMap(source => pages[tab].items.filter(item => (item.source_app ?? item.item?.source_app ?? "workspace") === source));
      const index = items.findIndex(item => item.item_id === activeId);
      const item = items[index < 0 ? 0 : index];
      if (["j", "k"].includes(event.key) && items.length) {
        event.preventDefault();
        const next = items[index < 0 ? 0 : (index + (event.key === "j" ? 1 : -1) + items.length) % items.length];
        setActiveId(next.item_id); cardNodes.current.get(next.item_id)?.focus();
      } else if (event.key === "y" && item && tab === "Pending") {
        event.preventDefault(); void answer(item, item.label);
      } else if (/^[1-4]$/.test(event.key) && item) {
        const options = Object.keys(item.probabilities).filter(label => label !== (item.answer_label ?? item.label));
        const label = options[Number(event.key) - 1];
        if (label) { event.preventDefault(); void answer(item, label); }
      } else if (event.key === "u" && undo.length) { event.preventDefault(); void undoLast(); }
    }
    window.addEventListener("keydown", keyboard);
    return () => window.removeEventListener("keydown", keyboard);
  });
  function card(item: Decision) {
    const source = item.source_app ?? item.item?.source_app ?? "workspace";
    const choices = Object.keys(item.probabilities).length ? Object.keys(item.probabilities) : labels;
    const alternatives = choices.filter(label => label !== (item.answer_label ?? item.label));
    return (
      <article
        className={`inbox-card ${activeId === item.item_id ? "is-current" : ""}`}
        ref={node => { if (node) cardNodes.current.set(item.item_id, node); else cardNodes.current.delete(item.item_id); }}
        tabIndex={0}
        onFocus={() => setActiveId(item.item_id)}
        onClick={() => setActiveId(item.item_id)}
        key={item.item_id}
        aria-label={item.item?.text || item.item_id}
      >
        <div className="decision-meta">
          <span className={`source-badge source-${source}`}>
            <SourceIcon source={source} />{names[source]}
          </span>
          <span>
            {item.spot_check
              ? "Spot-check"
              : tab === "Pending"
                ? "Needs your judgment"
                : item.model_version}
          </span>
        </div>
        <h3>{item.item?.text || item.item_id}</h3>
        {item.item?.amount != null && (
          <p className="decision-amount">
            Amount: {item.item.amount.toLocaleString()}
            {item.item.date && ` · ${item.item.date}`}
          </p>
        )}
        <div className="inbox-proposal">
          <span>{tab === "Overridden" ? "Your answer" : "Ren suggests"}</span>
          <strong>{item.answer_label ?? item.label}</strong>
          {tab !== "Overridden" && <span>{percent(item.prob)} confidence</span>}
        </div>
        {tab === "Overridden" && (
          <p className="original-answer">
            Ren originally suggested {item.label}. Your correction is in its
            history.
          </p>
        )}
        {tab === "Answered automatically" && <p className="decision-proof">{decisionBasis(item)}</p>}
        <details className="inbox-evidence">
          <summary>Why · nearest past decisions</summary>
          {item.reasons.neighbours.length ? (
            <ol>
              {item.reasons.neighbours.map((row) => (
                <li key={row.item_id}>
                  <p>{row.text}</p>
                  <span>
                    {row.label} · {percent(row.similarity)} similarity
                  </span>
                </li>
              ))}
            </ol>
          ) : (
            <p>No past-decision evidence saved for this prediction.</p>
          )}
          <p className="evidence-note">
            Similar examples provide context. Ren makes the decision.
          </p>
        </details>
        <div className="inbox-actions">
          {tab === "Pending" && (
            <button
              className="primary-button"
              disabled={busy}
              onClick={() => void answer(item, item.label)}
            >
              Confirm {item.label}
            </button>
          )}
          <button
            className="secondary-button"
            disabled={busy}
            aria-expanded={other === item.item_id}
            onClick={() => {
              setOther(other === item.item_id ? null : item.item_id);
              setOtherLabel(
                choices.find((x) => x !== (item.answer_label ?? item.label)) ??
                  choices[0],
              );
            }}
          >
            Correct this
          </button>
        </div>
        <div className="quick-corrections" aria-label="Other actions">
          {alternatives.slice(0, 4).map((label, index) => <button className="text-button" key={label} disabled={busy} onClick={() => void answer(item, label)}><kbd>{index + 1}</kbd> {label}</button>)}
        </div>
        {other === item.item_id && (
          <form
            className="inbox-correction"
            onSubmit={(event) => {
              event.preventDefault();
              if (otherLabel) void answer(item, otherLabel);
            }}
          >
            <label>
              Choose action
              <select
                autoFocus
                value={otherLabel}
                onChange={(event) => setOtherLabel(event.target.value)}
                disabled={busy}
              >
                {choices.map((label) => (
                  <option key={label}>{label}</option>
                ))}
              </select>
            </label>
            <p className="evidence-note">
              All actions for this decision are available.
            </p>
            <button className="primary-button" disabled={busy || !otherLabel}>
              Save correction
            </button>
          </form>
        )}
      </article>
    );
  }
  return (
    <section className="decision-inbox" aria-labelledby="inbox-title">
      <div className="inbox-heading">
        <div>
          <p className="eyebrow">REN FOR YOUR META APPS</p>
          <h1 id="inbox-title" ref={heading} tabIndex={-1}>
            Your decision inbox.
          </h1>
          <p>One clear choice at a time. Confirm Ren’s suggestion or make your own call.</p>
        </div>
        <button
          className="secondary-button"
          disabled={busy}
          onClick={() => {
            setError("");
            void load();
          }}
        >
          Refresh inbox
        </button>
      </div>
      <details className="business-switch"><summary>Switch business</summary><form className="workspace-picker" method="get">
        <label htmlFor="workspace-id">Business</label>
        <input
          id="workspace-id"
          name="workspace"
          defaultValue={workspaceId}
          required
          pattern="[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}"
          maxLength={64}
        />
        <button className="secondary-button">Open business</button>
      </form></details>
      {fixture && (
        <p className="preview-banner">
          Preview · illustrative decisions. Answers are not saved. Add your past decisions to start.
        </p>
      )}
      {demo && (
        <p className="preview-banner">
          Synthetic demo · illustrative messages, local answers only.
          Certification below uses synthetic calibration data, not Meta traffic.
        </p>
      )}
      <p className="keyboard-hint"><kbd>j</kbd> / <kbd>k</kbd> move <kbd>y</kbd> confirm <kbd>1–4</kbd> correct <kbd>u</kbd> undo</p>
      <div className="inbox-tabs" role="tablist" aria-label="Decision status">
        {(["Pending", "Answered automatically", "Overridden"] as const).map(
          (value, index) => (
            <button
              key={value}
              ref={(node) => {
                tabButtons.current[index] = node;
              }}
              role="tab"
              id={`inbox-tab-${index}`}
              aria-controls={`inbox-panel-${index}`}
              aria-selected={tab === value}
              tabIndex={tab === value ? 0 : -1}
              onClick={() => changeTab(value)}
              onKeyDown={(event) => {
                let next = index;
                if (event.key === "ArrowRight") next = (index + 1) % 3;
                else if (event.key === "ArrowLeft") next = (index + 2) % 3;
                else if (event.key === "Home") next = 0;
                else if (event.key === "End") next = 2;
                else return;
                event.preventDefault();
                changeTab(
                  (
                    ["Pending", "Answered automatically", "Overridden"] as const
                  )[next],
                );
                tabButtons.current[next]?.focus();
              }}
            >
              {tabNames[value]}
              <span>
                {pages[value].items.length}
                {pages[value].next_cursor ? "+" : ""}
              </span>
            </button>
          ),
        )}
      </div>
      <div className="inbox-feedback">
        <p role="status" aria-live="polite">
          {notice || (loading ? "Loading your decisions…" : "")}
        </p>
        {undo.length > 0 && (
          <button
            className="secondary-button"
            disabled={busy}
            onClick={() => void undoLast()}
          >
            Undo last answer
          </button>
        )}
      </div>
      {undo.length > 0 && (
        <p className="evidence-note">
          Undo last answer updates history. This app cannot retract sent
          messages. Ad changes have a separate undo in Activity.
        </p>
      )}
      {error && (
        <div className="error-message" role="alert">
          <p>{error}</p>
          <button
            className="secondary-button"
            disabled={busy}
            onClick={() => {
              setError("");
              void load();
            }}
          >
            Try again
          </button>
        </div>
      )}
      {(["Pending", "Answered automatically", "Overridden"] as const).map(
        (value, index) => (
          <div
            role="tabpanel"
            id={`inbox-panel-${index}`}
            aria-labelledby={`inbox-tab-${index}`}
            key={value}
            hidden={tab !== value}
            aria-busy={busy}
          >
            {tab === value && !loading && (
              <>
                {!pages[value].items.length && (
                  <div className="inbox-empty">
                    <h2>
                      {value === "Pending"
                        ? "All caught up."
                        : value === "Overridden"
                          ? "No corrections yet."
                          : "Nothing automatic yet."}
                    </h2>
                    <p>
                      {value === "Pending"
                        ? "Connect Meta in Connections and pick your accounts. New questions will appear here; your Autonomy settings decide what Ren can handle."
                        : value === "Overridden"
                          ? "Correct this on a decision to teach Ren your preference."
                          : "Review the suggestions in Needs you. Ren earns permission separately for Ads, Inbox and Account in Autonomy."}
                    </p>
                  </div>
                )}
                {sources.map((source) => {
                  const items = pages[value].items.filter(
                    (item) =>
                      (item.source_app ??
                        item.item?.source_app ??
                        "workspace") === source,
                  );
                  return items.length ? (
                    <section
                      className="inbox-source"
                      aria-label={`${names[source]} decisions`}
                      key={source}
                    >
                      <h2>
                        {names[source]}
                        <span>{items.length}</span>
                      </h2>
                      <div className="inbox-cards">{items.map(card)}</div>
                    </section>
                  ) : null;
                })}
                {pages[value].next_cursor && (
                  <button
                    className="secondary-button load-more"
                    disabled={busy}
                    onClick={() => void more()}
                  >
                    Load more decisions
                  </button>
                )}
              </>
            )}
          </div>
        ),
      )}
      <p className="inbox-footnote">
        Built for Meta’s platform by Secnd. Independently developed.
      </p>
    </section>
  );
}
