import { useCallback, useEffect, useRef, useState } from "react";
import { api, percent, workspaceId } from "./api";
import type { Decision, Page } from "./api";
import "./Companion.css";

type Tab = "Pending" | "Answered automatically" | "Overridden";
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
  workspace: "Workspace",
};
const measured = (value: number | null | undefined) =>
  value == null ? "Unavailable" : `${Math.floor(value * 10000) / 100}%`;

export default function Companion({ onAnswer }: { onAnswer: () => void }) {
  const [tab, setTab] = useState<Tab>("Pending");
  const [pages, setPages] = useState<Record<Tab, Page<Decision>>>({
    Pending: empty,
    "Answered automatically": empty,
    Overridden: empty,
  });
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
          : `Saved ${label} to Ren’s history.`,
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
      heading.current?.focus();
    }
  }
  async function undoLast() {
    const last = undo.at(-1);
    if (!last || mutating.current) return;
    mutating.current = true;
    revision.current++;
    setBusy(true);
    setError("");
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
      heading.current?.focus();
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
    setOther(null);
  }
  function card(item: Decision) {
    const source = item.source_app ?? item.item?.source_app ?? "workspace";
    const choices = labels.length ? labels : Object.keys(item.probabilities);
    return (
      <article
        className="inbox-card"
        key={item.item_id}
        aria-label={item.item?.text || item.item_id}
      >
        <div className="decision-meta">
          <span className={`source-badge source-${source}`}>
            {names[source]}
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
        {tab === "Answered automatically" && (
          <dl className="certification">
            <div>
              <dt>Certified threshold</dt>
              <dd>{measured(item.threshold)}</dd>
            </div>
            <div>
              <dt>Precision lower bound</dt>
              <dd>{measured(item.cp_lower_bound)}</dd>
            </div>
          </dl>
        )}
        {tab === "Answered automatically" && (
          <p className="evidence-note">
            One-sided 95% calibration bound; target 99%. Applies to this saved
            model, not a guarantee for every new message.
          </p>
        )}
        <details className="inbox-evidence">
          <summary>Evidence · nearest past decisions</summary>
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
            Similar examples provide context. Ren makes the decision; Muse may
            write starting history.
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
            Pick another label
          </button>
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
              Choose label
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
              Every label in this workspace is available.
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
          <p>Ren handles certified decisions. You make the rest.</p>
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
      <form className="workspace-picker" method="get">
        <label htmlFor="workspace-id">Workspace</label>
        <input
          id="workspace-id"
          name="workspace"
          defaultValue={workspaceId}
          required
          pattern="[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}"
          maxLength={64}
        />
        <button className="secondary-button">Open workspace</button>
      </form>
      {fixture && (
        <p className="preview-banner">
          Preview · illustrative decisions. Answers are not saved. Teach this
          workspace to start.
        </p>
      )}
      {demo && (
        <p className="preview-banner">
          Synthetic demo · illustrative messages, local answers only.
          Certification below uses synthetic calibration data, not Meta traffic.
        </p>
      )}
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
              {value}
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
                        ? "New questions from your connected apps will appear here."
                        : value === "Overridden"
                          ? "Pick another label on a decision to teach Ren your preference."
                          : "Until the gate certifies enough evidence, decisions wait for you."}
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
