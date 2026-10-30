import { useEffect, useRef, useState } from "react";
import { api, percent } from "./api";
import type { ColdStartState, Decision, InventedExample } from "./api";
import "./ColdStart.css";

function ExampleRow({
  row,
  labels,
  onSave,
  onDelete,
}: {
  row: InventedExample;
  labels: string[];
  onSave: (row: InventedExample) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}) {
  const [text, setText] = useState(row.text);
  const [label, setLabel] = useState(row.label);
  const [busy, setBusy] = useState(false);
  async function apply(action: () => Promise<void>) {
    setBusy(true);
    try {
      await action();
    } finally {
      setBusy(false);
    }
  }
  return (
    <li className="invented-row">
      <label className="sr-label">
        Example message
        <textarea
          aria-label={`Example for ${row.label}`}
          value={text}
          maxLength={500}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
        />
      </label>
      <div className="invented-row-actions">
        <select
          aria-label="Example label"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          disabled={busy}
        >
          {labels.map((x) => (
            <option key={x}>{x}</option>
          ))}
        </select>
        <button
          disabled={
            busy || !text.trim() || (text === row.text && label === row.label)
          }
          onClick={() =>
            void apply(() => onSave({ ...row, text: text.trim(), label }))
          }
        >
          Save
        </button>
        <button
          className="delete-example"
          disabled={busy}
          onClick={() => void apply(() => onDelete(row.id))}
        >
          Delete
        </button>
      </div>
    </li>
  );
}

export default function ColdStart({ onChange }: { onChange: () => void }) {
  const [labels, setLabels] = useState("");
  const [state, setState] = useState<ColdStartState | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [page, setPage] = useState(0);
  const [message, setMessage] = useState("");
  const [decision, setDecision] = useState<Decision | null>(null);
  const [answer, setAnswer] = useState("");
  const [saved, setSaved] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const generating =
    state?.status === "queued" || state?.status === "generating";

  useEffect(() => {
    let active = true;
    api
      .coldState()
      .then(({ data }) => {
        if (active) {
          setState(data);
          setLabels(data.labels.join("\n"));
        }
      })
      .catch(() => {
        /* A new workspace has no saved invented history. */
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    if (!generating) return;
    let active = true;
    const timer = window.setInterval(() => {
      api
        .coldState()
        .then(({ data }) => {
          if (active) setState(data);
        })
        .catch((cause) => {
          if (active) setError(cause.message);
        });
    }, 1500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [generating]);

  async function generate(upload?: File) {
    setBusy(true);
    setError("");
    try {
      const result = upload
        ? await api.importLabels(upload)
        : await api.coldStart(
            labels
              .split(/\r?\n/)
              .map((x) => x.trim())
              .filter(Boolean),
          );
      setState(result.data);
      setLabels(result.data.labels.join("\n"));
      onChange();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not start examples.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function update(action: () => Promise<{ data: ColdStartState }>) {
    setError("");
    try {
      setState((await action()).data);
      onChange();
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not update this example.",
      );
    }
  }
  async function predict() {
    setBusy(true);
    setError("");
    setSaved(false);
    setDecision(null);
    try {
      const { data } = await api.predict(message, crypto.randomUUID());
      setDecision(data[0]);
      setAnswer(data[0].label);
      onChange();
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Ren could not read this message.",
      );
    } finally {
      setBusy(false);
    }
  }
  async function confirm() {
    if (!decision) return;
    setBusy(true);
    setError("");
    try {
      await api.answer(decision.item_id, answer, `cold-${decision.item_id}`);
      setSaved(true);
      onChange();
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not save your answer.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="teach-card cold-start" aria-labelledby="cold-title">
      <div className="section-heading">
        <div>
          <p className="eyebrow">NO LABELLED HISTORY YET?</p>
          <h2 id="cold-title">Start with only label names.</h2>
        </div>
        <span className="step-pill">Meta × Ren</span>
      </div>
      <p className="section-description">
        Muse Wonder writes starting examples through the Meta API. Review and
        edit that history. Ren reads it to make each suggestion.
      </p>
      {state?.demo && (
        <p className="cold-notice">Demo examples · no provider charge</p>
      )}
      <div className="label-entry">
        <label htmlFor="label-names">Your labels, one per line</label>
        <textarea
          id="label-names"
          rows={4}
          placeholder={"Travel\nSoftware\nMeals"}
          value={labels}
          onChange={(e) => setLabels(e.target.value)}
          disabled={busy || generating || !!state}
        />
        <div className="cold-controls">
          <button
            className="primary-button"
            disabled={busy || generating || !labels.trim()}
            onClick={() => void generate()}
          >
            {generating
              ? "Muse is writing examples…"
              : state
                ? "Use saved examples / resume"
                : "Generate examples · up to $1"}
          </button>
          {!state && (
            <button
              className="secondary-button"
              disabled={busy}
              onClick={() => file.current?.click()}
            >
              Import a label list
            </button>
          )}
          <input
            ref={file}
            type="file"
            className="visually-hidden"
            accept=".csv,.txt,text/csv,text/plain"
            aria-label="Import label list"
            disabled={busy || generating || !!state}
            onChange={(e) => {
              const picked = e.target.files?.[0];
              if (picked) void generate(picked);
              e.target.value = "";
            }}
          />
        </div>
        <small>
          10 examples requested per label · $1 workspace limit · saved examples
          are reused.
        </small>
      </div>
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      {state && (
        <>
          <div className="cold-progress" role="status">
            <strong>{state.history.length} starting examples</strong>
            <span>
              ${Number(state.spent_usd).toFixed(2)} of $
              {Number(state.cap_usd).toFixed(2)} budget
            </span>
            <span>
              {generating
                ? "Writing and saving history…"
                : state.status === "ready"
                  ? "History ready to review"
                  : "Generation paused"}
            </span>
          </div>
          {state.error && <p className="cold-notice">{state.error}</p>}
          {state.history.length > 0 && (
            <>
              <p className="history-intro">
                Invented starting history. Keep what fits your labels; save
                edits or delete examples that do not.
              </p>
              <ul className="invented-history">
                {state.history.slice(page * 10, (page + 1) * 10).map((row) => (
                  <ExampleRow
                    key={`${row.id}-${row.text}-${row.label}`}
                    row={row}
                    labels={state.labels}
                    onSave={(next) => update(() => api.editExample(next))}
                    onDelete={(id) => update(() => api.deleteExample(id))}
                  />
                ))}
              </ul>
              <div className="history-pages">
                <button disabled={page === 0} onClick={() => setPage(page - 1)}>
                  Previous
                </button>
                <span>
                  Page {page + 1} of{" "}
                  {Math.max(1, Math.ceil(state.history.length / 10))}
                </span>
                <button
                  disabled={(page + 1) * 10 >= state.history.length}
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </button>
              </div>
            </>
          )}
          <div className="cold-predict">
            <h3>Try a message with Ren</h3>
            <p>
              A person confirms each suggestion. This workspace has no
              independent calibration labels.
            </p>
            <label htmlFor="cold-message">Customer message</label>
            <textarea
              id="cold-message"
              rows={2}
              value={message}
              maxLength={10000}
              onChange={(e) => setMessage(e.target.value)}
            />
            <button
              className="primary-button"
              disabled={busy || !message.trim()}
              onClick={() => void predict()}
            >
              Ask Ren
            </button>
            {decision && (
              <div className="cold-decision" aria-live="polite">
                <strong>Ren suggests {decision.label}</strong>
                <span>
                  Raw score {percent(decision.prob)} · human review required
                </span>
                {decision.reasons.neighbours.length > 0 && (
                  <details>
                    <summary>History Ren read</summary>
                    <ul>
                      {decision.reasons.neighbours.map((n) => (
                        <li key={n.item_id}>
                          {n.text} <strong>{n.label}</strong>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
                <label htmlFor="cold-answer">Your answer</label>
                <select
                  id="cold-answer"
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  disabled={busy || saved}
                >
                  {state.labels.map((x) => (
                    <option key={x}>{x}</option>
                  ))}
                </select>
                <button disabled={busy || saved} onClick={() => void confirm()}>
                  {saved ? "Added to Ren’s history" : "Confirm or correct"}
                </button>
              </div>
            )}
          </div>
        </>
      )}
      <p className="cold-reference">
        Recorded MTOP benchmark: invented history with 32 nearest examples gave
        84.95% accuracy and 25.6% answered automatically at the 99% gate, about
        equal to 200 human history labels. This is a published benchmark,
        separate from your workspace.
      </p>
    </section>
  );
}
