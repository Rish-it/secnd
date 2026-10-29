import { useCallback, useEffect, useRef, useState } from "react";
import { api, percent } from "./api";
import type { Decision, Page } from "./api";
import "./Companion.css";

type Props = {
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
  onAnswer: () => void;
};
const empty: Page<Decision> = { items: [], next_cursor: null };

export default function Companion({ open, onOpen, onClose, onAnswer }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [tab, setTab] = useState<"Questions" | "Handled" | "Ask Muse">(
    "Questions",
  );
  const [queue, setQueue] = useState(empty);
  const [feed, setFeed] = useState(empty);
  const [fixture, setFixture] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [other, setOther] = useState<string | null>(null);
  const [otherLabel, setOtherLabel] = useState("");
  const [why, setWhy] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<{ role: string; text: string }[]>(
    [],
  );
  const keys = useRef(new Map<string, string>());

  const load = useCallback(async () => {
    try {
      const [questions, handled] = await Promise.all([api.queue(), api.feed()]);
      setError("");
      setQueue(questions.data);
      setFeed(handled.data);
      setFixture(questions.fixture || handled.fixture);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not load your decisions.",
      );
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);
  useEffect(() => {
    const node = dialog.current;
    if (open && !node?.open) node?.showModal();
    else if (!open && node?.open) node.close();
  }, [open]);

  async function answer(item: Decision, label: string) {
    if (busy) return;
    setBusy(item.item_id);
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
          ? "Example answer checked. Your answer has not been saved."
          : "Answer saved. Ren is checking the next model version.",
      );
      setOther(null);
      setOtherLabel("");
      if (!result.fixture) {
        onAnswer();
        await load();
      }
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not save this answer. Try again.",
      );
    } finally {
      setBusy(null);
    }
  }
  async function chat(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    const text = question.trim();
    setBusy("chat");
    setError("");
    try {
      const reply = await api.chat(text);
      setMessages((current) => [
        ...current,
        { role: "You", text },
        {
          role: reply.fixture ? "Muse preview" : "Muse",
          text: reply.data.answer,
        },
      ]);
      setQuestion("");
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Muse could not answer. Try again.",
      );
    } finally {
      setBusy(null);
    }
  }
  function choices(item: Decision) {
    return Object.entries(item.probabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
  }
  function otherForm(item: Decision) {
    return (
      other === item.item_id && (
        <form
          className="other-label"
          onSubmit={(event) => {
            event.preventDefault();
            if (otherLabel.trim()) void answer(item, otherLabel.trim());
          }}
        >
          <label>
            Correct category
            <input
              autoFocus
              required
              maxLength={128}
              value={otherLabel}
              onChange={(event) => setOtherLabel(event.target.value)}
            />
          </label>
          <button className="primary-button" disabled={busy !== null}>
            Save answer
          </button>
        </form>
      )
    );
  }

  return (
    <>
      <button
        className={`companion-dot ${queue.items.length ? "has-questions" : ""}`}
        onClick={onOpen}
        aria-haspopup="dialog"
        aria-controls="companion-panel"
        aria-expanded={open}
      >
        <span className="orb" aria-hidden="true">
          s
        </span>
        <span>
          <strong>Your second pair of eyes</strong>
          <small>
            {loading
              ? "Checking in…"
              : error
                ? "Connection needs attention"
                : `${fixture ? 0 : feed.items.length}${!fixture && feed.next_cursor ? "+" : ""} handled · ${queue.items.length}${queue.next_cursor ? "+" : ""} ${fixture ? "sample " : ""}${queue.items.length === 1 ? "question" : "questions"}`}
          </small>
        </span>
      </button>
      <dialog
        id="companion-panel"
        className="companion-panel"
        ref={dialog}
        onCancel={(event) => {
          event.preventDefault();
          onClose();
        }}
        onClose={onClose}
        aria-labelledby="companion-title"
      >
        <header className="panel-heading">
          <div>
            <p className="eyebrow">A LITTLE MORE HEADSPACE</p>
            <h2 id="companion-title">Your companion.</h2>
          </div>
          <button
            className="close-panel"
            onClick={onClose}
            aria-label="Close companion"
          >
            ×
          </button>
        </header>
        {fixture && (
          <p className="preview-banner">
            Preview workspace · These are sample decisions. Answers are not
            saved yet.
          </p>
        )}
        <nav className="panel-tabs" aria-label="Companion sections">
          {(["Questions", "Handled", "Ask Muse"] as const).map((value) => (
            <button
              key={value}
              aria-pressed={tab === value}
              onClick={() => setTab(value)}
            >
              {value}
              {value === "Questions" && queue.items.length
                ? ` (${queue.items.length})`
                : ""}
            </button>
          ))}
        </nav>
        <div className="panel-content">
          {loading && <p role="status">Gathering your decisions…</p>}
          {error && (
            <div className="error-message" role="alert">
              <p>{error}</p>
              <button
                onClick={() => {
                  setLoading(true);
                  void load();
                }}
              >
                Try again
              </button>
            </div>
          )}
          {notice && (
            <p className="answer-notice" role="status">
              {notice}
            </p>
          )}
          {!loading && tab === "Questions" && (
            <>
              <p className="panel-intro">
                A small question. A better next decision.
              </p>
              {!queue.items.length && (
                <div className="panel-empty">
                  <h3>All caught up.</h3>
                  <p>
                    When Ren needs your judgment, the question will appear here.
                  </p>
                </div>
              )}
              {queue.items.map((item) => (
                <article
                  className="question-card"
                  key={item.item_id}
                  aria-busy={busy === item.item_id}
                >
                  <div className="decision-meta">
                    <span>
                      {item.spot_check
                        ? "Spot-check · help verify an automatic answer"
                        : fixture
                          ? "Illustrative question"
                          : "A question for you"}
                    </span>
                    <span>{percent(item.prob)} sure</span>
                  </div>
                  <h3>{item.item?.text || "Decision without a description"}</h3>
                  {item.item?.amount != null && (
                    <p className="decision-amount">
                      Amount: {item.item.amount.toLocaleString()}{" "}
                      {item.item.date ? `· ${item.item.date}` : ""}
                    </p>
                  )}
                  <p className="question-prompt">
                    Which category would you choose?
                  </p>
                  <div className="answer-options">
                    {choices(item).map(([label, probability]) => (
                      <button
                        key={label}
                        disabled={busy !== null}
                        onClick={() => void answer(item, label)}
                      >
                        <span>{label}</span>
                        <small>{percent(probability)}</small>
                      </button>
                    ))}
                    <button
                      disabled={busy !== null}
                      onClick={() => {
                        setOther(item.item_id);
                        setOtherLabel("");
                      }}
                    >
                      Other category
                    </button>
                  </div>
                  {otherForm(item)}
                  <small className="version-caption">
                    {fixture ? "Sample model" : item.model_version}
                  </small>
                </article>
              ))}
              {queue.next_cursor && (
                <button
                  className="text-button"
                  disabled={loading}
                  onClick={async () => {
                    setLoading(true);
                    try {
                      const next = await api.queue(queue.next_cursor!);
                      setQueue((previous) => ({
                        ...next.data,
                        items: [...previous.items, ...next.data.items],
                      }));
                    } catch (cause) {
                      setError(
                        cause instanceof Error
                          ? cause.message
                          : "Could not load more questions.",
                      );
                    } finally {
                      setLoading(false);
                    }
                  }}
                >
                  More questions
                </button>
              )}
            </>
          )}
          {!loading && tab === "Handled" && (
            <>
              <p className="panel-intro">
                Every automatic decision, with its reasons.
              </p>
              {!feed.items.length && (
                <div className="panel-empty">
                  <h3>Answers nothing yet.</h3>
                  <p>
                    Until there is enough evidence for the precision target,
                    decisions stay with you.
                  </p>
                </div>
              )}
              {feed.items.map((item) => (
                <article className="question-card" key={item.item_id}>
                  <div className="decision-meta">
                    <span>
                      {fixture ? "Illustrative decision" : "Handled by Ren"}
                    </span>
                    <span>{percent(item.prob)}</span>
                  </div>
                  <h3>{item.item?.text || item.item_id}</h3>
                  <p className="handled-label">{item.label}</p>
                  <button
                    className="text-button"
                    aria-expanded={why === item.item_id}
                    onClick={() =>
                      setWhy(why === item.item_id ? null : item.item_id)
                    }
                  >
                    Why this decision? <span aria-hidden="true">↗</span>
                  </button>
                  {why === item.item_id && (
                    <div className="decision-evidence">
                      <p>
                        {fixture
                          ? "Sample probabilities"
                          : "Calibrated probabilities"}
                      </p>
                      {choices(item).map(([label, probability]) => (
                        <p key={label}>
                          {label}
                          <strong>{percent(probability)}</strong>
                        </p>
                      ))}
                      <p>
                        Threshold<strong>{percent(item.threshold)}</strong>
                      </p>
                      <small>
                        {fixture ? "Sample model" : item.model_version}
                      </small>
                      <h4>Nearest past decisions</h4>
                      {item.reasons.neighbours.length ? (
                        item.reasons.neighbours.map((row) => (
                          <p key={row.item_id}>
                            {row.text} · {row.label} · {percent(row.similarity)}{" "}
                            similarity
                          </p>
                        ))
                      ) : (
                        <p>No past-decision evidence available.</p>
                      )}
                      {item.reasons.merchant_memory && (
                        <p>
                          A previous merchant decision contributed to the
                          prediction.
                        </p>
                      )}
                    </div>
                  )}
                  <div className="correction-options">
                    <span>Needs a correction?</span>
                    {choices(item)
                      .filter(([label]) => label !== item.label)
                      .map(([label]) => (
                        <button
                          key={label}
                          disabled={busy !== null}
                          onClick={() => void answer(item, label)}
                        >
                          Change to {label}
                        </button>
                      ))}
                    <button
                      disabled={busy !== null}
                      onClick={() => {
                        setOther(item.item_id);
                        setOtherLabel("");
                      }}
                    >
                      Choose another
                    </button>
                  </div>
                  {otherForm(item)}
                </article>
              ))}
              {feed.next_cursor && (
                <button
                  className="text-button"
                  onClick={async () => {
                    try {
                      const next = await api.feed(feed.next_cursor!);
                      setFeed((previous) => ({
                        ...next.data,
                        items: [...previous.items, ...next.data.items],
                      }));
                    } catch (cause) {
                      setError(
                        cause instanceof Error
                          ? cause.message
                          : "Could not load more decisions.",
                      );
                    }
                  }}
                >
                  More decisions
                </button>
              )}
            </>
          )}
          {tab === "Ask Muse" && (
            <section className="muse-chat">
              <p className="panel-intro">
                Ask Muse about your decisions, counts, or corrections.
              </p>
              {!messages.length && (
                <div className="panel-empty">
                  <span className="muse-mark">✳</span>
                  <h3>Put your history into words.</h3>
                  <p>
                    {fixture
                      ? "Muse is not connected in this preview. Try a question to see the conversation."
                      : "Muse reads your decision history when it needs evidence. The decision model still makes your predictions."}
                  </p>
                </div>
              )}
              <div className="chat-messages" aria-live="polite">
                {messages.map((message, index) => (
                  <article
                    className={`chat-message ${message.role === "You" ? "from-you" : ""}`}
                    key={index}
                  >
                    <small>{message.role}</small>
                    <p>{message.text}</p>
                  </article>
                ))}
              </div>
              <form
                className="chat-form"
                onSubmit={(event) => void chat(event)}
              >
                <label className="visually-hidden" htmlFor="muse-question">
                  Ask Muse
                </label>
                <textarea
                  id="muse-question"
                  placeholder="How many decisions needed a correction?"
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  maxLength={4000}
                  required
                  rows={3}
                />
                <button
                  className="primary-button"
                  disabled={busy !== null || !question.trim()}
                >
                  {busy === "chat" ? "Muse is thinking…" : "Ask Muse ↗"}
                </button>
              </form>
            </section>
          )}
        </div>
      </dialog>
    </>
  );
}
