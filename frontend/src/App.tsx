import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import "./App.css";

type View = "Teach" | "Companion" | "Metrics";
type Connection = "connecting" | "online" | "offline";
type Preview = {
  filename: string;
  columns: string[];
  row_count: number;
  rows: Record<string, string>[];
};

const example = `description,amount,category
Train to client meeting,28.50,Travel
Monthly design software,24.00,Software
Team lunch,56.80,Meals
Office notebooks,12.00,Office supplies
Return train ticket,31.00,Travel
Cloud hosting,18.00,Software
Coffee with client,8.50,Meals
Printer paper,9.90,Office supplies
`;

function Icon({ kind }: { kind: View | "upload" | "arrow" }) {
  const paths = {
    Teach: (
      <>
        <path d="m3 9 9-5 9 5-9 5-9-5Z" />
        <path d="M7 12v5c3 2 7 2 10 0v-5M21 9v7" />
      </>
    ),
    Companion: (
      <>
        <path d="M20 11a8 8 0 0 1-8 8H4v-8a8 8 0 1 1 16 0Z" />
        <path d="M8 10h8M8 14h5" />
      </>
    ),
    Metrics: (
      <>
        <path d="M4 4v16h16M8 15v-4M13 15V7M18 15v-6" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4m-5 5 5-5 5 5M5 15v5h14v-5" />
      </>
    ),
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
  };
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[kind]}
    </svg>
  );
}

export default function App() {
  const [view, setView] = useState<View>("Teach");
  const [connection, setConnection] = useState<Connection>("connecting");
  const [retry, setRetry] = useState(0);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [target, setTarget] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 5000);
    fetch("/api/health", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("API unavailable");
        const health = await response.json();
        if (health.status !== "ok" || health.database !== "ready")
          throw new Error("Not ready");
        if (active) setConnection("online");
      })
      .catch(() => {
        if (active) setConnection("offline");
      })
      .finally(() => window.clearTimeout(timeout));
    return () => {
      active = false;
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [retry]);

  async function inspect(file: File) {
    setError("");
    setPreview(null);
    setTarget("");
    if (file.size > 5 * 1024 * 1024) {
      setError("Choose a CSV smaller than 5 MiB.");
      return;
    }
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch("/api/preview", {
        method: "POST",
        body: form,
        signal: AbortSignal.timeout(30000),
      });
      if (!response.ok) {
        const problem = await response.json().catch(() => null);
        throw new Error(
          typeof problem?.detail === "string"
            ? problem.detail
            : "Could not read this CSV. Please try again.",
        );
      }
      const result: Preview = await response.json();
      setPreview(result);
      setTarget(
        result.columns.find((column) =>
          /^(category|label|target)$/i.test(column),
        ) ??
          result.columns.at(-1) ??
          "",
      );
    } catch (cause) {
      setError(
        cause instanceof Error &&
          cause.name !== "TypeError" &&
          cause.name !== "TimeoutError"
          ? cause.message
          : "Could not reach Secnd. Check your connection and try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) void inspect(file);
    event.target.value = "";
  }

  const connectionText = {
    connecting: "Connecting…",
    online: "Workspace connected",
    offline: "Workspace unavailable",
  }[connection];

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="Secnd home">
          <span className="brand-symbol">s</span>secnd
          <span className="brand-period">.</span>
        </a>
        <div className="workspace-label">
          <span className="workspace-avatar">Y</span>
          <span>
            Your workspace<small>A little more headspace.</small>
          </span>
        </div>
        <nav aria-label="Workspace">
          {(["Teach", "Companion", "Metrics"] as const).map((item, index) => (
            <button
              key={item}
              className={`nav-item ${view === item ? "active" : ""}`}
              aria-current={view === item ? "page" : undefined}
              onClick={() => setView(item)}
            >
              <Icon kind={item} />
              <span>{item}</span>
              <span className="nav-number">0{index + 1}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-note">
          <span className="small-spark">✳</span>
          <p>
            Good judgment.
            <br />A little less repetition.
          </p>
        </div>
        <div className="workspace-footer">
          <span className="local-dot" />
          Local workspace
        </div>
      </aside>

      <div className="workspace-main">
        <header className="topbar">
          <span>
            YOUR WORKSPACE <span className="breadcrumb-slash">/</span>{" "}
            <strong>{view}</strong>
          </span>
          <div className={`connection ${connection}`} role="status">
            <span />
            {connectionText}
            {connection === "offline" && (
              <button
                onClick={() => {
                  setConnection("connecting");
                  setRetry((value) => value + 1);
                }}
              >
                Retry
              </button>
            )}
          </div>
        </header>
        <main id="main" tabIndex={-1}>
          {view === "Teach" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">A COMPANION THAT LEARNS YOUR CALLS</p>
                  <h1>
                    Teach it once.
                    <br />
                    <span>Make room for more.</span>
                  </h1>
                  <p className="intro">
                    The decisions you make every day, learned from your
                    examples.
                    <br className="desktop-break" /> A little help with the
                    routine. The final say stays yours.
                  </p>
                </div>
                <div className="orbit" aria-hidden="true">
                  <div className="orbit-inner">
                    <span>
                      your
                      <br />
                      judgment
                    </span>
                    <span className="orbit-plus">+</span>
                    <span>
                      less
                      <br />
                      repetition
                    </span>
                  </div>
                  <span className="orbit-dot one" />
                  <span className="orbit-dot two" />
                  <span className="orbit-dot three" />
                </div>
              </div>
              <section className="teach-card" aria-labelledby="upload-title">
                <div className="section-heading">
                  <div>
                    <p className="eyebrow">START WITH WHAT YOU KNOW</p>
                    <h2 id="upload-title">Show Secnd your decisions.</h2>
                  </div>
                  <span className="step-pill">Step 01</span>
                </div>
                <p className="section-description">
                  Bring a CSV of past transactions or invoices, including the
                  labels you chose.
                </p>
                <input
                  ref={inputRef}
                  className="visually-hidden"
                  type="file"
                  accept=".csv,text/csv"
                  aria-label="Upload labelled decisions CSV"
                  onChange={chooseFile}
                  disabled={busy}
                />
                <div className="upload-area">
                  <div className="upload-icon">
                    <Icon kind="upload" />
                  </div>
                  <h3>
                    {busy
                      ? "Reading your decisions…"
                      : "Your examples are the starting point."}
                  </h3>
                  <p>A description, an amount, and the category you picked.</p>
                  <button
                    className="primary-button"
                    disabled={busy}
                    onClick={() => inputRef.current?.click()}
                  >
                    {busy ? "Reading CSV…" : "Choose a CSV"}
                    <Icon kind="arrow" />
                  </button>
                  <small>UTF-8 CSV · up to 5 MiB · preview only</small>
                </div>
                <div className="example-row">
                  <span>Just looking around?</span>
                  <button
                    disabled={busy}
                    onClick={() =>
                      void inspect(
                        new File([example], "example-decisions.csv", {
                          type: "text/csv",
                        }),
                      )
                    }
                  >
                    Try an example CSV <span aria-hidden="true">↗</span>
                  </button>
                </div>
                {error && (
                  <p className="error-message" role="alert">
                    {error}
                  </p>
                )}
                {preview && (
                  <section
                    className="preview"
                    aria-labelledby="preview-title"
                    aria-live="polite"
                  >
                    <div className="preview-heading">
                      <div>
                        <h3 id="preview-title">{preview.filename}</h3>
                        <p>
                          {preview.row_count.toLocaleString()} rows ·{" "}
                          {preview.columns.length} columns · first{" "}
                          {preview.rows.length} shown
                        </p>
                      </div>
                      <label>
                        The decision to learn
                        <select
                          value={target}
                          onChange={(event) => setTarget(event.target.value)}
                        >
                          {preview.columns.map((column) => (
                            <option key={column}>{column}</option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <div className="table-scroll">
                      <table>
                        <caption className="visually-hidden">
                          Preview of your uploaded decisions
                        </caption>
                        <thead>
                          <tr>
                            {preview.columns.map((column) => (
                              <th
                                key={column}
                                scope="col"
                                className={
                                  target === column ? "target-column" : ""
                                }
                              >
                                {column}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {preview.rows.map((row, index) => (
                            <tr key={index}>
                              {preview.columns.map((column) => (
                                <td
                                  key={column}
                                  className={
                                    target === column ? "target-column" : ""
                                  }
                                >
                                  {row[column]}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="preview-note">
                      Your preview is ready. Training is the next step and isn’t
                      available yet. This file hasn’t been saved.
                    </p>
                  </section>
                )}
              </section>
              <section
                className="how-it-works"
                aria-label="How Secnd will learn"
              >
                <article>
                  <span>01 / TEACH</span>
                  <h3>A few examples from you.</h3>
                  <p>Your past decisions give Secnd a place to start.</p>
                </article>
                <article>
                  <span>02 / DECIDE</span>
                  <h3>Confident enough? Handled.</h3>
                  <p>
                    Calibrated confidence will decide when to act and when to
                    ask.
                  </p>
                </article>
                <article>
                  <span>03 / LEARN</span>
                  <h3>Your answers make it better.</h3>
                  <p>Each answer will become another example to learn from.</p>
                </article>
              </section>
            </>
          )}

          {view === "Companion" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">A LITTLE HELP WITH THE EVERYDAY</p>
                  <h1>
                    Your decisions,
                    <br />
                    <span>with a second pair of eyes.</span>
                  </h1>
                  <p className="intro">
                    Automatic decisions and questions for you will live here.
                  </p>
                </div>
              </div>
              <section className="empty-card">
                <div className="empty-icon">
                  <Icon kind="Companion" />
                </div>
                <h2>Every good companion starts by listening.</h2>
                <p>
                  Once your first model is trained, Secnd will handle confident
                  decisions and bring the uncertain ones to you.
                </p>
                <button
                  className="primary-button"
                  onClick={() => setView("Teach")}
                >
                  Start with your examples
                  <Icon kind="arrow" />
                </button>
              </section>
            </>
          )}

          {view === "Metrics" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">TRUST, WITH THE NUMBERS TO MATCH</p>
                  <h1>
                    Watch your
                    <br />
                    <span>headspace grow.</span>
                  </h1>
                  <p className="intro">
                    See what Secnd can handle, how often it’s right, and what it
                    learns.
                  </p>
                </div>
              </div>
              <div className="metric-grid">
                {[
                  {
                    label: "Coverage",
                    note: "Decisions handled automatically",
                  },
                  { label: "Precision", note: "Correct automatic decisions" },
                  {
                    label: "Calibration error",
                    note: "How well confidence matches outcomes",
                  },
                ].map((metric) => (
                  <section className="metric-card" key={metric.label}>
                    <h2>{metric.label}</h2>
                    <div className="metric-value" aria-label="Not measured">
                      —
                    </div>
                    <p>{metric.note}</p>
                  </section>
                ))}
              </div>
              <section className="empty-card">
                <div className="empty-icon">
                  <Icon kind="Metrics" />
                </div>
                <h2>Real progress starts with a first model.</h2>
                <p>
                  Training and evaluation will unlock your learning curve and
                  the Muse Spark comparison. No measurements yet.
                </p>
                <button
                  className="primary-button"
                  onClick={() => setView("Teach")}
                >
                  Explore your examples
                  <Icon kind="arrow" />
                </button>
              </section>
            </>
          )}
          <footer className="page-footer">
            <span>Built to learn your judgment.</span>
            <span>Secnd / early preview</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
