import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { workspaceId } from "./api";
import "./Harness.css";

const root = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}`;
async function harnessRequest<T>(
  path: string,
  method = "GET",
  body?: unknown,
): Promise<T> {
  const response = await fetch(path, {
    method,
    credentials: "same-origin",
    signal: AbortSignal.timeout(30000),
    ...(body === undefined
      ? {}
      : {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(data.message ?? "Could not save. Try again.");
  return data as T;
}
const errorMessage = (cause: unknown) =>
  cause instanceof Error ? cause.message : "Could not connect. Try again.";
const measured = (value: number) => `${Math.floor(value * 10000) / 100}%`;

type Session = { signed_in: boolean; demo: boolean };
export function SessionGate({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [workspaces, setWorkspaces] = useState<string[]>([]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [create, setCreate] = useState(false);
  const [workspace, setWorkspace] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const load = useCallback(async () => {
    const result = await harnessRequest<Session>("/api/v1/auth/session");
    if (result.signed_in || result.demo) {
      const list = await harnessRequest<{ items: string[] }>(
        "/api/v1/workspaces",
      );
      setWorkspaces(list.items);
    }
    setSession(result);
  }, []);
  useEffect(() => {
    void load().catch((cause) => setError(errorMessage(cause)));
  }, [load]);
  if (!session && !error)
    return (
      <main className="sign-in">
        <p role="status">Opening Secnd…</p>
      </main>
    );
  if (!session || (!session.signed_in && !session.demo))
    return (
      <main className="sign-in">
        <a className="brand" href="/">
          secnd.
        </a>
        <p className="eyebrow">A LITTLE MORE HEADSPACE</p>
        <h1>
          Your judgment.
          <br />
          Less repetition.
        </h1>
        <p>
          Ren watches your Meta ads and inboxes, learns from your decisions, and
          asks until it has earned permission to act.
        </p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            setBusy(true);
            setError("");
            void harnessRequest(
              "/api/v1/auth/" + (create ? "register" : "login"),
              "POST",
              { username, password },
            )
              .then(() => {
                setPassword("");
                return load();
              })
              .catch((cause) => setError(errorMessage(cause)))
              .finally(() => setBusy(false));
          }}
        >
          <h2>{create ? "Create your account" : "Sign in"}</h2>
          <label>
            Sign-in name
            <input
              autoComplete="username"
              required
              minLength={3}
              maxLength={128}
              value={username}
              onChange={(event) => setUsername(event.target.value)}
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete={create ? "new-password" : "current-password"}
              minLength={12}
              maxLength={256}
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>
          <button className="primary-button" disabled={busy}>
            {busy ? "Opening…" : create ? "Create account" : "Sign in"}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setCreate((value) => !value)}
          >
            {create ? "I have an account" : "Create an account"}
          </button>
        </form>
        {error && (
          <p role="alert" className="error-message">
            {error}
          </p>
        )}
        <p className="small-print">
          Built for Meta’s platform by Secnd. Independently developed; not an
          official Meta product.
        </p>
      </main>
    );
  if (session.signed_in && !workspaces.includes(workspaceId))
    return (
      <main className="sign-in">
        <a className="brand" href="/">
          secnd.
        </a>
        <h1>Your workspaces</h1>
        {workspaces.map((ws) => (
          <a
            className="workspace-link"
            key={ws}
            href={`/?workspace=${encodeURIComponent(ws)}`}
          >
            Open {ws}
          </a>
        ))}
        <form
          onSubmit={(event) => {
            event.preventDefault();
            setBusy(true);
            setError("");
            void harnessRequest("/api/v1/workspaces", "POST", {
              workspace_id: workspace,
            })
              .then(() => {
                location.href = `/?workspace=${encodeURIComponent(workspace)}`;
              })
              .catch((cause) => setError(errorMessage(cause)))
              .finally(() => setBusy(false));
          }}
        >
          <label>
            New workspace name
            <input
              required
              pattern="[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}"
              maxLength={64}
              value={workspace}
              onChange={(event) => setWorkspace(event.target.value)}
            />
          </label>
          <button className="primary-button" disabled={busy}>
            Create workspace
          </button>
        </form>
        {error && <p role="alert">{error}</p>}
        <button
          className="secondary-button"
          onClick={() =>
            void harnessRequest("/api/v1/auth/logout", "POST").then(() =>
              location.reload(),
            )
          }
        >
          Sign out
        </button>
      </main>
    );
  return (
    <>
      {session.demo && (
        <div className="synthetic-ribbon" role="status">
          Synthetic demo · fictional workspaces and decisions · no live Meta
          actions
        </div>
      )}
      <div className="session-tools">
        <span>{session.demo ? "Demo workspace" : "Signed in"}</span>
        {session.signed_in && (
          <button
            onClick={() =>
              void harnessRequest("/api/v1/auth/logout", "POST")
                .then(() => location.reload())
                .catch((cause) => setError(errorMessage(cause)))
            }
          >
            Sign out
          </button>
        )}
        {error && <span role="alert">{error}</span>}
      </div>
      {children}
    </>
  );
}

type Asset = { key: string; kind: string; id: string; name: string };
type Connection = {
  connected: boolean;
  assets: Asset[];
  selected: string[];
  access_note: string;
};
export function MetaLogin() {
  const [connection, setConnection] = useState<Connection | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState(() =>
    new URLSearchParams(location.search).has("meta_error")
      ? "Meta refused the connection or sign-in expired. Check your app role and try connecting again."
      : "",
  );
  useEffect(() => {
    void harnessRequest<Session>("/api/v1/auth/session")
      .then((session) => {
        if (!session.signed_in) return;
        return harnessRequest<Connection>(
          `${root}/integrations/meta/connection`,
        ).then((result) => {
          setConnection(result);
          setSelected(result.selected);
        });
      })
      .catch((cause) => setError(errorMessage(cause)));
  }, []);
  return (
    <div className="meta-login">
      <h3>
        {connection?.connected
          ? "Choose the accounts Ren can watch"
          : "Connect Meta"}
      </h3>
      <p>
        Connect through Facebook Login for Business. Your access stays on the
        server. Only sandbox ad accounts can be selected.
      </p>
      <p className="access-note">
        Until the Meta app has Advanced Access, sign-in is limited to people
        with an admin, developer or tester role on the app. If Meta refuses
        access, ask the app owner to add your role.
      </p>
      <a
        className="primary-button"
        href={`${root}/integrations/meta/oauth/start`}
      >
        {connection?.connected ? "Reconnect Meta" : "Continue with Facebook"}
      </a>
      {connection?.connected && (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            setBusy(true);
            setError("");
            void harnessRequest<Connection>(
              `${root}/integrations/meta/accounts`,
              "PUT",
              { assets: selected },
            )
              .then((result) => {
                setConnection(result);
                setNotice(
                  "Accounts saved. Ren uses only your selections for this workspace.",
                );
              })
              .catch((cause) => setError(errorMessage(cause)))
              .finally(() => setBusy(false));
          }}
        >
          <fieldset disabled={busy}>
            <legend>Accounts for {workspaceId}</legend>
            {connection.assets.map((asset) => (
              <label className="check-row" key={asset.key}>
                <input
                  type="checkbox"
                  checked={selected.includes(asset.key)}
                  onChange={(event) =>
                    setSelected((old) =>
                      event.target.checked
                        ? [...old, asset.key]
                        : old.filter((key) => key !== asset.key),
                    )
                  }
                />
                <span>
                  <strong>{asset.name}</strong>
                  <small>
                    {asset.kind === "messenger" ? "Facebook Page" : asset.kind}{" "}
                    · {asset.id}
                  </small>
                </span>
              </label>
            ))}
            {!connection.assets.length && (
              <p>
                No eligible assets were returned. Check this person’s Meta
                permissions.
              </p>
            )}
          </fieldset>
          <button className="primary-button" disabled={busy}>
            Use selected accounts
          </button>
        </form>
      )}
      {connection?.connected && (
        <button
          className="secondary-button"
          disabled={busy}
          onClick={() => {
            setBusy(true);
            void harnessRequest("/api/v1/meta/disconnect", "POST")
              .then(() => {
                setConnection(null);
                setSelected([]);
                setNotice(
                  "Disconnected. Stored access and workspace mappings have been deleted.",
                );
              })
              .catch((cause) => setError(errorMessage(cause)))
              .finally(() => setBusy(false));
          }}
        >
          Disconnect Meta from all my workspaces
        </button>
      )}
      {notice && <p role="status">{notice}</p>}
      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}
    </div>
  );
}

type Policy = {
  decision_type: "ad" | "inbox" | "account";
  level: "watch" | "ask" | "act";
  effective_level: string;
  bar: 90 | 95 | 98 | 99;
  confidence_cut: number;
  allow_raise_budget: boolean;
  allow_customer_messages: boolean;
  reviewed: number;
  confirmed: number;
  corrected: number;
  lower_bound: number;
  unlocked: boolean;
  confirmations_left: number;
  window: number;
};
type Counts = {
  decision_type: string;
  proposals: number;
  confirmed: number;
  corrected: number;
  workspaces: number;
};
const decisionNames = { ad: "Ads", inbox: "Inbox", account: "Account" };
export function AutonomyScreen() {
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [counts, setCounts] = useState<Counts[]>([]);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    void Promise.all([
      harnessRequest<Policy[]>(`${root}/autonomy`).then(setPolicies),
      harnessRequest<{ counts: Counts[] }>("/api/v1/track-record").then(
        (result) => setCounts(result.counts),
      ),
    ]).catch((cause) => setError(errorMessage(cause)));
  }, []);
  async function save(policy: Policy, patch: Partial<Policy>) {
    if (busy) return;
    setBusy(true);
    setError("");
    const updated = { ...policy, ...patch };
    const {
      level,
      bar,
      confidence_cut,
      allow_raise_budget,
      allow_customer_messages,
    } = updated;
    try {
      const result = await harnessRequest<Policy>(
        `${root}/autonomy/${policy.decision_type}`,
        "PUT",
        {
          level,
          bar,
          confidence_cut,
          allow_raise_budget,
          allow_customer_messages,
        },
      );
      setPolicies((old) =>
        old.map((row) =>
          row.decision_type === result.decision_type ? result : row,
        ),
      );
      setNotice(
        `${decisionNames[result.decision_type]} preference saved. Current level: ${result.effective_level}.`,
      );
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="harness-screen" aria-labelledby="autonomy-title">
      <p className="eyebrow">PERMISSION COMES FROM YOUR DECISIONS</p>
      <h1 id="autonomy-title">How much can Ren do?</h1>
      <p>
        Watch: propose only. Ask: wait for your confirmation. Act: handle
        eligible decisions automatically, with random spot-checks and the
        model’s existing certification.
      </p>
      <p role="status">{notice}</p>
      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}
      {policies.map((policy) => (
        <section
          className="autonomy-row"
          key={policy.decision_type}
          aria-label={`${decisionNames[policy.decision_type]} autonomy`}
        >
          <div className="autonomy-heading">
            <h2>{decisionNames[policy.decision_type]}</h2>
            <span>
              Current level: <strong>{policy.effective_level}</strong>
            </span>
          </div>
          <div className="policy-fields">
            <label>
              Permission
              <select
                disabled={busy}
                value={policy.level}
                onChange={(event) =>
                  void save(policy, {
                    level: event.target.value as Policy["level"],
                  })
                }
              >
                <option value="watch">Watch</option>
                <option value="ask">Ask</option>
                <option value="act">Act when earned</option>
              </select>
            </label>
            <label>
              Your required bar
              <select
                disabled={busy}
                value={policy.bar}
                onChange={(event) =>
                  void save(policy, {
                    bar: Number(event.target.value) as Policy["bar"],
                  })
                }
              >
                {[90, 95, 98, 99].map((bar) => (
                  <option key={bar} value={bar}>
                    {bar}% target
                  </option>
                ))}
              </select>
            </label>
            <label>
              Confidence cut
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                defaultValue={policy.confidence_cut}
                key={policy.confidence_cut}
                disabled={busy}
                onBlur={(event) => {
                  const value = Number(event.target.value);
                  if (
                    event.target.validity.valid &&
                    value !== policy.confidence_cut
                  )
                    void save(policy, { confidence_cut: value });
                }}
              />
            </label>
          </div>
          <dl className="autonomy-numbers">
            <div>
              <dt>Reviewed</dt>
              <dd>{policy.reviewed}</dd>
            </div>
            <div>
              <dt>Confirmed</dt>
              <dd>{policy.confirmed}</dd>
            </div>
            <div>
              <dt>Corrected</dt>
              <dd>{policy.corrected}</dd>
            </div>
            <div>
              <dt>Measured lower bound</dt>
              <dd>{measured(policy.lower_bound)}</dd>
            </div>
          </dl>
          <progress
            max={policy.bar / 100}
            value={Math.min(policy.lower_bound, policy.bar / 100)}
            aria-label={`${decisionNames[policy.decision_type]} lower bound toward selected bar`}
          />
          <p>
            {policy.unlocked
              ? "Review evidence meets your bar. Act still requires the model’s certification and execution switches."
              : `${policy.confirmations_left} more consecutive confirmations needed at this confidence cut, assuming no corrections.`}
          </p>
          <p className="small-print">
            One-sided 95% Clopper–Pearson lower bound over the latest{" "}
            {policy.window} reviewed proposals above the cut. Each decision
            counts once. Corrections can remove Act access immediately.
          </p>
          {policy.decision_type !== "account" && (
            <label className="check-row">
              <input
                type="checkbox"
                disabled={busy || policy.bar !== 99}
                checked={
                  policy.decision_type === "ad"
                    ? policy.allow_raise_budget
                    : policy.allow_customer_messages
                }
                onChange={(event) =>
                  void save(
                    policy,
                    policy.decision_type === "ad"
                      ? { allow_raise_budget: event.target.checked }
                      : { allow_customer_messages: event.target.checked },
                  )
                }
              />
              <span>
                Allow{" "}
                {policy.decision_type === "ad"
                  ? "budget increases"
                  : "customer messages"}{" "}
                automatically only after the 99% target is met. Otherwise always
                ask.
              </span>
            </label>
          )}
        </section>
      ))}
      <details className="community-context">
        <summary>Track record across opted-in workspaces</summary>
        <p>
          Counts only, shared with permission. This context never unlocks Act.
          No customer messages, names or identifiers are shared.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Type</th>
                <th>Proposals</th>
                <th>Confirmed</th>
                <th>Corrected</th>
                <th>Workspaces</th>
              </tr>
            </thead>
            <tbody>
              {counts.map((row) => (
                <tr key={row.decision_type}>
                  <th scope="row">{row.decision_type}</th>
                  <td>{row.proposals}</td>
                  <td>{row.confirmed}</td>
                  <td>{row.corrected}</td>
                  <td>{row.workspaces}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}

type ActivityState = {
  enabled: boolean;
  share_counts: boolean;
  global_killed: boolean;
  worker_enabled: boolean;
  activity: { id: number; event: string; created_at: string }[];
};
type Outcome = {
  item_id: string;
  action: string;
  status: string;
  due_date: string;
  before: Metrics;
  after: Metrics | null;
};
type Metrics = {
  spend: string;
  cpa: string | null;
  ctr: string | null;
  currency: string;
};
export function ActivityScreen() {
  const [state, setState] = useState<ActivityState | null>(null);
  const [outcomes, setOutcomes] = useState<Outcome[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const load = useCallback(async () => {
    const [activity, results] = await Promise.all([
      harnessRequest<ActivityState>(`${root}/agent`),
      harnessRequest<Outcome[]>(`${root}/outcomes`),
    ]);
    setState(activity);
    setOutcomes(results);
  }, []);
  useEffect(() => {
    void load().catch((cause) => setError(errorMessage(cause)));
    const timer = window.setInterval(() => {
      void load().catch((cause) => setError(errorMessage(cause)));
    }, 10000);
    return () => clearInterval(timer);
  }, [load]);
  async function configure(patch: Partial<ActivityState>) {
    if (!state || busy) return;
    setBusy(true);
    setError("");
    try {
      setState(
        await harnessRequest<ActivityState>(`${root}/agent`, "PUT", {
          enabled: state.enabled,
          share_counts: state.share_counts,
          ...patch,
        }),
      );
      setNotice("Preference saved.");
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="harness-screen" aria-labelledby="activity-title">
      <p className="eyebrow">EVERY STEP, ACCOUNTED FOR</p>
      <h1 id="activity-title">Ren’s activity</h1>
      <p role="status">{notice}</p>
      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}
      <div className="activity-controls">
        <label className="check-row">
          <input
            type="checkbox"
            checked={state?.enabled ?? false}
            disabled={busy || !state || state.global_killed}
            onChange={(event) =>
              void configure({ enabled: event.target.checked })
            }
          />
          <span>Watch this workspace in the background</span>
        </label>
        {!state?.worker_enabled && (
          <p>
            Background processing is off at the server. Your workspace
            preference is saved for when it is enabled.
          </p>
        )}
        {state?.global_killed && (
          <p role="alert">
            All automatic actions are stopped. The operator must reset the
            global stop before processing resumes.
          </p>
        )}
        <button
          className="kill-button"
          disabled={busy || state?.global_killed}
          onClick={() => {
            setBusy(true);
            void harnessRequest("/api/v1/agent/kill", "POST")
              .then(load)
              .then(() => setNotice("All automatic actions stopped."))
              .catch((cause) => setError(errorMessage(cause)))
              .finally(() => setBusy(false));
          }}
        >
          Stop all automatic actions
        </button>
        <label className="check-row">
          <input
            type="checkbox"
            checked={state?.share_counts ?? false}
            disabled={busy || !state}
            onChange={(event) =>
              void configure({ share_counts: event.target.checked })
            }
          />
          <span>
            Share anonymous aggregate decision counts. Off by default.
          </span>
        </label>
      </div>
      <h2>Recent activity</h2>
      {!state?.activity.length && (
        <p>
          No activity yet. Connect Meta, choose accounts, and enable this
          workspace.
        </p>
      )}
      <ol className="activity-timeline">
        {state?.activity.map((row) => (
          <li key={row.id}>
            <span>{row.event.replaceAll("_", " ")}</span>
            <time dateTime={row.created_at}>
              {new Date(row.created_at).toLocaleString()}
            </time>
          </li>
        ))}
      </ol>
      <div className="outcome-heading">
        <h2>What happened after</h2>
        <a href={`${root}/outcomes.csv`}>Export observed results as CSV</a>
      </div>
      <p>
        Before and seven days after each ad decision. Observed results are not
        proof that the decision caused a change. Review the result in your inbox
        before it becomes a training label.
      </p>
      {!outcomes.length && (
        <p>
          Observed ad results will appear here after decisions have had seven
          days.
        </p>
      )}
      {outcomes.map((row) => (
        <article className="outcome-row" key={row.item_id}>
          <h3>
            {row.action} · {row.status.replaceAll("_", " ")}
          </h3>
          {row.status === "executed" && (
            <button
              className="secondary-button"
              disabled={busy}
              onClick={() => {
                setBusy(true);
                void harnessRequest(
                  `${root}/integrations/ads/proposals/${encodeURIComponent(row.item_id)}:undo`,
                  "POST",
                )
                  .then(load)
                  .then(() =>
                    setNotice(
                      "Undo request recorded. Check the action log for the result.",
                    ),
                  )
                  .catch((cause) => setError(errorMessage(cause)))
                  .finally(() => setBusy(false));
              }}
            >
              Undo ad change
            </button>
          )}
          {row.after ? (
            <dl className="autonomy-numbers">
              {(["spend", "cpa", "ctr"] as const).map((metric) => (
                <div key={metric}>
                  <dt>{metric.toUpperCase()}</dt>
                  <dd>
                    {row.before[metric] ?? "Unavailable"} →{" "}
                    {row.after?.[metric] ?? "Unavailable"}
                    {metric === "ctr" ? "%" : ` ${row.before.currency}`}
                  </dd>
                </div>
              ))}
            </dl>
          ) : (
            <p>Observation due {row.due_date}. No result measured yet.</p>
          )}
        </article>
      ))}
    </section>
  );
}
