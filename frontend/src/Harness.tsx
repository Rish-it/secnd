import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { workspaceId } from "./api";
import { businessRequest as harnessRequest, businessRoot as root, errorMessage } from "./business";
import type { MetaConnection as Connection } from "./business";
import FirstRun, { SetupSteps } from "./FirstRun";
import SourceIcon from "./SourceIcon";
import PermissionHistory from "./PermissionHistory";
import { recordPermission, observePermissions } from "./permissionLedger";
import "./Harness.css";
const measured = (value: number) => `${Math.floor(value * 10000) / 100}%`;
type Session = { signed_in: boolean; demo: boolean };

export function SessionGate({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [businesses, setBusinesses] = useState<string[]>([]);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [create, setCreate] = useState(false);
  const [business, setBusiness] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [setup, setSetup] = useState<boolean | null>(null);
  const load = useCallback(async () => {
    const result = await harnessRequest<Session>("/api/v1/auth/session");
    if (result.signed_in || result.demo) {
      const list = await harnessRequest<{ items: string[] }>("/api/v1/workspaces");
      setBusinesses(list.items);
      const requested = new URLSearchParams(location.search).has("setup");
      if (requested) setSetup(true);
      else if (result.demo || localStorage.getItem(`secnd-setup-${workspaceId}`)) setSetup(false);
      else if (list.items.includes(workspaceId)) {
        try {
          const connection = await harnessRequest<Connection>(`${root}/integrations/meta/connection`);
          setSetup(!connection.selected.length);
        } catch { setSetup(true); }
      } else setSetup(true);
    }
    setSession(result);
  }, []);
  useEffect(() => { void load().catch(cause => setError(errorMessage(cause))); }, [load]);
  if (!session && !error) return <main className="sign-in"><p role="status">Opening Secnd…</p></main>;
  if (!session || (!session.signed_in && !session.demo)) return <main className="sign-in">
    <a className="brand" href="/">secnd<span className="brand-period">.</span></a>
    <SetupSteps step={0} />
    <p className="eyebrow">YOUR BUSINESS. YOUR JUDGMENT.</p><h1>Meet your second.</h1>
    <p>Ren watches your Meta ads and inboxes, asks one tap at a time, and earns your permission to act. You choose the standard.</p>
    <form onSubmit={event => {
      event.preventDefault(); setBusy(true); setError("");
      void harnessRequest("/api/v1/auth/" + (create ? "register" : "login"), "POST", { username, password })
        .then(() => { setPassword(""); return load(); }).catch(cause => setError(errorMessage(cause))).finally(() => setBusy(false));
    }}>
      <h2>{create ? "Create your account" : "Sign in"}</h2>
      <label>Sign-in name<input autoComplete="username" required minLength={3} maxLength={128} value={username} onChange={e => setUsername(e.target.value)} /></label>
      <label>Password<input type="password" autoComplete={create ? "new-password" : "current-password"} minLength={12} maxLength={256} required value={password} onChange={e => setPassword(e.target.value)} /></label>
      <small>Use at least 12 characters.</small>
      <button className="primary-button" disabled={busy}>{busy ? "Opening…" : create ? "Create account" : "Sign in"}</button>
      <button type="button" className="text-button" onClick={() => setCreate(v => !v)}>{create ? "I have an account" : "Create an account"}</button>
    </form>
    {error && <p role="alert" className="error-message">{error}</p>}
    <p className="small-print">Built for Meta's platform. Built with Muse Spark Open. Secnd is an independent product, not an official Meta product.</p>
  </main>;
  if (session.signed_in && !businesses.includes(workspaceId)) return <main className="sign-in">
    <a className="brand" href="/">secnd.</a><SetupSteps step={0} /><h1>Your business.</h1>
    <p>{businesses.length ? "Open your business or add another." : "Give your business a short name to continue."}</p>
    {businesses.map(name => <a className="workspace-link" key={name} href={`/?workspace=${encodeURIComponent(name)}`}>Open {name}</a>)}
    <form onSubmit={event => {
      event.preventDefault(); setBusy(true); setError("");
      void harnessRequest("/api/v1/workspaces", "POST", { workspace_id: business })
        .then(() => { location.href = `/?workspace=${encodeURIComponent(business)}&setup=1`; })
        .catch(cause => setError(errorMessage(cause))).finally(() => setBusy(false));
    }}><label>Business name<input required pattern="[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}" maxLength={64} value={business} onChange={e => setBusiness(e.target.value)} /></label>
    <small>Letters, numbers and hyphens, without spaces.</small><button className="primary-button" disabled={busy}>Continue to Connect Meta</button></form>
    {error && <p role="alert">{error}</p>}
  </main>;
  return <>
    {session.demo && <div className="synthetic-ribbon" role="status">Synthetic demo · fictional business and decisions · no live Meta actions</div>}
    {setup ? <FirstRun demo={session.demo} onDone={() => setSetup(false)} /> : <>
      <div className="session-tools"><span>{session.demo ? "Synthetic business" : "Signed in"}</span>
      {session.signed_in && <button onClick={() => void harnessRequest("/api/v1/auth/logout", "POST").then(() => location.reload()).catch(cause => setError(errorMessage(cause)))}>Sign out</button>}
      {error && <span role="alert">{error}</span>}</div>{children}
    </>}
  </>;
}

export function MetaLogin() {
  const [connection, setConnection] = useState<Connection | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState(new URLSearchParams(location.search).has("meta_error") ? "Meta refused access. Check your app role and reconnect." : "");
  useEffect(() => {
    void harnessRequest<Connection>(`${root}/integrations/meta/connection`).then(result => { setConnection(result); setSelected(result.selected); }).catch(cause => setError(errorMessage(cause)));
  }, []);
  return <div className="meta-login">
    <h3>{connection?.connected ? "Accounts Ren can watch" : "Connect Meta"}</h3>
    <p>Connect through Facebook Login for Business, then choose your accounts.</p>
    <details><summary>Who can connect?</summary><p>Until the Meta app has Advanced Access, sign-in is limited to people with an admin, developer or tester role. Ask the app owner to add your role if access is refused.</p></details>
    {!connection?.connected && <a className="primary-button" href={`${root}/integrations/meta/oauth/start`}>Connect Meta</a>}
    {connection?.connected && <form onSubmit={event => {
      event.preventDefault(); setBusy(true); setError("");
      void harnessRequest<Connection>(`${root}/integrations/meta/accounts`, "PUT", { assets: selected }).then(result => { setConnection(result); setSelected(result.selected); setNotice("Accounts saved. Ren watches only your selections."); }).catch(cause => setError(errorMessage(cause))).finally(() => setBusy(false));
    }}><fieldset disabled={busy}><legend>Choose accounts</legend>{connection.assets.map(asset => <label className="check-row" key={asset.key}>
      <input type="checkbox" checked={selected.includes(asset.key)} onChange={e => setSelected(old => e.target.checked ? [...old, asset.key] : old.filter(key => key !== asset.key))} />
      <SourceIcon source={asset.kind} /><span><strong>{asset.name}</strong><small>{asset.kind === "messenger" ? "Facebook Page" : asset.kind}</small></span>
    </label>)}{!connection.assets.length && <p>No eligible accounts. Reconnect Meta and grant access to your business accounts.</p>}</fieldset>
    <button className="primary-button" disabled={busy}>Use selected accounts</button></form>}
    {connection?.connected && <details><summary>Manage connection</summary>
      <a className="secondary-button" href={`${root}/integrations/meta/oauth/start`}>Reconnect Meta</a>
      <button className="text-button" disabled={busy} onClick={() => { setBusy(true); void harnessRequest("/api/v1/meta/disconnect", "POST").then(() => { setConnection(null); setSelected([]); setNotice("Disconnected. Stored access has been deleted."); }).catch(cause => setError(errorMessage(cause))).finally(() => setBusy(false)); }}>Disconnect Meta from my businesses</button>
    </details>}
    {notice && <p role="status">{notice}</p>}{error && <p role="alert" className="error-message">{error}</p>}
  </div>;
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
      harnessRequest<Policy[]>(`${root}/autonomy`).then(result => { observePermissions(result); setPolicies(result); }),
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
      recordPermission(`${decisionNames[result.decision_type]}: ${policy.level} → ${result.level}; bar ${policy.bar}% → ${result.bar}%. Measured lower bound ${measured(result.lower_bound)}. Current level: ${result.effective_level}.`);
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
      {!policies.length && !error && <p role="status">Loading your permissions…</p>}
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
            <fieldset className="level-control" disabled={busy}><legend>Permission</legend>
              {(["watch", "ask", "act"] as const).map(level => <label key={level}>
                <input type="radio" name={`level-${policy.decision_type}`} checked={policy.level === level} onChange={() => void save(policy, { level })} />
                <span>{level === "watch" ? "Watch" : level === "ask" ? "Ask" : "Act"}</span>
              </label>)}
            </fieldset>
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
              Ask below this confidence
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
            aria-valuetext={`${measured(policy.lower_bound)} measured; ${policy.bar}% required; ${policy.confirmations_left} confirmations left`}
          />
          <p>
            {policy.unlocked
              ? "Your reviews meet the bar. Act also needs model certification and permission to perform the action."
              : `${policy.confirmations_left} more consecutive confirmations to reach your bar, assuming no corrections.`}
          </p>
          <p className="small-print">
            Conservative lower bound at 95% statistical confidence over the latest{" "}
            {policy.window} reviewed proposals above your confidence choice. Each decision
            counts once. Corrections can remove Act access immediately.
          </p>
          <p className="always-asks"><strong>Always asks:</strong> decisions below your confidence choice, spot-checks{policy.decision_type === "account" ? ", and all account changes (no account changes are performed yet)." : policy.decision_type === "ad" ? ", and budget increases unless you separately allow them at 99%." : ", and customer replies unless you separately allow them at 99%."}</p>
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
      <PermissionHistory />
      <details className="community-context">
        <summary>Track record across opted-in businesses</summary>
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
                <th>Businesss</th>
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
  const [confirmStop, setConfirmStop] = useState(false);
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
          <span>Watch this business in the background</span>
        </label>
        {!state?.worker_enabled && (
          <p>
            Background processing is off at the server. Your business
            preference is saved for when it is enabled.
          </p>
        )}
        {state?.global_killed && (
          <p role="alert">
            All automatic actions are stopped. The operator must reset the
            global stop before processing resumes.
          </p>
        )}
        <div className="stop-control">
          <h2>Stop Ren</h2><p>Stop automatic actions across all your businesses. Already submitted requests may finish.</p>
          {!confirmStop ? <button className="kill-button" disabled={busy || !state || state.global_killed} onClick={() => setConfirmStop(true)}>{state?.global_killed ? "Automatic actions stopped" : "Stop all automatic actions"}</button> :
          <div className="stop-confirm" role="alertdialog" aria-modal="false" aria-labelledby="stop-title" aria-describedby="stop-description">
            <h3 id="stop-title">Stop all automatic actions?</h3><p id="stop-description">Ren will stop sending replies and changing ads. An operator must reset the stop before actions resume.</p>
            <button className="kill-button" autoFocus disabled={busy} onClick={() => {
              setBusy(true); setError("");
              void harnessRequest("/api/v1/agent/kill", "POST").then(load).then(() => { setNotice("All automatic actions stopped."); setConfirmStop(false); }).catch(cause => setError(errorMessage(cause))).finally(() => setBusy(false));
            }}>Confirm stop</button>
            <button className="secondary-button" disabled={busy} onClick={() => setConfirmStop(false)}>Keep running</button>
          </div>}
        </div>
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
      <PermissionHistory />
      <h2>Agent log</h2>
      {!state?.activity.length && (
        <p>
          No activity yet. Connect Meta, choose accounts, and enable this
          business.
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
                  <dt>{metric === "spend" ? "Spend" : metric === "cpa" ? "Cost per result" : "Click-through"}</dt>
                  <dd>
                    <span><small>Before</small>{row.before[metric] ?? "Unavailable"}</span> →{" "}
                    <span><small>After 7 days</small>{row.after?.[metric] ?? "Unavailable"}</span>
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
