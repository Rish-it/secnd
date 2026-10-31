import { useEffect, useRef, useState } from "react";
import { workspaceId } from "./api";
import { businessRequest, businessRoot, errorMessage } from "./business";
import type { MetaConnection } from "./business";
import SourceIcon from "./SourceIcon";

export function SetupSteps({ step }: { step: number }) {
  return <ol className="setup-steps" aria-label="Set up your business">
    {["Sign in", "Connect Meta", "Pick accounts", "Set your bar", "Done"].map((name, i) =>
      <li key={name} aria-current={i === step ? "step" : undefined} className={i < step ? "complete" : ""}>
        <span aria-hidden="true">{i < step ? "✓" : i + 1}</span>{name}
      </li>)}
  </ol>;
}

export default function FirstRun({ demo, onDone }: { demo: boolean; onDone: () => void }) {
  const heading = useRef<HTMLHeadingElement>(null);
  const [step, setStep] = useState(demo ? 0 : 1);
  const [connection, setConnection] = useState<MetaConnection | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [bar, setBar] = useState(95);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => { heading.current?.focus(); }, [step]);
  useEffect(() => {
    if (demo) return;
    void businessRequest<MetaConnection>(`${businessRoot}/integrations/meta/connection`).then(result => {
      setConnection(result); setSelected(result.selected);
      if (result.connected) setStep(2);
    }).catch(cause => setError(errorMessage(cause)));
  }, [demo]);
  async function next() {
    setBusy(true); setError("");
    try {
      if (step === 0) setStep(1);
      if (step === 1 && demo) {
        const result = await businessRequest<MetaConnection>("/api/demo/connect", "POST");
        setConnection(result); setSelected(result.selected); setStep(2);
      }
      if (step === 2) {
        await businessRequest(`${businessRoot}/integrations/meta/accounts`, "PUT", { assets: selected });
        setStep(3);
      }
      if (step === 3) {
        // Preserve permission and special-action choices; setup changes only the chosen bar.
        const policies = await businessRequest<{ decision_type: string; level: string; confidence_cut: number; allow_raise_budget: boolean; allow_customer_messages: boolean }[]>(`${businessRoot}/autonomy`);
        for (const p of policies) await businessRequest(`${businessRoot}/autonomy/${p.decision_type}`, "PUT", {
          level: p.level, bar, confidence_cut: p.confidence_cut,
          allow_raise_budget: p.allow_raise_budget && bar === 99,
          allow_customer_messages: p.allow_customer_messages && bar === 99,
        });
        setStep(4);
      }
      if (step === 4) {
        localStorage.setItem(`secnd-setup-${workspaceId}`, "done");
        const url = new URL(location.href); url.searchParams.delete("setup"); url.searchParams.delete("connected");
        history.replaceState(null, "", url); onDone();
      }
    } catch (cause) { setError(errorMessage(cause)); } finally { setBusy(false); }
  }
  return <main className="setup-page">
    <a className="brand" href="/">secnd<span className="brand-period">.</span></a>
    <SetupSteps step={step} />
    {demo && <p className="synthetic-label">Synthetic walkthrough · no real sign-in or Meta connection</p>}
    <p className="eyebrow">YOUR BUSINESS, WITH A SECOND PAIR OF EYES</p>
    <h1 ref={heading} tabIndex={-1}>{["Meet your second.", "Bring your business along.", "Only the accounts you choose.", "You set the standard.", "You’re in charge."][step]}</h1>
    <p className="setup-intro">{[
      "Ren watches your Meta ads and inboxes, brings you one clear choice, and earns your permission to act.",
      "Connect with Facebook Login for Business so Ren can watch your ads and messages.",
      "Choose what Ren can watch. You can change this any time in Connections.",
      "Ren must reach this measured lower bound on your reviews before it can earn Act. Each decision type has its own progress.",
      "Your inbox is ready. Review a suggestion, see the past decisions behind it, and make your first call.",
    ][step]}</p>
    {step === 1 && <div className="setup-apps">{["ads", "whatsapp", "messenger", "instagram"].map(app => <SourceIcon key={app} source={app} />)}</div>}
    {step === 1 && !demo && <p className="small-print">Before public access is approved, you need an admin, developer or tester role on the Meta app. If access is refused, ask the app owner to add you.</p>}
    {step === 2 && <fieldset className="account-list" disabled={busy}><legend>Accounts for your business</legend>
      {connection?.assets.map(asset => <label className="check-row" key={asset.key}>
        <input type="checkbox" checked={selected.includes(asset.key)} onChange={e => setSelected(old => e.target.checked ? [...old, asset.key] : old.filter(key => key !== asset.key))} />
        <SourceIcon source={asset.kind} /><span><strong>{asset.name}</strong><small>{asset.kind === "messenger" ? "Facebook Page" : asset.kind === "ads" ? "Ad account" : asset.kind}</small></span>
      </label>)}
      {!connection?.assets.length && <p>No eligible accounts came back. Reconnect Meta with access to an ad account, Page or business inbox.</p>}
    </fieldset>}
    {step === 3 && <fieldset className="bar-options" disabled={busy}><legend>Required bar</legend>{[90, 95, 98, 99].map(value => <label key={value}>
      <input type="radio" name="setup-bar" checked={bar === value} onChange={() => setBar(value)} /><span>{value}%</span>
    </label>)}</fieldset>}
    {step === 3 && <p className="small-print">This is an evidence target, not a promise about every future decision. Budget increases and automatic customer replies need 99% plus your separate permission.</p>}
    {step === 4 && <p className="setup-result">Your bar: <strong>{bar}%</strong>. Ren asks whenever a decision has not earned permission. You can stop all automatic actions in Activity.</p>}
    {error && <p className="error-message" role="alert">{error}</p>}
    <div className="setup-next">{step === 1 && !demo ? <a className="primary-button" href={`${businessRoot}/integrations/meta/oauth/start`}>Connect Meta</a> :
      <button className="primary-button" disabled={busy || (step === 2 && !selected.length)} onClick={() => void next()}>{busy ? "Saving…" : ["Sign in to synthetic business", "Connect synthetic Meta", "Use selected accounts", "Save my bar", "Open my inbox"][step]}</button>}
      {step === 2 && !connection?.assets.length && <a className="secondary-button" href={`${businessRoot}/integrations/meta/oauth/start`}>Reconnect Meta</a>}
    </div>
    <p className="small-print">Built for Meta's platform · Built with Muse Spark Open. Independent product by Secnd.</p>
  </main>;
}
