import { useEffect, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { api, percent, workspaceId } from "./api";
import type { Preview, EvalCard } from "./api";
import Companion from "./Companion";
import MetaConnections from "./MetaConnections";
import { SessionGate, AutonomyScreen, ActivityScreen } from "./Harness";
import { businessRequest, businessRoot } from "./business";
import { observePermissions } from "./permissionLedger";
import type { PermissionSnapshot } from "./permissionLedger";
import "./App.css";
import "./Product.css";

type View = "Inbox" | "Autonomy" | "Activity" | "Connections" | "Past decisions" | "About";
type Serving = { model_id: string; model_card_url?: string; card_url?: string; device?: string; synthetic?: boolean; business_name?: string };
function NavIcon({ view }: { view: View }) {
  const paths: Record<View, string> = {
    Inbox: "M4 4h16v16H4V4Zm0 10h5l2 3h2l2-3h5",
    Autonomy: "M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6l-8-3Zm-4 9 3 3 5-6",
    Activity: "M3 12h4l3-7 4 14 3-7h4",
    Connections: "M8 7H5a4 4 0 0 0 0 8h4m6 0h4a4 4 0 0 0 0-8h-4M7 11h10",
    "Past decisions": "M4 4h16v16H4V4Zm4 4h8M8 12h8M8 16h5",
    About: "M12 11v6m0-10v1M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  };
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[view]} /></svg>;
}
export default function App() { return <SessionGate><AppBody /></SessionGate>; }
function AppBody() {
  const [view, setView] = useState<View>(new URLSearchParams(location.search).has("meta_error") ? "Connections" : "Inbox");
  const [theme, setTheme] = useState(() => localStorage.getItem("secnd-theme") ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  const [serving, setServing] = useState<Serving | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("secnd-theme", theme); }, [theme]);
  useEffect(() => {
    let active = true;
    void businessRequest<Serving>(`${businessRoot}/serving`).then(result => { if (active) setServing(result); }).catch(() => {});
    void businessRequest<{ status: string; database: string }>("/api/health").then(result => { if (active) setOnline(result.status === "ok" && result.database === "ready"); }).catch(() => { if (active) setOnline(false); });
    void businessRequest<PermissionSnapshot[]>(`${businessRoot}/autonomy`).then(observePermissions).catch(() => {});
    return () => { active = false; };
  }, []);
  const rawCardUrl = serving?.model_card_url ?? serving?.card_url;
  const cardUrl = rawCardUrl && /^(https?:\/\/|\/(?!\/))/.test(rawCardUrl) ? rawCardUrl : undefined;
  const identity = serving?.model_id ?? "Model details unavailable";
  function modelIdentity() { return cardUrl ? <a href={cardUrl} target="_blank" rel="noreferrer">{identity}</a> : <span>{identity}</span>; }
  return <div className="app-shell">
    <a className="skip-link" href="#main">Skip to content</a>
    <aside className="sidebar">
      <a className="brand" href="/" aria-label="Secnd home"><span className="brand-symbol">s</span>secnd<span className="brand-period">.</span></a>
      <div className="workspace-label"><span className="workspace-avatar">S</span><span>{serving?.business_name ?? (workspaceId === "default" ? "Your business" : workspaceId.replaceAll(/[-_]/g, " "))}<small>A second pair of eyes.</small></span></div>
      <nav aria-label="Business">{(["Inbox", "Autonomy", "Activity", "Connections", "Past decisions", "About"] as const).map(item => <button key={item} className={`nav-item ${view === item ? "active" : ""}`} aria-current={view === item ? "page" : undefined} onClick={() => { setView(item); window.scrollTo({ top: 0 }); }}><NavIcon view={item} /><span>{item}</span></button>)}</nav>
      <div className="sidebar-note"><p>Your standards.<br />A little more headspace.</p></div>
      <button className="text-button theme-toggle" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label={`Use ${theme === "light" ? "dark" : "light"} appearance`}>{theme === "light" ? "Dark appearance" : "Light appearance"}</button>
    </aside>
    <div className="workspace-main"><header className="topbar"><span>YOUR BUSINESS <span className="breadcrumb-slash">/</span> <strong>{view}</strong></span><span className={`connection ${online ? "online" : "offline"}`} role="status"><span />{online === null ? "Connecting…" : online ? "Connected" : "Connection unavailable"}</span></header>
      <main id="main" tabIndex={-1}>
        {view === "Inbox" && <Companion onAnswer={() => { void businessRequest<PermissionSnapshot[]>(`${businessRoot}/autonomy`).then(observePermissions).catch(() => {}); }} />}
        {view === "Autonomy" && <AutonomyScreen />}
        {view === "Activity" && <ActivityScreen />}
        {view === "Connections" && <MetaConnections />}
        {view === "Past decisions" && <PastDecisions />}
        {view === "About" && <section className="harness-screen about-product" aria-labelledby="about-title"><p className="eyebrow">A SECOND PAIR OF EYES</p><h1 id="about-title">Your business. With you.</h1>
          <p>Secnd runs your Meta business with you. Sign in, connect your accounts, and let Ren watch your ads and inboxes. It asks one tap at a time and earns the right to act, separately for each kind of decision, up to a bar you choose.</p>
          <h2>Ren makes the call.</h2><p>Ren is our 400M decision model, trained once. It reads your past decisions as examples. Your reviews change its evidence and permissions.</p>
          <h2>Muse never decides.</h2><p>Muse Spark Open is Meta’s language model. It supports the project’s language work; the decision and permission checks belong to Ren. Reviewing a decision makes no call to Muse.</p>
          <div className="running-model"><p className="eyebrow">RUNNING IN THIS BUSINESS</p><p>{modelIdentity()}{serving?.device && ` · ${serving.device.toUpperCase()}`}</p><p className="small-print">{serving?.synthetic ? "Synthetic scripted proposals for the demo. The 400M model is not running here." : "This identifier comes from the running API. Deployment configuration determines which model is active."}</p></div>
          <p>Built for Meta's platform. Secnd is independently developed and is not an official Meta product.</p>
        </section>}
        <footer className="page-footer"><span>Built for Meta's platform<br />Built with Muse Spark Open</span><span>{modelIdentity()}</span></footer>
      </main>
    </div>
  </div>;
}
function PastDecisions() {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [target, setTarget] = useState("");
  const [card, setCard] = useState<EvalCard | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function inspect(event: ChangeEvent<HTMLInputElement>) {
    const chosen = event.target.files?.[0]; if (!chosen) return;
    setError(""); setCard(null); setPreview(null); setFile(null);
    if (chosen.size > 5 * 1024 * 1024) { setError("Choose a CSV smaller than 5 MiB."); return; }
    setBusy(true);
    try { const result = await api.preview(chosen); setPreview(result.data); setFile(chosen); setTarget(result.data.columns.find(column => /^(label|category|target)$/i.test(column)) ?? result.data.columns.at(-1) ?? ""); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not read this file."); }
    finally { setBusy(false); event.target.value = ""; }
  }
  return <section className="harness-screen"><p className="eyebrow">YOUR JUDGMENT IS THE STARTING POINT</p><h1>Bring your past decisions.</h1><p>Upload a CSV with the situation and the action you chose. Ren uses your decisions as examples. Start with your existing history, then confirm or correct new suggestions in your inbox.</p>
    <div className="upload-area"><h2>{preview ? preview.filename : "Start with what you know."}</h2><p>A situation column and an action or label column · CSV · up to 5 MiB</p><input className="visually-hidden" ref={input} type="file" accept=".csv,text/csv" aria-label="Upload past decisions CSV" onChange={event => void inspect(event)} /><button className="primary-button" disabled={busy} onClick={() => input.current?.click()}>{busy ? "Reading…" : "Choose a CSV"}</button></div>
    {preview && <div className="preview"><p>{preview.row_count} past decisions found.</p><label className="form-label">The action you chose<select value={target} onChange={e => setTarget(e.target.value)}>{preview.columns.map(column => <option key={column}>{column}</option>)}</select></label><button className="primary-button" disabled={busy || !file || !target} onClick={() => {
      if (!file) return; setBusy(true); setError("");
      void api.teach(file, target).then(result => { if (result.fixture) setError("Preview only. Examples were not saved. Connect the API and try again."); else setCard(result.data.card); }).catch(cause => setError(cause instanceof Error ? cause.message : "Could not save examples.")).finally(() => setBusy(false));
    }}>Save past decisions</button></div>}
    {card && <p role="status">Examples saved. Measured automatic coverage: {percent(card.coverage)}. Open Inbox to review suggestions.</p>}
    {error && <p role="alert" className="error-message">{error}</p>}
  </section>;
}
