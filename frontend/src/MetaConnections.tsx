import { useEffect, useState } from "react";
import { MetaLogin } from "./Harness";
import SourceIcon from "./SourceIcon";
import { metaApi } from "./metaApi";
import type { MetaStatus, AdsStatus } from "./metaApi";
import "./MetaConnections.css";

export default function MetaConnections() {
  const [meta, setMeta] = useState<MetaStatus | null>(null);
  const [ads, setAds] = useState<AdsStatus | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    void Promise.all([metaApi.status(), metaApi.adsStatus()]).then(([messages, adStatus]) => { setMeta(messages); setAds(adStatus); }).catch(() => setError("Connection status is unavailable. Reopen Connections to retry."));
  }, []);
  return <section className="harness-screen" aria-labelledby="connections-title">
    <p className="eyebrow">YOUR BUSINESS, CONNECTED</p><h1 id="connections-title">Bring Ren into the loop.</h1>
    <p>Choose the accounts Ren can watch. Set what it can do in Autonomy.</p>
    {error && <p role="alert" className="error-message">{error}</p>}
    <MetaLogin />
    <div className="connection-list">{(["ads", "whatsapp", "messenger", "instagram"] as const).map(source => <div key={source}>
      <SourceIcon source={source} /><strong>{source === "ads" ? "Meta Ads" : source === "whatsapp" ? "WhatsApp" : source === "messenger" ? "Messenger" : "Instagram"}</strong>
      <span>{source === "ads" ? (ads?.sandbox_allowed ? "Sandbox account selected" : "Choose an ad account above") : meta?.accounts[source].length ? `${meta.accounts[source].length} account connected` : "Choose an account above"}</span>
    </div>)}</div>
    <p className="small-print">Customer replies and ad changes need separate permission. This version supports sandbox ad accounts. Account-management changes are saved as decisions only.</p>
  </section>;
}
