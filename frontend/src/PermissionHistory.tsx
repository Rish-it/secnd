import { useEffect, useState } from "react";
import { businessRequest, businessRoot } from "./business";
import { savedHistory } from "./permissionLedger";
import type { Entry } from "./permissionLedger";

export default function PermissionHistory() {
  const [local, setLocal] = useState(savedHistory);
  const [remote, setRemote] = useState<Entry[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    const update = () => setLocal(savedHistory());
    window.addEventListener("secnd-permission", update);
    void Promise.all([
      businessRequest<{ items: { item_id: string; label: string; answer_label: string; item?: { text?: string } }[] }>(`${businessRoot}/overridden`),
      businessRequest<{ activity: { event: string; created_at: string }[] }>(`${businessRoot}/agent`),
    ]).then(([page, agent]) => {
      setRemote([
        ...agent.activity.filter(row => /lower bound|level|correct/i.test(row.event)).map(row => ({ message: row.event.replaceAll("_", " "), created_at: row.created_at })),
        ...page.items.map(item => ({ message: `Corrected: ${item.item?.text ?? item.item_id} · ${item.label} → ${item.answer_label}`, created_at: "" })),
      ]);
    }).catch(() => setError("Correction history is unavailable. Reopen this screen to retry."));
    return () => window.removeEventListener("secnd-permission", update);
  }, []);
  return <section className="permission-history" aria-label="Permission and correction history">
    <h2>Permission & correction history</h2>
    <p className="small-print">Corrections are saved with this business. Permission changes below are recorded on this browser.</p>
    {error && <p role="status">{error}</p>}
    {!local.length && !remote.length && <p>No changes yet. Your corrections and changes to Ren’s level will appear here.</p>}
    <ol className="activity-timeline">{[...local, ...remote].map((entry, index) => <li key={`${entry.created_at}-${index}`}><span>{entry.message}</span>{entry.created_at && <time dateTime={entry.created_at}>{new Date(entry.created_at).toLocaleString()}</time>}</li>)}</ol>
  </section>;
}
