import { workspaceId } from "./api";
export type Entry = { message: string; created_at: string };
const key = `secnd-permission-history-${workspaceId}`;
export function savedHistory(): Entry[] {
  try { return JSON.parse(localStorage.getItem(key) ?? "[]"); } catch { return []; }
}
export function recordPermission(message: string) {
  localStorage.setItem(key, JSON.stringify([{ message, created_at: new Date().toISOString() }, ...savedHistory()].slice(0, 100)));
  window.dispatchEvent(new Event("secnd-permission"));
}
export type PermissionSnapshot = { decision_type: string; effective_level: string; lower_bound: number };
export function observePermissions(policies: PermissionSnapshot[]) {
  const snapshotKey = `secnd-permission-snapshot-${workspaceId}`;
  let previous: PermissionSnapshot[] = [];
  try { previous = JSON.parse(localStorage.getItem(snapshotKey) ?? "[]"); } catch { /* New browser history. */ }
  for (const p of policies) {
    const old = previous.find(row => row.decision_type === p.decision_type);
    if (old && (p.lower_bound < old.lower_bound || old.effective_level !== p.effective_level))
      recordPermission(`${p.decision_type === "ad" ? "Ads" : p.decision_type === "inbox" ? "Inbox" : "Account"}: measured lower bound ${(old.lower_bound * 100).toFixed(2)}% → ${(p.lower_bound * 100).toFixed(2)}%; current level ${old.effective_level} → ${p.effective_level}.`);
  }
  localStorage.setItem(snapshotKey, JSON.stringify(policies));
}
