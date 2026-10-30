import type { Decision } from "./api";

const chosen =
  new URLSearchParams(location.search).get("workspace") ?? "default";
const workspace = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(chosen)
  ? chosen
  : "default";
const base = `/api/v1/workspaces/${encodeURIComponent(workspace)}/integrations`;
export type MetaStatus = {
  accounts: Record<"whatsapp" | "messenger" | "instagram", string[]>;
  model_ready: boolean;
  signing_configured: boolean;
  whatsapp_send_enabled: boolean;
};
export type AdsStatus = {
  demo: boolean;
  auto_act: boolean;
  account: string | null;
  sandbox_allowed: boolean;
  token_configured: boolean;
  server_send_enabled: boolean;
  workspace_ready: boolean;
  budget_step_percent: number;
};
export type AdsProposal = {
  item_id: string;
  action: "pause" | "keep" | "raise budget" | "lower budget";
  status: string;
  decision: Decision;
  insight: {
    level: "campaign" | "adset";
    name: string;
    spend: string;
    cpa: string | null;
    cpa_action_type: string;
    ctr: string | null;
    currency: string;
    since: string;
    until: string;
  };
};
export type LogPage = {
  items: {
    id: number;
    event: string;
    item_id: string | null;
    details: Record<string, unknown>;
    created_at: string;
  }[];
  next_after: number | null;
};
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${base}/${path}`, {
    ...init,
    signal: AbortSignal.timeout(60000),
  });
  const body = await response.json();
  if (!response.ok)
    throw new Error(body.message ?? "Could not update this connection.");
  return body as T;
}
function json(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
export const metaApi = {
  status: () => request<MetaStatus>("meta/status"),
  adsStatus: () => request<AdsStatus>("ads/status"),
  preferences: (auto_act: boolean) =>
    request<AdsStatus>("ads/preferences", json("PUT", { auto_act })),
  sync: (level: "campaign" | "adset", since: string, until: string) =>
    request<AdsProposal[]>(
      "ads/insights:sync",
      json("POST", { level, since, until }),
    ),
  execute: (id: string) =>
    request<AdsProposal>(`ads/proposals/${encodeURIComponent(id)}:execute`, {
      method: "POST",
    }),
  log: (after = 0) => request<LogPage>(`ads/log?after=${after}`),
};
