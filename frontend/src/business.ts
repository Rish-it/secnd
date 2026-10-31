import { workspaceId } from "./api";

export const businessRoot = `/api/v1/workspaces/${encodeURIComponent(workspaceId)}`;
export async function businessRequest<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method, credentials: "same-origin", signal: AbortSignal.timeout(30000),
    ...(body === undefined ? {} : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.message === "string" ? data.message.replace(/workspace/gi, "business") : "Could not save. Try again.");
  return data as T;
}
export type MetaAsset = { key: string; kind: string; id: string; name: string };
export type MetaConnection = { connected: boolean; assets: MetaAsset[]; selected: string[]; access_note?: string };
export const errorMessage = (cause: unknown) => cause instanceof Error ? cause.message : "Could not connect. Try again.";
