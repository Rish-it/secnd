// Mirrors the workspace types in docs/api/openapi.json.
export type Item = { item_id: string; text: string; amount?: number | null; date?: string | null };
export type Decision = {
  item_id: string; label: string; probabilities: Record<string, number>;
  prob: number; auto: boolean; threshold: number | null; model_version: string;
  reasons: { merchant_memory: boolean; neighbours: { item_id: string; label: string; similarity: number; text: string }[] };
  item?: Item | null; spot_check: boolean;
};
export type EvalCard = {
  model_version: string; coverage: number;
  accuracy?: number | null; ece?: number | null; threshold?: number | null;
  precision?: number | null; cp_lower_bound?: number | null;
  robustness_pass_rate?: number | null; retrain_ms?: number | null;
  cost_per_1000_usd?: number | null;
};
export type Page<T> = { items: T[]; next_cursor: string | null };
export type ModelVersion = { model_version: string; current: boolean; card: EvalCard };
export type Metrics = {
  card: EvalCard;
  history: { timestamp: string; model_version: string; coverage: number; precision: number | null }[];
};
export type Preview = { filename: string; columns: string[]; row_count: number; rows: Record<string, string>[] };
export type Result<T> = { data: T; fixture: boolean };
const base = '/api/v1/workspaces/default';

async function request<T>(path: string, init: RequestInit = {}): Promise<Result<T>> {
  let response: Response;
  try {
    response = await fetch(path, { ...init, signal: AbortSignal.timeout(30000) });
  } catch {
    throw new Error('Could not reach Secnd. Try again when your connection is ready.');
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(typeof data?.message === 'string' ? data.message
      : typeof data?.detail === 'string' ? data.detail : 'This request could not be completed. Please try again.');
  }
  if (data === null) throw new Error('Secnd returned an unreadable response. Please try again.');
  return { data: data as T, fixture: response.headers.get('X-Secnd-Mode') === 'fixture' };
}
function json(body: unknown): RequestInit {
  return { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
}
export const api = {
  preview(file: File) {
    const form = new FormData(); form.append('file', file);
    return request<Preview>('/api/preview', { method: 'POST', body: form });
  },
  teach(file: File, target: string) {
    const form = new FormData(); form.append('file', file); form.append('target_column', target);
    return request<{ workspace_id: string; row_count: number; target_column: string; card: EvalCard }>(`${base}/decisions:import`, { method: 'POST', body: form });
  },
  feed: (cursor?: string) => request<Page<Decision>>(`${base}/feed${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ''}`),
  queue: (cursor?: string) => request<Page<Decision>>(`${base}/queue${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ''}`),
  metrics: () => request<Metrics>(`${base}/metrics`),
  models: () => request<Page<ModelVersion>>(`${base}/models`),
  answer(item_id: string, label: string, idempotencyKey: string) {
    return request<{ item_id: string; coverage: number; model_version: string; retrain: 'queued' | 'fixture' }>(`${base}/answers`, {
      ...json({ item_id, label }), headers: { 'Content-Type': 'application/json', 'Idempotency-Key': idempotencyKey },
    });
  },
  rollback: (version: string) => request<ModelVersion>(`${base}/models/${encodeURIComponent(version)}:rollback`, { method: 'POST' }),
  chat: (question: string) => request<{ answer: string; model_version: string | null }>(`${base}/chat`, json({ question })),
};
export const percent = (value: number | null | undefined) => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
