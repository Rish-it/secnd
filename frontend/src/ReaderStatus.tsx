import { useEffect, useState } from "react";
import "./ReaderStatus.css";
import { workspaceId } from "./api";

function percent(value: number | null | undefined, digits = 1) {
  return value == null ? "Unavailable" : `${(value * 100).toFixed(digits)}%`;
}

type Status = {
  demo?: boolean;
  backend: "reader" | "head";
  model_id: string;
  revision?: string;
  device: string;
  examples?: number;
  history_rows?: number;
  policy_complete?: boolean;
  scope_matches?: boolean;
  guarded?: boolean;
  threshold?: number | null;
  temperature?: number | null;
  support_floor?: number | null;
  scope_note: string;
  benchmark?: {
    dataset: string;
    history_rows: number;
    accuracy: number | null;
    with_floor_coverage: number | null;
    with_floor_precision: number | null;
    with_floor_sample: number;
    with_floor_accepted: number | null;
    with_floor_errors: number | null;
    with_floor_low95: number | null;
    audited_calibration_labels: number | null;
    floor_result_file: string;
  };
};

export default function ReaderStatus() {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    async function load() {
      try {
        const response = await fetch(
          `/api/v1/workspaces/${encodeURIComponent(workspaceId)}/serving`,
          { signal: AbortSignal.timeout(10000) },
        );
        if (response.status === 404) {
          if (active) {
            setStatus(null);
            setError("");
          }
          return;
        }
        if (!response.ok)
          throw new Error("Serving details unavailable. Reload to retry.");
        const data: Status = await response.json();
        if (active) {
          setStatus({
            ...data,
            demo: response.headers.get("X-Secnd-Mode") === "demo",
          });
          setError("");
        }
      } catch {
        if (active) setError("Serving details unavailable. Reload to retry.");
      }
    }
    void load();
    const timer = window.setInterval(() => void load(), 5000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);
  if (!status) return error ? <p role="status">{error}</p> : null;
  const benchmark = status.benchmark;
  return (
    <section className="reader-status" aria-labelledby="reader-status-title">
      <p className="eyebrow">ACTIVE DECISION MODEL</p>
      {status.demo && (
        <p>
          Fixture workspace · model inference simulated. Published evaluations
          below use committed results.
        </p>
      )}
      <h2 id="reader-status-title">
        Ren · {status.backend === "reader" ? "Reader" : "Workspace classifier"}
      </h2>
      <p className="reader-identity">
        {status.model_id} · {status.device.toUpperCase()}
        {status.examples != null && ` · ${status.examples} examples`}
      </p>
      {status.revision && (
        <p className="reader-revision">Revision {status.revision}</p>
      )}
      {status.backend === "reader" && (
        <>
          <p role="status">
            {status.guarded
              ? "Automatic answers paused: spot-check guard active."
              : status.scope_matches && status.policy_complete
                ? "Frozen confidence and support checks active."
                : "Human confirmation required for this history."}
          </p>
          <dl className="reader-gate">
            <div>
              <dt>Confidence threshold</dt>
              <dd>{percent(status.threshold, 3)}</dd>
            </div>
            <div>
              <dt>Intent support floor</dt>
              <dd>{status.support_floor?.toFixed(2) ?? "Unavailable"}</dd>
            </div>
            <div>
              <dt>Temperature</dt>
              <dd>{status.temperature?.toFixed(4) ?? "Unavailable"}</dd>
            </div>
          </dl>
          <p>{status.scope_note}</p>
        </>
      )}
      {benchmark && (
        <details open>
          <summary>
            Published {benchmark.dataset} evaluation · full{" "}
            {benchmark.history_rows.toLocaleString()}-row history
          </summary>
          <dl className="reader-gate">
            <div>
              <dt>Test accuracy</dt>
              <dd>{percent(benchmark.accuracy, 1)}</dd>
            </div>
            <div>
              <dt>
                Share with support floor ·{" "}
                {benchmark.with_floor_sample.toLocaleString()} test messages
              </dt>
              <dd>{percent(benchmark.with_floor_coverage, 2)}</dd>
            </div>
            <div>
              <dt>Observed test precision with support floor</dt>
              <dd>{percent(benchmark.with_floor_precision, 1)}</dd>
            </div>
            <div>
              <dt>95% precision lower bound with support floor</dt>
              <dd>{percent(benchmark.with_floor_low95, 2)}</dd>
            </div>
          </dl>
          <p>
            Calibration labels were changed by a model audit, not a human audit
            ({benchmark.audited_calibration_labels} changed labels). Test labels
            are unchanged. {benchmark.with_floor_accepted} accepted answers,
            {benchmark.with_floor_errors} errors with the support floor.
          </p>
          <p>
            <a
              href={`https://github.com/Rish-it/secnd/blob/stage/c3-serving/${benchmark.floor_result_file}`}
            >
              Full test results with support floor
            </a>
          </p>
          <p>
            Recorded benchmark results. These measurements do not establish a
            universal 99% guarantee. These numbers do not measure this workspace
            or a smaller history.
          </p>
        </details>
      )}
      {error && <p role="status">{error}</p>}
    </section>
  );
}
