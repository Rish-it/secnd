import ReaderStatus from "./ReaderStatus";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, percent } from "./api";
import type { Dataset, Metrics as WorkspaceMetrics, ModelVersion } from "./api";

import { ReliabilityChart, TradeoffChart } from "./EvidenceCharts";

const measuredPercent = (value: number | null | undefined) =>
  value == null ? "—" : `${(value * 100).toFixed(2)}%`;

export default function Metrics({ refresh }: { refresh: number }) {
  const [dataset, setDataset] = useState<Dataset | "workspace">("workspace");
  const requestId = useRef(0);
  const [versionsError, setVersionsError] = useState("");
  const [metrics, setMetrics] = useState<WorkspaceMetrics | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [fixture, setFixture] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    const request = ++requestId.current;
    try {
      const [result, models] = await Promise.all([
        api.metrics(dataset === "workspace" ? undefined : dataset),
        api.models().catch(() => null),
      ]);
      if (request !== requestId.current) return;
      setMetrics(result.data);
      setVersions(models?.data.items ?? []);
      setVersionsError(
        models
          ? ""
          : "Workspace versions are unavailable. Published evaluations are still shown.",
      );
      setFixture(result.fixture);
      setError("");
    } catch (cause) {
      if (request !== requestId.current) return;
      setError(
        cause instanceof Error ? cause.message : "Could not load your metrics.",
      );
    } finally {
      if (request === requestId.current) setLoading(false);
    }
  }, [dataset]);
  useEffect(() => {
    void load();
    const timer = window.setInterval(() => void load(), 2000);
    return () => window.clearInterval(timer);
  }, [load, refresh]);
  async function rollback(version: string) {
    setBusy(true);
    setNotice("");
    setError("");
    try {
      const result = await api.rollback(version);
      setNotice(
        result.fixture
          ? "Example rollback checked. No model version was changed."
          : `Restored ${result.data.model_version}.`,
      );
      await load();
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not restore this version.",
      );
    } finally {
      setBusy(false);
    }
  }
  const selected = dataset === "workspace" || metrics?.dataset === dataset;
  if (loading && (!metrics || !selected))
    return (
      <p role="status" className="metrics-status">
        Gathering the evidence…
      </p>
    );
  if (!metrics || !selected)
    return (
      <div className="empty-card" role="alert">
        <h2>Metrics are taking a moment.</h2>
        <p>{error}</p>
        <button
          className="primary-button"
          onClick={() => {
            setLoading(true);
            void load();
          }}
        >
          Try again
        </button>
      </div>
    );
  const card = metrics.card;
  const history = metrics.history;
  const curves = metrics.curves ?? { reliability: [], coverage_precision: [] };
  const missedTarget = card.precision != null && card.precision < 0.99;
  const points = history
    .map(
      (row, index) =>
        `${30 + (index / Math.max(1, history.length - 1)) * 380},${148 - row.coverage * 125}`,
    )
    .join(" ");
  const readerWorkspace =
    dataset === "workspace" && card.model_version.startsWith("ren-reader-");
  const datasetPicker = (
    <section className="dataset-picker" aria-label="Evaluation dataset">
      <div>
        <p className="eyebrow">PUBLISHED EVALUATIONS</p>
        <p>
          {metrics.dataset
            ? "Fixed held-out test sets · 2,000 training labels · seed 0"
            : readerWorkspace
              ? `Workspace history · ${metrics.labels} rows`
              : `Fixed workspace calibration · ${metrics.labels} training labels`}
        </p>
      </div>
      <div className="dataset-buttons">
        {(["workspace", "mtop", "banking77", "ipsa"] as const).map((name) => (
          <button
            key={name}
            aria-pressed={dataset === name}
            onClick={() => {
              if (name === dataset) return;
              setLoading(true);
              setDataset(name);
            }}
          >
            {name.toUpperCase()}
          </button>
        ))}
      </div>
    </section>
  );
  if (readerWorkspace)
    return (
      <div className="metrics-workspace">
        <ReaderStatus />
        {datasetPicker}
        <p className="evidence-note">
          Confirmed answers update the reader history. Published results above
          are separate from workspace measurements.
        </p>
      </div>
    );
  return (
    <div className="metrics-workspace">
      <ReaderStatus />
      {datasetPicker}
      {metrics.dataset && (
        <p className="evidence-note">
          Recorded {metrics.dataset.toUpperCase()} evaluation. These numbers
          describe a benchmark, not your uploaded workspace.
          {card.featurizer_id && ` Encoder/head: ${card.featurizer_id}.`}
        </p>
      )}
      {fixture && (
        <p className="metrics-preview">
          Preview workspace · No trained model or measured workspace results
          yet.
        </p>
      )}
      {metrics.guarded && (
        <p role="status" className="gate-warning">
          Guarded: automatic answers paused. Spot-checks show precision below
          the target; every new item goes to a person until a new version
          passes.
        </p>
      )}
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className="answer-notice" role="status">
          {notice}
        </p>
      )}
      <section className="gate-summary">
        <div>
          <p className="eyebrow">REN / THE PRECISION GATE</p>
          <h2>
            {metrics.guarded
              ? "Automatic answers paused (guarded)."
              : card.coverage === 0
                ? "Not certified yet. Answers nothing yet."
                : `${measuredPercent(card.coverage)} answered automatically${missedTarget ? "; precision target missed" : ""}.`}
          </h2>
          <p>
            {card.threshold == null
              ? "There is not enough held-out evidence to certify automatic answers. Until there is, every decision stays with you."
              : "Decisions below the certified threshold come back to you. Each model version is checked before release."}
          </p>
        </div>
        <div className="gate-target">
          <strong>99%</strong>
          <span>precision target</span>
          <small>A target requires evidence.</small>
        </div>
      </section>
      {missedTarget && (
        <p role="status" className="gate-warning">
          The fixed test precision is {measuredPercent(card.precision)}, below
          the 99% target. This result does not meet that target, even though the
          re-split pass rate clears 95%.
        </p>
      )}
      <div className="metric-grid">
        {[
          {
            label: "Coverage",
            value: measuredPercent(metrics.guarded ? 0 : card.coverage),
            note: "Share answered automatically",
          },
          {
            label: "Automatic precision",
            value: measuredPercent(metrics.guarded ? null : card.precision),
            note:
              card.precision == null
                ? "No measured automatic traffic"
                : "Correct answers above the threshold",
          },
          {
            label: "Calibration error",
            value: percent(card.ece),
            note: "Confidence compared with outcomes",
          },
          {
            label: "Cost per 1,000",
            value:
              card.cost_per_1000_usd == null
                ? "—"
                : `$${card.cost_per_1000_usd.toFixed(4)}`,
            note: "Not measured until the cost run is ready",
          },
        ].map((tile) => (
          <section className="metric-card" key={tile.label}>
            <h2>{tile.label}</h2>
            <div className="metric-value">{tile.value}</div>
            <p>{tile.note}</p>
          </section>
        ))}
      </div>
      <div className="evidence-grid">
        <section className="evidence-card">
          <p className="eyebrow">GETTING A LITTLE MORE HEADSPACE</p>
          <h3>Coverage over time.</h3>
          {history.length ? (
            <figure>
              <svg
                viewBox="0 0 440 190"
                role="img"
                aria-label={`Coverage over ${history.length} model updates`}
              >
                <path d="M30 20V148H415" fill="none" stroke="#ced5c6" />
                <text x="0" y="24">
                  100%
                </text>
                <text x="7" y="151">
                  0%
                </text>
                <polyline
                  points={points}
                  fill="none"
                  stroke="#587c48"
                  strokeWidth="3"
                />
                {history.map((row, i) => (
                  <circle
                    key={`${row.timestamp}-${i}`}
                    cx={30 + (i / Math.max(1, history.length - 1)) * 380}
                    cy={148 - row.coverage * 125}
                    r="4"
                    fill="#587c48"
                  >
                    <title>
                      {row.model_version}: {percent(row.coverage)}
                    </title>
                  </circle>
                ))}
                <text x="30" y="178">
                  First version
                </text>
                <text x="341" y="178">
                  Latest version
                </text>
              </svg>
              <figcaption>
                Measured model updates. Answer counts will appear with the
                learning history.
              </figcaption>
            </figure>
          ) : (
            <div className="chart-empty">
              <span className="empty-chart-lines" aria-hidden="true" />
              <p>Your first measured learning curve belongs here.</p>
              <small>
                Answering a question will not always increase coverage.
              </small>
            </div>
          )}
        </section>
        <section className="evidence-card">
          <p className="eyebrow">CONFIDENCE THAT EARNS ITS KEEP</p>
          <h3>Reliability.</h3>
          <ReliabilityChart bins={curves.reliability} />
        </section>
      </div>
      <section className="evidence-card">
        <p className="eyebrow">HOW MUCH CAN IT HANDLE?</p>
        <h3>Coverage and precision.</h3>
        <TradeoffChart points={curves.coverage_precision} />
      </section>
      <section className="evidence-card handoff-card">
        <div className="section-heading">
          <div>
            <p className="eyebrow">TAUGHT BY MUSE. LEARNED BY REN.</p>
            <h3>The handoff.</h3>
          </div>
          <div className="chart-legend">
            <span className="legend-muse">Muse</span>
            <span className="legend-ren">Ren auto</span>
            <span className="legend-human">You</span>
          </div>
        </div>
        <div className="chart-empty">
          <p>No measured handoff run yet.</p>
          <small>
            This chart will show the share of decisions made by Muse, Ren and
            you as human answers arrive. A cold start can begin with zero
            automatic coverage.
          </small>
        </div>
      </section>
      <div className="evidence-grid">
        <section className="evidence-card">
          <p className="eyebrow">THE CLAIM, WITH ITS EVIDENCE</p>
          <h3>Precision above the threshold.</h3>
          <dl className="bound-stats">
            <div>
              <dt>Threshold</dt>
              <dd>{percent(card.threshold)}</dd>
            </div>
            <div>
              <dt>Test precision</dt>
              <dd>{measuredPercent(card.precision)}</dd>
            </div>
            <div>
              <dt>Calibration lower bound</dt>
              <dd>{measuredPercent(card.cp_lower_bound)}</dd>
            </div>
            <div>
              <dt>Re-split pass rate</dt>
              <dd>{measuredPercent(card.robustness_pass_rate)}</dd>
            </div>
          </dl>
          <p className="evidence-note">
            The lower bound uses 95% confidence. Roughly 300 clean held-out
            answers above the threshold are needed before even a zero-error
            sample can certify 99% precision.
          </p>
          {card.robustness_pass_rate != null &&
            card.robustness_pass_rate < 0.95 && (
              <p className="gate-warning">
                The measured re-split pass rate is below the 95% release target.
              </p>
            )}
        </section>
        <section className="evidence-card">
          <p className="eyebrow">A RECORD OF EVERY RELEASE</p>
          <h3>Workspace model versions.</h3>
          {versionsError && (
            <p role="status" className="gate-warning">
              {versionsError}
            </p>
          )}
          <ul className="model-versions">
            {versions.map((version) => (
              <li key={version.model_version}>
                <div>
                  <strong>
                    {fixture
                      ? "Preview · no trained version"
                      : version.model_version}
                  </strong>
                  <small>
                    {percent(version.card.coverage)} coverage{" "}
                    {version.current ? "· current" : ""}
                  </small>
                </div>
                {!version.current && (
                  <button
                    disabled={busy}
                    onClick={() => void rollback(version.model_version)}
                  >
                    {busy ? "Restoring…" : "Restore version"}
                  </button>
                )}
              </li>
            ))}
          </ul>
          {!versions.length && (
            <p className="evidence-note">
              Your first approved model version will appear here.
            </p>
          )}
          <button
            className="text-button"
            disabled={loading}
            onClick={() => {
              setLoading(true);
              void load();
            }}
          >
            {loading ? "Refreshing…" : "Refresh measurements"}
          </button>
        </section>
      </div>
    </div>
  );
}
