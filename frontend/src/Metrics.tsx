import { useCallback, useEffect, useState } from 'react';
import { api, percent } from './api';
import type { Metrics as WorkspaceMetrics, ModelVersion } from './api';

export default function Metrics({ refresh }: { refresh: number }) {
  const [metrics, setMetrics] = useState<WorkspaceMetrics | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [fixture, setFixture] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);
  const load = useCallback(async () => {
    try {
      const [result, models] = await Promise.all([api.metrics(), api.models()]);
      setMetrics(result.data); setVersions(models.data.items);
      setFixture(result.fixture || models.fixture); setError('');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not load your metrics.'); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void load(); }, [load, refresh]);
  async function rollback(version: string) {
    setBusy(true); setNotice(''); setError('');
    try {
      const result = await api.rollback(version);
      setNotice(result.fixture ? 'Example rollback checked. No model version was changed.' : `Restored ${result.data.model_version}.`);
      await load();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not restore this version.'); }
    finally { setBusy(false); }
  }
  if (loading && !metrics) return <p role="status" className="metrics-status">Gathering the evidence…</p>;
  if (!metrics) return <div className="empty-card" role="alert"><h2>Metrics are taking a moment.</h2><p>{error}</p><button className="primary-button" onClick={() => { setLoading(true); void load(); }}>Try again</button></div>;
  const card = metrics.card;
  const history = metrics.history;
  const points = history.map((row, index) => `${30 + index / Math.max(1, history.length - 1) * 380},${148 - row.coverage * 125}`).join(' ');
  return <div className="metrics-workspace">
    {fixture && <p className="metrics-preview">Preview workspace · No trained model or measured workspace results yet.</p>}
    {error && <p className="error-message" role="alert">{error}</p>}
    {notice && <p className="answer-notice" role="status">{notice}</p>}
    <section className="gate-summary"><div><p className="eyebrow">REN / THE PRECISION GATE</p><h2>{card.coverage === 0 ? 'Answers nothing yet.' : `${percent(card.coverage)} handled on its own.`}</h2><p>{card.threshold == null ? 'There is not enough held-out evidence to certify automatic answers. Until there is, every decision stays with you.' : 'Decisions below the certified threshold come back to you. Each model version is checked before release.'}</p></div><div className="gate-target"><strong>99%</strong><span>precision target</span><small>A target requires evidence.</small></div></section>
    <div className="metric-grid">{[
      { label: 'Coverage', value: percent(card.coverage), note: 'Share answered automatically' },
      { label: 'Automatic precision', value: percent(card.precision), note: card.precision == null ? 'No measured automatic traffic' : 'Correct answers above the threshold' },
      { label: 'Calibration error', value: percent(card.ece), note: 'Confidence compared with outcomes' },
      { label: 'Cost per 1,000', value: card.cost_per_1000_usd == null ? '—' : `$${card.cost_per_1000_usd.toFixed(4)}`, note: 'Not measured until the cost run is ready' },
    ].map(tile => <section className="metric-card" key={tile.label}><h2>{tile.label}</h2><div className="metric-value">{tile.value}</div><p>{tile.note}</p></section>)}</div>
    <div className="evidence-grid">
      <section className="evidence-card"><p className="eyebrow">GETTING A LITTLE MORE HEADSPACE</p><h3>Coverage over time.</h3>{history.length ? <figure><svg viewBox="0 0 440 190" role="img" aria-label={`Coverage over ${history.length} model updates`}><path d="M30 20V148H415" fill="none" stroke="#ced5c6" /><text x="0" y="24">100%</text><text x="7" y="151">0%</text><polyline points={points} fill="none" stroke="#587c48" strokeWidth="3" />{history.map((row, i) => <circle key={`${row.timestamp}-${i}`} cx={30 + i / Math.max(1, history.length - 1) * 380} cy={148 - row.coverage * 125} r="4" fill="#587c48"><title>{row.model_version}: {percent(row.coverage)}</title></circle>)}<text x="30" y="178">First version</text><text x="341" y="178">Latest version</text></svg><figcaption>Measured model updates. Answer counts will appear with the learning history.</figcaption></figure> : <div className="chart-empty"><span className="empty-chart-lines" aria-hidden="true" /><p>Your first measured learning curve belongs here.</p><small>Answering a question will not always increase coverage.</small></div>}</section>
      <section className="evidence-card"><p className="eyebrow">CONFIDENCE THAT EARNS ITS KEEP</p><h3>Reliability.</h3><div className="chart-empty"><span className="reliability-empty" aria-hidden="true">↗</span><p>No reliability bins measured yet.</p><small>When confidence says 80%, roughly 80 in 100 should be right. The diagram will compare this with held-out outcomes.</small></div></section>
    </div>
    <section className="evidence-card handoff-card"><div className="section-heading"><div><p className="eyebrow">TAUGHT BY MUSE. LEARNED BY REN.</p><h3>The handoff.</h3></div><div className="chart-legend"><span className="legend-muse">Muse</span><span className="legend-ren">Ren auto</span><span className="legend-human">You</span></div></div><div className="chart-empty"><p>No measured handoff run yet.</p><small>This chart will show the share of decisions made by Muse, Ren and you as human answers arrive. A cold start can begin with zero automatic coverage.</small></div></section>
    <div className="evidence-grid">
      <section className="evidence-card"><p className="eyebrow">THE CLAIM, WITH ITS EVIDENCE</p><h3>Precision above the threshold.</h3><dl className="bound-stats"><div><dt>Threshold</dt><dd>{percent(card.threshold)}</dd></div><div><dt>Test precision</dt><dd>{percent(card.precision)}</dd></div><div><dt>Calibration lower bound</dt><dd>{percent(card.cp_lower_bound)}</dd></div><div><dt>Re-split pass rate</dt><dd>{percent(card.robustness_pass_rate)}</dd></div></dl><p className="evidence-note">The lower bound uses 95% confidence. Roughly 300 clean held-out answers above the threshold are needed before even a zero-error sample can certify 99% precision.</p>{card.robustness_pass_rate != null && card.robustness_pass_rate < 0.95 && <p className="gate-warning">The measured re-split pass rate is below the 95% release target.</p>}</section>
      <section className="evidence-card"><p className="eyebrow">A RECORD OF EVERY RELEASE</p><h3>Model versions.</h3><ul className="model-versions">{versions.map(version => <li key={version.model_version}><div><strong>{fixture ? 'Preview · no trained version' : version.model_version}</strong><small>{percent(version.card.coverage)} coverage {version.current ? '· current' : ''}</small></div>{!version.current && <button disabled={busy} onClick={() => void rollback(version.model_version)}>{busy ? 'Restoring…' : 'Restore version'}</button>}</li>)}</ul>{!versions.length && <p className="evidence-note">Your first approved model version will appear here.</p>}<button className="text-button" disabled={loading} onClick={() => { setLoading(true); void load(); }}>{loading ? 'Refreshing…' : 'Refresh measurements'}</button></section>
    </div>
  </div>;
}
