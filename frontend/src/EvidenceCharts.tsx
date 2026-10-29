import type { EvalCurves } from "./api";

const x = (value: number) => 46 + value * 350;
const y = (value: number) => 195 - value * 165;
const pct = (value: number) => `${(value * 100).toFixed(2)}%`;

function Axes({ xLabel, yLabel }: { xLabel: string; yLabel: string }) {
  return (
    <>
      {[0, 0.5, 1].map((value) => (
        <g key={value}>
          <path d={`M46 ${y(value)}H396`} stroke="#e6e9df" />
          <text x="8" y={y(value) + 4}>
            {value * 100}%
          </text>
          <text x={x(value)} y="215" textAnchor="middle">
            {value * 100}%
          </text>
        </g>
      ))}
      <text x="46" y="15">
        {yLabel}
      </text>
      <text x="221" y="238" textAnchor="middle">
        {xLabel}
      </text>
    </>
  );
}

export function ReliabilityChart({
  bins,
}: {
  bins: EvalCurves["reliability"];
}) {
  if (!bins.length)
    return <p className="chart-empty">No reliability bins measured yet.</p>;
  return (
    <figure>
      <svg
        viewBox="0 0 440 250"
        role="img"
        aria-label="Reliability: confidence compared with held-out accuracy"
      >
        <Axes xLabel="Model confidence" yLabel="Observed accuracy" />
        <path d="M46 195L396 30" stroke="#899681" strokeDasharray="5 5" />
        <polyline
          points={bins
            .map((row) => `${x(row.confidence)},${y(row.accuracy)}`)
            .join(" ")}
          fill="none"
          stroke="#587c48"
          strokeWidth="2"
        />
        {bins.map((row, i) => (
          <circle
            key={i}
            cx={x(row.confidence)}
            cy={y(row.accuracy)}
            r="4"
            fill="#587c48"
          >
            <title>
              {pct(row.confidence)} confidence · {pct(row.accuracy)} correct ·{" "}
              {row.count} items
            </title>
          </circle>
        ))}
      </svg>
      <figcaption>
        The dashed diagonal is perfect calibration. Each dot is a measured
        confidence bin.
      </figcaption>
      <details>
        <summary>Read reliability measurements</summary>
        <div className="chart-table">
          <table>
            <thead>
              <tr>
                <th>Confidence</th>
                <th>Accuracy</th>
                <th>Items</th>
              </tr>
            </thead>
            <tbody>
              {bins.map((row, i) => (
                <tr key={i}>
                  <td>{pct(row.confidence)}</td>
                  <td>{pct(row.accuracy)}</td>
                  <td>{row.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

export function TradeoffChart({
  points,
}: {
  points: EvalCurves["coverage_precision"];
}) {
  const observed = points.filter((row) => row.precision != null);
  if (!observed.length)
    return (
      <p className="chart-empty">No coverage–precision measurements yet.</p>
    );
  return (
    <figure>
      <svg
        viewBox="0 0 440 250"
        role="img"
        aria-label="Coverage and precision trade-off with the 99 percent target"
      >
        <Axes xLabel="Share answered automatically" yLabel="Precision" />
        <path d={`M46 ${y(0.99)}H396`} stroke="#b46f42" strokeDasharray="5 5" />
        <text x="394" y={y(0.99) + 15} textAnchor="end">
          99% target
        </text>
        <polyline
          points={observed
            .map((row) => `${x(row.coverage)},${y(row.precision!)}`)
            .join(" ")}
          fill="none"
          stroke="#587c48"
          strokeWidth="2"
        />
        {observed.map((row, i) => (
          <circle
            key={i}
            cx={x(row.coverage)}
            cy={y(row.precision!)}
            r="2.5"
            fill="#587c48"
          >
            <title>
              {pct(row.coverage)} coverage · {pct(row.precision!)} precision ·
              threshold {row.threshold}
            </title>
          </circle>
        ))}
      </svg>
      <figcaption>
        Held-out outcomes across thresholds. This curve is not a learning
        history or a certification rule.
      </figcaption>
      <details>
        <summary>Read threshold measurements</summary>
        <div className="chart-table">
          <table>
            <thead>
              <tr>
                <th>Threshold</th>
                <th>Coverage</th>
                <th>Precision</th>
              </tr>
            </thead>
            <tbody>
              {points.map((row, i) => (
                <tr key={i}>
                  <td>{row.threshold.toFixed(6)}</td>
                  <td>{pct(row.coverage)}</td>
                  <td>{row.precision == null ? "—" : pct(row.precision)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
