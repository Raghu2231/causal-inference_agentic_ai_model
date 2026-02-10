import Plot from "react-plotly.js";
import KpiCard from "../components/KpiCard";

export default function EdaDashboard({ eda }) {
  if (!eda) {
    return null;
  }

  const trends = eda.volume_trends || [];
  const xKey = trends.length ? Object.keys(trends[0])[0] : null;
  const yKeys = trends.length ? Object.keys(trends[0]).filter((key) => key !== xKey) : [];

  return (
    <div className="card">
      <h2>EDA Dashboard</h2>
      <div className="grid">
        <KpiCard label="Action uptake rate" value={`${(eda.action_uptake_rate * 100).toFixed(2)}%`} />
        <KpiCard
          label="Suggestion → Action conversion"
          value={`${(eda.suggestion_to_action_conversion * 100).toFixed(2)}%`}
        />
      </div>
      {!!trends.length && (
        <Plot
          data={yKeys.map((key) => ({
            x: trends.map((item) => item[xKey]),
            y: trends.map((item) => item[key]),
            mode: "lines+markers",
            type: "scatter",
            name: key,
          }))}
          layout={{ title: "Volume Trends", autosize: true, height: 360 }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      )}
      <details>
        <summary>Missing values</summary>
        <pre>{JSON.stringify(eda.missing_values, null, 2)}</pre>
      </details>
      <details>
        <summary>Distribution checks</summary>
        <pre>{JSON.stringify(eda.distributions, null, 2)}</pre>
      </details>
      <details>
        <summary>Correlation heatmap matrix</summary>
        <pre>{JSON.stringify(eda.correlation_heatmap, null, 2)}</pre>
      </details>
    </div>
  );
}
