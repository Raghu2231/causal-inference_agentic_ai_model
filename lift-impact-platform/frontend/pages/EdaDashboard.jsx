import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import KpiCard from "../components/KpiCard";

export default function EdaDashboard({ eda }) {
  const [selectedVariable, setSelectedVariable] = useState("");

  if (!eda) {
    return null;
  }

  const trends = eda.volume_trends || [];
  const xKey = trends.length ? Object.keys(trends[0])[0] : null;
  const yKeys = trends.length ? Object.keys(trends[0]).filter((key) => key !== xKey) : [];

  const profiles = eda.variable_profiles || {};
  const variableOptions = Object.keys(profiles);
  const selected = selectedVariable || variableOptions[0] || "";
  const stats = profiles[selected] || {};

  const feRows = useMemo(() => eda.feature_engineering_view || [], [eda]);

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

      {!!variableOptions.length && (
        <div className="card nested">
          <h3>Variable-level stats (mean, std, quartiles)</h3>
          <label>
            Select variable
            <select value={selected} onChange={(event) => setSelectedVariable(event.target.value)}>
              {variableOptions.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <div className="grid">
            <KpiCard label="Mean" value={(stats.mean ?? 0).toFixed(4)} />
            <KpiCard label="Std Dev" value={(stats.std ?? 0).toFixed(4)} />
            <KpiCard label="Q1" value={(stats.q25 ?? 0).toFixed(4)} />
            <KpiCard label="Median" value={(stats.median ?? 0).toFixed(4)} />
            <KpiCard label="Q3" value={(stats.q75 ?? 0).toFixed(4)} />
          </div>
          <Plot
            data={[
              {
                x: ["min", "q25", "median", "q75", "max"],
                y: [stats.min ?? 0, stats.q25 ?? 0, stats.median ?? 0, stats.q75 ?? 0, stats.max ?? 0],
                type: "bar",
                marker: { color: "#3f7bf8" },
                name: selected,
              },
            ]}
            layout={{ title: `Distribution checkpoints: ${selected}`, autosize: true, height: 300 }}
            style={{ width: "100%" }}
            useResizeHandler
          />
        </div>
      )}

      {!!feRows.length && (
        <div className="card nested">
          <h3>Feature-engineering view (adstock + treatment-effect ready signals)</h3>
          <table className="lift-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Q1</th>
                <th>Median</th>
                <th>Q3</th>
              </tr>
            </thead>
            <tbody>
              {feRows.map((row) => (
                <tr key={row.feature}>
                  <td>{row.feature}</td>
                  <td>{(row.mean ?? 0).toFixed(4)}</td>
                  <td>{(row.std ?? 0).toFixed(4)}</td>
                  <td>{(row.q25 ?? 0).toFixed(4)}</td>
                  <td>{(row.median ?? 0).toFixed(4)}</td>
                  <td>{(row.q75 ?? 0).toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
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
