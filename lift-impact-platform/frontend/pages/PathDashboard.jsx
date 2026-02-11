import Plot from "react-plotly.js";

function toRows(obj) {
  if (!obj || typeof obj !== "object") return [];
  return Object.entries(obj).map(([key, value]) => ({ key, value }));
}

export default function PathDashboard({ summary }) {
  if (!summary) {
    return null;
  }

  const pathA = summary.path_a || {};
  const pathB = summary.path_b || {};
  const channelLift = pathA.lift_per_channel || {};
  const channelKeys = Object.keys(channelLift);

  return (
    <>
      <div className="card">
        <h2>Path A Dashboard</h2>
        {!!channelKeys.length && (
          <Plot
            data={[
              {
                x: channelKeys,
                y: channelKeys.map((key) => channelLift[key] ?? 0),
                type: "bar",
                marker: { color: "#2463eb" },
              },
            ]}
            layout={{ title: "Incremental Action Lift by Channel", autosize: true, height: 340 }}
            style={{ width: "100%" }}
            useResizeHandler
          />
        )}
        <details>
          <summary>Path A details</summary>
          <pre>{JSON.stringify(pathA, null, 2)}</pre>
        </details>
      </div>

      <div className="card">
        <h2>Path B Dashboard</h2>
        <div className="grid">
          <div className="metric">
            <div>Incremental TRX</div>
            <strong>{pathB.aggregated_incremental_trx ?? 0}</strong>
          </div>
          <div className="metric">
            <div>Incremental NBRX</div>
            <strong>{pathB.aggregated_incremental_nbrx ?? 0}</strong>
          </div>
        </div>

        <table className="lift-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {toRows(pathB).map((row) => (
              <tr key={row.key}>
                <td>{row.key}</td>
                <td>{typeof row.value === "object" ? JSON.stringify(row.value) : String(row.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
