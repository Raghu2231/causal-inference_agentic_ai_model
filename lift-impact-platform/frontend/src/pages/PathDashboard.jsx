export default function PathDashboard({ summary }) {
  if (!summary) {
    return null;
  }

  return (
    <>
      <div className="card">
        <h2>Path A Dashboard</h2>
        <pre>{JSON.stringify(summary.path_a, null, 2)}</pre>
      </div>
      <div className="card">
        <h2>Path B Dashboard</h2>
        <pre>{JSON.stringify(summary.path_b, null, 2)}</pre>
      </div>
      <div className="card">
        <h2>Final Lift Summary</h2>
        <div className="grid">
          <div className="metric">
            <div>Total incremental TRX</div>
            <strong>{summary.path_b.aggregated_incremental_trx}</strong>
          </div>
          <div className="metric">
            <div>Total incremental NBRX</div>
            <strong>{summary.path_b.aggregated_incremental_nbrx}</strong>
          </div>
        </div>
      </div>
    </>
  );
}
