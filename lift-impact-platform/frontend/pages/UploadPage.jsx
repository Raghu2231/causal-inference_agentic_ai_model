export default function UploadPage({ onUpload, loading, backendReady, onRetryBackend }) {
  const handleChange = (event) => {
    if (!event.target.files?.length) {
      return;
    }
    onUpload(event.target.files[0]);
  };

  return (
    <div className="card">
      <h2>Upload Page</h2>
      <p>Select a .xlsx or .xls file to auto-detect schema and start the workflow.</p>
      {!backendReady && (
        <div className="warning">
          Backend is unreachable. Start API server at <code>http://localhost:8000</code> and click retry.
          <div>
            <button type="button" onClick={onRetryBackend} disabled={loading}>
              Retry backend check
            </button>
          </div>
        </div>
      )}
      <input type="file" accept=".xlsx,.xls" onChange={handleChange} disabled={loading || !backendReady} />
    </div>
  );
}
