import { useMemo, useState } from "react";

export default function UploadPage({ onUpload, loading, backendReady, onRetryBackend, uploadProgress }) {
  const [dragging, setDragging] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState("");

  const helperText = useMemo(() => {
    if (loading) {
      return `Uploading and validating file... ${uploadProgress}%`;
    }
    if (!backendReady) {
      return "Backend offline: start FastAPI server and retry health check.";
    }
    return "Drop an Excel file here or click to browse.";
  }, [backendReady, loading, uploadProgress]);

  const submitFile = (file) => {
    if (!file) return;
    setSelectedFileName(file.name);
    onUpload(file);
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    if (!backendReady || loading) {
      return;
    }
    const file = event.dataTransfer?.files?.[0];
    submitFile(file);
  };

  return (
    <div className="card">
      <h2>Upload Page</h2>
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

      <div
        className={`dropzone ${dragging ? "dropzone-active" : ""} ${!backendReady ? "dropzone-disabled" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          if (backendReady && !loading) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p className="dropzone-title">Excel Upload</p>
        <p>{helperText}</p>
        {selectedFileName && <p className="muted">Selected: {selectedFileName}</p>}
        {loading && (
          <div className="progress-wrap">
            <div className="progress-bar" style={{ width: `${uploadProgress}%` }} />
          </div>
        )}
        <input
          type="file"
          accept=".xlsx,.xls"
          onChange={(event) => submitFile(event.target.files?.[0])}
          disabled={loading || !backendReady}
        />
      </div>
    </div>
  );
}
