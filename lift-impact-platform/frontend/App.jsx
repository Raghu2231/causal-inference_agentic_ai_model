import { useEffect, useMemo, useState } from "react";
import KpiCard from "./components/KpiCard";
import EdaDashboard from "./pages/EdaDashboard";
import PathDashboard from "./pages/PathDashboard";
import UploadPage from "./pages/UploadPage";
import { fetchEda, pingBackend, runModel, uploadExcel } from "./services/apiClient";

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const STEPS = ["Upload", "EDA", "Run Models", "Review Lift"];

export default function App() {
  const [fileId, setFileId] = useState(null);
  const [schema, setSchema] = useState(null);
  const [eda, setEda] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [scenarioMultiplier, setScenarioMultiplier] = useState(1.0);
  const [isolateChannel, setIsolateChannel] = useState("");
  const [backendReady, setBackendReady] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      const ok = await pingBackend();
      if (mounted) {
        setBackendReady(ok);
      }
    };
    check();
    const intervalId = setInterval(check, 10000);
    return () => {
      mounted = false;
      clearInterval(intervalId);
    };
  }, []);

  const actionOptions = useMemo(() => schema?.action_cols || [], [schema]);
  const currentStep = summary ? 3 : fileId ? 2 : eda ? 1 : 0;

  const handleUpload = async (file) => {
    setLoading(true);
    setError("");
    setUploadProgress(0);
    setSummary(null);
    try {
      const ok = await pingBackend();
      setBackendReady(ok);
      if (!ok) {
        throw new Error("Backend is not reachable. Start backend and retry upload.");
      }
      const payload = await uploadExcel(file, (value) => setUploadProgress(value));
      setFileId(payload.file_id);
      setSchema(payload.schema);
      const edaPayload = await fetchEda(payload.file_id);
      setEda(edaPayload);
      setUploadProgress(100);
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunModel = async () => {
    if (!fileId) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const runPayload = await runModel(fileId, scenarioMultiplier, isolateChannel || null);
      setSummary(runPayload.summary);
    } catch (runError) {
      setError(runError.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header className="hero card">
        <div>
          <h1>Lift Impact Platform</h1>
          <p className="muted">Suggestions → Actions → Outcomes with Path A and Path B causal lift modeling.</p>
        </div>
        <div className={`status-pill ${backendReady ? "status-online" : "status-offline"}`}>
          {backendReady ? "Backend connected" : "Backend offline"}
        </div>
      </header>

      <div className="stepper card">
        {STEPS.map((label, index) => (
          <div key={label} className={`step ${index <= currentStep ? "step-active" : ""}`}>
            <span>{index + 1}</span>
            <p>{label}</p>
          </div>
        ))}
      </div>

      {error && <div className="error">{error}</div>}
      {fileId && <div className="success">Loaded file_id: {fileId}</div>}

      <UploadPage
        onUpload={handleUpload}
        loading={loading}
        backendReady={backendReady}
        uploadProgress={uploadProgress}
        onRetryBackend={async () => setBackendReady(await pingBackend())}
      />

      {schema && (
        <div className="card">
          <h2>Detected Schema</h2>
          {!!schema.warnings?.length && (
            <div className="warning">
              <strong>Auto-detection notes:</strong>
              <ul>
                {schema.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
          <pre>{JSON.stringify(schema, null, 2)}</pre>
        </div>
      )}

      <EdaDashboard eda={eda} />

      {eda && (
        <div className="card">
          <h2>Downloadable EDA Summary</h2>
          <button onClick={() => downloadJson("eda_summary.json", eda)}>Download EDA JSON</button>
        </div>
      )}

      {fileId && (
        <div className="card">
          <h2>Model Execution</h2>
          <div className="grid">
            <KpiCard label="Scenario multiplier" value={String(scenarioMultiplier)} />
            <KpiCard label="Isolated channel" value={isolateChannel || "All channels"} />
          </div>
          <label>
            Scenario multiplier
            <input
              type="number"
              value={scenarioMultiplier}
              min="0"
              step="0.1"
              max="3"
              onChange={(event) => setScenarioMultiplier(event.target.value)}
            />
          </label>
          <label>
            Isolate channel (optional)
            <select value={isolateChannel} onChange={(event) => setIsolateChannel(event.target.value)}>
              <option value="">All channels</option>
              {actionOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </label>
          <button onClick={handleRunModel} disabled={loading}>
            Run Path A + Path B
          </button>
        </div>
      )}

      {summary && (
        <div className="card">
          <h2>Downloadable Final Summary</h2>
          <button onClick={() => downloadJson("lift_summary.json", summary)}>Download Lift Summary JSON</button>
        </div>
      )}

      <PathDashboard summary={summary} />
    </div>
  );
}
