import { useMemo, useState } from "react";
import EdaDashboard from "./pages/EdaDashboard";
import PathDashboard from "./pages/PathDashboard";
import UploadPage from "./pages/UploadPage";
import { fetchEda, runModel, uploadExcel } from "./services/apiClient";

export default function App() {
  const [fileId, setFileId] = useState(null);
  const [schema, setSchema] = useState(null);
  const [eda, setEda] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [scenarioMultiplier, setScenarioMultiplier] = useState(1.0);
  const [isolateChannel, setIsolateChannel] = useState("");

  const actionOptions = useMemo(() => schema?.action_cols || [], [schema]);

  const handleUpload = async (file) => {
    setLoading(true);
    setError("");
    try {
      const payload = await uploadExcel(file);
      setFileId(payload.file_id);
      setSchema(payload.schema);
      setSummary(null);
      const edaPayload = await fetchEda(payload.file_id);
      setEda(edaPayload);
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
      <h1>Lift Impact Platform</h1>
      {error && <div className="error">{error}</div>}
      {fileId && <div className="success">Loaded file_id: {fileId}</div>}

      <UploadPage onUpload={handleUpload} loading={loading} />

      {schema && (
        <div className="card">
          <h2>Detected Schema</h2>
          <pre>{JSON.stringify(schema, null, 2)}</pre>
        </div>
      )}

      <EdaDashboard eda={eda} />

      {fileId && (
        <div className="card">
          <h2>Model Execution</h2>
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

      <PathDashboard summary={summary} />
    </div>
  );
}
