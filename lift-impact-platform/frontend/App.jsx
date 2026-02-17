import { useEffect, useMemo, useState } from "react";
import KpiCard from "./components/KpiCard";
import EdaDashboard from "./pages/EdaDashboard";
import PathDashboard from "./pages/PathDashboard";
import UploadPage from "./pages/UploadPage";
import { fetchEda, fetchInsights, fetchPreview, pingBackend, runModel, uploadExcel } from "./services/apiClient";

function downloadJson(filename, payload) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const TABS = ["Upload", "Schema & Variables", "EDA", "Modeling"];

export default function App() {
  const [activeTab, setActiveTab] = useState("Upload");
  const [fileId, setFileId] = useState(null);
  const [schema, setSchema] = useState(null);
  const [preview, setPreview] = useState(null);
  const [eda, setEda] = useState(null);
  const [summary, setSummary] = useState(null);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [scenarioMultiplier, setScenarioMultiplier] = useState(1.0);
  const [isolateChannel, setIsolateChannel] = useState("");
  const [backendReady, setBackendReady] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [edaLoading, setEdaLoading] = useState(false);

  useEffect(() => {
    pingBackend().then(setBackendReady);
  }, []);

  const actionOptions = useMemo(() => schema?.action_cols || [], [schema]);

  const handleUpload = async (file) => {
    setLoading(true);
    setError("");
    setUploadProgress(0);
    setSummary(null);
    setInsights(null);
    try {
      const ok = await pingBackend();
      setBackendReady(ok);
      if (!ok) {
        throw new Error("Backend is not reachable. Start backend and retry upload.");
      }
      const payload = await uploadExcel(file, setUploadProgress);
      setFileId(payload.file_id);
      setSchema(payload.schema);
      const previewPayload = await fetchPreview(payload.file_id);
      setPreview(previewPayload);
      setEda(null);
      setActiveTab("Schema & Variables");
    } catch (uploadError) {
      setError(uploadError.message);
    } finally {
      setLoading(false);
    }
  };



  const handleLoadEda = async () => {
    if (!fileId) return;
    setEdaLoading(true);
    setError("");
    try {
      const edaPayload = await fetchEda(fileId);
      setEda(edaPayload);
      setActiveTab("EDA");
    } catch (edaError) {
      setError(edaError.message);
    } finally {
      setEdaLoading(false);
    }
  };

  const handleGenerateInsights = async () => {
    if (!fileId || !summary) return;
    setLoading(true);
    setError("");
    try {
      const insightPayload = await fetchInsights(fileId, summary, "Generate tactical insights for field execution.");
      setInsights(insightPayload);
    } catch (insightError) {
      setError(insightError.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRunModel = async () => {
    if (!fileId) return;
    setLoading(true);
    setError("");
    try {
      const runPayload = await runModel(fileId, scenarioMultiplier, isolateChannel || null);
      setSummary(runPayload.summary);
      setInsights(null);
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
          <h1>Causal Lift Workflow Studio</h1>
          <p className="muted">Upload → auto-detect schema → classify variables → EDA → Path A/Path B modeling.</p>
        </div>
        <div className={`status-pill ${backendReady ? "status-online" : "status-offline"}`}>
          {backendReady ? "Backend connected" : "Backend offline"}
        </div>
      </header>

      <div className="tabbar card">
        {TABS.map((tab) => (
          <button
            key={tab}
            className={`tab-btn ${activeTab === tab ? "tab-btn-active" : ""}`}
            onClick={() => setActiveTab(tab)}
            disabled={tab !== "Upload" && !fileId && tab !== "Modeling"}
            type="button"
          >
            {tab}
          </button>
        ))}
      </div>

      {error && <div className="error">{error}</div>}
      {fileId && <div className="success">Loaded file_id: {fileId}</div>}

      {activeTab === "Upload" && (
        <UploadPage
          onUpload={handleUpload}
          loading={loading}
          backendReady={backendReady}
          uploadProgress={uploadProgress}
          onRetryBackend={async () => setBackendReady(await pingBackend())}
        />
      )}

      {activeTab === "Schema & Variables" && preview && (
        <div className="card">
          <h2>Detected Schema & Variable Classification</h2>
          {!!preview.schema?.warnings?.length && (
            <div className="warning">
              <strong>Auto-detection notes:</strong>
              <ul>
                {preview.schema.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="grid">
            <KpiCard label="Rows" value={String(preview.shape?.rows ?? 0)} />
            <KpiCard label="Columns" value={String(preview.shape?.columns ?? 0)} />
            <KpiCard label="Treatments" value={String(preview.groups?.treatments?.length ?? 0)} />
            <KpiCard label="Confounders" value={String(preview.groups?.confounders?.length ?? 0)} />
          </div>
          <div className="grid">
            <div className="metric"><h4>Treatments</h4><pre>{JSON.stringify(preview.groups?.treatments || [], null, 2)}</pre></div>
            <div className="metric"><h4>Covariates</h4><pre>{JSON.stringify(preview.groups?.covariates || [], null, 2)}</pre></div>
            <div className="metric"><h4>Confounders</h4><pre>{JSON.stringify(preview.groups?.confounders || [], null, 2)}</pre></div>
          </div>
          <details>
            <summary>Sample data preview (first 20 rows)</summary>
            <pre>{JSON.stringify(preview.sample, null, 2)}</pre>
          </details>
          <details>
            <summary>Schema JSON</summary>
            <pre>{JSON.stringify(preview.schema, null, 2)}</pre>
          </details>
          <div className="btn-row">
            <button onClick={handleLoadEda} disabled={edaLoading}>
              {edaLoading ? "Loading EDA..." : "Load EDA Summary"}
            </button>
          </div>
        </div>
      )}

      {activeTab === "EDA" && (
        <>
          {!eda && (
            <div className="card">
              <h2>EDA Dashboard</h2>
              <p>EDA has not been loaded yet for this file.</p>
              <button onClick={handleLoadEda} disabled={edaLoading}>
                {edaLoading ? "Loading EDA..." : "Load EDA Summary"}
              </button>
            </div>
          )}
          <EdaDashboard eda={eda} />
        </>
      )}

      {activeTab === "Modeling" && fileId && (
        <>
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
          {summary && (
            <div className="card">
              <h2>Downloadable Final Summary</h2>
              <div className="btn-row">
                <button onClick={() => downloadJson("lift_summary.json", summary)}>Download Lift Summary JSON</button>
                <button onClick={handleGenerateInsights} disabled={loading}>Generate AI Insights</button>
              </div>
            </div>
          )}
          {insights && (
            <div className="card">
              <h2>Automated Insights ({insights.source})</h2>
              <p>{insights.narrative}</p>
              <ul>
                {(insights.bullets || []).map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
            </div>
          )}
          <PathDashboard summary={summary} />
        </>
      )}

      {eda && (
        <div className="card">
          <h2>Downloadable EDA Summary</h2>
          <button onClick={() => downloadJson("eda_summary.json", eda)}>Download EDA JSON</button>
        </div>
      )}
    </div>
  );
}
