import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS || 120000);
const UPLOAD_TIMEOUT_MS = Number(import.meta.env.VITE_UPLOAD_TIMEOUT_MS || 180000);

const api = axios.create({
  baseURL: BASE_URL,
  timeout: REQUEST_TIMEOUT_MS,
});

const extractError = (error) => {
  if (error.code === "ECONNABORTED") {
    return `Request timed out while connecting to ${BASE_URL}. If backend is running, increase timeout with VITE_API_TIMEOUT_MS / VITE_UPLOAD_TIMEOUT_MS.`;
  }
  if (error.response) {
    return `API error ${error.response.status}: ${JSON.stringify(error.response.data)}`;
  }
  if (error.request) {
    return `Could not connect to backend at ${BASE_URL}. Start API server with: uvicorn backend.api.main:app --reload --port 8000`;
  }
  return error.message || "Unexpected frontend request error";
};

export async function pingBackend() {
  try {
    const response = await api.get("/health", { timeout: 10000 });
    return response.data?.status === "ok";
  } catch {
    return false;
  }
}

export async function uploadExcel(file, onProgress) {
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const response = await api.post("/upload", form, {
      timeout: UPLOAD_TIMEOUT_MS,
      onUploadProgress: (evt) => {
        if (!onProgress) return;
        const total = evt.total || 0;
        const percent = total > 0 ? Math.round((evt.loaded / total) * 100) : 0;
        onProgress(percent);
      },
    });
    return response.data;
  } catch (error) {
    throw new Error(extractError(error));
  }
}

export async function fetchEda(fileId) {
  try {
    const response = await api.get(`/eda/${fileId}`);
    return response.data;
  } catch (error) {
    throw new Error(extractError(error));
  }
}

export async function runModel(fileId, scenarioMultiplier, isolateChannel) {
  try {
    const response = await api.post(`/run/${fileId}`, {
      scenario_multiplier: Number(scenarioMultiplier),
      isolate_channel: isolateChannel || null,
    });
    return response.data;
  } catch (error) {
    throw new Error(extractError(error));
  }
}
