import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS || 120000);
const UPLOAD_TIMEOUT_MS = Number(import.meta.env.VITE_UPLOAD_TIMEOUT_MS || 180000);
const EDA_TIMEOUT_MS = Number(import.meta.env.VITE_EDA_TIMEOUT_MS || 300000);
const RETRY_COUNT = Number(import.meta.env.VITE_API_RETRY_COUNT || 3);
const RETRY_DELAY_MS = Number(import.meta.env.VITE_API_RETRY_DELAY_MS || 1200);

const api = axios.create({
  baseURL: BASE_URL,
  timeout: REQUEST_TIMEOUT_MS,
});

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const isRetryableNetworkError = (error) => {
  if (!error) return false;
  if (error.code === "ECONNABORTED") return true;
  if (error.code === "ERR_NETWORK") return true;
  if (error.code === "ECONNREFUSED") return true;
  return Boolean(error.request && !error.response);
};

async function withRetry(fn, attempts = RETRY_COUNT) {
  let lastError;
  for (let i = 0; i < attempts; i += 1) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (!isRetryableNetworkError(error) || i === attempts - 1) {
        throw error;
      }
      await sleep(RETRY_DELAY_MS * (i + 1));
    }
  }
  throw lastError;
}

const extractError = (error) => {
  if (error.code === "ECONNABORTED") {
    return `Request timed out while connecting to ${BASE_URL}. If backend is running, increase timeout with VITE_API_TIMEOUT_MS / VITE_UPLOAD_TIMEOUT_MS / VITE_EDA_TIMEOUT_MS.`;
  }
  if (error.response) {
    return `API error ${error.response.status}: ${JSON.stringify(error.response.data)}`;
  }
  if (error.request) {
    return `Could not connect to backend at ${BASE_URL}. Backend may be down or may have crashed during processing. Start API server with: uvicorn backend.api.main:app --reload --port 8000 and check backend terminal logs for traceback.`;
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
    const response = await withRetry(() =>
      api.post("/upload", form, {
        timeout: UPLOAD_TIMEOUT_MS,
        onUploadProgress: (evt) => {
          if (!onProgress) return;
          const total = evt.total || 0;
          const percent = total > 0 ? Math.round((evt.loaded / total) * 100) : 0;
          onProgress(percent);
        },
      }),
    );
    return response.data;
  } catch (error) {
    throw new Error(extractError(error));
  }
}

export async function fetchPreview(fileId) {
  try {
    const response = await withRetry(() => api.get(`/preview/${fileId}`));
    return response.data;
  } catch (error) {
    throw new Error(extractError(error));
  }
}

export async function fetchEda(fileId) {
  try {
    const response = await withRetry(() => api.get(`/eda/${fileId}`, { timeout: EDA_TIMEOUT_MS }));
    return response.data;
  } catch (error) {
    throw new Error(extractError(error));
  }
}

export async function runModel(fileId, scenarioMultiplier, isolateChannel) {
  try {
    const response = await withRetry(() =>
      api.post(`/run/${fileId}`, {
        scenario_multiplier: Number(scenarioMultiplier),
        isolate_channel: isolateChannel || null,
      }),
    );
    return response.data;
  } catch (error) {
    throw new Error(extractError(error));
  }
}

export async function fetchInsights(fileId, summary, context = "") {
  try {
    const response = await withRetry(() => api.post(`/insights/${fileId}`, { summary, context }));
    return response.data;
  } catch (error) {
    throw new Error(extractError(error));
  }
}
