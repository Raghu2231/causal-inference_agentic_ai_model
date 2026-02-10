import axios from "axios";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

const extractError = (error) => {
  if (error.code === "ECONNABORTED") {
    return `Request timed out while connecting to ${BASE_URL}`;
  }
  if (error.response) {
    return `API error ${error.response.status}: ${JSON.stringify(error.response.data)}`;
  }
  if (error.request) {
    return `Could not connect to backend at ${BASE_URL}. Start API server with: uvicorn backend.api.main:app --reload --port 8000`;
  }
  return error.message || "Unexpected frontend request error";
};

export async function uploadExcel(file) {
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    const response = await api.post("/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
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
