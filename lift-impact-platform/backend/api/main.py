from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.eda.analyzer import run_eda
from backend.services.lift_engine import LiftComputationEngine
from backend.services.session_store import DATASETS, EDA_CACHE
from backend.utils.checklist import build_variable_checklist
from backend.utils.schema_detection import detect_schema, schema_to_dict
from data_contracts.contracts import RunRequest

app = FastAPI(title="Pharma Causal Lift Model API")
engine = LiftComputationEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
ASSETS_DIR = DIST_DIR / "assets"
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


def _frontend_help_page() -> HTMLResponse:
    instructions = """
    <html>
      <head><title>Pharma Causal Lift Model</title></head>
      <body style="font-family: Arial, sans-serif; padding: 24px; background: #f7f9fc; color: #111827;">
        <h2>Pharma Causal Lift Model</h2>
        <p>Frontend build not found. Please run one of the options below.</p>
        <h3>Option A: Development mode (recommended)</h3>
        <pre>cd lift-impact-platform
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn backend.api.main:app --reload --port 8000

cd frontend
npm install
npm run dev
# Open http://localhost:5173</pre>
        <h3>Option B: Built mode via FastAPI</h3>
        <pre>cd lift-impact-platform/frontend
npm install
npm run build

cd ..
uvicorn backend.api.main:app --reload --port 8000
# Open http://localhost:8000</pre>
      </body>
    </html>
    """
    return HTMLResponse(instructions)


@app.get("/", response_class=HTMLResponse, response_model=None)
def root():
    built_index = DIST_DIR / "index.html"
    if built_index.exists():
        return FileResponse(built_index)
    return _frontend_help_page()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_excel(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        df = pd.read_excel(BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Excel upload: {exc}")

    schema = detect_schema(df)
    file_id = str(uuid4())
    DATASETS[file_id] = df
    EDA_CACHE.pop(file_id, None)
    return {"file_id": file_id, "schema": schema_to_dict(schema), "rows": len(df)}


@app.get("/checklist/{file_id}")
def checklist(file_id: str) -> dict:
    if file_id not in DATASETS:
        raise HTTPException(status_code=404, detail="file_id not found")
    schema = schema_to_dict(detect_schema(DATASETS[file_id]))
    return {"items": build_variable_checklist(DATASETS[file_id], schema)}


@app.get("/eda/{file_id}")
def eda(
    file_id: str,
    metric_group: str = Query(default="Suggestions"),
    variable: str | None = Query(default=None),
    include_zscore: bool = Query(default=False),
) -> dict:
    if file_id not in DATASETS:
        raise HTTPException(status_code=404, detail="file_id not found")
    cache_key = f"{file_id}:{metric_group}:{variable}:{include_zscore}"
    if cache_key in EDA_CACHE:
        return EDA_CACHE[cache_key]

    schema = schema_to_dict(detect_schema(DATASETS[file_id]))
    result = run_eda(DATASETS[file_id], schema, metric_group=metric_group, variable=variable, include_zscore=include_zscore)
    EDA_CACHE[cache_key] = result
    return result


@app.post("/run/{file_id}")
def run(file_id: str, request: RunRequest) -> dict:
    if file_id not in DATASETS:
        raise HTTPException(status_code=404, detail="file_id not found")
    summary = engine.run(
        DATASETS[file_id],
        scenario_multiplier=request.scenario_multiplier,
        isolate_channel=request.isolate_channel,
    )
    return {"summary": summary}
