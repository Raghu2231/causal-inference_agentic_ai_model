from __future__ import annotations

from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
if (DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")


@app.get("/", response_class=HTMLResponse, response_model=None)
def root():
    built_index = DIST_DIR / "index.html"
    if built_index.exists():
        return FileResponse(built_index)
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))


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
