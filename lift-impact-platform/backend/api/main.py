from __future__ import annotations

from io import BytesIO
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.eda.analyzer import run_eda
from backend.services.insight_generator import InsightGenerator
from backend.services.lift_engine import LiftComputationEngine
from backend.services.session_store import DATASETS, EDA_CACHE
from backend.utils.schema_detection import apply_schema_defaults, detect_schema, schema_to_dict
from backend.utils.variable_partition import partition_variables
from data_contracts.contracts import InsightRequest, RunRequest

app = FastAPI(title="Lift Impact Platform API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
engine = LiftComputationEngine()
insight_generator = InsightGenerator()


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

    try:
        schema = detect_schema(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    prepared_df = apply_schema_defaults(df, schema)
    file_id = str(uuid4())
    DATASETS[file_id] = prepared_df
    EDA_CACHE[file_id] = {}
    return {"file_id": file_id, "schema": schema_to_dict(schema), "rows": len(prepared_df)}




@app.get("/preview/{file_id}")
def preview(file_id: str) -> dict:
    if file_id not in DATASETS:
        raise HTTPException(status_code=404, detail="file_id not found")

    df = DATASETS[file_id]
    schema = schema_to_dict(detect_schema(df))
    groups = partition_variables(df, schema)
    return {
        "schema": schema,
        "groups": groups,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "sample": df.head(20).to_dict(orient="records"),
        "columns": list(df.columns),
    }

@app.get("/eda/{file_id}")
def eda(file_id: str) -> dict:
    if file_id not in DATASETS:
        raise HTTPException(status_code=404, detail="file_id not found")
    if file_id in EDA_CACHE and EDA_CACHE[file_id]:
        return EDA_CACHE[file_id]
    schema = schema_to_dict(detect_schema(DATASETS[file_id]))
    eda_payload = run_eda(DATASETS[file_id], schema)
    EDA_CACHE[file_id] = eda_payload
    return eda_payload


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


@app.post("/insights/{file_id}")
def insights(file_id: str, request: InsightRequest) -> dict:
    if file_id not in DATASETS:
        raise HTTPException(status_code=404, detail="file_id not found")

    summary = request.summary
    if summary is None:
        summary = engine.run(DATASETS[file_id])

    insight = insight_generator.generate(summary, prompt_context=request.context)
    return {"narrative": insight.narrative, "bullets": insight.bullets, "source": insight.source}
