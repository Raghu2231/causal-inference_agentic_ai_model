from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

# Mirrors model feature pattern philosophy in Causal_Inference/config/ml.py
CONFOUND_PATTERN = re.compile(r"^confound_path_(A|B)_.*$", re.IGNORECASE)
COVARIATE_PATTERN = re.compile(r"^covariate_path_(A|B)_.*$", re.IGNORECASE)
PROPENSITY_PATTERN = re.compile(r"^propensity_score_path_(A|B)_[^_]+$", re.IGNORECASE)


def partition_variables(df: pd.DataFrame, schema: Dict) -> Dict[str, List[str]]:
    cols = list(df.columns)
    treatment_cols = list(dict.fromkeys(schema.get("suggestion_cols", []) + schema.get("action_cols", [])))

    confounders: List[str] = []
    covariates: List[str] = []

    for col in cols:
        low = col.lower()
        if col in treatment_cols or col in schema.get("outcome_cols", []):
            continue
        if col in {schema.get("time_col"), schema.get("hcp_id_col"), schema.get("rep_id_col")}:
            continue

        if CONFOUND_PATTERN.match(col):
            confounders.append(col)
            continue
        if COVARIATE_PATTERN.match(col) or PROPENSITY_PATTERN.match(col):
            covariates.append(col)
            continue

        # heuristic mapping aligned with workflow concepts from Causal_Inference utilities
        if any(k in low for k in ["lag", "rolling", "baseline", "season", "month", "quarter", "trend"]):
            confounders.append(col)
        elif any(k in low for k in ["region", "district", "specialty", "segment", "tier", "gender", "channel"]):
            covariates.append(col)

    if not covariates:
        numeric_cols = [
            c for c in df.select_dtypes(include=["number"]).columns.tolist() if c not in treatment_cols and c not in schema.get("outcome_cols", [])
        ]
        covariates = numeric_cols[:10]

    return {
        "treatments": treatment_cols,
        "covariates": sorted(set(covariates)),
        "confounders": sorted(set(confounders)),
        "outcomes": schema.get("outcome_cols", []),
    }
