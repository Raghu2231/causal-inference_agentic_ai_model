from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class DetectedSchema:
    suggestion_cols: List[str]
    action_cols: List[str]
    outcome_cols: List[str]
    time_col: str
    hcp_id_col: str
    rep_id_col: str
    warnings: List[str] = field(default_factory=list)


def _contains_any(name: str, needles: List[str]) -> bool:
    n = name.lower().replace(" ", "_")
    return any(k in n for k in needles)


def _pick_text_id_columns(df: pd.DataFrame, excluded: set[str]) -> List[str]:
    candidates: List[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        series = df[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            candidates.append(col)
    return candidates


def _pick_numeric_columns(df: pd.DataFrame, excluded: set[str]) -> List[str]:
    return [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c not in excluded]


def detect_schema(df: pd.DataFrame) -> DetectedSchema:
    cols = list(df.columns)
    warnings: List[str] = []

    suggestion_cols = [
        c
        for c in cols
        if _contains_any(c, ["suggest", "insight_suggest", "recommend", "nudge", "prompt", "next_best"]) 
    ]
    action_cols = [
        c
        for c in cols
        if _contains_any(c, ["action", "call", "email", "insight", "visit", "engage"]) and c not in suggestion_cols
    ]
    outcome_cols = [
        c
        for c in cols
        if _contains_any(c, ["trx", "nbrx", "outcome", "script", "rx", "new_rx", "total_rx"])
    ]

    time_candidates = [c for c in cols if _contains_any(c, ["date", "week", "month", "time", "period"])]
    hcp_candidates = [
        c
        for c in cols
        if _contains_any(c, ["hcp", "customer", "npi", "physician", "doctor", "prescriber", "account"])
    ]
    rep_candidates = [
        c
        for c in cols
        if _contains_any(c, ["rep", "territory", "salesperson", "employee", "owner", "agent", "user"])
    ]

    used: set[str] = set()

    if not time_candidates:
        datetime_like = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
        if datetime_like:
            time_candidates = [datetime_like[0]]
        else:
            time_candidates = ["__auto_time"]
            warnings.append("Could not auto-detect time column. Created synthetic __auto_time index.")
    used.add(time_candidates[0])

    if not hcp_candidates:
        text_ids = _pick_text_id_columns(df, used)
        if text_ids:
            hcp_candidates = [text_ids[0]]
            warnings.append(f"HCP id column inferred as '{text_ids[0]}' from text-like columns.")
        else:
            hcp_candidates = ["__auto_hcp_id"]
            warnings.append("Could not auto-detect HCP id column. Created synthetic __auto_hcp_id.")
    used.add(hcp_candidates[0])

    if not rep_candidates:
        text_ids = _pick_text_id_columns(df, used)
        if text_ids:
            rep_candidates = [text_ids[0]]
            warnings.append(f"Rep id column inferred as '{text_ids[0]}' from text-like columns.")
        else:
            rep_candidates = ["__auto_rep_id"]
            warnings.append("Could not auto-detect Rep id column. Created synthetic __auto_rep_id.")
    used.add(rep_candidates[0])

    if not suggestion_cols:
        numeric = _pick_numeric_columns(df, used)
        if numeric:
            suggestion_cols = [numeric[0]]
            warnings.append(f"Suggestion column inferred as '{numeric[0]}'.")
        else:
            suggestion_cols = ["__auto_suggestion"]
            warnings.append("Could not auto-detect suggestion column. Created synthetic __auto_suggestion.")
    used.update(suggestion_cols)

    if not action_cols:
        numeric = _pick_numeric_columns(df, used)
        if numeric:
            action_cols = [numeric[0]]
            warnings.append(f"Action column inferred as '{numeric[0]}'.")
        else:
            action_cols = ["__auto_action"]
            warnings.append("Could not auto-detect action column. Created synthetic __auto_action.")
    used.update(action_cols)

    if not outcome_cols:
        numeric = _pick_numeric_columns(df, used)
        if numeric:
            outcome_cols = [numeric[0]]
            warnings.append(f"Outcome column inferred as '{numeric[0]}'.")
        else:
            outcome_cols = ["__auto_trx"]
            warnings.append("Could not auto-detect outcome columns. Created synthetic __auto_trx.")

    return DetectedSchema(
        suggestion_cols=suggestion_cols,
        action_cols=action_cols,
        outcome_cols=outcome_cols,
        time_col=time_candidates[0],
        hcp_id_col=hcp_candidates[0],
        rep_id_col=rep_candidates[0],
        warnings=warnings,
    )


def apply_schema_defaults(df: pd.DataFrame, schema: DetectedSchema) -> pd.DataFrame:
    out = df.copy()

    if schema.time_col not in out.columns:
        out[schema.time_col] = pd.date_range("2023-01-01", periods=len(out), freq="D")

    if schema.hcp_id_col not in out.columns:
        out[schema.hcp_id_col] = "AUTO_HCP"

    if schema.rep_id_col not in out.columns:
        out[schema.rep_id_col] = "AUTO_REP"

    for col in schema.suggestion_cols + schema.action_cols + schema.outcome_cols:
        if col not in out.columns:
            out[col] = 0.0

    return out


def schema_to_dict(schema: DetectedSchema) -> Dict:
    return {
        "suggestion_cols": schema.suggestion_cols,
        "action_cols": schema.action_cols,
        "outcome_cols": schema.outcome_cols,
        "time_col": schema.time_col,
        "hcp_id_col": schema.hcp_id_col,
        "rep_id_col": schema.rep_id_col,
        "warnings": schema.warnings,
    }
