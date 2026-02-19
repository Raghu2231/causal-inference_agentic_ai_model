from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from backend.utils.schema_detection import DetectedSchema


def _coerce_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


@dataclass
class TransformationOutput:
    data: pd.DataFrame
    feature_columns: List[str]


def build_features(df: pd.DataFrame, schema: DetectedSchema) -> TransformationOutput:
    out = df.copy()
    out[schema.time_col] = pd.to_datetime(out[schema.time_col], errors="coerce")
    out = out.dropna(subset=[schema.time_col])
    out = _coerce_numeric_columns(out, schema.suggestion_cols + schema.action_cols + schema.outcome_cols)
    out = out.sort_values([schema.hcp_id_col, schema.rep_id_col, schema.time_col])

    out["suggestion_total"] = out[schema.suggestion_cols].sum(axis=1)
    out["action_total"] = out[schema.action_cols].sum(axis=1)

    # Treatment indicators
    out["treated_suggestion"] = (out["suggestion_total"] > 0).astype(int)
    out["treated_action"] = (out["action_total"] > 0).astype(int)

    # Time alignment and lags/windows
    grp = out.groupby([schema.hcp_id_col, schema.rep_id_col], dropna=False)
    out["lag_suggestion_total_1"] = grp["suggestion_total"].shift(1).fillna(0)
    out["lag_action_total_1"] = grp["action_total"].shift(1).fillna(0)
    out["rolling4_suggestion_total"] = grp["suggestion_total"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    out["rolling4_action_total"] = grp["action_total"].transform(lambda s: s.rolling(4, min_periods=1).mean())

    # baseline prescribing and seasonality controls
    if "trx" in " ".join(c.lower() for c in schema.outcome_cols):
        trx_col = next(c for c in schema.outcome_cols if "trx" in c.lower())
        out["baseline_trx_8w"] = grp[trx_col].transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean()).fillna(0)
    else:
        out["baseline_trx_8w"] = 0.0

    out["month"] = out[schema.time_col].dt.month
    out["quarter"] = out[schema.time_col].dt.quarter

    # normalization
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in {"treated_suggestion", "treated_action"}:
            continue
        std = out[col].std()
        if std and not np.isnan(std):
            out[f"z_{col}"] = (out[col] - out[col].mean()) / std
        else:
            out[f"z_{col}"] = 0.0

    # one-hot encoding ids to capture HCP/rep behavior profiles
    out = pd.get_dummies(out, columns=[schema.hcp_id_col, schema.rep_id_col, "month", "quarter"], drop_first=True)

    excluded = set(schema.suggestion_cols + schema.action_cols + schema.outcome_cols + [schema.time_col])
    feature_cols = [c for c in out.columns if c not in excluded]

    return TransformationOutput(data=out.fillna(0), feature_columns=feature_cols)
