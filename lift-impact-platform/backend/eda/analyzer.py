from __future__ import annotations

from typing import Dict

import pandas as pd


def run_eda(df: pd.DataFrame, schema: Dict) -> Dict:
    out = {}
    time_col = schema["time_col"]
    suggestions = schema["suggestion_cols"]
    actions = schema["action_cols"]
    outcomes = schema["outcome_cols"]

    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col])
    tmp["suggestion_total"] = tmp[suggestions].sum(axis=1)
    tmp["action_total"] = tmp[actions].sum(axis=1)

    out["volume_trends"] = (
        tmp.groupby(time_col)[suggestions + actions + outcomes]
        .sum()
        .reset_index()
        .to_dict(orient="records")
    )
    out["action_uptake_rate"] = float((tmp["action_total"] > 0).mean())
    out["suggestion_to_action_conversion"] = float(((tmp["suggestion_total"] > 0) & (tmp["action_total"] > 0)).mean())
    out["missing_values"] = tmp.isna().sum().to_dict()
    out["distributions"] = tmp[suggestions + actions + outcomes].describe().to_dict()
    out["correlation_heatmap"] = tmp[suggestions + actions + outcomes].corr().fillna(0).to_dict()
    return out
