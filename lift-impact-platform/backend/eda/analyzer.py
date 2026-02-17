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
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.dropna(subset=[time_col])
    tmp["suggestion_total"] = tmp[suggestions].sum(axis=1)
    tmp["action_total"] = tmp[actions].sum(axis=1)

    # Reduce payload size and latency for large uploads.
    trend_cols = list(dict.fromkeys(suggestions + actions + outcomes))
    out["volume_trends"] = (
        tmp.groupby(time_col)[trend_cols]
        .sum()
        .reset_index()
        .tail(120)
        .to_dict(orient="records")
    )
    out["action_uptake_rate"] = float((tmp["action_total"] > 0).mean())
    out["suggestion_to_action_conversion"] = float(((tmp["suggestion_total"] > 0) & (tmp["action_total"] > 0)).mean())
    out["missing_values"] = tmp.isna().sum().to_dict()

    eda_cols = trend_cols[:30]
    stats_frame = tmp[eda_cols]
    out["distributions"] = stats_frame.describe().to_dict()

    corr_sample = stats_frame.sample(n=min(len(stats_frame), 5000), random_state=42) if len(stats_frame) else stats_frame
    out["correlation_heatmap"] = corr_sample.corr().fillna(0).to_dict()
    out["eda_columns_used"] = eda_cols
    return out
