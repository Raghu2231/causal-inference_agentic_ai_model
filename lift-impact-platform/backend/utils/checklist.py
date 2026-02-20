from __future__ import annotations

from typing import Dict, List

import pandas as pd


CHECKLIST_ORDER = [
    "parseable_time_column",
    "stable_hcp_rep_ids",
    "numeric_convertible_suggestion_action_outcome",
    "grain_consistency",
    "temporal_depth_for_lags",
]


def build_variable_checklist(df: pd.DataFrame, schema: Dict) -> List[Dict[str, str]]:
    time_col = schema["time_col"]
    hcp_col = schema["hcp_id_col"]
    rep_col = schema["rep_id_col"]
    metric_cols = schema["suggestion_cols"] + schema["action_cols"] + schema["outcome_cols"]

    parsed_time = pd.to_datetime(df[time_col], errors="coerce")
    parseable_ratio = parsed_time.notna().mean()

    stable_ids = df[hcp_col].notna().all() and df[rep_col].notna().all()
    numeric_ratio = (
        pd.to_numeric(df[metric_cols].stack(), errors="coerce").notna().mean() if metric_cols else 1.0
    )

    keyed = df.assign(_parsed_time=parsed_time).dropna(subset=["_parsed_time"]) \
        .groupby([hcp_col, rep_col, "_parsed_time"]).size()
    grain_consistent = bool((keyed <= 1).all()) if not keyed.empty else False

    monthly_depth = parsed_time.dt.to_period("M").nunique()

    checks = {
        "parseable_time_column": {
            "label": "Parseable time column",
            "status": "pass" if parseable_ratio >= 0.95 else "warn",
            "detail": f"{parseable_ratio:.1%} rows parse to datetime.",
        },
        "stable_hcp_rep_ids": {
            "label": "Stable HCP/Rep IDs",
            "status": "pass" if stable_ids else "warn",
            "detail": "All rows have HCP and Rep IDs." if stable_ids else "Missing HCP/Rep IDs detected.",
        },
        "numeric_convertible_suggestion_action_outcome": {
            "label": "Numeric-convertible suggestions/actions/outcomes",
            "status": "pass" if numeric_ratio >= 0.9 else "warn",
            "detail": f"{numeric_ratio:.1%} metric values are numeric-convertible.",
        },
        "grain_consistency": {
            "label": "Grain consistency",
            "status": "pass" if grain_consistent else "warn",
            "detail": "No duplicate HCP/Rep/time grain rows." if grain_consistent else "Duplicate grain rows found.",
        },
        "temporal_depth_for_lags": {
            "label": "Enough temporal depth for lags/rolling/adstock",
            "status": "pass" if monthly_depth >= 6 else "warn",
            "detail": f"Detected {monthly_depth} unique months.",
        },
    }

    return [checks[key] for key in CHECKLIST_ORDER]
