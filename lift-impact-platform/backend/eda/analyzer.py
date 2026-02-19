from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _coerce_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _profile_numeric(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0}
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "q25": float(s.quantile(0.25)),
        "median": float(s.quantile(0.5)),
        "q75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


def _adstock_by_group(df: pd.DataFrame, value_col: str, group_cols: List[str], decay: float = 0.5) -> pd.Series:
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby(group_cols, dropna=False).groups.items():
        values = df.loc[idx, value_col].astype(float).to_numpy()
        acc = np.zeros_like(values, dtype=float)
        for i, v in enumerate(values):
            acc[i] = v + (decay * acc[i - 1] if i > 0 else 0.0)
        out.loc[idx] = acc
    return out.fillna(0.0)


def run_eda(df: pd.DataFrame, schema: Dict) -> Dict:
    out: Dict = {}
    time_col = schema["time_col"]
    suggestions = schema["suggestion_cols"]
    actions = schema["action_cols"]
    outcomes = schema["outcome_cols"]

    tmp = df.copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.dropna(subset=[time_col]).sort_values(time_col)
    tmp = _coerce_numeric_columns(tmp, suggestions + actions + outcomes)
    tmp["suggestion_total"] = tmp[suggestions].sum(axis=1)
    tmp["action_total"] = tmp[actions].sum(axis=1)

    trend_cols = list(dict.fromkeys(suggestions + actions + outcomes))
    trend_frame = tmp.groupby(time_col)[trend_cols].sum().reset_index().tail(120)
    trend_frame[time_col] = trend_frame[time_col].dt.strftime("%Y-%m-%d")
    out["volume_trends"] = trend_frame.to_dict(orient="records")

    out["action_uptake_rate"] = float((tmp["action_total"] > 0).mean())
    out["suggestion_to_action_conversion"] = float(((tmp["suggestion_total"] > 0) & (tmp["action_total"] > 0)).mean())
    out["missing_values"] = tmp.isna().sum().to_dict()

    eda_cols = trend_cols[:30]
    stats_frame = tmp[eda_cols]
    out["distributions"] = stats_frame.describe().to_dict()

    corr_sample = stats_frame.sample(n=min(len(stats_frame), 5000), random_state=42) if len(stats_frame) else stats_frame
    out["correlation_heatmap"] = corr_sample.corr().fillna(0).to_dict()
    out["eda_columns_used"] = eda_cols

    variable_profiles: Dict[str, Dict[str, float]] = {}
    for col in eda_cols:
        variable_profiles[col] = _profile_numeric(tmp[col])
    out["variable_profiles"] = variable_profiles

    hcp_col = schema.get("hcp_id_col")
    rep_col = schema.get("rep_id_col")
    grp_cols = [c for c in [hcp_col, rep_col] if c and c in tmp.columns]
    if not grp_cols:
        tmp["__global_group"] = "all"
        grp_cols = ["__global_group"]

    tmp = tmp.sort_values(grp_cols + [time_col])
    tmp["lag_suggestion_total_1"] = tmp.groupby(grp_cols, dropna=False)["suggestion_total"].shift(1).fillna(0)
    tmp["lag_action_total_1"] = tmp.groupby(grp_cols, dropna=False)["action_total"].shift(1).fillna(0)
    tmp["rolling4_suggestion_total"] = tmp.groupby(grp_cols, dropna=False)["suggestion_total"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    tmp["rolling4_action_total"] = tmp.groupby(grp_cols, dropna=False)["action_total"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    tmp["adstock_suggestion_0_5"] = _adstock_by_group(tmp, "suggestion_total", grp_cols, decay=0.5)
    tmp["adstock_action_0_5"] = _adstock_by_group(tmp, "action_total", grp_cols, decay=0.5)

    fe_cols = [
        "suggestion_total",
        "action_total",
        "lag_suggestion_total_1",
        "lag_action_total_1",
        "rolling4_suggestion_total",
        "rolling4_action_total",
        "adstock_suggestion_0_5",
        "adstock_action_0_5",
    ]
    out["feature_engineering_view"] = [
        {"feature": c, **_profile_numeric(tmp[c])}
        for c in fe_cols
    ]

    return out
