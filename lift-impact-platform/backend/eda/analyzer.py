from __future__ import annotations

from typing import Dict, List

import pandas as pd

MAX_POINTS = 1500
CORRELATION_THRESHOLD = 0.7


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _variable_profile(series: pd.Series) -> Dict[str, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    return {
        "mean": float(series.mean()),
        "std_dev": float(series.std(ddof=0)),
        "min": float(series.min()),
        "q1": float(q1),
        "median": float(series.median()),
        "q3": float(q3),
        "max": float(series.max()),
    }


def _outliers(series: pd.Series, include_zscore: bool = False) -> Dict:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    iqr_mask = (series < lower) | (series > upper)

    response: Dict[str, object] = {
        "iqr": {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "count": int(iqr_mask.sum()),
        }
    }

    if include_zscore:
        std = series.std(ddof=0)
        if std > 0:
            z_scores = (series - series.mean()) / std
            z_mask = z_scores.abs() > 3
        else:
            z_mask = pd.Series([False] * len(series), index=series.index)
        response["z_score"] = {"threshold": 3.0, "count": int(z_mask.sum())}

    return response


def run_eda(
    df: pd.DataFrame,
    schema: Dict,
    metric_group: str = "Suggestions",
    variable: str | None = None,
    include_zscore: bool = False,
) -> Dict:
    out = {}
    time_col = schema["time_col"]
    suggestions = schema["suggestion_cols"]
    actions = schema["action_cols"]
    outcomes = schema["outcome_cols"]

    group_map = {
        "Suggestions": suggestions,
        "Actions": actions,
        "Outcomes": outcomes,
    }
    other_cols = [
        col
        for col in df.columns
        if col not in set([time_col] + suggestions + actions + outcomes)
        and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce"))
    ]
    group_map["Other Variables"] = other_cols

    selected_group = metric_group if metric_group in group_map else "Suggestions"
    selected_columns = group_map[selected_group]
    selected_variable = variable if variable in selected_columns else (selected_columns[0] if selected_columns else None)

    all_numeric_cols = suggestions + actions + outcomes + other_cols
    tmp = _safe_numeric(df, all_numeric_cols)
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
    tmp = tmp.dropna(subset=[time_col]).copy()
    tmp["month"] = tmp[time_col].dt.to_period("M").astype(str)

    tmp["suggestion_total"] = tmp[suggestions].sum(axis=1) if suggestions else 0
    tmp["action_total"] = tmp[actions].sum(axis=1) if actions else 0

    metric_cols = set([time_col] + suggestions + actions + outcomes)
    candidate_cols = [c for c in df.columns if c not in metric_cols]
    quantitative_variables: List[str] = []
    categorical_variables: List[str] = []
    for col in candidate_cols:
        parsed = pd.to_numeric(df[col], errors="coerce")
        if float(parsed.notna().mean()) >= 0.8:
            quantitative_variables.append(col)
        else:
            categorical_variables.append(col)

    quantitative_variables = sorted(set(quantitative_variables + suggestions + actions + outcomes))
    categorical_variables = sorted(set(categorical_variables))

    monthly = tmp.groupby("month", as_index=False)[suggestions + actions + outcomes].sum() if (suggestions + actions + outcomes) else pd.DataFrame()
    if len(monthly) > MAX_POINTS:
        monthly = monthly.tail(MAX_POINTS)

    out["metric_groups"] = {name: cols for name, cols in group_map.items()}
    out["selected_group"] = selected_group
    out["selected_variable"] = selected_variable
    out["volume_trends"] = monthly.to_dict(orient="records") if not monthly.empty else []
    out["monthly_totals_by_type"] = {
        "suggestions": tmp.groupby("month", as_index=False)[suggestions].sum().to_dict(orient="records") if suggestions else [],
        "actions": tmp.groupby("month", as_index=False)[actions].sum().to_dict(orient="records") if actions else [],
        "outcomes": tmp.groupby("month", as_index=False)[outcomes].sum().to_dict(orient="records") if outcomes else [],
    }
    out["monthly_counts_by_type"] = {
        "suggestions": (
            tmp.assign(**{col: (tmp[col] > 0).astype(int) for col in suggestions})
            .groupby("month", as_index=False)[suggestions]
            .sum()
            .to_dict(orient="records")
            if suggestions
            else []
        ),
        "actions": (
            tmp.assign(**{col: (tmp[col] > 0).astype(int) for col in actions})
            .groupby("month", as_index=False)[actions]
            .sum()
            .to_dict(orient="records")
            if actions
            else []
        ),
        "outcomes": (
            tmp.assign(**{col: (tmp[col] > 0).astype(int) for col in outcomes})
            .groupby("month", as_index=False)[outcomes]
            .sum()
            .to_dict(orient="records")
            if outcomes
            else []
        ),
    }

    out["action_uptake_rate"] = float((tmp["action_total"] > 0).mean())
    out["suggestion_to_action_conversion"] = float(((tmp["suggestion_total"] > 0) & (tmp["action_total"] > 0)).mean())
    out["missing_values"] = tmp.isna().sum().to_dict()

    profile_source = tmp[selected_variable] if selected_variable else pd.Series([0.0])
    out["variable_profile"] = _variable_profile(profile_source)
    out["outliers"] = _outliers(profile_source, include_zscore=include_zscore)

    monthly_variable_trend = []
    if selected_variable:
        trend_df = tmp.groupby("month", as_index=False)[selected_variable].sum()
        trend_outliers = _outliers(trend_df[selected_variable], include_zscore=include_zscore)
        bounds = trend_outliers["iqr"]
        trend_df["is_outlier"] = (trend_df[selected_variable] < bounds["lower_bound"]) | (
            trend_df[selected_variable] > bounds["upper_bound"]
        )
        monthly_variable_trend = trend_df.to_dict(orient="records")
    out["monthly_variable_trend"] = monthly_variable_trend

    dist_cols = selected_columns if selected_columns else (suggestions + actions + outcomes)
    out["distributions"] = tmp[dist_cols].describe().to_dict() if dist_cols else {}
    out["sampled_rows"] = int(min(len(tmp), MAX_POINTS))
    out["total_rows"] = int(len(tmp))

    covariates = [c for c in quantitative_variables if c not in set(suggestions + actions + outcomes)]
    prefixed_covariate_confounders = [
        c for c in covariates if c.lower().startswith("covariate") or c.lower().startswith("confound")
    ]
    out["quantitative_variables"] = quantitative_variables
    out["categorical_variables"] = categorical_variables
    out["quantitative_covariates"] = covariates
    out["quantitative_covariate_confounders"] = prefixed_covariate_confounders
    out["monthly_covariate_trends"] = {
        col: tmp.groupby("month", as_index=False)[col].mean().to_dict(orient="records") for col in covariates
    }

    out["quantitative_diagnostics"] = [
        {
            "variable": col,
            "variance": float(tmp[col].var(ddof=0)),
            "sparsity_ratio": float((tmp[col] == 0).mean()),
            "non_zero_ratio": float((tmp[col] != 0).mean()),
        }
        for col in quantitative_variables
        if col in tmp.columns
    ]

    quantitative_matrix_cols = [c for c in quantitative_variables if c in tmp.columns]
    if len(quantitative_matrix_cols) >= 2:
        corr_df = tmp[quantitative_matrix_cols].corr().fillna(0)
        pairs = []
        for i, left in enumerate(quantitative_matrix_cols):
            for right in quantitative_matrix_cols[i + 1 :]:
                corr_val = float(corr_df.loc[left, right])
                if abs(corr_val) >= CORRELATION_THRESHOLD:
                    pairs.append(
                        {
                            "left": left,
                            "right": right,
                            "correlation": round(corr_val, 4),
                            "strength": "strong positive" if corr_val > 0 else "strong negative",
                            "insight": (
                                f"{left} and {right} move together strongly; check if both are needed to avoid multicollinearity."
                                if corr_val > 0
                                else f"{left} and {right} move inversely; review lag direction and business assumptions."
                            ),
                        }
                    )
        pairs.sort(key=lambda r: abs(r["correlation"]), reverse=True)
        out["quantitative_correlation_matrix"] = corr_df.to_dict()
        out["high_correlation_pairs"] = pairs
    else:
        out["quantitative_correlation_matrix"] = {}
        out["high_correlation_pairs"] = []

    model_cols = [c for c in suggestions + actions + outcomes if c in tmp.columns]
    out["correlation_heatmap"] = tmp[model_cols].corr().fillna(0).to_dict() if len(model_cols) >= 2 else {}

    id_col = schema.get("hcp_id_col") or next((c for c in df.columns if "customer" in c.lower()), None)
    categorical_unique_counts = {}
    if id_col and id_col in df.columns:
        for col in [c for c in categorical_variables if c != id_col]:
            counts = (
                df[[id_col, col]]
                .fillna("Missing")
                .groupby(col, as_index=False)[id_col]
                .nunique()
                .rename(columns={col: "category", id_col: "unique_id_count"})
                .sort_values("unique_id_count", ascending=False)
            )
            categorical_unique_counts[col] = counts.to_dict(orient="records")
    out["categorical_id_column"] = id_col
    out["categorical_unique_counts"] = categorical_unique_counts
    return out
