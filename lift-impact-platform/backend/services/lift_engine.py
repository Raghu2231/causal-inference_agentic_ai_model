from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from backend.models.path_a.propensity import PathAPropensityModel
from backend.models.path_a.treatment_effect import PathATreatmentEffectModel
from backend.models.path_b.outcome_lift import PathBOutcomeLiftModel
from backend.transformations.feature_engineering import build_features
from backend.utils.schema_detection import detect_schema, schema_to_dict


class LiftComputationEngine:
    def __init__(self, artifact_dir: str = "artifacts") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _monthly_rollup(model_df: pd.DataFrame, time_col: str, value_cols: List[str]) -> List[Dict]:
        out = model_df.copy()
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out = out.dropna(subset=[time_col]).copy()
        out["month"] = out[time_col].dt.to_period("M").astype(str)
        return out.groupby("month", as_index=False)[value_cols].sum().to_dict(orient="records")

    def run(self, raw_df: pd.DataFrame, scenario_multiplier: float = 1.0, isolate_channel: str | None = None) -> Dict:
        schema = detect_schema(raw_df)
        schema_dict = schema_to_dict(schema)
        transformed = build_features(raw_df, schema)
        model_df = transformed.data

        pa_prop = PathAPropensityModel()
        pa_prop_res = pa_prop.fit_predict(model_df, transformed.feature_columns)
        model_df["propensity_score_path_a"] = pa_prop_res.propensity_scores

        pa_te = PathATreatmentEffectModel()
        path_a_result = pa_te.fit_predict(
            model_df,
            transformed.feature_columns + ["propensity_score_path_a"],
            channel_cols=schema.action_cols,
        )

        model_df["incremental_actions"] = path_a_result.incremental_action_by_row * scenario_multiplier
        if isolate_channel and isolate_channel in schema.action_cols:
            denom = raw_df[schema.action_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).replace(0, 1)
            ch_weight = (pd.to_numeric(raw_df[isolate_channel], errors="coerce").fillna(0) / denom).fillna(0)
            model_df["incremental_actions"] *= ch_weight

        trx_col = next(c for c in schema.outcome_cols if "trx" in c.lower())
        nbrx_col = next((c for c in schema.outcome_cols if "nbrx" in c.lower()), trx_col)

        pb = PathBOutcomeLiftModel()
        path_b_result = pb.fit_predict(
            model_df,
            transformed.feature_columns + ["incremental_actions"],
            trx_col=trx_col,
            nbrx_col=nbrx_col,
        )

        model_df["incremental_trx"] = path_b_result.incremental_trx
        model_df["incremental_nbrx"] = path_b_result.incremental_nbrx

        path_a_monthly = self._monthly_rollup(model_df, schema.time_col, ["incremental_actions"])
        path_b_monthly = self._monthly_rollup(model_df, schema.time_col, ["incremental_trx", "incremental_nbrx"])

        path_a_diagnostics = {
            "auc": pa_prop_res.metrics.get("auc", 0.0),
            "treated_cohort_size": int((model_df["treated_suggestion"] == 1).sum()),
            "control_cohort_size": int((model_df["treated_suggestion"] == 0).sum()),
            "feature_count": int(len(transformed.feature_columns)),
            "warnings": [
                "Low treated cohort size" if (model_df["treated_suggestion"] == 1).sum() < 10 else ""
            ],
        }
        path_a_diagnostics["warnings"] = [w for w in path_a_diagnostics["warnings"] if w]

        path_b_diagnostics = {
            **path_b_result.metrics,
            "treated_action_rows": int((model_df["treated_action"] == 1).sum()),
            "control_action_rows": int((model_df["treated_action"] == 0).sum()),
            "feature_count": int(len(transformed.feature_columns) + 1),
        }

        channel_monthly_rollup = []
        raw_num = raw_df.copy()
        raw_num[schema.time_col] = pd.to_datetime(raw_num[schema.time_col], errors="coerce")
        raw_num = raw_num.dropna(subset=[schema.time_col])
        raw_num["month"] = raw_num[schema.time_col].dt.to_period("M").astype(str)
        for ch in schema.action_cols:
            if ch in raw_num.columns:
                share = (
                    pd.to_numeric(raw_num[ch], errors="coerce").fillna(0)
                    / raw_num[schema.action_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).replace(0, 1)
                ).fillna(0)
                tmp = pd.DataFrame({"month": raw_num["month"], "channel": ch, "incremental_actions": model_df["incremental_actions"] * share})
                channel_monthly_rollup.extend(tmp.groupby(["month", "channel"], as_index=False)["incremental_actions"].sum().to_dict(orient="records"))

        summary = {
            "schema": schema_dict,
            "path_a": {
                "setup": {
                    "propensity_inputs": transformed.feature_columns,
                    "treatment_effect_inputs": transformed.feature_columns + ["propensity_score_path_a"],
                    "guidance": "Use propensity to validate treatment assignment balance, then interpret treatment effect as directional uplift.",
                    "disclaimer": "Causal estimates depend on observed confounders and data quality checks.",
                },
                "propensity_metrics": pa_prop_res.metrics,
                "diagnostics": path_a_diagnostics,
                "avg_incremental_actions": float(model_df["incremental_actions"].mean()),
                "channel_lift": path_a_result.channel_lift,
                "channel_monthly_rollup": channel_monthly_rollup,
                "monthly_rollup": path_a_monthly,
                "aggregated_incremental_actions": float(model_df["incremental_actions"].sum()),
            },
            "path_b": {
                "classification": {
                    "outcomes": schema.outcome_cols,
                    "treatment_action_drivers": schema.action_cols,
                    "confounders_covariates": transformed.feature_columns,
                },
                "metrics": path_b_result.metrics,
                "diagnostics": path_b_diagnostics,
                "monthly_rollup": path_b_monthly,
                "aggregated_incremental_trx": float(model_df["incremental_trx"].sum()),
                "aggregated_incremental_nbrx": float(model_df["incremental_nbrx"].sum()),
            },
        }

        model_df.to_parquet(self.artifact_dir / "scored_output.parquet", index=False)
        (self.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(path_a_monthly).to_csv(self.artifact_dir / "path_a_monthly_rollup.csv", index=False)
        pd.DataFrame(path_b_monthly).to_csv(self.artifact_dir / "path_b_monthly_rollup.csv", index=False)
        pd.DataFrame(path_a_diagnostics.items(), columns=["metric", "value"]).to_csv(
            self.artifact_dir / "path_a_diagnostics.csv", index=False
        )
        (self.artifact_dir / "path_a_diagnostics.json").write_text(json.dumps(path_a_diagnostics, indent=2))

        return summary
