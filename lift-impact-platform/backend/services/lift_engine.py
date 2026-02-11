from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from backend.models.path_a.propensity import PathAPropensityModel
from backend.models.path_a.treatment_effect import PathATreatmentEffectModel
from backend.models.path_b.outcome_lift import PathBOutcomeLiftModel
from backend.transformations.feature_engineering import build_features
from backend.utils.schema_detection import apply_schema_defaults, detect_schema, schema_to_dict


class LiftComputationEngine:
    def __init__(self, artifact_dir: str = "artifacts") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def run(self, raw_df: pd.DataFrame, scenario_multiplier: float = 1.0, isolate_channel: str | None = None) -> Dict:
        schema = detect_schema(raw_df)
        schema_dict = schema_to_dict(schema)
        prepared_df = apply_schema_defaults(raw_df, schema)
        transformed = build_features(prepared_df, schema)
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
            ch_weight = (prepared_df[isolate_channel] / prepared_df[schema.action_cols].sum(axis=1).replace(0, 1)).fillna(0)
            model_df["incremental_actions"] *= ch_weight

        trx_col = next((c for c in schema.outcome_cols if "trx" in c.lower() and "nbrx" not in c.lower()), schema.outcome_cols[0])
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

        rep_lift = model_df.groupby([c for c in model_df.columns if c.startswith(schema.rep_id_col + "_")]).agg(
            incremental_actions=("incremental_actions", "sum"),
            incremental_trx=("incremental_trx", "sum"),
            incremental_nbrx=("incremental_nbrx", "sum"),
        ) if any(c.startswith(schema.rep_id_col + "_") for c in model_df.columns) else pd.DataFrame()

        hcp_lift = model_df.groupby([c for c in model_df.columns if c.startswith(schema.hcp_id_col + "_")]).agg(
            incremental_actions=("incremental_actions", "sum"),
            incremental_trx=("incremental_trx", "sum"),
            incremental_nbrx=("incremental_nbrx", "sum"),
        ) if any(c.startswith(schema.hcp_id_col + "_") for c in model_df.columns) else pd.DataFrame()

        summary = {
            "schema": schema_dict,
            "path_a": {
                "propensity_metrics": pa_prop_res.metrics,
                "avg_incremental_actions": float(model_df["incremental_actions"].mean()),
                "channel_lift": path_a_result.channel_lift,
                "aggregated_incremental_actions": float(model_df["incremental_actions"].sum()),
            },
            "path_b": {
                "metrics": path_b_result.metrics,
                "aggregated_incremental_trx": float(model_df["incremental_trx"].sum()),
                "aggregated_incremental_nbrx": float(model_df["incremental_nbrx"].sum()),
            },
        }

        model_df.to_parquet(self.artifact_dir / "scored_output.parquet", index=False)
        (self.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if not rep_lift.empty:
            rep_lift.to_csv(self.artifact_dir / "rep_lift.csv")
        if not hcp_lift.empty:
            hcp_lift.to_csv(self.artifact_dir / "hcp_lift.csv")

        return summary
