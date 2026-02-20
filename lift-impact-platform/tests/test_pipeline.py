from __future__ import annotations

import pandas as pd

from backend.eda.analyzer import run_eda
from backend.services.lift_engine import LiftComputationEngine
from backend.utils.schema_detection import detect_schema, schema_to_dict


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "week_end_date": pd.date_range("2024-01-07", periods=20, freq="W"),
            "hcp_id": [f"H{i%5}" for i in range(20)],
            "rep_id": [f"R{i%3}" for i in range(20)],
            "suggestion_email": ["1", "2", "bad", "4", "5"] * 4,
            "suggestion_call": [1, 0, 1, 0, 1] * 4,
            "action_email": [1 if i % 3 == 0 else 0 for i in range(20)],
            "action_call": [1 if i % 4 == 0 else 0 for i in range(20)],
            "trx": [10 + (i % 4) for i in range(20)],
            "nbrx": [5 + (i % 3) for i in range(20)],
        }
    )


def test_mixed_type_eda_robustness() -> None:
    df = _sample_df()
    schema = schema_to_dict(detect_schema(df))
    result = run_eda(df, schema, metric_group="Suggestions", variable="suggestion_email")
    assert "volume_trends" in result
    assert result["variable_profile"]["mean"] >= 0


def test_monthly_aggregation_correctness() -> None:
    df = _sample_df()
    schema = schema_to_dict(detect_schema(df))
    result = run_eda(df, schema, metric_group="Outcomes", variable="trx")
    months = {row["month"] for row in result["volume_trends"]}
    assert months == {"2024-01", "2024-02", "2024-03", "2024-04", "2024-05"}


def test_outlier_detection_output() -> None:
    df = _sample_df()
    df.loc[0, "trx"] = 1000
    schema = schema_to_dict(detect_schema(df))
    result = run_eda(df, schema, metric_group="Outcomes", variable="trx", include_zscore=True)
    assert "iqr" in result["outliers"]
    assert result["outliers"]["iqr"]["count"] >= 1
    assert "z_score" in result["outliers"]


def test_path_a_monthly_rollup_output() -> None:
    df = _sample_df()
    engine = LiftComputationEngine(artifact_dir="test_artifacts")
    summary = engine.run(df)
    assert summary["path_a"]["aggregated_incremental_actions"] >= 0
    assert len(summary["path_a"]["monthly_rollup"]) > 0
    assert "diagnostics" in summary["path_a"]


def test_deep_eda_quantitative_and_categorical_sections() -> None:
    df = _sample_df()
    df["segment"] = ["A", "B", "A", "C", "B"] * 4
    df["covariate_score"] = [0.2 * i for i in range(20)]
    schema = schema_to_dict(detect_schema(df))
    result = run_eda(df, schema, metric_group="Actions", variable="action_email")

    assert "segment" in result["categorical_variables"]
    assert "covariate_score" in result["quantitative_covariates"]
    assert len(result["monthly_counts_by_type"]["actions"]) > 0
    assert "segment" in result["categorical_unique_counts"]


def test_quantitative_correlation_insights() -> None:
    df = _sample_df()
    df["highly_related"] = df["trx"] * 2
    schema = schema_to_dict(detect_schema(df))
    result = run_eda(df, schema, metric_group="Outcomes", variable="trx")

    assert "trx" in result["quantitative_correlation_matrix"]
    assert any(
        (pair["left"] == "trx" and pair["right"] == "highly_related")
        or (pair["left"] == "highly_related" and pair["right"] == "trx")
        for pair in result["high_correlation_pairs"]
    )


def test_covariate_confound_and_variance_sparsity_outputs() -> None:
    df = _sample_df()
    df["covariate_age"] = [40 + i for i in range(20)]
    df["confound_score"] = [0 if i % 2 == 0 else 1 for i in range(20)]
    schema = schema_to_dict(detect_schema(df))
    result = run_eda(df, schema, metric_group="Suggestions")

    assert "covariate_age" in result["quantitative_covariate_confounders"]
    assert "confound_score" in result["quantitative_covariate_confounders"]
    assert any(row["variable"] == "covariate_age" for row in result["quantitative_diagnostics"])
