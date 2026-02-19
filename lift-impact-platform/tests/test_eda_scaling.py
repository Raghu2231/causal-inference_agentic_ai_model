from __future__ import annotations

import pandas as pd

from backend.eda.analyzer import run_eda


def test_eda_limits_payload_for_large_inputs() -> None:
    rows = 250
    df = pd.DataFrame(
        {
            "week_end_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "suggestion_email": [1 if i % 2 else 0 for i in range(rows)],
            "action_email": [1 if i % 3 == 0 else 0 for i in range(rows)],
            "trx": [10 + (i % 5) for i in range(rows)],
        }
    )
    schema = {
        "time_col": "week_end_date",
        "suggestion_cols": ["suggestion_email"],
        "action_cols": ["action_email"],
        "outcome_cols": ["trx"],
    }

    result = run_eda(df, schema)
    assert len(result["volume_trends"]) <= 120
    assert len(result["eda_columns_used"]) <= 30
    assert "variable_profiles" in result
    assert "feature_engineering_view" in result
    if result["volume_trends"]:
        assert isinstance(result["volume_trends"][0]["week_end_date"], str)


def test_eda_handles_mixed_numeric_and_string_values() -> None:
    df = pd.DataFrame(
        {
            "week_end_date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "suggestion_email": ["1", "bad", 2, None, "3"],
            "action_email": [0, "1", "x", 1, 0],
            "trx": [10, "11", "bad", 8, 9],
        }
    )
    schema = {
        "time_col": "week_end_date",
        "suggestion_cols": ["suggestion_email"],
        "action_cols": ["action_email"],
        "outcome_cols": ["trx"],
    }

    result = run_eda(df, schema)
    assert "action_uptake_rate" in result
    assert isinstance(result["action_uptake_rate"], float)
