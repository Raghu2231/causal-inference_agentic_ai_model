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
