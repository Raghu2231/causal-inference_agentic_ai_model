from __future__ import annotations

import pandas as pd

from backend.utils.schema_detection import apply_schema_defaults, detect_schema


def test_schema_detection_handles_missing_hcp_rep_names() -> None:
    df = pd.DataFrame(
        {
            "period": pd.date_range("2024-01-01", periods=6, freq="W"),
            "provider_name": ["Dr A", "Dr B", "Dr C", "Dr A", "Dr B", "Dr C"],
            "owner_name": ["Rep 1", "Rep 2", "Rep 1", "Rep 2", "Rep 1", "Rep 2"],
            "email_suggested": [1, 0, 1, 0, 1, 0],
            "call_made": [1, 0, 0, 1, 0, 1],
            "total_rx": [10, 11, 9, 12, 8, 13],
        }
    )

    schema = detect_schema(df)
    prepared = apply_schema_defaults(df, schema)

    assert schema.hcp_id_col in prepared.columns
    assert schema.rep_id_col in prepared.columns
    assert schema.suggestion_cols
    assert schema.action_cols
    assert schema.outcome_cols
