from __future__ import annotations

import pandas as pd

from backend.utils.variable_partition import partition_variables


def test_partition_variables_maps_groups() -> None:
    df = pd.DataFrame(
        {
            "week_end_date": pd.date_range("2024-01-01", periods=4, freq="W"),
            "hcp": ["h1", "h2", "h1", "h2"],
            "rep": ["r1", "r1", "r2", "r2"],
            "suggestion_call": [1, 0, 1, 0],
            "action_call": [1, 0, 0, 1],
            "trx": [10, 11, 12, 13],
            "covariate_path_A_rep_region": [1, 1, 2, 2],
            "confound_path_A_baseline_rx": [9, 9, 8, 8],
            "lag_action_total_1": [0, 1, 0, 1],
        }
    )
    schema = {
        "suggestion_cols": ["suggestion_call"],
        "action_cols": ["action_call"],
        "outcome_cols": ["trx"],
        "time_col": "week_end_date",
        "hcp_id_col": "hcp",
        "rep_id_col": "rep",
    }
    groups = partition_variables(df, schema)

    assert "suggestion_call" in groups["treatments"]
    assert "action_call" in groups["treatments"]
    assert "covariate_path_A_rep_region" in groups["covariates"]
    assert "confound_path_A_baseline_rx" in groups["confounders"]
    assert "lag_action_total_1" in groups["confounders"]
