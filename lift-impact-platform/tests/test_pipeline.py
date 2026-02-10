from __future__ import annotations

import pandas as pd

from backend.services.lift_engine import LiftComputationEngine


def test_end_to_end_pipeline_runs() -> None:
    df = pd.DataFrame(
        {
            "week_end_date": pd.date_range("2024-01-07", periods=20, freq="W"),
            "hcp_id": [f"H{i%5}" for i in range(20)],
            "rep_id": [f"R{i%3}" for i in range(20)],
            "suggestion_email": [i % 2 for i in range(20)],
            "suggestion_call": [(i + 1) % 2 for i in range(20)],
            "action_email": [1 if i % 3 == 0 else 0 for i in range(20)],
            "action_call": [1 if i % 4 == 0 else 0 for i in range(20)],
            "trx": [10 + (i % 4) for i in range(20)],
            "nbrx": [5 + (i % 3) for i in range(20)],
        }
    )

    engine = LiftComputationEngine(artifact_dir="test_artifacts")
    summary = engine.run(df)
    assert summary["path_a"]["aggregated_incremental_actions"] >= 0
    assert summary["path_b"]["aggregated_incremental_trx"] >= 0
