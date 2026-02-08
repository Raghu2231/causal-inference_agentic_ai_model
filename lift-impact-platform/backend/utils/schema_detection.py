from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class DetectedSchema:
    suggestion_cols: List[str]
    action_cols: List[str]
    outcome_cols: List[str]
    time_col: str
    hcp_id_col: str
    rep_id_col: str


def _contains_any(name: str, needles: List[str]) -> bool:
    n = name.lower()
    return any(k in n for k in needles)


def detect_schema(df: pd.DataFrame) -> DetectedSchema:
    cols = list(df.columns)

    suggestion_cols = [c for c in cols if _contains_any(c, ["suggest", "insight_suggest", "recommend"]) ]
    action_cols = [c for c in cols if _contains_any(c, ["action", "call", "email", "insight"]) and c not in suggestion_cols]
    outcome_cols = [c for c in cols if _contains_any(c, ["trx", "nbrx", "outcome", "script"]) ]

    time_candidates = [c for c in cols if _contains_any(c, ["date", "week", "month", "time"]) ]
    hcp_candidates = [c for c in cols if _contains_any(c, ["hcp", "customer", "npi"]) ]
    rep_candidates = [c for c in cols if _contains_any(c, ["rep", "territory", "salesperson"]) ]

    if not time_candidates:
        raise ValueError("Could not auto-detect time column.")
    if not hcp_candidates:
        raise ValueError("Could not auto-detect HCP id column.")
    if not rep_candidates:
        raise ValueError("Could not auto-detect Rep id column.")
    if not suggestion_cols:
        raise ValueError("Could not auto-detect suggestion columns.")
    if not action_cols:
        raise ValueError("Could not auto-detect action columns.")
    if not outcome_cols:
        raise ValueError("Could not auto-detect outcome columns (TRX/NBRX).")

    return DetectedSchema(
        suggestion_cols=suggestion_cols,
        action_cols=action_cols,
        outcome_cols=outcome_cols,
        time_col=time_candidates[0],
        hcp_id_col=hcp_candidates[0],
        rep_id_col=rep_candidates[0],
    )


def schema_to_dict(schema: DetectedSchema) -> Dict:
    return {
        "suggestion_cols": schema.suggestion_cols,
        "action_cols": schema.action_cols,
        "outcome_cols": schema.outcome_cols,
        "time_col": schema.time_col,
        "hcp_id_col": schema.hcp_id_col,
        "rep_id_col": schema.rep_id_col,
    }
