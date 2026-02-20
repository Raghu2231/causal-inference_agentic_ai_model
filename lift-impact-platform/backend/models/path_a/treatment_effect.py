from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class PathATreatmentEffectResult:
    incremental_action_by_row: pd.Series
    channel_lift: Dict[str, float]
    avg_lift: float


class PathATreatmentEffectModel:
    """T-learner for action volume uplift from suggestion exposure."""

    def __init__(self) -> None:
        self.model_treated = GradientBoostingRegressor(random_state=42)
        self.model_control = GradientBoostingRegressor(random_state=42)

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        action_value_col: str = "action_total",
        treatment_col: str = "treated_suggestion",
        channel_cols: List[str] | None = None,
    ) -> PathATreatmentEffectResult:
        channel_cols = channel_cols or []
        X = df[feature_cols]
        treated = df[treatment_col] == 1

        if treated.sum() == 0 or (~treated).sum() == 0:
            base_treated = float(df.loc[treated, action_value_col].mean()) if treated.sum() else 0.0
            base_control = float(df.loc[~treated, action_value_col].mean()) if (~treated).sum() else 0.0
            y1 = pd.Series([base_treated] * len(df), index=df.index)
            y0 = pd.Series([base_control] * len(df), index=df.index)
        else:
            self.model_treated.fit(X[treated], df.loc[treated, action_value_col])
            self.model_control.fit(X[~treated], df.loc[~treated, action_value_col])
            y1 = pd.Series(self.model_treated.predict(X), index=df.index)
            y0 = pd.Series(self.model_control.predict(X), index=df.index)
        ite = pd.Series(y1 - y0, index=df.index).clip(lower=0)

        channel_lift = {}
        for ch in channel_cols:
            if ch in df.columns:
                share = (df[ch] / df[channel_cols].sum(axis=1).replace(0, 1)).fillna(0)
                channel_lift[ch] = float((ite * share).sum())

        return PathATreatmentEffectResult(
            incremental_action_by_row=ite,
            channel_lift=channel_lift,
            avg_lift=float(ite.mean()),
        )

    def save(self, treated_path: str, control_path: str) -> None:
        joblib.dump(self.model_treated, treated_path)
        joblib.dump(self.model_control, control_path)
