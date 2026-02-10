from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class PathBOutcomeLiftResult:
    incremental_trx: pd.Series
    incremental_nbrx: pd.Series
    metrics: Dict[str, float]


class PathBOutcomeLiftModel:
    """Estimate TRX/NBRX uplift from observed actions versus no-action counterfactual."""

    def __init__(self) -> None:
        self.trx_treated = GradientBoostingRegressor(random_state=11)
        self.trx_control = GradientBoostingRegressor(random_state=11)
        self.nbrx_treated = GradientBoostingRegressor(random_state=11)
        self.nbrx_control = GradientBoostingRegressor(random_state=11)

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        trx_col: str,
        nbrx_col: str,
        treatment_col: str = "treated_action",
    ) -> PathBOutcomeLiftResult:
        X = df[feature_cols]
        treated = df[treatment_col] == 1

        if treated.sum() == 0 or (~treated).sum() == 0:
            inc_trx = pd.Series(0.0, index=df.index)
            inc_nbrx = pd.Series(0.0, index=df.index)
        else:
            self.trx_treated.fit(X[treated], df.loc[treated, trx_col])
            self.trx_control.fit(X[~treated], df.loc[~treated, trx_col])
            self.nbrx_treated.fit(X[treated], df.loc[treated, nbrx_col])
            self.nbrx_control.fit(X[~treated], df.loc[~treated, nbrx_col])

            inc_trx = pd.Series(self.trx_treated.predict(X) - self.trx_control.predict(X), index=df.index).clip(lower=0)
            inc_nbrx = pd.Series(self.nbrx_treated.predict(X) - self.nbrx_control.predict(X), index=df.index).clip(lower=0)

        return PathBOutcomeLiftResult(
            incremental_trx=inc_trx,
            incremental_nbrx=inc_nbrx,
            metrics={"avg_incremental_trx": float(inc_trx.mean()), "avg_incremental_nbrx": float(inc_nbrx.mean())},
        )

    def save(self, artifact_prefix: str) -> None:
        joblib.dump(self.trx_treated, f"{artifact_prefix}_trx_treated.joblib")
        joblib.dump(self.trx_control, f"{artifact_prefix}_trx_control.joblib")
        joblib.dump(self.nbrx_treated, f"{artifact_prefix}_nbrx_treated.joblib")
        joblib.dump(self.nbrx_control, f"{artifact_prefix}_nbrx_control.joblib")
