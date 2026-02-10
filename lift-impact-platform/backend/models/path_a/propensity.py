from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class PathAPropensityResult:
    propensity_scores: pd.Series
    metrics: Dict[str, float]


class PathAPropensityModel:
    """P(action|suggestion,covariates): action uptake caused by suggestion exposure."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)

    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        treatment_col: str = "treated_suggestion",
        action_col: str = "treated_action",
    ) -> PathAPropensityResult:
        cols = [c for c in feature_cols if c != action_col]
        X = df[cols]
        y = df[action_col]

        if len(np.unique(y)) < 2:
            p = pd.Series(np.repeat(float(y.iloc[0]), len(y)), index=df.index)
            return PathAPropensityResult(propensity_scores=p, metrics={"auc": 0.5})

        self.model.fit(X, y)
        p = self.model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        return PathAPropensityResult(propensity_scores=pd.Series(p, index=df.index), metrics={"auc": float(auc)})

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
