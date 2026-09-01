from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import FEATURE_COLUMNS


CATEGORICAL = ["channel", "region"]
NUMERICAL = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]


def _preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERICAL),
            ("cat", categorical_pipe, CATEGORICAL),
        ]
    )


def _response_model(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=180,
                    max_leaf_nodes=20,
                    min_samples_leaf=40,
                    l2_regularization=0.6,
                    random_state=random_state,
                ),
            ),
        ]
    )


@dataclass
class TLearner:
    treated_model: Pipeline
    control_model: Pipeline

    def predict_components(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p1 = self.treated_model.predict_proba(X[FEATURE_COLUMNS])[:, 1]
        p0 = self.control_model.predict_proba(X[FEATURE_COLUMNS])[:, 1]
        return p1, p0

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p1, p0 = self.predict_components(X)
        return p1 - p0


def fit_t_learner(train: pd.DataFrame, seed: int = 42) -> TLearner:
    treated = train.loc[train["treatment"] == 1]
    control = train.loc[train["treatment"] == 0]

    treated_model = _response_model(seed)
    control_model = _response_model(seed + 1)

    treated_model.fit(treated[FEATURE_COLUMNS], treated["returned_60d"])
    control_model.fit(control[FEATURE_COLUMNS], control["returned_60d"])

    return TLearner(treated_model=treated_model, control_model=control_model)


def fit_propensity_benchmark(train: pd.DataFrame, seed: int = 123) -> Pipeline:
    """Predict return propensity while intentionally ignoring treatment.

    This is a benchmark to contrast prediction with causal targeting.
    """
    model = _response_model(seed)
    model.fit(train[FEATURE_COLUMNS], train["returned_60d"])
    return model


def predictive_auc(model: Pipeline, data: pd.DataFrame) -> float:
    p = model.predict_proba(data[FEATURE_COLUMNS])[:, 1]
    return float(roc_auc_score(data["returned_60d"], p))
