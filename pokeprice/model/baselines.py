"""Simple baselines the LightGBM model has to beat."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class NoChange:
    """Predict that the price stays the same (log return 0). Hard to beat in noisy markets."""

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def ridge():
    """Linear model on the same features."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
