"""
Preprocessing pipeline.

WHAT: The exact sequence of transformations applied to raw input before it
      reaches a model:

        raw 8 columns
          -> ZeroToNaN        impossible zeros (e.g. BMI = 0) become missing
          -> SimpleImputer    fill missing with the TRAINING median
                              + add 0/1 "was missing" indicator columns
          -> FeatureEngineer  optional engineered features (on/off is tuned)
          -> StandardScaler   mean 0, std 1 (needed by LR, SVM, KNN)
          -> model

WHY:  Wrapping all of this in ONE sklearn Pipeline means
      1. no data leakage - during cross-validation the medians and scaling
         are learned from the training folds only, never from validation data;
      2. the backend calls pipeline.predict_proba(raw_row) and gets exactly the
         same preprocessing that was used in training (saved together in one
         joblib file).
VIVA: "Preprocessing is part of the saved model object. The API sends raw values;
      the pipeline does the rest, so training and serving cannot diverge."
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.features import ZERO_AS_MISSING_COLUMNS


def _to_frame(X, columns) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    return pd.DataFrame(np.asarray(X, dtype=float), columns=columns)


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace 0 with NaN in columns where 0 is physiologically impossible.

    Stateless (learns nothing from data), so it cannot leak information.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(
            X.columns if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(X.shape[1])],
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        df = _to_frame(X, self.feature_names_in_)
        cols = [c for c in (self.columns or []) if c in df.columns]
        df[cols] = df[cols].replace(0, np.nan)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-motivated engineered features. `enabled` is a tuned hyperparameter.

    When enabled:
      - Glucose_x_BMI : interaction of the two strongest risk factors
      - Insulin, DiabetesPedigreeFunction are replaced by log transforms
        (both are strongly right-skewed; logs help linear models / SVM)
    When disabled the data passes through unchanged.
    Cross-validation decides per model whether this helps.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        df = _to_frame(X, self.feature_names_in_)
        if not self.enabled:
            return df
        df["Glucose_x_BMI"] = df["Glucose"] * df["BMI"] / 100.0
        df["Insulin"] = np.log1p(df["Insulin"])
        df["DiabetesPedigreeFunction"] = np.log(df["DiabetesPedigreeFunction"])
        return df

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        if self.enabled:
            names.append("Glucose_x_BMI")
        return np.asarray(names, dtype=object)


def build_preprocessor() -> Pipeline:
    pre = Pipeline([
        ("zero_to_nan", ZeroToNaN(columns=ZERO_AS_MISSING_COLUMNS)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("engineer", FeatureEngineer(enabled=False)),
        ("scale", StandardScaler()),
    ])
    pre.set_output(transform="pandas")   # keep column names through every step
    return pre


def build_pipeline(model) -> Pipeline:
    """Full pipeline: preprocessing + classifier."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])
