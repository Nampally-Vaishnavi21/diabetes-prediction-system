"""
Model uncertainty via a bootstrap ensemble.

WHAT: We train 50 copies of the final model, each on a different bootstrap
      resample of the training set (rows drawn with replacement). For a new
      patient we ask all 50 copies and look at the spread of their answers.

WHY:  A single probability (e.g. 0.62) hides how stable that number is. If
      small changes in the training data move the estimate from 0.40 to 0.80,
      the model is uncertain for this patient. A narrow spread means the
      estimate is stable.

WHAT IT IS NOT: It is not a clinical confidence interval and not the chance
      that the patient has diabetes. It measures the model's sensitivity to
      its training data (epistemic uncertainty) only.

VIVA: "We report the 5th-95th percentile of 50 bootstrap models' estimates.
      If that range crosses the decision threshold, the classification is
      flagged as borderline."
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

N_BOOTSTRAP = 50
LOWER_PCT, UPPER_PCT = 5, 95


def train_bootstrap_ensemble(estimator, X: pd.DataFrame, y: pd.Series,
                             n_models: int = N_BOOTSTRAP, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    models = []
    n = len(X)
    while len(models) < n_models:
        idx = rng.integers(0, n, n)
        yb = y.iloc[idx]
        if yb.nunique() < 2:
            continue
        m = clone(estimator)
        m.fit(X.iloc[idx], yb)
        models.append(m)
    return models


def bootstrap_uncertainty(models: list, row: pd.DataFrame, threshold: float, point: float) -> dict:
    probs = np.array([m.predict_proba(row)[0, 1] for m in models])
    lower, upper = np.percentile(probs, [LOWER_PCT, UPPER_PCT])
    median = float(np.median(probs))
    borderline = bool(lower < threshold <= upper)
    share_above = float((probs >= threshold).mean())
    return {
        "method": f"Bootstrap ensemble: {len(models)} models, each trained on a resample of the training set",
        "n_models": len(models),
        "interval_level_pct": UPPER_PCT - LOWER_PCT,
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
        "median": round(median, 4),
        "std": round(float(probs.std()), 4),
        "width": round(float(upper - lower), 4),
        "share_of_models_above_threshold": round(share_above, 3),
        "borderline": borderline,
        "point_estimate": round(float(point), 4),
        "explanation": (
            f"Across {len(models)} bootstrap models, 90% of estimates fall between "
            f"{lower:.1%} and {upper:.1%}. "
            + ("This range crosses the decision threshold, so the classification is borderline: "
               "slightly different training data could flip it."
               if borderline else
               "The whole range is on one side of the decision threshold, so the classification "
               "is stable under resampling of the training data.")
        ),
        "caveat": "This reflects the model's sensitivity to its training data, not a clinical "
                  "confidence interval or diagnostic certainty.",
    }
