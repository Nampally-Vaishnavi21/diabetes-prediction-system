"""
Explainable AI with SHAP (SHapley Additive exPlanations).

WHAT: For one patient, SHAP splits the model's predicted probability into a
      contribution from each of the 8 input features:

          predicted probability = base value + sum(contributions)

      base value   = the model's average prediction over a background sample
      contribution > 0 pushes the estimate UP (toward the positive class)
      contribution < 0 pushes the estimate DOWN

WHY:  The model is an SVM, which has no readable coefficients. SHAP works for
      any model, so it explains exactly the probability the user sees.

HOW:  With only 8 features there are 2^8 = 256 feature subsets
      ("coalitions"), so we compute EXACT Shapley values instead of an
      approximation:
        1. For every subset S, v(S) = average model output when the features
           in S keep the patient's values and the others take values from a
           background sample of 100 training rows.
        2. A feature's Shapley value = weighted average, over all subsets S
           without it, of v(S + feature) - v(S), with the classic weight
           |S|! (n-|S|-1)! / n!.
      This is the same calculation as the shap library's ExactExplainer with
      an Independent masker (a test checks they agree to 1e-6). It is written
      out in numpy so the web server does not need to load the shap library
      (~260 MB), which lets the app run on small free hosting.
      We explain the full pipeline on RAW inputs, so contributions are
      reported for "Glucose", "BMI", ... (not for scaled/engineered columns)
      and are in probability units.

VIVA: "SHAP shows how the model used each feature for this prediction. It is
      a description of the model, not proof that a feature causes diabetes."
"""
from math import factorial

import numpy as np
import pandas as pd

from app.ml.features import CSV_COLUMNS, FEATURES

BACKGROUND_SIZE = 100
METHOD = "Exact Shapley values (SHAP) over all 256 feature subsets, full pipeline, probability units"
SHAP_NOTE = (
    "SHAP values describe how the model used each input for this estimate. "
    "They show association within the model, not medical causation."
)


class ExactShapleyExplainer:
    """Exact (interventional) Shapley values for a model with a few features."""

    def __init__(self, model, background: pd.DataFrame):
        self.model = model
        self.background = background[CSV_COLUMNS].to_numpy(dtype=float)     # (B, n)
        self.n = len(CSV_COLUMNS)
        m = 2 ** self.n
        idx = np.arange(m)
        # masks[k, j] is True when feature j is "present" in coalition k
        self.masks = ((idx[:, None] >> np.arange(self.n)) & 1).astype(bool)  # (256, 8)
        self.sizes = self.masks.sum(axis=1)
        self.weights = np.array([factorial(s) * factorial(self.n - s - 1) / factorial(self.n)
                                 for s in range(self.n)])
        # For each feature i: the coalitions without i, and the same coalitions with i added
        self.without = [idx[(idx >> i) & 1 == 0] for i in range(self.n)]
        self.with_ = [w | (1 << i) for i, w in enumerate(self.without)]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(pd.DataFrame(X, columns=CSV_COLUMNS))[:, 1]

    def coalition_values(self, x: np.ndarray) -> np.ndarray:
        """v(S) for all 256 subsets S, in one batched model call."""
        b = len(self.background)
        data = np.where(self.masks[:, None, :], x[None, None, :], self.background[None, :, :])
        preds = self._predict(data.reshape(-1, self.n))
        return preds.reshape(len(self.masks), b).mean(axis=1)

    def explain_row(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        v = self.coalition_values(np.asarray(x, dtype=float))
        phi = np.array([
            np.sum(self.weights[self.sizes[wo]] * (v[wi] - v[wo]))
            for wo, wi in zip(self.without, self.with_)
        ])
        return phi, float(v[0])          # v(empty set) = base value


def build_explainer(model, background: pd.DataFrame) -> ExactShapleyExplainer:
    return ExactShapleyExplainer(model, background)


def shap_values_for(explainer: ExactShapleyExplainer, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (values [n_rows, 8], base_values [n_rows])."""
    rows = X[CSV_COLUMNS].to_numpy(dtype=float)
    out = [explainer.explain_row(r) for r in rows]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def local_explanation(explainer, row: pd.DataFrame, probability: float, raw_inputs: dict) -> dict:
    """Explanation for one patient, sorted by absolute contribution."""
    values, base = shap_values_for(explainer, row)
    vals = values[0]
    contributions = []
    for spec, v in zip(FEATURES, vals):
        contributions.append({
            "feature": spec.key,
            "label": spec.label,
            "unit": spec.unit,
            "value": raw_inputs.get(spec.key),         # None if left blank (imputed)
            "imputed": raw_inputs.get(spec.key) is None,
            "shap_value": round(float(v), 5),
            "direction": "increases" if v > 0 else ("decreases" if v < 0 else "neutral"),
        })
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
    base_value = float(base[0])
    total = base_value + float(vals.sum())
    return {
        "method": METHOD,
        "base_value": round(base_value, 5),
        "prediction": round(float(probability), 5),
        "sum_check": round(total, 5),          # equals prediction (additivity)
        "contributions": contributions,
        "top_increasing": [c["label"] for c in contributions if c["shap_value"] > 0][:3],
        "top_decreasing": [c["label"] for c in contributions if c["shap_value"] < 0][:3],
        "note": SHAP_NOTE,
    }


def global_explanation(explainer, X: pd.DataFrame, values=None, base=None) -> dict:
    """Global view over a dataset (the held-out test set): mean |SHAP| + beeswarm points.

    `values`/`base` may be passed in if already computed (avoids recomputing).
    """
    if values is None:
        values, base = shap_values_for(explainer, X)
    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    order = np.argsort(-mean_abs)

    importance = [{
        "feature": FEATURES[i].key, "label": FEATURES[i].label,
        "mean_abs_shap": round(float(mean_abs[i]), 5),
        "mean_shap": round(float(mean_signed[i]), 5),
    } for i in order]

    # Beeswarm data: one point per patient per feature, with the feature value
    # normalised to 0-1 (for colouring low -> high).
    clean = X[CSV_COLUMNS].copy().astype(float)
    beeswarm = []
    for i in order:
        col = clean.iloc[:, i]
        valid = col.replace(0, np.nan) if FEATURES[i].zero_means_missing else col
        lo, hi = valid.min(), valid.max()
        pts = []
        for raw, sv in zip(valid.to_numpy(), values[:, i]):
            norm = None if np.isnan(raw) or hi == lo else round(float((raw - lo) / (hi - lo)), 3)
            pts.append({"shap": round(float(sv), 4),
                        "value": None if np.isnan(raw) else round(float(raw), 3),
                        "norm": norm})
        beeswarm.append({"feature": FEATURES[i].key, "label": FEATURES[i].label, "points": pts})

    return {
        "method": f"{METHOD}; background = {BACKGROUND_SIZE} training rows; explained set = held-out test set",
        "n_explained": int(len(X)),
        "base_value": round(float(base.mean()), 5),
        "importance": importance,
        "beeswarm": beeswarm,
        "note": SHAP_NOTE,
    }
