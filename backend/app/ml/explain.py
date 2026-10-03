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

HOW:  We use shap's ExactExplainer. With only 8 features there are
      2^8 = 256 feature subsets, so SHAP can compute exact Shapley values
      instead of an approximation. A "missing" feature is simulated by
      replacing it with values from a background sample of 100 training rows.
      We explain the full pipeline on RAW inputs, so contributions are
      reported for "Glucose", "BMI", ... (not for scaled/engineered columns)
      and are in probability units.

VIVA: "SHAP shows how the model used each feature for this prediction. It is
      a description of the model, not proof that a feature causes diabetes."
"""
import numpy as np
import pandas as pd
import shap

from app.ml.features import CSV_COLUMNS, FEATURES

BACKGROUND_SIZE = 100
SHAP_NOTE = (
    "SHAP values describe how the model used each input for this estimate. "
    "They show association within the model, not medical causation."
)


def make_predict_fn(model):
    """SHAP passes numpy arrays; the pipeline expects a DataFrame with CSV column names."""
    def predict(X):
        return model.predict_proba(pd.DataFrame(np.asarray(X, dtype=float), columns=CSV_COLUMNS))[:, 1]
    return predict


def build_explainer(model, background: pd.DataFrame):
    masker = shap.maskers.Independent(background[CSV_COLUMNS].to_numpy(dtype=float),
                                      max_samples=len(background))
    return shap.explainers.Exact(make_predict_fn(model), masker)


def shap_values_for(explainer, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Returns (values [n_rows, 8], base_values [n_rows])."""
    exp = explainer(X[CSV_COLUMNS].to_numpy(dtype=float), silent=True)
    return np.asarray(exp.values), np.asarray(exp.base_values).reshape(-1)


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
        "method": "SHAP ExactExplainer on the full pipeline (probability units)",
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
        "method": "SHAP ExactExplainer, background = 100 training rows, explained set = held-out test set",
        "n_explained": int(len(X)),
        "base_value": round(float(base.mean()), 5),
        "importance": importance,
        "beeswarm": beeswarm,
        "note": SHAP_NOTE,
    }
