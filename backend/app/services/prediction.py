"""
Prediction service: everything that happens after a request is validated.

    PatientInput (validated by Pydantic)
        -> DataFrame with the original CSV column names (blank -> NaN)
        -> final pipeline .predict_proba()     (preprocessing + model)
        -> threshold                           (tuned during training)
        -> bootstrap uncertainty               (50 resampled models)
        -> model agreement                     (all 7 trained models)
        -> out-of-distribution check           (vs training-data ranges)
        -> SHAP explanation                    (exact Shapley values)
        -> JSON response

Every number in the response is computed here from the trained artifacts.
"""
import numpy as np
import pandas as pd

from app.ml.explain import local_explanation
from app.ml.features import CSV_COLUMNS, FEATURES, SPEC_BY_KEY, TARGET_COLUMN
from app.ml.uncertainty import LOWER_PCT, UPPER_PCT, bootstrap_uncertainty
from app.schemas.patient import PatientInput
from app.services.model_registry import registry
from app.utils.errors import AnalysisError

DISCLAIMER = (
    "This system is an academic/research prototype for diabetes risk prediction. It is not a "
    "medical diagnostic device and should not be used as a substitute for professional medical "
    "advice, diagnosis, or treatment."
)
LABELS = {1: "Higher estimated risk", 0: "Lower estimated risk"}


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def to_frame(patient: PatientInput) -> pd.DataFrame:
    return pd.DataFrame([patient.to_csv_row()], columns=CSV_COLUMNS)


def model_probability(row: pd.DataFrame) -> float:
    return float(registry.pipeline.predict_proba(row)[0, 1])


def threshold() -> float:
    return float(registry.metadata["threshold"])


def classify(prob: float) -> int:
    return int(prob >= threshold())


def interpretation(prob: float) -> str:
    thr = threshold()
    side = "above" if prob >= thr else "below"
    return (
        f"The model estimates a probability of approximately {prob:.0%} based on the input "
        f"features. This is {side} the model's decision threshold of {thr:.0%}, so the input is "
        f"classified as '{LABELS[classify(prob)]}'. This is a statistical estimate from a model "
        "trained on a small research dataset, not a diagnosis."
    )


def out_of_distribution(patient: PatientInput) -> list[dict]:
    ranges = registry.metadata.get("training_ranges", {})
    warnings = []
    for spec in FEATURES:
        value = getattr(patient, spec.key)
        r = ranges.get(spec.key)
        if value is None or r is None:
            continue
        if value < r["min"] or value > r["max"]:
            warnings.append({
                "feature": spec.key, "label": spec.label, "value": float(value),
                "training_min": r["min"], "training_max": r["max"],
                "message": (f"{spec.label} = {value:g} {spec.unit} is outside the range seen in "
                            f"training ({r['min']:g}–{r['max']:g}). The estimate is an "
                            "extrapolation and is less reliable."),
            })
    return warnings


def model_agreement(row: pd.DataFrame) -> dict:
    thr = threshold()
    names = registry.metadata.get("model_display_names", {})
    votes = []
    for key, model in registry.all_models.items():
        p = float(model.predict_proba(row)[0, 1])
        votes.append({"key": key, "display_name": names.get(key, key),
                      "probability": round(p, 4), "above_threshold": p >= thr})
    return {
        "note": ("Probabilities from every model trained during model comparison, judged against "
                 "the selected model's threshold. Disagreement between models signals uncertainty."),
        "votes": votes,
        "n_above_threshold": sum(v["above_threshold"] for v in votes),
        "n_models": len(votes),
    }


def explain(patient: PatientInput, row: pd.DataFrame, prob: float) -> dict:
    return local_explanation(registry.explainer, row, prob, patient.to_api_dict())


# --------------------------------------------------------------------------- #
# Endpoint logic
# --------------------------------------------------------------------------- #
def predict(patient: PatientInput, include_explanation: bool = True) -> dict:
    registry.require(complete=True)
    row = to_frame(patient)
    prob = model_probability(row)
    pred = classify(prob)
    return {
        "prediction": pred,
        "prediction_label": LABELS[pred],
        "probability": round(prob, 4),
        "probability_pct": round(100 * prob, 1),
        "threshold": round(threshold(), 4),
        "threshold_method": registry.metadata["threshold_method"],
        "model": registry.metadata["selected_model_display"],
        "interpretation": interpretation(prob),
        "uncertainty": bootstrap_uncertainty(registry.bootstrap_models, row, threshold(), prob),
        "model_agreement": model_agreement(row),
        "ood_warnings": out_of_distribution(patient),
        "imputed_features": [SPEC_BY_KEY[k].label for k, v in patient.to_api_dict().items() if v is None],
        "explanation": explain(patient, row, prob) if include_explanation else None,
        "input": patient.to_api_dict(),
        "disclaimer": DISCLAIMER,
    }


def explain_patient(patient: PatientInput) -> dict:
    registry.require(complete=True)
    row = to_frame(patient)
    prob = model_probability(row)
    pred = classify(prob)
    return {
        "probability": round(prob, 4),
        "prediction": pred,
        "prediction_label": LABELS[pred],
        "threshold": round(threshold(), 4),
        "explanation": explain(patient, row, prob),
        "global_importance": (registry.global_shap or {}).get("importance", []),
        "input": patient.to_api_dict(),
    }


def global_explanation() -> dict:
    registry.require()
    if not registry.global_shap:
        raise AnalysisError("Global SHAP results not found. Run 'python -m app.ml.train'.", 503)
    return registry.global_shap


def _scenario(patient: PatientInput) -> dict:
    row = to_frame(patient)
    prob = model_probability(row)
    pred = classify(prob)
    unc = bootstrap_uncertainty(registry.bootstrap_models, row, threshold(), prob)
    return {
        "probability": round(prob, 4), "prediction": pred, "prediction_label": LABELS[pred],
        "uncertainty_lower": unc["lower"], "uncertainty_upper": unc["upper"],
        "borderline": unc["borderline"],
        "explanation": explain(patient, row, prob),
        "input": patient.to_api_dict(),
    }


def what_if(original: PatientInput, modified: PatientInput) -> dict:
    registry.require(complete=True)
    o, m = original.to_api_dict(), modified.to_api_dict()
    changed = [{"feature": k, "label": SPEC_BY_KEY[k].label, "original": o[k], "modified": m[k]}
               for k in o if o[k] != m[k]]
    if not changed:
        raise AnalysisError("No values were changed. Modify at least one input to compare scenarios.")
    a, b = _scenario(original), _scenario(modified)
    diff = b["probability"] - a["probability"]
    return {
        "original": a,
        "modified": b,
        "probability_change": round(diff, 4),
        "percentage_point_change": round(100 * diff, 2),
        "prediction_changed": a["prediction"] != b["prediction"],
        "changed_features": changed,
        "threshold": round(threshold(), 4),
        "note": ("Model sensitivity analysis: this shows how the MODEL's output responds to changed "
                 "inputs. It does not show that changing a real patient's values would change their "
                 "medical outcome."),
    }


def sensitivity(patient: PatientInput, feature: str, n_points: int) -> dict:
    registry.require(complete=True)
    spec = SPEC_BY_KEY[feature]
    r = registry.metadata["training_ranges"][feature]
    lo, hi = r["min"], r["max"]
    if spec.integer:
        values = np.unique(np.round(np.linspace(lo, hi, n_points)))
    else:
        values = np.linspace(lo, hi, n_points)

    base = to_frame(patient)
    grid = pd.concat([base] * len(values), ignore_index=True)
    grid[spec.csv_column] = values
    probs = registry.pipeline.predict_proba(grid)[:, 1]
    boot = np.array([m.predict_proba(grid)[:, 1] for m in registry.bootstrap_models])
    lower = np.percentile(boot, LOWER_PCT, axis=0)
    upper = np.percentile(boot, UPPER_PCT, axis=0)

    return {
        "feature": feature, "label": spec.label, "unit": spec.unit,
        "range_source": f"Training-data range of {spec.label} ({lo:g}–{hi:g} {spec.unit})",
        "current_value": getattr(patient, feature),
        "current_probability": round(model_probability(base), 4),
        "threshold": round(threshold(), 4),
        "points": [{"value": round(float(v), 3), "probability": round(float(p), 4),
                    "lower": round(float(l), 4), "upper": round(float(u), 4)}
                   for v, p, l, u in zip(values, probs, lower, upper)],
        "note": ("Individual conditional expectation: only this feature is varied, all others are held "
                 "at the patient's values. Shaded band = 90% range of the bootstrap models. "
                 "Shows model behaviour, not causal effect."),
    }


def example(rng: np.random.Generator | None = None) -> dict:
    """A random REAL patient from the held-out test set (for the 'Load Example' button)."""
    registry.require()
    df = registry.test_set
    if df is None or df.empty:
        raise AnalysisError("Test set not found. Run 'python -m app.ml.train'.", 503)
    # Required fields must hold real measurements (0 = not measured cannot be typed into the form)
    required_zero_cols = [s.csv_column for s in FEATURES if s.required and s.zero_means_missing]
    usable = df[(df[required_zero_cols] > 0).all(axis=1)]
    rng = rng or np.random.default_rng()
    idx = int(rng.integers(0, len(usable)))
    row = usable.iloc[idx]
    patient = {}
    for spec in FEATURES:
        v = float(row[spec.csv_column])
        if spec.zero_means_missing and v == 0:
            patient[spec.key] = None              # not measured -> left blank
        else:
            patient[spec.key] = int(v) if spec.integer else v
    return {
        "patient": patient,
        "recorded_outcome": int(row[TARGET_COLUMN]),
        "source": f"Held-out test set, row {int(usable.index[idx])} of {len(df)} "
                  "(never used for training)",
    }
