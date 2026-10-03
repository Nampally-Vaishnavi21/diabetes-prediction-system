"""
Builds the /model-info response from the JSON files written by training.

Nothing here is hardcoded: every metric is read from outputs/metrics/*.json,
which `python -m app.ml.train` generates from real experiments. Only the
descriptive text (dataset source, limitations) is fixed wording.
"""
import json
from pathlib import Path

from app.config import settings
from app.services.model_registry import registry
from app.utils.errors import AnalysisError

METRIC_FILES = {
    "data_quality": "data_quality.json",
    "feature_selection": "feature_selection.json",
    "cv_results": "cv_results.json",
    "evaluation": "evaluation.json",
}

DATASET_SOURCE = {
    "name": "Pima Indians Diabetes Database",
    "origin": "National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)",
    "url": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
    "license": "CC0: Public Domain",
    "population": "Females aged 21 or older of Pima Indian heritage (Arizona, USA)",
}

LIMITATIONS = [
    "Small dataset (768 patients) from a single population: women of Pima heritage aged 21+. "
    "Results do not generalise to men, children or other populations.",
    "Several features use 0 to mean 'not measured' (Insulin is missing for about half the patients); "
    "these are imputed, which adds uncertainty.",
    "The test set has only 154 patients, so test metrics have wide confidence intervals.",
    "The model finds statistical associations, not causes.",
    "This is an academic prototype, not a validated clinical tool.",
]

METHODOLOGY = [
    "Load and validate the CSV; check duplicates; report zero-encoded missing values and IQR outliers.",
    "Stratified 80/20 train/test split; the test set is used only once, for final evaluation.",
    "Preprocessing inside an sklearn Pipeline: zeros -> missing, median imputation with "
    "missing-value indicators, optional engineered features, standard scaling.",
    "Feature-selection analysis (ANOVA F, mutual information, RFECV) on the training set.",
    "Six model families tuned with GridSearchCV and stratified 5-fold cross-validation (ROC-AUC).",
    "Soft-voting ensemble of the top three tuned models.",
    "Model selection by CV ROC-AUC with near-ties broken by CV Brier score.",
    "Calibration check: an extra sigmoid layer is kept only if it lowers CV Brier score.",
    "Decision threshold chosen by Youden's J on out-of-fold training predictions.",
    "Final evaluation on the held-out test set with a 1000-sample bootstrap 95% CI.",
]


def _load(name: str) -> dict:
    path: Path = settings.METRICS_DIR / METRIC_FILES[name]
    if not path.exists():
        raise AnalysisError(
            f"Metrics file '{path.name}' not found. Run 'python -m app.ml.train' to generate it.",
            status_code=503,
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_info() -> dict:
    registry.require()
    md = registry.metadata
    dq, fs, cv, ev = (_load(n) for n in METRIC_FILES)

    return {
        "selected_model": md["selected_model"],
        "selected_model_display": md["selected_model_display"],
        "selection_reason": md["selection_reason"],
        "best_params": md["best_params"],
        "engineered_features": md.get("engineered_features"),
        "trained_at": md["trained_at"],
        "sklearn_version": md["sklearn_version"],
        "threshold": ev["threshold"],
        "calibration": {**ev["calibration"], "summary": md.get("calibration_summary")},
        "dataset": {
            **DATASET_SOURCE,
            "n_rows": dq["n_rows"],
            "n_features": dq["n_features"],
            "n_train": md["dataset"]["n_train"],
            "n_test": md["dataset"]["n_test"],
            "class_counts": dq["class_counts"],
            "positive_rate_pct": dq["positive_rate_pct"],
            "n_duplicates": dq["n_duplicates"],
            "zero_encoded_missing": dq["zero_encoded_missing"],
            "rows_with_any_missing": dq["rows_with_any_missing"],
            "feature_stats": dq["feature_stats"],
            "correlation_matrix": dq["correlation_matrix"],
            "outlier_policy": dq["outlier_policy"],
        },
        "feature_selection": fs,
        "cross_validation": cv,
        "test_evaluation": {
            "test_set": ev["test_set"],
            "final_model": ev["final_model"],
            "uncertainty": ev.get("uncertainty"),
            "models": ev["models"],
        },
        "methodology": METHODOLOGY,
        "limitations": LIMITATIONS,
    }
