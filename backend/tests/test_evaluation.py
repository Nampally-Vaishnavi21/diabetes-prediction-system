"""
Phase 4 tests: metric functions, threshold, calibration outputs and GET /model-info.

The key test recomputes the final model's test metrics from the saved pipeline
and the saved test set, and checks they equal what /model-info reports. That
proves the displayed numbers come from the real model, not hardcoded values.
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.ml.evaluate import (bootstrap_ci, calibration_points, compute_metrics, pr_points,
                             roc_points, youden_threshold)
from app.ml.features import CSV_COLUMNS, TARGET_COLUMN
from app.services.model_registry import registry

ARTIFACTS_PRESENT = (settings.METRICS_DIR / "evaluation.json").exists() and \
    (settings.MODELS_DIR / settings.PIPELINE_FILE).exists()
needs_artifacts = pytest.mark.skipif(not ARTIFACTS_PRESENT,
                                     reason="Run 'python -m app.ml.train' first")


# ---------- metric functions on hand-checkable data ----------
def test_compute_metrics_hand_example():
    y = [0, 0, 1, 1]
    p = [0.1, 0.6, 0.4, 0.9]
    m = compute_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == 0.5 and m["specificity"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == 0.75
    assert m["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4, abs=1e-4)


def test_threshold_changes_predictions():
    y = [0, 0, 1, 1]
    p = [0.1, 0.35, 0.4, 0.9]
    assert compute_metrics(y, p, 0.5)["recall"] == 0.5
    assert compute_metrics(y, p, 0.38)["recall"] == 1.0


def test_youden_on_separable_data():
    best = youden_threshold([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert best["youden_j"] == 1.0
    assert 0.3 < best["threshold"] <= 0.7


def test_curves_have_valid_points():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
    roc = roc_points(y, p)
    assert roc[0] == {"fpr": 0.0, "tpr": 0.0} and roc[-1] == {"fpr": 1.0, "tpr": 1.0}
    assert all(0 <= pt["precision"] <= 1 for pt in pr_points(y, p))
    assert 1 <= len(calibration_points(y, p)) <= 10


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.3 + rng.random(300) * 0.7, 0, 1)
    ci = bootstrap_ci(y, p, n_boot=300)
    point = compute_metrics(y, p)["roc_auc"]
    assert ci["roc_auc_95ci"][0] <= point <= ci["roc_auc_95ci"][1]


# ---------- /model-info ----------
@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_model_info_503_without_model(client, tmp_path):
    registry.load(tmp_path)
    r = client.get("/model-info")
    assert r.status_code == 503
    registry.load()   # restore for later tests


@needs_artifacts
def test_model_info_structure(client):
    registry.load()
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    for key in ("selected_model_display", "threshold", "calibration", "dataset",
                "cross_validation", "test_evaluation", "methodology", "limitations"):
        assert key in body
    assert body["dataset"]["n_rows"] == 768
    assert len(body["test_evaluation"]["models"]) == 7
    cm = body["test_evaluation"]["final_model"]["metrics_at_threshold"]["confusion_matrix"]
    assert sum(cm.values()) == body["test_evaluation"]["test_set"]["n"] == 154
    assert body["calibration"]["cv_brier_uncalibrated"]["mean"] > 0


@needs_artifacts
def test_reported_metrics_match_saved_model(client):
    """Recompute test metrics from the saved pipeline: they must equal /model-info."""
    registry.load()
    body = client.get("/model-info").json()
    test = pd.read_csv(settings.MODELS_DIR / "test_set.csv")
    prob = registry.pipeline.predict_proba(test[CSV_COLUMNS])[:, 1]
    thr = registry.metadata["threshold"]
    recomputed = compute_metrics(test[TARGET_COLUMN], prob, thr)
    reported = body["test_evaluation"]["final_model"]["metrics_at_threshold"]
    for k in ("roc_auc", "pr_auc", "brier", "accuracy", "recall", "specificity", "f1"):
        assert recomputed[k] == pytest.approx(reported[k], abs=1e-4), k
    assert body["threshold"]["value"] == pytest.approx(thr)


@needs_artifacts
def test_threshold_not_from_test_set():
    ev = json.loads((settings.METRICS_DIR / "evaluation.json").read_text())
    assert "out-of-fold" in ev["threshold"]["method"]
    assert 0.0 < ev["threshold"]["value"] < 1.0


@needs_artifacts
def test_figures_written():
    for name in ("roc_curves.png", "pr_curves.png", "confusion_matrix.png", "calibration_curve.png"):
        assert (settings.FIGURES_DIR / name).stat().st_size > 1000
