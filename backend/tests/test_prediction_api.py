"""
Phase 5 tests: /predict, /explain, /explain/global, /what-if, /what-if/sensitivity, /example.

These tests check the API against the saved model directly, so they would fail
if any value were hardcoded or simulated.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.ml.features import CSV_COLUMNS
from app.schemas.patient import PatientInput
from app.services.model_registry import registry

ARTIFACTS_PRESENT = (settings.MODELS_DIR / settings.BOOTSTRAP_FILE).exists()
pytestmark = pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="Run 'python -m app.ml.train' first")

P = {"pregnancies": 2, "glucose": 150, "blood_pressure": 70, "skin_thickness": 20,
     "insulin": 79, "bmi": 32, "diabetes_pedigree": 0.35, "age": 45}


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        registry.load()
        yield c


def direct_probability(patient: dict) -> float:
    row = pd.DataFrame([PatientInput(**patient).to_csv_row()], columns=CSV_COLUMNS)
    return float(registry.pipeline.predict_proba(row)[0, 1])


# ---------- /predict ----------
def test_predict_returns_real_model_probability(client):
    r = client.post("/predict", json=P)
    assert r.status_code == 200
    b = r.json()
    assert b["probability"] == pytest.approx(direct_probability(P), abs=1e-4)
    assert b["prediction"] == int(b["probability"] >= b["threshold"])
    assert b["threshold"] == pytest.approx(registry.metadata["threshold"], abs=1e-4)
    assert "approximately" in b["interpretation"] and "not a diagnosis" in b["interpretation"]
    assert "not a medical diagnostic device" in b["disclaimer"]


def test_different_inputs_give_different_outputs(client):
    low = client.post("/predict", json={**P, "glucose": 85, "bmi": 22, "age": 23}).json()
    high = client.post("/predict", json={**P, "glucose": 190, "bmi": 42}).json()
    assert low["probability"] < high["probability"]


def test_shap_values_are_additive(client):
    b = client.post("/predict", json=P).json()
    e = b["explanation"]
    total = e["base_value"] + sum(c["shap_value"] for c in e["contributions"])
    assert total == pytest.approx(b["probability"], abs=1e-3)
    assert len(e["contributions"]) == 8
    mags = [abs(c["shap_value"]) for c in e["contributions"]]
    assert mags == sorted(mags, reverse=True)


def test_uncertainty_is_consistent(client):
    u = client.post("/predict", json=P).json()["uncertainty"]
    assert 0 <= u["lower"] <= u["median"] <= u["upper"] <= 1
    assert u["n_models"] == 50
    thr = registry.metadata["threshold"]
    assert u["borderline"] == (u["lower"] < thr <= u["upper"])


def test_model_agreement_lists_all_models(client):
    a = client.post("/predict", json=P).json()["model_agreement"]
    assert a["n_models"] == len(registry.all_models) == 7
    assert a["n_above_threshold"] == sum(v["above_threshold"] for v in a["votes"])


def test_blank_optional_fields_are_imputed(client):
    b = client.post("/predict", json={**P, "insulin": None, "skin_thickness": None}).json()
    assert set(b["imputed_features"]) == {"Insulin (2-hour serum)", "Skin Thickness (triceps)"}
    imputed = [c for c in b["explanation"]["contributions"] if c["imputed"]]
    assert len(imputed) == 2


def test_out_of_distribution_warning(client):
    b = client.post("/predict", json={**P, "glucose": 300}).json()
    assert [w["feature"] for w in b["ood_warnings"]] == ["glucose"]
    assert client.post("/predict", json=P).json()["ood_warnings"] == []


def test_predict_rejects_invalid_input(client):
    r = client.post("/predict", json={**P, "age": -4})
    assert r.status_code == 422
    assert r.json()["details"][0]["message"].startswith("Please enter a valid age")


def test_predict_503_without_model(client, tmp_path):
    registry.load(tmp_path)
    try:
        assert client.post("/predict", json=P).status_code == 503
    finally:
        registry.load()


# ---------- /explain ----------
def test_explain_matches_predict(client):
    p = client.post("/predict", json=P).json()
    e = client.post("/explain", json=P).json()
    assert e["probability"] == p["probability"]
    assert [c["shap_value"] for c in e["explanation"]["contributions"]] == \
           [c["shap_value"] for c in p["explanation"]["contributions"]]
    assert len(e["global_importance"]) == 8


def test_global_explanation(client):
    g = client.get("/explain/global").json()
    vals = [d["mean_abs_shap"] for d in g["importance"]]
    assert vals == sorted(vals, reverse=True)
    assert g["n_explained"] == 154
    assert all(len(f["points"]) == 154 for f in g["beeswarm"])


# ---------- /what-if ----------
def test_what_if_uses_real_model(client):
    mod = {**P, "glucose": 120, "bmi": 28}
    b = client.post("/what-if", json={"original": P, "modified": mod}).json()
    assert b["original"]["probability"] == pytest.approx(direct_probability(P), abs=1e-4)
    assert b["modified"]["probability"] == pytest.approx(direct_probability(mod), abs=1e-4)
    assert b["probability_change"] == pytest.approx(
        b["modified"]["probability"] - b["original"]["probability"], abs=1e-4)
    assert {c["feature"] for c in b["changed_features"]} == {"glucose", "bmi"}
    assert "does not show" in b["note"]


def test_what_if_requires_a_change(client):
    r = client.post("/what-if", json={"original": P, "modified": P})
    assert r.status_code == 400


def test_what_if_validates_both_scenarios(client):
    r = client.post("/what-if", json={"original": P, "modified": {**P, "bmi": 900}})
    assert r.status_code == 422
    assert r.json()["details"][0]["field"] == "modified.bmi"


# ---------- /what-if/sensitivity ----------
def test_sensitivity_curve(client):
    b = client.post("/what-if/sensitivity", json={"patient": P, "feature": "glucose", "n_points": 25}).json()
    vals = [p["value"] for p in b["points"]]
    assert vals == sorted(vals) and len(vals) == 25
    r = registry.metadata["training_ranges"]["glucose"]
    assert vals[0] == pytest.approx(r["min"]) and vals[-1] == pytest.approx(r["max"])
    assert all(p["lower"] <= p["upper"] for p in b["points"])
    assert b["current_probability"] == pytest.approx(direct_probability(P), abs=1e-4)


def test_sensitivity_integer_feature(client):
    b = client.post("/what-if/sensitivity", json={"patient": P, "feature": "age"}).json()
    assert all(float(p["value"]).is_integer() for p in b["points"])


def test_sensitivity_rejects_unknown_feature(client):
    r = client.post("/what-if/sensitivity", json={"patient": P, "feature": "cholesterol"})
    assert r.status_code == 422


# ---------- /example ----------
def test_example_is_a_real_valid_test_row(client):
    for _ in range(5):
        b = client.get("/example").json()
        patient = PatientInput(**b["patient"])          # must pass validation
        assert b["recorded_outcome"] in (0, 1)
        test = registry.test_set
        match = test[np.isclose(test["Glucose"], patient.glucose) &
                     np.isclose(test["BMI"], patient.bmi) &
                     np.isclose(test["Age"], patient.age)]
        assert len(match) >= 1
