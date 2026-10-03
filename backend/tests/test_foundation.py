"""
Phase 2 tests: API foundation (health, feature-info, validation, error handling, CORS).

Run from backend/:   pytest
"""
import math

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app
from app.ml.features import FEATURE_KEYS, FEATURES
from app.schemas.patient import PatientInput
from app.services.model_registry import registry

VALID = {
    "pregnancies": 2, "glucose": 120, "blood_pressure": 70, "skin_thickness": 20,
    "insulin": 79, "bmi": 28.5, "diabetes_pedigree": 0.35, "age": 25,
}


@pytest.fixture()
def app_with_test_routes(tmp_path, monkeypatch):
    """A fresh API-only app plus routes that exist only for testing the error handlers."""
    from app.config import settings
    monkeypatch.setattr(settings, "FRONTEND_DIST", tmp_path / "no-frontend-build")
    app = create_app()

    @app.post("/_test/patient")
    def echo_patient(p: PatientInput):
        return p.to_api_dict()

    @app.get("/_test/needs-model")
    def needs_model():
        registry.require()
        return {"ok": True}

    @app.get("/_test/crash")
    def crash():
        raise RuntimeError("secret internal detail")

    return app


@pytest.fixture()
def client(app_with_test_routes):
    with TestClient(app_with_test_routes, raise_server_exceptions=False) as c:
        yield c


# ---------- /health ----------
def test_health_without_model(client, tmp_path):
    registry.load(tmp_path)            # empty folder -> no model
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert "python -m app.ml.train" in body["message"]


# ---------- /feature-info ----------
def test_feature_info_lists_all_features(client):
    r = client.get("/feature-info")
    assert r.status_code == 200
    body = r.json()
    assert [f["key"] for f in body["features"]] == FEATURE_KEYS
    assert len(body["features"]) == 8
    assert body["target"] == "Outcome"
    optional = {f["key"] for f in body["features"] if not f["required"]}
    assert optional == {"skin_thickness", "insulin"}


# ---------- PatientInput schema ----------
def test_valid_patient_accepted():
    p = PatientInput(**VALID)
    row = p.to_csv_row()
    assert row["Glucose"] == 120.0 and row["Age"] == 25.0
    assert list(row.keys()) == [f.csv_column for f in FEATURES]


def test_optional_fields_become_nan():
    data = {k: v for k, v in VALID.items() if k not in ("insulin", "skin_thickness")}
    row = PatientInput(**data).to_csv_row()
    assert math.isnan(row["Insulin"]) and math.isnan(row["SkinThickness"])


@pytest.mark.parametrize("field,value", [
    ("age", -5), ("age", 10), ("age", 30.5), ("bmi", 500), ("bmi", 0),
    ("glucose", 0), ("glucose", 1000), ("pregnancies", -1), ("pregnancies", 2.5),
    ("blood_pressure", 10), ("diabetes_pedigree", 9), ("insulin", 5000),
    ("skin_thickness", -3), ("glucose", float("nan")), ("bmi", float("inf")),
    ("glucose", "abc"),
])
def test_invalid_values_rejected(field, value):
    with pytest.raises(ValidationError):
        PatientInput(**{**VALID, field: value})


def test_missing_required_rejected():
    data = dict(VALID)
    del data["glucose"]
    with pytest.raises(ValidationError):
        PatientInput(**data)


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        PatientInput(**VALID, cholesterol=200)


# ---------- error handlers ----------
def test_validation_error_is_friendly_json(client):
    r = client.post("/_test/patient", json={**VALID, "age": -5, "bmi": 500})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error"
    messages = {d["field"]: d["message"] for d in body["details"]}
    assert messages["age"].startswith("Please enter a valid age")
    assert messages["bmi"].startswith("Please enter a valid BMI")


def test_missing_field_message(client):
    data = dict(VALID)
    del data["glucose"]
    r = client.post("/_test/patient", json=data)
    assert r.status_code == 422
    assert r.json()["details"][0]["message"] == "Glucose is required."


def test_valid_body_passes_through(client):
    r = client.post("/_test/patient", json=VALID)
    assert r.status_code == 200
    assert r.json()["glucose"] == 120


def test_model_not_loaded_returns_503(client, tmp_path):
    registry.load(tmp_path)
    r = client.get("/_test/needs-model")
    assert r.status_code == 503
    assert r.json()["error"] == "model_not_loaded"


def test_unexpected_error_hides_details(client):
    r = client.get("/_test/crash")
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "internal_error"
    assert "secret" not in r.text and "Traceback" not in r.text


def test_unknown_route_404_json(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"


# ---------- CORS ----------
def test_cors_preflight_allows_frontend(client):
    r = client.options("/feature-info", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
    })
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_blocks_unknown_origin(client):
    r = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


# ---------- deployed-website mode ----------
def test_api_also_available_under_api_prefix(client):
    r = client.get("/api/feature-info")
    assert r.status_code == 200
    assert len(r.json()["features"]) == 8


def test_serves_react_app_when_built(tmp_path, monkeypatch):
    from app.config import settings
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<div id=root>react app</div>")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "favicon.svg").write_text("<svg/>")
    monkeypatch.setattr(settings, "FRONTEND_DIST", tmp_path)
    with TestClient(create_app()) as c:
        for page in ("/", "/predict", "/results", "/what-if", "/model"):
            r = c.get(page)
            assert r.status_code == 200 and "react app" in r.text, page
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/favicon.svg").text == "<svg/>"
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.get("/api/nope").status_code == 404
        assert c.get("/../backend/app/main.py").text.startswith("<div")   # no path traversal
