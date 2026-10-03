"""
Phase 3 tests: data loading, preprocessing, leakage checks, saved artifacts.

Artifact tests need training to have been run first:
    python -m app.ml.train
"""
import json

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml.data import DatasetError, data_quality_report, load_dataset
from app.ml.features import CSV_COLUMNS, TARGET_COLUMN, ZERO_AS_MISSING_COLUMNS
from app.ml.preprocessing import FeatureEngineer, ZeroToNaN, build_preprocessor
from app.schemas.patient import PatientInput
from app.services.model_registry import ModelRegistry

ARTIFACTS_PRESENT = (settings.MODELS_DIR / settings.PIPELINE_FILE).exists()
needs_artifacts = pytest.mark.skipif(not ARTIFACTS_PRESENT,
                                     reason="Run 'python -m app.ml.train' first")


@pytest.fixture(scope="module")
def df():
    return load_dataset(settings.DATA_PATH)


# ---------- data loading / validation ----------
def test_dataset_loads_with_expected_shape(df):
    assert df.shape == (768, 9)
    assert list(df.columns) == CSV_COLUMNS + [TARGET_COLUMN]


def test_missing_file_gives_download_help(tmp_path):
    with pytest.raises(DatasetError, match="kaggle"):
        load_dataset(tmp_path / "nope.csv")


def test_wrong_columns_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"a": range(200), "b": range(200)}).to_csv(p, index=False)
    with pytest.raises(DatasetError, match="missing columns"):
        load_dataset(p)


def test_non_binary_target_rejected(tmp_path, df):
    bad = df.copy()
    bad.loc[0, TARGET_COLUMN] = 2
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(DatasetError, match="only 0 and 1"):
        load_dataset(p)


def test_quality_report_matches_known_dataset_facts(df):
    q = data_quality_report(df)
    assert q["class_counts"] == {"non_diabetic_0": 500, "diabetic_1": 268}
    assert q["n_duplicates"] == 0
    zeros = {c: v["count"] for c, v in q["zero_encoded_missing"].items()}
    assert zeros == {"Glucose": 5, "BloodPressure": 35, "SkinThickness": 227,
                     "Insulin": 374, "BMI": 11}


# ---------- preprocessing ----------
def test_zero_to_nan_only_touches_listed_columns(df):
    out = ZeroToNaN(columns=ZERO_AS_MISSING_COLUMNS).fit(df[CSV_COLUMNS]).transform(df[CSV_COLUMNS])
    assert out["Insulin"].isna().sum() == 374
    assert out["Pregnancies"].isna().sum() == 0          # 0 pregnancies is a real value
    assert (out["Pregnancies"] == 0).sum() == (df["Pregnancies"] == 0).sum()


def test_imputer_learns_from_training_rows_only(df):
    """Leakage check: the median must come from the rows passed to fit(), not the full data."""
    train = df[CSV_COLUMNS].iloc[:300]
    pre = build_preprocessor().fit(train)
    learned = dict(zip(CSV_COLUMNS, pre.named_steps["impute"].statistics_))
    expected = train["Insulin"].replace(0, np.nan).median()
    full = df["Insulin"].replace(0, np.nan).median()
    assert learned["Insulin"] == pytest.approx(expected)
    assert expected != pytest.approx(full)   # proves the test can detect leakage


def test_preprocessor_output_has_no_missing_values(df):
    out = build_preprocessor().fit_transform(df[CSV_COLUMNS])
    assert not out.isna().any().any()
    assert "missingindicator_Insulin" in out.columns


def test_feature_engineer_toggle(df):
    X = df[CSV_COLUMNS].replace(0, 1.0)
    off = FeatureEngineer(enabled=False).fit(X).transform(X)
    on = FeatureEngineer(enabled=True).fit(X).transform(X)
    assert "Glucose_x_BMI" not in off.columns and "Glucose_x_BMI" in on.columns
    assert on["Insulin"].iloc[0] == pytest.approx(np.log1p(X["Insulin"].iloc[0]))


# ---------- saved artifacts ----------
@needs_artifacts
def test_registry_loads_artifacts():
    reg = ModelRegistry()
    assert reg.load() is True
    assert reg.is_loaded
    assert reg.metadata["selected_model"] in reg.all_models


@needs_artifacts
def test_pipeline_predicts_from_raw_api_input():
    reg = ModelRegistry()
    reg.load()
    p = PatientInput(pregnancies=2, glucose=120, blood_pressure=70, skin_thickness=20,
                     insulin=79, bmi=28.5, diabetes_pedigree=0.35, age=25)
    row = pd.DataFrame([p.to_csv_row()])
    prob = reg.pipeline.predict_proba(row)[0, 1]
    assert 0.0 <= prob <= 1.0

    # Higher glucose and BMI should not lower the estimate for this model family
    high = row.copy()
    high[["Glucose", "BMI"]] = [190, 40]
    assert reg.pipeline.predict_proba(high)[0, 1] > prob


@needs_artifacts
def test_pipeline_handles_blank_optional_fields():
    reg = ModelRegistry()
    reg.load()
    p = PatientInput(pregnancies=2, glucose=120, blood_pressure=70,
                     bmi=28.5, diabetes_pedigree=0.35, age=25)
    prob = reg.pipeline.predict_proba(pd.DataFrame([p.to_csv_row()]))[0, 1]
    assert 0.0 <= prob <= 1.0


@needs_artifacts
def test_test_set_was_held_out():
    test = pd.read_csv(settings.MODELS_DIR / "test_set.csv")
    md = json.loads((settings.MODELS_DIR / settings.METADATA_FILE).read_text())
    assert len(test) == md["dataset"]["n_test"] == 154
    assert md["dataset"]["n_train"] + md["dataset"]["n_test"] == 768


@needs_artifacts
def test_cv_results_written_for_every_model():
    cv = json.loads((settings.METRICS_DIR / "cv_results.json").read_text())
    names = {m["key"] for m in cv["models"]}
    assert {"logistic_regression", "decision_tree", "random_forest", "svm",
            "gradient_boosting", "soft_voting_ensemble"} <= names
    for m in cv["models"]:
        assert 0.5 < m["cv"]["roc_auc"]["mean"] <= 1.0
