"""
Response schemas.

WHAT: Pydantic models describing what each endpoint returns.
WHY:  FastAPI checks every response against these models and documents them
      at /docs, so the frontend team (you) knows exactly what JSON to expect.
"""
from typing import Any, Optional

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str


class ErrorResponse(BaseModel):
    """Every error from this API has this shape - no Python stack traces."""
    error: str                          # machine-readable code, e.g. "validation_error"
    message: str                        # human-readable summary for the UI
    details: Optional[list[ErrorDetail]] = None


class HealthResponse(BaseModel):
    status: str                         # "ok" (API is up)
    api_version: str
    model_loaded: bool
    model_name: Optional[str] = None
    trained_at: Optional[str] = None
    message: str


class TrainingRange(BaseModel):
    min: float
    max: float


class FeatureInfoItem(BaseModel):
    key: str
    csv_column: str
    label: str
    unit: str
    min_value: float
    max_value: float
    step: float
    integer: bool
    required: bool
    zero_means_missing: bool
    placeholder: str
    description: str
    error_message: str
    training_range: Optional[TrainingRange] = None


class FeatureInfoResponse(BaseModel):
    features: list[FeatureInfoItem]
    target: str
    note: str


# Generic container used by later phases for JSON loaded from disk
JSONDict = dict[str, Any]


class ModelInfoResponse(BaseModel):
    """GET /model-info. Nested sections are passed through from the training JSON files."""
    selected_model: str
    selected_model_display: str
    selection_reason: str
    best_params: dict[str, Any]
    engineered_features: Optional[bool] = None
    trained_at: str
    sklearn_version: str
    threshold: dict[str, Any]
    calibration: dict[str, Any]
    dataset: dict[str, Any]
    feature_selection: dict[str, Any]
    cross_validation: dict[str, Any]
    test_evaluation: dict[str, Any]
    methodology: list[str]
    limitations: list[str]


# --------------------------------------------------------------------------- #
# Prediction / explanation / what-if responses
# --------------------------------------------------------------------------- #
class Contribution(BaseModel):
    feature: str
    label: str
    unit: str
    value: Optional[float] = None       # None = left blank and imputed
    imputed: bool
    shap_value: float
    direction: str                      # increases | decreases | neutral


class Explanation(BaseModel):
    method: str
    base_value: float
    prediction: float
    sum_check: float
    contributions: list[Contribution]
    top_increasing: list[str]
    top_decreasing: list[str]
    note: str


class Uncertainty(BaseModel):
    method: str
    n_models: int
    interval_level_pct: int
    lower: float
    upper: float
    median: float
    std: float
    width: float
    share_of_models_above_threshold: float
    borderline: bool
    point_estimate: float
    explanation: str
    caveat: str


class ModelVote(BaseModel):
    key: str
    display_name: str
    probability: float
    above_threshold: bool


class ModelAgreement(BaseModel):
    note: str
    votes: list[ModelVote]
    n_above_threshold: int
    n_models: int


class OODWarning(BaseModel):
    feature: str
    label: str
    value: float
    training_min: float
    training_max: float
    message: str


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    probability_pct: float
    threshold: float
    threshold_method: str
    model: str
    interpretation: str
    uncertainty: Uncertainty
    model_agreement: ModelAgreement
    ood_warnings: list[OODWarning]
    imputed_features: list[str]
    explanation: Optional[Explanation] = None
    input: dict[str, Optional[float]]
    disclaimer: str


class ExplainResponse(BaseModel):
    probability: float
    prediction: int
    prediction_label: str
    threshold: float
    explanation: Explanation
    global_importance: list[dict[str, Any]]
    input: dict[str, Optional[float]]


class GlobalExplanationResponse(BaseModel):
    method: str
    n_explained: int
    base_value: float
    importance: list[dict[str, Any]]
    beeswarm: list[dict[str, Any]]
    note: str


class ScenarioResult(BaseModel):
    probability: float
    prediction: int
    prediction_label: str
    uncertainty_lower: float
    uncertainty_upper: float
    borderline: bool
    explanation: Explanation
    input: dict[str, Optional[float]]


class ChangedFeature(BaseModel):
    feature: str
    label: str
    original: Optional[float] = None
    modified: Optional[float] = None


class WhatIfResponse(BaseModel):
    original: ScenarioResult
    modified: ScenarioResult
    probability_change: float
    percentage_point_change: float
    prediction_changed: bool
    changed_features: list[ChangedFeature]
    threshold: float
    note: str


class SensitivityPoint(BaseModel):
    value: float
    probability: float
    lower: float
    upper: float


class SensitivityResponse(BaseModel):
    feature: str
    label: str
    unit: str
    range_source: str
    current_value: Optional[float] = None
    current_probability: float
    threshold: float
    points: list[SensitivityPoint]
    note: str


class ExampleResponse(BaseModel):
    patient: dict[str, Optional[float]]
    recorded_outcome: int
    source: str
