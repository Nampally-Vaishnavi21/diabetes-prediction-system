"""Request bodies for what-if and sensitivity analysis."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.ml.features import FEATURE_KEYS
from app.schemas.patient import PatientInput

FeatureKey = Literal[tuple(FEATURE_KEYS)]  # e.g. "glucose", "bmi", ...


class WhatIfRequest(BaseModel):
    """Both scenarios are full, validated patient records."""
    model_config = ConfigDict(extra="forbid")
    original: PatientInput
    modified: PatientInput


class SensitivityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient: PatientInput
    feature: FeatureKey
    n_points: int = Field(default=40, ge=5, le=100)
