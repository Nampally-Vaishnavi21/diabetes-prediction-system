"""
Request schema for patient input.

WHAT: A Pydantic model describing the JSON body sent by the frontend.
WHY:  FastAPI uses it to validate every request automatically. A wrong type,
      missing required field or out-of-range value is rejected with HTTP 422
      BEFORE any ML code runs. This is the "never trust the frontend" layer.
HOW:  The limits (ge = greater-or-equal, le = less-or-equal) are taken from
      app/ml/features.py, so they always match the frontend form.
"""
import math
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.ml.features import FEATURES, SPEC_BY_KEY


def _field(key: str):
    s = SPEC_BY_KEY[key]
    return Field(
        default=... if s.required else None,
        ge=s.min_value,
        le=s.max_value,
        description=f"{s.description} Unit: {s.unit}. Valid range {s.min_value}–{s.max_value}.",
    )


class PatientInput(BaseModel):
    # forbid unknown keys; reject NaN / Infinity; strip nothing silently
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "pregnancies": 2,
                "glucose": 120,
                "blood_pressure": 70,
                "skin_thickness": 20,
                "insulin": 79,
                "bmi": 28.5,
                "diabetes_pedigree": 0.35,
                "age": 25,
            }
        },
    )

    pregnancies: int = _field("pregnancies")
    glucose: float = _field("glucose")
    blood_pressure: float = _field("blood_pressure")
    skin_thickness: Optional[float] = _field("skin_thickness")
    insulin: Optional[float] = _field("insulin")
    bmi: float = _field("bmi")
    diabetes_pedigree: float = _field("diabetes_pedigree")
    age: int = _field("age")

    def to_csv_row(self) -> dict:
        """
        Convert to a dict keyed by the ORIGINAL CSV column names, which is
        what the trained pipeline expects. Blank optional fields become NaN
        so the pipeline's imputer handles them exactly as in training.
        """
        row = {}
        for spec in FEATURES:
            value = getattr(self, spec.key)
            row[spec.csv_column] = np.nan if value is None else float(value)
        return row

    def to_api_dict(self) -> dict:
        return {
            k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in self.model_dump().items()
        }
