"""
GET /feature-info — field definitions for the React form.

The frontend builds its form and its validation rules from this response,
so frontend and backend limits are always identical. When the model is
loaded, each feature also includes the range seen in the training data,
used for out-of-distribution warnings.
"""
from fastapi import APIRouter

from app.ml.features import FEATURES, TARGET_COLUMN, feature_as_dict
from app.schemas.responses import FeatureInfoResponse
from app.services.model_registry import registry

router = APIRouter(tags=["System"])


@router.get("/feature-info", response_model=FeatureInfoResponse)
def feature_info() -> FeatureInfoResponse:
    ranges = (registry.metadata or {}).get("training_ranges", {}) if registry.is_loaded else {}
    return FeatureInfoResponse(
        features=[feature_as_dict(f, ranges.get(f.key)) for f in FEATURES],
        target=TARGET_COLUMN,
        note=(
            "Valid ranges reject physiologically implausible input. Values that are valid "
            "but outside the training range are accepted with an out-of-distribution warning."
        ),
    )
