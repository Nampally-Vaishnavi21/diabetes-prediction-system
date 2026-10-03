"""GET /health — is the API running and is the model loaded?"""
from fastapi import APIRouter

from app.config import settings
from app.schemas.responses import HealthResponse
from app.services.model_registry import registry

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if registry.is_loaded:
        md = registry.metadata
        return HealthResponse(
            status="ok",
            api_version=settings.APP_VERSION,
            model_loaded=True,
            model_name=md.get("selected_model_display"),
            trained_at=md.get("trained_at"),
            message=("API is running and the model is loaded."
                     if registry.is_complete else
                     "Model loaded, but some artifacts (SHAP/bootstrap) are missing. "
                     "Re-run 'python -m app.ml.train'."),
        )
    return HealthResponse(
        status="ok",
        api_version=settings.APP_VERSION,
        model_loaded=False,
        message=registry.load_error or "Model not loaded.",
    )
