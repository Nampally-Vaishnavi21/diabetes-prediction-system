"""GET /model-info — everything the Model Information page shows, from real training outputs."""
from fastapi import APIRouter

from app.schemas.responses import ModelInfoResponse
from app.services.model_info import build_model_info

router = APIRouter(tags=["Model"])


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> dict:
    return build_model_info()
