"""
Custom exceptions + FastAPI exception handlers.

WHAT: Turns every kind of failure into a clean JSON error with the right
      HTTP status code.
WHY:  The requirement says users must never see raw Python stack traces.
      Full technical details are written to the server log instead.

Status codes used:
  422  invalid / missing input            (RequestValidationError)
  503  model artifacts not available yet  (ModelNotLoadedError)
  404  unknown URL                        (Starlette HTTPException)
  500  anything unexpected                (Exception)
"""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.ml.features import SPEC_BY_KEY
from app.utils.logging import get_logger

logger = get_logger("errors")


class ModelNotLoadedError(Exception):
    """Raised when an endpoint needs the trained model but it is not available."""


class AnalysisError(Exception):
    """Raised for a known, explainable failure inside the ML services (e.g. bad feature name)."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": code, "message": message, "details": details}
    return JSONResponse(status_code=status, content=body)


def _friendly_validation_message(field: str | None, err: dict) -> str:
    """Use the per-feature message from features.py when we know the field."""
    if field in SPEC_BY_KEY:
        spec = SPEC_BY_KEY[field]
        if err.get("type") == "missing":
            return f"{spec.label} is required."
        return spec.error_message
    if err.get("type") == "extra_forbidden":
        return f"Unknown field '{field}' is not allowed."
    return err.get("msg", "Invalid value.")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            loc = [str(p) for p in err.get("loc", []) if p != "body"]
            # nested bodies (e.g. what-if) -> use the last element as the field name
            field = loc[-1] if loc else None
            details.append({"field": ".".join(loc) or None,
                            "message": _friendly_validation_message(field, err)})
        logger.info("Validation failed on %s: %s", request.url.path, details)
        return _error(422, "validation_error",
                      "Some inputs are missing or out of range. Please correct the highlighted fields.",
                      details)

    @app.exception_handler(ModelNotLoadedError)
    async def model_handler(request: Request, exc: ModelNotLoadedError):
        logger.warning("Model unavailable for %s: %s", request.url.path, exc)
        return _error(503, "model_not_loaded",
                      "The prediction model is not available. Train it first with "
                      "'python -m app.ml.train', then restart the backend.")

    @app.exception_handler(AnalysisError)
    async def analysis_handler(request: Request, exc: AnalysisError):
        logger.info("Analysis error on %s: %s", request.url.path, exc.message)
        return _error(exc.status_code, "analysis_error", exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException):
        code = "not_found" if exc.status_code == 404 else "http_error"
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        # Full traceback goes to the server log only
        logger.exception("Unexpected error on %s %s", request.method, request.url.path)
        return _error(500, "internal_error",
                      "An unexpected server error occurred. Please try again.")
