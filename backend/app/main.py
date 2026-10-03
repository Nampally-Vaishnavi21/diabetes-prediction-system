"""
FastAPI application entry point.

Run from the backend/ folder:
    uvicorn app.main:app --reload

Interactive API docs: http://localhost:8000/docs

Two ways to run:
  * Development: React dev server (port 5173) calls this API on port 8000.
  * Deployed website: after `npm run build`, this same server also serves the
    React app from frontend/dist, and the frontend calls the API under /api.
    One process, one URL, no CORS needed.

Startup sequence (the "lifespan" function):
  1. configure logging
  2. load the trained model artifacts once (see services/model_registry.py)
  3. start serving requests
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import features, health, model_info, predict
from app.services.model_registry import registry
from app.utils.errors import register_exception_handlers
from app.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    registry.load()
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Academic/research prototype for diabetes risk prediction. "
            "Not a medical diagnostic device."
        ),
        lifespan=lifespan,
    )

    # CORS: lets the React dev server (a different origin/port) call this API.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path,
                    response.status_code, ms)
        return response

    register_exception_handlers(app)

    for router in (health.router, features.router, model_info.router, predict.router):
        app.include_router(router)                                       # e.g. /predict
        app.include_router(router, prefix="/api", include_in_schema=False)  # e.g. /api/predict

    mount_frontend(app)
    return app


def mount_frontend(app: FastAPI) -> None:
    """Serve the built React app (if present) so the whole system runs at one URL.

    React Router handles page addresses like /results in the browser, so any
    GET path that is not an API route or a real file returns index.html.
    """
    dist = settings.FRONTEND_DIST
    index = dist / "index.html"
    if not index.exists():
        @app.get("/", include_in_schema=False)
        def root():
            return {"message": f"{settings.APP_NAME} is running. See /docs for the API."}
        return

    logger.info("Serving the React app from %s", dist)
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    dist_root = dist.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Unknown API endpoint.")
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and dist_root in candidate.parents:
            return FileResponse(candidate)          # e.g. /favicon.svg
        return FileResponse(index)                  # every page of the React app


app = create_app()
