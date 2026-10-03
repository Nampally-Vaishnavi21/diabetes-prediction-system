"""
Central configuration.

WHAT: One place that holds file paths and settings.
WHY:  Avoids hardcoding paths/URLs across many files; values can be changed
      through environment variables (or a .env file) without editing code.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# backend/ folder (this file is backend/app/config.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent

# Load backend/.env if it exists (does nothing otherwise)
load_dotenv(BACKEND_DIR / ".env")


def _path(env_name: str, default: str) -> Path:
    """Read a path from the environment; relative paths are resolved from backend/."""
    value = Path(os.getenv(env_name, default))
    return value if value.is_absolute() else (BACKEND_DIR / value).resolve()


class Settings:
    APP_NAME = "Diabetes Prediction System API"
    APP_VERSION = "1.0.0"

    CORS_ORIGINS = [
        o.strip()
        for o in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if o.strip()
    ]
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    DATA_PATH = _path("DATA_PATH", "data/raw/diabetes.csv")
    MODELS_DIR = _path("MODELS_DIR", "models")
    OUTPUTS_DIR = _path("OUTPUTS_DIR", "../outputs")
    # Built React app (npm run build). If it exists, the API also serves the website.
    FRONTEND_DIST = _path("FRONTEND_DIST", "../frontend/dist")

    # Artifact file names (written by app/ml/train.py, read by the API)
    PIPELINE_FILE = "final_pipeline.joblib"
    BOOTSTRAP_FILE = "bootstrap_ensemble.joblib"
    ALL_MODELS_FILE = "all_models.joblib"
    SHAP_BACKGROUND_FILE = "shap_background.joblib"
    METADATA_FILE = "model_metadata.json"

    @property
    def METRICS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "metrics"

    @property
    def FIGURES_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "figures"


settings = Settings()
