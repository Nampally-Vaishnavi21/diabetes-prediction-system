"""
Model registry — loads the trained artifacts ONCE when the API starts.

WHAT: Holds the trained pipeline and related files in memory.
WHY:  Loading a model from disk on every request would be slow. We load at
      startup and every request reuses the same objects.
HOW:  `load()` reads the files written by `python -m app.ml.train`.
      If they are missing or corrupt, the API still starts (so /health works
      and can tell the user what is wrong) but prediction endpoints return
      HTTP 503 with a clear message.

Artifacts (all in backend/models/):
  final_pipeline.joblib      preprocessing + calibrated classifier (REQUIRED)
  model_metadata.json        selected model, threshold, ranges, etc. (REQUIRED)
  bootstrap_ensemble.joblib  bootstrap copies for uncertainty
  all_models.joblib          every tuned model, for model-disagreement view
  shap_background.joblib     background rows for SHAP
  test_set.csv               held-out rows used by "Load Example"
Also read: outputs/metrics/shap_global.json (precomputed global SHAP)
"""
import json
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd

from app.config import settings
from app.ml.explain import build_explainer, shap_values_for
from app.utils.errors import ModelNotLoadedError
from app.utils.logging import get_logger

logger = get_logger("model_registry")


class ModelRegistry:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.pipeline: Any = None
        self.metadata: Optional[dict] = None
        self.bootstrap_models: Optional[list] = None
        self.all_models: Optional[dict] = None
        self.shap_background: Any = None
        self.explainer: Any = None
        self.global_shap: Optional[dict] = None
        self.test_set: Any = None
        self.load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None and self.metadata is not None

    @property
    def is_complete(self) -> bool:
        """All artifacts needed by /predict are present."""
        return self.is_loaded and self.explainer is not None and bool(self.bootstrap_models) \
            and bool(self.all_models)

    def load(self, models_dir: Optional[Path] = None) -> bool:
        """Load artifacts. Returns True on success; never raises."""
        self.reset()
        models_dir = Path(models_dir or settings.MODELS_DIR)
        pipeline_path = models_dir / settings.PIPELINE_FILE
        metadata_path = models_dir / settings.METADATA_FILE

        if not pipeline_path.exists() or not metadata_path.exists():
            self.load_error = (
                f"Model artifacts not found in '{models_dir}'. "
                "Run 'python -m app.ml.train' from the backend folder."
            )
            logger.warning(self.load_error)
            return False

        try:
            self.pipeline = joblib.load(pipeline_path)
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            optional = {
                "bootstrap_models": settings.BOOTSTRAP_FILE,
                "all_models": settings.ALL_MODELS_FILE,
                "shap_background": settings.SHAP_BACKGROUND_FILE,
            }
            for attr, filename in optional.items():
                path = models_dir / filename
                if path.exists():
                    setattr(self, attr, joblib.load(path))
                else:
                    logger.warning("Optional artifact missing: %s", path)

            test_path = models_dir / "test_set.csv"
            if test_path.exists():
                self.test_set = pd.read_csv(test_path)

            global_path = settings.METRICS_DIR / "shap_global.json"
            if global_path.exists():
                self.global_shap = json.loads(global_path.read_text(encoding="utf-8"))

            if self.shap_background is not None:
                # Build the SHAP explainer once and run it on one row so the first
                # user request is not slowed down by one-time initialisation.
                self.explainer = build_explainer(self.pipeline, self.shap_background)
                shap_values_for(self.explainer, self.shap_background.iloc[[0]])

            logger.info("Loaded model '%s' (trained %s)",
                        self.metadata.get("selected_model"), self.metadata.get("trained_at"))
            return True
        except Exception as exc:  # corrupt file, sklearn version mismatch, ...
            logger.exception("Failed to load model artifacts")
            self.reset()
            self.load_error = (
                f"Model artifacts could not be loaded ({type(exc).__name__}). "
                "Re-run 'python -m app.ml.train' with the installed library versions."
            )
            return False

    def require(self, complete: bool = False) -> "ModelRegistry":
        """Call at the start of any endpoint that needs the model.

        complete=True also requires the SHAP explainer and bootstrap/comparison models.
        """
        if not self.is_loaded or (complete and not self.is_complete):
            raise ModelNotLoadedError(self.load_error or "Model artifacts incomplete")
        return self


registry = ModelRegistry()
