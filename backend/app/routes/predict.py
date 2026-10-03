"""Prediction, explanation, what-if and example endpoints."""
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.schemas.patient import PatientInput
from app.schemas.requests import SensitivityRequest, WhatIfRequest
from app.schemas.responses import (ExampleResponse, ExplainResponse, GlobalExplanationResponse,
                                   PredictionResponse, SensitivityResponse, WhatIfResponse)
from app.services import prediction as svc

router = APIRouter(tags=["Prediction"])

# Model calls are CPU work; run_in_threadpool keeps the server responsive meanwhile.


@router.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    """Prediction + probability + uncertainty + model agreement + OOD warnings + SHAP."""
    return await run_in_threadpool(svc.predict, patient)


@router.post("/explain", response_model=ExplainResponse)
async def explain(patient: PatientInput):
    """Local SHAP explanation for one patient, plus global importance for context."""
    return await run_in_threadpool(svc.explain_patient, patient)


@router.get("/explain/global", response_model=GlobalExplanationResponse)
def explain_global():
    """Global SHAP importance and beeswarm data, precomputed on the test set during training."""
    return svc.global_explanation()


@router.post("/what-if", response_model=WhatIfResponse)
async def what_if(body: WhatIfRequest):
    """Compare the model output for an original and a modified set of inputs."""
    return await run_in_threadpool(svc.what_if, body.original, body.modified)


@router.post("/what-if/sensitivity", response_model=SensitivityResponse)
async def what_if_sensitivity(body: SensitivityRequest):
    """Vary one feature across its training range, holding the others fixed."""
    return await run_in_threadpool(svc.sensitivity, body.patient, body.feature, body.n_points)


@router.get("/example", response_model=ExampleResponse)
def example():
    """A random real patient from the held-out test set, for the 'Load Example' button."""
    return svc.example()
