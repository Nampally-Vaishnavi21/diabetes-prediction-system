# Diabetes Prediction System

An end-to-end, explainable machine-learning web application that estimates the probability of a diabetes outcome from eight clinical measurements.

**React** frontend → **FastAPI** backend → **scikit-learn** pipeline → **SHAP** explanations, bootstrap uncertainty and what-if analysis.

> **Medical disclaimer.** This system is an academic/research prototype for diabetes risk prediction. It is not a medical diagnostic device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Technology stack](#technology-stack)
4. [Project structure](#project-structure)
5. [Dataset](#dataset)
6. [Setup and running (step by step)](#setup-and-running)
7. [Model training](#model-training)
8. [API endpoints](#api-endpoints)
9. [Testing](#testing)
10. [Results](#results)
11. [Limitations](#limitations)
12. [Troubleshooting](#troubleshooting)
13. [Publish as a website](#publish-as-a-website)

---

## Features

| Area | What it does |
|---|---|
| **Prediction** | An 8-field form with frontend and backend validation. Insulin and skin thickness are optional and are imputed when blank. |
| **Probability** | The model's estimated probability, compared against a decision threshold tuned on training data. |
| **Uncertainty** | 50 bootstrap models give a 90% range for the estimate. A borderline flag appears when that range crosses the threshold. The results page also shows whether the other models agree. |
| **Out-of-distribution warning** | Flags inputs outside the range seen in the training data. |
| **Explainability** | SHAP local explanation (exact Shapley values over all 256 feature subsets, verified identical to the `shap` library) for each patient, plus global feature importance and a SHAP summary plot. |
| **What-if analysis** | Compares an original and a modified scenario with the real model. Also draws a sensitivity curve with a bootstrap band. |
| **Model information** | Comparison of 7 models (cross-validation and test results), ROC / precision–recall / calibration curves, confusion matrix, dataset quality, feature selection and methodology. |
| **Robustness** | Clear messages for backend unavailable, timeout, invalid input, model not trained and server errors. Users never see a stack trace. |
| **Responsive** | Works on phones, tablets, laptops and desktops. The navigation collapses into a menu on small screens. |

No value in the UI is hardcoded. Every number comes from the trained model or from files the training script writes.

---

## Architecture

```
 Browser (React, Vite)                FastAPI backend                         Offline training
 ───────────────────────              ─────────────────────────────           ──────────────────────────────
 Pages ─► services/api.js ──HTTP──►  routes/  (Pydantic validation)          python -m app.ml.train
   ▲        (axios, VITE_API_URL)       │                                       │
   │                                    ▼                                       ├─ load + validate CSV
   │                                 services/prediction.py                     ├─ stratified 80/20 split
   │                                    │  ├─ final_pipeline.predict_proba      ├─ GridSearchCV × 6 models
   │                                    │  ├─ threshold                         ├─ soft-voting ensemble
   │                                    │  ├─ bootstrap uncertainty (50)        ├─ selection, calibration check
   │                                    │  ├─ model agreement (7 models)        ├─ threshold (out-of-fold)
   │                                    │  ├─ out-of-distribution check         ├─ test-set evaluation (once)
   │                                    │  └─ exact SHAP (numpy)                ├─ bootstrap ensemble, SHAP
   └──────────── JSON ◄─────────────────┘                                       └─► backend/models/*.joblib
                                       services/model_registry.py loads ◄────────    outputs/metrics/*.json
                                       artifacts once at startup                     outputs/figures/*.png
```

**The key design rule.** Preprocessing (zero→missing, imputation, feature engineering, scaling) lives inside one scikit-learn `Pipeline` together with the model, and that object is saved with joblib. The API sends raw values to it. As a result, training-time and prediction-time preprocessing are guaranteed to be identical.

---

## Technology stack

| Layer | Tools |
|---|---|
| Frontend | React 18, Vite 5, React Router 6, Recharts 2, axios |
| Backend | Python 3.10–3.13, FastAPI, Pydantic v2, Uvicorn |
| Machine learning | scikit-learn, pandas, numpy, joblib, SHAP |
| Figures | matplotlib, seaborn |
| Testing | pytest + FastAPI TestClient; Vitest + React Testing Library; Playwright (end-to-end) |

No database is used: nothing needs to persist between requests.

---

## Project structure

```
diabetes-prediction-system/
├── backend/
│   ├── app/
│   │   ├── main.py                  FastAPI app: CORS, logging, error handlers, routers
│   │   ├── config.py                paths and settings (.env)
│   │   ├── routes/                  health, feature-info, model-info, predict/explain/what-if/example
│   │   ├── schemas/                 Pydantic request + response models
│   │   ├── services/                model_registry, prediction, model_info
│   │   ├── ml/
│   │   │   ├── features.py          ★ single source of truth for the 8 inputs and their limits
│   │   │   ├── data.py              load, validate, data-quality report
│   │   │   ├── preprocessing.py     ZeroToNaN → imputer → FeatureEngineer → scaler
│   │   │   ├── models.py            6 models + hyperparameter grids
│   │   │   ├── evaluate.py          metrics, curves, threshold, bootstrap CI, figures
│   │   │   ├── explain.py           SHAP local + global
│   │   │   ├── uncertainty.py       bootstrap ensemble
│   │   │   ├── eda.py               EDA figures
│   │   │   └── train.py             end-to-end training script
│   │   └── utils/                   logging, errors
│   ├── data/raw/diabetes.csv        dataset (bundled, CC0)
│   ├── models/                      trained artifacts (created by training)
│   ├── tests/                       76 pytest tests
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── pages/                   Dashboard, Predict, Results, Explainability, WhatIf, ModelInfo, About
│   │   ├── components/              Layout, PatientForm, RiskRuler, charts, Feedback, Disclaimer
│   │   ├── services/api.js          ★ every backend call + error mapping
│   │   ├── context/                 shared prediction state
│   │   ├── hooks/ utils/
│   │   └── test/                    Vitest tests
│   ├── package.json
│   └── .env.example                 VITE_API_URL
├── notebooks/                       01–05, executed, import the same app.ml code
├── outputs/{metrics,figures}/       written by training
├── e2e/run_e2e.py                   browser test of every page and button
├── docs/VIVA_GUIDE.md               component-by-component explanations for the viva
├── requirements-notebooks.txt
└── README.md
```

---

## Dataset

**Pima Indians Diabetes Database.** It comes from the National Institute of Diabetes and Digestive and Kidney Diseases and is published on Kaggle under a CC0 (public domain) licence.

- 768 patients, all women aged 21 or older of Pima heritage
- 8 features and a binary `Outcome`: 500 negative, 268 positive
- Zeros in Glucose (5), BloodPressure (35), SkinThickness (227), Insulin (374) and BMI (11) mean *not measured*. The pipeline treats them as missing values.

The file is already included at `backend/data/raw/diabetes.csv`. If you ever need to re-download it:

1. Go to https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database and download `diabetes.csv`.
2. Place it at `backend/data/raw/diabetes.csv`.

The training script checks the file's columns, data types and target values. If anything is wrong, it stops with a clear message.

---

## Setup and running

You need **Python 3.10+** and **Node.js 18+**. Open two terminals: one for the backend, one for the frontend.

### 1. Backend (terminal 1)

```bash
cd diabetes-prediction-system/backend

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# macOS / Linux:         source .venv/bin/activate

pip install -r requirements.txt

python -m app.ml.train          # trains everything, about 3–5 minutes
uvicorn app.main:app --reload   # API at http://localhost:8000  (docs at /docs)
```

Optional: copy `.env.example` to `.env` if you need to change the CORS origins or file paths.

### 2. Frontend (terminal 2)

```bash
cd diabetes-prediction-system/frontend
cp .env.example .env            # Windows: copy .env.example .env
npm install
npm run dev                     # app at http://localhost:5173
```

The backend address is configured in one place: `VITE_API_URL` in `frontend/.env`. The default is `http://localhost:8000`.

### 3. Use it

Open http://localhost:5173.
- The **Dashboard** should show *Connected* and the model name.
- Go to **Prediction**, then click **Load example** or type the values yourself, then click **Predict**.

---

## Model training

`python -m app.ml.train` (run from `backend/`) performs these steps:

1. Load and validate the CSV; check for duplicates; report missing values (zeros) and IQR outliers.
2. Generate EDA figures.
3. Split the data 80/20, stratified, with `random_state=42`. The test set is not touched again until step 10.
4. Feature-selection analysis on the training set: ANOVA F-test, mutual information and RFECV.
5. Tune six models with `GridSearchCV` and stratified 5-fold cross-validation, refitting on ROC-AUC: Logistic Regression, Decision Tree, Random Forest, SVM (RBF with Platt scaling), Gradient Boosting and KNN. Whether to use the engineered features is itself one of the tuned settings.
6. Build a soft-voting ensemble from the top three models.
7. Select the final model: best cross-validated ROC-AUC, with models within 0.005 counted as tied and the tie broken by the lowest Brier score.
8. Calibration check: an extra sigmoid calibration layer is kept only if it lowers the cross-validated Brier score.
9. Choose the decision threshold with Youden's J on out-of-fold training predictions.
10. Evaluate once on the held-out test set, with a 1000-resample bootstrap 95% confidence interval.
11. Train 50 bootstrap models for uncertainty estimates.
12. Compute global SHAP values on the test set.
13. Save everything:
    - `backend/models/`: model artifacts
    - `outputs/metrics/`: JSON and CSV results
    - `outputs/figures/`: PNG figures

**Notebooks.** The five notebooks in `notebooks/` cover data understanding, EDA, feature engineering, training and evaluation/XAI. They import the same code as the training script, so they always match it. To open them:

```bash
pip install -r requirements-notebooks.txt   # from the project root, inside the venv
jupyter notebook notebooks/
```

---

## API endpoints

Interactive documentation is available at http://localhost:8000/docs.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Whether the API is running, the model is loaded, and the model name |
| GET | `/feature-info` | Field definitions and valid ranges, used to build the form; also the training-data ranges |
| GET | `/model-info` | All metrics, curves, calibration, dataset and selection details |
| POST | `/predict` | Prediction, probability, threshold, uncertainty, model agreement, out-of-distribution warnings and SHAP values |
| POST | `/explain` | Local SHAP explanation for one patient, plus global importance |
| GET | `/explain/global` | Global SHAP importance and summary-plot data |
| POST | `/what-if` | Compare an `original` and a `modified` patient |
| POST | `/what-if/sensitivity` | Vary one feature across its training range |
| GET | `/example` | A random real patient from the held-out test set |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "pregnancies": 2, "glucose": 120, "blood_pressure": 70, "skin_thickness": 20,
  "insulin": 79, "bmi": 28.5, "diabetes_pedigree": 0.35, "age": 25 }'
```

**Response** (abridged; all values are computed by the model):

```json
{
  "prediction": 0,
  "prediction_label": "Lower estimated risk",
  "probability": 0.1357,
  "threshold": 0.3082,
  "model": "Support Vector Machine (RBF)",
  "interpretation": "The model estimates a probability of approximately 14% based on the input features. ...",
  "uncertainty": { "lower": 0.0951, "upper": 0.1697, "borderline": false, "...": "..." },
  "model_agreement": { "n_above_threshold": 0, "n_models": 7, "votes": ["..."] },
  "ood_warnings": [],
  "explanation": { "base_value": 0.3298, "contributions": [ { "feature": "bmi", "shap_value": -0.0744, "...": "..." } ] },
  "disclaimer": "This system is an academic/research prototype ..."
}
```

**Errors** always use the same JSON shape: `{"error": "...", "message": "...", "details": [...]}`.

| Status | Meaning |
|---|---|
| 422 | Invalid or missing input, with a per-field message |
| 400 | A what-if request with nothing changed |
| 503 | Model not trained yet |
| 500 | Unexpected error. The details go to the server log only. |

---

## Testing

```bash
# Backend: 76 tests (validation, errors, CORS, data, leakage, metrics, every endpoint)
cd backend && pytest

# Frontend: 17 tests (validation rules, error mapping, prediction form behaviour)
cd frontend && npm test

# End-to-end: 18 browser checks against the running app (start the backend and frontend first)
pip install playwright && python -m playwright install chromium
python e2e/run_e2e.py
```

The end-to-end script covers the following:
- every navigation link and every button
- the empty-form and out-of-range validation messages
- **Load example** and **Reset**
- backend unavailable → error message → **Retry**
- the loading state, and that the displayed probability equals the backend's
- explanation, what-if, sensitivity curve, model-information tabs, the About page and the 404 page
- no JavaScript errors in the console
- no horizontal overflow at tablet (820 px) and mobile (390 px) widths
- the mobile menu

Several tests prove that results are not hardcoded. For example, `test_reported_metrics_match_saved_model` recomputes the test metrics from the saved model and compares them with what `/model-info` reports.

---

## Results

These come from the training run included with this project. Your numbers are regenerated whenever you train and should be identical, because `random_state` is fixed.

**Final model:** SVM (RBF kernel) with Platt-scaled probabilities and a decision threshold of 0.308.

**Held-out test set (154 patients):**

| Metric | Value |
|---|---|
| ROC-AUC | 0.815 (95% CI 0.745–0.875) |
| PR-AUC | 0.682 |
| Brier score | 0.175 |
| Recall | 0.778 |
| Specificity | 0.720 |
| Precision | 0.600 |
| F1 | 0.677 |
| Accuracy | 0.740 |

Uncertainty: the average 90% bootstrap range is 16 percentage points wide, and 27% of test patients are borderline. The full comparison table and curves are on the **Model information** page and in `outputs/`.

---

## Limitations

- The dataset is small (768 patients) and comes from one population: women aged 21+ of Pima heritage. The model does not generalise to men, children or other populations.
- Insulin is missing for about half of the patients and is imputed.
- With 154 test patients, the test metrics have wide confidence intervals.
- SHAP and what-if results describe how the model behaves, not medical causes.
- The model has not been clinically validated. It is not a diagnostic device.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard says **Backend unavailable** | Start the backend (`uvicorn app.main:app --reload` in `backend/`) and check that `VITE_API_URL` in `frontend/.env` matches its address. Restart `npm run dev` after editing `.env`. |
| **Model not available** (503) | Run `python -m app.ml.train` in `backend/`, then restart uvicorn. |
| Model files fail to load after upgrading scikit-learn | Saved models are tied to the library version that created them. Re-run training. |
| CORS error in the browser console | Add your frontend's address to `CORS_ORIGINS` in `backend/.env`. |
| Port 8000 or 5173 is already in use | Use `uvicorn app.main:app --port 8001` and set `VITE_API_URL=http://localhost:8001`, or use `npm run dev -- --port 5174` and add that origin to `CORS_ORIGINS`. |
| PowerShell will not activate the venv | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |

---

## Publish as a website

The `Dockerfile` packages everything into **one container**:
- it builds the React app;
- it installs the Python libraries;
- it trains the model and runs the tests (if anything fails, the build stops);
- it starts one server that serves both the website and the API (`/api/...`).

Step-by-step instructions for a free public URL on **Render** are in **[DEPLOY.md](DEPLOY.md)**.

To run the same single-server setup on your own computer, without Docker:

```bash
cd frontend
VITE_API_URL=/api npm run build      # Windows PowerShell: $env:VITE_API_URL="/api"; npm run build
cd ../backend
uvicorn app.main:app --port 8000     # website and API both at http://localhost:8000
```

