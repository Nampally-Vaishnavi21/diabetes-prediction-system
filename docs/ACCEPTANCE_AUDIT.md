# Final audit against the 24 acceptance criteria

**How this was checked.** The audit was run on a **fresh copy** of the project: no trained models, no generated metrics, no `node_modules`. Every step followed the README exactly:

1. `python -m venv`
2. `pip install -r requirements.txt`
3. `python -m app.ml.train`
4. `pytest`
5. `npm install`
6. `npm test`
7. `npm run build`
8. Started both servers.
9. Ran `python e2e/run_e2e.py`.

**Environment used:** Python 3.13, scikit-learn 1.9.1, SHAP 0.52.0, FastAPI 0.142, Node 22.

**Results:**
- 74/74 backend tests passed.
- 17/17 frontend tests passed.
- 18/18 end-to-end browser checks passed.
- The fresh training run produced exactly the same test metrics as the original run (fixed `random_state`).

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Frontend starts | ✅ | `npm run dev` serves on port 5173; `npm run build` succeeds |
| 2 | Backend starts | ✅ | `uvicorn app.main:app` responds; `/health` returns 200 |
| 3 | Frontend connects to backend | ✅ | e2e "Dashboard shows live backend status"; `VITE_API_URL` is set in one place (`services/api.js`) |
| 4 | Dataset loads | ✅ | `test_dataset_loads_with_expected_shape` (768×9); schema validation in `data.py` |
| 5 | Models train | ✅ | `python -m app.ml.train` trains 6 tuned models plus an ensemble in about 4 minutes |
| 6 | Artifacts saved | ✅ | `backend/models/`: final pipeline, bootstrap ensemble, all models, SHAP background, metadata, test set |
| 7 | Backend loads model | ✅ | `test_registry_loads_artifacts`; `/health` reports `model_loaded: true` |
| 8 | Prediction works with real input | ✅ | `test_predict_returns_real_model_probability`; e2e checks the UI value equals the API value |
| 9 | Probability works | ✅ | Probability is compared to the tuned threshold; wording is "The model estimates a probability of approximately…" |
| 10 | Explainability works | ✅ | `/explain` and `/explain/global` tests; e2e Explainability page |
| 11 | SHAP local explanation | ✅ | `test_shap_values_are_additive` (base + contributions = probability) |
| 12 | What-if works | ✅ | `test_what_if_uses_real_model`; e2e what-if with sensitivity curve |
| 13 | Calibration metrics | ✅ | Brier score, reliability curves, and the CV calibration decision in `evaluation.json` and on the Model information page |
| 14 | Model comparison | ✅ | 7 models: CV table, test table and ROC/PR curves on the Model information page |
| 15 | Charts display real data | ✅ | Every chart is drawn from API arrays; `test_reported_metrics_match_saved_model` |
| 16 | Input validation | ✅ | 16 parametrised backend cases plus frontend tests; e2e checks empty and out-of-range input |
| 17 | Error handling | ✅ | Backend unavailable (e2e), model missing (manual check with an empty models folder), invalid input (422), what-if with no change (400), crash (500 with no traceback), dataset missing (exit 1 with download instructions), timeout (unit test) |
| 18 | Reset works | ✅ | e2e "Reset form clears every field" and the what-if "Reset changes" button |
| 19 | Navigation works | ✅ | e2e checks every sidebar link, the mobile menu and the 404 link |
| 20 | No non-functional buttons | ✅ | Every button is covered in e2e: Predict, Load example, Reset form, Retry prediction, Run prediction again, View full explanation, Try what-if, Back to inputs, Back to results, Run comparison, Reset changes, curve tabs, feature and scenario selectors, Menu, Retry/Check again |
| 21 | No fake data | ✅ | No mock, sample or placeholder data in the app. The only test doubles are in `src/test/Predict.test.jsx`, used for isolated form tests |
| 22 | No hardcoded prediction | ✅ | Recompute-and-match tests; different inputs give different outputs (`test_different_inputs_give_different_outputs`) |
| 23 | Runs from a fresh environment | ✅ | The fresh-copy run described above |
| 24 | Understandable for a viva | ✅ | `docs/VIVA_GUIDE.md`; What/Why/How docstrings in every ML module; 5 executed notebooks |

**Known, accepted constraints:**
- Web fonts load from Google Fonts. Offline, the app falls back to system fonts and still works.
- The saved model files only load with the same scikit-learn version that created them. The README says to retrain after upgrading libraries.
