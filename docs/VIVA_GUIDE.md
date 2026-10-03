# Viva guide — Diabetes Prediction System

How to explain each part of the project. For every component: **What** it is, **Why** we use it, **How** it works, and a short **Viva answer** you can say out loud.

---

## The 60-second summary

> "This is a full-stack diabetes risk prediction system. A React frontend collects eight clinical values and validates them. A FastAPI backend validates them again with Pydantic. It then passes them to a scikit-learn pipeline that was trained on the Pima Indians Diabetes dataset.
>
> I compared six model families plus a voting ensemble, using stratified 5-fold cross-validation with grid search. I selected an SVM using ROC-AUC, with Brier score as the tie-breaker. The decision threshold was tuned on out-of-fold predictions, and the held-out test set was used exactly once, at the end.
>
> Every prediction comes with three things:
> - a SHAP explanation that shows which features pushed the estimate up or down;
> - an uncertainty range from 50 bootstrap models;
> - a what-if tool for model sensitivity analysis.
>
> On 154 unseen patients the model reaches ROC-AUC 0.815, with a 95% confidence interval of 0.745 to 0.875, and 78% recall. It is an academic prototype, not a diagnostic device."

---

## 1. Architecture

**What.** Three parts:
- the React user interface;
- the FastAPI REST backend;
- an offline training script that writes model files, which the backend loads at startup.

**Why.** Each part has one job and can be tested on its own. The frontend never contains any ML logic.

**How.**
1. React calls `services/api.js`, which uses axios.
2. The request reaches a FastAPI route.
3. Pydantic validates the request body.
4. `services/prediction.py` runs the saved pipeline and returns JSON.

**Viva answer.** "The frontend is just a client of a REST API. All validation and ML happens on the server, so the same API could serve a mobile app."

## 2. Single source of truth for the inputs (`features.py`)

**What.** One list that defines all 8 inputs: name, unit, valid range, whether it is optional, and the error message.

**Why.** If the frontend and backend each had their own copy of the limits, the two copies would eventually disagree.

**How.**
- Pydantic builds its `ge`/`le` (greater-or-equal / less-or-equal) constraints from this list.
- React fetches the same list from `GET /feature-info` and builds the form from it.

**Viva answer.** "Validation exists in two places, but the rules exist in only one place."

## 3. Data quality and missing values

**What.** In the CSV, zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI mean the value was not measured. Insulin, for example, is missing in 374 of 768 rows.

**Why.** Treating these zeros as real values would teach the model that someone can have a BMI of 0.

**How.** Inside the pipeline:
- `ZeroToNaN` turns those zeros into missing values;
- `SimpleImputer(median, add_indicator=True)` fills them with the training median and adds 0/1 "was missing" columns.

**Viva answer.** "Missing values are imputed inside the pipeline, so the medians are learned from training data only."

## 4. Leakage prevention

**What.** No information from the validation or test data may influence training.

**How.**
- The test set is split off first and is used once, at the end.
- All preprocessing is inside the `Pipeline`, so cross-validation refits it on each training fold.
- The threshold is tuned on out-of-fold training predictions.
- The test `pytest test_imputer_learns_from_training_rows_only` proves the imputer uses only the rows it was fitted on.

**Viva answer.** "Every decision, including the model, its hyperparameters, calibration and the threshold, was made without the test set."

## 5. Preprocessing pipeline and joblib

**What.** The pipeline runs these steps in order: ZeroToNaN, then Imputer, then FeatureEngineer, then StandardScaler, then the model. The whole pipeline is saved as one `.joblib` file.

**Why.** The API sends raw values to the saved pipeline, so prediction-time preprocessing is exactly the same as training-time preprocessing.

**Viva answer.** "Preprocessing is part of the saved model object, so training and serving cannot diverge."

## 6. Feature engineering and feature selection

**Feature engineering**
- **What:** adds `Glucose×BMI` and log transforms of the skewed Insulin and DiabetesPedigreeFunction.
- **How:** whether to use these features is a GridSearchCV parameter (`prep__engineer__enabled`), so cross-validation decides for each model.

**Feature selection**
- **What:** three methods on the training set: ANOVA F-test, mutual information, and RFECV (recursive feature elimination with cross-validation).
- **Result:** RFECV kept all 13 columns, so we keep all 8 inputs.

**Viva answer.** "Feature engineering and selection decisions were made by cross-validation, not by guessing."

## 7. Models, cross-validation and tuning

**What.** Six models were tuned:
- Logistic Regression
- Decision Tree
- Random Forest
- SVM with an RBF kernel
- Gradient Boosting
- KNN

A soft-voting ensemble of the top 3 was added as a seventh candidate.

**How.**
- `GridSearchCV` tries every hyperparameter combination using `StratifiedKFold(5)`.
- Stratified means each fold keeps the 35% positive rate of the full dataset.
- The best configuration is refit on all 614 training rows.

**Selection rule.**
- Pick the highest cross-validated ROC-AUC.
- Models within 0.005 of the best are treated as tied, because that gap is far smaller than the fold-to-fold standard deviation of about 0.03.
- Among tied models, pick the lowest Brier score. That picked the SVM (CV ROC-AUC 0.844, Brier 0.152).

**Viva answer.** "The ensemble had a slightly higher cross-validated AUC, but its members were tuned on the same folds, so that number is slightly optimistic. Within noise, I preferred the model with the best probability quality."

## 8. Evaluation metrics

| Metric | Meaning | Our test value |
|---|---|---|
| ROC-AUC | How well the model ranks diabetic patients above non-diabetic ones, across all thresholds | 0.815 |
| PR-AUC | Area under the precision–recall curve; more informative than ROC-AUC when the classes are imbalanced | 0.682 |
| Recall (sensitivity) | Share of truly diabetic patients that are flagged | 0.778 |
| Specificity | Share of non-diabetic patients correctly not flagged | 0.720 |
| Precision | Share of flagged patients who are truly diabetic | 0.600 |
| F1 | Harmonic mean of precision and recall | 0.677 |
| Brier score | Mean squared error of the predicted probabilities (lower is better) | 0.175 |

**Viva answer for "why is the test AUC lower than the CV AUC (0.84)?"**
> "154 test patients is a small sample. The bootstrap 95% confidence interval (0.745–0.875) includes the CV value. Gradient Boosting scored 0.821 on the test set, but switching to it would mean selecting the model on test data, which biases the final result."

## 9. Decision threshold

**What.** The probability above which a patient is labelled "higher estimated risk". Ours is 0.308.

**Why not 0.5?**
- With only 35% of patients positive, a 0.5 threshold gives the SVM a recall of just 0.52.
- In screening, missing diabetic patients is the costlier error.

**How.** We maximise Youden's J (sensitivity + specificity − 1) on out-of-fold training predictions.

## 10. Calibration

**What.** Whether the predicted probabilities match how often the outcome actually happens. For example, of the patients given 30%, about 30% should really be diabetic.

**How.**
- The SVM already produces probabilities through Platt scaling: a sigmoid fitted with internal cross-validation.
- We tested adding a second sigmoid layer with `CalibratedClassifierCV`. The cross-validated Brier score got worse (0.1525 with it vs 0.1520 without), so we did not use it.
- The reliability diagram on the Model information page shows the effect.

**Viva answer.** "Calibration was tested and then rejected based on evidence, not applied automatically."

**How to phrase a probability:** "The model estimates a probability of approximately 47% based on the input features." Never say: "the patient has a 47% chance of having diabetes."

## 11. Uncertainty (bootstrap ensemble)

**What.** 50 copies of the final model, each trained on a resample of the training data drawn with replacement. For each patient we report the 5th–95th percentile of their 50 estimates.

**Why.** It shows how stable the estimate is. If the range crosses the threshold, the patient is flagged as *borderline*.

**What it is NOT.** It is not a clinical confidence interval and not diagnostic certainty. It only measures how sensitive the model is to its training data, which is called epistemic uncertainty.

**Numbers.** On the test set the average range width was 16 percentage points, and 27% of patients were borderline.

**Model agreement.** The results page also shows all 7 trained models' probabilities. When they disagree, that is another signal of uncertainty.

## 12. Explainable AI (SHAP)

**What.** SHAP gives each feature a contribution, so that:

> base value + sum of contributions = predicted probability

**How.**
- We use `ExactExplainer`. With 8 features there are only 2⁸ = 256 feature subsets, so we can compute exact Shapley values instead of approximations.
- A "missing" feature is simulated with 100 background training rows.
- We explain the whole pipeline on the raw inputs, so contributions are reported for Glucose, BMI and so on, in probability units.

**Global view.** The mean |SHAP| over 154 test patients ranks the features: Glucose (0.17), then BMI (0.085), then Pregnancies.

**Viva answer.** "SHAP shows how the model used each feature for this patient. It shows association inside the model, not causation."

## 13. What-if analysis

**What.**
- `/what-if` compares two complete patient records using the real model.
- `/what-if/sensitivity` varies one feature across its training range while holding the others fixed. This is called an Individual Conditional Expectation (ICE) curve. It is drawn with a bootstrap band.

**Viva answer.** "It is sensitivity analysis of the model. It does not show that lowering a real patient's glucose would change their outcome."

## 14. Out-of-distribution warning

Inputs that are valid but outside the range seen in training get a warning, for example glucose 250 when the training data's maximum is 199. Predictions there are extrapolations.

## 15. Error handling

| Situation | Backend response | What the frontend shows |
|---|---|---|
| Invalid or missing input | 422, with a message per field | The message under each field (it is also caught by frontend validation first) |
| Model not trained | 503 | "Model not available — train it first" |
| Unexpected exception | 500, generic message; full traceback in the server log | "Server error", with a Retry button |
| Backend down | (no response) | "Backend unavailable", with a Retry button |
| Timeout (30 s) | (no response) | "Request timed out" |
| Dataset missing during training | (training script) | Exits with download instructions |

## 16. Testing

| Suite | Count | What it covers |
|---|---|---|
| Backend `pytest` | 74 | Validation, error handlers, CORS, data loading, leakage, metric functions, every endpoint |
| Frontend `vitest` | 17 | Validation rules, API error mapping, prediction form behaviour |
| End-to-end (`e2e/run_e2e.py`) | 18 browser checks | Every page and button against the real backend, at desktop, tablet and mobile widths |

**Proof that nothing is hardcoded.** These tests compare the API's output with values recomputed directly from the saved model:
- `test_reported_metrics_match_saved_model`
- `test_predict_returns_real_model_probability`
- `test_what_if_uses_real_model`

---

## Likely viva questions

**Q. Why the Pima dataset?**
It is public (CC0) and well documented, and its 8 features match the form. Its limitations, a single population and many missing values, are stated in the app.

**Q. Why an SVM rather than a neural network?**
With 614 training rows, classical models are appropriate. A neural network would overfit and is harder to explain.

**Q. Why ROC-AUC as the main metric?**
It does not depend on any particular threshold, and it is not misleading under class imbalance the way accuracy is. We also report PR-AUC and the Brier score.

**Q. What is stratification?**
Splitting the data so that every part keeps the same class ratio as the whole (about 35% positive).

**Q. Why a 0.005 tie tolerance?**
The fold-to-fold standard deviation of the AUC is about 0.03, so differences smaller than 0.005 are noise. When models are that close, probability quality (Brier score) is a better tie-breaker.

**Q. How would you improve the project?**
- A larger and more diverse dataset.
- External validation on a different population.
- Nested cross-validation for an unbiased comparison of models.
- Clinical review of the input limits.
- Prospective evaluation in practice.
