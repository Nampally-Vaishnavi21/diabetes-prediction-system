"""
END-TO-END TRAINING SCRIPT.

Run from the backend/ folder:
    python -m app.ml.train

Steps:
  1. Load + validate the CSV, remove duplicates
  2. Data-quality report + EDA figures
  3. Stratified 80/20 train/test split (test set is locked away)
  4. Feature-selection analysis (training set only)
  5. Tune every model with GridSearchCV + stratified 5-fold CV (training set only)
  6. Build a soft-voting ensemble of the top 3 tuned models
  7. Select the final model (best CV ROC-AUC; near-ties broken by Brier score)
  8. Calibration: keep a sigmoid calibration layer only if it lowers CV Brier
  9. Decision threshold from out-of-fold training predictions (Youden's J)
 10. Evaluate everything ONCE on the held-out test set
 11. Bootstrap ensemble (50 resampled copies) for uncertainty
 12. SHAP background sample + global explanation on the test set
 13. Save artifacts for the API and metrics for the frontend

Leakage prevention, in one sentence for the viva:
  "The test set is split off first and never used for any decision; all
   preprocessing is inside the Pipeline, so medians/scaling are learned from
   training folds only during cross-validation."
"""
import json
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import RFECV, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, cross_validate, train_test_split)

from app.config import settings
from app.ml.data import DatasetError, data_quality_report, load_dataset, remove_duplicates
from app.ml.eda import generate_eda_figures
from app.ml.explain import BACKGROUND_SIZE, build_explainer, global_explanation, shap_values_for
from app.ml.uncertainty import N_BOOTSTRAP, bootstrap_uncertainty, train_bootstrap_ensemble
from app.ml.evaluate import (bootstrap_ci, calibration_points, compute_metrics, plot_calibration,
                             plot_confusion, plot_pr, plot_roc, pr_points, roc_points,
                             youden_threshold)
from app.ml.features import (CSV_COLUMNS, CSV_TO_KEY, FEATURES, TARGET_COLUMN,
                             ZERO_AS_MISSING_COLUMNS)
from app.ml.models import (CV_SCORING, ENSEMBLE_DISPLAY, ENSEMBLE_KEY, ENSEMBLE_TOP_N,
                           MODEL_ZOO, N_JOBS, PRIMARY_METRIC, RANDOM_STATE, TIE_TOLERANCE)
from app.ml.preprocessing import build_pipeline, build_preprocessor

TEST_SIZE = 0.20
N_FOLDS = 5


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def to_jsonable(obj):
    """Convert numpy types so json.dump works."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_json(data, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Step 4: feature selection analysis
# --------------------------------------------------------------------------- #
def feature_selection_analysis(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Three complementary views, all on the TRAINING set only:
      - ANOVA F-test   : linear separation of the two classes per feature
      - Mutual info    : any (also non-linear) dependency with the outcome
      - RFECV (LogReg) : recursively drop the weakest feature, measure CV ROC-AUC
    Preprocessing is fitted on the training set before scoring features.
    """
    pre = build_preprocessor()
    Z = pre.fit_transform(X_train, y_train)
    cols = list(Z.columns)

    f_scores, p_values = f_classif(Z, y_train)
    mi = mutual_info_classif(Z, y_train, random_state=RANDOM_STATE)

    rfecv = RFECV(LogisticRegression(max_iter=2000), step=1, cv=make_cv(), scoring="roc_auc")
    rfecv.fit(Z, y_train)
    curve = rfecv.cv_results_["mean_test_score"]
    best_n = int(rfecv.n_features_)
    best_score = float(curve[best_n - 1])
    n_all = len(cols)
    score_all = float(curve[-1])

    keep_all = best_n == n_all or (score_all >= best_score - TIE_TOLERANCE)
    decision = (
        f"Keep all {len(CSV_COLUMNS)} input features. RFECV's best subset has {best_n} of "
        f"{n_all} columns (CV ROC-AUC {best_score:.4f}) vs {score_all:.4f} with all columns, "
        f"a difference within the {TIE_TOLERANCE} noise tolerance. With only 8 inputs, "
        "dropping features gains nothing measurable and every form field stays used."
        if keep_all else
        f"RFECV suggests {best_n} columns (CV ROC-AUC {best_score:.4f} vs {score_all:.4f} with all)."
    )

    return {
        "method": "ANOVA F-test, mutual information, RFECV with Logistic Regression "
                  f"({N_FOLDS}-fold stratified CV, ROC-AUC); training set only",
        "columns_after_preprocessing": cols,
        "anova_f": {c: round(float(f), 3) for c, f in zip(cols, f_scores)},
        "anova_p_value": {c: float(p) for c, p in zip(cols, p_values)},
        "mutual_information": {c: round(float(m), 4) for c, m in zip(cols, mi)},
        "rfecv": {
            "n_features_selected": best_n,
            "selected": [c for c, s in zip(cols, rfecv.support_) if s],
            "ranking": {c: int(r) for c, r in zip(cols, rfecv.ranking_)},
            "cv_roc_auc_by_n_features": [round(float(s), 4) for s in curve],
        },
        "decision_keep_all": bool(keep_all),
        "decision": decision,
    }


# --------------------------------------------------------------------------- #
# Step 5-6: tuning and cross-validation
# --------------------------------------------------------------------------- #
def _summarise_cv(cv_metrics: dict) -> dict:
    """cv_metrics: {metric: array of per-fold scores}. Brier is un-negated."""
    out = {}
    for name, scores in cv_metrics.items():
        scores = np.asarray(scores, dtype=float)
        if name == "brier":
            scores = -scores
        out[name] = {"mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}
    return out


def tune_model(key: str, cfg: dict, X: pd.DataFrame, y: pd.Series) -> dict:
    t0 = time.perf_counter()
    search = GridSearchCV(
        estimator=build_pipeline(cfg["estimator"]),
        param_grid=cfg["param_grid"],
        scoring=CV_SCORING,
        refit=PRIMARY_METRIC,       # final model = best mean ROC-AUC
        cv=make_cv(),
        n_jobs=N_JOBS,
        error_score="raise",
    )
    search.fit(X, y)
    i = search.best_index_
    fold_scores = {
        m: [search.cv_results_[f"split{k}_test_{m}"][i] for k in range(N_FOLDS)]
        for m in CV_SCORING
    }
    secs = time.perf_counter() - t0
    n_candidates = len(search.cv_results_["params"])
    log(f"  {cfg['display_name']:<30} CV ROC-AUC {search.best_score_:.4f}  "
        f"({n_candidates} configs x {N_FOLDS} folds, {secs:.1f}s)")
    return {
        "key": key,
        "display_name": cfg["display_name"],
        "why": cfg["why"],
        "best_params": {k: v for k, v in search.best_params_.items()},
        "engineered_features": bool(search.best_params_.get("prep__engineer__enabled", False)),
        "n_configs_tried": n_candidates,
        "cv": _summarise_cv(fold_scores),
        "tuning_seconds": round(secs, 1),
        "estimator": search.best_estimator_,   # refit on the full training set
    }


def build_ensemble(results: list[dict], X: pd.DataFrame, y: pd.Series) -> dict:
    top = sorted(results, key=lambda r: r["cv"][PRIMARY_METRIC]["mean"], reverse=True)[:ENSEMBLE_TOP_N]
    members = [(r["key"], r["estimator"]) for r in top]
    ensemble = VotingClassifier(estimators=members, voting="soft", n_jobs=None)
    cv = cross_validate(ensemble, X, y, scoring=CV_SCORING, cv=make_cv(), n_jobs=N_JOBS)
    fold_scores = {m: cv[f"test_{m}"] for m in CV_SCORING}
    ensemble.fit(X, y)
    summary = _summarise_cv(fold_scores)
    log(f"  {ENSEMBLE_DISPLAY:<30} CV ROC-AUC {summary[PRIMARY_METRIC]['mean']:.4f}  "
        f"(members: {', '.join(k for k, _ in members)})")
    return {
        "key": ENSEMBLE_KEY,
        "display_name": ENSEMBLE_DISPLAY,
        "why": (f"Averages the predicted probabilities of the top {ENSEMBLE_TOP_N} tuned models. "
                "Its CV score is slightly optimistic because the members were tuned on the same folds."),
        "best_params": {"members": [k for k, _ in members], "voting": "soft"},
        "engineered_features": None,
        "n_configs_tried": 1,
        "cv": summary,
        "tuning_seconds": None,
        "estimator": ensemble,
    }


def select_model(results: list[dict]) -> tuple[dict, str]:
    best_auc = max(r["cv"][PRIMARY_METRIC]["mean"] for r in results)
    tied = [r for r in results if r["cv"][PRIMARY_METRIC]["mean"] >= best_auc - TIE_TOLERANCE]
    chosen = min(tied, key=lambda r: r["cv"]["brier"]["mean"])
    reason = (
        f"Highest mean CV ROC-AUC was {best_auc:.4f}. Models within {TIE_TOLERANCE} of it "
        f"({', '.join(r['display_name'] for r in tied)}) were treated as tied and the one with "
        f"the lowest CV Brier score ({chosen['cv']['brier']['mean']:.4f}) was selected: "
        f"{chosen['display_name']}."
    )
    return chosen, reason


# --------------------------------------------------------------------------- #
# Step 8: calibration (decided by CV on the training set)
# --------------------------------------------------------------------------- #
def calibration_step(selected_estimator, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compare the selected model with and without an extra sigmoid calibration layer.

    Both versions are scored on the SAME outer CV folds (paired comparison) by
    Brier score. Calibration is kept only if it lowers the mean CV Brier score.
    Sigmoid (Platt) is used rather than isotonic because isotonic regression
    overfits with only ~490 rows per training fold.
    """
    folds = make_cv()
    uncal = clone(selected_estimator)
    cal = CalibratedClassifierCV(clone(selected_estimator), method="sigmoid", cv=make_cv())
    b_uncal = -cross_val_score(uncal, X, y, cv=folds, scoring="neg_brier_score", n_jobs=N_JOBS)
    b_cal = -cross_val_score(cal, X, y, cv=folds, scoring="neg_brier_score", n_jobs=N_JOBS)
    use = bool(b_cal.mean() < b_uncal.mean())
    decision = (
        f"Calibration kept: mean CV Brier {b_cal.mean():.4f} vs {b_uncal.mean():.4f} without it."
        if use else
        f"Calibration not applied: mean CV Brier {b_cal.mean():.4f} with it vs "
        f"{b_uncal.mean():.4f} without it, so the extra layer did not help."
    )
    log(f"  {decision}")
    uncal.fit(X, y)
    cal.fit(X, y)
    return {
        "method": "CalibratedClassifierCV(method='sigmoid', cv=5) wrapped around the selected pipeline",
        "cv_brier_uncalibrated": {"mean": round(float(b_uncal.mean()), 4), "std": round(float(b_uncal.std()), 4)},
        "cv_brier_calibrated": {"mean": round(float(b_cal.mean()), 4), "std": round(float(b_cal.std()), 4)},
        "used": use,
        "decision": decision,
        "uncalibrated_estimator": uncal,
        "calibrated_estimator": cal,
    }


# --------------------------------------------------------------------------- #
# Step 9: decision threshold (out-of-fold training predictions)
# --------------------------------------------------------------------------- #
def threshold_step(final_estimator, X: pd.DataFrame, y: pd.Series) -> dict:
    oof = cross_val_predict(clone(final_estimator), X, y, cv=make_cv(),
                            method="predict_proba", n_jobs=N_JOBS)[:, 1]
    best = youden_threshold(y, oof)
    log(f"  Threshold {best['threshold']:.3f} (Youden J {best['youden_j']:.3f} on out-of-fold "
        f"training predictions; sens {best['sensitivity']:.3f}, spec {best['specificity']:.3f})")
    return {
        "value": best["threshold"],
        "method": "Youden's J (sensitivity + specificity - 1) maximised on out-of-fold "
                  f"predictions from {N_FOLDS}-fold CV on the training set; the test set was not used",
        "oof_youden": best,
        "oof_metrics_at_threshold": compute_metrics(y, oof, best["threshold"]),
        "oof_metrics_at_0_5": compute_metrics(y, oof, 0.5),
    }


# --------------------------------------------------------------------------- #
# Step 10: final evaluation on the held-out test set (used exactly once)
# --------------------------------------------------------------------------- #
def evaluate_on_test(results, final_estimator, final_name, calib, threshold, X_test, y_test,
                     figures_dir) -> dict:
    per_model = []
    roc_c, pr_c, aucs, aps = {}, {}, {}, {}
    for r in results:
        prob = r["estimator"].predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, prob, 0.5)
        per_model.append({
            "key": r["key"], "display_name": r["display_name"],
            "test_metrics_at_0_5": m,
            "roc_curve": roc_points(y_test, prob),
            "pr_curve": pr_points(y_test, prob),
            "calibration_curve": calibration_points(y_test, prob),
        })
        roc_c[r["display_name"]], pr_c[r["display_name"]] = per_model[-1]["roc_curve"], per_model[-1]["pr_curve"]
        aucs[r["display_name"]], aps[r["display_name"]] = m["roc_auc"], m["pr_auc"]

    prob = final_estimator.predict_proba(X_test)[:, 1]
    final = {
        "display_name": final_name,
        "metrics_at_threshold": compute_metrics(y_test, prob, threshold),
        "metrics_at_0_5": compute_metrics(y_test, prob, 0.5),
        "bootstrap": bootstrap_ci(y_test, prob),
        "roc_curve": roc_points(y_test, prob),
        "pr_curve": pr_points(y_test, prob),
        "calibration_curve": calibration_points(y_test, prob),
    }

    # Calibration before/after on the test set (reported only; the decision was made by CV)
    p_unc = calib["uncalibrated_estimator"].predict_proba(X_test)[:, 1]
    p_cal = calib["calibrated_estimator"].predict_proba(X_test)[:, 1]
    calib_test = {
        "uncalibrated": {"brier": round(float(brier_score_loss(y_test, p_unc)), 4),
                         "points": calibration_points(y_test, p_unc)},
        "calibrated": {"brier": round(float(brier_score_loss(y_test, p_cal)), 4),
                       "points": calibration_points(y_test, p_cal)},
    }

    figs = [
        plot_roc(roc_c, aucs, figures_dir / "roc_curves.png"),
        plot_pr(pr_c, aps, float(np.mean(y_test)), figures_dir / "pr_curves.png"),
        plot_confusion(final["metrics_at_threshold"]["confusion_matrix"],
                       f"{final_name}\nthreshold {threshold:.3f}", figures_dir / "confusion_matrix.png"),
        plot_calibration({
            "Before extra calibration": (calib_test["uncalibrated"]["points"], calib_test["uncalibrated"]["brier"]),
            "After sigmoid calibration": (calib_test["calibrated"]["points"], calib_test["calibrated"]["brier"]),
        }, figures_dir / "calibration_curve.png"),
    ]
    return {"models": per_model, "final_model": final, "calibration_test": calib_test, "figures": figs}


# --------------------------------------------------------------------------- #
# Step 11: bootstrap ensemble for uncertainty
# --------------------------------------------------------------------------- #
def bootstrap_step(final_estimator, X_train, y_train, X_test, threshold) -> tuple[list, dict]:
    t0 = time.perf_counter()
    models = train_bootstrap_ensemble(final_estimator, X_train, y_train)
    widths, borderline = [], 0
    for i in range(len(X_test)):
        row = X_test.iloc[[i]]
        u = bootstrap_uncertainty(models, row, threshold, final_estimator.predict_proba(row)[0, 1])
        widths.append(u["width"])
        borderline += u["borderline"]
    summary = {
        "n_models": len(models),
        "interval": "5th-95th percentile of bootstrap model estimates",
        "test_mean_interval_width": round(float(np.mean(widths)), 4),
        "test_median_interval_width": round(float(np.median(widths)), 4),
        "test_share_borderline": round(borderline / len(X_test), 4),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    log(f"  {len(models)} bootstrap models; mean 90% interval width on test set "
        f"{summary['test_mean_interval_width']:.3f}; {summary['test_share_borderline']:.1%} of test "
        f"patients borderline ({summary['seconds']}s)")
    return models, summary


# --------------------------------------------------------------------------- #
# Step 12: SHAP
# --------------------------------------------------------------------------- #
def shap_step(final_estimator, X_train, X_test, figures_dir) -> tuple[pd.DataFrame, dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    t0 = time.perf_counter()
    background = X_train.sample(BACKGROUND_SIZE, random_state=RANDOM_STATE)
    explainer = build_explainer(final_estimator, background)
    values, base = shap_values_for(explainer, X_test)
    glob = global_explanation(explainer, X_test, values, base)
    glob["seconds"] = round(time.perf_counter() - t0, 1)

    # Figures for the report: shap's own summary (beeswarm) and bar plots
    disp = X_test[CSV_COLUMNS].copy().astype(float)
    disp[ZERO_AS_MISSING_COLUMNS] = disp[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    for kind, name in (("dot", "shap_summary_beeswarm.png"), ("bar", "shap_importance_bar.png")):
        shap.summary_plot(values, disp, plot_type=kind, show=False)
        plt.title("SHAP - held-out test set (contribution to predicted probability)", fontsize=10)
        plt.tight_layout()
        plt.savefig(figures_dir / name, dpi=130)
        plt.close("all")
    top = ", ".join(f"{d['label']} ({d['mean_abs_shap']:.3f})" for d in glob["importance"][:3])
    log(f"  Global SHAP on {len(X_test)} test patients: top features {top} ({glob['seconds']}s)")
    return background, glob


def training_ranges(X_train: pd.DataFrame) -> dict:
    """Observed min/max of each feature in the training set (zeros treated as missing)."""
    clean = X_train.copy()
    clean[ZERO_AS_MISSING_COLUMNS] = clean[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    return {CSV_TO_KEY[c]: {"min": float(clean[c].min()), "max": float(clean[c].max())}
            for c in CSV_COLUMNS}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    t_start = time.perf_counter()
    metrics_dir, figures_dir, models_dir = settings.METRICS_DIR, settings.FIGURES_DIR, settings.MODELS_DIR
    for d in (metrics_dir, figures_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load
    log(f"Loading dataset: {settings.DATA_PATH}")
    try:
        raw = load_dataset(settings.DATA_PATH)
    except DatasetError as exc:
        log(f"ERROR: {exc}")
        return 1
    df, n_dupes = remove_duplicates(raw)
    log(f"  {len(raw)} rows, {n_dupes} duplicates removed -> {len(df)} rows")

    # 2. Data quality + EDA
    quality = data_quality_report(df)
    quality["n_duplicates_removed"] = n_dupes
    save_json(quality, metrics_dir / "data_quality.json")
    figs = generate_eda_figures(df, figures_dir)
    log(f"  Data-quality report + {len(figs)} EDA figures written")

    # 3. Split
    X, y = df[CSV_COLUMNS], df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    log(f"  Train {len(X_train)} rows ({y_train.mean():.1%} positive) | "
        f"Test {len(X_test)} rows ({y_test.mean():.1%} positive)")
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df.to_csv(models_dir / "test_set.csv", index=False)

    # 4. Feature selection
    log("Feature-selection analysis (training set only)")
    fs = feature_selection_analysis(X_train, y_train)
    save_json(fs, metrics_dir / "feature_selection.json")
    log(f"  {fs['decision']}")

    # 5. Tune all models
    log(f"Hyperparameter tuning: GridSearchCV, stratified {N_FOLDS}-fold CV, refit on {PRIMARY_METRIC}")
    results = [tune_model(k, cfg, X_train, y_train) for k, cfg in MODEL_ZOO.items()]

    # 6. Ensemble
    results.append(build_ensemble(results, X_train, y_train))

    # 7. Select
    chosen, reason = select_model(results)
    log(f"Selected: {chosen['display_name']}")

    # 8. Calibration
    log("Calibration check (paired CV on the training set)")
    calib = calibration_step(chosen["estimator"], X_train, y_train)
    final_estimator = calib["calibrated_estimator"] if calib["used"] else calib["uncalibrated_estimator"]
    final_name = chosen["display_name"] + (" + sigmoid calibration" if calib["used"] else "")

    # 9. Threshold
    log("Decision threshold")
    thr = threshold_step(final_estimator, X_train, y_train)

    # 10. Test-set evaluation
    log("Evaluating on the held-out test set (first and only use)")
    evaluation = evaluate_on_test(results, final_estimator, final_name, calib, thr["value"],
                                  X_test, y_test, figures_dir)
    fm = evaluation["final_model"]["metrics_at_threshold"]
    ci = evaluation["final_model"]["bootstrap"]
    log(f"  Final model test ROC-AUC {fm['roc_auc']:.4f} (95% CI {ci['roc_auc_95ci'][0]:.3f}-"
        f"{ci['roc_auc_95ci'][1]:.3f}), Brier {fm['brier']:.4f}, recall {fm['recall']:.3f}, "
        f"specificity {fm['specificity']:.3f}")

    # 11. Bootstrap ensemble
    log("Bootstrap ensemble for uncertainty")
    boot_models, boot_summary = bootstrap_step(final_estimator, X_train, y_train, X_test, thr["value"])

    # 12. SHAP
    log("SHAP explanations")
    background, shap_global = shap_step(final_estimator, X_train, X_test, figures_dir)
    save_json(shap_global, metrics_dir / "shap_global.json")

    # 13. Save
    joblib.dump(final_estimator, models_dir / settings.PIPELINE_FILE)
    joblib.dump(boot_models, models_dir / settings.BOOTSTRAP_FILE)
    joblib.dump(background, models_dir / settings.SHAP_BACKGROUND_FILE)
    joblib.dump({r["key"]: r["estimator"] for r in results}, models_dir / settings.ALL_MODELS_FILE)

    cv_table = [{k: v for k, v in r.items() if k != "estimator"} for r in results]
    save_json({
        "cv_method": f"Stratified {N_FOLDS}-fold cross-validation (shuffled, random_state={RANDOM_STATE}) "
                     "on the training set; tuning by GridSearchCV",
        "primary_metric": PRIMARY_METRIC,
        "selection_rule": f"Best mean CV ROC-AUC; models within {TIE_TOLERANCE} treated as tied, "
                          "tie broken by lowest CV Brier score",
        "selected_model": chosen["key"],
        "selection_reason": reason,
        "models": cv_table,
    }, metrics_dir / "cv_results.json")
    pd.DataFrame([
        {"model": r["display_name"],
         **{f"cv_{m}_mean": r["cv"][m]["mean"] for m in CV_SCORING},
         **{f"cv_{m}_std": r["cv"][m]["std"] for m in CV_SCORING}}
        for r in results
    ]).to_csv(metrics_dir / "cv_results.csv", index=False)

    calibration_info = {k: v for k, v in calib.items() if not k.endswith("_estimator")}
    save_json({
        "test_set": {"n": int(len(y_test)), "n_positive": int(y_test.sum()),
                     "positive_rate": round(float(y_test.mean()), 4)},
        "threshold": thr,
        "calibration": {**calibration_info, "test_curves": evaluation["calibration_test"],
                        "n_bins": 10, "binning": "quantile"},
        "final_model": evaluation["final_model"],
        "uncertainty": boot_summary,
        "models": evaluation["models"],
        "figures": evaluation["figures"] + ["shap_summary_beeswarm.png", "shap_importance_bar.png"],
    }, metrics_dir / "evaluation.json")
    pd.DataFrame([
        {"model": m["display_name"], **{k: v for k, v in m["test_metrics_at_0_5"].items()
                                         if k != "confusion_matrix"}}
        for m in evaluation["models"]
    ] + [
        {"model": final_name + " (tuned threshold)",
         **{k: v for k, v in fm.items() if k != "confusion_matrix"}}
    ]).to_csv(metrics_dir / "test_metrics.csv", index=False)

    metadata = {
        "selected_model": chosen["key"],
        "selected_model_display": final_name,
        "base_model_display": chosen["display_name"],
        "selection_reason": reason,
        "best_params": chosen["best_params"],
        "engineered_features": chosen["engineered_features"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "dataset": {"path": str(settings.DATA_PATH.name), "n_rows": len(df),
                    "n_train": len(X_train), "n_test": len(X_test),
                    "test_size": TEST_SIZE, "positive_rate_train": float(y_train.mean())},
        "features": [f.key for f in FEATURES],
        "training_ranges": training_ranges(X_train),
        "threshold": thr["value"],
        "threshold_method": thr["method"],
        "calibrated": calib["used"],
        "calibration_decision": calib["decision"],
        "calibration_summary": (
            "Probabilities come from Platt (sigmoid) scaling built into the SVM step. "
            if chosen["key"] == "svm" else ""
        ) + calib["decision"],
        "n_bootstrap": N_BOOTSTRAP,
        "shap_background_size": BACKGROUND_SIZE,
        "model_keys": [r["key"] for r in results],
        "model_display_names": {r["key"]: r["display_name"] for r in results},
    }
    save_json(metadata, models_dir / settings.METADATA_FILE)

    log(f"Artifacts saved to {models_dir}")
    log(f"Metrics saved to {metrics_dir}")
    log(f"Done in {time.perf_counter() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
