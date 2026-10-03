"""
Model evaluation: metrics, curves, threshold selection, bootstrap confidence intervals.

Every number here is computed from real predictions. Curves are returned as
JSON arrays so the React app can draw them with Recharts, and also saved as
PNG figures for the written report.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

CALIBRATION_BINS = 10


def _r(x, n=4):
    return None if x is None else round(float(x), n)


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """All requested metrics for one set of predicted probabilities."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": _r(threshold),
        "accuracy": _r(accuracy_score(y_true, pred)),
        "precision": _r(precision_score(y_true, pred, zero_division=0)),
        "recall": _r(recall_score(y_true, pred, zero_division=0)),          # sensitivity
        "specificity": _r(tn / (tn + fp) if (tn + fp) else 0.0),           # true-negative rate
        "f1": _r(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": _r(roc_auc_score(y_true, prob)),
        "pr_auc": _r(average_precision_score(y_true, prob)),
        "brier": _r(brier_score_loss(y_true, prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
    }


def youden_threshold(y_true, prob) -> dict:
    """Threshold maximising Youden's J = sensitivity + specificity - 1.

    Applied to OUT-OF-FOLD training predictions, never to the test set.
    """
    fpr, tpr, thr = roc_curve(y_true, prob)
    j = tpr - fpr
    finite = np.isfinite(thr)            # first roc threshold is +inf
    i = int(np.argmax(np.where(finite, j, -np.inf)))
    return {"threshold": _r(thr[i]), "youden_j": _r(j[i]),
            "sensitivity": _r(tpr[i]), "specificity": _r(1 - fpr[i])}


def roc_points(y_true, prob) -> list[dict]:
    fpr, tpr, thr = roc_curve(y_true, prob)
    return [{"fpr": _r(f), "tpr": _r(t)} for f, t in zip(fpr, tpr)]


def pr_points(y_true, prob) -> list[dict]:
    precision, recall, _ = precision_recall_curve(y_true, prob)
    # sklearn returns recall in decreasing order; reverse for left-to-right plotting
    pts = [{"recall": _r(r), "precision": _r(p)} for p, r in zip(precision, recall)]
    return pts[::-1]


def calibration_points(y_true, prob) -> list[dict]:
    """Reliability diagram: mean predicted probability vs observed positive rate per bin.

    Quantile bins put ~equal numbers of patients in each bin, which is more
    stable than equal-width bins on a small test set.
    """
    frac_pos, mean_pred = calibration_curve(y_true, prob, n_bins=CALIBRATION_BINS, strategy="quantile")
    return [{"mean_predicted": _r(m), "fraction_positive": _r(f)} for m, f in zip(mean_pred, frac_pos)]


def bootstrap_ci(y_true, prob, n_boot: int = 1000, seed: int = 42) -> dict:
    """95% percentile bootstrap CI of test ROC-AUC and Brier score.

    WHY: with 154 test patients one headline number hides a lot of uncertainty.
    Resampling the test set shows how much that number could move.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)
    aucs, briers = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], prob[idx]))
        briers.append(brier_score_loss(y_true[idx], prob[idx]))
    return {
        "n_bootstrap": len(aucs),
        "roc_auc_95ci": [_r(np.percentile(aucs, 2.5)), _r(np.percentile(aucs, 97.5))],
        "brier_95ci": [_r(np.percentile(briers, 2.5)), _r(np.percentile(briers, 97.5))],
    }


# --------------------------------------------------------------------------- #
# Figures for the report
# --------------------------------------------------------------------------- #
def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def plot_roc(curves: dict, aucs: dict, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, pts in curves.items():
        ax.plot([p["fpr"] for p in pts], [p["tpr"] for p in pts], label=f"{name} (AUC {aucs[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (recall)",
           title="ROC curves - held-out test set")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_pr(curves: dict, aps: dict, base_rate: float, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, pts in curves.items():
        ax.plot([p["recall"] for p in pts], [p["precision"] for p in pts], label=f"{name} (AP {aps[name]:.3f})")
    ax.axhline(base_rate, color="k", ls="--", lw=1, label=f"Baseline ({base_rate:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves - held-out test set",
           ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, path)


def plot_confusion(cm: dict, title: str, path: Path) -> str:
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=16,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["True 0", "True 1"])
    ax.set_title(title, fontsize=10)
    return _save(fig, path)


def plot_calibration(series: dict, path: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for name, (pts, brier) in series.items():
        ax.plot([p["mean_predicted"] for p in pts], [p["fraction_positive"] for p in pts],
                marker="o", label=f"{name} (Brier {brier:.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction diabetic",
           title="Calibration (reliability) curve - test set", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper left")
    return _save(fig, path)
