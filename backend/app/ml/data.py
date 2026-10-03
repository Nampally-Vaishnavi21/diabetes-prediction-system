"""
Data loading, validation and data-quality analysis.

WHAT: Reads diabetes.csv, checks it is the file we expect, and produces a
      data-quality report (missing values, duplicates, outliers, class balance).
WHY:  Training on a wrong or corrupted file would silently produce a bad model.
      Every number in the report is computed from the real CSV.
VIVA: "Before training, the pipeline validates the schema and target column,
      checks duplicates, counts the zero-encoded missing values and reports
      IQR outliers. Nothing is fixed here; actual cleaning happens inside the
      sklearn Pipeline so it is learned from training folds only."
"""
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features import CSV_COLUMNS, FEATURES, TARGET_COLUMN, ZERO_AS_MISSING_COLUMNS

DOWNLOAD_HELP = (
    "Download 'diabetes.csv' from "
    "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database "
    "and place it at backend/data/raw/diabetes.csv"
)


class DatasetError(Exception):
    """The dataset is missing or does not have the expected structure."""


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the CSV. Raises DatasetError with a clear message."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found at '{path}'. {DOWNLOAD_HELP}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not read '{path}' as CSV: {exc}") from exc

    expected = CSV_COLUMNS + [TARGET_COLUMN]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise DatasetError(
            f"Dataset is missing columns {missing_cols}. Expected header: {','.join(expected)}"
        )
    df = df[expected].copy()

    non_numeric = [c for c in expected if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DatasetError(f"Columns must be numeric: {non_numeric}")

    if df[expected].isna().any().any():
        # The original file has no blank cells (missing values are encoded as 0)
        raise DatasetError("Unexpected blank cells in dataset; the original file has none.")

    if not set(df[TARGET_COLUMN].unique()) <= {0, 1}:
        raise DatasetError(f"'{TARGET_COLUMN}' must contain only 0 and 1.")

    if (df[CSV_COLUMNS] < 0).any().any():
        raise DatasetError("Dataset contains negative feature values.")

    if len(df) < 100:
        raise DatasetError(f"Dataset has only {len(df)} rows; expected the full 768-row file.")

    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n = int(df.duplicated().sum())
    return df.drop_duplicates().reset_index(drop=True), n


def _iqr_outliers(series: pd.Series) -> dict:
    """Tukey's rule: outlier if below Q1 - 1.5*IQR or above Q3 + 1.5*IQR."""
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lo) | (s > hi)
    return {
        "q1": round(float(q1), 3), "q3": round(float(q3), 3), "iqr": round(float(iqr), 3),
        "lower_fence": round(float(lo), 3), "upper_fence": round(float(hi), 3),
        "n_outliers": int(mask.sum()), "pct_outliers": round(100 * float(mask.mean()), 2),
    }


def data_quality_report(df: pd.DataFrame) -> dict:
    """Compute the data-quality report from the raw dataframe."""
    n = len(df)
    counts = df[TARGET_COLUMN].value_counts().to_dict()

    # Zeros that really mean "not measured"
    zero_missing = {
        c: {"count": int((df[c] == 0).sum()), "pct": round(100 * float((df[c] == 0).mean()), 2)}
        for c in ZERO_AS_MISSING_COLUMNS
    }

    # Treat those zeros as missing before computing statistics/outliers
    clean = df.copy()
    clean[ZERO_AS_MISSING_COLUMNS] = clean[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)

    stats = {}
    for spec in FEATURES:
        s = clean[spec.csv_column]
        stats[spec.key] = {
            "column": spec.csv_column,
            "count_valid": int(s.notna().sum()),
            "mean": round(float(s.mean()), 3), "std": round(float(s.std()), 3),
            "min": round(float(s.min()), 3), "median": round(float(s.median()), 3),
            "max": round(float(s.max()), 3),
            "mean_diabetic": round(float(s[df[TARGET_COLUMN] == 1].mean()), 3),
            "mean_non_diabetic": round(float(s[df[TARGET_COLUMN] == 0].mean()), 3),
            "outliers_iqr": _iqr_outliers(s),
        }

    corr = clean[CSV_COLUMNS + [TARGET_COLUMN]].corr(method="pearson").round(3)

    return {
        "n_rows": n,
        "n_features": len(CSV_COLUMNS),
        "target": TARGET_COLUMN,
        "class_counts": {"non_diabetic_0": int(counts.get(0, 0)), "diabetic_1": int(counts.get(1, 0))},
        "positive_rate_pct": round(100 * float(df[TARGET_COLUMN].mean()), 2),
        "n_duplicates": int(df.duplicated().sum()),
        "zero_encoded_missing": zero_missing,
        "rows_with_any_missing": int(clean[ZERO_AS_MISSING_COLUMNS].isna().any(axis=1).sum()),
        "feature_stats": stats,
        "correlation_matrix": {
            "columns": list(corr.columns),
            "values": corr.values.tolist(),   # pairwise, ignores missing values
        },
        "outlier_policy": (
            "Outliers are reported (Tukey IQR rule) but not removed: the dataset is small "
            "and extreme values such as high insulin are clinically plausible. "
            "Physiologically impossible zeros are treated as missing instead."
        ),
    }
