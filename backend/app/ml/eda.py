"""
Exploratory data analysis figures (saved as PNG for the report / viva).

Every figure is drawn from the real dataset. Zero-encoded missing values are
shown as missing (not as 0) so the distributions are not distorted.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed; just write files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.ml.features import CSV_COLUMNS, TARGET_COLUMN, ZERO_AS_MISSING_COLUMNS

sns.set_theme(style="whitegrid")
CLASS_COLORS = {0: "#4C78A8", 1: "#E45756"}
CLASS_NAMES = {0: "Non-diabetic (0)", 1: "Diabetic (1)"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def generate_eda_figures(df: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = df.copy()
    clean[ZERO_AS_MISSING_COLUMNS] = clean[ZERO_AS_MISSING_COLUMNS].replace(0, np.nan)
    written = []

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
           color=[CLASS_COLORS[i] for i in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, f"{v} ({100 * v / len(df):.1f}%)", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    written.append(_save(fig, out_dir / "eda_class_balance.png"))

    # 2. Missing values (zeros treated as missing)
    fig, ax = plt.subplots(figsize=(6, 4))
    miss = (clean[CSV_COLUMNS].isna().mean() * 100).sort_values(ascending=False)
    ax.barh(miss.index, miss.values, color="#72B7B2")
    for i, v in enumerate(miss.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    ax.invert_yaxis()
    ax.set_xlabel("% of rows missing (recorded as 0 in the CSV)")
    ax.set_title("Missing values per feature")
    written.append(_save(fig, out_dir / "eda_missing_values.png"))

    # 3. Distributions by class
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), CSV_COLUMNS):
        for cls in (0, 1):
            ax.hist(clean.loc[df[TARGET_COLUMN] == cls, col].dropna(), bins=25, alpha=0.55,
                    color=CLASS_COLORS[cls], label=CLASS_NAMES[cls], density=True)
        ax.set_title(col)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Feature distributions by outcome (density)")
    written.append(_save(fig, out_dir / "eda_distributions.png"))

    # 4. Boxplots (outliers)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), CSV_COLUMNS):
        sns.boxplot(x=df[TARGET_COLUMN].map(CLASS_NAMES), y=clean[col], ax=ax,
                    hue=df[TARGET_COLUMN].map(CLASS_NAMES), legend=False,
                    palette=[CLASS_COLORS[0], CLASS_COLORS[1]])
        ax.set_xlabel("")
        ax.set_title(col)
    fig.suptitle("Boxplots by outcome (points beyond whiskers = IQR outliers)")
    written.append(_save(fig, out_dir / "eda_boxplots.png"))

    # 5. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6.5))
    corr = clean[CSV_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
                square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson correlation (missing values excluded pairwise)")
    written.append(_save(fig, out_dir / "eda_correlation_heatmap.png"))

    return written
