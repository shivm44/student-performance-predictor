"""
augment_data.py — Augment AI-Data.csv with statistically-grounded synthetic records.

Method: Gaussian KDE (Kernel Density Estimation) fitted independently per class
on the real distribution of each continuous feature. Categorical feature
(StudentAbsenceDays) is sampled from the real conditional probability per class.
All other columns are sampled from the original rows to preserve real demographic
distributions.

KDE-based augmentation is the standard approach used in published educational
data mining research when the source dataset is a closed, fixed-size repository
(Xapi-Edu-Data has exactly 480 real rows — no larger real version exists).

Output: data/AI-Data-Augmented.csv (original 480 + 1120 synthetic = 1600 rows)

Usage:
    python augment_data.py
"""

import logging
import sys
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Real distributions confirmed from the dataset (480 rows)
# Used as ground-truth priors for KDE sampling
CLASS_STATS = {
    "L": {
        "n_original": 127,
        "n_synthetic": 373,      # → 500 total L rows
        "under7_prob": 11 / 127, # 0.087 — Low performers almost always absent >7 days
    },
    "M": {
        "n_original": 211,
        "n_synthetic": 389,      # → 600 total M rows
        "under7_prob": 140 / 211,# 0.663
    },
    "H": {
        "n_original": 142,
        "n_synthetic": 358,      # → 500 total H rows
        "under7_prob": 138 / 142,# 0.972 — High performers almost never miss >7 days
    },
}

NUMERIC_FEATURES = ["raisedhands", "VisITedResources", "AnnouncementsView", "Discussion"]
CATEGORICAL_COLS = [
    "gender", "NationalITy", "PlaceofBirth", "StageID", "GradeID",
    "SectionID", "Topic", "Semester", "Relation",
    "ParentAnsweringSurvey", "ParentschoolSatisfaction",
]

SRC  = r"C:\Users\SHIVAM\OneDrive - Shivam Infotech Pvt. Ltd\Desktop\student performance predictor\data\AI-Data.csv"
DEST = r"C:\Users\SHIVAM\OneDrive - Shivam Infotech Pvt. Ltd\Desktop\student performance predictor\data\AI-Data-Augmented.csv"


def kde_sample(values: np.ndarray, n: int, lo: int = 0, hi: int = 100, rng: np.random.Generator = None) -> np.ndarray:
    """
    Draw n samples from a Gaussian KDE fitted on values, clipped to [lo, hi]
    and rounded to integers.  Rejection-samples until n valid values are collected.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    kde = gaussian_kde(values, bw_method="scott")
    result = []
    while len(result) < n:
        batch = kde.resample(n * 3, seed=rng.integers(1e9))[0]
        batch = np.round(batch).astype(int)
        batch = batch[(batch >= lo) & (batch <= hi)]
        result.extend(batch.tolist())
    return np.array(result[:n])


def generate_synthetic(df: pd.DataFrame, cls: str, info: dict, rng: np.random.Generator) -> pd.DataFrame:
    n = info["n_synthetic"]
    sub = df[df["Class"] == cls].reset_index(drop=True)

    # ── Numeric features via KDE ─────────────────────────────────────────────
    synth_numeric: dict = {}
    for feat in NUMERIC_FEATURES:
        synth_numeric[feat] = kde_sample(sub[feat].values, n, rng=rng)

    # ── Absence: sample from real conditional probability ────────────────────
    absence = rng.choice(
        ["Under-7", "Above-7"],
        size=n,
        p=[info["under7_prob"], 1 - info["under7_prob"]],
    )

    # ── Categorical: bootstrap from real rows ────────────────────────────────
    idx = rng.integers(0, len(sub), size=n)
    cat_df = sub[CATEGORICAL_COLS].iloc[idx].reset_index(drop=True)

    synth_df = pd.DataFrame(synth_numeric)
    synth_df["StudentAbsenceDays"] = absence
    synth_df["Class"] = cls
    synth_df = pd.concat([cat_df, synth_df[NUMERIC_FEATURES + ["StudentAbsenceDays", "Class"]]], axis=1)

    # Reorder columns to match original
    synth_df = synth_df[df.columns]
    return synth_df


def main() -> int:
    rng = np.random.default_rng(42)

    logger.info("Loading original dataset...")
    df = pd.read_csv(SRC)
    logger.info("Original: %d rows", len(df))

    parts = [df]  # Start with original 480 rows

    for cls, info in CLASS_STATS.items():
        logger.info("Generating %d synthetic '%s' rows...", info["n_synthetic"], cls)
        synthetic = generate_synthetic(df, cls, info, rng)
        parts.append(synthetic)
        logger.info(
            "  Class %s: original=%d + synthetic=%d = %d total",
            cls, info["n_original"], info["n_synthetic"],
            info["n_original"] + info["n_synthetic"],
        )

    combined = pd.concat(parts, ignore_index=True)
    # Shuffle so original and synthetic rows are interleaved
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("Combined dataset: %d rows", len(combined))
    logger.info("Class distribution:")
    for cls, cnt in combined["Class"].value_counts().items():
        logger.info("  %s: %d (%.1f%%)", cls, cnt, 100 * cnt / len(combined))

    # Validate numeric ranges
    for feat in NUMERIC_FEATURES:
        assert combined[feat].min() >= 0, f"{feat} has values < 0"
        assert combined[feat].max() <= 100, f"{feat} has values > 100"
    assert set(combined["StudentAbsenceDays"].unique()) == {"Under-7", "Above-7"}
    assert set(combined["Class"].unique()) == {"L", "M", "H"}

    combined.to_csv(DEST, index=False)
    logger.info("Saved to %s", DEST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
