"""
data_pipeline.py — Data loading, zero-target-leakage feature synthesis,
feature preparation, and strict input validation.

Guarantees:
1. Zero Target Leakage: When synthesizing AttendancePercentage and StudyHours,
   only observable student behaviors (absence category and engagement metrics)
   are used. The ground truth target 'Class' is NEVER used during synthesis.
2. Preprocessing Isolation: Feature matrices are extracted as raw numbers;
   all scaling and transformations are deferred strictly to scikit-learn
   Pipelines during training, preventing train/test data contamination.
"""

import logging
from typing import Any
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Synthetic feature generator (Zero Target Leakage) ─────────────────────────

def generate_synthetic_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic AttendancePercentage and StudyHours based strictly
    on observable student engagement and absence records.

    Zero Target Leakage:
    The ground-truth 'Class' column is strictly NEVER used in generating
    either AttendancePercentage or StudyHours. Both features are functions
    of observable student behavior plus realistic stochastic variance.
    """
    df = df.copy()
    rng = np.random.default_rng(seed)

    att_list = []
    study_list = []

    for _, row in df.iterrows():
        abs_cat = str(row.get("StudentAbsenceDays", "Under-7"))
        rh = float(row.get("raisedhands", 50))
        vr = float(row.get("VisITedResources", 50))
        dis = float(row.get("Discussion", 50))

        # Overall LMS engagement ratio: 0.0 to 1.0
        eng = (rh + vr + dis) / 300.0

        # Attendance Percentage:
        # Grounded in official absence category (Under-7 vs Above-7)
        if abs_cat == "Under-7":
            # Absent < 7 days in semester -> high attendance (85% to 100%)
            base_att = 92.0 + 5.0 * eng
            att = rng.normal(base_att, 2.5)
            att = np.clip(att, 85.0, 100.0)
        else:
            # Absent >= 7 days in semester -> reduced attendance (40% to 84%)
            base_att = 62.0 + 12.0 * eng
            att = rng.normal(base_att, 5.5)
            att = np.clip(att, 40.0, 84.0)

        # Study Hours per day:
        # Correlated with resource utilization, lecture participation, and attendance habit
        base_sh = 1.2 + 4.2 * (0.45 * (vr / 100.0) + 0.35 * (rh / 100.0) + 0.20 * (dis / 100.0))
        if abs_cat == "Under-7":
            base_sh += 0.8
        sh = rng.normal(base_sh, 0.75)
        sh = np.clip(sh, 0.5, 9.0)

        att_list.append(round(float(att), 1))
        study_list.append(round(float(sh), 1))

    df["AttendancePercentage"] = att_list
    df["StudyHours"] = study_list
    return df


# ── Dataset loading & validation ─────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """
    Load the source CSV and ensure all required feature columns exist.
    If AttendancePercentage or StudyHours are missing, generates them
    with zero target leakage.
    """
    path = config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Check for synthetic features; synthesize with zero target leakage if missing
    if "AttendancePercentage" not in df.columns or "StudyHours" not in df.columns:
        logger.info("Synthesizing AttendancePercentage and StudyHours without target leakage...")
        df = generate_synthetic_features(df)
        df.to_csv(path, index=False)

    required = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {missing}")

    logger.info("Dataset loaded: %d rows, %d columns", *df.shape)
    return df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Return summary statistics calculated dynamically from the loaded dataframe.
    Calculates total records (excluding CSV header), valid rows, missing value
    counts, class distribution, and feature counts. Zero hardcoded values.
    """
    total_records = int(len(df))
    required_cols = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    valid_mask = df[required_cols].notnull().all(axis=1)
    valid_records = int(valid_mask.sum())
    missing_records = total_records - valid_records

    counts = df[config.TARGET_COLUMN].value_counts().to_dict()
    class_distribution = {
        cls: int(counts.get(cls, 0))
        for cls in config.CLASS_ORDER
    }

    return {
        "total_records": total_records,
        "valid_records": valid_records,
        "missing_records": missing_records,
        "num_features": len(config.FEATURE_COLUMNS),
        "num_classes": int(df[config.TARGET_COLUMN].nunique()),
        "class_distribution": class_distribution,
        "num_models": len(config.MODEL_KEYS),
        "dataset_name": config.DATA_PATH.name,
    }


# ── Feature extraction (Raw vectors, no scaling applied here) ────────────────

def _encode_absence(series: pd.Series) -> np.ndarray:
    """Map StudentAbsenceDays to int: Under-7 -> 1, Above-7 -> 0."""
    return series.map(config.ABSENCE_ENCODING).fillna(0).astype(int).values


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract raw feature matrix X and label array y from dataframe.

    Order strictly matches config.FEATURE_COLUMNS:
      [raisedhands, VisITedResources, Discussion, StudentAbsenceDays_encoded, AttendancePercentage, StudyHours]

    Note: Scaling is NOT performed here to prevent train/test leakage.
    Scaling is applied inside sklearn Pipelines during cross-validation/training.
    """
    X_num1 = df[["raisedhands", "VisITedResources", "Discussion"]].values.astype(float)
    X_abs  = _encode_absence(df["StudentAbsenceDays"]).reshape(-1, 1).astype(float)
    X_num2 = df[["AttendancePercentage", "StudyHours"]].values.astype(float)
    X = np.hstack([X_num1, X_abs, X_num2])
    y = df[config.TARGET_COLUMN].values
    return X, y


def prepare_single_input(
    raisedhands: int,
    visited_resources: int,
    discussion: int,
    absence_days: str,
    attendance_pct: float | None = None,
    study_hours: float | None = None,
) -> np.ndarray:
    """
    Convert a single prediction request into the exact 2D feature array
    expected by fitted sklearn pipelines.
    """
    absence_encoded = float(config.ABSENCE_ENCODING.get(absence_days, 0))

    # Derive sensible defaults if omitted
    if attendance_pct is None:
        attendance_pct = 94.0 if absence_days == "Under-7" else 68.0
    if study_hours is None:
        study_hours = 4.0 if absence_days == "Under-7" else 2.5

    return np.array([[
        float(raisedhands),
        float(visited_resources),
        float(discussion),
        float(absence_encoded),
        float(attendance_pct),
        float(study_hours),
    ]], dtype=float)


# ── Strict Server-Side Input Validation ──────────────────────────────────────

def validate_inputs(
    raisedhands: Any,
    visited_resources: Any,
    discussion: Any,
    absence_days: Any,
    attendance_pct: Any = 85.0,
    study_hours: Any = 4.0,
) -> tuple[bool, str, dict]:
    """
    Validate user inputs strictly on the server side before inference.
    Prevents malformed data, type mismatches, and boundary violations.
    """
    errors = []
    cleaned: dict = {}

    def _check_num(name: str, val: Any, lo: float, hi: float, is_int: bool = True) -> float | None:
        try:
            v = int(val) if is_int else float(val)
        except (TypeError, ValueError):
            errors.append(f"{name} must be a valid number between {lo} and {hi}.")
            return None
        if not (lo <= v <= hi):
            errors.append(f"{name} must be between {lo} and {hi} (got {v}).")
            return None
        return v

    cleaned["raisedhands"] = _check_num(
        "Raised Hands", raisedhands, config.INT_FEATURE_MIN, config.INT_FEATURE_MAX, is_int=True
    )
    cleaned["visited_resources"] = _check_num(
        "Visited Resources", visited_resources, config.INT_FEATURE_MIN, config.INT_FEATURE_MAX, is_int=True
    )
    cleaned["discussion"] = _check_num(
        "Discussion", discussion, config.INT_FEATURE_MIN, config.INT_FEATURE_MAX, is_int=True
    )

    valid_absences = list(config.ABSENCE_ENCODING.keys())
    if absence_days not in valid_absences:
        errors.append(f"Absence Days must be one of: {valid_absences}.")
    else:
        cleaned["absence_days"] = absence_days

    # Attendance Percentage (0-100)
    att_val = attendance_pct if attendance_pct is not None else 85.0
    cleaned["attendance_pct"] = _check_num(
        "Attendance Percentage", att_val, config.ATTENDANCE_MIN, config.ATTENDANCE_MAX, is_int=False
    )

    # Study Hours (0-24)
    sh_val = study_hours if study_hours is not None else 4.0
    cleaned["study_hours"] = _check_num(
        "Study Hours per Day", sh_val, config.STUDY_HOURS_MIN, config.STUDY_HOURS_MAX, is_int=False
    )

    if errors:
        return False, " | ".join(errors), {}
    return True, "", cleaned
