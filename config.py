"""
config.py — Centralized application configuration.

All tuneable constants live here. No secrets or credentials are hardcoded;
sensitive values should be supplied via environment variables.
"""

import os
from pathlib import Path

# ── Project root ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

# ── Data & model paths ───────────────────────────────────────────────────────
# Always use the canonical dataset AI-Data.csv (480 rows) matching the reference project.
DATA_PATH   = BASE_DIR / "data" / "AI-Data.csv"
MODELS_PATH = BASE_DIR / "models" / "pipelines.joblib"

# ── Dataset / feature config ─────────────────────────────────────────────────
# 6 core features: 4 engagement & attendance features from original dataset
# plus 2 synthesized features (AttendancePercentage, StudyHours) authorized by user.
# Order matters — pipelines are fitted and predictions built in this exact order.
FEATURE_COLUMNS = [
    "raisedhands",          # int 0–100
    "VisITedResources",     # int 0–100
    "Discussion",           # int 0–100
    "StudentAbsenceDays",   # categorical: "Under-7" | "Above-7"
    "AttendancePercentage", # float/int 0–100 %
    "StudyHours",           # float 0–24 hrs/day
]

FEATURE_DISPLAY_NAMES = {
    "raisedhands": "Raised Hands",
    "VisITedResources": "Visited Resources",
    "Discussion": "Discussion",
    "StudentAbsenceDays": "Absence Days",
    "AttendancePercentage": "Attendance %",
    "StudyHours": "Study Hours",
}
TARGET_COLUMN = "Class"                  # L / M / H
CLASS_ORDER = ["L", "M", "H"]           # display order
CLASS_LABELS = {"L": "Low", "M": "Medium", "H": "High"}

# Absence encoding (match original Project.py logic, inverted: Under-7 → 1)
ABSENCE_ENCODING = {"Under-7": 1, "Above-7": 0}

# ── Pass / Fail rule definition ──────────────────────────────────────────────
# In the Kalboard 360 / UCI dataset, performance classes are defined by grade intervals:
#   Low (L): 0–69 marks
#   Medium (M): 70–89 marks
#   High (H): 90–100 marks
# Low performance represents failing to meet academic benchmarks.
PASS_FAIL_MAPPING = {"L": "Fail", "M": "Pass", "H": "Pass"}
PASS_FAIL_RULE_DESC = "Low (0–69) = Fail | Middle (70–89) & High (90–100) = Pass (Kalboard 360 Grade Intervals)"

# ── Ranking criteria documentation ───────────────────────────────────────────
RANKING_CRITERIA_DOC = (
    "Models are evaluated on held-out test data (30% split, stratified). "
    "Ranked primarily by weighted F1-Score to account for class distribution, "
    "with Test Accuracy, Precision, and Recall as tie-breakers."
)

# ── Train / test split ───────────────────────────────────────────────────────
TEST_SIZE = 0.30
RANDOM_STATE = 42          # reproducible splits everywhere

# ── Model names ──────────────────────────────────────────────────────────────
MODEL_KEYS = [
    "decision_tree",
    "random_forest",
    "logistic_regression",
    "perceptron",
    "mlp",
]
MODEL_DISPLAY_NAMES = {
    "decision_tree":      "Decision Tree",
    "random_forest":      "Random Forest",
    "logistic_regression":"Logistic Regression",
    "perceptron":         "Perceptron",
    "mlp":                "MLP Neural Network",
}

# ── Input validation bounds ──────────────────────────────────────────────────
INT_FEATURE_MIN = 0
INT_FEATURE_MAX = 100
ATTENDANCE_MIN = 0.0
ATTENDANCE_MAX = 100.0
STUDY_HOURS_MIN = 0.0
STUDY_HOURS_MAX = 24.0

# ── Flask / Dash ─────────────────────────────────────────────────────────────
# Never enable DEBUG in production; use env var to override.
DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
PORT = int(os.environ.get("FLASK_PORT", "8050"))

# Secret key — override via environment variable in production.
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-in-production")

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
