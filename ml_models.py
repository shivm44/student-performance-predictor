"""
ml_models.py — Machine Learning pipelines, model training, evaluation,
model ranking, artifact persistence, and multi-model inference.

Data Leakage Prevention:
1. Preprocessing is encapsulated inside sklearn Pipelines. Feature scaling
   is fitted strictly on the training partition (X_train) and applied
   out-of-sample to the held-out test partition (X_test).
2. Saved artifacts store only fitted estimators, evaluation metrics, and
   aggregate counts. Raw test sets are stripped prior to serialization.
"""

import logging
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from data_pipeline import prepare_features

logger = logging.getLogger(__name__)


# ── Model Pipeline Definitions ────────────────────────────────────────────────

def _build_pipelines() -> dict[str, Pipeline]:
    """
    Construct sklearn Pipelines for all 5 algorithms.

    Each pipeline encapsulates:
      StandardScaler -> Estimator
    Scaling parameters (mean, std) are learned EXCLUSIVELY on training data
    during pipeline.fit(), eliminating train-test data contamination.
    """
    return {
        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=config.RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
            )),
        ]),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=500,
                random_state=config.RANDOM_STATE,
            )),
        ]),
        "perceptron": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", Perceptron(
                max_iter=1000,
                random_state=config.RANDOM_STATE,
                tol=1e-3,
            )),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="logistic",
                max_iter=500,
                random_state=config.RANDOM_STATE,
            )),
        ]),
    }


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_all_models(df: pd.DataFrame) -> dict:
    """
    Train all 5 pipelines on the dataset, evaluate on the held-out test set,
    and return results.

    Methodology:
    - Train/test split: 70% train / 30% test with stratified sampling on Class.
    - Zero Preprocessing Leakage: StandardScaler is fitted strictly on X_train.
    - Zero Target Leakage: Target 'Class' is used only as label for supervision.
    """
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    class_names = sorted(df[config.TARGET_COLUMN].unique())

    pipelines = _build_pipelines()
    results: dict = {
        "pipelines": {},
        "metrics": {},
        "X_test": X_test,
        "y_test": y_test,
        "class_names": class_names,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    for key, pipeline in pipelines.items():
        display = config.MODEL_DISPLAY_NAMES[key]
        logger.info("Training %s...", display)
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = pipeline.predict(X_test)
        has_proba = hasattr(pipeline.named_steps["clf"], "predict_proba")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=class_names).tolist()
        report = classification_report(
            y_test, y_pred,
            labels=class_names,
            target_names=[config.CLASS_LABELS[c] for c in class_names],
            zero_division=0,
            output_dict=True,
        )

        results["pipelines"][key] = pipeline
        results["metrics"][key] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "train_time_s": round(train_time, 4),
            "has_proba": has_proba,
            "y_pred": y_pred.tolist(),
        }
        logger.info(
            "%s — acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  time=%.3fs",
            display, acc, prec, rec, f1, train_time,
        )

    return results


# ── Persistence (No Raw Data Saved) ───────────────────────────────────────────

def save_results(results: dict) -> None:
    """
    Persist only fitted pipelines, metrics, and metadata.
    Raw test vectors (X_test, y_test) are deliberately excluded to prevent
    data exposure or storage leakage.
    """
    path = config.MODELS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipelines": results["pipelines"],
        "metrics": results["metrics"],
        "class_names": results["class_names"],
        "train_size": results["train_size"],
        "test_size": results["test_size"],
    }
    joblib.dump(payload, path)
    logger.info("Results saved to %s", path)


def load_results() -> dict | None:
    """Load persisted pipelines and metrics from disk."""
    path = config.MODELS_PATH
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        logger.info("Models loaded from %s", path)
        return payload
    except Exception as exc:
        logger.error("Failed to load models: %s", exc)
        return None


# ── Prediction & Multi-Model Inference ────────────────────────────────────────

def predict(pipeline: Pipeline, X: np.ndarray, class_names: list[str]) -> dict:
    """
    Execute inference with a fitted pipeline on a single feature vector.
    Returns predicted class label and genuine probabilities if supported.
    """
    raw_pred = pipeline.predict(X)[0]
    pred_label = str(raw_pred)
    probabilities = {}
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        proba = pipeline.predict_proba(X)[0]
        probabilities = {
            str(cls): round(float(p), 4)
            for cls, p in zip(pipeline.classes_, proba)
        }
    return {
        "predicted_class": pred_label,
        "predicted_label": config.CLASS_LABELS.get(pred_label, pred_label),
        "probabilities": probabilities,
    }


def get_model_rankings(metrics: dict[str, dict]) -> list[dict]:
    """
    Rank models dynamically by held-out test F1-score (weighted average).
    Tie-breakers: Accuracy, Precision, Recall.
    """
    scored = []
    for key, m in metrics.items():
        scored.append({
            "key": key,
            "display_name": config.MODEL_DISPLAY_NAMES.get(key, key),
            "f1": float(m.get("f1", 0.0)),
            "accuracy": float(m.get("accuracy", 0.0)),
            "precision": float(m.get("precision", 0.0)),
            "recall": float(m.get("recall", 0.0)),
            "has_proba": bool(m.get("has_proba", False)),
            "train_time_s": float(m.get("train_time_s", 0.0)),
        })

    scored.sort(
        key=lambda x: (x["f1"], x["accuracy"], x["precision"], x["recall"]),
        reverse=True,
    )

    for i, item in enumerate(scored, 1):
        item["rank"] = i
        item["is_top_3"] = (i <= 3)

    return scored


def predict_top_models(
    pipelines: dict[str, Pipeline],
    top_models: list[dict],
    X: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    """
    Pass the validated feature vector through all top-3 pipelines simultaneously.
    """
    results = []
    for model_info in top_models:
        key = model_info["key"]
        pipeline = pipelines.get(key)
        item = dict(model_info)
        if pipeline is None:
            item["predicted_class"] = "N/A"
            item["predicted_label"] = "Unavailable"
            item["pass_fail"] = "N/A"
            item["probabilities"] = {}
            item["error"] = f"Trained pipeline for '{model_info['display_name']}' is not available."
        else:
            try:
                pred = predict(pipeline, X, class_names)
                cls_str = str(pred["predicted_class"])
                item["predicted_class"] = cls_str
                item["predicted_label"] = pred["predicted_label"]
                item["pass_fail"] = config.PASS_FAIL_MAPPING.get(cls_str, "N/A")
                item["probabilities"] = pred["probabilities"]
                item["error"] = None
            except Exception as exc:
                logger.exception("Prediction failed for %s: %s", key, exc)
                item["predicted_class"] = "N/A"
                item["predicted_label"] = "Error"
                item["pass_fail"] = "N/A"
                item["probabilities"] = {}
                item["error"] = f"Inference failed: {exc}"
        results.append(item)
    return results
