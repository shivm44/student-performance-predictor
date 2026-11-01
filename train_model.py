"""
train_model.py — One-time script to train all models and save artifacts.

Run this before starting the application:
    python train_model.py

This script fits preprocessing and models on training data only, evaluates
on the held-out test set, and persists the results to models/pipelines.joblib.
"""

import logging
import sys

import config
import modules

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("=== Student Performance Predictor — Training Script ===")

    # 1. Load dataset
    try:
        df = modules.load_dataset()
    except FileNotFoundError as exc:
        logger.error("Dataset not found: %s", exc)
        logger.error("Expected at: %s", config.DATA_PATH)
        return 1

    stats = modules.get_dataset_stats(df)
    logger.info(
        "Dataset: %d records | %d features | %d classes",
        stats["total_records"],
        stats["num_features"],
        stats["num_classes"],
    )
    logger.info("Class distribution: %s", stats["class_distribution"])

    # 2. Train all models
    logger.info("Training %d models...", len(config.MODEL_KEYS))
    try:
        results = modules.train_all_models(df)
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        return 1

    # 3. Print summary
    logger.info("\n── Evaluation Summary (Test Set) ──")
    logger.info("Train size: %d | Test size: %d", results["train_size"], results["test_size"])
    for key in config.MODEL_KEYS:
        m = results["metrics"][key]
        logger.info(
            "%-22s  acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f  time=%.3fs",
            config.MODEL_DISPLAY_NAMES[key],
            m["accuracy"], m["precision"], m["recall"], m["f1"], m["train_time_s"],
        )

    # 4. Save
    try:
        modules.save_results(results)
        logger.info("Artifacts saved to: %s", config.MODELS_PATH)
    except Exception as exc:
        logger.exception("Save failed: %s", exc)
        return 1

    logger.info("=== Training complete. Run app.py to start the dashboard. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
