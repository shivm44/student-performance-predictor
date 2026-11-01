"""
modules.py — Central unified facade for the Student Performance Predictor.

Architecture:
To prevent data leakage, concerns are segregated into distinct modules:
- data_pipeline.py: Zero-target-leakage feature synthesis, data loading, preparation, and input validation.
- ml_models.py: Scikit-learn pipelines with internal scaling, training strictly on training folds, leak-free artifact persistence, and inference.
- visualizations.py: Plotly charts, dark-theme styling, and evidence-based personalized feedback.

This facade re-exports all public functions, classes, and constants for seamless
backwards compatibility with app.py, train_model.py, and pytest test suites.
"""

# Re-export from data_pipeline
from data_pipeline import (
    generate_synthetic_features,
    load_dataset,
    get_dataset_stats,
    _encode_absence,
    prepare_features,
    prepare_single_input,
    validate_inputs,
)

# Re-export from ml_models
from ml_models import (
    _build_pipelines,
    train_all_models,
    save_results,
    load_results,
    predict,
    get_model_rankings,
    predict_top_models,
)

# Re-export from visualizations
from visualizations import (
    _COLORS,
    _CLASS_COLORS,
    _PAPER_BG,
    _PLOT_BG,
    _FONT_COLOR,
    _CLASS_MEANS,
    _base_layout,
    chart_class_distribution,
    chart_absence_distribution,
    chart_scatter_engagement,
    chart_model_accuracy,
    chart_metrics_comparison,
    chart_confusion_matrix,
    chart_feature_importance,
    chart_top3_prediction_comparison,
    chart_top3_probability_comparison,
    chart_top3_metrics_comparison,
    get_improvement_recommendations,
)

__all__ = [
    # data_pipeline
    "generate_synthetic_features",
    "load_dataset",
    "get_dataset_stats",
    "_encode_absence",
    "prepare_features",
    "prepare_single_input",
    "validate_inputs",
    # ml_models
    "_build_pipelines",
    "train_all_models",
    "save_results",
    "load_results",
    "predict",
    "get_model_rankings",
    "predict_top_models",
    # visualizations
    "_COLORS",
    "_CLASS_COLORS",
    "_PAPER_BG",
    "_PLOT_BG",
    "_FONT_COLOR",
    "_CLASS_MEANS",
    "_base_layout",
    "chart_class_distribution",
    "chart_absence_distribution",
    "chart_scatter_engagement",
    "chart_model_accuracy",
    "chart_metrics_comparison",
    "chart_confusion_matrix",
    "chart_feature_importance",
    "chart_top3_prediction_comparison",
    "chart_top3_probability_comparison",
    "chart_top3_metrics_comparison",
    "get_improvement_recommendations",
]
