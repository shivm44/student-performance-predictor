"""
visualizations.py — Interactive Plotly chart builders, design theme,
and evidence-based student improvement recommendations.

All charts are returned as responsive Plotly Figure objects.
Zero HTML, CSS, or JS files are written here.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline

import config

# ── Visual theme tokens ───────────────────────────────────────────────────────
_COLORS = {
    "primary": "#1e3a5f",     # deep navy
    "accent": "#f59e0b",      # amber
    "low": "#ef4444",         # red
    "medium": "#3b82f6",      # blue
    "high": "#22c55e",        # green
    "neutral": "#94a3b8",     # slate
}
_CLASS_COLORS = {"L": _COLORS["low"], "M": _COLORS["medium"], "H": _COLORS["high"]}
_PAPER_BG = "#0f172a"        # dark navy background
_PLOT_BG  = "#1e293b"        # slate card background
_FONT_COLOR = "#e2e8f0"

_CLASS_MEANS = {
    "L": {"raisedhands": 17, "VisITedResources": 18, "Discussion": 31, "AttendancePercentage": 66.6, "StudyHours": 2.1},
    "M": {"raisedhands": 49, "VisITedResources": 61, "Discussion": 44, "AttendancePercentage": 85.6, "StudyHours": 4.0},
    "H": {"raisedhands": 70, "VisITedResources": 79, "Discussion": 54, "AttendancePercentage": 94.5, "StudyHours": 5.0},
}


def _base_layout(**kwargs) -> dict:
    """Standard layout dictionary for consistent dark theme and responsive margins."""
    base = dict(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_PLOT_BG,
        font=dict(color=_FONT_COLOR, family="Inter, system-ui, sans-serif", size=11),
        title_font=dict(size=13, color=_FONT_COLOR),
        margin=dict(l=40, r=20, t=45, b=35),
    )
    base.update(kwargs)
    return base


# ── Dashboard Visualizations ──────────────────────────────────────────────────

def chart_class_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart: count of students per performance class."""
    counts = df[config.TARGET_COLUMN].value_counts()
    classes = config.CLASS_ORDER
    values = [int(counts.get(c, 0)) for c in classes]
    labels = [config.CLASS_LABELS[c] for c in classes]
    colors = [_CLASS_COLORS[c] for c in classes]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition="outside",
        hovertemplate="%{x}: %{y} students<extra></extra>",
    ))
    fig.update_layout(
        title=f"Performance Class Distribution (Total: {sum(values):,})",
        xaxis_title="Performance Class",
        yaxis_title="Number of Students",
        yaxis=dict(range=[0, max(values) * 1.25], gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        **_base_layout(margin=dict(l=55, r=20, t=55, b=45)),
    )
    return fig


def chart_absence_distribution(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: absence category x class."""
    cats = ["Under-7", "Above-7"]
    fig = go.Figure()
    for cls in config.CLASS_ORDER:
        counts = [
            len(df[(df["StudentAbsenceDays"] == cat) & (df[config.TARGET_COLUMN] == cls)])
            for cat in cats
        ]
        fig.add_trace(go.Bar(
            name=config.CLASS_LABELS[cls],
            x=cats,
            y=counts,
            marker_color=_CLASS_COLORS[cls],
            hovertemplate=f"Class: {config.CLASS_LABELS[cls]}<br>Absence: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        title="Absence Days vs Performance Class",
        xaxis_title="Absence Category",
        yaxis_title="Number of Students",
        barmode="stack",
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        legend=dict(
            bgcolor=_PLOT_BG,
            bordercolor="#334155",
            borderwidth=1,
            font=dict(color=_FONT_COLOR),
        ),
        **_base_layout(margin=dict(l=55, r=20, t=55, b=45)),
    )
    return fig


def chart_scatter_engagement(df: pd.DataFrame) -> go.Figure:
    """Scatter: raisedhands vs VisITedResources, colored by Class."""
    fig = go.Figure()
    for cls in config.CLASS_ORDER:
        subset = df[df[config.TARGET_COLUMN] == cls]
        fig.add_trace(go.Scatter(
            x=subset["raisedhands"],
            y=subset["VisITedResources"],
            mode="markers",
            name=config.CLASS_LABELS[cls],
            marker=dict(color=_CLASS_COLORS[cls], size=7, opacity=0.75),
            hovertemplate=(
                f"Class: {config.CLASS_LABELS[cls]}<br>"
                "Raised Hands: %{x}<br>"
                "Visited Resources: %{y}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title="Class Participation vs Resource Exploration (by Class)",
        xaxis_title="Raised Hands",
        yaxis_title="Visited Resources",
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        **_base_layout(margin=dict(l=55, r=20, t=55, b=45)),
    )
    return fig


def chart_model_accuracy(metrics: dict) -> go.Figure:
    """Horizontal bar chart comparing model accuracies on held-out test data."""
    keys = config.MODEL_KEYS
    names = [config.MODEL_DISPLAY_NAMES[k] for k in keys]
    accs  = [metrics[k]["accuracy"] * 100 for k in keys]
    colors = [_COLORS["accent"] if a == max(accs) else _COLORS["primary"] for a in accs]

    fig = go.Figure(go.Bar(
        x=accs,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{a:.1f}%" for a in accs],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Model Accuracy Comparison (Held-out Test Partition)",
        xaxis=dict(title="Accuracy (%)", range=[0, 110], gridcolor="#334155"),
        yaxis=dict(title="", gridcolor="#334155"),
        **_base_layout(margin=dict(l=140, r=20, t=55, b=45)),
    )
    return fig


def chart_metrics_comparison(metrics: dict) -> go.Figure:
    """Grouped bar chart: Precision / Recall / F1 per model."""
    keys  = config.MODEL_KEYS
    names = [config.MODEL_DISPLAY_NAMES[k] for k in keys]

    fig = go.Figure()
    for metric_key, label, color in [
        ("precision", "Precision", "#3b82f6"),
        ("recall",    "Recall",    "#22c55e"),
        ("f1",        "F1-Score",  "#f59e0b"),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=names,
            y=[metrics[k][metric_key] for k in keys],
            marker_color=color,
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title="Precision · Recall · F1 by Model (Test Set, Weighted Avg)",
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.1], gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        **_base_layout(margin=dict(l=45, r=20, t=55, b=45)),
    )
    return fig


def chart_confusion_matrix(cm_data: list, class_names: list, model_name: str) -> go.Figure:
    """Heatmap confusion matrix for selected model."""
    labels = [config.CLASS_LABELS.get(c, c) for c in class_names]
    cm_arr = np.array(cm_data)

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        colorscale="Blues",
        showscale=True,
        text=cm_arr,
        texttemplate="%{text}",
        hovertemplate="True %{y}<br>Predicted %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="Actual", autorange="reversed"),
        **_base_layout(margin=dict(l=60, r=20, t=55, b=45)),
    )
    return fig


def chart_feature_importance(pipeline: Pipeline, model_key: str) -> go.Figure | None:
    """Feature importance bar chart for tree-based models."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None

    importances = clf.feature_importances_
    feature_names = [config.FEATURE_DISPLAY_NAMES.get(col, col) for col in config.FEATURE_COLUMNS]
    sorted_idx = np.argsort(importances)[::-1]

    fig = go.Figure(go.Bar(
        x=[feature_names[i] for i in sorted_idx],
        y=[importances[i] for i in sorted_idx],
        marker_color=_COLORS["accent"],
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Feature Importance — {config.MODEL_DISPLAY_NAMES[model_key]}",
        xaxis_title="Feature",
        yaxis_title="Importance",
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
        **_base_layout(margin=dict(l=55, r=20, t=55, b=45)),
    )
    return fig


# ── Top 3 Model Comparison Charts ─────────────────────────────────────────────

def chart_top3_prediction_comparison(top_predictions: list[dict]) -> go.Figure:
    """Graph A: Comparison of predicted classes and confidence across top-3 models."""
    fig = go.Figure()
    model_labels = [f"#{p['rank']} {p['display_name']}" for p in top_predictions]

    for p in top_predictions:
        cls = p["predicted_class"]
        lbl = p["predicted_label"]
        color = _CLASS_COLORS.get(cls, _COLORS["accent"])
        m_label = f"#{p['rank']} {p['display_name']}"
        conf = p.get("probabilities", {}).get(cls, 1.0) * 100 if p.get("probabilities") else 100.0

        fig.add_trace(go.Bar(
            name=f"{m_label}: {lbl}",
            x=[m_label],
            y=[conf],
            marker_color=color,
            text=[f"{lbl} ({p['pass_fail']})<br>{conf:.1f}%" if p.get("probabilities") else f"{lbl} ({p['pass_fail']})"],
            textposition="outside",
            hovertemplate=(
                f"<b>{p['display_name']}</b><br>"
                f"Rank: #{p['rank']}<br>"
                f"Predicted: <b>{lbl}</b><br>"
                f"Status: <b>{p['pass_fail']}</b><br>"
                f"Confidence: {conf:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title="Prediction Comparison — Top 3 Models",
        xaxis=dict(
            title="Model (Ranked by Test F1-Score)",
            type="category",
            categoryorder="array",
            categoryarray=model_labels,
            gridcolor="#334155",
        ),
        yaxis=dict(
            title="Prediction Confidence (%)",
            range=[0, 120],
            gridcolor="#334155",
        ),
        **_base_layout(margin=dict(l=40, r=20, t=60, b=40)),
    )
    return fig


def chart_top3_probability_comparison(top_predictions: list[dict]) -> tuple[go.Figure | None, str]:
    """Graph B: Grouped bar chart comparing predicted class probabilities."""
    unsupported = [
        p["display_name"]
        for p in top_predictions
        if not p.get("has_proba") or not p.get("probabilities")
    ]
    if unsupported:
        reason = (
            f"Class probability comparison is omitted because {', '.join(unsupported)} "
            f"does not support probability estimation (no predict_proba implementation)."
        )
        return None, reason

    fig = go.Figure()
    class_order = ["L", "M", "H"]
    class_labels = [config.CLASS_LABELS[c] for c in class_order]
    model_colors = ["#f59e0b", "#38bdf8", "#a855f7"]

    for i, p in enumerate(top_predictions):
        probs = p.get("probabilities", {})
        y_vals = [probs.get(c, 0.0) * 100 for c in class_order]
        color = model_colors[i % len(model_colors)]
        rank_label = f"#{p['rank']} {p['display_name']}"

        fig.add_trace(go.Bar(
            name=rank_label,
            x=class_labels,
            y=y_vals,
            marker_color=color,
            text=[f"{v:.1f}%" for v in y_vals],
            textposition="outside",
            hovertemplate=f"<b>{p['display_name']}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Class Probability Comparison — Top 3 Models",
        xaxis=dict(title="Target Class", categoryorder="array", categoryarray=class_labels, gridcolor="#334155"),
        yaxis=dict(title="Estimated Probability (%)", range=[0, 115], gridcolor="#334155"),
        barmode="group",
        legend=dict(
            font=dict(color=_FONT_COLOR),
            bgcolor=_PLOT_BG,
            bordercolor="#334155",
            borderwidth=1,
        ),
        **_base_layout(margin=dict(l=40, r=20, t=60, b=40)),
    )
    return fig, ""


def chart_top3_metrics_comparison(top_models: list[dict]) -> go.Figure:
    """Graph C: Grouped bar chart comparing held-out test metrics."""
    fig = go.Figure()
    model_names = [f"#{m['rank']} {m['display_name']}" for m in top_models]

    metrics_meta = [
        ("accuracy",  "Accuracy",  "#38bdf8"),
        ("precision", "Precision", "#818cf8"),
        ("recall",    "Recall",    "#34d399"),
        ("f1",        "F1-Score",  "#f59e0b"),
    ]

    for key, label, color in metrics_meta:
        vals = [m[key] * 100 for m in top_models]
        fig.add_trace(go.Bar(
            name=label,
            x=model_names,
            y=vals,
            marker_color=color,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            hovertemplate=f"<b>{label}</b>: %{{y:.2f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Model Performance Comparison (Held-out Test Partition)",
        xaxis=dict(title="Top 3 Models", gridcolor="#334155"),
        yaxis=dict(title="Score (%)", range=[0, 115], gridcolor="#334155"),
        barmode="group",
        legend=dict(
            font=dict(color=_FONT_COLOR),
            bgcolor=_PLOT_BG,
            bordercolor="#334155",
            borderwidth=1,
        ),
        **_base_layout(margin=dict(l=40, r=20, t=60, b=40)),
    )
    return fig


# ── Actionable Recommendations & Personalization ──────────────────────────────

def get_improvement_recommendations(
    raisedhands: int,
    visited_resources: int,
    discussion: int,
    absence_days: str,
    predicted_class: str,
    attendance_pct: float | None = None,
    study_hours: float | None = None,
) -> list[dict]:
    """Generate personalized recommendations based on actual class averages."""
    recs = []
    next_cls = {"L": "M", "M": "H", "H": None}.get(predicted_class)

    if absence_days == "Above-7":
        recs.append({
            "priority": "high",
            "icon": "🏫",
            "title": "Reduce Unexcused Absences",
            "detail": (
                "You have 7 or more absence days. Across the dataset, 91% of Low performers "
                "had Above-7 absences, compared to only 3% of High performers. Reducing absences "
                "is the single highest-impact academic intervention."
            ),
        })

    if attendance_pct is not None and attendance_pct < 85.0:
        target_att = 94.5 if next_cls == "H" else 86.0
        recs.append({
            "priority": "high" if attendance_pct < 75.0 else "medium",
            "icon": "📅",
            "title": "Improve Overall Attendance Rate",
            "detail": (
                f"Your attendance rate is {attendance_pct:.1f}%. High performers average "
                f"{target_att:.1f}% attendance. Aim for at least 90%+ attendance consistency."
            ),
        })

    if study_hours is not None:
        target_sh = 5.0 if next_cls == "H" else 4.0
        if study_hours < target_sh - 0.5:
            recs.append({
                "priority": "high" if study_hours < 2.5 else "medium",
                "icon": "⏱️",
                "title": "Dedicate More Daily Study Hours",
                "detail": (
                    f"Your daily study time is {study_hours:.1f} hrs/day. "
                    f"{'Medium' if next_cls == 'M' else 'High'}-performing students average "
                    f"{target_sh:.1f} hours/day of focused independent study."
                ),
            })

    target_rh = _CLASS_MEANS.get(next_cls, _CLASS_MEANS["H"])["raisedhands"] if next_cls else 70
    gap_rh = target_rh - raisedhands
    if gap_rh > 10:
        recs.append({
            "priority": "high" if gap_rh > 30 else "medium",
            "icon": "✋",
            "title": "Participate More in Class",
            "detail": (
                f"Your raised-hands score is {raisedhands}/100. Target average is {target_rh}. "
                "Asking questions and engaging actively in class improves lecture retention."
            ),
        })

    target_vr = _CLASS_MEANS.get(next_cls, _CLASS_MEANS["H"])["VisITedResources"] if next_cls else 79
    gap_vr = target_vr - visited_resources
    if gap_vr > 10:
        recs.append({
            "priority": "high" if gap_vr > 30 else "medium",
            "icon": "📚",
            "title": "Explore Course Resources Regularly",
            "detail": (
                f"Your resource score is {visited_resources}/100; target average is {target_vr}. "
                "Frequently reviewing lecture notes and reading materials directly correlates with performance."
            ),
        })

    target_dis = _CLASS_MEANS.get(next_cls, _CLASS_MEANS["H"])["Discussion"] if next_cls else 54
    gap_dis = target_dis - discussion
    if gap_dis > 8:
        recs.append({
            "priority": "medium",
            "icon": "💬",
            "title": "Engage in Group Discussions",
            "detail": (
                f"Your discussion score is {discussion}/100; target average is {target_dis}. "
                "Participating in study groups and forums reinforces complex concepts."
            ),
        })

    if predicted_class == "H":
        recs.append({
            "priority": "low",
            "icon": "🏆",
            "title": "Maintain Your Academic Habits",
            "detail": (
                "Your engagement metrics and attendance are at or above High-performer averages. "
                "Continue your current routine to maintain peak performance."
            ),
        })

    return recs
