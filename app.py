"""
app.py — Student Performance Prediction Dashboard

Built entirely in Python using Dash (Plotly Dash) which integrates with Flask.
No HTML, CSS, or JavaScript files are written or read by this application.
All layout, styling, and interactivity is expressed in Python.

Design direction: Precision Utility — dark slate / deep-navy dominant,
amber/orange accents, rounded panels, responsive two-column layout.
Font size scaled to 80% for compact, sleek presentation.
"""

import logging
import sys

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask import Flask

import config
import modules

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask server (security headers via after_request) ─────────────────────────
server = Flask(__name__)
server.secret_key = config.SECRET_KEY


@server.after_request
def add_security_headers(response):
    """Apply defensive HTTP response headers."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


# ── Application state — loaded once at startup ────────────────────────────────
_state: dict = {}


def _bootstrap_state() -> bool:
    """Load dataset and models into _state. Returns True on success."""
    try:
        df = modules.load_dataset()
        _state["df"] = df
        _state["stats"] = modules.get_dataset_stats(df)
    except FileNotFoundError as exc:
        logger.error("Dataset missing: %s", exc)
        return False
    except Exception as exc:
        logger.exception("Failed to load dataset: %s", exc)
        return False

    results = modules.load_results()
    if results is None:
        logger.warning("No trained models found. Running training now...")
        try:
            results = modules.train_all_models(df)
            modules.save_results(results)
        except Exception as exc:
            logger.exception("Auto-training failed: %s", exc)
            return False

    _state["pipelines"]   = results["pipelines"]
    _state["metrics"]     = results["metrics"]
    _state["class_names"] = results["class_names"]
    _state["train_size"]  = results["train_size"]
    _state["test_size"]   = results["test_size"]
    _state["rankings"]    = modules.get_model_rankings(results["metrics"])
    _state["top_3_keys"]  = [m["key"] for m in _state["rankings"][:3]]

    logger.info(
        "Application state ready. %d models loaded. Top 3 models: %s",
        len(_state["pipelines"]),
        [m["display_name"] for m in _state["rankings"][:3]],
    )
    return True


# ── Design tokens (pure Python, compact 80% proportional scaling) ──────────────
NAVY    = "#0f172a"
PANEL   = "#1e293b"
BORDER  = "#334155"
AMBER   = "#f59e0b"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
SUCCESS = "#22c55e"
DANGER  = "#ef4444"
INFO    = "#3b82f6"
PURPLE  = "#a855f7"

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"


def _card(children, style_extra=None):
    base = {
        "background": PANEL,
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "18px 22px",
        "marginBottom": "16px",
        "boxSizing": "border-box",
    }
    if style_extra:
        base.update(style_extra)
    return html.Div(children, style=base)


def _stat_card(label: str, value: str, color: str = AMBER):
    return html.Div(
        [
            html.Div(value, style={
                "fontSize": "2.2rem",
                "fontWeight": "800",
                "color": color,
                "lineHeight": "1",
            }),
            html.Div(label, style={
                "fontSize": "0.78rem",
                "color": MUTED,
                "marginTop": "6px",
                "textTransform": "uppercase",
                "letterSpacing": "0.08em",
                "fontWeight": "600",
            }),
        ],
        style={
            "background": PANEL,
            "border": f"1px solid {BORDER}",
            "borderRadius": "12px",
            "padding": "18px 20px",
            "flex": "1 1 180px",
            "minWidth": "160px",
            "textAlign": "center",
            "boxSizing": "border-box",
        },
    )


def _section_title(text: str, sub: str = ""):
    return html.Div([
        html.H2(text, style={
            "color": TEXT,
            "fontSize": "1.35rem",
            "fontWeight": "700",
            "margin": "0 0 4px 0",
        }),
        html.P(sub, style={"color": MUTED, "fontSize": "0.82rem", "margin": "0 0 14px 0"}) if sub else None,
    ])


def _section_header(title: str):
    """Amber-bordered section header."""
    return html.H3(title, style={
        "color": TEXT,
        "fontSize": "0.95rem",
        "fontWeight": "700",
        "borderLeft": f"3px solid {AMBER}",
        "paddingLeft": "10px",
        "margin": "0 0 14px 0",
    })


def _input_row(label: str, control):
    return html.Div([
        html.Label(label, style={
            "color": MUTED,
            "fontSize": "0.78rem",
            "fontWeight": "600",
            "display": "block",
            "marginBottom": "5px",
            "textTransform": "uppercase",
            "letterSpacing": "0.06em",
        }),
        control,
    ], style={"marginBottom": "16px"})


# ── Input style helper ────────────────────────────────────────────────────────
_INPUT_STYLE = {
    "width": "100%",
    "height": "38px",
    "background": NAVY,
    "border": f"1px solid {BORDER}",
    "borderRadius": "6px",
    "color": TEXT,
    "padding": "6px 12px",
    "fontSize": "0.88rem",
    "boxSizing": "border-box",
    "outline": "none",
}
_DROPDOWN_STYLE = {"width": "100%"}


# ── Tab content builders ──────────────────────────────────────────────────────

def _tab_home():
    stats = _state.get("stats", {})
    total_rec = stats.get("total_records", 480)
    dataset_name = stats.get("dataset_name", config.DATA_PATH.name)
    rankings = _state.get("rankings", [])

    return html.Div([
        # Hero banner
        html.Div([
            html.Div("🎓", style={"fontSize": "2.4rem", "marginBottom": "6px"}),
            html.H1("Student Performance Predictor", style={
                "color": TEXT,
                "fontSize": "1.9rem",
                "fontWeight": "800",
                "margin": "0 0 6px 0",
            }),
            html.P(
                "A machine learning dashboard that predicts student academic performance "
                "(Low / Medium / High) using engagement behaviour data from the "
                "UCI Student Academics Performance dataset.",
                style={"color": MUTED, "fontSize": "0.88rem", "maxWidth": "620px", "margin": "0 auto 20px", "lineHeight": "1.5"},
            ),
        ], style={"textAlign": "center", "padding": "16px 20px 10px"}),

        # Stat cards (matching reference screenshot: 480 records, 4 features, 3 classes, 5 models)
        html.Div([
            _stat_card("Dataset Records", f"{total_rec:,}"),
            _stat_card("Input Features", "4", INFO),
            _stat_card("Performance Classes", str(stats.get("num_classes", 3)), SUCCESS),
            _stat_card("Models Evaluated", str(stats.get("num_models", len(config.MODEL_KEYS))), AMBER),
        ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "16px"}),

        # Data Integrity Callout
        html.Div([
            html.Span("ℹ Data Integrity: ", style={"fontWeight": "700", "color": AMBER}),
            html.Span(
                f"Loaded {total_rec:,} total verified rows from {dataset_name}. "
                f"All {stats.get('valid_records', total_rec):,} rows are verified and complete with zero missing values.",
                style={"color": MUTED, "fontSize": "0.82rem"},
            ),
        ], style={
            "background": PANEL,
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "padding": "10px 16px",
            "marginBottom": "16px",
        }),

        # Dataset overview charts (side-by-side responsive grid)
        _card([
            _section_title(
                "Dataset Overview",
                f"UCI Student Academics Performance ({dataset_name}) — {total_rec:,} dynamically loaded rows",
            ),
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=modules.chart_class_distribution(_state["df"]),
                        config={"displayModeBar": False, "responsive": True},
                        style={"width": "100%", "height": "300px"},
                        responsive=True,
                    ),
                ], style={"minWidth": "0", "flex": "1 1 380px"}),
                html.Div([
                    dcc.Graph(
                        figure=modules.chart_absence_distribution(_state["df"]),
                        config={"displayModeBar": False, "responsive": True},
                        style={"width": "100%", "height": "300px"},
                        responsive=True,
                    ),
                ], style={"minWidth": "0", "flex": "1 1 380px"}),
            ], style={
                "display": "flex",
                "gap": "16px",
                "flexWrap": "wrap",
                "width": "100%",
                "boxSizing": "border-box",
            }),
        ]),

        # Engagement relationship scatter
        _card([
            _section_title(
                "Engagement Feature Relationship",
                "Student participation (Raised Hands) vs resource interaction (Visited Resources) across performance classes",
            ),
            html.Div([
                dcc.Graph(
                    figure=modules.chart_scatter_engagement(_state["df"]),
                    config={"displayModeBar": False, "responsive": True},
                    style={"width": "100%", "height": "320px"},
                    responsive=True,
                ),
            ], style={"minWidth": "0", "width": "100%"}),
        ]),

        # Feature descriptions card
        _card([
            _section_title(
                "Model Features Used",
                "These 4 primary engagement features from the UCI dataset drive performance prediction:",
            ),
            html.Div([
                html.Div([
                    html.Strong("Raised Hands", style={"color": AMBER}),
                    html.Span(" — Number of times the student raised their hand during lectures (0–100)", style={"color": MUTED, "marginLeft": "6px"}),
                ], style={"padding": "8px 0", "borderBottom": f"1px solid {BORDER}", "fontSize": "0.85rem"}),
                html.Div([
                    html.Strong("Visited Resources", style={"color": AMBER}),
                    html.Span(" — Number of times the student interacted with online course resources (0–100)", style={"color": MUTED, "marginLeft": "6px"}),
                ], style={"padding": "8px 0", "borderBottom": f"1px solid {BORDER}", "fontSize": "0.85rem"}),
                html.Div([
                    html.Strong("Discussion", style={"color": AMBER}),
                    html.Span(" — Number of times the student participated in group or forum discussions (0–100)", style={"color": MUTED, "marginLeft": "6px"}),
                ], style={"padding": "8px 0", "borderBottom": f"1px solid {BORDER}", "fontSize": "0.85rem"}),
                html.Div([
                    html.Strong("Absence Days", style={"color": AMBER}),
                    html.Span(" — Whether student absences were under 7 days or 7 or more days (Under-7 / Above-7)", style={"color": MUTED, "marginLeft": "6px"}),
                ], style={"padding": "8px 0", "fontSize": "0.85rem"}),
            ]),
        ]),

        # Quick Model Performance Summary on Dashboard
        _card([
            _section_title("Evaluated ML Models Summary", "Genuine test performance across all 5 models (30% held-out test partition)."),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Rank", style={"padding": "8px 12px", "textAlign": "left", "color": AMBER, "fontSize": "0.82rem"}),
                    html.Th("Model", style={"padding": "8px 12px", "textAlign": "left", "color": AMBER, "fontSize": "0.82rem"}),
                    html.Th("Test Accuracy", style={"padding": "8px 12px", "color": AMBER, "fontSize": "0.82rem", "textAlign": "center"}),
                    html.Th("Test F1-Score", style={"padding": "8px 12px", "color": AMBER, "fontSize": "0.82rem", "textAlign": "center"}),
                    html.Th("Status", style={"padding": "8px 12px", "color": AMBER, "fontSize": "0.82rem", "textAlign": "center"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            f"#{m['rank']}" if m["rank"] > 1 else "🏆 #1",
                            style={"padding": "8px 12px", "fontWeight": "700", "color": AMBER if m["rank"] == 1 else TEXT, "fontSize": "0.85rem"},
                        ),
                        html.Td(m["display_name"], style={"padding": "8px 12px", "fontWeight": "600", "color": TEXT, "fontSize": "0.85rem"}),
                        html.Td(f"{m['accuracy']:.1%}", style={"padding": "8px 12px", "textAlign": "center", "color": TEXT, "fontSize": "0.85rem"}),
                        html.Td(f"{m['f1']:.3f}", style={"padding": "8px 12px", "textAlign": "center", "color": TEXT, "fontSize": "0.85rem"}),
                        html.Td(
                            html.Span(
                                "★ Top 3 Performer" if m["is_top_3"] else "Evaluated",
                                style={
                                    "background": "rgba(245, 158, 11, 0.15)" if m["is_top_3"] else "rgba(148, 163, 184, 0.12)",
                                    "color": AMBER if m["is_top_3"] else MUTED,
                                    "padding": "3px 8px",
                                    "borderRadius": "10px",
                                    "fontSize": "0.75rem",
                                    "fontWeight": "600",
                                },
                            ),
                            style={"padding": "8px 12px", "textAlign": "center"},
                        ),
                    ], style={
                        "borderTop": f"1px solid {BORDER}",
                        "background": "rgba(245, 158, 11, 0.05)" if m["rank"] == 1 else "transparent",
                    })
                    for m in rankings
                ]),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
        ]),
    ])


def _tab_predict():
    """
    Two-column side-by-side Predict layout matching reference image:
    Left: Student Prediction form with dataset features & model selector.
    Right: Live Prediction Result & Top 3 Model Comparison.
    """
    model_options = [
        {"label": config.MODEL_DISPLAY_NAMES[k], "value": k}
        for k in config.MODEL_KEYS
    ]

    # Left Column: Form Card
    form_card = _card([
        _section_title("Student Prediction", "Enter student engagement data to predict performance."),

        # Raised Hands
        _input_row("Raised Hands (0–100)",
            dcc.Slider(
                id="inp-hands",
                min=0, max=100, step=1, value=43,
                marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ),

        # Visited Resources
        _input_row("Visited Resources (0–100)",
            dcc.Slider(
                id="inp-resources",
                min=0, max=100, step=1, value=76,
                marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ),

        # Discussion
        _input_row("Discussion (0–100)",
            dcc.Slider(
                id="inp-discussion",
                min=0, max=100, step=1, value=43,
                marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ),

        # Absence Days
        _input_row("Absence Days",
            dcc.Dropdown(
                id="inp-absence",
                options=[
                    {"label": "Under 7 days absent", "value": "Under-7"},
                    {"label": "7 or more days absent", "value": "Above-7"},
                ],
                value="Under-7",
                clearable=False,
                style=_DROPDOWN_STYLE,
                className="dark-dropdown",
            ),
        ),

        # Select Model
        _input_row("Select Model",
            dcc.Dropdown(
                id="inp-model",
                options=model_options,
                value="random_forest",
                clearable=False,
                style=_DROPDOWN_STYLE,
                className="dark-dropdown",
            ),
        ),

        # Supplementary Context Fields (Clean expandable details)
        html.Details([
            html.Summary("Additional Academic Factors (Optional)", style={
                "color": MUTED,
                "fontSize": "0.78rem",
                "fontWeight": "600",
                "cursor": "pointer",
                "marginBottom": "12px",
                "outline": "none",
            }),
            html.Div([
                _input_row("Attendance Percentage (0–100%)",
                    dcc.Slider(
                        id="inp-attendance",
                        min=0, max=100, step=1, value=85,
                        marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ),
                _input_row("Study Hours per Day (0–24h)",
                    dcc.Input(
                        id="inp-study-hours",
                        type="number", min=0, max=24, step=0.5,
                        value=4.0, placeholder="e.g. 4.0",
                        style=_INPUT_STYLE,
                    ),
                ),
                _input_row("Previous Exam Score (0–100)",
                    dcc.Input(
                        id="inp-prev-score",
                        type="number", min=0, max=100,
                        placeholder="e.g. 72 (optional)",
                        style=_INPUT_STYLE,
                    ),
                ),
                _input_row("Assignment Completion (0–100%)",
                    dcc.Slider(
                        id="inp-assignment",
                        min=0, max=100, step=1, value=75,
                        marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ),
            ], style={"background": NAVY, "padding": "12px 14px", "borderRadius": "8px", "border": f"1px solid {BORDER}", "marginBottom": "14px"}),
        ], style={"marginBottom": "16px"}),

        # Error display
        html.Div(id="predict-error", style={"color": DANGER, "fontSize": "0.82rem", "marginBottom": "10px"}),

        # Buttons
        html.Div([
            html.Button("Predict", id="btn-predict", n_clicks=0, style={
                "background": AMBER,
                "color": NAVY,
                "border": "none",
                "borderRadius": "6px",
                "padding": "8px 28px",
                "fontSize": "0.9rem",
                "fontWeight": "700",
                "cursor": "pointer",
                "marginRight": "10px",
                "height": "38px",
                "boxShadow": "0 2px 6px rgba(245, 158, 11, 0.25)",
                "transition": "all 0.15s ease",
            }),
            html.Button("Reset", id="btn-reset", n_clicks=0, style={
                "background": "transparent",
                "color": MUTED,
                "border": f"1px solid {BORDER}",
                "borderRadius": "6px",
                "padding": "8px 20px",
                "fontSize": "0.9rem",
                "fontWeight": "600",
                "cursor": "pointer",
                "height": "38px",
                "transition": "all 0.15s ease",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style_extra={"flex": "1 1 380px", "maxWidth": "460px", "minWidth": "320px", "marginBottom": "0"})

    # Right Column: Result Panel Container
    result_container = html.Div(
        id="predict-result-panel",
        children=_build_default_result_panel(),
        style={"flex": "2 1 520px", "minWidth": "340px"},
    )

    # Side-by-side two column container
    return html.Div(
        [form_card, result_container],
        style={
            "display": "flex",
            "gap": "20px",
            "alignItems": "flex-start",
            "flexWrap": "wrap",
        },
    )


def _tab_models():
    metrics = _state.get("metrics", {})
    rankings = _state.get("rankings", [])
    model_options = [
        {"label": config.MODEL_DISPLAY_NAMES[k], "value": k}
        for k in config.MODEL_KEYS
    ]

    return html.Div([
        # Model Rankings & Selection Card
        _card([
            _section_title("Model Rankings & Top 3 Selection", "Dynamically ranked based on held-out test evaluation."),
            html.P(
                config.RANKING_CRITERIA_DOC,
                style={"color": MUTED, "fontSize": "0.82rem", "marginBottom": "14px"},
            ),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Rank", style={"padding": "8px 12px", "textAlign": "left", "color": AMBER, "fontSize": "0.82rem"}),
                    html.Th("Model", style={"padding": "8px 12px", "textAlign": "left", "color": AMBER, "fontSize": "0.82rem"}),
                    html.Th("Test F1-Score", style={"padding": "8px 12px", "color": AMBER, "fontSize": "0.82rem", "textAlign": "center"}),
                    html.Th("Test Accuracy", style={"padding": "8px 12px", "color": AMBER, "fontSize": "0.82rem", "textAlign": "center"}),
                    html.Th("Precision", style={"padding": "8px 12px", "color": AMBER, "fontSize": "0.82rem", "textAlign": "center"}),
                    html.Th("Recall", style={"padding": "8px 12px", "color": AMBER, "fontSize": "0.82rem", "textAlign": "center"}),
                    html.Th("Status", style={"padding": "8px 12px", "color": AMBER, "fontSize": "0.82rem", "textAlign": "center"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            f"#{m['rank']}" if m["rank"] > 1 else "🏆 #1",
                            style={"padding": "8px 12px", "fontWeight": "700", "color": AMBER if m["rank"] == 1 else TEXT, "fontSize": "0.85rem"},
                        ),
                        html.Td(m["display_name"], style={"padding": "8px 12px", "fontWeight": "600", "color": TEXT, "fontSize": "0.85rem"}),
                        html.Td(f"{m['f1']:.3f}", style={"padding": "8px 12px", "textAlign": "center", "color": TEXT, "fontSize": "0.85rem"}),
                        html.Td(f"{m['accuracy']:.1%}", style={"padding": "8px 12px", "textAlign": "center", "color": TEXT, "fontSize": "0.85rem"}),
                        html.Td(f"{m['precision']:.3f}", style={"padding": "8px 12px", "textAlign": "center", "color": MUTED, "fontSize": "0.85rem"}),
                        html.Td(f"{m['recall']:.3f}", style={"padding": "8px 12px", "textAlign": "center", "color": MUTED, "fontSize": "0.85rem"}),
                        html.Td(
                            html.Span(
                                "★ Top 3 Performer" if m["is_top_3"] else "Evaluated Model",
                                style={
                                    "background": "rgba(245, 158, 11, 0.15)" if m["is_top_3"] else "rgba(148, 163, 184, 0.15)",
                                    "color": AMBER if m["is_top_3"] else MUTED,
                                    "padding": "3px 8px",
                                    "borderRadius": "10px",
                                    "fontSize": "0.75rem",
                                    "fontWeight": "600",
                                },
                            ),
                            style={"padding": "8px 12px", "textAlign": "center"},
                        ),
                    ], style={
                        "borderTop": f"1px solid {BORDER}",
                        "background": "rgba(245, 158, 11, 0.05)" if m["rank"] == 1 else "transparent",
                    })
                    for m in rankings
                ]),
            ], style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "10px"}),
        ]),

        # Model accuracy comparison
        _card([
            _section_title("Model Accuracy Comparison", "All metrics computed on the held-out test set (30% split)."),
            html.Div([
                dcc.Graph(
                    figure=modules.chart_model_accuracy(metrics),
                    config={"displayModeBar": False, "responsive": True},
                    style={"width": "100%", "height": "320px"},
                    responsive=True,
                ),
            ], style={"minWidth": "0", "width": "100%"}),
        ]),

        # Precision / Recall / F1
        _card([
            html.Div([
                dcc.Graph(
                    figure=modules.chart_metrics_comparison(metrics),
                    config={"displayModeBar": False, "responsive": True},
                    style={"width": "100%", "height": "320px"},
                    responsive=True,
                ),
            ], style={"minWidth": "0", "width": "100%"}),
        ]),

        # Confusion Matrix & Feature Importance
        _card([
            _section_title("Confusion Matrix & Feature Importance"),
            html.Div([
                html.Label("Select model:", style={"color": MUTED, "fontSize": "0.78rem", "marginBottom": "5px", "display": "block", "textTransform": "uppercase"}),
                dcc.Dropdown(
                    id="cm-model-selector",
                    options=model_options,
                    value="random_forest",
                    clearable=False,
                    style={"width": "260px"},
                    className="dark-dropdown",
                ),
            ], style={"marginBottom": "14px"}),
            html.Div([
                html.Div([
                    dcc.Graph(id="cm-graph", config={"displayModeBar": False, "responsive": True}, style={"width": "100%", "height": "320px"}, responsive=True),
                ], style={"flex": "1 1 300px", "minWidth": "0"}),
                html.Div([
                    dcc.Graph(id="fi-graph", config={"displayModeBar": False, "responsive": True}, style={"width": "100%", "height": "320px"}, responsive=True),
                ], style={"flex": "1 1 300px", "minWidth": "0"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        ]),
    ])


# ── Dash app creation ─────────────────────────────────────────────────────────

def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.SLATE],
        title="Student Performance Predictor",
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    INLINE_CSS = """
        html {
            font-size: 80% !important; /* Scaled to 80% per specification */
            overflow-y: auto !important;
            overflow-x: hidden !important;
        }
        body {
            background: #0f172a !important;
            font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: #e2e8f0 !important;
            margin: 0 !important;
            padding: 0 !important;
            font-size: 1rem !important; /* Exactly 80% of browser standard (12.8px) */
            overflow-y: auto !important;
            overflow-x: hidden !important;
            min-height: 100vh !important;
        }
        * { box-sizing: border-box !important; }

        /* Sleek Dropdown Styling */
        .dark-dropdown .Select-control {
            background: #0f172a !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
            color: #e2e8f0 !important;
            height: 38px !important;
            font-size: 0.85rem !important;
        }
        .dark-dropdown .Select-placeholder, .dark-dropdown .Select--single > .Select-control .Select-value {
            line-height: 36px !important;
            padding-left: 10px !important;
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
        }
        .dark-dropdown .Select-menu-outer {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
            z-index: 1000 !important;
        }
        .dark-dropdown .Select-option {
            color: #e2e8f0 !important;
            background: #1e293b !important;
            padding: 8px 12px !important;
            font-size: 0.85rem !important;
        }
        .dark-dropdown .Select-option.is-focused {
            background: #334155 !important;
            color: #f59e0b !important;
        }
        .dark-dropdown .Select-arrow { border-top-color: #94a3b8 !important; }

        input[type=number]::-webkit-outer-spin-button, input[type=number]::-webkit-inner-spin-button {
            -webkit-appearance: none; margin: 0;
        }
        input[type=number] { -moz-appearance: textfield; }

        /* Custom Sleek Scrollbar */
        ::-webkit-scrollbar { width: 7px; height: 7px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #f59e0b; }

        .tab-selected { border-bottom: 2px solid #f59e0b !important; color: #f59e0b !important; }
        .js-plotly-plot, .plot-container { width: 100% !important; max-width: 100% !important; }

        /* Slider styling */
        .rc-slider-track { background-color: #f59e0b !important; height: 5px !important; }
        .rc-slider-rail { height: 5px !important; background-color: #334155 !important; }
        .rc-slider-handle { border-color: #f59e0b !important; background: #f59e0b !important; width: 14px !important; height: 14px !important; margin-top: -5px !important; }
        .rc-slider-tooltip-inner { background-color: #1e293b !important; border: 1px solid #334155 !important; color: #e2e8f0 !important; font-size: 0.8rem !important; font-weight: 600 !important; padding: 2px 6px !important; }
        .rc-slider-mark-text { color: #64748b !important; font-size: 0.75rem !important; }
    """

    app.index_string = f"""<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>{INLINE_CSS}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
  </body>
</html>"""

    NAV_STYLE = {
        "background": PANEL,
        "borderBottom": f"1px solid {BORDER}",
        "padding": "0 24px",
        "display": "flex",
        "alignItems": "center",
        "position": "sticky",
        "top": "0",
        "zIndex": "100",
        "height": "50px",
    }

    TAB_STYLE = {
        "color": MUTED,
        "background": "transparent",
        "border": "none",
        "borderBottom": "2px solid transparent",
        "padding": "14px 18px",
        "fontSize": "0.88rem",
        "fontWeight": "600",
        "cursor": "pointer",
        "transition": "color 0.15s, border-color 0.15s",
    }
    TAB_SELECTED_STYLE = {**TAB_STYLE, "color": AMBER, "borderBottom": f"2px solid {AMBER}"}

    app.layout = html.Div([
        # Navigation
        html.Div([
            html.Div("🎓", style={"fontSize": "1.3rem", "marginRight": "8px"}),
            html.Span("SPP", style={"color": AMBER, "fontWeight": "800", "fontSize": "1.15rem", "marginRight": "28px"}),
            dcc.Tabs(
                id="main-tabs",
                value="tab-home",
                children=[
                    dcc.Tab(label="Dashboard", value="tab-home",
                            style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="Predict", value="tab-predict",
                            style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                    dcc.Tab(label="Model Comparison", value="tab-models",
                            style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                ],
                style={"border": "none", "background": "transparent"},
                colors={"border": "transparent", "primary": AMBER, "background": "transparent"},
            ),
        ], style=NAV_STYLE),

        # Content container
        html.Div(
            id="tab-content",
            style={
                "maxWidth": "1400px",
                "margin": "0 auto",
                "padding": "20px 24px",
                "boxSizing": "border-box",
            },
        ),
    ], style={"background": NAVY, "minHeight": "100vh"})

    # ── Callbacks ─────────────────────────────────────────────────────────────

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab):
        if tab == "tab-home":
            return _tab_home()
        elif tab == "tab-predict":
            return _tab_predict()
        elif tab == "tab-models":
            return _tab_models()
        return html.Div("Page not found.", style={"color": DANGER})

    @app.callback(
        Output("predict-result-panel", "children"),
        Output("predict-error", "children"),
        Output("inp-hands", "value"),
        Output("inp-resources", "value"),
        Output("inp-discussion", "value"),
        Output("inp-absence", "value"),
        Output("inp-attendance", "value"),
        Output("inp-study-hours", "value"),
        Output("inp-prev-score", "value"),
        Output("inp-assignment", "value"),
        Input("btn-predict", "n_clicks"),
        Input("btn-reset", "n_clicks"),
        State("inp-hands", "value"),
        State("inp-resources", "value"),
        State("inp-discussion", "value"),
        State("inp-absence", "value"),
        State("inp-model", "value"),
        State("inp-attendance", "value"),
        State("inp-study-hours", "value"),
        State("inp-prev-score", "value"),
        State("inp-assignment", "value"),
        prevent_initial_call=True,
    )
    def handle_predict_or_reset(n_pred, n_reset, hands, resources, discussion, absence, model_key,
                                 attendance, study_hours, prev_score, assignment):
        ctx = callback_context
        if not ctx.triggered:
            return (no_update,) * 10

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if triggered_id == "btn-reset":
            return (
                _build_default_result_panel(),
                "",
                43, 76, 43, "Under-7", 85, 4.0, None, 75,
            )

        # --- Predict ---
        # Ensure absence default
        if absence is None:
            absence = "Under-7"
        if model_key is None:
            model_key = "random_forest"

        is_valid, err_msg, cleaned = modules.validate_inputs(
            hands, resources, discussion, absence, attendance, study_hours
        )
        if not is_valid:
            return (no_update, err_msg) + (no_update,) * 8

        pipelines = _state.get("pipelines", {})
        if model_key not in pipelines:
            return (no_update, "Selected model is not available. Please retrain.") + (no_update,) * 8

        class_names = _state["class_names"]
        X = modules.prepare_single_input(
            cleaned["raisedhands"],
            cleaned["visited_resources"],
            cleaned["discussion"],
            cleaned["absence_days"],
            cleaned["attendance_pct"],
            cleaned["study_hours"],
        )

        try:
            selected_pipeline = pipelines[model_key]
            result = modules.predict(selected_pipeline, X, class_names)
            top_rankings = _state.get("rankings", [])[:3]
            top_predictions = modules.predict_top_models(pipelines, top_rankings, X, class_names)
        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            return (no_update, "Prediction failed due to an internal error.") + (no_update,) * 8

        # Build extended inputs dict for display
        extended_inputs = dict(cleaned)
        extended_inputs["prev_score"] = prev_score
        extended_inputs["assignment_pct"] = assignment

        panel = _build_result_panel(
            result=result,
            model_key=model_key,
            inputs=extended_inputs,
            top_predictions=top_predictions,
            top_rankings=top_rankings,
        )
        return (panel, "", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update)

    @app.callback(
        Output("cm-graph", "figure"),
        Output("fi-graph", "figure"),
        Input("cm-model-selector", "value"),
        prevent_initial_call=False,
    )
    def update_cm_and_fi(model_key):
        if not model_key or model_key not in _state.get("metrics", {}):
            empty = go.Figure()
            return empty, empty
        metrics = _state["metrics"][model_key]
        class_names = _state["class_names"]
        pipeline = _state["pipelines"][model_key]

        cm_fig = modules.chart_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            config.MODEL_DISPLAY_NAMES[model_key],
        )
        fi_fig = modules.chart_feature_importance(pipeline, model_key)
        if fi_fig is None:
            fi_fig = go.Figure().update_layout(
                title=f"Feature Importance — not available for {config.MODEL_DISPLAY_NAMES[model_key]}",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#1e293b",
                font=dict(color="#e2e8f0", size=11),
            )
        return cm_fig, fi_fig

    return app


# ── Result panel helpers ──────────────────────────────────────────────────────

def _empty_result_panel():
    return _card([
        _section_title("Prediction Result & Model Comparison", "Awaiting student engagement inputs."),
        html.Div([
            html.Div(style={
                "display": "inline-flex",
                "alignItems": "flex-end",
                "gap": "5px",
                "background": "#e2e8f0",
                "padding": "10px 12px",
                "borderRadius": "12px",
                "marginBottom": "14px",
            }, children=[
                html.Div(style={"width": "6px", "height": "16px", "background": "#22c55e", "borderRadius": "3px"}),
                html.Div(style={"width": "6px", "height": "22px", "background": "#ec4899", "borderRadius": "3px"}),
                html.Div(style={"width": "6px", "height": "28px", "background": "#3b82f6", "borderRadius": "3px"}),
            ]),
            html.H3("Ready for Prediction", style={"color": TEXT, "fontSize": "1.15rem", "fontWeight": "700", "marginBottom": "8px"}),
            html.P(
                "Adjust student parameters on the left and click Predict. "
                "The system will evaluate performance across the top 3 best-performing "
                "machine learning models simultaneously.",
                style={"color": MUTED, "fontSize": "0.85rem", "maxWidth": "440px", "margin": "0 auto 18px", "lineHeight": "1.5"},
            ),
            html.Div([
                html.Div([
                    html.Span("1", style={"background": AMBER, "color": NAVY, "borderRadius": "50%", "width": "20px", "height": "20px", "display": "inline-flex", "alignItems": "center", "justifyContent": "center", "fontSize": "0.75rem", "fontWeight": "800", "marginRight": "10px"}),
                    html.Span("Selected Model Result with Probabilities", style={"color": TEXT, "fontSize": "0.82rem"}),
                ], style={"background": NAVY, "padding": "10px 14px", "borderRadius": "6px", "border": f"1px solid {BORDER}", "display": "flex", "alignItems": "center"}),
                html.Div([
                    html.Span("2", style={"background": INFO, "color": "#ffffff", "borderRadius": "50%", "width": "20px", "height": "20px", "display": "inline-flex", "alignItems": "center", "justifyContent": "center", "fontSize": "0.75rem", "fontWeight": "800", "marginRight": "10px"}),
                    html.Span("Top 3 Multi-Model Comparison Table", style={"color": TEXT, "fontSize": "0.82rem"}),
                ], style={"background": NAVY, "padding": "10px 14px", "borderRadius": "6px", "border": f"1px solid {BORDER}", "display": "flex", "alignItems": "center"}),
                html.Div([
                    html.Span("3", style={"background": SUCCESS, "color": "#ffffff", "borderRadius": "50%", "width": "20px", "height": "20px", "display": "inline-flex", "alignItems": "center", "justifyContent": "center", "fontSize": "0.75rem", "fontWeight": "800", "marginRight": "10px"}),
                    html.Span("Comparative Graphs & Pass/Fail Benchmark", style={"color": TEXT, "fontSize": "0.82rem"}),
                ], style={"background": NAVY, "padding": "10px 14px", "borderRadius": "6px", "border": f"1px solid {BORDER}", "display": "flex", "alignItems": "center"}),
            ], style={"display": "flex", "flexDirection": "column", "gap": "10px", "maxWidth": "380px", "margin": "0 auto"}),
        ], style={"textAlign": "center", "padding": "32px 16px"}),
    ], style_extra={"boxSizing": "border-box", "marginBottom": "0"})


def _build_default_result_panel():
    pipelines = _state.get("pipelines", {})
    if not pipelines or "random_forest" not in pipelines:
        return _empty_result_panel()
    class_names = _state.get("class_names", config.CLASS_ORDER)
    X = modules.prepare_single_input(43, 76, 43, "Under-7", 85.0, 4.0)
    try:
        result = modules.predict(pipelines["random_forest"], X, class_names)
        top_rankings = _state.get("rankings", [])[:3]
        top_predictions = modules.predict_top_models(pipelines, top_rankings, X, class_names)
        return _build_result_panel(
            result=result,
            model_key="random_forest",
            inputs={"raisedhands": 43, "visited_resources": 76, "discussion": 43, "absence_days": "Under-7", "attendance_pct": 85.0, "study_hours": 4.0},
            top_predictions=top_predictions,
            top_rankings=top_rankings,
        )
    except Exception as exc:
        logger.exception("Failed to build default result panel: %s", exc)
        return _empty_result_panel()


def _build_result_panel(
    result: dict,
    model_key: str,
    inputs: dict,
    top_predictions: list,
    top_rankings: list,
):
    cls = result["predicted_class"]
    label = result["predicted_label"]
    probabilities = result["probabilities"]

    cls_color = {"L": DANGER, "M": INFO, "H": SUCCESS}.get(cls, AMBER)

    best_model = top_rankings[0] if top_rankings else None
    selected_rank = None
    for r in _state.get("rankings", []):
        if r["key"] == model_key:
            selected_rank = r["rank"]
            break

    # Recommendations for actionable student improvement
    recs = modules.get_improvement_recommendations(
        raisedhands=inputs.get("raisedhands", 43),
        visited_resources=inputs.get("visited_resources", 76),
        discussion=inputs.get("discussion", 43),
        absence_days=inputs.get("absence_days", "Under-7"),
        predicted_class=cls,
        attendance_pct=inputs.get("attendance_pct"),
        study_hours=inputs.get("study_hours"),
    )

    # Probability comparison graph
    prob_fig, prob_reason = modules.chart_top3_probability_comparison(top_predictions)

    # Graph tabs sub-component
    graph_tabs = dcc.Tabs(
        children=[
            dcc.Tab(
                label="Prediction Comparison",
                children=[
                    dcc.Graph(
                        figure=modules.chart_top3_prediction_comparison(top_predictions),
                        config={"displayModeBar": False, "responsive": True},
                        style={"width": "100%", "height": "280px"},
                        responsive=True,
                    ),
                ],
                style={"padding": "8px 12px", "color": MUTED, "background": "transparent", "border": "none", "fontSize": "0.82rem"},
                selected_style={"padding": "8px 12px", "color": AMBER, "borderBottom": f"2px solid {AMBER}", "background": "transparent", "fontSize": "0.82rem"},
            ),
            dcc.Tab(
                label="Class Probabilities",
                children=[
                    dcc.Graph(
                        figure=prob_fig,
                        config={"displayModeBar": False, "responsive": True},
                        style={"width": "100%", "height": "280px"},
                        responsive=True,
                    ) if prob_fig is not None else html.Div(
                        prob_reason,
                        style={"color": MUTED, "padding": "30px 16px", "textAlign": "center", "fontSize": "0.85rem"},
                    ),
                ],
                style={"padding": "8px 12px", "color": MUTED, "background": "transparent", "border": "none", "fontSize": "0.82rem"},
                selected_style={"padding": "8px 12px", "color": AMBER, "borderBottom": f"2px solid {AMBER}", "background": "transparent", "fontSize": "0.82rem"},
            ),
            dcc.Tab(
                label="Test Performance",
                children=[
                    dcc.Graph(
                        figure=modules.chart_top3_metrics_comparison(top_rankings),
                        config={"displayModeBar": False, "responsive": True},
                        style={"width": "100%", "height": "280px"},
                        responsive=True,
                    ),
                ],
                style={"padding": "8px 12px", "color": MUTED, "background": "transparent", "border": "none", "fontSize": "0.82rem"},
                selected_style={"padding": "8px 12px", "color": AMBER, "borderBottom": f"2px solid {AMBER}", "background": "transparent", "fontSize": "0.82rem"},
            ),
        ],
        style={"borderBottom": f"1px solid {BORDER}", "marginBottom": "12px"},
        colors={"border": "transparent", "primary": AMBER, "background": "transparent"},
    )

    children = [
        _section_title("Prediction Result"),

        # Big result display (matching reference image)
        html.Div([
            html.Div(style={
                "display": "inline-flex",
                "alignItems": "flex-end",
                "gap": "5px",
                "background": "#e2e8f0",
                "padding": "8px 10px",
                "borderRadius": "10px",
                "marginBottom": "10px",
            }, children=[
                html.Div(style={"width": "5px", "height": "14px", "background": "#22c55e", "borderRadius": "2px"}),
                html.Div(style={"width": "5px", "height": "20px", "background": "#ec4899", "borderRadius": "2px"}),
                html.Div(style={"width": "5px", "height": "26px", "background": "#3b82f6", "borderRadius": "2px"}),
            ]),
            html.Div(label.upper(), style={
                "fontSize": "2.2rem",
                "fontWeight": "800",
                "color": cls_color,
                "letterSpacing": "0.06em",
                "lineHeight": "1.1",
            }),
            html.Div("PERFORMANCE LEVEL", style={
                "color": MUTED,
                "fontSize": "0.78rem",
                "textTransform": "uppercase",
                "letterSpacing": "0.1em",
                "marginTop": "4px",
            }),
            html.Div([
                html.Span(f"Model: {config.MODEL_DISPLAY_NAMES[model_key]}", style={"color": TEXT, "fontSize": "0.85rem", "fontWeight": "600"}),
                html.Span(
                    "  •  🏆 Rank 1 (Best Model)" if selected_rank == 1 else
                    f"  •  Rank {selected_rank} (Top Model: {best_model['display_name'] if best_model else 'N/A'})",
                    style={"color": AMBER if selected_rank == 1 else MUTED, "fontSize": "0.78rem", "marginLeft": "6px"},
                ),
            ], style={"marginTop": "8px", "marginBottom": "16px"}),
        ], style={"textAlign": "center", "padding": "10px 0 8px"}),

        # Class Probabilities
        _render_probabilities(probabilities) if probabilities else html.Div(
            "Probability estimates not available for this model.",
            style={"color": MUTED, "fontSize": "0.82rem", "textAlign": "center", "marginBottom": "12px"},
        ),

        # Input Summary
        html.Div([
            html.Div("INPUT SUMMARY", style={
                "color": MUTED, "fontSize": "0.75rem",
                "textTransform": "uppercase", "letterSpacing": "0.08em",
                "marginTop": "14px", "marginBottom": "8px",
                "fontWeight": "700",
            }),
            html.Div([
                html.Div([
                    html.Span(k + ": ", style={"color": MUTED, "fontSize": "0.82rem"}),
                    html.Span(str(v), style={"color": TEXT, "fontSize": "0.82rem", "fontWeight": "600"}),
                ], style={"padding": "4px 0", "borderBottom": f"1px solid {BORDER}"})
                for k, v in [
                    ("Raised Hands", inputs["raisedhands"]),
                    ("Visited Resources", inputs["visited_resources"]),
                    ("Discussion", inputs["discussion"]),
                    ("Absence Days", inputs["absence_days"]),
                    ("Attendance %", f"{inputs.get('attendance_pct', 'N/A')}%" if inputs.get("attendance_pct") is not None else "N/A"),
                    ("Study Hours/Day", f"{inputs.get('study_hours', 'N/A')} hrs" if inputs.get("study_hours") is not None else "N/A"),
                ]
            ]),
        ]),

        # Top 3 Models Comparison Table
        html.Div([
            html.Div([
                html.H3("Top 3 ML Model Comparison", style={
                    "color": TEXT, "fontSize": "1rem", "fontWeight": "700", "margin": "0 0 4px 0",
                }),
                html.P(
                    f"Evaluated on held-out test data (30% split, {_state.get('test_size', '—')} samples). Ranked by Test F1-Score.",
                    style={"color": MUTED, "fontSize": "0.78rem", "margin": "0 0 10px 0"},
                ),
            ], style={"marginTop": "18px"}),

            html.Table([
                html.Thead(html.Tr([
                    html.Th("Rank", style={"padding": "6px 8px", "textAlign": "left", "color": AMBER, "fontSize": "0.78rem"}),
                    html.Th("Model", style={"padding": "6px 8px", "textAlign": "left", "color": AMBER, "fontSize": "0.78rem"}),
                    html.Th("Prediction", style={"padding": "6px 8px", "color": AMBER, "fontSize": "0.78rem", "textAlign": "center"}),
                    html.Th("Pass / Fail", style={"padding": "6px 8px", "color": AMBER, "fontSize": "0.78rem", "textAlign": "center"}),
                    html.Th("Accuracy", style={"padding": "6px 8px", "color": AMBER, "fontSize": "0.78rem", "textAlign": "center"}),
                    html.Th("F1-Score", style={"padding": "6px 8px", "color": AMBER, "fontSize": "0.78rem", "textAlign": "center"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(
                            f"#{p['rank']}" if p["rank"] > 1 else "🏆 #1",
                            style={"padding": "6px 8px", "fontWeight": "700", "color": AMBER if p["rank"] == 1 else TEXT, "fontSize": "0.8rem"},
                        ),
                        html.Td([
                            html.Span(p["display_name"], style={"fontWeight": "600", "color": TEXT, "fontSize": "0.82rem"}),
                            html.Span(" (Selected)", style={"color": AMBER, "fontSize": "0.72rem", "fontWeight": "700", "marginLeft": "4px"}) if p["key"] == model_key else None,
                        ], style={"padding": "6px 8px"}),
                        html.Td(
                            html.Span(
                                p["predicted_label"],
                                style={
                                    "background": {"L": "rgba(239, 68, 68, 0.2)", "M": "rgba(59, 130, 246, 0.2)", "H": "rgba(34, 197, 94, 0.2)"}.get(p["predicted_class"], "transparent"),
                                    "color": {"L": DANGER, "M": INFO, "H": SUCCESS}.get(p["predicted_class"], TEXT),
                                    "padding": "2px 6px",
                                    "borderRadius": "4px",
                                    "fontSize": "0.78rem",
                                    "fontWeight": "600",
                                },
                            ),
                            style={"padding": "6px 8px", "textAlign": "center"},
                        ),
                        html.Td(
                            html.Span(
                                p["pass_fail"],
                                style={
                                    "color": SUCCESS if p["pass_fail"] == "Pass" else DANGER,
                                    "fontWeight": "700",
                                    "fontSize": "0.78rem",
                                },
                            ),
                            style={"padding": "6px 8px", "textAlign": "center"},
                        ),
                        html.Td(f"{p['accuracy']:.1%}", style={"padding": "6px 8px", "textAlign": "center", "color": TEXT, "fontSize": "0.8rem"}),
                        html.Td(f"{p['f1']:.3f}", style={"padding": "6px 8px", "textAlign": "center", "color": TEXT, "fontSize": "0.8rem"}),
                    ], style={
                        "borderTop": f"1px solid {BORDER}",
                        "background": "rgba(245, 158, 11, 0.08)" if p["rank"] == 1 else "transparent",
                    })
                    for p in top_predictions
                ]),
            ], style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "10px"}),

            # Pass/Fail rule callout box
            html.Div([
                html.Span("ℹ Pass/Fail Benchmark: ", style={"fontWeight": "700", "color": AMBER, "fontSize": "0.78rem"}),
                html.Span(config.PASS_FAIL_RULE_DESC, style={"color": MUTED, "fontSize": "0.78rem"}),
            ], style={
                "background": NAVY,
                "border": f"1px solid {BORDER}",
                "borderRadius": "6px",
                "padding": "8px 12px",
                "marginBottom": "14px",
            }),
        ]),

        # Actionable Recommendations for Improvement (evidence-based student guidance)
        *( [
            html.Div([
                html.H3("Actionable Recommendations for Improvement", style={"color": TEXT, "fontSize": "0.95rem", "fontWeight": "700", "marginBottom": "8px"}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Span(r["icon"], style={"fontSize": "1.1rem", "marginRight": "8px"}),
                            html.Strong(r["title"], style={"color": TEXT, "fontSize": "0.85rem"}),
                            html.Span(
                                r["priority"].upper(),
                                style={
                                    "background": "rgba(239, 68, 68, 0.15)" if r["priority"] == "high" else ("rgba(245, 158, 11, 0.15)" if r["priority"] == "medium" else "rgba(34, 197, 94, 0.15)"),
                                    "color": DANGER if r["priority"] == "high" else (AMBER if r["priority"] == "medium" else SUCCESS),
                                    "fontSize": "0.68rem",
                                    "fontWeight": "700",
                                    "padding": "2px 6px",
                                    "borderRadius": "4px",
                                    "marginLeft": "auto",
                                },
                            ),
                        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
                        html.P(r["detail"], style={"color": MUTED, "fontSize": "0.8rem", "margin": "0", "lineHeight": "1.4"}),
                    ], style={
                        "background": NAVY,
                        "border": f"1px solid {BORDER}",
                        "borderLeft": f"3px solid {DANGER if r['priority'] == 'high' else (AMBER if r['priority'] == 'medium' else SUCCESS)}",
                        "borderRadius": "6px",
                        "padding": "8px 12px",
                        "marginBottom": "8px",
                    })
                    for r in recs
                ]),
            ], style={"marginTop": "6px", "marginBottom": "12px"})
        ] if recs else [] ),

        # Comparison Graphs section
        html.Div([
            html.H3("Comparative Analysis", style={"color": TEXT, "fontSize": "0.95rem", "fontWeight": "700", "marginBottom": "8px"}),
            graph_tabs,
        ], style={"marginTop": "8px"}),

        # Footer disclaimer
        html.P(
            "⚠ Prediction is based on engagement patterns in the training dataset only. "
            "It does not account for all factors affecting student performance.",
            style={"color": MUTED, "fontSize": "0.75rem", "marginTop": "14px", "fontStyle": "italic", "lineHeight": "1.4"},
        ),
    ]
    return _card(children, style_extra={"boxSizing": "border-box", "marginBottom": "0"})


def _render_probabilities(probabilities: dict):
    """Render probability bars for each class matching reference design."""
    rows = []
    for cls in config.CLASS_ORDER:
        if cls not in probabilities:
            continue
        prob = probabilities[cls]
        label = config.CLASS_LABELS[cls]
        bar_color = {"L": DANGER, "M": INFO, "H": SUCCESS}.get(cls, AMBER)
        rows.append(html.Div([
            html.Div([
                html.Span(label, style={"color": TEXT, "fontSize": "0.82rem", "minWidth": "55px", "display": "inline-block"}),
                html.Span(f"{prob:.1%}", style={"color": MUTED, "fontSize": "0.82rem", "float": "right"}),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "3px"}),
            html.Div(
                html.Div(style={
                    "width": f"{prob * 100:.1f}%",
                    "background": bar_color,
                    "height": "6px",
                    "borderRadius": "3px",
                    "transition": "width 0.4s ease",
                }),
                style={
                    "background": BORDER,
                    "borderRadius": "3px",
                    "height": "6px",
                    "marginBottom": "10px",
                },
            ),
        ]))
    return html.Div([
        html.Div("CLASS PROBABILITIES", style={
            "color": MUTED, "fontSize": "0.75rem",
            "textTransform": "uppercase", "letterSpacing": "0.08em",
            "marginBottom": "10px",
            "fontWeight": "700",
        }),
        *rows,
    ])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ok = _bootstrap_state()
    if not ok:
        logger.error("Failed to initialise application. Exiting.")
        sys.exit(1)

    app = create_app()
    logger.info("Starting dashboard at http://%s:%d/", config.HOST, config.PORT)
    app.run(
        debug=config.DEBUG,
        host=config.HOST,
        port=config.PORT,
    )
