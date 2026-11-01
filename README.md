# Student Performance Predictor

A machine learning dashboard that predicts student academic performance (Low / Medium / High)
using the UCI Student Academics Performance dataset.

Built entirely in Python — Flask + Dash (no HTML/CSS/JS files).

## Features

- 5 scikit-learn models: Decision Tree, Random Forest, Logistic Regression, Perceptron, MLP
- Consistent scikit-learn Pipeline (preprocessing → classifier) for training and prediction
- Dash dashboard: home stats, prediction form with probability bars, model comparison charts
- Automated tests with pytest

## Project Structure

```
student performance predictor/
├── app.py              # Dash + Flask application (pure Python UI)
├── modules.py          # ML logic: data, preprocessing, training, prediction, charts
├── train_model.py      # One-time training script
├── config.py           # Centralised configuration
├── requirements.txt    # Pinned dependencies
├── README.md
├── data/
│   └── AI-Data.csv     # Dataset (480 rows, 17 columns)
├── models/
│   └── pipelines.joblib  # Trained model artifacts (generated at runtime)
└── tests/
    └── test_app.py     # Automated tests
```

## Features Used for Prediction

The model uses exactly 4 features from the dataset (matching the original project):

| Feature | Type | Range |
|---------|------|-------|
| `raisedhands` | integer | 0–100 |
| `VisITedResources` | integer | 0–100 |
| `Discussion` | integer | 0–100 |
| `StudentAbsenceDays` | categorical | Under-7 / Above-7 |

Target: `Class` → L (Low) / M (Medium) / H (High)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify dataset

Ensure `data/AI-Data.csv` exists (copied from the original project).

### 3. Train models (required before first run)

```bash
python train_model.py
```

This trains all 5 models, evaluates them on the held-out test set (30%),
and saves `models/pipelines.joblib`.

### 4. Run the application

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:8050/**

## Running Tests

```bash
python -m pytest tests/test_app.py -v
```

## Security Notes

- Inputs are validated server-side with strict range and type checks
- No stack traces are exposed to users
- Debug mode is disabled by default (set `FLASK_DEBUG=true` only for development)
- Model files are loaded only from the controlled local `models/` path
- No secrets are hardcoded; set `SECRET_KEY` and `LOG_LEVEL` via environment variables

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `false` | Enable Flask/Dash debug mode |
| `FLASK_HOST` | `127.0.0.1` | Bind address |
| `FLASK_PORT` | `8050` | Port |
| `SECRET_KEY` | dev placeholder | Flask session secret (must be changed in production) |
| `LOG_LEVEL` | `INFO` | Python logging level |

## Bugs Fixed from Original `Project.py`

1. **LabelEncoder bug** — `dtype == type(object)` never matched; fixed with a proper pipeline
2. **Logistic Regression reported Perceptron accuracy** — fixed
3. **No reproducibility** — `random_state=42` used throughout
4. **No preprocessing pipeline** — sklearn Pipeline prevents data leakage
5. **Scale-sensitive models unscaled** — StandardScaler applied to all pipelines
6. **AnnouncementsView inconsistency** — dropped from training and not collected
7. **`plt.show()` would block a server** — replaced with Plotly figures
8. **Interactive `input()` calls** — replaced by Dash form callbacks
9. **No input validation** — full server-side validation in `modules.validate_inputs()`
10. **No error handling** — custom error handlers added throughout
