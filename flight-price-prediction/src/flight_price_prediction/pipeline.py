"""Data preparation, model selection, evaluation, and inference for flight fares."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Clean_Dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
TARGET = "price"
RAW_FEATURES = (
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
)
NUMERIC_FEATURES = ("duration", "days_left")
CATEGORICAL_FEATURES = (
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "route",
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the source CSV and verify that it has the expected prediction fields."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    frame = pd.read_csv(data_path)
    required = set(RAW_FEATURES) | {TARGET}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return frame


def clean_flight_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove duplicate rows and invalid targets; retain feature gaps for imputation."""
    required = set(RAW_FEATURES) | {TARGET}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    cleaned = frame.drop(
        columns=[column for column in frame.columns if column.startswith("Unnamed:")],
        errors="ignore",
    )
    initial_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    duplicates_removed = initial_rows - len(cleaned)

    numeric_target = pd.to_numeric(cleaned[TARGET], errors="coerce")
    valid_target = numeric_target.notna() & np.isfinite(numeric_target)
    invalid_targets_removed = int((~valid_target).sum())
    cleaned = cleaned.loc[valid_target].copy()
    cleaned[TARGET] = numeric_target.loc[valid_target].astype(float)

    for column in NUMERIC_FEATURES:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].replace([np.inf, -np.inf], np.nan)

    report = {
        "raw_rows": (len(frame)),
        "duplicates_removed": int(duplicates_removed),
        "invalid_targets_removed": invalid_targets_removed,
        "clean_rows": (len(cleaned)),
    }
    return cleaned.reset_index(drop=True), report


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model inputs, including a route feature, without using fare or flight ID."""
    missing = sorted(set(RAW_FEATURES).difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required fields: {', '.join(missing)}")

    features = frame.loc[:, list(RAW_FEATURES)].copy()
    features["route"] = (
        features["source_city"].astype("string").fillna("Unknown")
        + " -> "
        + features["destination_city"].astype("string").fillna("Unknown")
    )

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce")
        features[column] = features[column].replace([np.inf, -np.inf], np.nan)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("object")
        features[column] = features[column].where(features[column].notna(), np.nan)
    return features.loc[:, list(FEATURES)]


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def _build_pipeline(regressor: Any) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", _build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def _regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _predict(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(features))


def _save_eda_plots(frame: pd.DataFrame, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(data=frame, x=TARGET, bins=60, log_scale=(True, False), ax=axis)
    axis.set(title="Flight fare distribution (log-scaled x-axis)", xlabel="Fare")
    fig.tight_layout()
    fig.savefig(directory / "fare_distribution.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=frame, x="class", y=TARGET, showfliers=False, ax=axis)
    axis.set(
        title="Fare by travel class (outliers hidden for readability)", ylabel="Fare"
    )
    fig.tight_layout()
    fig.savefig(directory / "fare_by_class.png", dpi=150)
    plt.close(fig)

    airline_fares = (
        frame.groupby("airline", observed=True)[TARGET].median().sort_values()
    )
    fig, axis = plt.subplots(figsize=(9, 5))
    airline_fares.plot.barh(ax=axis, color="steelblue")
    axis.set(title="Median fare by airline", xlabel="Median fare", ylabel="Airline")
    fig.tight_layout()
    fig.savefig(directory / "median_fare_by_airline.png", dpi=150)
    plt.close(fig)

    numeric = frame.select_dtypes(include="number")
    fig, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=axis)
    axis.set(title="Numeric feature correlations")
    fig.tight_layout()
    fig.savefig(directory / "numeric_correlations.png", dpi=150)
    plt.close(fig)


def run_experiment(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    search_iterations: int = 4,
    cv_folds: int = 3,
) -> dict[str, Any]:
    """Train, tune on training data, select on validation data, and test once."""
    if search_iterations < 1:
        raise ValueError("search_iterations must be at least 1")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    output_path = Path(output_dir)
    frame, cleaning_report = clean_flight_data(load_dataset(data_path))
    if len(frame) < cv_folds + 2:
        raise ValueError(
            "Not enough valid rows for the requested train/validation/test split."
        )
    _save_eda_plots(frame, output_path / "figures")

    features = make_features(frame)
    target = frame[TARGET]
    train_validation_x, test_x, train_validation_y, test_y = train_test_split(
        features, target, test_size=0.10, random_state=42
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_validation_x,
        train_validation_y,
        test_size=1 / 9,
        random_state=42,
    )

    validation_scores: dict[str, dict[str, float]] = {}
    validation_models: dict[str, Pipeline] = {}

    median_baseline = _build_pipeline(DummyRegressor(strategy="median"))
    median_baseline.fit(train_x, train_y)
    validation_models["Median baseline"] = median_baseline
    validation_scores["Median baseline"] = _regression_metrics(
        validation_y, _predict(median_baseline, validation_x)
    )

    ridge = _build_pipeline(Ridge(alpha=1.0))
    ridge.fit(train_x, train_y)
    validation_models["Ridge regression"] = ridge
    validation_scores["Ridge regression"] = _regression_metrics(
        validation_y, _predict(ridge, validation_x)
    )

    search = RandomizedSearchCV(
        estimator=_build_pipeline(HistGradientBoostingRegressor(random_state=42)),
        param_distributions={
            "regressor__learning_rate": [0.05, 0.1],
            "regressor__max_iter": [80, 120],
            "regressor__max_leaf_nodes": [15, 31],
            "regressor__min_samples_leaf": [20, 50],
            "regressor__l2_regularization": [0.0, 1.0],
        },
        n_iter=search_iterations,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    search.fit(train_x, train_y)
    tuned_model = search.best_estimator_
    if not isinstance(tuned_model, Pipeline):
        raise TypeError("Hyperparameter search did not return a pipeline.")
    validation_models["Tuned HistGradientBoosting"] = tuned_model
    validation_scores["Tuned HistGradientBoosting"] = _regression_metrics(
        validation_y, _predict(tuned_model, validation_x)
    )

    selected_name = min(
        validation_scores, key=lambda name: validation_scores[name]["rmse"]
    )
    if selected_name == "Median baseline":
        final_model = _build_pipeline(DummyRegressor(strategy="median"))
    elif selected_name == "Ridge regression":
        final_model = _build_pipeline(Ridge(alpha=1.0))
    else:
        final_model = tuned_model

    final_model.fit(
        pd.concat([train_x, validation_x]),
        pd.concat([train_y, validation_y]),
    )
    test_metrics = _regression_metrics(test_y, _predict(final_model, test_x))
    baseline_final = _build_pipeline(DummyRegressor(strategy="median"))
    combined_x = pd.concat([train_x, validation_x])
    combined_y = pd.concat([train_y, validation_y])
    baseline_final.fit(combined_x, combined_y)
    baseline_test_metrics = _regression_metrics(
        test_y, _predict(baseline_final, test_x)
    )

    model_dir = output_path / "models"
    report_dir = output_path / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "flight_price_model.joblib"
    joblib.dump(final_model, model_path)

    validation_table = pd.DataFrame(validation_scores).T
    validation_table.index.name = "model"
    validation_table = validation_table.sort_values("rmse")
    validation_table.to_csv(report_dir / "validation_metrics.csv")

    results: dict[str, Any] = {
        "dataset": str(Path(data_path)),
        "cleaning": cleaning_report,
        "split_rows": {
            "train": (len(train_x)),
            "validation": (len(validation_x)),
            "test": (len(test_x)),
        },
        "selected_model": selected_name,
        "validation_metrics": validation_table.to_dict(orient="index"),
        "test_metrics": test_metrics,
        "median_baseline_test_metrics": baseline_test_metrics,
        "best_hist_gradient_boosting_parameters": search.best_params_,
        "model_path": str(model_path),
    }
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results


def predict_fares(model_path: str | Path, records: list[dict[str, Any]]) -> list[float]:
    """Load a trained pipeline and predict fares for one or more flight records."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    model = joblib.load(path)
    predictions = model.predict(make_features(pd.DataFrame(records)))
    return [float(value) for value in predictions]
