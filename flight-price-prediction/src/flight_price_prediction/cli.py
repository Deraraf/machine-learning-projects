"""Command-line entry points for model training and local fare predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flight_price_prediction.pipeline import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    predict_fares,
    run_experiment,
)


def _read_prediction_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(record, dict) for record in payload):
        return payload
    raise ValueError("Prediction JSON must contain a flight object or a list of flight objects.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flight-price-prediction",
        description="Train and use a flight fare regression model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Run EDA, tune, evaluate, and save a model.")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    train_parser.add_argument("--search-iterations", type=int, default=4)
    train_parser.add_argument("--cv-folds", type=int, default=3)

    predict_parser = commands.add_parser("predict", help="Predict fare from a JSON flight record.")
    predict_parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "models" / "flight_price_model.joblib",
    )
    predict_parser.add_argument("--input-json", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.command == "train":
        results = run_experiment(
            arguments.data,
            arguments.output,
            search_iterations=arguments.search_iterations,
            cv_folds=arguments.cv_folds,
        )
        print(json.dumps(results, indent=2))
        return

    records = _read_prediction_records(arguments.input_json)
    predictions = predict_fares(arguments.model, records)
    print(json.dumps([{"predicted_price": price} for price in predictions], indent=2))
