# Flight Price Prediction — Assignment 1

This project follows a supervised **regression** workflow using the Kaggle flight
price dataset in `data/raw/Clean_Dataset.csv`. Its goal is to estimate the listed
fare from the itinerary and booking details. The target is `price`, and the
primary evaluation metric is RMSE; MAE and R² are reported alongside it.

The measurable success check is that the selected model has lower held-out test
RMSE than a median-fare baseline. An R² of 0.80 or higher is a stretch goal, not
a promised result. Use the actual generated report when discussing results.

## Setup

The project requires Python 3.14 or newer. From this directory, install the
declared dependencies with:

```bash
uv sync
```

The CSV is expected at `data/raw/Clean_Dataset.csv`. The supplied dataset is
read locally; training does not download data or contact Kaggle.

## Run the workflow

Train, tune and evaluate the model:

```bash
uv run flight-price-prediction train
```

The defaults use a reproducible 80% training / 10% validation / 10% test split.
The training portion is used for 3-fold randomized hyperparameter search. The
validation set selects between a median baseline, Ridge regression, and tuned
HistGradientBoosting; the test set is held back until final evaluation. To
change the search budget or paths:

```bash
uv run flight-price-prediction train \
  --data data/raw/Clean_Dataset.csv \
  --output outputs \
  --search-iterations 4 \
  --cv-folds 3
```

The command saves:

- `outputs/figures/` — fare distribution, fare by class, median fare by airline,
  and numeric-correlation plots.
- `outputs/reports/metrics.json` — cleaning counts, split sizes, selected
  model, validation scores, test metrics, baseline test metrics, and search
  parameters.
- `outputs/reports/validation_metrics.csv` — validation-set model comparison.
- `outputs/models/flight_price_model.joblib` — fitted preprocessing and model
  pipeline for local inference.

Open `notebooks/flight_price_prediction.ipynb` to review the workflow in
notebook form:

```bash
uv run jupyter notebook notebooks/flight_price_prediction.ipynb
```

## Local inference demo

After training, predict the fare for the example record:

```bash
uv run flight-price-prediction predict \
  --model outputs/models/flight_price_model.joblib \
  --input-json examples/flight_record.json
```

The prediction command accepts one JSON object or a list of objects. Each record
must contain `airline`, `source_city`, `departure_time`, `stops`,
`arrival_time`, `destination_city`, `class`, `duration`, and `days_left`.
`flight` and `price` are deliberately not model inputs. This command is a local
deployment/inference demo, not a hosted production service.

## Method and decisions

1. **Problem and data:** predict Indian flight fares from the supplied
   tabular dataset.
2. **Data checks:** count missing values and duplicates; remove duplicate rows,
   the CSV row-number column, and rows with missing or invalid targets. Missing
   feature values are imputed inside the model pipeline.
3. **Exploration:** inspect fare distribution, compare fare by class and
   airline, and inspect numeric correlations. Fares are strongly skewed and
   unusually high fares can represent the premium class, so the workflow does
   not delete valid fare outliers.
4. **Feature preparation:** one-hot encode categorical details, scale and
   median-impute numeric details, and create a source-to-destination `route`.
   The high-cardinality `flight` code and CSV row index are excluded to avoid
   memorizing identifiers.
5. **Training and tuning:** compare Ridge with a tuned
   HistGradientBoosting regressor. Preprocessing is part of each scikit-learn
   pipeline, so it is fit on training folds only. Search settings and the
   random seed are fixed for reproducibility.
6. **Evaluation:** choose using validation RMSE, then report test RMSE, MAE, and
   R² once. RMSE penalizes large fare errors more; MAE gives the typical
   absolute error in the dataset's fare units; R² measures explained variance.
7. **Deployment and maintenance:** save a reusable fitted pipeline and use the
   local prediction command. For real deployment, monitor prediction errors
   and input drift and retrain when needed.

Use the generated figures and metrics in your presentation, and explain what
they mean in your own words. Do not report the success goal as an achieved
result unless the generated test metrics support it.

## Tests

Run the focused data-preparation tests from this directory:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```
