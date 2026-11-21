# Machine Learning Assignment Projects

This repository contains two supervised machine learning projects.

| Project | Problem type | Directory |
|---|---|---|
| Customer Telecommunications Churn | Classification | `customer-telecommunications-churn-prediction/` |
| Flight Price Prediction | Regression | `flight-price-prediction/` |

## Clone the repository

```bash
git clone https://github.com/Deraraf/flight-price-prediction.git
cd flight-price-prediction
```

The current GitHub repository name is `flight-price-prediction`, but it contains both assignment projects.

## Run the classification project

Requirements:

- Python 3.14 or newer
- `uv`

```bash
cd customer-telecommunications-churn-prediction
uv sync
uv run python train.py
uv run jupyter notebook customer_churn_assignment.ipynb
uv run streamlit run app.py
```

The Streamlit app is available at `http://localhost:8501`.

Submission files in this project include:

- Jupyter notebook
- Python training and deployment code
- Dataset
- PowerPoint presentation
- Generated evaluation artifacts

## Run the regression project

```bash
cd ../flight-price-prediction
uv sync
PYTHONPATH=src uv run python -m unittest discover -s tests -v
uv run flight-price-prediction train
uv run jupyter notebook notebooks/flight_price_prediction.ipynb
```

See [flight-price-prediction/README.md](flight-price-prediction/README.md) for complete regression-project instructions.

## Repository setup

Both projects use separate local environments and lock files. Run setup and commands inside the project directory being reviewed.

Do not commit virtual environments, `.next`, `node_modules`, or local cache directories. The root `.gitignore` excludes these generated files.
