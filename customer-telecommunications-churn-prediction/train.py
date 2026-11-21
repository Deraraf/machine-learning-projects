import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"


def load_data() -> pd.DataFrame:
    raw = pd.read_excel(DATA_PATH)
    column_line = raw.columns[0]
    columns = [name.strip() for name in str(column_line).split(",")]
    rows = raw.iloc[:0].dropna().astype(str).str.split(",", expand=True)
    rows.columns = columns

    for column in ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")

    rows["Churn"] = rows["Churn"].str.strip()
    rows = rows.drop(columns=["customerID"])
    rows = rows.drop_duplicates()
    return rows


def save_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="Churn", ax=axes[0])
    axes[0].set_title("Customer Churn Distribution")
    df["Churn"].value_counts(normalize=True).mul(100).plot(
        kind="bar", ax=axes[1], color=["#4c78a8", "#f58518"]
    )
    axes[1].set_title("Churn Percentage")
    axes[1].set_ylabel("Percent")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Churn", y="tenure", ax=ax)
    ax.set_title("Tenure by Churn Status")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tenure_by_churn.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
    ax.set_title("Monthly Charges by Churn Status")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_charges_by_churn.png", dpi=150)
    plt.close(fig)


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    save_eda(df)

    df["Target"] = (df["Churn"].str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn", "Target"])
    y = df["Target"]

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = build_preprocessor(numeric_columns, categorical_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [8, 16, None],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }

    results = {}
    fitted_models = {}
    for name, (model, parameter_grid) in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        fitted_models[name] = search

        predictions = search.predict(X_test)
        probabilities = search.predict_proba(X_test)[:, 1]
        results[name] = {
            "best_parameters": search.best_params_,
            "cv_roc_auc": round(float(search.best_score_), 4),
            "test_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "test_precision": round(float(precision_score(y_test, predictions)), 4),
            "test_recall": round(float(recall_score(y_test, predictions)), 4),
            "test_f1": round(float(f1_score(y_test, predictions)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "classification_report": classification_report(
                y_test, predictions, target_names=["No churn", "Churn"]
            ),
        }

        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, display_labels=["No churn", "Churn"], ax=ax
        )
        ax.set_title(f"{name} - Confusion Matrix")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{name}_confusion_matrix.png", dpi=150)
        plt.close(fig)

    best_name = max(results, key=lambda name: results[name]["test_roc_auc"])
    best_search = fitted_models[best_name]
    joblib.dump(best_search.best_estimator_, OUTPUT_DIR / "churn_model.joblib")
    joblib.dump(
        {
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        },
        OUTPUT_DIR / "feature_metadata.joblib",
    )

    with (OUTPUT_DIR / "model_results.json").open("w", encoding="utf-8") as file:
        json.dump({"best_model": best_name, "models": results}, file, indent=2)

    print(f"Dataset shape: {df.shape}")
    print(f"Training rows: {len(X_train)}, test rows: {len(X_test)}")
    print(json.dumps({"best_model": best_name, "models": results}, indent=2))


if __name__ == "__main__":
    main()
