import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

OUTPUT_DIR = PROJECT_ROOT / "artifacts"

MODEL_PATH = OUTPUT_DIR / "customer_churn_pipeline.joblib"


# --------------------------------------------------
# 2. Load and clean dataset
# --------------------------------------------------


def load_data() -> pd.DataFrame:
    """Load and clean the customer churn dataset."""

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges from string to numeric.
    # Invalid or empty values become NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Replace missing TotalCharges with 0.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


# --------------------------------------------------
# 3. Generate EDA plots
# --------------------------------------------------


def save_eda_plots(df: pd.DataFrame) -> None:
    """Generate and save exploratory data analysis plots."""

    sns.set_theme(style="whitegrid")

    # 3.1 Customer churn distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(data=df, x="Churn", ax=axes[0])

    axes[0].set_title("Customer Churn Distribution")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Customers")

    churn_percentage = df["Churn"].value_counts(normalize=True).mul(100)

    sns.barplot(x=churn_percentage.index, y=churn_percentage.values, ax=axes[1])

    axes[1].set_title("Customer Churn Percentage")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Percentage (%)")

    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)

    plt.close(fig)

    # 3.2 Customer tenure by churn
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(data=df, x="Churn", y="tenure", ax=ax)

    ax.set_title("Customer Tenure by Churn Status")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Tenure (Months)")

    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "tenure_by_churn.png", dpi=150)

    plt.close(fig)

    # 3.3 Monthly charges by churn
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)

    ax.set_title("Monthly Charges by Churn Status")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges")

    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "monthly_charges_by_churn.png", dpi=150)

    plt.close(fig)

    print("\nEDA plots generated successfully!")


# --------------------------------------------------
# 4. Build machine learning pipeline
# --------------------------------------------------


def build_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Build the preprocessing and classification pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    return pipeline


# --------------------------------------------------
# 5. Save confusion matrix
# --------------------------------------------------


def save_confusion_matrix(y_test, y_pred) -> None:
    """Generate and save the model confusion matrix."""

    fig, ax = plt.subplots(figsize=(7, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No Churn", "Churn"], cmap="Blues", ax=ax
    )

    ax.set_title("Balanced Logistic Regression - Confusion Matrix")

    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)

    plt.close(fig)

    print("Confusion matrix saved successfully!")


# --------------------------------------------------
# 6. Save model evaluation results
# --------------------------------------------------


def save_model_results(y_test, y_pred, y_proba) -> None:
    """Calculate and save model evaluation results."""

    results = {
        "model": "Balanced Logistic Regression",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True
        ),
    }

    results_path = OUTPUT_DIR / "model_results.json"

    with results_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"Model results saved to: {results_path}")


# --------------------------------------------------
# 7. Main training function
# --------------------------------------------------


def main() -> None:

    # Create artifacts directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 7.1 Load dataset
    print("Loading dataset...")

    df = load_data()

    print(f"Dataset shape: {df.shape}")

    # 7.2 Generate EDA plots
    print("\nGenerating EDA plots...")

    save_eda_plots(df)

    # 7.3 Separate features and target
    X = df.drop(columns=["customerID", "Churn"])

    y = df["Churn"].map({"No": 0, "Yes": 1})

    # 7.4 Identify feature types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["str", "object", "category"]
    ).columns.tolist()

    print("\nFeature information:")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Total features: {X.shape[1]}")

    # 7.5 Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nDataset split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 7.6 Build pipeline
    print("\nBuilding Balanced Logistic Regression pipeline...")

    final_model = build_pipeline(numeric_features, categorical_features)

    # 7.7 Train model
    print("Training model...")

    final_model.fit(X_train, y_train)

    print("Model trained successfully!")

    # 7.8 Generate predictions
    y_pred = final_model.predict(X_test)

    y_proba = final_model.predict_proba(X_test)[:, 1]

    # 7.9 Evaluate model
    print("\nBalanced Logistic Regression Evaluation")
    print("---------------------------------------")

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("Precision:", precision_score(y_test, y_pred))

    print("Recall:", recall_score(y_test, y_pred))

    print("F1 Score:", f1_score(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    print("\nClassification Report:")

    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # 7.10 Save confusion matrix
    save_confusion_matrix(y_test, y_pred)

    # 7.11 Save model results
    save_model_results(y_test, y_pred, y_proba)

    # 7.12 Save trained pipeline
    joblib.dump(final_model, MODEL_PATH)

    print("\nFinal pipeline saved successfully!")
    print(f"Model location: {MODEL_PATH}")

    print("\nAll training tasks completed successfully!")


# --------------------------------------------------
# 8. Execute program
# --------------------------------------------------

if __name__ == "__main__":
    main()
