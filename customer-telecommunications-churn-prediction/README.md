# Customer Telecommunications Churn Prediction

## Problem Definition

This project predicts whether a telecommunications customer is likely to cancel their service. It is a binary classification problem:

- **Positive class:** customer churned (`Churn = Yes`)
- **Negative class:** customer stayed (`Churn = No`)

The business goal is to identify at-risk customers early so that retention campaigns can be targeted. Success is primarily measured using ROC-AUC and recall, because missed churners can represent lost revenue. Accuracy, precision, F1-score, and the confusion matrix are also reported.

## Dataset

The project uses the public Telco Customer Churn dataset. The source file is stored at `data/Telco-Customer-Churn.csv.xlsx`.

- Rows: 7,021 after removing duplicates
- Features: customer demographic, service, account, and billing attributes
- Target: `Churn`
- Unit: one row per customer

The input file is an Excel container with comma-separated rows. `train.py` normalizes this format into a standard table.

## Data Preparation

1. Normalized the dataset into named columns.
2. Removed the customer identifier and duplicate rows.
3. Converted numeric fields into numeric types.
4. Split data into 80% training and 20% test sets using stratification.
5. Built separate preprocessing pipelines for numeric and categorical features.
6. Used median imputation and standard scaling for numeric features.
7. Used most-frequent imputation and one-hot encoding for categorical features.

## Models

Two supervised models were compared:

- Logistic Regression
- Random Forest

Five-fold cross-validation with ROC-AUC as the optimization metric was used to tune hyperparameters.

## Results

The Random Forest model was selected as the best test-set model:

| Metric | Result |
|---|---:|
| Accuracy | 0.7530 |
| Precision | 0.5224 |
| Recall | 0.7823 |
| F1-score | 0.6265 |
| ROC-AUC | 0.8407 |

The model prioritizes recall and can identify approximately 78% of churners in the test set.

## Generated Artifacts

Running `python train.py` creates:

- `artifacts/churn_model.joblib`: trained deployment model
- `artifacts/model_results.json`: metrics and tuned parameters
- `artifacts/*_confusion_matrix.png`: test confusion matrices
- `artifacts/churn_distribution.png`: churn class distribution
- `artifacts/tenure_by_churn.png`: tenure relationship
- `artifacts/monthly_charges_by_churn.png`: billing relationship

## Deployment

Start the Streamlit interface with:

```bash
uv run streamlit run app.py
```

Enter customer attributes in the browser and receive a churn probability.

## Presentation

Generate the PowerPoint with:

```bash
python create_presentation.py
```

The presentation includes the problem, dataset, EDA, preparation, algorithms, tuning, evaluation, deployment, and monitoring sections.

## Jupyter Notebook

Open and run the complete ML process in Jupyter:

```bash
uv run jupyter notebook customer_churn_assignment.ipynb
```

The notebook contains Markdown explanations and executable code for problem definition, data collection, EDA, cleaning, feature engineering, train-test splitting, model development, hyperparameter tuning, testing, deployment guidance, and monitoring.

## Monitoring and Maintenance

In production, monitor:

- ROC-AUC, precision, recall, and F1-score
- Prediction volume and response latency
- Feature and target distributions
- Data drift
- Changes in customer behavior and retention outcomes

Retraining should be triggered periodically or when sustained performance degradation or drift is detected.