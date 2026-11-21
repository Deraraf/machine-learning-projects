from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
OUTPUT = PROJECT_ROOT / "Customer_Churn_Machine_Learning_Assignment.pptx"


def add_title_slide(prs, title, subtitle):
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle


def add_bullet_slide(prs, title, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    body = slide.placeholders[1].text_frame
    body.clear()
    for index, bullet in enumerate(bullets):
        paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
        paragraph.text = bullet
        paragraph.level = 0
        paragraph.font.size = Pt(22)


def add_image_slide(prs, title, image_path):
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title
    slide.shapes.add_picture(
        str(image_path),
        Inches(1.0),
        Inches(1.5),
        width=Inches(8),
    )


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    add_title_slide(
        prs,
        "Customer Telecommunications Churn Prediction",
        "Machine Learning Process Assignment\nClassification Project",
    )
    add_bullet_slide(
        prs,
        "1. Problem Definition",
        [
            "Goal: predict whether a customer will discontinue telecommunications service.",
            "Type: binary supervised classification.",
            "Target: Churn = Yes or No.",
            "Business value: identify at-risk customers for proactive retention.",
            "Success criteria: maximize ROC-AUC and recall while monitoring precision.",
        ],
    )
    add_bullet_slide(
        prs,
        "2. Data Collection",
        [
            "Public Telco Customer Churn dataset.",
            "7,043 source records; 7,021 records after duplicate removal.",
            "Demographic, service, contract, and billing features.",
            "Customer identifier excluded because it has no predictive meaning.",
        ],
    )
    add_image_slide(
        prs,
        "3. Exploratory Data Analysis",
        ARTIFACTS / "churn_distribution.png",
    )
    add_bullet_slide(
        prs,
        "4. Data Cleaning and Preparation",
        [
            "Converted numeric fields and handled invalid numeric values.",
            "Removed duplicate records and the customer identifier.",
            "Applied median imputation and scaling to numeric features.",
            "Applied most-frequent imputation and one-hot encoding to categorical features.",
            "Used an 80/20 stratified train-test split.",
        ],
    )
    add_bullet_slide(
        prs,
        "5. Algorithm Selection",
        [
            "Compared Logistic Regression and Random Forest.",
            "Logistic Regression provides a simple, interpretable baseline.",
            "Random Forest captures nonlinear feature interactions.",
            "Used class weighting and ROC-AUC during tuning.",
        ],
    )
    add_bullet_slide(
        prs,
        "6. Training and Hyperparameter Tuning",
        [
            "Five-fold cross-validation.",
            "Grid search for regularization, tree depth, leaf size, and number of trees.",
            "Cross-validation ROC-AUC used for model selection.",
            "Best model: Random Forest.",
        ],
    )
    add_image_slide(
        prs,
        "7. Model Evaluation - Confusion Matrix",
        ARTIFACTS / "random_forest_confusion_matrix.png",
    )
    add_bullet_slide(
        prs,
        "8. Test Results",
        [
            "Accuracy: 75.30%",
            "Precision: 52.24%",
            "Recall: 78.23%",
            "F1-score: 62.65%",
            "ROC-AUC: 84.07%",
            "The model identifies approximately 78% of churners in the test set.",
        ],
    )
    add_bullet_slide(
        prs,
        "9. Deployment",
        [
            "Saved the final Random Forest pipeline with joblib.",
            "Created an interactive Streamlit application.",
            "The app accepts customer details and returns a churn probability.",
            "Launch with: uv run streamlit run app.py",
        ],
    )
    add_bullet_slide(
        prs,
        "10. Monitoring and Maintenance",
        [
            "Track predictive metrics over time.",
            "Monitor feature drift and target drift.",
            "Monitor response latency and prediction volume.",
            "Retrain when performance deteriorates or the data distribution changes.",
        ],
    )
    add_bullet_slide(
        prs,
        "Conclusion",
        [
            "Random Forest outperformed Logistic Regression on test ROC-AUC.",
            "The model provides useful churn recall for retention targeting.",
            "Further work: cost-sensitive thresholds, explainability, and larger datasets.",
        ],
    )

    prs.save(str(OUTPUT))
    print(OUTPUT)


if __name__ == "__main__":
    main()
