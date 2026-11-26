from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "artifacts" / "customer_churn_pipeline.joblib"


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------


@st.cache_resource
def load_model():
    """Load the trained customer churn pipeline."""
    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction", page_icon="📊", layout="wide"
)

st.title("Customer Churn Prediction")

st.write("Enter customer details to estimate the probability of customer churn.")


# --------------------------------------------------
# 4. Check and load model
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error("Trained model not found. Please run train.py first.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# 5. Customer information form
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior = st.selectbox("Senior citizen", ["No", "Yes"])

    partner = st.selectbox("Partner", ["No", "Yes"])

    dependents = st.selectbox("Dependents", ["No", "Yes"])

    tenure = st.slider("Tenure (months)", 0, 72, 12)

    phone = st.selectbox("Phone service", ["No", "Yes"])

    multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])


with col2:
    st.subheader("Services and Billing")

    online_security = st.selectbox(
        "Online security", ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])

    device_protection = st.selectbox(
        "Device protection", ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])

    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

    streaming_movies = st.selectbox(
        "Streaming movies", ["No", "Yes", "No internet service"]
    )

    paperless = st.selectbox("Paperless billing", ["No", "Yes"])

    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    monthly_charges = st.number_input(
        "Monthly charges", min_value=0.0, max_value=300.0, value=70.0
    )

    total_charges = st.number_input(
        "Total charges", min_value=0.0, max_value=10000.0, value=840.0
    )


# --------------------------------------------------
# 6. Predict customer churn
# --------------------------------------------------

if st.button("Predict churn", type="primary"):
    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    # Predict churn probability
    probability = model.predict_proba(customer)[0][1]

    # Classify using the default 0.5 threshold
    prediction = "Likely to churn" if probability >= 0.5 else "Likely to stay"

    st.divider()

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.warning(prediction)
    else:
        st.success(prediction)

    st.metric(label="Estimated churn probability", value=f"{probability:.1%}")

    st.progress(float(probability))

    st.caption("This is a model estimate, not a guarantee of future customer behavior.")
