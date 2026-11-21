import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    return joblib.load("artifacts/churn_model.joblib")


st.set_page_config(page_title="Customer Churn Prediction", page_icon=":bar_chart:")
st.title("Customer Churn Prediction")
st.write("Enter customer details to estimate churn risk.")

model = load_model()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone service", ["No", "Yes"])
    multiple_lines = st.selectbox(
        "Multiple lines", ["No", "Yes", "No phone service"]
    )
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )

with col2:
    online_security = st.selectbox(
        "Online security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.selectbox(
        "Online backup", ["No", "Yes", "No internet service"]
    )
    device_protection = st.selectbox(
        "Device protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.selectbox(
        "Tech support", ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
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
    monthly_charges = st.number_input("Monthly charges", 0.0, 300.0, 70.0)
    total_charges = st.number_input("Total charges", 0.0, 10000.0, 840.0)

if st.button("Predict churn"):
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
    probability = model.predict_proba(customer)[0][1]
    prediction = "Likely to churn" if probability >= 0.5 else "Likely to stay"
    st.subheader(prediction)
    st.metric("Churn probability", f"{probability:.1%}")
