"""
app.py
------
Streamlit app for live customer churn predictions using the trained model.

Run locally:
    streamlit run app.py

Deploy free:
    1. Push this whole folder to a public GitHub repo.
    2. Go to https://share.streamlit.io, sign in with GitHub.
    3. Click 'New app', select your repo, set main file to 'app.py'.
    4. Deploy — you'll get a live public URL to put on your resume.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path("models")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_DIR / "churn_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    with open(MODEL_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    metrics = {}
    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    return model, scaler, encoders, feature_names, metrics


def main():
    st.title("📉 Customer Churn Predictor")
    st.write(
        "Enter a customer's details below to predict whether they're likely to churn."
    )

    try:
        model, scaler, encoders, feature_names, metrics = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model files not found. Run `python train_model.py` first to "
            "train and save the model."
        )
        st.stop()

    if metrics:
        best_model = metrics.get("best_model", "N/A")
        f1 = metrics.get("results", {}).get(best_model, {}).get("f1_score")
        auc = metrics.get("results", {}).get(best_model, {}).get("roc_auc")
        with st.expander("ℹ️ Model info"):
            st.write(f"**Best model:** {best_model}")
            if f1 is not None:
                st.write(f"**F1-score:** {f1:.3f}")
            if auc is not None:
                st.write(f"**ROC-AUC:** {auc:.3f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )

    with col2:
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0
        )

    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, value=float(tenure) * monthly_charges
    )

    raw_input = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    if st.button("Predict Churn", type="primary", use_container_width=True):
        input_df = pd.DataFrame([raw_input])

        # Apply the same label encoders used during training
        for col, encoder in encoders.items():
            if col == "Churn":
                continue
            if col in input_df.columns:
                # Handle unseen categories gracefully
                val = input_df.at[0, col]
                if val in encoder.classes_:
                    input_df[col] = encoder.transform([val])
                else:
                    input_df[col] = 0

        input_df = input_df[feature_names]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Likely to churn — probability: {probability:.1%}")
        else:
            st.success(f"✅ Likely to stay — churn probability: {probability:.1%}")

        st.progress(float(probability))


if __name__ == "__main__":
    main()
