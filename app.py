import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"  # change this after deploying the API

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.markdown("<h1 style='text-align: center;'>📉 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict churn risk and see exactly why, powered by SHAP explainability.</p>", unsafe_allow_html=True)
st.markdown("---")


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_URL}/", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


if not check_api_health():
    st.error(
        "⚠️ Cannot reach the prediction API. Make sure the FastAPI backend is running "
        f"at {API_URL} (locally: `uvicorn main:app --reload --port 8000` inside the `api/` folder)."
    )
    st.stop()


with st.form("churn_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

    submitted = st.form_submit_button("🔮 Predict Churn Risk", type="primary")


if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
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

    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {e}")
        st.stop()

    st.markdown("---")
    proba = result["churn_probability"]
    will_churn = result["will_churn"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Probability", f"{proba:.1%}")
    with col2:
        if will_churn:
            st.error("⚠️ High Risk — Likely to Churn")
        else:
            st.success("✅ Low Risk — Likely to Stay")

    st.progress(proba)

    st.subheader("🔍 Why this prediction? (Top contributing factors)")
    factors_df = pd.DataFrame(result["top_factors"])
    factors_df["direction"] = factors_df["impact"].apply(lambda x: "Increases Risk" if x > 0 else "Decreases Risk")
    factors_df["abs_impact"] = factors_df["impact"].abs()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d62728" if x > 0 else "#2ca02c" for x in factors_df["impact"]]
    ax.barh(factors_df["feature"], factors_df["impact"], color=colors)
    ax.set_xlabel("SHAP Impact (+ increases churn risk, - decreases it)")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(factors_df[["feature", "impact", "direction"]], use_container_width=True, hide_index=True)