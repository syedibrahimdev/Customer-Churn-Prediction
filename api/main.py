import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability with SHAP explainability",
    version="1.0.0",
)

MODEL_PATH = "../models/churn_model.pkl"
FEATURES_PATH = "../models/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)
explainer = shap.TreeExplainer(model)


# Request schema: matches the RAW Telco dataset columns, before encoding
class CustomerInput(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    top_factors: list[dict]


def preprocess_input(customer: CustomerInput) -> pd.DataFrame:
    """
    Replicates the exact preprocessing from notebooks/02_preprocessing.ipynb
    so the API's encoding matches what the model was trained on.
    """
    data = customer.model_dump()
    df = pd.DataFrame([data])

    # Feature engineering (must match training exactly)
    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, np.inf],
        labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5yr+']
    )

    addon_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['num_addon_services'] = sum((df[col] == 'Yes').astype(int) for col in addon_services)

    df['avg_monthly_spend'] = df['TotalCharges'] / df['tenure'].replace(0, 1)

    # Binary encoding
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map(binary_map).astype(int)

    # One-hot encode multi-category columns
    multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod', 'tenure_group']
    df = pd.get_dummies(df, columns=multi_cat_cols)

    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


@app.get("/")
def root():
    return {"status": "ok", "message": "Customer Churn Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerInput):
    try:
        processed = preprocess_input(customer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    proba = model.predict_proba(processed)[0][1]
    will_churn = bool(proba >= 0.5)

    # SHAP explanation for this specific prediction
    shap_values = explainer.shap_values(processed)
    feature_impact = list(zip(feature_columns, shap_values[0]))
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = [
        {"feature": name, "impact": round(float(value), 4)}
        for name, value in feature_impact[:5]
    ]

    return PredictionResponse(
        churn_probability=round(float(proba), 4),
        will_churn=will_churn,
        top_factors=top_factors,
    )