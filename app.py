"""
Task 1: FastAPI Deployment
REST API for Lead Recommendation System
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize FastAPI
app = FastAPI(
    title="Lead Recommendation API",
    description="AI-Powered Lead Scoring and Recommendation System",
    version="1.0.0",
)

# Load model
try:
    model = joblib.load("notebooks/saved_models/best_lead_model.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


# Request Schema - EXACT 4 FEATURES
class LeadInput(BaseModel):
    website_exists: int
    contact_form: int
    services_count: int
    country_score: int


# Response Schema
class LeadOutput(BaseModel):
    prediction: str
    recommendation: str
    timestamp: str


# Recommendation mapping
recommendations = {
    "High": "🔴 Priority Lead - Contact within 24 hours",
    "Medium": "🟡 Potential Opportunity - Add to nurture campaign",
    "Low": "🟢 Low Priority - Monitor for future engagement",
}

# Numeric to String mapping
prediction_map = {0: "High", 1: "Medium", 2: "Low"}


@app.get("/")
def root():
    return {
        "message": "Lead Recommendation API is running!",
        "endpoints": {
            "GET /health": "Check API status",
            "POST /predict": "Predict lead quality",
        },
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=LeadOutput)
def predict_lead(lead: LeadInput):
    """
    Predict lead quality based on input features
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        features = pd.DataFrame(
            [
                [
                    lead.website_exists,
                    lead.contact_form,
                    lead.services_count,
                    lead.country_score,
                ]
            ],
            columns=[
                "website_exists",
                "contact_form",
                "services_count",
                "country_score",
            ],
        )

        # Make prediction
        prediction_raw = model.predict(features)[0]

        # Convert numeric to string if needed
        if isinstance(prediction_raw, (int, np.integer, np.int64)):
            prediction = prediction_map.get(prediction_raw, "Medium")
        else:
            prediction = str(prediction_raw)

        recommendation = recommendations.get(prediction, "Unknown")

        # Log prediction
        logging.info(f"Prediction: {prediction} | Features: {features.values.tolist()}")

        return LeadOutput(
            prediction=prediction,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Run with: uvicorn app:app --reload
