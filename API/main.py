from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from SRC.predict import load_predictor


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="CarMarket AI API",
    description=(
        "REST API for CarMarket AI. "
        "Provides real-time car market price predictions "
        "using a trained machine learning model."
    ),
    version="1.0.0"
)


# ============================================================
# Load Model
# ============================================================

try:
    predictor = load_predictor()
    MODEL_LOADED = True
    MODEL_ERROR = None

except Exception as exc:
    predictor = None
    MODEL_LOADED = False
    MODEL_ERROR = str(exc)


# ============================================================
# Request / Response Models
# ============================================================

class PredictionRequest(BaseModel):
    """
    Request body for car price prediction.

    The features dictionary must contain the same
    feature names used during model training.
    """

    features: Dict[str, Any] = Field(
        ...,
        description=(
            "Car features used by the trained model. "
            "Feature names must match the training dataset."
        ),
        examples=[
            {
                "year": 2019,
                "km_driven": 50000,
                "fuel": "Diesel"
            }
        ]
    )


class PredictionResponse(BaseModel):
    predicted_price: float = Field(
        ...,
        description="Predicted car market price in INR."
    )

    currency: str = Field(
        default="INR",
        description="Currency of the prediction."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ============================================================
# Root Endpoint
# ============================================================

@app.get(
    "/",
    tags=["General"]
)
def root():
    """
    Basic API information.
    """

    return {
        "message": "Welcome to CarMarket AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict"
    }


# ============================================================
# Health Endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["General"]
)
def health():
    """
    Check whether the API and ML model are available.
    """

    return {
        "status": (
            "healthy"
            if MODEL_LOADED
            else "unhealthy"
        ),
        "model_loaded": MODEL_LOADED
    }


# ============================================================
# Prediction Endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict(request: PredictionRequest):
    """
    Predict the market price of a car.
    """

    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail=(
                "ML model could not be loaded. "
                f"Error: {MODEL_ERROR}"
            )
        )

    try:

        predicted_price = predictor.predict(
            request.features
        )

        return {
            "predicted_price": round(
                predicted_price,
                2
            ),
            "currency": "INR"
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(exc)}"
            )
        )