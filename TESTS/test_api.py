from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# ============================================================
# Load Feature Dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "DATA"
    / "PROCESSED"
    / "cars_features.csv"
)


@pytest.fixture
def sample_features():

    if not FEATURE_FILE.exists():

        pytest.skip(
            "cars_features.csv not found."
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    target_candidates = [
        "selling_price",
        "price",
        "resale_price"
    ]

    target = next(
        (
            column
            for column in target_candidates
            if column in df.columns
        ),
        None
    )

    if target is None:

        pytest.skip(
            "Price target column not found."
        )

    features = {}

    for column in df.columns:

        if column == target:
            continue

        if column == "log_price":
            continue

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            features[column] = float(
                df[column].median()
            )

        else:

            mode = (
                df[column]
                .dropna()
                .astype(str)
                .mode()
            )

            if len(mode) > 0:
                features[column] = mode.iloc[0]

            else:
                features[column] = "Unknown"

    return features


# ============================================================
# Test Root Endpoint
# ============================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert "version" in data
    assert "docs" in data


# ============================================================
# Test Health Endpoint
# ============================================================

def test_health():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_loaded" in data


# ============================================================
# Test Prediction Endpoint
# ============================================================

def test_prediction(
    sample_features
):

    response = client.post(
        "/predict",
        json={
            "features": sample_features
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_price" in data
    assert "currency" in data

    assert isinstance(
        data["predicted_price"],
        (int, float)
    )

    assert data["predicted_price"] >= 0

    assert data["currency"] == "INR"


# ============================================================
# Test Missing Features
# ============================================================

def test_prediction_missing_features():

    response = client.post(
        "/predict",
        json={
            "features": {}
        }
    )

    assert response.status_code == 422


# ============================================================
# Test Invalid Endpoint
# ============================================================

def test_invalid_endpoint():

    response = client.get(
        "/does-not-exist"
    )

    assert response.status_code == 404