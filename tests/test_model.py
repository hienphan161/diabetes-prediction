import joblib
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from utils import engineer_features

# Define path to our model
MODEL_DIR = ROOT_DIR / "results/4_best_model"


def test_model_loads():
    """Test that model loads correctly."""
    model_data = joblib.load(MODEL_DIR / "best_model.joblib")
    assert "model" in model_data
    assert "features" in model_data
    assert model_data["model"] is not None


def test_model_healthy_patient():
    """Test prediction for a healthy patient (expected: no diabetes)."""
    model_data = joblib.load(MODEL_DIR / "best_model.joblib")
    
    # Healthy patient data
    data = {
        "gender": "Female",
        "age": 25.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 22.0,
        "HbA1c_level": 5.0,
        "blood_glucose_level": 100
    }
    
    df = pd.DataFrame([data])
    df = engineer_features(df)
    df = df[model_data["features"]]
    
    if model_data["prep"] is not None:
        X = model_data["prep"].transform(df)
    else:
        X = df.values
    
    pred = model_data["model"].predict(X)[0]
    assert pred == 0, f"Expected 0 (no diabetes), got {pred}"


def test_model_high_risk_patient():
    """Test prediction for a high-risk patient (expected: diabetes)."""
    model_data = joblib.load(MODEL_DIR / "best_model.joblib")
    
    # High-risk patient data
    data = {
        "gender": "Male",
        "age": 70.0,
        "hypertension": 1,
        "heart_disease": 1,
        "smoking_history": "former",
        "bmi": 35.0,
        "HbA1c_level": 8.5,
        "blood_glucose_level": 250
    }
    
    df = pd.DataFrame([data])
    df = engineer_features(df)
    df = df[model_data["features"]]
    
    if model_data["prep"] is not None:
        X = model_data["prep"].transform(df)
    else:
        X = df.values
    
    pred = model_data["model"].predict(X)[0]
    assert pred == 1, f"Expected 1 (diabetes), got {pred}"


def test_model_probability_range():
    """Test that probability is between 0 and 1."""
    model_data = joblib.load(MODEL_DIR / "best_model.joblib")
    
    data = {
        "gender": "Female",
        "age": 45.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 27.5,
        "HbA1c_level": 6.2,
        "blood_glucose_level": 140
    }
    
    df = pd.DataFrame([data])
    df = engineer_features(df)
    df = df[model_data["features"]]
    
    if model_data["prep"] is not None:
        X = model_data["prep"].transform(df)
    else:
        X = df.values
    
    prob = model_data["model"].predict_proba(X)[0][1]
    assert 0 <= prob <= 1, f"Probability {prob} is out of range [0, 1]"


def test_batch_prediction():
    """Test batch prediction with multiple patients."""
    model_data = joblib.load(MODEL_DIR / "best_model.joblib")
    
    patients = [
        {"gender": "Female", "age": 25.0, "hypertension": 0, "heart_disease": 0, 
         "smoking_history": "never", "bmi": 22.0, "HbA1c_level": 5.0, "blood_glucose_level": 100},
        {"gender": "Male", "age": 70.0, "hypertension": 1, "heart_disease": 1,
         "smoking_history": "former", "bmi": 35.0, "HbA1c_level": 8.5, "blood_glucose_level": 250},
    ]
    
    df = pd.DataFrame(patients)
    df = engineer_features(df)
    df = df[model_data["features"]]
    
    if model_data["prep"] is not None:
        X = model_data["prep"].transform(df)
    else:
        X = df.values
    
    preds = model_data["model"].predict(X)
    assert len(preds) == 2, f"Expected 2 predictions, got {len(preds)}"
