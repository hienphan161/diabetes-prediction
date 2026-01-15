"""
5: FastAPI for Diabetes Prediction Model Serving with OpenTelemetry Metrics
"""

import os
import sys
import joblib
import pandas as pd
from time import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# OpenTelemetry imports
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import start_http_server

# Add parent to path for imports (works in both local and Docker)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from utils import engineer_features, get_risk_level
except ImportError:
    # Fallback for local development
    ROOT_DIR = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from utils import engineer_features, get_risk_level

# =============================================================================
# Configuration
# =============================================================================
# Default path for local development, can be overridden by env var
DEFAULT_MODEL_PATH = Path(__file__).parent.parent.parent / "results/4_best_model/best_model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8099"))

# =============================================================================
# OpenTelemetry Metrics Setup
# =============================================================================
# Start Prometheus metrics server
start_http_server(port=METRICS_PORT, addr="0.0.0.0")

# Create resource with service name
resource = Resource(attributes={SERVICE_NAME: "diabetes-prediction-api"})

# Create Prometheus metric reader
reader = PrometheusMetricReader()

# Create meter provider
provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(provider)

# Create meter
meter = metrics.get_meter("diabetes_prediction", "1.0.0")

# Create metrics
request_counter = meter.create_counter(
    name="diabetes_prediction_requests_total",
    description="Total number of prediction requests",
    unit="1"
)

prediction_histogram = meter.create_histogram(
    name="diabetes_prediction_latency_seconds",
    description="Prediction request latency",
    unit="seconds"
)

diabetes_positive_counter = meter.create_counter(
    name="diabetes_positive_predictions_total",
    description="Total number of positive diabetes predictions",
    unit="1"
)

batch_size_histogram = meter.create_histogram(
    name="diabetes_prediction_batch_size",
    description="Batch prediction request sizes",
    unit="1"
)

# =============================================================================
# Pydantic Models
# =============================================================================
class PatientData(BaseModel):
    """Input schema for a single patient."""
    gender: str = Field(..., description="Gender: Male, Female, or Other")
    age: float = Field(..., ge=0, le=100, description="Age in years")
    hypertension: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    heart_disease: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    smoking_history: str = Field(..., description="never, current, former, ever, not current, No Info")
    bmi: float = Field(..., ge=10, le=100, description="Body Mass Index")
    HbA1c_level: float = Field(..., ge=3, le=15, description="HbA1c level")
    blood_glucose_level: int = Field(..., ge=50, le=400, description="Blood glucose level")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "age": 45.0,
                "hypertension": 0,
                "heart_disease": 0,
                "smoking_history": "never",
                "bmi": 27.5,
                "HbA1c_level": 6.2,
                "blood_glucose_level": 140
            }
        }

class PredictionResponse(BaseModel):
    """Output schema for prediction."""
    prediction: int = Field(..., description="0=No Diabetes, 1=Diabetes")
    probability: float = Field(..., description="Probability of diabetes")
    risk_level: str = Field(..., description="Low, Medium, or High risk")

class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""
    patients: list[PatientData]

class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: list[PredictionResponse]
    count: int

class ModelInfo(BaseModel):
    """Model metadata."""
    model_name: str
    preprocessor: str
    metrics: dict
    features: list[str]

# =============================================================================
# Model Loading
# =============================================================================
model_data = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_data
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    model_data = joblib.load(MODEL_PATH)
    print(f"Loaded model: {model_data['model_name']} with {model_data['prep_name']} preprocessor")
    yield
    model_data = None

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes risk based on patient health data",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Diabetes Prediction API"}

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    return ModelInfo(
        model_name=model_data["model_name"],
        preprocessor=model_data["prep_name"],
        metrics=model_data["metrics"],
        features=model_data["features"]
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientData):
    """Predict diabetes risk for a single patient."""
    start_time = time()
    labels = {"endpoint": "/predict", "method": "POST"}
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([patient.model_dump()])
        
        # Engineer features
        df = engineer_features(df)
        
        # Ensure column order matches training
        df = df[model_data["features"]]
        
        # Apply preprocessor if exists
        if model_data["prep"] is not None:
            X = model_data["prep"].transform(df)
        else:
            X = df.values
        
        # Predict
        pred = model_data["model"].predict(X)[0]
        prob = model_data["model"].predict_proba(X)[0][1]
        
        # Record metrics
        request_counter.add(1, labels)
        if pred == 1:
            diabetes_positive_counter.add(1, labels)
        
        # Record latency
        elapsed_time = time() - start_time
        prediction_histogram.record(elapsed_time, labels)
        
        return PredictionResponse(
            prediction=int(pred),
            probability=round(float(prob), 4),
            risk_level=get_risk_level(prob)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """Predict diabetes risk for multiple patients."""
    start_time = time()
    labels = {"endpoint": "/predict/batch", "method": "POST"}
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([p.model_dump() for p in request.patients])
        
        # Engineer features
        df = engineer_features(df)
        
        # Ensure column order
        df = df[model_data["features"]]
        
        # Apply preprocessor
        if model_data["prep"] is not None:
            X = model_data["prep"].transform(df)
        else:
            X = df.values
        
        # Predict
        preds = model_data["model"].predict(X)
        probs = model_data["model"].predict_proba(X)[:, 1]
        
        results = [
            PredictionResponse(
                prediction=int(p),
                probability=round(float(prob), 4),
                risk_level=get_risk_level(prob)
            )
            for p, prob in zip(preds, probs)
        ]
        
        # Record metrics
        batch_count = len(request.patients)
        request_counter.add(1, labels)
        batch_size_histogram.record(batch_count, labels)
        positive_count = sum(1 for p in preds if p == 1)
        if positive_count > 0:
            diabetes_positive_counter.add(positive_count, labels)
        
        # Record latency
        elapsed_time = time() - start_time
        prediction_histogram.record(elapsed_time, labels)
        
        return BatchPredictionResponse(predictions=results, count=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Run Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

