# Diabetes Prediction

Machine learning project to predict diabetes risk based on patient health data.

## Project Structure

```
diabetes-prediction/
├── data/                          # Raw data
│   └── diabetes_prediction_dataset.csv
├── src/                           # Source code
│   ├── eda/                       # Exploratory Data Analysis
│   │   ├── basic_eda.py          # Basic EDA (distributions, correlations)
│   │   └── advanced_eda.py       # Advanced EDA (statistical tests, interactions)
│   ├── preprocessing/             # Data preprocessing
│   │   └── feature_engineering.py # Feature engineering pipeline
│   ├── training/                  # Model training
│   │   └── train.py              # Training with GridSearch + MLflow
│   ├── api/                       # API serving
│   │   └── main.py               # FastAPI application
│   └── utils.py                   # Shared utilities
├── tests/                         # Unit tests
│   └── test_model.py
├── results/                       # Generated outputs
│   ├── 1_*.png/txt/md            # Basic EDA results
│   ├── 2_*.png/txt/md            # Advanced EDA results
│   ├── 3_*.csv/txt               # Feature engineering outputs
│   ├── 4_checkpoints/            # Model checkpoints
│   ├── 4_best_model/             # Best model
│   └── mlruns/                   # MLflow tracking
├── infra/                         # Infrastructure
│   ├── ansible/                  # Ansible playbooks
│   ├── terraform/                # Terraform configs
│   ├── helm_charts/              # Kubernetes Helm charts
│   └── prometheus/               # Monitoring configs
├── Dockerfile                     # Docker configuration
└── requirements.txt               # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run EDA

```bash
python src/eda/basic_eda.py
python src/eda/advanced_eda.py
```

### 3. Feature Engineering

```bash
python src/preprocessing/feature_engineering.py
```

### 4. Train Models

```bash
python src/training/train.py
```

### 5. Run API

```bash
# Local
uvicorn src.api.main:app --reload --port 8000

# Or using Python
python -m src.api.main
```

### 6. Run Tests

```bash
pytest tests/ -v
```

## Docker

```bash
# Build
docker build -t diabetes-prediction-api .

# Run
docker run -p 30000:30000 diabetes-prediction-api
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/model/info` | Model metadata |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |

### Example Request

```bash
curl -X POST http://localhost:30000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "age": 45.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 27.5,
    "HbA1c_level": 6.2,
    "blood_glucose_level": 140
  }'
```

## MLflow

View experiment tracking:

```bash
mlflow ui --backend-store-uri file://$(pwd)/results/mlruns
```

Then open http://localhost:5000

## Model Performance

Best model metrics are saved in `results/4_training_report.txt` and `results/4_model_comparison.csv`.
