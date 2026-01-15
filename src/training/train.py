"""
4: Model Training with Grid Search and MLflow Tracking
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, PowerTransformer, FunctionTransformer)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
ROOT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT_DIR / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "4_checkpoints"
BEST_MODEL_DIR = RESULTS_DIR / "4_best_model"
MLFLOW_DIR = RESULTS_DIR / "mlruns"
EXPERIMENT_NAME = "diabetes_prediction"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Create directories
for d in [CHECKPOINTS_DIR, BEST_MODEL_DIR, MLFLOW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Preprocessor Definitions
# =============================================================================
def log_transform(X):
    return np.log1p(np.abs(X))

PREPROCESSORS = {
    "none": None,
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
    "maxabs": MaxAbsScaler(),
    "power_yj": PowerTransformer(method="yeo-johnson"),
    "log_standard": Pipeline([("log", FunctionTransformer(log_transform)), ("scaler", StandardScaler())]),
    "log_minmax": Pipeline([("log", FunctionTransformer(log_transform)), ("scaler", MinMaxScaler())])
}

# =============================================================================
# Model Definitions
# =============================================================================
SCALING_PREPS = ["standard", "minmax", "robust", "maxabs", "power_yj", "log_standard", "log_minmax"]
NO_SCALING_PREPS = ["none"]

MODELS = {
    "logistic_regression": {
        "model": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "params": {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]},
        "preps": SCALING_PREPS
    },
    "decision_tree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "params": {"max_depth": [5, 10, 20, None], "min_samples_split": [2, 5, 10], "class_weight": [None, "balanced"]},
        "preps": NO_SCALING_PREPS
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {"n_estimators": [100, 200], "max_depth": [10, 20, None], "class_weight": [None, "balanced"]},
        "preps": NO_SCALING_PREPS
    },
    "extra_trees": {
        "model": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "params": {"n_estimators": [100, 200], "max_depth": [10, 20, None], "class_weight": [None, "balanced"]},
        "preps": NO_SCALING_PREPS
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1]},
        "preps": NO_SCALING_PREPS
    },
    "adaboost": {
        "model": AdaBoostClassifier(random_state=RANDOM_STATE),
        "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
        "preps": NO_SCALING_PREPS
    },
    "xgboost": {
        "model": XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss"),
        "params": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1], "scale_pos_weight": [1, 10]},
        "preps": NO_SCALING_PREPS
    },
    "lightgbm": {
        "model": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "params": {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1], "class_weight": [None, "balanced"]},
        "preps": NO_SCALING_PREPS
    },
    "svm": {
        "model": SVC(random_state=RANDOM_STATE, probability=True),
        "params": {"C": [0.1, 1, 10], "kernel": ["rbf"], "class_weight": [None, "balanced"]},
        "preps": SCALING_PREPS
    },
    "knn": {
        "model": KNeighborsClassifier(n_jobs=-1),
        "params": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]},
        "preps": SCALING_PREPS
    },
    "naive_bayes": {
        "model": GaussianNB(),
        "params": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        "preps": SCALING_PREPS
    },
    "mlp": {
        "model": MLPClassifier(random_state=RANDOM_STATE, max_iter=500, early_stopping=True),
        "params": {"hidden_layer_sizes": [(64,), (128,), (64, 32)], "alpha": [0.0001, 0.001], "learning_rate_init": [0.001, 0.01]},
        "preps": SCALING_PREPS
    }
}

# =============================================================================
# Helper Functions
# =============================================================================
def load_data():
    """Load and split the processed dataset."""
    df = pd.read_csv(RESULTS_DIR / "3_processed_data.csv")
    X, y = df.drop(columns=["diabetes"]), df["diabetes"]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

def preprocess_data(X_train, X_test):
    """Apply all preprocessors to the data."""
    data = {}
    for name, prep in PREPROCESSORS.items():
        try:
            if prep is None:
                data[name] = {"train": X_train.values, "test": X_test.values, "prep": None}
            else:
                data[name] = {"train": prep.fit_transform(X_train), "test": prep.transform(X_test), "prep": prep}
        except Exception as e:
            print(f"  Preprocessor {name} failed: {e}")
    return data

def evaluate(model, X_test, y_test):
    """Calculate evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }

def log_to_mlflow(name, prep_name, params, best_params, cv_score, metrics, model, is_best=False):
    """Log a training run to MLflow."""
    run_name = "BEST_MODEL" if is_best else f"{name}_{prep_name}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", name)
        mlflow.set_tag("preprocessor", prep_name)
        if is_best:
            mlflow.set_tag("is_best", "true")
        mlflow.log_param("model_name", name)
        mlflow.log_param("preprocessor", prep_name)
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("cv_f1", cv_score)
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(f"test_{k}", v)
        mlflow.sklearn.log_model(model, "model")
        return run.info.run_id

def train_single(name, config, prep_data, y_train, y_test):
    """Train a model with a single preprocessor."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(clone(config["model"]), config["params"], cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(prep_data["train"], y_train)
    metrics = evaluate(grid.best_estimator_, prep_data["test"], y_test)
    return grid.best_estimator_, grid.best_params_, grid.best_score_, metrics

def train_model(name, config, all_prep_data, y_train, y_test):
    """Train a model with all applicable preprocessors, return best result."""
    checkpoint = CHECKPOINTS_DIR / f"{name}.joblib"
    
    # Skip if checkpoint exists
    if checkpoint.exists():
        print(f"[SKIP] {name}")
        return joblib.load(checkpoint)
    
    print(f"[TRAIN] {name}")
    best = {"f1": -1}
    
    for prep_name in config["preps"]:
        if prep_name not in all_prep_data:
            continue
        try:
            model, params, cv_score, metrics = train_single(name, config, all_prep_data[prep_name], y_train, y_test)
            print(f"  {prep_name}: CV={cv_score:.4f}, Test F1={metrics['f1']:.4f}")
            
            # Log to MLflow
            run_id = log_to_mlflow(name, prep_name, config["params"], params, cv_score, metrics, model)
            
            if metrics["f1"] > best["f1"]:
                best = {
                    "model": model, "params": params, "cv_score": cv_score,
                    "metrics": metrics, "prep_name": prep_name,
                    "prep": all_prep_data[prep_name]["prep"], "run_id": run_id, "f1": metrics["f1"]
                }
        except Exception as e:
            print(f"  {prep_name}: FAILED - {e}")
    
    if best["f1"] < 0:
        return None
    
    # Save checkpoint
    result = {k: v for k, v in best.items() if k != "f1"}
    joblib.dump(result, checkpoint)
    print(f"  Best: {best['prep_name']} (F1={best['f1']:.4f})")
    return result

# =============================================================================
# Main
# =============================================================================
def main():
    # Setup MLflow
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR.absolute()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow: {MLFLOW_DIR.absolute()}")
    
    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution: {y_train.value_counts().to_dict()}")
    
    # Preprocess
    print("\nPreprocessing...")
    prep_data = preprocess_data(X_train, X_test)
    print(f"Preprocessors ready: {list(prep_data.keys())}")
    
    # Train all models
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    results = {}
    for name, config in MODELS.items():
        results[name] = train_model(name, config, prep_data, y_train, y_test)
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    comparison = []
    for name, data in results.items():
        if data:
            m = data["metrics"]
            comparison.append({
                "Model": name, "Preprocessor": data["prep_name"],
                "Accuracy": m["accuracy"], "Precision": m["precision"],
                "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"]
            })
    
    df = pd.DataFrame(comparison).sort_values("F1", ascending=False)
    print("\n" + df.to_string(index=False))
    
    # Save best model
    best_name = df.iloc[0]["Model"]
    best_data = results[best_name]
    print(f"\nBest: {best_name} + {best_data['prep_name']} (F1={best_data['metrics']['f1']:.4f})")
    
    # Log best to MLflow
    best_run_id = log_to_mlflow(
        best_name, best_data["prep_name"], {}, best_data["params"],
        best_data["cv_score"], best_data["metrics"], best_data["model"], is_best=True
    )
    
    # Save best model file
    best_info = {
        "model_name": best_name, "model": best_data["model"],
        "params": best_data["params"], "metrics": best_data["metrics"],
        "prep_name": best_data["prep_name"], "prep": best_data["prep"],
        "features": list(X_train.columns), "run_id": best_run_id
    }
    joblib.dump(best_info, BEST_MODEL_DIR / "best_model.joblib")
    
    # Save comparison CSV
    df.to_csv(RESULTS_DIR / "4_model_comparison.csv", index=False)
    
    # Save report
    report = f"""
{'='*70}
MODEL TRAINING REPORT
{'='*70}

Data: {X_train.shape[0]:,} train, {X_test.shape[0]:,} test, {X_train.shape[1]} features
Preprocessors: {', '.join(prep_data.keys())}
Models: {len(MODELS)}

{df.to_string(index=False)}

Best Model: {best_name}
  Preprocessor: {best_data['prep_name']}
  Parameters: {best_data['params']}
  CV F1: {best_data['cv_score']:.4f}
  Test Metrics: {best_data['metrics']}
  MLflow Run: {best_run_id}

MLflow UI: mlflow ui --backend-store-uri file://{MLFLOW_DIR.absolute()}
{'='*70}
"""
    (RESULTS_DIR / "4_training_report.txt").write_text(report)
    print(report)

if __name__ == "__main__":
    main()

