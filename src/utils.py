"""
Shared utilities for diabetes prediction.
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to match training pipeline.
    Must match 3_feature_engineering.py transformations.
    """
    df = df.copy()
    
    # Age features
    df["age_group"] = pd.cut(df["age"], bins=[0, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df["age_squared"] = df["age"] ** 2
    
    # BMI features
    df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    df["bmi_squared"] = df["bmi"] ** 2
    
    # Clinical thresholds
    df["hba1c_high"] = (df["HbA1c_level"] > 6.5).astype(int)
    df["glucose_high"] = (df["blood_glucose_level"] >= 200).astype(int)
    
    # Comorbidity
    df["comorbidity_score"] = df["hypertension"] + df["heart_disease"]
    df["has_comorbidity"] = (df["comorbidity_score"] > 0).astype(int)
    
    # Interactions
    df["age_bmi_interaction"] = df["age"] * df["bmi"]
    df["age_hba1c_interaction"] = df["age"] * df["HbA1c_level"]
    df["bmi_glucose_interaction"] = df["bmi"] * df["blood_glucose_level"]
    
    # Risk score
    df["risk_score"] = df["age_group"] + df["bmi_category"] + df["hba1c_high"] * 2 + df["glucose_high"] * 2 + df["comorbidity_score"]
    
    # Encode categorical
    for col in ["gender_Female", "gender_Male", "gender_Other"]:
        gender = col.split("_")[1]
        df[col] = (df["gender"] == gender).astype(int)
    
    smoking_cats = ["No Info", "current", "ever", "former", "never", "not current"]
    for cat in smoking_cats:
        df[f"smoking_{cat}"] = (df["smoking_history"] == cat).astype(int)
    
    # Drop original categorical
    df.drop(columns=["gender", "smoking_history"], inplace=True)
    
    return df


def get_risk_level(prob: float) -> str:
    """Convert probability to risk level."""
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    return "High"
