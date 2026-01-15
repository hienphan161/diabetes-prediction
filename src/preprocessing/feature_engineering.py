"""
3: Feature Engineering for Diabetes Prediction Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Setup
ROOT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(ROOT_DIR / "data/diabetes_prediction_dataset.csv")
print(f"Original shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# =============================================================================
# 1. Age-based Features
# =============================================================================
# Age groups based on EDA findings
df["age_group"] = pd.cut(df["age"], bins=[0, 20, 40, 60, 80, 100],
                         labels=[0, 1, 2, 3, 4]).astype(int)

# Age squared (captures non-linear relationship)
df["age_squared"] = df["age"] ** 2

# =============================================================================
# 2. BMI-based Features
# =============================================================================
# BMI categories (medical thresholds)
df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100],
                            labels=[0, 1, 2, 3]).astype(int)
# 0=Underweight, 1=Normal, 2=Overweight, 3=Obese

# BMI squared
df["bmi_squared"] = df["bmi"] ** 2

# =============================================================================
# 3. HbA1c-based Features
# =============================================================================
# HbA1c diabetic threshold (>6.5% indicates diabetes)
df["hba1c_high"] = (df["HbA1c_level"] > 6.5).astype(int)

# =============================================================================
# 4. Blood Glucose Features
# =============================================================================
# High blood glucose flag (>=200 is high)
df["glucose_high"] = (df["blood_glucose_level"] >= 200).astype(int)

# =============================================================================
# 5. Comorbidity Features
# =============================================================================
# Comorbidity score (sum of hypertension and heart_disease)
df["comorbidity_score"] = df["hypertension"] + df["heart_disease"]

# Has any comorbidity flag
df["has_comorbidity"] = (df["comorbidity_score"] > 0).astype(int)

# =============================================================================
# 6. Interaction Features
# =============================================================================
# Age x BMI interaction (high-risk combination from heatmap)
df["age_bmi_interaction"] = df["age"] * df["bmi"]

# Age x HbA1c interaction
df["age_hba1c_interaction"] = df["age"] * df["HbA1c_level"]

# BMI x Glucose interaction
df["bmi_glucose_interaction"] = df["bmi"] * df["blood_glucose_level"]

# =============================================================================
# 7. Risk Score Feature
# =============================================================================
# Composite risk score based on EDA findings
df["risk_score"] = (
    df["age_group"] +
    df["bmi_category"] +
    df["hba1c_high"] * 2 +
    df["glucose_high"] * 2 +
    df["comorbidity_score"]
)

# =============================================================================
# 8. Encode Categorical Variables
# =============================================================================
# Gender encoding (one-hot)
gender_dummies = pd.get_dummies(df["gender"], prefix="gender", drop_first=False)
df = pd.concat([df, gender_dummies], axis=1)

# Smoking history encoding (one-hot)
smoking_dummies = pd.get_dummies(df["smoking_history"], prefix="smoking", drop_first=False)
df = pd.concat([df, smoking_dummies], axis=1)

# Drop original categorical columns
df.drop(columns=["gender", "smoking_history"], inplace=True)

# =============================================================================
# 9. Feature Summary
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)

# Separate features by type
original_numerical = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
original_binary = ["hypertension", "heart_disease"]
engineered_categorical = ["age_group", "bmi_category"]
engineered_binary = ["hba1c_high", "glucose_high", "has_comorbidity"]
engineered_numerical = ["age_squared", "bmi_squared", "comorbidity_score",
                        "age_bmi_interaction", "age_hba1c_interaction",
                        "bmi_glucose_interaction", "risk_score"]
encoded_features = list(gender_dummies.columns) + list(smoking_dummies.columns)

print(f"\nOriginal Numerical ({len(original_numerical)}): {original_numerical}")
print(f"Original Binary ({len(original_binary)}): {original_binary}")
print(f"Engineered Categorical ({len(engineered_categorical)}): {engineered_categorical}")
print(f"Engineered Binary ({len(engineered_binary)}): {engineered_binary}")
print(f"Engineered Numerical ({len(engineered_numerical)}): {engineered_numerical}")
print(f"Encoded Features ({len(encoded_features)}): {encoded_features}")

print(f"\nFinal shape: {df.shape}")
print(f"Final columns ({len(df.columns)}): {list(df.columns)}")

# =============================================================================
# 10. Save Processed Data
# =============================================================================
output_path = RESULTS_DIR / "3_processed_data.csv"
df.to_csv(output_path, index=False)
print(f"\nProcessed data saved to: {output_path}")

# Save feature list for reference
feature_info = f"""
================================================================================
                    FEATURE ENGINEERING - REFERENCE
================================================================================

ORIGINAL FEATURES (kept):
- age: Patient age (0-80)
- bmi: Body Mass Index
- HbA1c_level: Glycated hemoglobin level
- blood_glucose_level: Blood glucose level
- hypertension: 0/1
- heart_disease: 0/1

ENGINEERED FEATURES:

1. Age-based:
   - age_group: 0=0-20, 1=21-40, 2=41-60, 3=61-80, 4=80+
   - age_squared: age^2 (non-linear relationship)

2. BMI-based:
   - bmi_category: 0=Underweight, 1=Normal, 2=Overweight, 3=Obese
   - bmi_squared: bmi^2 (non-linear relationship)

3. Clinical Thresholds:
   - hba1c_high: 1 if HbA1c > 6.5% (diabetes indicator)
   - glucose_high: 1 if blood_glucose >= 200

4. Comorbidity:
   - comorbidity_score: hypertension + heart_disease (0-2)
   - has_comorbidity: 1 if any comorbidity present

5. Interactions:
   - age_bmi_interaction: age * bmi
   - age_hba1c_interaction: age * HbA1c_level
   - bmi_glucose_interaction: bmi * blood_glucose_level

6. Risk Score:
   - risk_score: Composite score based on multiple risk factors

ENCODED FEATURES:
- gender_Female, gender_Male, gender_Other
- smoking_No Info, smoking_current, smoking_ever, smoking_former, 
  smoking_never, smoking_not current

TARGET: diabetes (0/1)

Total Features: {len(df.columns) - 1} (excluding target)
================================================================================
"""

with open(RESULTS_DIR / "3_feature_reference.txt", "w") as f:
    f.write(feature_info)

print(feature_info)
print("\nFeature engineering completed.")

