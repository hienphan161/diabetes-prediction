"""
1: Basic Exploratory Data Analysis for Diabetes Prediction Dataset

Column Definitions:
- gender: Biological sex (Male, Female, Other)
- age: Patient age (0-80 years). Diabetes more common in older adults.
- hypertension: 0=No, 1=Yes. High blood pressure condition.
- heart_disease: 0=No, 1=Yes. Associated with increased diabetes risk.
- smoking_history: not current, former, No Info, current, never, ever
- bmi: Body Mass Index (10.16-71.55). <18.5=underweight, 18.5-24.9=normal,
       25-29.9=overweight, >=30=obese. Higher BMI = higher diabetes risk.
- HbA1c_level: Average blood sugar over 2-3 months. >6.5% indicates diabetes.
- blood_glucose_level: Current blood glucose. High levels indicate diabetes.
- diabetes: Target variable. 0=No diabetes, 1=Has diabetes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
ROOT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# Load data
df = pd.read_csv(ROOT_DIR / "data/diabetes_prediction_dataset.csv")

# Add derived features based on medical thresholds
df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 24.9, 29.9, 100],
                            labels=["Underweight", "Normal", "Overweight", "Obese"])
df["hba1c_diabetic"] = (df["HbA1c_level"] > 6.5).map({True: "High (>6.5%)", False: "Normal (<=6.5%)"})

# =============================================================================
# 1. Basic Dataset Info
# =============================================================================
print("=" * 60)
print("BASIC DATASET INFORMATION")
print("=" * 60)
print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# =============================================================================
# 2. Missing Values
# =============================================================================
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
print(f"\nMissing values per column:\n{missing[missing > 0] if missing.sum() > 0 else 'No missing values'}")

# Check for 'No Info' in smoking_history
no_info_count = (df["smoking_history"] == "No Info").sum()
print(f"\n'No Info' in smoking_history: {no_info_count:,} ({no_info_count/len(df)*100:.2f}%)")

# =============================================================================
# 3. Statistical Summary
# =============================================================================
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY - NUMERICAL FEATURES")
print("=" * 60)
print(df.describe().round(2).to_string())

print("\n" + "=" * 60)
print("CATEGORICAL FEATURES VALUE COUNTS")
print("=" * 60)
for col in ["gender", "smoking_history"]:
    print(f"\n{col}:\n{df[col].value_counts()}")

print("\n" + "=" * 60)
print("BMI CATEGORIES (Medical Thresholds)")
print("=" * 60)
print("Underweight: <18.5 | Normal: 18.5-24.9 | Overweight: 25-29.9 | Obese: >=30")
print(f"\n{df['bmi_category'].value_counts()}")

print("\n" + "=" * 60)
print("HbA1c LEVEL (Diabetes Threshold: >6.5%)")
print("=" * 60)
print(f"\n{df['hba1c_diabetic'].value_counts()}")

# =============================================================================
# 4. Target Variable Distribution
# =============================================================================
print("\n" + "=" * 60)
print("TARGET VARIABLE (diabetes) DISTRIBUTION")
print("=" * 60)
target_counts = df["diabetes"].value_counts()
print(f"\n{target_counts}")
print(f"\nClass Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1 (No Diabetes : Diabetes)")

# =============================================================================
# 5. Visualizations
# =============================================================================

# Figure 1: Target distribution and categorical features
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Target distribution
ax = axes[0]
colors = ["#2ecc71", "#e74c3c"]
target_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
ax.set_title("Diabetes Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Diabetes (0=No, 1=Yes)")
ax.set_ylabel("Count")
ax.set_xticklabels(["No Diabetes", "Diabetes"], rotation=0)
for i, v in enumerate(target_counts):
    ax.text(i, v + 1000, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

# Gender distribution
ax = axes[1]
df["gender"].value_counts().plot(kind="bar", ax=ax, color=["#3498db", "#9b59b6"], edgecolor="black")
ax.set_title("Gender Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Gender")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Smoking history distribution
ax = axes[2]
df["smoking_history"].value_counts().plot(kind="bar", ax=ax, color="#f39c12", edgecolor="black")
ax.set_title("Smoking History Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Smoking History")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "1_categorical_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: Numerical features distributions
numerical_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    ax = axes[i]
    df[col].hist(bins=50, ax=ax, color="#3498db", edgecolor="black", alpha=0.7)
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"Mean: {df[col].mean():.2f}")
    ax.axvline(df[col].median(), color="green", linestyle="-", label=f"Median: {df[col].median():.2f}")
    ax.set_title(f"{col} Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "1_numerical_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
numerical_df = df[["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]]
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap="RdYlBu_r", center=0, fmt=".2f", ax=ax, linewidths=0.5)
ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4: Diabetes rate by categorical features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# By gender
ax = axes[0, 0]
diabetes_by_gender = df.groupby("gender")["diabetes"].mean() * 100
diabetes_by_gender.plot(kind="bar", ax=ax, color=["#3498db", "#9b59b6"], edgecolor="black")
ax.set_title("Diabetes Rate by Gender", fontsize=12, fontweight="bold")
ax.set_xlabel("Gender")
ax.set_ylabel("Diabetes Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(diabetes_by_gender):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

# By smoking history
ax = axes[0, 1]
diabetes_by_smoking = df.groupby("smoking_history")["diabetes"].mean() * 100
diabetes_by_smoking.sort_values(ascending=False).plot(kind="bar", ax=ax, color="#f39c12", edgecolor="black")
ax.set_title("Diabetes Rate by Smoking History", fontsize=12, fontweight="bold")
ax.set_xlabel("Smoking History")
ax.set_ylabel("Diabetes Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# By BMI category (medical thresholds)
ax = axes[1, 0]
bmi_order = ["Underweight", "Normal", "Overweight", "Obese"]
diabetes_by_bmi = df.groupby("bmi_category", observed=True)["diabetes"].mean().reindex(bmi_order) * 100
diabetes_by_bmi.plot(kind="bar", ax=ax, color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"], edgecolor="black")
ax.set_title("Diabetes Rate by BMI Category", fontsize=12, fontweight="bold")
ax.set_xlabel("BMI Category")
ax.set_ylabel("Diabetes Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(diabetes_by_bmi):
    ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10)

# By HbA1c threshold
ax = axes[1, 1]
diabetes_by_hba1c = df.groupby("hba1c_diabetic")["diabetes"].mean() * 100
diabetes_by_hba1c.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black")
ax.set_title("Diabetes Rate by HbA1c Level", fontsize=12, fontweight="bold")
ax.set_xlabel("HbA1c Level (Threshold: 6.5%)")
ax.set_ylabel("Diabetes Rate (%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(diabetes_by_hba1c):
    ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "1_diabetes_rate_by_category.png", dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
# 6. Save Summary Report
# =============================================================================
# Calculate BMI category stats
bmi_cat_counts = df["bmi_category"].value_counts()
hba1c_high_count = (df["HbA1c_level"] > 6.5).sum()

report = f"""
================================================================================
                    DIABETES PREDICTION - BASIC EDA REPORT
================================================================================

1. DATASET OVERVIEW
-------------------
- Total Records: {df.shape[0]:,}
- Total Features: {df.shape[1] - 3} (excluding target)
- Target Variable: diabetes (binary: 0/1)

2. FEATURES
-----------
Numerical (4):
  - age: {df['age'].min():.0f} to {df['age'].max():.0f} years (diabetes more common in older adults)
  - bmi: {df['bmi'].min():.2f} to {df['bmi'].max():.2f} (higher BMI = higher diabetes risk)
  - HbA1c_level: {df['HbA1c_level'].min():.1f} to {df['HbA1c_level'].max():.1f} (>6.5% indicates diabetes)
  - blood_glucose_level: {df['blood_glucose_level'].min():.0f} to {df['blood_glucose_level'].max():.0f}

Binary (2):
  - hypertension: {df['hypertension'].sum():,} cases ({df['hypertension'].mean()*100:.2f}%)
  - heart_disease: {df['heart_disease'].sum():,} cases ({df['heart_disease'].mean()*100:.2f}%)

Categorical (2):
  - gender: {df['gender'].nunique()} unique values (Male, Female, Other)
  - smoking_history: {df['smoking_history'].nunique()} categories (never, current, former, ever, not current, No Info)

3. BMI CATEGORIES (Medical Thresholds)
--------------------------------------
  - Underweight (<18.5): {bmi_cat_counts.get('Underweight', 0):,}
  - Normal (18.5-24.9): {bmi_cat_counts.get('Normal', 0):,}
  - Overweight (25-29.9): {bmi_cat_counts.get('Overweight', 0):,}
  - Obese (>=30): {bmi_cat_counts.get('Obese', 0):,}

4. HbA1c LEVEL (Diabetes Indicator)
-----------------------------------
  - Normal (<=6.5%): {len(df) - hba1c_high_count:,} ({(len(df) - hba1c_high_count)/len(df)*100:.2f}%)
  - High (>6.5%): {hba1c_high_count:,} ({hba1c_high_count/len(df)*100:.2f}%)

5. DATA QUALITY
---------------
- Missing Values: {df.isnull().sum().sum()}
- 'No Info' in smoking_history: {no_info_count:,} ({no_info_count/len(df)*100:.2f}%)

6. TARGET DISTRIBUTION
----------------------
- No Diabetes (0): {target_counts[0]:,} ({target_counts[0]/len(df)*100:.2f}%)
- Diabetes (1): {target_counts[1]:,} ({target_counts[1]/len(df)*100:.2f}%)
- Class Imbalance Ratio: {target_counts[0]/target_counts[1]:.2f}:1

7. KEY CORRELATIONS WITH DIABETES
---------------------------------
{corr_matrix['diabetes'].drop('diabetes').sort_values(ascending=False).to_string()}

================================================================================
Generated Plots:
- 1_categorical_distributions.png
- 1_numerical_distributions.png
- 1_correlation_heatmap.png
- 1_diabetes_rate_by_category.png (includes BMI category & HbA1c threshold)
================================================================================
"""

with open(RESULTS_DIR / "1_eda_report.txt", "w") as f:
    f.write(report)

print(report)
print("\nBasic EDA completed. Results saved to 'results/' folder.")

