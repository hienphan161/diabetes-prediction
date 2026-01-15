"""
2: Advanced Exploratory Data Analysis for Diabetes Prediction Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Setup
ROOT_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# Load data
df = pd.read_csv(ROOT_DIR / "data/diabetes_prediction_dataset.csv")

# =============================================================================
# 1. Statistical Tests - Feature Significance
# =============================================================================
print("=" * 70)
print("STATISTICAL TESTS - FEATURE SIGNIFICANCE")
print("=" * 70)

# Chi-square tests for categorical features
print("\nChi-Square Tests (Categorical vs Diabetes):")
print("-" * 50)
categorical_cols = ["gender", "smoking_history", "hypertension", "heart_disease"]
chi2_results = []
for col in categorical_cols:
    contingency = pd.crosstab(df[col], df["diabetes"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    chi2_results.append({"Feature": col, "Chi2": chi2, "p-value": p_value, "Significant": p_value < 0.05})
    print(f"{col:20} Chi2={chi2:12.2f}  p-value={p_value:.2e}  {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")

# T-tests for numerical features
print("\nT-Tests (Numerical: Diabetes vs No Diabetes):")
print("-" * 50)
numerical_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
ttest_results = []
for col in numerical_cols:
    group0 = df[df["diabetes"] == 0][col]
    group1 = df[df["diabetes"] == 1][col]
    t_stat, p_value = stats.ttest_ind(group0, group1)
    effect_size = (group1.mean() - group0.mean()) / df[col].std()  # Cohen's d
    ttest_results.append({"Feature": col, "t-stat": t_stat, "p-value": p_value, "Effect Size": effect_size})
    print(f"{col:20} t={t_stat:10.2f}  p-value={p_value:.2e}  Cohen's d={effect_size:.3f}")

# =============================================================================
# 2. Distribution Comparison by Diabetes Status
# =============================================================================
print("\n" + "=" * 70)
print("DISTRIBUTION COMPARISON BY DIABETES STATUS")
print("=" * 70)
for col in numerical_cols:
    no_diab = df[df["diabetes"] == 0][col]
    diab = df[df["diabetes"] == 1][col]
    print(f"\n{col}:")
    print(f"  No Diabetes: mean={no_diab.mean():.2f}, median={no_diab.median():.2f}, std={no_diab.std():.2f}")
    print(f"  Diabetes:    mean={diab.mean():.2f}, median={diab.median():.2f}, std={diab.std():.2f}")

# =============================================================================
# 3. Outlier Analysis
# =============================================================================
print("\n" + "=" * 70)
print("OUTLIER ANALYSIS (IQR Method)")
print("=" * 70)
outlier_summary = []
for col in numerical_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_pct = len(outliers) / len(df) * 100
    outlier_summary.append({"Feature": col, "Outliers": len(outliers), "Percentage": outlier_pct})
    print(f"{col:20} Outliers: {len(outliers):6,} ({outlier_pct:.2f}%)  Range: [{lower:.2f}, {upper:.2f}]")

# =============================================================================
# 4. Feature Interactions
# =============================================================================
print("\n" + "=" * 70)
print("FEATURE INTERACTIONS - DIABETES RATE")
print("=" * 70)

# Age groups
df["age_group"] = pd.cut(df["age"], bins=[0, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "80+"])
age_diab = df.groupby("age_group", observed=True)["diabetes"].agg(["mean", "count"])
age_diab["mean"] *= 100
print("\nDiabetes Rate by Age Group:")
print(age_diab.rename(columns={"mean": "Rate (%)", "count": "Count"}).to_string())

# BMI groups
df["bmi_group"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 100], labels=["Underweight", "Normal", "Overweight", "Obese"])
bmi_diab = df.groupby("bmi_group", observed=True)["diabetes"].agg(["mean", "count"])
bmi_diab["mean"] *= 100
print("\nDiabetes Rate by BMI Group:")
print(bmi_diab.rename(columns={"mean": "Rate (%)", "count": "Count"}).to_string())

# Hypertension + Heart Disease interaction
print("\nDiabetes Rate by Hypertension x Heart Disease:")
interaction = df.groupby(["hypertension", "heart_disease"])["diabetes"].agg(["mean", "count"])
interaction["mean"] *= 100
print(interaction.rename(columns={"mean": "Rate (%)", "count": "Count"}).to_string())

# =============================================================================
# 5. Visualizations
# =============================================================================

# Figure 1: Distribution comparison by diabetes status
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(numerical_cols):
    ax = axes[i // 2, i % 2]
    for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        data = df[df["diabetes"] == label][col]
        ax.hist(data, bins=50, alpha=0.6, label=f"Diabetes={label}", color=color, density=True)
    ax.set_title(f"{col} Distribution by Diabetes Status", fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "2_distribution_by_diabetes.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 2: Box plots by diabetes status
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(numerical_cols):
    ax = axes[i // 2, i % 2]
    df.boxplot(column=col, by="diabetes", ax=ax, patch_artist=True,
               boxprops=dict(facecolor="#3498db", alpha=0.7))
    ax.set_title(f"{col} by Diabetes Status", fontweight="bold")
    ax.set_xlabel("Diabetes (0=No, 1=Yes)")
    ax.set_ylabel(col)
plt.suptitle("")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "2_boxplots_by_diabetes.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 3: Diabetes rate heatmap - Age x BMI
fig, ax = plt.subplots(figsize=(10, 6))
pivot = df.pivot_table(values="diabetes", index="bmi_group", columns="age_group", aggfunc="mean", observed=True) * 100
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", ax=ax, cbar_kws={"label": "Diabetes Rate (%)"})
ax.set_title("Diabetes Rate (%) by Age Group x BMI Group", fontsize=14, fontweight="bold")
ax.set_xlabel("Age Group")
ax.set_ylabel("BMI Group")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "2_diabetes_heatmap_age_bmi.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 4: Pair plot for key features
key_features = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "diabetes"]
sample_df = df[key_features].sample(n=min(5000, len(df)), random_state=42)
g = sns.pairplot(sample_df, hue="diabetes", palette={0: "#2ecc71", 1: "#e74c3c"}, 
                 diag_kind="kde", plot_kws={"alpha": 0.5, "s": 20})
g.fig.suptitle("Pair Plot of Numerical Features by Diabetes Status", y=1.02, fontweight="bold")
plt.savefig(RESULTS_DIR / "2_pairplot.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 5: Violin plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(numerical_cols):
    ax = axes[i // 2, i % 2]
    parts = ax.violinplot([df[df["diabetes"] == 0][col], df[df["diabetes"] == 1][col]], 
                          positions=[0, 1], showmeans=True, showmedians=True)
    for pc, color in zip(parts["bodies"], ["#2ecc71", "#e74c3c"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax.set_title(f"{col} Distribution", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Diabetes", "Diabetes"])
    ax.set_ylabel(col)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "2_violin_plots.png", dpi=150, bbox_inches="tight")
plt.close()

# Figure 6: Feature importance correlation
fig, ax = plt.subplots(figsize=(8, 5))
correlations = df[numerical_cols + ["diabetes"]].corr()["diabetes"].drop("diabetes").sort_values()
colors = ["#e74c3c" if x > 0 else "#3498db" for x in correlations]
correlations.plot(kind="barh", ax=ax, color=colors, edgecolor="black")
ax.set_title("Feature Correlation with Diabetes", fontsize=14, fontweight="bold")
ax.set_xlabel("Correlation Coefficient")
ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "2_feature_correlation.png", dpi=150, bbox_inches="tight")
plt.close()

# =============================================================================
# 6. Save Advanced Report
# =============================================================================
report = f"""
================================================================================
                 DIABETES PREDICTION - ADVANCED EDA REPORT
================================================================================

1. STATISTICAL SIGNIFICANCE
---------------------------
Chi-Square Tests (Categorical Features):
{pd.DataFrame(chi2_results).to_string(index=False)}

T-Tests (Numerical Features):
{pd.DataFrame(ttest_results).to_string(index=False)}

2. OUTLIER SUMMARY
------------------
{pd.DataFrame(outlier_summary).to_string(index=False)}

3. DIABETES RATE BY AGE GROUP
-----------------------------
{age_diab.rename(columns={'mean': 'Rate (%)', 'count': 'Count'}).to_string()}

4. DIABETES RATE BY BMI GROUP
-----------------------------
{bmi_diab.rename(columns={'mean': 'Rate (%)', 'count': 'Count'}).to_string()}

5. DIABETES RATE BY HYPERTENSION x HEART DISEASE
------------------------------------------------
{interaction.rename(columns={'mean': 'Rate (%)', 'count': 'Count'}).to_string()}

================================================================================
Generated Plots:
- 2_distribution_by_diabetes.png
- 2_boxplots_by_diabetes.png
- 2_diabetes_heatmap_age_bmi.png
- 2_pairplot.png
- 2_violin_plots.png
- 2_feature_correlation.png
================================================================================
"""

with open(RESULTS_DIR / "2_advanced_eda_report.txt", "w") as f:
    f.write(report)

print(report)
print("\nAdvanced EDA completed. Results saved to 'results/' folder.")

# Cleanup temporary columns
df.drop(columns=["age_group", "bmi_group"], inplace=True)

