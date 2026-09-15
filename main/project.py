"""
EV Battery Health Prediction ML Project
Course: 303-05 Fundamentals of Machine Learning
S.Y.B.C.A (AI), Semester III

Dataset: EV_Battery_Dataset.csv (provided by subject faculty)

Main task:
    Predict SoH_Percent (State of Health) using vehicle, battery and usage features.

Course mapping:
    Unit 1 -> data understanding, EDA, data types, distributions, missing values, outliers
    Unit 2 -> automated EDA, covariance/correlation, Simple Linear Regression
    Unit 3 -> supervised learning, train/test split, features/labels, MSE/MAE, overfitting
    Unit 4 -> Linear Regression, OLS, R2/MSE/RMSE, Ridge/Lasso, cross-validation

Run:
    python project.py

Optional automated EDA packages:
    pip install ydata-profiling sweetviz
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. File and output folders
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "EV_Battery_Dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "Reports", "project_outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("EV BATTERY HEALTH PREDICTION ML PROJECT")
print("=" * 70)

# ------------------------------------------------------------
# 2. Basic EDA
# ------------------------------------------------------------
print("\n1. FIRST FIVE RECORDS")
print(df.head())

print("\n2. DATASET SHAPE")
print("Rows:", df.shape[0], "Columns:", df.shape[1])

print("\n3. COLUMN NAMES")
print(df.columns.tolist())

print("\n4. DATA TYPES")
print(df.dtypes)

print("\n5. DATASET INFORMATION")
df.info()

print("\n6. MISSING VALUES")
print(df.isnull().sum())

print("\n7. DUPLICATE ROWS")
print(df.duplicated().sum())

print("\n8. NUMERICAL SUMMARY")
print(df.describe())

print("\n9. CATEGORICAL SUMMARY")
for col in df.select_dtypes(include="object").columns:
    print(f"\n{col}:")
    print(df[col].value_counts())

# ------------------------------------------------------------
# 3. Quantitative / qualitative data
# ------------------------------------------------------------
quantitative = df.select_dtypes(include=np.number).columns.tolist()
qualitative = df.select_dtypes(exclude=np.number).columns.tolist()

print("\n10. QUANTITATIVE COLUMNS")
print(quantitative)

print("\n11. QUALITATIVE COLUMNS")
print(qualitative)

# Vehicle_ID is an identifier, not a useful predictive feature.
print("\nNote: Vehicle_ID is an identifier, so it is excluded from modelling.")

# ------------------------------------------------------------
# 4. Skewness and kurtosis
# ------------------------------------------------------------
print("\n12. SKEWNESS")
print(df[quantitative].skew().round(4))

print("\n13. KURTOSIS (Pandas excess kurtosis)")
print(df[quantitative].kurt().round(4))

# ------------------------------------------------------------
# 5. Outlier detection using IQR
# ------------------------------------------------------------
print("\n14. IQR OUTLIER CHECK")
outlier_summary = []

for col in quantitative:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[col] < lower) | (df[col] > upper)).sum())

    outlier_summary.append(
        [col, q1, q3, iqr, lower, upper, count]
    )

outlier_df = pd.DataFrame(
    outlier_summary,
    columns=["Column", "Q1", "Q3", "IQR", "Lower_Bound", "Upper_Bound", "Outlier_Count"]
)
print(outlier_df.round(4).to_string(index=False))

print(
    "\nOutliers are reviewed rather than automatically deleted. "
    "Some extreme battery/temperature/resistance values can be valid real-world observations."
)

# ------------------------------------------------------------
# 6. Correlation
# ------------------------------------------------------------
print("\n15. CORRELATION WITH SoH_Percent")
corr_with_target = (
    df[quantitative]
    .corr()["SoH_Percent"]
    .sort_values(ascending=False)
)
print(corr_with_target.round(4))

# ------------------------------------------------------------
# 7. Simple Linear Regression demonstration
#    Strongest non-target numerical predictor: Internal Resistance
# ------------------------------------------------------------
slr_x = df[["Internal_Resistance_Ohm"]]
slr_y = df["SoH_Percent"]

slr = LinearRegression()
slr.fit(slr_x, slr_y)

slr_slope = slr.coef_[0]
slr_intercept = slr.intercept_
slr_pred = slr.predict(slr_x)

covariance = df["Internal_Resistance_Ohm"].cov(df["SoH_Percent"])
correlation = df["Internal_Resistance_Ohm"].corr(df["SoH_Percent"])
slr_r2 = r2_score(slr_y, slr_pred)

print("\n16. SIMPLE LINEAR REGRESSION")
print("Independent variable: Internal_Resistance_Ohm")
print("Dependent variable: SoH_Percent")
print("Covariance:", round(covariance, 6))
print("Correlation:", round(correlation, 6))
print("Slope:", round(slr_slope, 6))
print("Intercept:", round(slr_intercept, 6))
print(f"Equation: SoH = {slr_slope:.4f} * Internal_Resistance + {slr_intercept:.4f}")
print("R2:", round(slr_r2, 6))

median_resistance = df["Internal_Resistance_Ohm"].median()
print(
    f"Predicted SoH at median internal resistance ({median_resistance:.4f} Ohm): "
    f"{slr.predict([[median_resistance]])[0]:.2f}%"
)

# Plot SLR
plt.figure(figsize=(8, 5))
plt.scatter(df["Internal_Resistance_Ohm"], df["SoH_Percent"], s=10, alpha=0.30, label="Observed data")
x_line = np.linspace(
    df["Internal_Resistance_Ohm"].min(),
    df["Internal_Resistance_Ohm"].max(),
    200
)
plt.plot(
    x_line,
    slr.predict(x_line.reshape(-1, 1)),
    linewidth=2,
    label="SLR line"
)
plt.xlabel("Internal Resistance (Ohm)")
plt.ylabel("State of Health (SoH %)")
plt.title("Internal Resistance vs SoH")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "slr_internal_resistance_soh.png"), dpi=160)
plt.close()

# ------------------------------------------------------------
# 8. Main supervised-learning problem
# ------------------------------------------------------------
TARGET = "SoH_Percent"

FEATURES = [
    "Car_Model",
    "Battery_Type",
    "Battery_Capacity_kWh",
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
    "Driving_Style",
    "Internal_Resistance_Ohm",
]

X = df[FEATURES]
y = df[TARGET]

categorical_features = [
    "Car_Model",
    "Battery_Type",
    "Driving_Style",
]

numeric_features = [
    "Battery_Capacity_kWh",
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
    "Internal_Resistance_Ohm",
]

# ------------------------------------------------------------
# 9. Preprocessing pipeline
# ------------------------------------------------------------
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# 80:20 split, as recommended in the practical instructions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n17. TRAIN / TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# ------------------------------------------------------------
# 10. Model training and evaluation
# ------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
}

results = {}
trained_models = {}

for name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    results[name] = {
        "Train R2": r2_score(y_train, train_pred),
        "Test R2": r2_score(y_test, test_pred),
        "Test MAE": mean_absolute_error(y_test, test_pred),
        "Test MSE": mean_squared_error(y_test, test_pred),
        "Test RMSE": mean_squared_error(y_test, test_pred) ** 0.5,
    }

    trained_models[name] = pipeline

results_df = pd.DataFrame(results).T

print("\n18. MODEL RESULTS")
print(results_df.round(6))

# ------------------------------------------------------------
# 11. Best model
# ------------------------------------------------------------
best_model_name = results_df["Test R2"].idxmax()
best_model = trained_models[best_model_name]
best_pred = best_model.predict(X_test)

print("\n19. BEST MODEL")
print(best_model_name)

print("\nActual vs Predicted sample:")
comparison = pd.DataFrame(
    {
        "Actual_SoH": y_test.values,
        "Predicted_SoH": best_pred
    }
)
print(comparison.head(10).round(3).to_string(index=False))

# ------------------------------------------------------------
# 12. Actual vs predicted plot
# ------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, best_pred, s=12, alpha=0.35)
min_value = min(y_test.min(), best_pred.min())
max_value = max(y_test.max(), best_pred.max())
plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linewidth=2,
    label="Perfect prediction"
)
plt.xlabel("Actual SoH (%)")
plt.ylabel("Predicted SoH (%)")
plt.title(f"Actual vs Predicted SoH — {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "actual_vs_predicted.png"), dpi=160)
plt.close()

# ------------------------------------------------------------
# 13. Residual analysis
# ------------------------------------------------------------
residuals = y_test - best_pred

plt.figure(figsize=(8, 5))
plt.scatter(best_pred, residuals, s=12, alpha=0.35)
plt.axhline(0, linestyle="--", linewidth=2)
plt.xlabel("Fitted SoH (%)")
plt.ylabel("Residual")
plt.title("Residuals vs Fitted Values")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "residuals_vs_fitted.png"), dpi=160)
plt.close()

print("\n20. RESIDUAL SUMMARY")
print("Mean residual:", round(residuals.mean(), 6))
print("Residual standard deviation:", round(residuals.std(), 6))

# ------------------------------------------------------------
# 14. Adjusted R2
# ------------------------------------------------------------
transformed_X_train = best_model.named_steps["preprocessor"].transform(X_train)
p = transformed_X_train.shape[1]
n = len(y_test)
test_r2 = r2_score(y_test, best_pred)

# Adjusted R2 is most meaningful when calculated with the same n and p.
adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1)

print("\n21. ADJUSTED R2")
print("Number of transformed predictors:", p)
print("Test R2:", round(test_r2, 6))
print("Adjusted R2:", round(adjusted_r2, 6))

# ------------------------------------------------------------
# 15. Cross-validation
# ------------------------------------------------------------
print("\n22. 5-FOLD CROSS-VALIDATION")

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_table = {}

for name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="r2"
    )

    cv_table[name] = {
        "Fold 1": scores[0],
        "Fold 2": scores[1],
        "Fold 3": scores[2],
        "Fold 4": scores[3],
        "Fold 5": scores[4],
        "Mean R2": scores.mean(),
        "Std R2": scores.std(),
    }

cv_df = pd.DataFrame(cv_table).T
print(cv_df.round(6))

# ------------------------------------------------------------
# 16. Model coefficients for Linear Regression
# ------------------------------------------------------------
linear_model = trained_models["Linear Regression"]
feature_names = linear_model.named_steps["preprocessor"].get_feature_names_out()
coefficients = linear_model.named_steps["model"].coef_

coef_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Coefficient": coefficients
    }
)
coef_df["Absolute_Coefficient"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values("Absolute_Coefficient", ascending=False)

print("\n23. TOP LINEAR-REGRESSION COEFFICIENTS")
print(coef_df.head(15).round(6).to_string(index=False))

# ------------------------------------------------------------
# 17. Optional automated EDA reports
# ------------------------------------------------------------
print("\n24. AUTOMATED EDA")

# ydata-profiling is preferred for the course practical.
try:
    from ydata_profiling import ProfileReport

    profile = ProfileReport(
        df,
        title="EV Battery Health Prediction - Automated EDA Report",
        explorative=True
    )
    profile_path = os.path.join(OUTPUT_DIR, "EDA_Report_ydata.html")
    profile.to_file(profile_path)
    print("ydata-profiling report created:", profile_path)
except Exception as e:
    print("ydata-profiling is not installed or could not run.")
    print("Install with: pip install ydata-profiling")
    print("A compact automated EDA HTML report will still be created.")

# Sweetviz is optional and useful for a visual second report.
try:
    import sweetviz as sv

    sweet_report = sv.analyze(df)
    sweet_path = os.path.join(OUTPUT_DIR, "EDA_Report_Sweetviz.html")
    sweet_report.show_html(sweet_path, open_browser=False)
    print("Sweetviz report created:", sweet_path)
except Exception:
    print("Sweetviz is optional. Install with: pip install sweetviz")

# ------------------------------------------------------------
# 18. Save numerical results
# ------------------------------------------------------------
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_results.csv"))
cv_df.to_csv(os.path.join(OUTPUT_DIR, "cross_validation_results.csv"))
outlier_df.to_csv(os.path.join(OUTPUT_DIR, "outlier_summary.csv"), index=False)
coef_df.to_csv(os.path.join(OUTPUT_DIR, "linear_coefficients.csv"), index=False)

# ------------------------------------------------------------
# 19. Final prediction example
# ------------------------------------------------------------
# Example: use the first test record as an unseen vehicle.
sample_vehicle = X_test.iloc[[0]]
sample_prediction = best_model.predict(sample_vehicle)[0]

print("\n25. SAMPLE NEW-VEHICLE PREDICTION")
print(sample_vehicle.to_string(index=False))
print(f"Predicted SoH using {best_model_name}: {sample_prediction:.2f}%")

print("\n" + "=" * 70)
print("PROJECT COMPLETE")
print("=" * 70)
print("Best model:", best_model_name)
print(f"Test R2: {results_df.loc[best_model_name, 'Test R2']:.4f}")
print(f"Test MAE: {results_df.loc[best_model_name, 'Test MAE']:.4f}%")
print(f"Test RMSE: {results_df.loc[best_model_name, 'Test RMSE']:.4f}%")
print("=" * 70)
