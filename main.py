# PART 1 : IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# PART 2 : LOAD DATASET

df = pd.read_csv("Dataset/EV_Battery_Dataset.csv")

print("\n" + "--- Dataset Loaded ---".center(100))
print(df)

# PART 3 : UNDERSTANDING DATA

print("\n" + "--- Dataset Information ---".center(100))
print(df.info())

print("\n" + "--- First Five Rows ---".center(100))
print(df.head())

print("\n" + "--- Last Five Rows ---".center(100))
print(df.tail())

print("\n" + "--- Dataset Data Types ---".center(100))
print(df.dtypes)

print("\n" + "--- Total Rows and Columns ---".center(100))
print(df.shape)

# PART 4 : QUANTITATIVE AND QUALITATIVE DATA

print("\n" + "--- Quantitative Data ---".center(100))
quantitative = df.select_dtypes(include=["int64", "float64"])
print(quantitative)

print("\n" + "--- Qualitative Data ---".center(100))
qualitative = df.select_dtypes(include=["object", "category"])
print(qualitative)

# PART 5 : NULL VALUES

print("\n" + "--- Null Values ---".center(100))
print(df.isnull().sum())

# PART 6 : UNIQUE VALUES

print("\n" + "--- Unique Values ---".center(100))
print(df.nunique())

# PART 7 : DUPLICATE VALUES

print("\n" + "--- Duplicate Rows ---".center(100))
print(df.duplicated().sum())

# PART 8 : STATISTICAL SUMMARY

print("\n" + "--- Statistical Summary ---".center(100))
print(df.describe())

# PART 9 : MISSING VALUE HANDLING

print("\n" + "--- Missing Values Before Handling ---".center(100))
print(df.isnull().sum())

# Fill numerical missing values with median

for column in quantitative.columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values with mode

for column in qualitative.columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\n" + "--- Missing Values After Handling ---".center(100))
print(df.isnull().sum())

# PART 10 : SKEWNESS

print("\n" + "--- Skewness ---".center(100))
for column in quantitative.columns:
    print(column,"=",df[column].skew())

# PART 11 : KURTOSIS

print("\n" + "--- Kurtosis ---".center(100))
for column in quantitative.columns:
    print(column,"=",df[column].kurtosis())

# PART 12 : HISTOGRAM

print("\n" + "--- Histogram of SoH ---".center(100))
plt.figure(figsize=(8, 5))
plt.hist(df["SoH_Percent"],bins=30)
plt.xlabel("SoH Percent")
plt.ylabel("Frequency")
plt.title("Distribution of Battery SoH")
plt.show()

# PART 13 : NORMAL CURVE

print("\n" + "--- Histogram with Normal Curve ---".center(100))
mean = df["SoH_Percent"].mean()
std = df["SoH_Percent"].std()
x = np.linspace(df["SoH_Percent"].min(),df["SoH_Percent"].max(),100)
normal_curve = (1/(std*np.sqrt(2*np.pi)))*np.exp(-((x-mean)**2)/(2*std**2))
plt.figure(figsize=(8, 5))
plt.hist(df["SoH_Percent"],bins=30,density=True,alpha=0.7)
plt.plot(x,normal_curve)
plt.xlabel("SoH Percent")
plt.ylabel("Density")
plt.title("SoH Distribution with Normal Curve")
plt.show()

# PART 14 : BOX PLOT

print("\n" + "--- Box Plot of SoH ---".center(100))
plt.figure(figsize=(7, 5))
plt.boxplot(df["SoH_Percent"])
plt.ylabel("SoH Percent")
plt.title("SoH Box Plot")
plt.show()

# PART 15 : PIE CHART

print("\n" + "--- Battery Status Pie Chart ---".center(100))
battery_status = df["Battery_Status"].value_counts()
plt.figure(figsize=(7, 7))
plt.pie(battery_status,labels=battery_status.index,autopct="%1.1f%%")
plt.title("Battery Status Distribution")
plt.show()

# PART 16 : BAR CHART

print("\n" + "--- Average SoH by Battery Type ---".center(100))
average_soh = df.groupby("Battery_Type")["SoH_Percent"].mean()
print(average_soh)
plt.figure(figsize=(8, 5))
plt.bar(average_soh.index,average_soh.values)
plt.xlabel("Battery Type")
plt.ylabel("Average SoH")
plt.title("Average SoH by Battery Type")
plt.xticks(rotation=20)
plt.show()

# PART 17 : COVARIANCE

print("\n" + "--- Covariance Matrix ---".center(100))
covariance = quantitative.cov()
print(covariance)

# PART 18 : CORRELATION

print("\n" + "--- Correlation Matrix ---".center(100))
correlation = quantitative.corr()
print(correlation)

# PART 19 : CORRELATION WITH SoH

print("\n" + "--- Correlation with SoH ---".center(100))
soh_correlation = correlation["SoH_Percent"].sort_values(ascending=False)
print(soh_correlation)

# PART 20 : CORRELATION HEATMAP

print("\n" + "--- Correlation Heatmap ---".center(100))
plt.figure(figsize=(12, 8))
plt.imshow(correlation,interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(correlation.columns)),correlation.columns,rotation=90)
plt.yticks(range(len(correlation.columns)),correlation.columns)
plt.title("Correlation Heatmap")
plt.show()

# PART 21 : SIMPLE LINEAR REGRESSION

print("\n" + "--- Simple Linear Regression ---".center(100))

# Independent Variable
X = df[["Internal_Resistance_Ohm"]]

# Dependent Variable
y = df["SoH_Percent"]

model = LinearRegression()
model.fit(X,y)

# Slope
slope = model.coef_[0]

# Intercept
intercept = model.intercept_

# Prediction
y_pred = model.predict(X)


# R2
r2 = r2_score(y,y_pred)

# Correlation
correlation_value = df["Internal_Resistance_Ohm"].corr(df["SoH_Percent"])
print("Independent Variable =","Internal_Resistance_Ohm")
print("Dependent Variable =","SoH_Percent")
print("Slope =",slope)
print("Intercept =",intercept)
print("Correlation =",correlation_value)
print("R2 =",r2)
print("\nEquation:")
print("SoH_Percent =","* Internal_Resistance_Ohm +",intercept)

# PART 22 : SLR GRAPH

print("\n" + "--- Simple Linear Regression Graph ---".center(100))
plt.figure(figsize=(8, 5))
plt.scatter(df["Internal_Resistance_Ohm"],df["SoH_Percent"])
plt.plot(df["Internal_Resistance_Ohm"],y_pred)
plt.xlabel("Internal Resistance (Ohm)")
plt.ylabel("SoH Percent")
plt.title("Internal Resistance vs Battery SoH")
plt.show()

# PART 23 : SLR PREDICTION

print("\n" + "--- SLR Prediction ---".center(100))
new_resistance = 0.0287
new_prediction = model.predict([[new_resistance]])
print("Internal Resistance =",new_resistance)
print("Predicted SoH =",new_prediction[0])

# PART 24 : OUTLIER DETECTION USING IQR

print("\n" + "--- Outlier Detection Using IQR ---".center(100))
for column in quantitative.columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print("\nColumn =", column)
    print("Q1 =",Q1)
    print("Q3 =",Q3)
    print("IQR =",IQR)
    print("Lower Limit =",lower_limit)
    print("Upper Limit =",upper_limit)
    print("Number of Outliers =",len(outliers))

# PART 25 : FEATURE SELECTION

print("\n" + "--- Feature Selection ---".center(100))
target = "SoH_Percent"
features = [
    "Car_Model",
    "Battery_Type",
    "Battery_Capacity_kWh",
    "Vehicle_Age_Months",
    "Total_Charging_Cycles",
    "Avg_Temperature_C",
    "Fast_Charge_Ratio",
    "Avg_Discharge_Rate_C",
    "Driving_Style",
    "Internal_Resistance_Ohm"
]

print("Dependent Variable =",target)
print("\nIndependent Variables:")

for feature in features:
    print(feature)

# Vehicle_ID is not used because
# it is only an identifier.
print("\nExcluded Variable = Vehicle_ID")

# PART 26 : CREATE X AND y

X = df[features]
y = df[target]

# PART 27 : CONVERT CATEGORICAL DATA TO NUMBERS

print("\n" + "--- Encoding Categorical Data ---".center(100))
X = pd.get_dummies(X,drop_first=True)
print(X.head())

# PART 28 : TRAIN TEST SPLIT

print("\n" + "--- Train Test Split ---".center(100))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

print("Training Rows =",len(X_train))
print("Testing Rows =",len(X_test))
print("Training Data = 80%")
print("Testing Data = 20%")

# PART 29 : STANDARDIZATION

print("\n" + "--- Standardization ---".center(100))
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PART 30 : LINEAR REGRESSION

print("\n" + "--- Linear Regression ---".center(100))
linear_model = LinearRegression()
linear_model.fit(X_train_scaled,y_train)
linear_train_prediction = linear_model.predict(X_train_scaled)
linear_test_prediction = linear_model.predict(X_test_scaled)
linear_train_r2 = r2_score(y_train,linear_train_prediction)
linear_test_r2 = r2_score(y_test,linear_test_prediction)
linear_mae = mean_absolute_error(y_test,linear_test_prediction)
linear_mse = mean_squared_error(y_test,linear_test_prediction)
linear_rmse = linear_mse ** 0.5

print("Train R2 =",linear_train_r2)
print("Test R2 =",linear_test_r2)
print("MAE =",linear_mae)
print("MSE =",linear_mse)
print("RMSE =",linear_rmse)

# PART 31 : RIDGE REGRESSION

print("\n" + "--- Ridge Regression ---".center(100))
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled,y_train)
ridge_prediction = ridge_model.predict(X_test_scaled)
ridge_r2 = r2_score(y_test,ridge_prediction)
ridge_mae = mean_absolute_error(y_test,ridge_prediction)
ridge_mse = mean_squared_error(y_test,ridge_prediction)
ridge_rmse = ridge_mse ** 0.5

print("Ridge R2 =",ridge_r2)
print("Ridge MAE =",ridge_mae)
print("Ridge MSE =",ridge_mse)
print("Ridge RMSE =",ridge_rmse)

# PART 32 : LASSO REGRESSION

print("\n" + "--- Lasso Regression ---".center(100))
lasso_model = Lasso(alpha=0.01,max_iter=10000)
lasso_model.fit(X_train_scaled,y_train)
lasso_prediction = lasso_model.predict(X_test_scaled)
lasso_r2 = r2_score(y_test,lasso_prediction)
lasso_mae = mean_absolute_error(y_test,lasso_prediction)
lasso_mse = mean_squared_error(y_test,lasso_prediction)
lasso_rmse = lasso_mse ** 0.5

print("Lasso R2 =",lasso_r2)
print("Lasso MAE =",lasso_mae)
print("Lasso MSE =",lasso_mse)
print("Lasso RMSE =",lasso_rmse)

# PART 33 : MODEL COMPARISON

print("\n" + "--- Model Comparison ---".center(100))
results = pd.DataFrame({
    "Model": ["Linear Regression","Ridge Regression","Lasso Regression"],
    "R2": [linear_test_r2,ridge_r2,lasso_r2],
    "MAE": [linear_mae,ridge_mae,lasso_mae],
    "MSE": [linear_mse,ridge_mse,lasso_mse],
    "RMSE": [linear_rmse,ridge_rmse,lasso_rmse]
    })

print(results)

# PART 34 : ADJUSTED R2

print("\n" + "--- Adjusted R2 ---".center(100))
n = len(y_test)
p = X_test.shape[1]
adjusted_r2 = (1 -((1 - linear_test_r2)*(n - 1)/(n - p - 1)))

print("R2 =",linear_test_r2)
print("Adjusted R2 =",adjusted_r2)

# PART 35 : ACTUAL VS PREDICTED

print("\n" + "--- Actual vs Predicted ---".center(100))
plt.figure(figsize=(8, 5))
plt.scatter(y_test,linear_test_prediction)
plt.xlabel("Actual SoH")
plt.ylabel("Predicted SoH")
plt.title("Actual vs Predicted SoH")
plt.show()

# PART 36 : RESIDUAL ANALYSIS

print("\n" + "--- Residual Analysis ---".center(100))
residuals = (y_test -linear_test_prediction)
plt.figure(figsize=(8, 5))
plt.scatter(linear_test_prediction,residuals)
plt.axhline(y=0,linestyle="--")
plt.xlabel("Predicted SoH")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values")
plt.show()

# PART 37 : RESIDUAL HISTOGRAM

print("\n" + "--- Residual Histogram ---".center(100))
plt.figure(figsize=(8, 5))
plt.hist(residuals,bins=30)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual Histogram")
plt.show()

# PART 38 : 5-FOLD CROSS VALIDATION

print("\n" + "--- 5 Fold Cross Validation ---".center(100))
linear_cv = cross_val_score(linear_model,X,y,cv=5,scoring="r2")
ridge_cv = cross_val_score(ridge_model,X,y,cv=5,scoring="r2")
lasso_cv = cross_val_score(lasso_model,X,y,cv=5,scoring="r2")

print("\nLinear Regression Fold Scores:")
print(linear_cv)
print("Mean Linear CV R2 =",linear_cv.mean())
print("\nRidge Regression Fold Scores:")
print(ridge_cv)
print("Mean Ridge CV R2 =",ridge_cv.mean())
print("\nLasso Regression Fold Scores:")
print(lasso_cv)
print("Mean Lasso CV R2 =",lasso_cv.mean())

# PART 39 : BEST MODEL

print("\n" + "--- Best Model ---".center(100))
if linear_test_r2 >= ridge_r2 and linear_test_r2 >= lasso_r2:
    print("Best Model = Linear Regression")
elif ridge_r2 >= linear_test_r2 and ridge_r2 >= lasso_r2:
    print("Best Model = Ridge Regression")
else:
    print("Best Model = Lasso Regression")

# PART 40 : FINAL PREDICTION

print("\n" + "--- Final Battery Health Prediction ---".center(100))
new_ev = X_test.iloc[[0]]
new_ev_scaled = scaler.transform(new_ev)
final_prediction = linear_model.predict(new_ev_scaled)
print("Example EV Battery:")
print(new_ev)
print("\nPredicted Battery SoH =",final_prediction[0],"%")