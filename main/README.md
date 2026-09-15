# 🔋 EV Battery Health Prediction ML Project

> **Machine Learning Mini Project | S.Y.B.C.A (AI), Semester III | 303 Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C)](https://matplotlib.org/)

---

## 📌 Project Information

| Item | Details |
|---|---|
| **Project Title** | EV Battery Health Prediction ML Project |
| **Subject** | 303-05 Fundamentals of Machine Learning |
| **Program** | S.Y.B.C.A (AI), Semester III |
| **College** | Sutex Bank College of Computer Applications & Science, Amroli |
| **Faculty** | Dr. Hiral Patel |
| **Dataset** | `EV_Battery_Dataset.csv` |
| **ML Problem** | Supervised Regression |
| **Target Variable** | `SoH_Percent` |
| **Dataset Records** | 10,000 |
| **Dataset Columns** | 13 |
| **Missing Values** | 0 |
| **Duplicate Rows** | 0 |
| **Train/Test Split** | 80% / 20% |
| **Final Model** | Linear Regression |

---

## 📚 Course Coverage — Units 1 to 4

This project is designed around the concepts covered in Units 1–4:

- **Unit 1:** Fundamentals of AI/ML, ML lifecycle, data types, EDA, distributions, skewness, kurtosis, missing values and duplicates.
- **Unit 2:** Automated EDA, covariance, correlation and Simple Linear Regression.
- **Unit 3:** Supervised learning, datasets, training/testing sets, features, labels, classification, regression and loss functions.
- **Unit 4:** Linear Regression, OLS, R², MSE, RMSE, Ridge Regression, Lasso Regression, regularization and cross-validation.

The faculty project instructions require dataset understanding, EDA, preprocessing, feature selection, model building, evaluation, improvement, code, results, conclusion and future scope. The required soft-copy structure is `Dataset`, `Python_Code` and `Reports`. 

---

# 🎯 Problem Statement

Develop a supervised Machine Learning model to predict the **State of Health (SoH) percentage of an Electric Vehicle battery** using battery, vehicle and usage-related characteristics.

The project aims to demonstrate a complete Machine Learning workflow in a simple and understandable way.

---

# 🎯 Objectives

1. Understand the EV battery dataset.
2. Identify numerical and categorical variables.
3. Check missing values and duplicate records.
4. Perform Exploratory Data Analysis (EDA).
5. Study distributions and relationships between variables.
6. Calculate covariance and correlation.
7. Demonstrate Simple Linear Regression.
8. Select features and target variables.
9. Split the dataset into training and testing sets.
10. Build supervised regression models.
11. Compare Linear, Ridge and Lasso Regression.
12. Evaluate models using R², MAE, MSE and RMSE.
13. Apply 5-fold cross-validation.
14. Select the best-performing model.
15. Document the complete workflow and results.

---

# 📂 Dataset Source & Data Collection Location

## Dataset Collection Location

**The dataset used in this project was provided by the subject faculty (Ma'am) for the college ML project.**

- **Dataset file:** `Dataset/EV_Battery_Dataset.csv`
- **Collection method:** Faculty-provided project dataset
- **Original external download URL:** Not specified in the supplied project materials
- **External source claim:** No external website is claimed for this specific dataset

This is intentionally stated this way rather than inventing a Kaggle/UCI/GitHub source.

The faculty's project instructions list possible public dataset sources such as Kaggle, UCI Machine Learning Repository, Data.gov portals, GitHub public datasets and OpenML, but the dataset used here was supplied by the faculty. 

### Dataset in this repository

**[Open the dataset →](Dataset/EV_Battery_Dataset.csv)**

---

# 🗃️ Dataset Description

The dataset contains EV battery and vehicle information such as:

- Vehicle ID
- Car Model
- Battery Type
- Battery Capacity
- Vehicle Age
- Total Charging Cycles
- Average Temperature
- Fast Charge Ratio
- Average Discharge Rate
- Driving Style
- Internal Resistance
- State of Health (SoH)
- Battery Status

The actual CSV header uses `Vehicle_Age_Months` and `Internal_Resistance_Ohm`.

---

# 🧠 Why `SoH_Percent` Is the Target?

`SoH_Percent` represents the battery's State of Health as a numerical percentage.

Because the output is a continuous numerical value, this is a:

> **Supervised Regression Problem**

### Target

```text
SoH_Percent
```

### Example features

```text
Battery_Capacity_kWh
Vehicle_Age_Months
Total_Charging_Cycles
Avg_Temperature_C
Fast_Charge_Ratio
Avg_Discharge_Rate_C
Internal_Resistance_Ohm
```

---

# ⚠️ Why `Battery_Status` Is Not the Main Target

The dataset contains a `Battery_Status` column, but it is extremely imbalanced.

The supplied data contains approximately:

| Battery Status | Count |
|---|---:|
| Healthy | 9,996 |
| Replace Required | 4 |

Therefore, `Battery_Status` is **not used as the main prediction target**.

Using `SoH_Percent` gives a more appropriate regression problem for the project.

---

# 🔄 Machine Learning Lifecycle

```text
Problem Definition
        ↓
Data Collection
        ↓
Data Understanding
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
Model Building
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Cross-Validation
        ↓
Final Model
        ↓
Conclusion & Future Scope
```

---

# 🔍 Exploratory Data Analysis

EDA was performed to understand the dataset before model building.

The project includes:

- Dataset structure
- Descriptive statistics
- Numerical/categorical variables
- Missing-value analysis
- Duplicate analysis
- Histogram
- Box plot
- Correlation matrix
- Scatter plots
- Category-wise analysis
- Outlier analysis
- Automated EDA report

---

## 📊 EDA Visualizations

### 1. SoH Distribution

![SoH Distribution](Reports/figures/01_soh_distribution.png)

The histogram shows the distribution of battery State of Health values.

---

### 2. SoH Box Plot

![SoH Box Plot](Reports/figures/02_soh_boxplot.png)

The box plot helps identify the spread and possible outliers in SoH.

---

### 3. Correlation Heatmap

![Correlation Heatmap](Reports/figures/03_correlation_heatmap.png)

The correlation heatmap provides a visual overview of relationships between numerical variables.

---

### 4. Vehicle Age vs SoH

![Vehicle Age vs SoH](Reports/figures/04_age_soh.png)

This plot helps visualize the relationship between vehicle age and battery health.

---

### 5. Charging Cycles vs SoH

![Charging Cycles vs SoH](Reports/figures/05_cycles_soh.png)

This plot shows how SoH varies with the total number of charging cycles.

---

### 6. Internal Resistance vs SoH — SLR

![Internal Resistance vs SoH](Reports/figures/06_resistance_soh_slr.png)

This is the Simple Linear Regression demonstration.

---

### 7. SoH by Battery Type

![SoH by Battery Type](Reports/figures/07_soh_battery_type.png)

This visualization compares SoH across battery types.

---

# 📈 Important Correlation Findings

The actual analysis produced the following correlations with `SoH_Percent`:

| Feature | Correlation with SoH |
|---|---:|
| `Internal_Resistance_Ohm` | **-0.9720** |
| `Total_Charging_Cycles` | **-0.8732** |
| `Vehicle_Age_Months` | **-0.7262** |
| `Battery_Capacity_kWh` | -0.2383 |
| `Fast_Charge_Ratio` | -0.1500 |
| `Avg_Temperature_C` | -0.1122 |
| `Avg_Discharge_Rate_C` | -0.0147 |

### Interpretation

`Internal_Resistance_Ohm` has the strongest numerical relationship with `SoH_Percent`.

Its correlation is approximately:

```text
-0.9720
```

This indicates a very strong negative linear relationship in this dataset.

> Correlation indicates association; it does not by itself prove causation.

---

# 📐 Simple Linear Regression

For the SLR demonstration:

- **Independent variable:** `Internal_Resistance_Ohm`
- **Dependent variable:** `SoH_Percent`

General equation:

\[
y = mx + b
\]

Actual fitted equation:

\[
SoH = -322.6926(Internal\ Resistance) + 105.9282
\]

### SLR R²

```text
0.944756
```

This demonstrates the strong linear relationship between internal resistance and SoH in the supplied dataset.

---

# 🧹 Data Preprocessing

The dataset was checked for:

- Missing values
- Duplicate rows
- Data types
- Numerical variables
- Categorical variables
- Identifier columns
- Target and feature variables

### Actual dataset checks

```text
Records:        10,000
Columns:        13
Missing values: 0
Duplicates:     0
```

`Vehicle_ID` is treated as an identifier rather than a meaningful numerical predictor.

Categorical variables such as car model, battery type and driving style are encoded for Machine Learning using one-hot encoding.

---

# 🤖 Machine Learning Models

Three regression models were evaluated:

## 1. Linear Regression

Linear Regression models a relationship between input variables and a continuous output.

It is simple, interpretable and suitable for the main project problem.

## 2. Ridge Regression

Ridge Regression adds **L2 regularization**.

It penalizes large coefficient values and can help control overfitting and multicollinearity.

## 3. Lasso Regression

Lasso Regression adds **L1 regularization**.

It can shrink some coefficients toward zero and can therefore also perform feature-selection-like behavior.

---

# 📏 Model Evaluation Metrics

## R²

R² measures the proportion of target variation explained by the model.

Higher values are generally better.

## MAE

\[
MAE = \frac{1}{n}\sum |y_i-\hat{y_i}|
\]

MAE represents the average absolute prediction error.

## MSE

\[
MSE = \frac{1}{n}\sum(y_i-\hat{y_i})^2
\]

MSE squares the errors, so large errors receive greater penalty.

## RMSE

\[
RMSE = \sqrt{MSE}
\]

RMSE is expressed in the same units as the target.

---

# 🏆 Model Results

The models were evaluated on the actual 2,000-record test set.

| Model | Test R² | MAE | RMSE |
|---|---:|---:|---:|
| **Linear Regression** | **0.983993** | **0.309681** | **0.412555** |
| Ridge Regression | 0.906562 | 0.700031 | 0.996742 |
| Lasso Regression | 0.886085 | 0.769021 | 1.100548 |

## Best Model

**Linear Regression**

It achieved:

- Highest R²
- Lowest MAE
- Lowest RMSE

---

# 📊 Actual vs Predicted

![Actual vs Predicted](Reports/figures/08_actual_predicted.png)

The plot compares actual SoH values with the values predicted by the final model.

Points closer to the diagonal indicate better prediction.

---

# 📉 Residual Analysis

![Residual Plot](Reports/figures/09_residuals.png)

Residuals are calculated as:

\[
Residual = Actual - Predicted
\]

Residual analysis helps identify prediction errors and possible model problems.

---

# 🔬 Cross-Validation

A **5-fold cross-validation** procedure was used to check model stability.

| Model | Mean 5-Fold CV R² |
|---|---:|
| **Linear Regression** | **0.984945** |
| Ridge Regression | 0.910504 |
| Lasso Regression | 0.890548 |

Linear Regression achieved the highest mean CV R².

Its CV standard deviation was approximately:

```text
0.000825
```

This indicates stable performance across folds.

---

# 📊 Model Comparison

![Model Comparison](Reports/figures/10_model_comparison.png)

The comparison confirms that Linear Regression performed best among the three tested regression models.

---

# 🏅 Final Model

## Linear Regression

The final model was selected because it provided the best overall performance on the actual dataset.

### Final test performance

```text
R²   = 0.983993
MAE  = 0.309681
RMSE = 0.412555
```

### Cross-validation

```text
Mean 5-Fold CV R² = 0.984945
```

---

# 💡 Key Findings

1. `Internal_Resistance_Ohm` has the strongest numerical relationship with SoH.
2. `Total_Charging_Cycles` has a strong negative relationship with SoH.
3. `Vehicle_Age_Months` also has a strong negative relationship with SoH.
4. The dataset contains no missing values.
5. The dataset contains no duplicate rows.
6. `SoH_Percent` is a continuous target, making regression appropriate.
7. Linear Regression performed better than the tested Ridge and Lasso models.
8. Cross-validation confirms stable Linear Regression performance.
9. `Battery_Status` is highly imbalanced and is therefore not used as the primary target.

---

# 🗂️ Project Structure

```text
EV_Battery_Health_Prediction_ML_Project/
│
├── Dataset/
│   └── EV_Battery_Dataset.csv
│
├── Python_Code/
│   └── project.py
│
├── Reports/
│   ├── EDA_Report.html
│   ├── Project_Report.pdf
│   ├── EV_Battery_Health_Prediction_ML_Project.md
│   │
│   ├── figures/
│   │   ├── 01_soh_distribution.png
│   │   ├── 02_soh_boxplot.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 04_age_soh.png
│   │   ├── 05_cycles_soh.png
│   │   ├── 06_resistance_soh_slr.png
│   │   ├── 07_soh_battery_type.png
│   │   ├── 08_actual_predicted.png
│   │   ├── 09_residuals.png
│   │   └── 10_model_comparison.png
│   │
│   └── project_outputs/
│       ├── correlation_with_soh.csv
│       ├── cross_validation_results.csv
│       ├── linear_coefficients.csv
│       ├── model_results.csv
│       └── outlier_summary.csv
│
├── README.md
├── README.txt
└── requirements.txt
```

---

# 📄 Project Files

| File | Purpose |
|---|---|
| [📊 Dataset CSV](Dataset/EV_Battery_Dataset.csv) | Faculty-provided project dataset |
| [🐍 Python Code](Python_Code/project.py) | Complete executable ML program |
| [📘 Project Report PDF](Reports/Project_Report.pdf) | Complete college project report |
| [🔎 Automated EDA Report](Reports/EDA_Report.html) | Automated exploratory analysis |
| [📝 Detailed Markdown](Reports/EV_Battery_Health_Prediction_ML_Project.md) | Detailed project notes and full code |
| [📈 Model Results](Reports/project_outputs/model_results.csv) | Model evaluation results |
| [🔬 Cross-Validation Results](Reports/project_outputs/cross_validation_results.csv) | 5-fold CV results |
| [📊 Correlation Results](Reports/project_outputs/correlation_with_soh.csv) | Correlations with SoH |
| [📐 Linear Coefficients](Reports/project_outputs/linear_coefficients.csv) | Linear model coefficients |
| [⚠️ Outlier Summary](Reports/project_outputs/outlier_summary.csv) | Outlier analysis |

---

# 🚀 How to Run the Project

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd EV_Battery_Health_Prediction_ML_Project
```

## 2. Install required libraries

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install pandas numpy matplotlib scikit-learn
```

Optional automated EDA libraries:

```bash
pip install ydata-profiling sweetviz
```

## 3. Run the Python program

```bash
python Python_Code/project.py
```

The program:

- Loads the dataset
- Checks data quality
- Performs EDA calculations
- Calculates correlation
- Demonstrates SLR
- Preprocesses categorical variables
- Splits the data
- Trains regression models
- Evaluates the models
- Performs cross-validation
- Saves result files
- Generates visualizations

---

# 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **ydata-profiling / automated EDA tools** where required

---

# 📖 Concepts Demonstrated

### Unit 1

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Machine Learning lifecycle
- Dataset understanding
- Quantitative data
- Qualitative data
- Discrete data
- Continuous data
- Nominal data
- Ordinal data
- EDA
- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Skewness
- Kurtosis
- Missing values
- Duplicate values

### Unit 2

- Automated EDA
- Covariance
- Correlation
- Simple Linear Regression
- Slope
- Intercept
- Best-fit line
- Residuals

### Unit 3

- Supervised Learning
- Dataset
- Training set
- Testing set
- Features
- Labels
- Classification
- Regression
- Loss functions
- MSE
- MAE

### Unit 4

- Linear Regression
- Ordinary Least Squares
- R²
- MSE
- RMSE
- Ridge Regression
- Lasso Regression
- Regularization
- Cross-validation
- Model comparison

---

# ⚠️ Limitations

- The dataset was supplied for the academic project.
- The external original source of the supplied dataset is not specified in the project materials.
- Correlation does not prove causation.
- Linear models may not represent every nonlinear battery-aging relationship.
- Real-world battery health can depend on additional variables not present in this dataset.
- The results should not be interpreted as universal battery-health performance for all EVs.

---

# 🔮 Future Scope

Possible future improvements include:

- Larger real-world battery datasets
- Time-series battery measurements
- Nonlinear regression
- Random Forest Regression
- Gradient Boosting
- XGBoost
- Neural Networks
- Remaining Useful Life prediction
- Hyperparameter tuning
- Real-time battery monitoring
- Web-based battery health prediction application

---

# 🎓 Viva Questions

### 1. What is the target variable?

`SoH_Percent`.

### 2. Why is this a regression problem?

Because SoH is a continuous numerical value.

### 3. What is a feature?

A feature is an input variable used to make predictions.

### 4. What is a label?

A label is the output value that the model learns to predict.

### 5. What is EDA?

Exploratory Data Analysis is the process of understanding data through statistics and visualizations.

### 6. What does correlation -0.972 mean?

It indicates a very strong negative linear relationship between internal resistance and SoH in this dataset.

### 7. What is SLR?

Simple Linear Regression models the relationship between one independent variable and one dependent variable.

### 8. What is MAE?

Mean Absolute Error is the average absolute difference between actual and predicted values.

### 9. What is RMSE?

Root Mean Squared Error is the square root of MSE and is expressed in the target's units.

### 10. Why use cross-validation?

To check model performance and stability across different subsets of data.

### 11. Which model performed best?

Linear Regression.

### 12. Why was Battery_Status not used as the main target?

Because its classes are extremely imbalanced: 9,996 Healthy versus only 4 Replace Required.

### 13. What is Ridge Regression?

Linear Regression with L2 regularization.

### 14. What is Lasso Regression?

Linear Regression with L1 regularization.

---

# ✅ Conclusion

The **EV Battery Health Prediction ML Project** successfully demonstrates a complete Machine Learning workflow using the faculty-provided EV battery dataset.

The project performs data understanding, preprocessing, EDA, correlation analysis, Simple Linear Regression, supervised regression, model evaluation, regularization comparison and cross-validation.

The best-performing model is:

> **Linear Regression**

with actual test performance of:

```text
R²   = 0.983993
MAE  = 0.309681
RMSE = 0.412555
```

and:

```text
5-Fold CV Mean R² = 0.984945
```

The project therefore provides a complete and beginner-friendly demonstration of the Machine Learning concepts covered in **Units 1–4**.

---

## 📌 Academic Note

This repository is prepared for an academic Machine Learning mini project. The student should understand the dataset, preprocessing, EDA, model building and evaluation rather than submitting the project without understanding it. The faculty instructions specifically emphasize that students should be able to explain every step of their project during practical sessions.

---

## 📜 License

This repository is intended for **academic/educational use**.

The dataset was provided by the subject faculty for the project. No external ownership or public redistribution rights are claimed for the dataset.
