# Advanced Analytics & ML — Task 1: Predicting House Prices with Linear Regression

**Portfolio Scope:** Data Analytics Portfolio  
**Track:** Data Analytics (`Business-Analytics-Portfolio`)  
**Task Code:** `PRJ-05` (Task #5 in Master Roadmap)  
**Author:** `Dhruv Goyal`  
**Status:** `Testing Complete` (Automated Tests 12/12 Passing; Notebook Executed Headlessly; Pipeline Serialized)  
**Submission Gate:** Strictly **PENDING** until student customizes title cards and records video demonstration.

---

## 1. Project Overview & Official Checklist

This project delivers an econometric analysis and predictive modeling pipeline to forecast California residential property values using **Ordinary Least Squares (OLS) Linear Regression**, regularized regression benchmarks (**Ridge** and **Lasso**), and non-linear spatial tree modeling (**Random Forest**).

The implementation strictly eliminates data leakage by encapsulating median imputation (`SimpleImputer`), continuous standardization (`StandardScaler`), and dummy encoding (`OneHotEncoder(drop='first')`) within Scikit-Learn `ColumnTransformer` and `Pipeline` constructs.

### Official Requirement Verification Matrix

| Checklist Item | Official Requirement | Implementation Detail | Status |
|:---|:---|:---|:---:|
| **1. Load Dataset & EDA** | Inspect nulls, descriptive statistics, and target variable distribution | Section 1 & 2 of `House_Price_Prediction.ipynb`; `fig1_target_distribution.png` | Verified |
| **2. Feature Selection** | Identify likely predictors and explain reasoning in markdown | Section 3 discussion on purchasing power (`median_income`), coordinates, and capacity vs crowding | Verified |
| **3. Handle Nulls & Encoding** | Impute missing values and encode categorical features (One-Hot) | Median imputation for `total_bedrooms` (207 nulls); `OneHotEncoder(drop='first')` for `ocean_proximity` | Verified |
| **4. Correlation Heatmap** | Pearson correlation matrix identifying top predictors of price | `fig2_correlation_heatmap.png` & `feature_correlations.csv` (`median_income` $r = +0.688$) | Verified |
| **5. Train/Test Split** | 80/20 train/test split with deterministic seed | 16,512 train records / 4,128 test records (`random_state=42`) | Verified |
| **6. Train Linear Regression** | Train scikit-learn Linear Regression model | `LinearRegression()` trained via `Pipeline` | Verified |
| **7. Model Evaluation** | Compute MSE, RMSE, and $R^2$ score (plus MAE, 5-Fold CV) | Test $R^2 = 0.6254$, RMSE = \\$70,059, MAE = \\$50,670, 5-Fold CV RMSE = \\$68,623 | Verified |
| **8. Scatter Plot** | Actual vs. Predicted prices scatter plot with $y = x$ reference line | `fig3_actual_vs_predicted.png` annotated with $R^2$ and RMSE | Verified |
| **9. Residual Plot** | Check that residuals are randomly distributed without systematic bias | `fig4_residual_diagnostics.png` (homoscedasticity, Q-Q normality, and ceiling censoring audit) | Verified |
| **10. Coefficient Analysis** | Highest positive and negative impact features | `fig5_coefficient_analysis.png` & `regression_coefficients.csv` | Verified |
| **11. (Bonus) Regularization** | Compare Linear Regression against Ridge and Lasso regularized models | Ridge ($\alpha=1.0$) and Lasso ($\alpha=100.0$) benchmarks in `model_performance_comparison.csv` | Verified |
| **12. (Bonus) Non-Linear Benchmark**| Compare against Random Forest Regressor | Random Forest achieves $R^2 = 0.8143$ and RMSE = \\$49,324, demonstrating spatial non-linearity | Verified |

---

## 2. Dataset Lineage & Integrity

- **Dataset Name:** California Housing Dataset (StatLib / 1990 U.S. Census Bureau)
- **Primary Source:** R. Kelley Pace and Ronald Barry (1997), *"Sparse Spatial Autoregressions"*, Statistics & Probability Letters, 33(3), pp. 291–297.
- **Repository Mirror:** Aurélien Géron (`ageron/handson-ml2`) verified distribution.
- **File Location:** `data/housing.csv`
- **SHA-256 Checksum:** `8A3727F4CF54AC1A327F69B1D5B4DB54C5834EA81C6E4EFC0D163300022A685E`
- **Total Dimensions:** 20,640 census block group observations across 10 columns.
- **Target Variable:** `median_house_value` (USD continuous).
- **Missing Value Profile:** Exactly 207 rows (1.003%) missing in `total_bedrooms`. All other 9 columns are 100% complete.

---

## 3. Authoritative Performance Benchmark

All metrics evaluated on the held-out 20% test split ($N = 4,128$ observations):

| Model Architecture | Test MAE ($) | Test MSE ($^2$) | Test RMSE ($) | Test $R^2$ | 5-Fold CV RMSE (Train) | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Dummy Regressor (Mean Baseline)** | \\$90,606.85 | $1.3107 \\times 10^{10}$ | \\$114,485.64 | -0.0002 | N/A | Naive mean predictor |
| **Linear Regression (OLS)** | **\\$50,670.49** | **$4.9083 \\times 10^9$** | **\\$70,059.19** | **0.6254** | **\\$68,622.54 $\\pm$ \\$1,438.77** | Champion linear model |
| **Ridge Regression ($\alpha=1.0$)** | \\$50,678.11 | $4.9095 \\times 10^9$ | \\$70,067.48 | 0.6253 | \\$68,621.67 $\\pm$ \\$1,440.72$ | $L_2$ regularized |
| **Lasso Regression ($\alpha=100.0$)** | \\$50,669.87 | $4.9116 \\times 10^9$ | \\$70,082.90 | 0.6252 | \\$68,650.88 $\\pm$ \\$1,416.77$ | $L_1$ regularized |
| **Random Forest (100 Trees, Depth 16)**| \\$32,066.11 | $2.4328 \\times 10^9$ | \\$49,323.89 | 0.8143 | \\$49,642.78 $\\pm$ \\$583.20$ | Non-linear spatial comparison |

*Key finding on regularization:* Because $N = 16,512$ training records vastly exceeds $P = 12$ encoded features ($N/P > 1,375$), OLS is already well-conditioned, resulting in near-identical performance across OLS, Ridge, and Lasso.

*Role of Models:* Linear Regression is the primary educational and interpretable model. Random Forest is included strictly as an instructive non-linear comparison to illustrate the structural limitations of planar linear functional forms on complex 2D geographic coordinate surfaces.

---

## 4. Standardized Beta Weights & Conditional Associations

Standardized coefficients represent the expected difference in median house value in USD for every 1 standard deviation shift in continuous predictors (or presence of categorical indicator relative to `<1H OCEAN` baseline), **conditional on all other covariates remaining constant (*ceteris paribus* within this linear model)**:

*Methodological Warning:* Because these estimates are derived from observational cross-sectional census data without experimental identification, they represent descriptive conditional associations and must not be interpreted as established causal economic drivers.

| Impact Rank | Feature | Standardized Beta ($) | Direction | Conditional Association & Spatial Context |
|:---:|:---|:---:|:---:|:---|
| 1 | `ocean_proximity_ISLAND` | +\\$136,125.07 | Positive (+) | Conditional premium for island tracts (small sample: $N=5$) |
| 2 | `median_income` | +\\$75,167.77 | Positive (+) | Primary purchasing power association (+$75.2k per 1 SD income) |
| 3 | `latitude` | -\\$54,415.70 | Negative (-) | Northward geographic shift away from prime coastal metros |
| 4 | `longitude` | -\\$53,826.65 | Negative (-) | Eastward geographic shift towards inland mountain/desert tracts |
| 5 | `population` | -\\$43,403.43 | Negative (-) | High block group density conditionally associated with lower values |
| 6 | `total_bedrooms` | +\\$43,068.18 | Positive (+) | Dwelling capacity association |
| 7 | `ocean_proximity_INLAND` | -\\$39,786.66 | Negative (-) | Inland conditional difference relative to coastal baseline |
| 8 | `households` | +\\$18,382.20 | Positive (+) | Urban aggregation density |
| 9 | `housing_median_age` | +\\$13,889.87 | Positive (+) | Mature neighborhood amenity association |
| 10 | `total_rooms` | -\\$13,094.25 | Negative (-) | Room fragmentation effect when bedrooms/households are held constant |
| 11 | `ocean_proximity_NEAR BAY` | -\\$5,136.64 | Negative (-) | Minor conditional difference relative to `<1H OCEAN` reference |
| 12 | `ocean_proximity_NEAR OCEAN` | +\\$3,431.14 | Positive (+) | Coastal amenity alignment |

**Regression Intercept:** \\$219,899.78 (Expected property value at mean numerical levels for `<1H OCEAN`).

---

## 5. Residual Diagnostics, Gauss-Markov Nuance & Ceiling Effects

Residual analysis reveals critical econometric properties:
1. **Low Aggregate Test Bias:** The residual mean on test data is small (\\$\\approx 1,040.85$), showing absence of severe global level bias.
   - *Gauss-Markov Correction:* In OLS with an intercept, the training residual mean is identically zero by mathematical construction ($\sum e_i = 0$). On held-out test data, a near-zero mean residual indicates low global aggregate bias, **but does NOT prove that Gauss-Markov assumptions hold**. The Gauss-Markov theorem requires strict exogeneity ($\mathbb{E}[\epsilon \mid \mathbf{X}] = 0$), homoscedasticity ($\text{Var}(\epsilon \mid \mathbf{X}) = \sigma^2$), and uncorrelated errors ($\text{Cov}(\epsilon_i, \epsilon_j) = 0$). Heteroscedasticity is clearly present, and spatial autocorrelation likely affects error independence.
2. **Heteroscedasticity:** Error variance expands noticeably above \\$350,000 fitted values.
3. **Census Capping & Validation Limitation:**
   - **Uncapped Properties (<\\$500k, $N = 3,944$ / 95.54%):** Test MAE = \\$46,428.91, Test RMSE = **\\$61,848.69**.
   - **Capped Properties (\\$500k+, $N = 184$ / 4.46%):** Test MAE = \\$141,587.86, Test RMSE = **\\$167,698.58** (**2.71x higher RMSE**).
   - *Validation Limitation:* The historical 1990 census administratively capped values at \\$500,001. For properties at this ceiling, true market values are unrecorded. While the model predicts values above \\$500,000 for many of these tracts, **it is impossible to validate whether those predictions are accurate, underestimated, or overestimated against actual uncapped values**. The apparent \\$167k RMSE reflects deviation from the artificial ceiling, not verified error against actual market transactions.
4. **Data Leakage vs. Spatial Dependence:**
   - *Preprocessing Leakage:* Strictly eliminated. Imputer medians and scaler means/variances were fitted solely on training folds.
   - *Spatial Dependence:* In spatial census data, neighboring block groups are correlated. Under a standard random 80/20 split, adjacent tracts may fall into both train and test splits, potentially resulting in somewhat optimistic generalization metrics compared to spatial block cross-validation.

---

## 6. Project Directory Structure

```
Task_1-Predicting_House_Prices_with_Linear_Regression/
├── data/
│   └── housing.csv                         # Raw California housing dataset (20,640 x 10)
├── models/
│   ├── linear_regression_pipeline.joblib   # Serialized OLS pipeline artifact
│   ├── ridge_pipeline.joblib               # Serialized Ridge pipeline artifact
│   ├── lasso_pipeline.joblib               # Serialized Lasso pipeline artifact
│   └── random_forest_pipeline.joblib       # Serialized Random Forest pipeline artifact
├── outputs/
│   ├── figures/
│   │   ├── fig1_target_distribution.png    # Target histogram/KDE + ocean proximity boxplot
│   │   ├── fig2_correlation_heatmap.png    # Pearson correlation matrix
│   │   ├── fig3_actual_vs_predicted.png    # Actual vs predicted prices scatter plot
│   │   ├── fig4_residual_diagnostics.png   # Homoscedasticity, Q-Q, and censoring breakdown
│   │   ├── fig5_coefficient_analysis.png   # Standardized coefficient horizontal bar chart
│   │   └── fig6_model_performance_comparison.png # Benchmark bar chart comparison
│   └── tables/
│       ├── housing_dataset_summary.csv     # Descriptive statistics & null audit
│       ├── feature_correlations.csv        # Feature correlations with price
│       ├── model_performance_comparison.csv# Multi-model MAE, RMSE, R2, and CV-RMSE
│       ├── regression_coefficients.csv     # Standardized beta weights and impact ranks
│       └── residual_diagnostics_summary.csv# Residual breakdown (uncapped vs capped)
├── build_l2_t1_notebook.py                 # Notebook generation script
├── house_price_prediction.py               # Standalone production pipeline script
├── test_house_price_prediction.py          # Automated unit test suite (12 tests)
├── House_Price_Prediction.ipynb            # Headlessly executed Jupyter Notebook (33 cells)
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 7. How to Reproduce

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Standalone Pipeline Script
```powershell
python house_price_prediction.py
```

### 3. Run Automated Unit Test Suite
```powershell
python test_house_price_prediction.py
```
*Expected result:* `Ran 12 tests in 1.8s ... OK`.

### 4. Re-execute Jupyter Notebook Headlessly
```powershell
jupyter nbconvert --to notebook --execute --inplace House_Price_Prediction.ipynb
```
