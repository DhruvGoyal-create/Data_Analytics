# DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO
## Formal Task Report: Advanced Analytics & ML — Task 1: Predicting House Prices with Linear Regression
### Econometric Modeling, Residual Diagnostics, Standardized Beta Weights & Regularization Analysis on the California Housing Benchmark

**Candidate:** `Dhruv Goyal` *(Notice: Template document; customize with actual student legal name prior to final submission)*  
**Track:** Data Analytics  
**Task Code:** `PRJ-05` (Task #5 in Master Roadmap)  
**Execution Date:** October 2026  
**Status:** `Completed & Quality Gate Verified` (12/12 Automated Unit Tests Passing, Notebook Executed Headlessly)  
**Target Repository Path:** `projects/05_House_Price_Prediction/`  

---

## Executive Summary

Predicting residential property prices is a fundamental problem in real estate economics, mortgage credit risk evaluation, and property tax assessment. In econometric valuation, estimating market value requires understanding both continuous household financial capacity and discrete spatial amenities while managing collinear structural housing dimensions.

This report documents the design, implementation, and empirical verification of an **end-to-end econometric regression pipeline** that predicts California median house values from census block group characteristics. Using the canonical **California Housing Dataset** (StatLib / Pace & Barry, 1997) comprising 20,640 census block groups, the project implements a leakage-free Scikit-Learn `ColumnTransformer` architecture, handles missing value imputation, encodes categorical geographic locations, performs 5-fold cross-validation, conducts exhaustive residual diagnostics, evaluates regularized regression models (**Ridge** and **Lasso**), and quantifies the structural limitations of planar linear models against non-linear tree ensembles (**Random Forest**).

### Key Performance & Audit Highlights:
- **Clean Ingestion, Preprocessing Leakage Controls & Spatial Dependence:** The dataset contains 20,640 records across 10 features. Exactly **207 records (1.003%)** contain missing values, strictly isolated to `total_bedrooms`. Imputation was encapsulated via `SimpleImputer(strategy='median')` fitted solely on training splits, eliminating preprocessing test data leakage. We explicitly distinguish this from spatial dependence: because census block groups were partitioned via standard random 80/20 splitting, spatial proximity between neighboring tracts across splits may induce spatial autocorrelation.
- **Strict Leakage-Free Standardization & Dummy Encoding:** Continuous attributes were scaled via `StandardScaler()`. The multi-category `ocean_proximity` feature was encoded via `OneHotEncoder(drop='first')`, setting `<1H OCEAN` as the reference category and preventing the **dummy variable trap** (perfect multicollinearity with the regression intercept).
- **Benchmark Performance Hierarchy ($N=4,128$ Held-Out Test Records):**
  - **Dummy Regressor (Mean Baseline):** Predicts mean house price (\\$206,856), achieving Test RMSE = **\\$114,485.64**, Test MAE = **\\$90,606.85**, and $R^2 = -0.0002$.
  - **Linear Regression (Primary Educational Model):** Achieves Test $R^2 = \mathbf{0.6254}$, Test RMSE = **\\$70,059.19**, Test MAE = **\\$50,670.49**, and a 5-Fold Training CV-RMSE of **\\$68,622.54 $\pm$ \\$1,438.77**. This represents a **38.8% reduction in root mean squared error** over baseline.
  - **Ridge Regression ($\alpha=1.0$):** Achieves Test $R^2 = 0.6253$, Test RMSE = \\$70,067.48, and 5-Fold CV-RMSE = \\$68,621.67 $\pm$ \\$1,440.72$.
  - **Lasso Regression ($\alpha=100.0$):** Achieves Test $R^2 = 0.6252$, Test RMSE = \\$70,082.90, and 5-Fold CV-RMSE = \\$68,650.88 $\pm$ \\$1,416.77$.
  - **Non-Linear Comparison Benchmark (Random Forest, Depth 16):** Achieves Test $R^2 = \mathbf{0.8143}$, Test RMSE = **\\$49,323.89**, Test MAE = **\\$32,066.11**, and 5-Fold CV-RMSE = **\\$49,642.78 $\pm$ \\$583.20$**.
- **The Asymptotic Conditioning Discovery:** Because the training sample size ($N = 16,512$) vastly exceeds feature dimensionality ($P = 12$ encoded features; $N/P > 1,375$), Ordinary Least Squares is exceptionally well-conditioned. Consequently, $L_1$ and $L_2$ regularization produce near-identical metrics, confirming that standard OLS is already at the empirical limit of linear functional form for this feature set.
- **Dominant Econometric Associations (Standardized Beta Weights):**
  1. `median_income` exhibits the strongest positive conditional association (**+\\$75,167.77** per 1 standard deviation increase; $r = +0.688$).
  2. `ocean_proximity_ISLAND` commands a large spatial scarcity premium (**+\\$136,125.07** over `<1H OCEAN`, small sample: $N=5$).
  3. `ocean_proximity_INLAND` incurs a substantial conditional locational discount (**-\\$39,786.66**).
  4. Geographic coordinates exhibit strong negative conditional associations (`latitude`: **-\\$54,415.70**; `longitude`: **-\\$53,826.65**), reflecting lower interior valuations away from prime coastal economic hubs.
  *Note:* All coefficients reflect conditional partial associations (ceteris paribus within the specification), not established causal economic mechanisms.
- **Forensic Discovery: Historical Census Ceiling Bias & Validation Limitation:** 
  - Exactly **965 properties (4.68%) in the dataset are artificially capped at \\$500,001** due to 1990 U.S. Census Bureau administrative truncation.
  - On uncapped properties (<\\$500k, $N = 3,944$), the linear model achieves an RMSE of **\\$61,848.69**.
  - On capped properties (\\$500k+, $N = 184$), apparent RMSE inflates by **2.71x to \\$167,698.58**.
  - *Validation Limitation:* Because actual uncapped transaction prices were never recorded, it is impossible to validate whether model predictions above \\$500k are accurate against true market reality. The apparent \\$167k RMSE reflects truncation against the artificial administrative ceiling, not verified valuation error against actual market transactions.

---

## 1. Problem Statement & Real Estate Econometrics

In real estate economics and mortgage finance, determining property valuation through hedonic pricing models provides transparency and automated risk underwriting:
1. **Automated Valuation Models (AVMs):** Financial institutions rely on statistical regressions to estimate collateral property value for mortgage lending without costly manual appraisals.
2. **Economic Deconstruction:** Decomposing real estate values into constituent components (local purchasing power, spatial location, and dwelling capacity) enables municipal planners and tax assessors to evaluate zoning changes.
3. **Model Interpretability vs Non-Linearity:** While non-linear machine learning models often achieve higher fit, linear regression provides unambiguous, audited coefficient weights required by fair lending and regulatory oversight bodies.

---

## 2. Dataset Lineage, Provenance & Quality Audit

- **Canonical Repository:** StatLib / Carnegie Mellon University
- **Primary Literature Reference:**  
  R. Kelley Pace and Ronald Barry. *Sparse Spatial Autoregressions*, Statistics & Probability Letters, 33(3), pp. 291–297, 1997.
- **Verified Download URL:** `https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv`
- **SHA-256 Checksum:** `8A3727F4CF54AC1A327F69B1D5B4DB54C5834EA81C6E4EFC0D163300022A685E`
- **File Size:** 1,423,529 bytes.
- **Dimensions:** 20,640 census block group observations $\times$ 10 columns.
- **Target Variable:** `median_house_value` (USD continuous, ranging from \\$14,999 to \\$500,001).

### Structural Quality Audit

| Attribute | Data Type | Null Count | Null % | Range / Distinct Values | Role |
|:---|:---:|:---:|:---:|:---|:---:|
| `longitude` | float64 | 0 | 0.00% | -124.35 to -114.31 | Spatial coordinate |
| `latitude` | float64 | 0 | 0.00% | 32.54 to 41.95 | Spatial coordinate |
| `housing_median_age` | float64 | 0 | 0.00% | 1.0 to 52.0 years | Structural age |
| `total_rooms` | float64 | 0 | 0.00% | 2.0 to 39,320.0 | Tract physical scale |
| `total_bedrooms` | float64 | 207 | 1.003% | 1.0 to 6,445.0 (Imputed) | Living capacity |
| `population` | float64 | 0 | 0.00% | 3.0 to 35,682.0 | Tract population |
| `households` | float64 | 0 | 0.00% | 1.0 to 6,082.0 | Household count |
| `median_income` | float64 | 0 | 0.00% | 0.4999 to 15.0001 ($10k USD) | Economic purchasing power |
| `ocean_proximity` | object | 0 | 0.00% | 5 categories (`<1H OCEAN`, `INLAND`, `NEAR OCEAN`, `NEAR BAY`, `ISLAND`) | Geographic categorical |
| `median_house_value` | float64 | 0 | 0.00% | \\$14,999 to \\$500,001 | **Target Variable** |

---

## 3. Exploratory Data Analysis & Empirical Correlations

### Target Variable Distribution (`median_house_value`)
- **Descriptive Metrics:** Mean = \\$206,856; Median = \\$179,700; Standard Deviation = \\$115,396; Interquartile Range (IQR) = \\$145,125.
- **Skewness & Kurtosis:** Skewness = +0.978 (moderately right-skewed, typical of asset prices); Kurtosis = +0.154.
- **Census Capping Artifact:** 965 observations (4.68%) possess a value of exactly \\$500,001. In 1990 census procedures, values exceeding \\$500k were capped to preserve confidentiality.

### Pearson Correlation Analysis with House Prices

| Housing Attribute | Pearson Correlation ($r$) | Direction | Analytical Rationale |
|:---|:---:|:---:|:---|
| `median_income` | **+0.6881** | Positive | Primary purchasing power determinant in market pricing |
| `total_rooms` | +0.1342 | Positive | Neighborhood scale; positively correlated with tract size |
| `housing_median_age` | +0.1056 | Positive | Established, mature residential neighborhoods command slight premium |
| `households` | +0.0658 | Positive | Urban tract density |
| `total_bedrooms` | +0.0497 | Positive | Living capacity metric (collinear with rooms and households) |
| `population` | -0.0247 | Negative | Higher crowding per tract correlates with lower per-capita wealth |
| `longitude` | -0.0460 | Negative | Inland eastern shift away from Pacific coastal employment centers |
| `latitude` | -0.1442 | Negative | Northward shift towards agricultural and rural interior regions |

---

## 4. Leakage-Free Preprocessing Architecture

To ensure adherence to production standards and prevent data snooping:
1. **Partitioning:** An 80/20 train/test split was established with a fixed seed (`random_state=42`), producing 16,512 training observations and 4,128 test observations.
2. **Numerical Pipeline:** Numeric variables were routed to `SimpleImputer(strategy='median')` followed by `StandardScaler()`. Both imputer medians and scaler means/variances were computed strictly on `X_train`.
3. **Categorical Pipeline:** The `ocean_proximity` feature was routed to `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`. Dropping the first category (`<1H OCEAN`) prevents the dummy variable trap and defines `<1H OCEAN` as the reference baseline.
4. **Encapsulation:** The preprocessing logic was combined with the estimator into a unified Scikit-Learn `Pipeline`, ensuring identical transformations during cross-validation and test set inference.

```
Raw Input Data (20,640 x 10)
          │
          ▼
   80/20 Split (Seed 42)
   ├── Train: 16,512 records
   └── Test:   4,128 records
          │
          ▼
   ColumnTransformer
   ├── Numeric (8 Features): SimpleImputer(median) ──► StandardScaler()
   └── Categorical (1 Feature): OneHotEncoder(drop='first') ──► 4 Binary Dummies
          │
          ▼
   Transformed Design Matrix (12 Encoded Features)
          │
          ├──► LinearRegression (OLS)
          ├──► Ridge (alpha=1.0)
          ├──► Lasso (alpha=100.0)
          └──► RandomForestRegressor (Benchmark)
```

---

## 5. Comprehensive Model Benchmark & Evaluation

All models were evaluated on the exact same 4,128 held-out test observations. Cross-validation was performed using 5-fold CV exclusively on the 16,512 training observations.

### Authoritative Performance Comparison Table

| Model Architecture | Test MAE ($) | Test MSE ($^2$) | Test RMSE ($) | Test $R^2$ | 5-Fold CV RMSE (Train) | Rank |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy (Mean Baseline)** | \\$90,606.85 | $1.3107 \times 10^{10}$ | \\$114,485.64 | -0.0002 | N/A | 5 |
| **Linear Regression (OLS)** | **\\$50,670.49** | **$4.9083 \times 10^9$** | **\\$70,059.19** | **0.6254** | **\\$68,622.54 $\pm$ \\$1,438.77** | **1 (Linear)** |
| **Ridge Regression ($\alpha=1.0$)** | \\$50,678.11 | $4.9095 \times 10^9$ | \\$70,067.48 | 0.6253 | \\$68,621.67 $\pm$ \\$1,440.72$ | 2 (Linear) |
| **Lasso Regression ($\alpha=100.0$)** | \\$50,669.87 | $4.9116 \times 10^9$ | \\$70,082.90 | 0.6252 | \\$68,650.88 $\pm$ \\$1,416.77$ | 3 (Linear) |
| **Random Forest (100 Trees, Depth 16)**| \\$32,066.11 | $2.4328 \times 10^9$ | \\$49,323.89 | 0.8143 | \\$49,642.78 $\pm$ \\$583.20$ | 1 (Overall) |

### Key Regression Observations:
1. **OLS Supremacy among Linear Models:** Standard Ordinary Least Squares achieves the highest $R^2$ (0.6254) and lowest test RMSE (\\$70,059.19) among linear formulations.
2. **Behavior of Regularization Penalties:** Both Ridge ($L_2$) and Lasso ($L_1$) perform within 0.01% of OLS. In statistical theory, shrinkage estimators offer substantial variance reduction when $P \approx N$ or in severe multicollinearity settings. Here, with $N = 16,512$ observations across only 12 features, sample information dominates the prior penalty, leading to negligible shrinkage bias.
3. **Non-Linear Gap (Random Forest):** Random Forest achieves an $R^2$ of **0.8143** and an RMSE of **\\$49,323.89** (a \\$20,735 lower error than OLS). This discrepancy demonstrates that geographical coordinates form complex spatial price clusters that cannot be represented by a flat linear hyperplane without non-linear feature engineering.

---

### 6. Standardized Beta Weights & Conditional Associations

Because all numeric features were converted to standard score units ($z = \frac{x - \mu}{\sigma}$), the regression coefficients directly measure relative feature associations with property prices conditional on all other covariates remaining constant (*ceteris paribus* within this linear specification).

*Methodological Warning:* Because these estimates are derived from observational cross-sectional census data without experimental identification, they represent descriptive conditional associations and must not be interpreted as established causal economic drivers.

### Standardized Coefficient Impact Table

| Rank | Feature Name | Standardized Beta ($) | Direction | Conditional Association & Spatial Context |
|:---:|:---|:---:|:---:|:---|
| 1 | `ocean_proximity_ISLAND` | +\\$136,125.07 | Positive (+) | Conditional premium for island tracts (small sample: $N=5$) |
| 2 | `median_income` | +\\$75,167.77 | Positive (+) | Primary purchasing power association (+$75.2k per 1 SD income increase) |
| 3 | `latitude` | -\\$54,415.70 | Negative (-) | Northward geographic discount away from prime coastal metro areas |
| 4 | `longitude` | -\\$53,826.65 | Negative (-) | Eastward geographic discount toward interior mountain/desert tracts |
| 5 | `population` | -\\$43,403.43 | Negative (-) | Crowding penalty: higher density per block group conditionally associated with lower value |
| 6 | `total_bedrooms` | +\\$43,068.18 | Positive (+) | Living capacity association |
| 7 | `ocean_proximity_INLAND` | -\\$39,786.66 | Negative (-) | Severe inland discount relative to coastal baseline |
| 8 | `households` | +\\$18,382.20 | Positive (+) | Urban aggregation density |
| 9 | `housing_median_age` | +\\$13,889.87 | Positive (+) | Mature, established neighborhood amenity value |
| 10 | `total_rooms` | -\\$13,094.25 | Negative (-) | Negative sign due to multicollinearity conditioning on bedrooms |
| 11 | `ocean_proximity_NEAR BAY` | -\\$5,136.64 | Negative (-) | Minor variation relative to `<1H OCEAN` baseline |
| 12 | `ocean_proximity_NEAR OCEAN` | +\\$3,431.14 | Positive (+) | Coastal amenity alignment |

**Regression Intercept:** \\$219,899.78 (Expected property value at mean numerical levels for `<1H OCEAN`).

---

## 7. Residual Diagnostics, Gauss-Markov Nuance & Ceiling Effects

### 1. Low Aggregate Test Bias vs. Gauss-Markov Assumptions
- **Test Mean Residual:** \\$1,040.85 on test data.
- **Test Median Residual:** -\\$9,038.64.
- In Ordinary Least Squares, the residuals on the training set sum identically to zero by mathematical construction of the normal equations ($\mathbf{1}^T \mathbf{e} = 0$). On the held-out test data, a sample mean residual close to zero ($\approx \\$1,040.85$ relative to a mean price of $\approx \\$206k$) indicates low global aggregate bias on this test partition.
- **Crucially, this near-zero test residual mean does NOT verify that the Gauss-Markov assumptions hold.**
  - *Strict Exogeneity ($\mathbb{E}[\epsilon \mid \mathbf{X}] = 0$):* Cannot be verified merely by checking the sample mean of residuals.
  - *Homoscedasticity ($\text{Var}(\epsilon \mid \mathbf{X}) = \sigma^2$):* Is clearly violated, as residual variance expands noticeably at higher fitted values.
  - *Uncorrelated Errors ($\text{Cov}(\epsilon_i, \epsilon_j) = 0$):* Is likely violated by spatial proximity between neighboring census block groups (spatial autocorrelation).
  - Therefore, the small sample residual mean indicates only low global level offset, while Gauss-Markov assumptions of constant variance and error independence are violated in this spatial dataset.

### 2. Homoscedasticity & Variance Dispersion
- As shown in `fig4_residual_diagnostics.png`, residual variance is stable across low and moderate predicted price ranges (\\$100k–\\$300k).
- At fitted values above \\$350k, residual spread widens noticeably, indicating heteroscedasticity common in asset pricing.

### 3. Normal Q-Q Distribution
- Standardized residuals track the theoretical normal line closely between quantiles -2.0 and +1.5 ($R = 0.963$).
- Beyond +1.5 quantiles, heavy positive tails emerge, driven by premium properties whose true uncensored values far exceed the linear prediction.

### 4. Forensic Audit: Impact of the $500,001 Ceiling Cap & Validation Limitation

| Property Subset | Sample Count | Test MAE ($) | Test RMSE ($) | Analytical Interpretation |
|:---|:---:|:---:|:---:|:---|
| **Uncapped Homes (<\\$500k)** | 3,944 (95.54%) | \\$46,428.91 | **\\$61,848.69** | Normal market pricing; residuals are symmetric |
| **Capped Homes (\\$500k+)** | 184 (4.46%) | \\$141,587.86 | **\\$167,698.58** | **2.71x higher RMSE** due to administrative truncation |

*Validation Limitation for Valuation Practice:* The historical 1990 census administratively capped values at \\$500,001. For properties at this ceiling, true market values were unrecorded. While the model predicts values above \\$500,000 for many of these tracts based on high local income and coastal proximity, **it is impossible to validate whether those predictions are accurate, underestimated, or overestimated against actual uncapped values**. The apparent \\$167k RMSE reflects deviation from the artificial ceiling, not verified error against actual market transactions. In production deployment, automated valuation models must exclude administratively capped records, apply Tobit/censored regression, or ingest uncapped deed registries.

---

## 8. Strategic Business & Valuation Recommendations

### Recommendation 1: Deploy Hierarchical Regional Sub-Market Regressions
- **Observation:** A single statewide linear hyperplane cannot model localized coastal micro-climates without error.
- **Action:** Segment mortgage valuation portfolios into regional geographic zones (e.g., Bay Area Urban, Southern California Coastal, Central Valley Agricultural) and fit distinct linear regressions per region. This preserves regulatory compliance while drastically reducing spatial bias.

### Recommendation 2: Transition from Census Top-Coded Data to Deed Records
- **Observation:** Truncated census data inflates pricing error by 2.71x on high-value properties.
- **Action:** For loan underwriting above conventional conforming limits (jumbo mortgages), automated appraisal systems must ingest uncapped public county deed transaction registries rather than census summaries.

### Recommendation 3: Weight Neighborhood Income Trends in Collateral Monitoring
- **Observation:** Median income contributes +\\$75.2k per standard deviation shift, exceeding physical tract room counts.
- **Action:** Mortgage servicing institutions should monitor local wage growth and employment dynamics as early warning indicators for collateral risk before structural physical deterioration occurs.

---

## 9. Verification & Automated Test Summary

The implementation was verified using the automated test suite in `test_house_price_prediction.py`:

```
test_01_raw_data_file_exists_and_sha256: PASSED
test_02_raw_data_schema: PASSED
test_03_missing_value_profile: PASSED
test_04_preprocessor_pipeline_structure: PASSED
test_05_linear_regression_pipeline_artifact: PASSED
test_06_model_performance_metrics: PASSED
test_07_regularized_models_performance: PASSED
test_08_regression_coefficients_consistency: PASSED
test_09_residual_diagnostics_characteristics: PASSED
test_10_figure_artifacts_exist: PASSED
test_11_summary_tables_exist: PASSED
test_12_non_linear_benchmark_superiority: PASSED

Ran 12 tests in 1.804s — ALL 12 TESTS PASSED
```

All figures (`fig1` through `fig6`), summary CSV tables, serialized `.joblib` models, and the 33-cell Jupyter Notebook have been generated, executed headlessly, and archived.
