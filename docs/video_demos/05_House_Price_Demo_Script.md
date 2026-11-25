# Video Demo Script: Predicting House Prices with Linear Regression

**Task:** Project 05 (House Price Prediction) — Predicting House Prices with Linear Regression (Task #5 in Master Roadmap)  
**Project:** DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO (`Business-Analytics-Portfolio`)  
**Target Video Duration:** 3 to 5 Minutes  
**Presenter:** `Dhruv Goyal` *(Template document — replace with student full legal name prior to recording)*  
**Static Title Card:** Display `title_card_task5.png` for the first 2 seconds of the recording.

---

## 1. Video Recording Setup Checklist
- [ ] Screen resolution set to 1080p (1920x1080).
- [ ] Display `title_card_task5.png` for exactly 2 seconds as the opening title card.
- [ ] Have `House_Price_Prediction.ipynb` open in VS Code or Jupyter Lab with all cells pre-executed.
- [ ] Terminal open at project directory showing passing unit tests (`test_house_price_prediction.py`).
- [ ] Microphone tested with clear audio levels.

---

## 2. Timed Presentation Walkthrough

### 0:00 – 0:25: Opening & Title Card (25 Seconds)
- **Visual:** Display static title card `title_card_task5.png` for 2 seconds, then transition camera to webcam + screen sharing.
- **Presenter Dialogue:**
  > "Hello everyone! My name is `Dhruv Goyal`, and welcome to my submission demonstration for Project 05 (House Price Prediction): Predicting House Prices with Linear Regression in the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO program.
  > Today, I'll walk you through an econometric predictive modeling pipeline that forecasts California residential property values, enforces strict data leakage prevention, evaluates regularized regression models, and conducts comprehensive residual diagnostics."

### 0:25 – 1:00: Problem Statement & Dataset Lineage (35 Seconds)
- **Visual:** Scroll to Section 1 of `House_Price_Prediction.ipynb` showing the dataset overview and missing value profile.
- **Presenter Dialogue:**
  > "Accurate real estate valuation is fundamental to mortgage lending, property taxation, and macroeconomic policy. 
  > For this project, we utilize the canonical California Housing benchmark compiled by Pace and Barry from the 1990 U.S. Census Bureau data. It comprises 20,640 census block group observations across 10 continuous and categorical dimensions.
  > Data quality inspection reveals that exactly 207 records—representing 1% of the dataset—are missing values in the total_bedrooms column. All other 9 attributes are 100% complete, and dataset integrity has been verified via SHA-256 checksum."

### 1:00 – 1:45: Exploratory Data Analysis & Target Distribution (45 Seconds)
- **Visual:** Scroll to Section 2 showing Figure 1 (Target Distribution & Boxplot) and Figure 2 (Correlation Heatmap).
- **Presenter Dialogue:**
  > "Examining the target variable, median house value spans from $15,000 to $500,000, with a mean of $206,856 and a median of $179,700.
  > Notice a critical data provenance artifact: 965 block groups are artificially capped at exactly $500,001 due to historical census administrative truncation. We will see shortly how this truncates model residuals.
  > Looking at the correlation matrix in Figure 2, median household income is by far the single strongest linear predictor, showing a massive positive correlation of r = +0.688. 
  > Meanwhile, latitude shows a negative correlation of -0.144, capturing lower interior agricultural values compared to coastal southern California."

### 1:45 – 2:30: Leakage-Free Preprocessing & Model Training (45 Seconds)
- **Visual:** Scroll to Section 4 and Section 5 showing the `ColumnTransformer` code and Dummy Baseline evaluation.
- **Presenter Dialogue:**
  > "To guarantee production rigor and zero data leakage, all preprocessing is encapsulated inside a Scikit-Learn ColumnTransformer.
  > We impute missing bedrooms using the training median, standardize continuous features, and one-hot encode ocean proximity with drop='first' to prevent the dummy variable trap.
  > We partition the dataset into an 80% training set (16,512 rows) and a 20% held-out test set (4,128 rows).
  > Against a naive mean baseline that achieves an RMSE of $114,486 and an R² of zero, our Ordinary Least Squares Linear Regression model reduces RMSE by 38.8% to $70,059, achieving an R² of 0.6254 and a 5-fold cross-validation RMSE of $68,623."

### 2:30 – 3:15: Regularization & Non-Linear Benchmark (45 Seconds)
- **Visual:** Scroll to Section 7 and Section 11 showing the master benchmark table and Figure 6.
- **Presenter Dialogue:**
  > "We also evaluate regularized linear models—Ridge and Lasso. Because our training sample size of 16,500 records vastly exceeds our 12 encoded features, OLS is already well-conditioned, and Ridge and Lasso produce nearly identical performance within $25 RMSE.
  > Furthermore, when we benchmark against a Random Forest Regressor, performance leaps to an R² of 0.8143 and an RMSE of $49,324. 
  > This highlights a vital econometric lesson: geographic coordinates define complex spatial non-linear manifolds that planar linear equations cannot wrap around without non-linear terms."

### 3:15 – 3:55: Residual Diagnostics, Gauss-Markov Nuance & Ceiling Effects (40 Seconds)
- **Visual:** Scroll to Section 9 showing Figure 4 (Residual Diagnostics and Censoring Breakdown).
- **Presenter Dialogue:**
  > "In Section 9, we inspect residual diagnostics. The test set residual mean is approximately $1,040, showing low global aggregate offset. Crucially, this does not verify Gauss-Markov assumptions, as variance expands at higher prices and spatial autocorrelation affects neighboring tracts.
  > Looking at the fourth subplot in Figure 4, on the 3,944 uncapped homes, the linear model achieves an RMSE of $61,848.
  > On the 184 capped homes at $500,001, apparent RMSE inflates to $167,698 because the 1990 census truncated values. Because true uncapped market values were never recorded, predictions above $500,000 cannot be empirically validated against actual transactions."

### 3:55 – 4:30: Standardized Coefficients as Conditional Associations (35 Seconds)
- **Visual:** Scroll to Section 10 showing Figure 5 (Standardized Coefficients) and Section 12 (Recommendations).
- **Presenter Dialogue:**
  > "Analyzing our standardized beta weights in Figure 5, which represent conditional partial associations holding all other covariates constant:
  > Median income shows the strongest positive association, with +$75,168 per standard deviation increase.
  > Island tracts show a +$136,125 conditional difference over the baseline, while an inland location reflects a -$39,787 conditional discount.
  > Because these are derived from observational census data, they reflect descriptive statistical associations, not established causal levers.
  > For methodology, we recommend regional sub-market modeling to address spatial autocorrelation, and ingesting uncapped deed registries for luxury underwriting."

### 4:30 – 4:50: Verification & Closing (20 Seconds)
- **Visual:** Switch window to terminal showing `test_house_price_prediction.py` passing 12/12 tests in 1.8 seconds.
- **Presenter Dialogue:**
  > "All 12 automated unit tests pass in 1.8 seconds, covering data integrity, pipeline isolation, model artifacts, and diagnostics.
  > Thank you for watching my presentation for Data Analytics Portfolio Advanced Analytics & ML Task 1!"
