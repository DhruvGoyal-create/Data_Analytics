# DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — TASK TRACKER

**Track:** Data Analytics  
**Repository Name:** `Business-Analytics-Portfolio`  
**Execution Window:** 16 October 2026 – 15 November 2026 (30 Calendar Days)  
**Minimum Requirement:** 3 Tasks from either Foundational Analytics or Advanced Analytics & ML combined  
**Advanced Portfolio Target:** All 9 Tasks across both Foundational Analytics and Advanced Analytics & ML  
**Status Legend:**
- Task Lifecycle: `Not Started` | `In Progress` | `Implementation Complete` | `Testing Complete` | `Documentation Complete` | `Submission Prepared` | `Submission Confirmed`
- Checklist Items: `[ ] Not Started` | `[-] In Progress` | `[x] Implemented` | `[*] Tested` | `[!] Blocked`

---

## 1. Executive Summary Portfolio Matrix

| # | Task Code | Official Task Name | Level | Status | Checklist Progress | Target Milestone | Priority Rank |
|---|-----------|-------------------|-------|--------|-------------------|------------------|---------------|
| 1 | `PRJ-01` | EDA on Retail Sales Data | Foundational Analytics | `Testing Complete` | 9 / 9 | Milestone 1 (Min Gate) | 1 |
| 2 | `PRJ-02` | Customer Segmentation Analysis | Foundational Analytics | `Testing Complete` | 9 / 9 | Milestone 2 (Expand) | 4 |
| 3 | `PRJ-03` | Cleaning Data | Foundational Analytics | `Testing Complete` | 8 / 8 | Milestone 1 (Min Gate) | 2 |
| 4 | `PRJ-04` | Sentiment Analysis | Foundational Analytics | `Testing Complete` | 9 / 9 | Milestone 2 (Expand) | 6 |
| 5 | `PRJ-05` | Predicting House Prices with Linear Regression | Advanced Analytics & ML | `Testing Complete` | 11 / 11 | Milestone 2 (Expand) | 5 |
| 6 | `PRJ-06` | Wine Quality Prediction | Advanced Analytics & ML | `Testing Complete` | 10 / 10 | Milestone 1 (Min Gate) | 3 |
| 7 | `PRJ-07` | Fraud Detection | Advanced Analytics & ML | `Testing Complete` | 10 / 10 | Milestone 4 (Advanced Portfolio Target) | 8 |
| 8 | `PRJ-08` | Google Play Store Analysis | Advanced Analytics & ML | `Not Started` | 0 / 10 | Milestone 3 (Expand) | 7 |
| 9 | `PRJ-09` | Autocomplete and Autocorrect Data Analytics | Advanced Analytics & ML | `Not Started` | 0 / 10 | Milestone 4 (Advanced Portfolio Target) | 9 |

---

## 2. Exploratory & Descriptive Analytics Projects Detailed Tracking

### Task 1: EDA on Retail Sales Data (`PRJ-01`)
- **Directory:** `projects/01_Retail_Sales_EDA/`
- **Current Status:** `Testing Complete` (Notebook executed cleanly end-to-end; outputs, report, architecture, and video assets verified)
- **Dataset Requirements:** Multi-Year Retail Store Sales Dataset with Date (2011–2016), Demographics (Age, Age Group, Gender), Catalog (3 Categories, 17 Sub-categories, 130 Products), Quantity, Unit Cost, Unit Price, Revenue, Profit.
- **Estimated Effort:** 8 Hours (Completed in Session 02)
- **Technical Dependencies:** `python 3.13.6`, `pandas 2.2.3`, `numpy 2.4.4`, `matplotlib 3.10.3`, `seaborn 0.13.2`
- **Risk Level:** Low (Executed headlessly in 4.2 seconds)
- **Required Deliverables:**
  - `EDA_Retail_Sales.ipynb` (clean, executed, 27 cells with embedded outputs)
  - `data/retail_sales_data.csv` (113,036 rows x 18 columns)
  - 6 High-res exported charts (`outputs/figures/fig1..fig6.png`)
  - 4 Statistical summary CSVs (`outputs/tables/*.csv`)
  - Standalone project `README.md`
  - Reproducible `requirements.txt`
  - Analytical report in `docs/Reports/Week_01/Task_1_EDA_Retail_Sales_Report.md`
  - Architecture doc in `docs/Architecture/Week_01/Task_1_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_01/Task_1_Demo_Script.md`
  - Pre-rendered 1080p static title card in `docs/Video_Demo/Week_01/title_card.png`
- **Checklist:**
  - [*] Load dataset and perform initial inspection: shape, column dtypes, null check
  - [*] Descriptive statistics: mean, median, mode, standard deviation for all numerical columns
  - [*] Time series analysis: plot monthly and quarterly sales trends using line charts
  - [*] Customer demographics analysis: distribution of customer age groups, gender breakdown
  - [*] Product analysis: top 10 best-selling products; revenue by product category (bar chart)
  - [*] Heatmap: correlation matrix between numerical variables
  - [*] Additional visualization: non-obvious insight (gross profit margin across countries & categories)
  - [*] Markdown cells throughout the notebook with written observations after each chart
  - [*] Conclusion section: at least 3 specific, actionable business recommendations

---

### Task 2: Customer Segmentation Analysis (`PRJ-02`)
- **Directory:** `projects/02_Customer_Segmentation/`
- **Current Status:** `Testing Complete` (8/8 unit tests passing, clean dataset verified on disk)
- **Dataset Requirements:** E-commerce / retail transaction log with CustomerID, InvoiceDate, Quantity, UnitPrice (UCI Online Retail dataset, 541,909 raw records).
- **Estimated Effort:** 10 Hours (Actual: ~6 Hours)
- **Technical Dependencies:** `python`, `pandas`, `scikit-learn` (StandardScaler, KMeans), `matplotlib`, `seaborn`, `nbconvert`
- **Risk Level:** Low to Medium (Resolved via log1p transformation and K=4 optimization)
- **Required Deliverables:**
  - `Customer_Segmentation.ipynb` (Executed headlessly, 18 cells, 1.33 MB)
  - `customer_segmentation.py` (Reproducible standalone script)
  - `test_customer_segmentation.py` (8 unit tests passing in 0.03s)
  - `raw_online_retail.csv` (541k rows) & `rfm_customer_segments.csv` (4,338 customers with cluster labels & personas)
  - Standalone project `README.md` & `requirements.txt`
  - Report in `docs/Reports/Week_02/Task_2_Customer_Segmentation_Report.md`
  - Architecture doc in `docs/Architecture/Week_02/Task_2_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_02/Task_2_Demo_Script.md` & template title card `title_card_task2.png`
- **Checklist:**
  - [x] Load dataset and inspect structure; handle missing values and inconsistent data (filtered 135k null CustomerIDs, cancellations 'C', non-positive quantities/prices; retained 397,884 clean purchases)
  - [x] Descriptive statistics: average purchase value (£481 mean), purchase frequency (4.3 mean, 2.0 median), customer lifetime value (£2,054 mean)
  - [x] Feature selection: identify 2–3 key behavioural features for clustering (Recency, Frequency, Monetary)
  - [x] Data normalisation/standardisation before clustering (log1p transform + StandardScaler)
  - [x] Apply K-Means clustering — use the Elbow Method to determine optimal number of clusters (evaluated K=2..8; selected K=4 at elbow inflection & silhouette peak 0.3371)
  - [x] Visualise clusters using scatter plots (Recency vs Monetary, Frequency vs Monetary on log scales)
  - [x] Profile each cluster: calculate mean/median feature values per cluster and describe 4 customer personas (Champions, Regulars, Promising, At-Risk)
  - [x] Bar chart: number of customers per cluster (Figure 4)
  - [x] Insights section: what marketing action would you recommend for each segment? (Strategic marketing matrix and actionable playbooks)

---

### Task 3: Cleaning Data (`PRJ-03`)
- **Directory:** `projects/03_Data_Cleaning_Pipeline/`
- **Current Status:** `Testing Complete` (9/9 unit tests passing, clean dataset verified on disk)
- **Dataset Requirements:** A deliberately messy real-world or curated dirty dataset containing missing values, duplicates, mixed casing, invalid dates, and outliers. Sourced Ahmed Mohamed's Dirty Cafe Sales benchmark (10,000 records).
- **Estimated Effort:** 6 Hours (Actual: ~5 Hours)
- **Technical Dependencies:** `python`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `nbconvert`
- **Risk Level:** Low (Pure data manipulation, deterministic validation)
- **Required Deliverables:**
  - `Data_Cleaning_Pipeline.ipynb` (Executed headlessly, 21 cells, 361 KB)
  - `raw_dirty_cafe_sales.csv` (10,000 rows) & `cleaned_cafe_sales.csv` (9,974 rows, 0 nulls, 100% accounting identity)
  - Automated before-and-after quality comparison matrix (`outputs/tables/before_after_metrics.csv`)
  - Automated unit test suite (`test_cleaning_pipeline.py`, 9 tests passing in 0.06s)
  - Standalone project `README.md` & `requirements.txt`
  - Report in `docs/Reports/Week_01/Task_3_Data_Cleaning_Report.md`
  - Architecture doc in `docs/Architecture/Week_01/Task_3_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_01/Task_3_Demo_Script.md` & static title card `title_card_task3.png`
- **Checklist:**
  - [x] Load dataset and produce a "data quality report": nulls per column, duplicate rows, data type issues, value range anomalies
  - [x] Missing data handling: choose an appropriate strategy for each column and justify each choice in markdown (canonical menu pricing + deterministic algebraic back-calculation)
  - [x] Duplicate removal: identify and audit duplicate rows (0 full-row duplicates, 100% unique primary keys, 141 natural attribute collisions documented)
  - [x] Standardisation: normalise inconsistent formatting (casing, location values, dates to ISO `YYYY-MM-DD`)
  - [x] Outlier detection: use IQR method and Z-score to identify outliers in numeric columns; document decisions (defended retaining 259 transactions at $25.00 representing 5 Salads @ $5.00)
  - [x] Data type correction: ensure all columns have correct dtype (dates, IDs, currency, integers)
  - [x] Produce "before vs. after" summary table: null count, duplicate count, row count, dtype accuracy (`before_after_metrics.csv`)
  - [x] Save the cleaned dataset to a new CSV file (`cleaned_cafe_sales.csv`)


---

### Task 4: Sentiment Analysis (`PRJ-04`)
- **Directory:** `projects/04_Sentiment_Analysis/`
- **Current Status:** `Testing Complete` (All models evaluated, 8/8 automated unit tests passing, clean notebook executed headlessly)
- **Dataset Requirements:** Labeled sentiment dataset (Positive, Negative, Neutral). Sourced Kaggle Twitter Entity Sentiment benchmark (75,682 raw; 61,949 clean 3-class records across 32 entities).
- **Estimated Effort:** 10 Hours (Actual: ~5 Hours)
- **Technical Dependencies:** `python 3.13.6`, `pandas 2.2.3`, `scikit-learn 1.8.0` (MultinomialNB, LogisticRegression, LinearSVC, TfidfVectorizer), `nltk 3.9.1`, `wordcloud 1.9.6`, `matplotlib 3.10.3`, `seaborn 0.13.2`, `joblib 1.5.3`
- **Risk Level:** Low (Executed headlessly in 8.5 seconds; 8/8 tests pass)
- **Required Deliverables:**
  - `Sentiment_Analysis_Modeling.ipynb` (Executed headlessly, 21 cells, 1.02 MB)
  - `sentiment_analysis.py` (Reproducible standalone script)
  - `test_sentiment_analysis.py` (8 automated unit tests passing in 0.57s)
  - `models/sentiment_pipeline_svc.joblib` (6.1 MB serialized model + vectorizer pipeline)
  - `data/twitter_entity_sentiment_raw.csv` (10.49 MB) & `data/twitter_sentiment_clean.csv` (7.78 MB, 61,949 records)
  - 5 High-res exported charts (`outputs/figures/fig1..fig5.png`)
  - 4 Summary tables (`outputs/tables/*.csv`) including 5 concrete error analysis failure modes
  - Standalone project `README.md` & `requirements.txt`
  - Report in `docs/Reports/Week_03/Task_4_Sentiment_Analysis_Report.md`
  - Architecture doc in `docs/Architecture/Week_03/Task_4_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_03/Task_4_Demo_Script.md` & static title cards
- **Checklist:**
  - [x] Load dataset and inspect class distribution (positive/negative/neutral counts: Negative 22,624 [36.5%], Positive 20,932 [33.8%], Neutral 18,393 [29.7%])
  - [x] Text preprocessing pipeline: lowercase, URL/handle removal, punctuation cleaning, WordNet lemmatization, and negation-preserving stopword filtering
  - [x] Feature extraction: TF-IDF Vectorizer (10,000 unigram/bigram features, sublinear term-frequency scaling, min_df=3)
  - [x] Train/test split (80/20 stratified split: 48,382 train / 12,096 test samples)
  - [x] Train at least 2 classifiers: Multinomial Naive Bayes (76.33%), Logistic Regression (81.15%), and LinearSVC (83.97%)
  - [x] Evaluation: accuracy, precision, recall, F1-score, confusion matrix for each model (LinearSVC champion at 83.97% Acc / 0.8372 Macro F1)
  - [x] Visualisation: bar chart of sentiment distribution (Figure 1); WordCloud for each sentiment class (Figure 2)
  - [x] Error analysis: show 5 examples the model misclassified and discuss why (`outputs/tables/error_analysis_cases.csv`: sarcasm, negation inversion, contrastive clauses, slang polysemy, factual noise)
  - [x] Conclusion: which model performed best and what real-world application could this serve? (LinearSVC superiority; CX escalation, brand crisis sentinel, competitor intelligence)

---

## 3. Predictive Modeling & Advanced Analytics Projects Detailed Tracking

### Task 1: Predicting House Prices with Linear Regression (`PRJ-05`)
- **Directory:** `projects/05_House_Price_Prediction/`
- **Current Status:** `Testing Complete` (All models evaluated, 12/12 automated unit tests passing in 1.8s, notebook executed headlessly, pipeline serialized)
- **Dataset Requirements:** California Housing Dataset (StatLib / 1990 U.S. Census, 20,640 records x 10 columns, SHA-256 verified, Public domain).
- **Estimated Effort:** 8 Hours (Actual: ~5 Hours)
- **Technical Dependencies:** `python 3.13.6`, `pandas 2.2.3`, `numpy 2.4.4`, `scikit-learn 1.8.0` (LinearRegression, Ridge, Lasso, RandomForestRegressor, ColumnTransformer), `matplotlib 3.10.3`, `seaborn 0.13.2`, `joblib 1.5.3`
- **Risk Level:** Low to Medium (Resolved multicollinearity analysis, verified well-conditioned OLS, and quantified $500k census ceiling truncation effect)
- **Required Deliverables:**
  - `House_Price_Prediction.ipynb` (Executed headlessly, 33 cells, 1.76 MB)
  - `house_price_prediction.py` (Reproducible standalone script with CLI)
  - `test_house_price_prediction.py` (12 automated unit tests passing in 1.8s)
  - `models/linear_regression_pipeline.joblib` (Champion OLS pipeline)
  - `models/ridge_pipeline.joblib` (Ridge regularized pipeline)
  - `models/lasso_pipeline.joblib` (Lasso regularized pipeline)
  - `models/random_forest_pipeline.joblib` (Non-linear spatial ensemble benchmark)
  - `data/housing.csv` (20,640 rows x 10 columns, SHA-256 verified)
  - 6 High-res exported charts (`outputs/figures/fig1..fig6.png`)
  - 5 Summary tables (`outputs/tables/*.csv`) including residual diagnostics breakdown
  - Standalone project `README.md` & `requirements.txt`
  - Report in `docs/Reports/Week_02/Task_5_House_Price_Prediction_Report.md`
  - Architecture doc in `docs/Architecture/Week_02/Task_5_House_Price_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_02/Task_5_Demo_Script.md` & 1080p static title cards
- **Checklist:**
  - [*] Load dataset and perform EDA: null check, descriptive statistics, distribution of target variable (`median_house_value`)
  - [*] Feature selection discussion: identify likely predictors; explain reasoning in markdown (`median_income`, spatial coordinates, living capacity vs density)
  - [*] Handle missing values (`total_bedrooms` 207 nulls) and encode categorical features (`ocean_proximity` One-Hot with `drop='first'`)
  - [*] Correlation heatmap: identify features most correlated with house price (`median_income` $r = +0.688$)
  - [*] Train/test split (80/20 with `random_state=42`)
  - [*] Train a Linear Regression model using scikit-learn
  - [*] Evaluate model using: MAE (\\$50,670.49), MSE ($4.908 \times 10^9$), RMSE (\\$70,059.19), and $R^2$ score (0.6254) plus 5-Fold CV (\\$68,622.54 $\pm$ \\$1,438.77)
  - [*] Scatter plot: actual prices vs. predicted prices with $y = x$ reference line
  - [*] Residual plot: check that residuals are randomly distributed without systematic bias and quantify $500k ceiling effect
  - [*] Coefficient analysis: highest positive impact (`median_income` +\\$75,168) and highest negative impact (`latitude` -\\$54,416, `ocean_proximity_INLAND` -\\$39,787)
  - [*] (Bonus) Compare Linear Regression against Ridge ($\alpha=1.0$) and Lasso ($\alpha=100.0$) regularized models, plus Random Forest benchmark ($R^2 = 0.8143$)

---

### Task 2: Wine Quality Prediction (`PRJ-06`)
- **Directory:** `projects/06_Wine_Quality_Prediction/`
- **Current Status:** `Testing Complete` (All models trained, 11/11 automated unit tests passing in 0.11s, notebook executed headlessly)
- **Dataset Requirements:** UCI Wine Quality dataset (1,599 red Vinho Verde observations, 11 continuous physicochemical features, 1 sensory target, SHA-256 verified, CC BY 4.0).
- **Estimated Effort:** 8 Hours (Actual: ~5 Hours)
- **Technical Dependencies:** `python 3.13.6`, `pandas 2.2.3`, `numpy 2.4.4`, `scikit-learn 1.8.0`, `matplotlib 3.10.3`, `seaborn 0.13.2`, `joblib 1.5.3`
- **Risk Level:** Low (Resolved class imbalance via binary threshold and balanced class weights)
- **Required Deliverables:**
  - `Wine_Quality_Prediction.ipynb` (Executed headlessly, 19 cells, 1.05 MB)
  - `wine_quality_prediction.py` (Reproducible standalone script with CLI)
  - `test_wine_quality.py` (11 automated unit tests passing in 0.11s)
  - `models/wine_quality_random_forest_pipeline.joblib` (Champion RF pipeline)
  - `models/wine_quality_sgd_classifier_pipeline.joblib` (SGD linear pipeline)
  - `models/wine_quality_support_vector_svc_pipeline.joblib` (SVC kernel pipeline)
  - `data/winequality-red.csv` (1,599 rows) & `data/winequality-white.csv` (4,898 rows)
  - 6 High-res exported charts (`outputs/figures/fig1..fig6.png`)
  - 5 Summary tables (`outputs/tables/*.csv`) including misclassification audit
  - Standalone project `README.md` & `requirements.txt`
  - Report in `docs/Reports/Week_02/Task_6_Wine_Quality_Report.md`
  - Architecture doc in `docs/Architecture/Week_02/Task_6_Wine_Quality_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_02/Task_6_Demo_Script.md` & 1080p static title card `title_card_task6.png`
- **Checklist:**
  - [*] Load Wine Quality dataset and inspect structure; check class distribution of quality scores
  - [*] EDA: distribution plots for all chemical features; correlation heatmap
  - [*] Discuss class imbalance: underrepresented quality scores and modeling implications (82.5% central concentration in scores 5 & 6)
  - [*] Feature engineering: bin quality scores into binary (good/bad, threshold >= 7) with justification and empirical proof of 3-tier failure
  - [*] Train/test split with stratification to preserve class ratios (13.57% positive across train and test splits)
  - [*] Train 3 classifiers: Random Forest, Stochastic Gradient Descent (SGD), and Support Vector Classifier (SVC)
  - [*] Evaluate each: accuracy, classification report (precision, recall, F1, ROC-AUC, PR-AUC), confusion matrix
  - [*] Feature importance chart for the Random Forest model (MDI Gini & Permutation Importance)
  - [*] Comparison table: summarise the performance of all 3 models side-by-side
  - [*] Conclusion: which model is most suitable for deployment and why? (Random Forest selected for 92.86% precision to protect brand equity)

---

### Task 3: Fraud Detection (`PRJ-07`)
- **Directory:** `projects/07_Credit_Card_Fraud_Detection/`
- **Current Status:** `Testing Complete` (12/12 automated unit tests passing, notebook executed with full outputs, all deliverables verified)
- **Dataset Requirements:** Credit Card Fraud Detection benchmark dataset (284,807 transactions, 492 fraud cases, 0.1727% fraud rate).
- **Estimated Effort:** 12 Hours (Actual: ~8 Hours)
- **Technical Dependencies:** `python 3.13.6`, `pandas 2.2.3`, `numpy 2.4.4`, `scikit-learn 1.6.1`, `imblearn 0.14.2`, `matplotlib 3.10.3`, `seaborn 0.13.2`, `nbconvert`
- **Risk Level:** High (Managed via `imblearn.pipeline.Pipeline` with SMOTE strictly inside cross-validation/training folds and `RobustScaler` on skewed features)
- **Required Deliverables:**
  - `Fraud_Detection.ipynb` (Executed headlessly, 27 cells, 540 KB, zero tracebacks)
  - `fraud_detection.py` (Reproducible standalone execution script with CLI)
  - `test_fraud_detection.py` (12 automated unit tests passing in 11.8s)
  - `creditcard.csv` (284,807 rows x 31 columns, SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`)
  - 6 High-res exported charts (`outputs/figures/fig1..fig6.png`)
  - 7 Summary tables (`outputs/tables/*.csv`) including financial loss prevention audit and scalability throughput
  - Serialized model artifacts: `models/fraud_detection_random_forest.joblib`, `models/fraud_detection_logistic_regression.joblib`, `models/fraud_detection_smote_lr.joblib`
  - Standalone project `README.md` & `requirements.txt`
  - Report in `docs/Reports/Week_03/Task_7_Fraud_Detection_Report.md`
  - Architecture doc in `docs/Architecture/Week_03/Task_7_Fraud_Detection_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_03/Task_7_Demo_Script.md` & 1080p static title cards
- **Checklist:**
  - [*] Load dataset and analyse class imbalance: percentage of fraudulent transactions (492 of 284,807 = 0.1727%, 1:578 imbalance)
  - [*] EDA: transaction amount distribution for fraud vs. non-fraud; time-of-day analysis (fraud mean $122.21 vs legitimate $88.29; observed nocturnal fraud prevalence peaking at 1.71% at 2:00 AM)
  - [*] Discuss why standard accuracy is misleading for imbalanced fraud datasets (Dummy baseline achieves 99.83% accuracy with 0% recall and $0 fraud recovered)
  - [*] Apply class imbalance technique: SMOTE oversampling or class_weight='balanced' without data leakage (pipeline encapsulation with SMOTE sampling_strategy=0.1 and balanced logistic regression strictly on training split)
  - [*] Stratified train/test split (80/20 stratified split: 227,845 train with 394 frauds, 56,962 test with 98 frauds)
  - [*] Train at least 2 models: Logistic Regression + (Random Forest or XGBoost) (Trained 5 models: Dummy, Unweighted LR, Balanced LR, SMOTE+LR, and Random Forest champion)
  - [*] Evaluate using: Precision, Recall, F1-Score, and AUC-ROC curve (Random Forest: 81.82% Prec, 82.65% Rec, 0.8223 F1, 0.9766 ROC-AUC, 0.8187 PR-AUC)
  - [*] Explain which metric matters most for fraud detection and why (Recall vs. Precision trade-off & financial cost curve: PR-AUC and ROC-AUC evaluated complementarily)
  - [*] Feature importance or coefficient analysis (Identified V14, V10, V12, V17 as primary negative indicators, V4 and V11 as positive indicators)
  - [*] Discuss scalability: handling 1 million transactions per hour in production (Proposed architecture blueprint for 277.8 TPS sustained, 833.3 TPS peak, with measured Python vs ONNX needs)

---

### Task 4: Google Play Store Analysis (`PRJ-08`)
- **Directory:** `projects/08_Google_Play_Store_Analytics/`
- **Current Status:** `Not Started`
- **Dataset Requirements:** Google Play Store Apps dataset + Google Play Store User Reviews dataset.
- **Estimated Effort:** 10 Hours
- **Technical Dependencies:** `python`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `textblob`, `plotly`
- **Risk Level:** Medium (Dirty strings in Installs/Price/Size, two-table join, sentiment scoring)
- **Required Deliverables:**
  - `Google_Play_Store_Analysis.ipynb`
  - Interactive Plotly HTML chart (`interactive_playstore_viz.html`)
  - Standalone project `README.md`
  - Report in `docs/Reports/Week_03/Task_4_Google_Play_Store_Report.md`
  - Architecture doc in `docs/Architecture/Week_03/Task_4_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_03/Task_4_Demo_Script.md`
- **Checklist:**
  - [ ] Load Play Store apps dataset and user reviews dataset separately
  - [ ] Data cleaning: fix incorrect data types (Installs, Price, Size), handle nulls, remove duplicates
  - [ ] Category analysis: bar chart of app distribution across categories; saturation analysis
  - [ ] Ratings analysis: distribution of app ratings; average rating by category
  - [ ] Size and installs analysis: scatter plot of app size vs. number of installs; correlation test
  - [ ] Pricing analysis: paid vs. free distribution; price distribution for paid apps; qualified revenue estimate
  - [ ] Sentiment analysis on user reviews: classify reviews as positive/negative/neutral using TextBlob/VADER
  - [ ] Sentiment by category: positive/negative sentiment breakdown per category
  - [ ] Interactive visualisation using Plotly
  - [ ] Conclusion: 3 data-driven insights for an app developer launching a new app

---

### Task 5: Autocomplete and Autocorrect Data Analytics (`PRJ-09`)
- **Directory:** `projects/09_Autocomplete_Autocorrect/`
- **Current Status:** `Not Started`
- **Dataset Requirements:** Public domain text corpus (Project Gutenberg books, e.g. Sherlock Holmes, Frankenstein, or Jane Austen).
- **Estimated Effort:** 12 Hours
- **Technical Dependencies:** `python`, `pandas`, `nltk`, `collections`, `matplotlib`, `pyspellchecker` / `textdistance`
- **Risk Level:** High (N-gram sparsity, algorithmic correctness of custom Levenshtein / Damerau-Levenshtein, evaluation definition)
- **Required Deliverables:**
  - `Autocomplete_Autocorrect_Engine.ipynb`
  - Prefix testing evaluation matrix (10 prefixes x 3 predictions)
  - Misspelling test set results (20 words with precision/recall)
  - Standalone project `README.md`
  - Report in `docs/Reports/Week_04/Task_5_Autocomplete_Autocorrect_Report.md`
  - Architecture doc in `docs/Architecture/Week_04/Task_5_Architecture.md`
  - Video demo script in `docs/Video_Demo/Week_04/Task_5_Demo_Script.md`
- **Checklist:**
  - [ ] Collect or download a large legally usable text corpus (Project Gutenberg)
  - [ ] NLP preprocessing: tokenisation, lowercasing, punctuation removal, stopword handling
  - [ ] Autocomplete implementation: frequency-based n-gram model (bigram/trigram) predicting next word given prefix
  - [ ] Test autocomplete on at least 10 input prefixes and display top 3 predictions per prefix
  - [ ] Autocorrect implementation: edit-distance based correction using Levenshtein / pyspellchecker
  - [ ] Test autocorrect on at least 20 deliberately misspelled words and measure correction accuracy
  - [ ] Performance metrics: define and calculate precision and recall for both autocomplete and autocorrect
  - [ ] Algorithm comparison: compare at least 2 approaches (frequency vs. smoothed/neural or Levenshtein vs. Jaro-Winkler)
  - [ ] Visualisation: bar chart of top 20 most frequent words; confusion matrix / error distribution for autocorrect
  - [ ] Discussion: limitations of implementation vs. production systems (Gboard, iOS keyboard)

---

## 4. Overall Portfolio Quality Gatekeeper

| Gate | Requirement Description | Verification Evidence | Completed Date |
|------|-------------------------|-----------------------|----------------|
| **Gate 1: Repository Standard** | Repo named `Business-Analytics-Portfolio`, clean Git history, no secrets, `.gitignore` active | `.git` repo active, `.gitignore` configured | In Progress |
| **Gate 2: Minimum Threshold** | At least 3 tasks fully completed, tested, and documented across Foundational Analytics & 2 | Milestone 1 verification tests passing | Target: 24-Oct-2026 |
| **Gate 3: Expanded Coverage** | At least 7 tasks fully completed, tested, and documented | Milestone 3 verification tests passing | Target: 06-Nov-2026 |
| **Gate 4: Advanced Portfolio Target (All 9)** | All 9 tasks completed with 100% official checklist fulfillment | 9/9 tasks verified end-to-end | Target: 11-Nov-2026 |
| **Gate 5: Video & Peer Compliance** | Demo scripts & recordings prepared with 2s title card; 2 peer comments planned | Video demo folder & LinkedIn post scripts | Target: 13-Nov-2026 |
| **Gate 6: Final Submission** | Data Analytics Portfolio Task Submission Form completed with repo link | Final form confirmation receipt | Target: 14-Nov-2026 |
