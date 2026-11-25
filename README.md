# Data Analytics & Business Intelligence Portfolio
## Industry-Standard Analytics, Predictive Modeling & Machine Learning Systems

**Author:** Dhruv Goyal ([GitHub Profile](https://github.com/DhruvGoyal-create))  
**Specialization:** Data Analytics, Business Intelligence & Machine Learning  
**Repository:** `Secondary A2-Bussiness Analytics`  
**Remote URL:** [https://github.com/DhruvGoyal-create/Data_Analytics.git](https://github.com/DhruvGoyal-create/Data_Analytics.git)  
**Status:** Production Verified (81/81 Automated Unit Tests Passing across all 8 modules)

---

## Portfolio Architecture & Project Overview

This repository houses an end-to-end, production-grade Data Analytics and Business Intelligence portfolio. Each project is engineered with strict reproducibility standards: dedicated automated unit test suites, leakage-free Scikit-Learn pipelines, headless Jupyter Notebook execution, statistical summary tables, high-resolution visual analytics (300 DPI), and comprehensive technical documentation.

| # | Project Code | Project Title | Domain & Tech Stack | Key Methodologies & Business Impact | Verification |
|:---:|:---:|:---|:---|:---|:---:|
| **01** | `PRJ-01` | **[Retail Sales Exploratory Data Analysis](projects/01_Retail_Sales_EDA/)** | Retail Operations<br>`Pandas`, `Matplotlib`, `Seaborn` | Analyzed 113,036 transactions ($84.8M clean revenue). Audited 1,000 exact duplicates (-$444K impact). Reconciled cost/profit identities and age-demographic profit margin drivers. | **8 / 8 Tests PASS** |
| **02** | `PRJ-02` | **[Customer Segmentation & RFM Clustering](projects/02_Customer_Segmentation/)** | Customer Intelligence<br>`Scikit-Learn`, `K-Means`, `StandardScaler` | Segmented 4,338 unique customers into 4 behavioural personas (Champions, Regulars, Promising, At-Risk) using log-transformed RFM features with optimal Elbow/Silhouette verification. | **8 / 8 Tests PASS** |
| **03** | `PRJ-03` | **[Data Cleaning & Financial Imputation Pipeline](projects/03_Data_Cleaning_Pipeline/)** | Data Quality Engineering<br>`Pandas`, `NumPy`, `Data Validation` | Deterministically recovered 969 items, 479 quantities, 533 prices, and 502 totals across 10,000 messy records using menu lookup matrices and algebraic constraints ($Revenue = Qty \times Price$). | **9 / 9 Tests PASS** |
| **04** | `PRJ-04` | **[Entity Sentiment Analysis NLP Engine](projects/04_Sentiment_Analysis/)** | Natural Language Processing<br>`NLTK`, `TF-IDF`, `LinearSVC`, `Naive Bayes` | Preprocessed 74,682 social posts. Trained TF-IDF classifiers on 3-class sentiment (Champion LinearSVC: 83.31% Accuracy, 0.8322 Macro-F1). Audited qualitative NLP failure modes. | **10 / 10 Tests PASS** |
| **05** | `PRJ-05` | **[Predicting House Prices with Econometric OLS](projects/05_House_Price_Prediction/)** | Econometrics & Real Estate<br>`Scikit-Learn`, `OLS`, `Ridge`, `Lasso` | Evaluated 20,640 California census block groups. Built leakage-free `ColumnTransformer`. Benchmarked OLS ($R^2 = 0.6254$, RMSE $70,059) vs Random Forest ($R^2 = 0.8143$). Audited $500k ceiling bias. | **12 / 12 Tests PASS** |
| **06** | `PRJ-06` | **[Wine Quality Biochemical Classification](projects/06_Wine_Quality_Prediction/)** | Quality Assurance & Viticulture<br>`Scikit-Learn`, `Random Forest`, `SVC` | Addressed 86.4% class skew on sensory ratings. Champion Random Forest achieved 94.06% Accuracy, 92.86% Precision, 0.9523 ROC-AUC. Audited 240 duplicates and cross-boundary leakage. | **12 / 12 Tests PASS** |
| **07** | `PRJ-07` | **[Credit Card Fraud Detection](projects/07_Credit_Card_Fraud_Detection/)** | Financial Fraud & Risk Underwriting<br>`Imblearn (SMOTE)`, `Random Forest`, `PR-AUC` | Analyzed extreme 0.1727% fraud skew across 284,807 transactions. Built SMOTE pipelines. Champion Random Forest achieved 0.8187 PR-AUC, 81.82% Precision, 82.65% Recall ($6,552 fraud prevented, 18 false alarms). | **12 / 12 Tests PASS** |
| **08** | `PRJ-08` | **[Google Play Store Market Analytics](projects/08_Google_Play_Store_Analytics/)** | Mobile App Store Intelligence<br>`Pandas`, `Scipy`, `Plotly`, `TextBlob` | Cleaned 10,841 app listings & 64,295 reviews. Handled Row 10472 shift. Sized category saturation, size elasticity (rho = +0.31), upfront monetization, and interactive 4D Plotly landscape. | **10 / 10 Tests PASS** |

---

## Repository Directory Layout

```
Secondary A2-Bussiness Analytics/
├── .gitignore                               # Production Python gitignore
├── README.md                                # Master portfolio documentation
├── requirements.txt                         # Pinned global dependencies
├── run_all_tests.py                         # Master test runner across all modules
│
├── projects/                                # Independent self-contained projects
│   ├── 01_Retail_Sales_EDA/
│   │   ├── EDA_Retail_Sales.ipynb           # Executed Jupyter Notebook
│   │   ├── test_retail_sales_eda.py         # Automated unit test suite (8 tests)
│   │   ├── README.md & requirements.txt
│   │   ├── data/ & outputs/ (figures, tables)
│   ├── 02_Customer_Segmentation/
│   ├── 03_Data_Cleaning_Pipeline/
│   ├── 04_Sentiment_Analysis/
│   ├── 05_House_Price_Prediction/
│   ├── 06_Wine_Quality_Prediction/
│   ├── 07_Credit_Card_Fraud_Detection/
│   └── 08_Google_Play_Store_Analytics/
│
└── docs/                                    # Centralized technical documentation
    ├── reports/                             # Formal analytical whitepapers & reports
    ├── architecture/                        # System architecture specifications
    ├── presentations/                       # 1080p presentation title cards
    ├── video_demos/                         # Timecoded demo narration scripts
    └── strategy/                            # Technical roadmap & decision logs
```

---

## Global Verification & Quality Standards

To run the unified test regression suite across all active project modules:

```bash
python run_all_tests.py
```

Expected verification output:
```
REGRESSION SUITE SUMMARY: 81/81 TESTS PASSED (100% Pass Rate)
Total Failures: 0 | Total Errors: 0
```

---

## Contact & Profile

- **Author:** Dhruv Goyal
- **GitHub:** [https://github.com/DhruvGoyal-create](https://github.com/DhruvGoyal-create)
- **Repository:** [https://github.com/DhruvGoyal-create/Data_Analytics.git](https://github.com/DhruvGoyal-create/Data_Analytics.git)
