# Advanced Analytics & ML — Task 3: Fraud Detection

## Machine Learning Pipeline for Extreme Class Imbalance & Real-Time Financial Fraud Prevention

**Track:** Data Analytics  
**Organization:** Data Analytics Portfolio (`Business-Analytics-Portfolio`)  
**Task Code:** `PRJ-07` (Task #7 in Master Roadmap)  
**Author:** `Dhruv Goyal` *(Template document — replace with student legal name prior to submission)*  
**Current Status:** `Testing Complete` (All models trained, 12/12 automated unit tests passing, notebook executed headlessly)  
**Submission Gate:** Strictly **PENDING** until student customizes title cards and records video demonstration.

---

## 1. Project Overview & Business Problem

Financial fraud costs the global payments ecosystem tens of billions of dollars annually. When deploying automated decision engines on live transaction streams, machine learning systems face an acute operational challenge: **extreme class imbalance**.

In legitimate banking operations, fraudulent events represent a tiny fraction of total activity. Consequently, standard evaluation metrics like classification accuracy completely collapse—a naive model predicting "legitimate" for all transactions yields 99.8% accuracy while missing 100% of financial crimes.

This project delivers an **end-to-end, production-grade fraud detection machine learning system** built on the canonical **Credit Card Fraud Detection benchmark** (ULB Machine Learning Group / Dal Pozzolo et al., 2015). The system explores diurnal fraud periodicity, addresses class imbalance through cost-sensitive weighting and pipeline-encapsulated SMOTE oversampling, benchmarks multiple model paradigms, conducts a comprehensive financial loss prevention audit, and designs a microservice architecture capable of processing **1,000,000 transactions per hour** within a strict <15 ms P99 latency budget.

---

## 2. Official Task Checklist & Compliance

| # | Official Checklist Requirement | Implementation Details | Verification Status |
|---|--------------------------------|------------------------|:-------------------:|
| 1 | **Load dataset & analyze class imbalance** | Loaded 284,807 transactions; confirmed 492 fraud cases (0.1727%) vs 284,315 legitimate transactions (99.8273%). | **Verified** |
| 2 | **EDA: Amount distributions & Time-of-day** | Analyzed log-transformed transaction amounts (`fig1`) and identified observed nocturnal fraud prevalence rise at 2:00 AM (`fig2`). | **Verified** |
| 3 | **Discuss why accuracy is misleading** | Detailed mathematical exposition of the Accuracy Paradox in markdown; documented financial cost of false negatives. | **Verified** |
| 4 | **Apply class imbalance handling** | Implemented `class_weight='balanced'` and `SMOTE(sampling_strategy=0.1)` inside leakage-free Scikit-Learn pipelines. | **Verified** |
| 5 | **Stratified train/test split** | Stratified 80/20 split (227,845 train / 56,962 test; seed 42) preserving exact 0.172% fraud proportion. | **Verified** |
| 6 | **Train at least 2 models** | Trained 4 models: Unweighted Logistic Regression, Balanced Logistic Regression, SMOTE + LR, and Balanced Random Forest. | **Verified** |
| 7 | **Evaluate using Precision, Recall, F1, PR-AUC** | Evaluated models on Precision, Recall, F1, Balanced Accuracy, ROC-AUC, and Average Precision (PR-AUC). | **Verified** |
| 8 | **Explain metric trade-offs (Recall vs Precision)** | In-depth operational analysis comparing fraud dollar recovery against customer checkout friction. | **Verified** |
| 9 | **Feature importance / coefficient analysis** | Extracted top PCA predictors (`V14`, `V10`, `V12`, `V17`, `V4`) via linear weights and tree MDI (`fig6`). | **Verified** |
| 10 | **Discuss scalability (1M transactions/hour)** | Detailed microservice architecture with TPS calculations, latency budgets, and caching strategy. | **Verified** |

---

## 3. Dataset Lineage & Integrity

- **Dataset Name:** Credit Card Fraud Detection Benchmark
- **Original Source:** Machine Learning Group (MLG), Université Libre de Bruxelles (ULB) & Worldline.
- **Reference Publication:** Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. In IEEE Symposium Series on Computational Intelligence (SSCI), 2015.
- **Local File Location:** `data/creditcard.csv` (150,828,752 bytes, 143.84 MB)
- **SHA-256 Checksum:** `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`
- **Total Dimensions:** 284,807 transactions $\times$ 31 columns.
- **Feature Structure:**
  - `Time`: Elapsed seconds from first transaction in the dataset (spanning 48.0 hours).
  - `V1` through `V28`: Principal Component Analysis (PCA) components engineered to preserve cardholder confidentiality.
  - `Amount`: Transaction amount in USD.
  - `Class`: Binary target variable (1 = Fraudulent, 0 = Legitimate).
- **Missing Values:** Exactly 0 null/missing cells across the entire dataset.
- **Duplicate Records:** 1,081 exact duplicate rows (1,064 legitimate, 17 fraud) reflecting automated identical micro-billing transactions.

---

## 4. Empirical Performance Benchmarks

All models were evaluated on the held-out stratified 20% test partition ($N = 56,962$ transactions containing 98 fraud cases):

| Model Architecture | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Avg Prec) | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Fit Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy (Majority Baseline)** | 99.83% | 50.00% | 0.00% | 0.00% | 0.0000 | *N/A* | *N/A* | 0 | 0 | 98 | 0.01s |
| **Logistic Regression (Unweighted)** | 99.92% | 82.13% | 82.89% | 64.29% | 0.7241 | 0.9592 | 0.7429 | 63 | 13 | 35 | 0.62s |
| **Logistic Regression (Balanced)** | 97.56% | 94.70% | 6.11% | **91.84%** | 0.1146 | 0.9721 | 0.7156 | **90** | 1,382 | **8** | 1.13s |
| **SMOTE + Logistic Regression** | 99.70% | 94.25% | 35.22% | 88.78% | 0.5043 | 0.9667 | 0.7523 | 87 | 160 | 11 | 2.14s |
| **Random Forest (Balanced Weights)**| **99.94%** | **91.31%** | **81.82%** | **82.65%** | **0.8223** | **0.9766** | **0.8187** | **81** | **18** | **17** | 10.62s |

### Key Analytical Insights:
1. **The Deception of Accuracy:** The Dummy Classifier achieves 99.83% accuracy while allowing \$10,645 in fraud to pass completely undetected.
2. **The Precision-Recall Trade-off:**
   - Balanced Logistic Regression achieves the highest recall (91.84%, catching 90 of 98 frauds), but its 6.11% precision generates 1,382 false alarms.
   - SMOTE + Logistic Regression cuts false alarms by 88% (down to 160) while maintaining an 88.78% recall.
   - **Random Forest** delivers the best overall commercial balance: 81.82% Precision, 82.65% Recall, and the highest PR-AUC (0.8187), generating only 18 false alarms across 56,864 legitimate transactions.

---

## 5. Financial Fraud Loss Prevention Audit

Translating statistical confusion matrices into transaction dollar exposure and customer friction on the test set (\$10,644.93 total fraud exposure):

| Model Architecture | Dollars Prevented ($) | Dollars Lost ($) | Detection Rate (%) | False Positive Count | Legitimate Volume Frictioned ($) | Operational Assessment |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Dummy Baseline** | \$0.00 | \$10,644.93 | 0.00% | 0 | \$0.00 | Complete financial loss |
| **Logistic Regression (Unweighted)** | \$4,506.29 | \$6,138.64 | 42.33% | 13 | \$902.67 | Unacceptable fraud leakage (misses \$6.1k) |
| **Logistic Regression (Balanced)** | **\$9,986.32** | **\$658.61** | **93.81%** | 1,382 | \$443,861.88 | Maximum loss prevention; heavy queue congestion |
| **SMOTE + Logistic Regression** | \$8,818.43 | \$1,826.50 | 82.84% | 160 | \$37,668.58 | Strong compromise for mid-tier risk tiers |
| **Random Forest (Balanced)** | \$6,551.61 | \$4,093.32 | 61.55% | **18** | **\$950.71** | Frictionless champion (18 false alarms) |

*Accounting Note:* "Dollars Prevented" represents detected fraud exposure (test true positives). Interception provides the opportunity for operational challenge; it does not equal guaranteed cash savings due to chargeback liability and dispute outcome dependencies. "Friction" represents legitimate dollars flagged for review (test false positives).

---

## 6. Scalability Architecture: Proposed Blueprint (1,000,000 Transactions per Hour)

*Status:* Proposed architectural blueprint and theoretical sizing framework; not a live production deployment.

### Throughput & Latency Specifications:
- **Target Volume:** 1,000,000 transactions / 3,600 seconds $\approx$ **277.8 TPS** steady-state.
- **Peak Burst Sizing (3x factor):** **833.3 TPS**.
- **Target P99 Latency Budget:** **< 15.0 milliseconds** (Design Objective).
- **Measured Local Python Latencies:**
  - Logistic Regression: **~0.44 ms** per transaction (~2,300 TPS / single core unbatched).
  - Random Forest: **~16.3 ms** per transaction unbatched (~61 TPS / single core); drops to **~0.035 ms** per transaction when batched (500 items).
  - *Engineering Note:* Sub-15 ms unbatched gateway SLAs require compiling the model to an optimized C++/ONNX Runtime engine or utilizing micro-batching.

### Proposed Microservice Architecture:
```
Payment Ingestion Gateway (Stripe / Bank Switch)
               │
               ▼
     Kafka Transaction Topic
               │
               ├──► Tier-1 Hard Rules Engine (< 2 ms)
               │    - Velocity checks, card blacklist, geographic geofencing
               │
               ▼
     Distributed In-Memory Feature Store (Redis Cluster)
     - Cardholder rolling 30-day velocity, median amounts, device fingerprints
               │
               ▼
  Inference Microservice Pods (Envoy Proxy Load Balancer)
  ├── Pod 1: Serialized Model (ONNX Runtime / C++ Engine)
  ├── Pod 2: Serialized Model (ONNX Runtime / C++ Engine)
  └── Pod 3: Serialized Model (ONNX Runtime / C++ Engine)
               │
               ├── ML Probability < 0.30 ──► Instant Approval (99.5% volume)
               ├── 0.30 <= ML Probability < 0.70 ──► Step-Up Challenge (3D Secure OTP)
               └── ML Probability >= 0.70 ──► Hard Decline & Automated Fraud Alert
```

---

## 7. Project Directory Structure

```
Task_3-Fraud_Detection/
├── data/
│   └── creditcard.csv                       # Benchmark dataset (284,807 x 31, 143.8 MB)
├── models/
│   ├── fraud_detection_random_forest.joblib # Champion Random Forest pipeline
│   ├── fraud_detection_logistic_regression.joblib # Balanced Logistic Regression pipeline
│   └── fraud_detection_smote_lr.joblib      # SMOTE oversampled pipeline
├── outputs/
│   ├── figures/
│   │   ├── fig1_class_imbalance_and_amounts.png # Class balance & log amount boxplots
│   │   ├── fig2_time_of_day_analysis.png    # 24h volume vs early morning fraud surge
│   │   ├── fig3_correlation_heatmap.png     # Top PCA feature correlations with fraud
│   │   ├── fig4_confusion_matrices.png      # Side-by-side test confusion matrices
│   │   ├── fig5_roc_and_pr_curves.png       # ROC & Precision-Recall curves
│   │   └── fig6_feature_importance.png      # MDI vs Logistic standardized weights
│   └── tables/
│       ├── fraud_dataset_summary.csv        # Dataset overview & dollar volumes
│       ├── class_imbalance_and_amount_stats.csv # Descriptive statistics by class
│       ├── time_of_day_fraud_rates.csv      # Hourly volume and fraud percentage
│       ├── model_performance_comparison.csv # Complete benchmark performance metrics
│       ├── financial_fraud_prevention_audit.csv # Fraud dollars prevented vs friction
│       ├── top_predictive_features.csv      # Top feature importances & weights
│       └── scalability_throughput_analysis.csv # Production throughput specifications
├── build_l2_t3_notebook.py                 # Notebook generation script
├── fraud_detection.py                       # Standalone reproducible pipeline script
├── test_fraud_detection.py                  # Automated 12-test unit test suite
├── Fraud_Detection.ipynb                    # Fully executed Jupyter Notebook (27 cells)
├── requirements.txt                        # Pinned package dependencies
└── README.md                               # Project documentation
```

---

## 8. How to Reproduce

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```powershell
python fraud_detection.py
```

### 3. Run Automated Unit Tests
```powershell
python -m unittest discover -s . -p "test_*.py" -v
```
