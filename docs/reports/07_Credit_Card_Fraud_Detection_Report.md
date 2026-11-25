# DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO
## Formal Task Report: Advanced Analytics & ML — Task 3: Fraud Detection
### Machine Learning for Extreme Class Imbalance, Financial Loss Mitigation & Real-Time Enterprise Scalability

**Candidate:** `Dhruv Goyal` *(Notice: Template document; customize with actual student legal name prior to final submission)*  
**Track:** Data Analytics  
**Task Code:** `PRJ-07` (Task #7 in Master Roadmap)  
**Execution Date:** October 2026  
**Status:** `Completed & Quality Gate Verified` (12/12 Automated Unit Tests Passing, Notebook Executed Headlessly)  
**Target Repository Path:** `projects/07_Credit_Card_Fraud_Detection/`  

---

## Executive Summary

Financial fraud inflicts tens of billions of dollars in losses annually on financial institutions, payment gateways, and retail merchants. When engineering automated machine learning systems for real-time transaction processing, engineers encounter an extreme operational condition: **severe class imbalance**. In legitimate payment flows, fraudulent transactions represent a minute fraction of overall volume, rendering traditional evaluation metrics—especially simple classification accuracy—completely deceptive.

This report documents the design, empirical benchmarking, and production architecture of an **end-to-end fraud detection machine learning system** for **Project 07 (Credit Card Fraud Detection)**. Utilizing the canonical **Credit Card Fraud Detection Benchmark** (ULB Machine Learning Group / Dal Pozzolo et al., 2015) comprising 284,807 European cardholder transactions, the pipeline profiles diurnal transaction dynamics, enforces leakage-free outlier-resistant scaling, evaluates cost-sensitive weighting and synthetic minority oversampling (SMOTE), translates statistical confusion matrices into financial loss recovery tables, and designs an enterprise microservice capable of scoring **1,000,000 transactions per hour** within a strict <15 ms latency budget.

### Key Performance & Audit Highlights:
- **Severe Class Imbalance Diagnosed:** Exactly **492 out of 284,807 transactions are fraudulent (0.1727%)**, establishing an extreme class imbalance ratio of **1 fraud per 578 legitimate transactions**.
- **Empirical Demonstration of the Accuracy Paradox:** A naive baseline model that uniformly predicts 'Legitimate' for every transaction achieves an apparent accuracy of **99.827%**, yet detects **zero fraud cases (0.00% Recall)**, allowing 100% of financial losses to pass through completely unmitigated.
- **Observation of Nocturnal Fraud Prevalence:** By mapping elapsed seconds into diurnal hours ($Hour = (\lfloor Time / 3600 \rfloor) \pmod{24}$), we observed that fraud prevalence rises significantly during early morning hours, reaching **1.71% at 2:00 AM** (57 frauds / 3,328 transactions) and **1.04% at 4:00 AM** (23 frauds / 2,209 transactions) compared to the 0.17% baseline rate. In payments operations, this is commonly hypothesized to reflect automated attacks executed during cardholder sleep cycles; however, the dataset does not contain timezone or device logs to establish attacker intent or causality.
- **Leakage-Safe Preprocessing Pipeline:** Continuous attributes (`Time`, `Amount`) are transformed using `RobustScaler()` to resist extreme financial transaction outliers (up to \$25,691). Scalers are fitted strictly on training folds within an 80/20 stratified partition (227,845 train / 56,962 test; seed 42), preventing test data leakage.
- **Five-Model Empirical Benchmark ($N = 56,962$ Held-Out Test Records, 98 Frauds):**
  - *Dummy Baseline:* 99.83% Accuracy | 0.00% Recall | PR-AUC Undefined (N/A) | TP: 0 | FP: 0 | FN: 98
  - *Unweighted Logistic Regression:* 99.92% Accuracy | 64.29% Recall | 0.7429 PR-AUC | TP: 63 | FP: 13 | FN: 35
  - *Balanced Logistic Regression:* 97.56% Accuracy | **91.84% Recall** | 0.7156 PR-AUC | **TP: 90** | FP: 1,382 | **FN: 8**
  - *SMOTE + Logistic Regression:* 99.70% Accuracy | 88.78% Recall | 0.7523 PR-AUC | TP: 87 | FP: 160 | FN: 11
  - *Balanced Random Forest (Champion):* **99.94% Accuracy** | **81.82% Precision** | **82.65% Recall** | **0.9766 ROC-AUC** | **0.8187 PR-AUC** | **TP: 81** | **FP: 18** | **FN: 17**
- **Financial Fraud Loss Recovery Audit (\$10,644.93 Total Test Exposure):**
  - *Balanced Logistic Regression* detects **\$9,986.32 (93.81%)** in fraud exposure, while placing **\$443,861.88** of legitimate customer volume under operational review across 1,382 false alarms.
  - *SMOTE + Logistic Regression* detects **\$8,818.43 (82.84%)** of fraud dollars while reducing legitimate volume friction to **\$37,668.58** across 160 false alarms.
  - *Balanced Random Forest* detects **\$6,551.61 (61.55%)** of fraud dollars with minimal customer disruption, placing only **\$950.71** under review across 18 false alarms.
  - *Accounting Note:* Detected fraud exposure represents transactions flagged for intervention; realized cash savings depend on downstream chargeback rules and dispute outcomes.
- **Scalability Engineering & Proposed Architecture (1,000,000 Transactions / Hour):** A 1M txn/hr workload requires **277.8 TPS steady-state** and **833.3 TPS peak burst**. Single-item Python inference latency was measured at **~0.44 ms** for Logistic Regression and **~16.3 ms** for Random Forest (dropping to **~0.035 ms/item** when batched). Meeting a sub-15 ms unbatched gateway SLA in production requires compiling the tree ensemble to an optimized C++/ONNX engine; the multi-pod Kubernetes topology is presented as a proposed architectural blueprint rather than a verified live deployment.

---

## 1. Problem Statement & Financial Fraud Operations

In modern electronic payments, fraud detection systems must balance two competing operational imperatives:
1. **Financial Loss Mitigation (Maximizing Recall):** Every undetected fraudulent transaction (False Negative) results in direct chargeback losses, card network penalty fees, and loss of merchant goods.
2. **Customer Experience & Checkout Friction (Maximizing Precision):** Every legitimate customer transaction erroneously declined or delayed (False Positive) damages customer lifetime value, causes transaction abandonment, and inflates call center investigation overhead.

Because fraud accounts for less than 0.2% of daily traffic, naive modeling fails. Machine learning architectures must explicitly integrate cost-sensitive loss functions, precision-recall optimization, and low-latency microservice design.

---

## 2. Dataset Lineage, Provenance & Quality Audit

- **Canonical Repository:** Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB) & Worldline.
- **Primary Literature Reference:**  
  Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification*. In IEEE Symposium Series on Computational Intelligence (SSCI), 2015.
- **Direct Download Source:** Hugging Face Datasets repository (`David-Egea/Creditcard-fraud-detection`) / Kaggle ULB MLG dataset.
- **Applicable License:** Database Contents License (DbCL) v1.0 / Open Database License (ODbL) v1.0. Anonymized benchmark used for educational and research evaluation.
- **File Location:** `data/creditcard.csv` (150,828,752 bytes, 143.84 MB).
- **SHA-256 Checksum:** `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
- **Dimensions:** 284,807 transaction observations $\times$ 31 columns.
- **Privacy Preservation:** Due to confidentiality agreements, 28 numerical attributes (`V1` through `V28`) represent principal components extracted via Principal Component Analysis (PCA). The only raw, un-transformed features are `Time` and `Amount`.
- **Target Variable:** `Class` (integer binary: 1 for fraud, 0 for legitimate).

### Structural Quality Audit

| Attribute | Data Type | Null Count | Null % | Range / Distinct Values | Role |
|:---|:---:|:---:|:---:|:---|:---:|
| `Time` | float64 | 0 | 0.00% | 0.0 to 172,792.0 seconds (48.0 hours) | Elapsed transaction timestamp |
| `V1` to `V28` | float64 | 0 | 0.00% | Orthogonal zero-centered PCA components | Anonymized behavioral dimensions |
| `Amount` | float64 | 0 | 0.00% | \$0.00 to \$25,691.16 | Transaction dollar magnitude |
| `Class` | int64 | 0 | 0.00% | 0 (284,315), 1 (492) | **Binary Target Label** |

Exported to `outputs/tables/fraud_dataset_summary.csv`.

---

## 3. Exploratory Data Analysis & Behavioral Profiling

### 3.1 Class Imbalance & The Accuracy Paradox
The dataset contains **284,315 legitimate transactions (99.8273%)** and **492 fraudulent transactions (0.1727%)**.

In a markdown analysis of metric validity:
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{0 + 284,315}{284,807} = 99.827\%$$

This illustrates the **Accuracy Paradox**: a trivial majority classifier that approves every transaction achieves near-perfect accuracy while providing zero security value. True fraud performance must be evaluated via Precision, Recall, and the Precision-Recall curve.

### 3.2 Transaction Amount Distribution
- **Legitimate:** Mean = \$88.29, Median = \$22.00, Standard Deviation = \$250.11, Maximum = \$25,691.16.
- **Fraudulent:** Mean = \$122.21, Median = \$9.25, Standard Deviation = \$256.68, Maximum = \$2,125.87.
- *Behavioral Finding:* While the mean fraud amount is 38% higher than legitimate transactions, the median fraud amount (\$9.25) is substantially lower than legitimate purchases (\$22.00). Fraud rings routinely conduct automated \$1.00 authorization pings to verify stolen credentials before attempting maximum-limit cashouts.

### 3.3 Diurnal Dynamics: Observed Nocturnal Fraud Prevalence
Converting continuous timestamps into diurnal 24-hour buckets reveals an empirical pattern (`outputs/tables/time_of_day_fraud_rates.csv`):
- Overall mean fraud prevalence across all hours is **0.1727%** (492 frauds / 284,807 transactions).
- At **2:00 AM (Hour 2)**, observed fraud prevalence rises to **1.7127%** (57 frauds across 3,328 transactions) — a **9.9x increase over the baseline average rate**.
- At **4:00 AM (Hour 4)**, observed fraud prevalence remains elevated at **1.0412%** (23 frauds across 2,209 transactions).
- During peak daytime consumer hours (12:00 PM – 8:00 PM), hourly volume exceeds 15,000 transactions, while observed fraud rates remain below 0.20%.
- *Descriptive vs. Causal Boundary:* Payments operations literature often hypothesizes that nocturnal spikes occur because automated transactions take advantage of cardholders being asleep, which reduces immediate verification responses. However, because this anonymized dataset does not contain cardholder timezones, merchant locations, IP addresses, or device logs, the data cannot establish attacker intent or specific operational mechanisms; the diurnal pattern is documented strictly as an observed descriptive association in this dataset.

---

## 4. Leakage-Free Preprocessing Architecture

To ensure adherence to production standards:
1. **Stratified Splitting:** Partitioned the corpus into an 80% training set (227,845 records) and a 20% test set (56,962 records) using stratified sampling on `Class` (`random_state=42`).
2. **Outlier-Resistant Scaling:** Continuous features `Time` and `Amount` possess extreme positive skewness. Applying standard z-score normalization would distort mean and variance estimates due to \$25k transactions. We employ `RobustScaler()`:
   $$x_{\text{scaled}} = \frac{x - \text{median}(x)}{IQR(x)}$$
   The median and IQR parameters are fitted strictly on `X_train` and applied transformationally to `X_test`.
3. **Pipeline Encapsulation:** Scaling and oversampling steps are combined with classifiers into unified Scikit-Learn and Imbalanced-Learn `Pipeline` constructs, eliminating out-of-fold data leakage.

```
Raw Input Data (284,807 x 31)
          │
          ▼
   80/20 Stratified Split (Seed 42)
   ├── Train: 227,845 records (394 Fraud, 227,451 Legit)
   └── Test:   56,962 records (98 Fraud, 56,864 Legit)
          │
          ▼
   Pipeline Preprocessing
   ├── RobustScaler(Time, Amount) [Fit on Train Only]
   └── PCA Features V1-V28 (Passed Through)
          │
          ▼
   Model Architectures
   ├──► DummyClassifier (Majority Baseline)
   ├──► LogisticRegression (Unweighted)
   ├──► LogisticRegression (Balanced Class Weights)
   ├──► SMOTE (sampling_strategy=0.1) + LogisticRegression
   └──► RandomForestClassifier (100 Trees, Depth 10, Balanced)
```

---

## 5. Comprehensive Model Benchmarks & Diagnostic Evaluation

All models were evaluated on the identical 56,962 held-out test transactions containing exactly 98 ground-truth fraud cases:

### Master Benchmark Performance Table

| Model Architecture | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | PR-AUC (Avg Prec) | True Positives | False Positives | False Negatives | Fit Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dummy (Majority Baseline)** | 99.83% | 50.00% | 0.00% | 0.00% | 0.0000 | *N/A* | *N/A* | 0 | 0 | 98 | 0.01s |
| **Logistic Regression (Unweighted)** | 99.92% | 82.13% | 82.89% | 64.29% | 0.7241 | 0.9592 | 0.7429 | 63 | 13 | 35 | 0.62s |
| **Logistic Regression (Balanced)** | 97.56% | 94.70% | 6.11% | **91.84%** | 0.1146 | 0.9721 | 0.7156 | **90** | 1,382 | **8** | 1.13s |
| **SMOTE + Logistic Regression** | 99.70% | 94.25% | 35.22% | 88.78% | 0.5043 | 0.9667 | 0.7523 | 87 | 160 | 11 | 2.14s |
| **Random Forest (Balanced Weights)**| **99.94%** | **91.31%** | **81.82%** | **82.65%** | **0.8223** | **0.9766** | **0.8187** | **81** | **18** | **17** | 10.62s |

Exported to `outputs/tables/model_performance_comparison.csv`.

---

## 6. Financial Fraud Loss Prevention Audit

In corporate risk underwriting, statistical confusion matrices must be translated into financial accounting balance sheets.

### Operational Definitions & Accounting Boundaries
- **Total Test Fraud Exposure (\$10,644.93):** The sum of transaction amounts across all 98 confirmed fraud cases in the held-out test partition ($N = 56,962$).
- **Dollars Prevented / Intercepted:** Sum of transaction amounts for True Positives ($\sum \text{Amount}_i$ where $\text{Actual}=1 \text{ and } \text{Predicted}=1$). *Crucial Operational Caveat:* Flagging a fraudulent transaction identifies it for operational challenge or decline; it represents *detected fraud exposure*, not guaranteed realized cash savings. True realized savings depend on payment-network chargeback liability shifts, merchant dispute resolution, and cardholder recovery protocols.
- **Dollars Lost:** Sum of transaction amounts for False Negatives ($\sum \text{Amount}_i$ where $\text{Actual}=1 \text{ and } \text{Predicted}=0$). Represents undetected fraud dollars that bypass the model entirely.
- **Legitimate Volume Frictioned / Dollars Held:** Sum of transaction amounts for False Positives ($\sum \text{Amount}_i$ where $\text{Actual}=0 \text{ and } \text{Predicted}=1$). Represents legitimate customer transaction dollars temporarily delayed, challenged with step-up authentication, or declined, introducing friction and potential checkout abandonment.

### Financial Loss & Friction Summary ($N = 56,962$ Test Transactions, \$10,644.93 Fraud Volume)

| Model Architecture | Total Test Fraud ($) | Dollars Prevented ($) | Dollars Lost ($) | Detection Rate (%) | False Positive Count | Legitimate Volume Frictioned ($) | Operational Trade-Off Profile |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Dummy Baseline** | \$10,644.93 | \$0.00 | \$10,644.93 | 0.00% | 0 | \$0.00 | Total loss; zero fraud protection |
| **Logistic Regression (Unweighted)** | \$10,644.93 | \$4,506.29 | \$6,138.64 | 42.33% | 13 | \$902.67 | High fraud leakage (misses \$6.1k in fraud) |
| **Logistic Regression (Balanced)** | \$10,644.93 | **\$9,986.32** | **\$658.61** | **93.81%** | 1,382 | \$443,861.88 | Maximum loss interception; high queue friction |
| **SMOTE + Logistic Regression** | \$10,644.93 | \$8,818.43 | \$1,826.50 | 82.84% | 160 | \$37,668.58 | Strong compromise for automated challenge rules |
| **Random Forest (Balanced)** | \$10,644.93 | \$6,551.61 | \$4,093.32 | 61.55% | **18** | **\$950.71** | **Optimal Commercial Champion** (only 18 false alarms) |

Exported to `outputs/tables/financial_fraud_prevention_audit.csv`.

---

## 7. Model Interpretability & Feature Importance Analysis

Feature importances were extracted via **Mean Decrease in Impurity (Gini)** for Random Forest and **standardized log-odds coefficients** for Logistic Regression (`outputs/tables/top_predictive_features.csv`):

### Top Discriminatory Predictors:
1. **`V14` (Random Forest MDI = 18.6%, LR $\beta = -1.40$):** Strongest individual predictor. Extreme negative values correlate heavily with compromised card credentials.
2. **`V10` (Random Forest MDI = 11.4%, LR $\beta = -1.19$):** Second strongest negative discriminator.
3. **`V12` (Random Forest MDI = 10.4%, LR $\beta = -1.12$):** Consistently weighted across both linear and non-linear paradigms.
4. **`V17` (Random Forest MDI = 9.7%, LR $\beta = -0.99$):** Highly influential PCA axis.
5. **`V4` (Random Forest MDI = 9.3%, LR $\beta = +1.50$):** Primary positive indicator of unauthorized transaction patterns.

Together, these top five features account for **59.4%** of Random Forest split decisions.

---

## 8. High-Throughput Production Scalability: Proposed Architectural Blueprint (1,000,000 Txns / Hour)

*Deployment Status Notice:* The following section outlines a theoretical capacity sizing framework and architectural proposal for enterprise deployment; it represents an engineering design specification, not an already deployed or verified live cluster.

### 8.1 Throughput Requirements & Design Targets
- **Target Hourly Volume:** 1,000,000 transactions / hour.
- **Required Ingestion Throughput:**
  $$\text{Throughput}_{\text{steady}} = \frac{1,000,000}{3,600} \approx \mathbf{277.8 \text{ Transactions / Second (TPS)}}$$
- **Peak Burst Factor (3x Design Target):** $\approx \mathbf{833.3 \text{ TPS}}$.
- **Target End-to-End P99 Latency SLA:** **< 15.0 milliseconds** (Design Objective).

### 8.2 In-Memory Benchmark Measurements vs. Production Needs
- **Measured Local Python Single-Item Latency (Unbatched):**
  - Logistic Regression Pipeline: **~0.44 ms** per transaction (~2,300 TPS on single core).
  - Random Forest Pipeline (100 Trees): **~16.3 ms** per transaction (~61 TPS on single core in sequential Python/scikit-learn).
- **Measured Local Batched Latency (Vectorized, 500 items):**
  - Random Forest Pipeline: **~0.035 ms** per transaction (~28,500 items/sec per core).
- *Engineering Discovery:* In pure sequential Python with single-item DataFrame conversion overhead, Random Forest requires ~16.3 ms, slightly exceeding a 15 ms end-to-end SLA. Therefore, to achieve the proposed sub-15 ms target in a live production environment, the model must either:
  1. Be compiled to an optimized C++/ONNX Runtime engine (e.g., via `onnxruntime` or `treelite`), which reduces single-item tree traversal latency by 5–10x; OR
  2. Be deployed behind a micro-batching inference gateway (accumulating incoming requests over a 2–3 ms window).
- *Proposed Pod Sizing (Theoretical Model):* Sizing for an 833.3 TPS peak burst with compiled ONNX runtime execution (~1–2 ms per item) at 60% CPU utilization would theoretically require **3 to 4 worker pods** (4 vCPUs each) running containerized inference engines behind an Envoy load balancer and a Redis distributed feature cache.

Exported to `outputs/tables/scalability_throughput_analysis.csv`.

---

## 9. Strategic Business Recommendations for Fraud Operations

### Recommendation 1: Implement Dynamic Three-Tier Risk Interventions
Rather than relying on a hard binary block/approve threshold, configure risk triage tiers based on predicted probability $p$:
- **Tier 1 — Frictionless Approval ($p < 0.30$):** 99.5% of legitimate volume passes instantly without challenge.
- **Tier 2 — Step-Up Authentication ($0.30 \le p < 0.75$):** Trigger biometric verification or 3D Secure SMS OTP. This salvages false positives while intercepting automated botnet attacks.
- **Tier 3 — Automated Hard Decline ($p \ge 0.75$):** Block transaction immediately and freeze card credentials.

### Recommendation 2: Deploy Nocturnal Velocity Throttles
Because fraud rates surge by 10x between 1:00 AM and 4:00 AM, apply dynamic nocturnal rules: lower the step-up threshold from $p=0.30$ to $p=0.15$ during nocturnal hours for cross-border and non-chip transactions.

### Recommendation 3: Implement Daily Chargeback Feedback Ingestion
Establish an automated retraining pipeline that ingests verified card-scheme chargeback files daily, adapting feature weights as fraud rings rotate their synthetic attack profiles.

---

## 10. Quality Assurance & Automated Test Suite Verification

The implementation was verified using the automated test suite in `test_fraud_detection.py`:

```
test_01_dataset_file_exists_and_sha256: PASSED
test_02_dataset_schema: PASSED
test_03_zero_missing_values: PASSED
test_04_class_imbalance_ratio: PASSED
test_05_stratified_train_test_split: PASSED
test_06_model_artifacts_saved: PASSED
test_07_model_prediction_shape_and_classes: PASSED
test_08_performance_table_metrics: PASSED
test_09_financial_prevention_audit_metrics: PASSED
test_10_figures_generated: PASSED
test_11_tables_generated: PASSED
test_12_time_of_day_peak_fraud_hour: PASSED

Ran 12 tests in 11.796s — ALL 12 TESTS PASSED (0 failures, 0 errors)
```

---

## 11. Artifacts & Deliverables Summary

| Deliverable Path | Description | Verification State |
|:---|:---|:---:|
| `data/creditcard.csv` | Benchmark dataset (284,807 rows, 31 cols, SHA-256 verified) | Verified |
| `outputs/figures/fig1_class_imbalance_and_amounts.png` | Class balance bar chart & log amount boxplots | Verified |
| `outputs/figures/fig2_time_of_day_analysis.png` | Diurnal transaction volume vs nocturnal fraud surge | Verified |
| `outputs/figures/fig3_correlation_heatmap.png` | Top positive and negative PCA correlations with fraud | Verified |
| `outputs/figures/fig4_confusion_matrices.png` | Side-by-side confusion matrices across 4 models | Verified |
| `outputs/figures/fig5_roc_and_pr_curves.png` | ROC and Precision-Recall discrimination curves | Verified |
| `outputs/figures/fig6_feature_importance.png` | Random Forest MDI vs Logistic standardized weights | Verified |
| `outputs/tables/fraud_dataset_summary.csv` | Dataset overview metadata & dollar volumes | Verified |
| `outputs/tables/class_imbalance_and_amount_stats.csv` | Descriptive statistics by transaction class | Verified |
| `outputs/tables/time_of_day_fraud_rates.csv` | Hourly transaction volume and fraud percentage | Verified |
| `outputs/tables/model_performance_comparison.csv` | Multi-model benchmark performance table | Verified |
| `outputs/tables/financial_fraud_prevention_audit.csv` | Dollars recovered vs customer checkout friction | Verified |
| `outputs/tables/top_predictive_features.csv` | Feature importance weights and rank order | Verified |
| `outputs/tables/scalability_throughput_analysis.csv` | 1M txns/hr production throughput specifications | Verified |
| `models/fraud_detection_random_forest.joblib` | Serialized champion Random Forest pipeline | Verified |
| `models/fraud_detection_logistic_regression.joblib` | Serialized balanced Logistic Regression pipeline | Verified |
| `models/fraud_detection_smote_lr.joblib` | Serialized SMOTE oversampled pipeline | Verified |
| `fraud_detection.py` | Standalone reproducible pipeline script | Verified |
| `test_fraud_detection.py` | Automated 12-test unit test suite | Verified |
| `Fraud_Detection.ipynb` | Fully executed Jupyter Notebook (27 cells, 540 KB) | Verified |
| `README.md` | Comprehensive standalone project documentation | Verified |
| `requirements.txt` | Pinned package dependencies | Verified |
| `docs/Reports/Week_03/Task_7_Fraud_Detection_Report.md` | Formal technical report (this document) | Verified |
| `docs/Architecture/Week_03/Task_7_Fraud_Detection_Architecture.md` | Complete engineering architecture specification | Verified |
| `docs/Video_Demo/Week_03/Task_7_Demo_Script.md` | Time-coded video demonstration script | Verified |
| `docs/Video_Demo/Week_03/title_card_task7.png` | 1080p high-resolution static video title card | Verified |
