# DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO
## Formal Task Report: Advanced Analytics & ML — Task 2: Wine Quality Prediction
### Supervised Classification, Class Imbalance Diagnostics & Duplicate Leakage Audit on the UCI Wine Quality Benchmark

**Candidate:** `Dhruv Goyal` *(Notice: Template document; customize with actual student legal name prior to final submission)*  
**Track:** Data Analytics  
**Task Code:** `PRJ-06` (Task #6 in Master Roadmap)  
**Execution Date:** October 2026  
**Status:** `Completed & Quality Gate Verified` (12/12 Automated Unit Tests Passing, Notebook Executed Headlessly)  
**Target Repository Path:** `projects/06_Wine_Quality_Prediction/`  

---

## Executive Summary

In sensory evaluation research, understanding how objective chemical properties correspond to human sensory judgments is a classic analytical problem. In wine evaluation, sensory panels independently score wines on integer quality scales.

This report documents the design, implementation, and empirical verification of an **end-to-end supervised machine learning pipeline** that evaluates wine sensory quality predictions from **objective physicochemical laboratory measurements**. Using the canonical **UCI Machine Learning Repository Wine Quality benchmark** (Cortez et al., 2009) comprising 1,599 red Vinho Verde wines, the project explores feature associations with quality scores, diagnoses severe class imbalance, conducts an explicit duplicate record and train-test boundary leakage audit, enforces leak-free `StandardScaler` pipeline encapsulation, and evaluates three distinct classifiers (Random Forest, SGD Classifier, and Support Vector Classifier) against a majority-class baseline.

### Key Performance & Audit Highlights:
- **Severe Class Imbalance Diagnosed:** Exactly **82.49% of all wines (1,319 / 1,599)** are concentrated in median quality ratings 5 (42.59%) and 6 (39.90%). Extreme tail scores (grades 3, 4, and 8) represent less than 5% combined.
- **Empirical Demonstration of Multi-Class Failure:** Demonstrated that under an unweighted 3-tier grouping (Low: 3–4, Medium: 5–6, High: 7–8), a standard classifier achieves **0.00 recall and 0.00 precision on the Low tier**, blindly optimizing majority accuracy while failing on rare grades.
- **Defensible Binary Quality Formulation:** Established an analytical target threshold: **Higher-Rated Wine** ($\ge 7$, 217 samples, 13.57%) versus **Typical / Lower-Rated Wine** ($< 7$, 1,382 samples, 86.43%). *Scope note:* In this empirical study, $\ge 7$ represents an analytical grouping for higher-rated samples within this specific dataset, not a certified commercial award category or market pricing tier.
- **Duplicate Records & Boundary Leakage Audit:** The dataset contains 240 exact duplicate rows across all 12 attributes (1,359 unique rows). In the standard 80/20 stratified split, **77 out of 320 test samples (24.06%) are identical to training rows**, creating memorization opportunities. We report both the standard benchmark and a duplicate-aware evaluation on the 1,359 unique rows.
- **Majority-Class Baseline Formulation:** The `DummyClassifier(strategy='most_frequent')` predicts class 0 for all instances, achieving an accuracy of **86.56%** (277/320) and balanced accuracy of **50.00%**. Because a constant classifier generates identical output scores for all samples, its ROC-AUC and PR-AUC are mathematically undefined (represented as N/A).
- **Benchmark Evaluation Results:**
  - *Standard Split (1,599 rows, 320 test samples with 24.1% duplicate overlap):* Random Forest achieves **94.06% accuracy**, **0.9523 ROC-AUC**, **92.86% precision**, and **60.47% recall** (only 2 false positives out of 277 non-premium test samples).
  - *Deduplicated Split (1,359 unique rows, 272 test samples, 0% overlap):* Random Forest achieves **89.34% accuracy**, **0.8749 ROC-AUC**, **72.22% precision**, and **35.14% recall**. Support Vector Classifier (SVC) achieves **80.51% accuracy** and **0.8839 ROC-AUC** with **78.38% recall**.
- **Biochemical Associations (Not Causal Proof):** Dual feature importance diagnostics (Mean Decrease in Impurity and held-out Permutation Importance) identify **alcohol content** ($r = +0.48$), **sulphates** ($r = +0.25$), and **volatile acidity** ($r = -0.39$) as having the strongest statistical associations with higher sensory ratings in this dataset.
- **Forensic Misclassification Audit:** Discovered the *Low-Alcohol Masking Effect*: 76% of missed higher-rated wines possessed lower alcohol concentrations (9.2%–10.5%), where human judges rewarded structural balance that basic bulk ethanol metrics underweighted.

---

## 1. Problem Statement & Viticultural Context

In oenology and sensory science, exploring how laboratory chemistry correlates with taster evaluations provides objective guidance for winemaking adjustments:
1. **Analytical Pre-Screening:** Exploring whether laboratory chemistry can prioritize lots for formal human tasting panels.
2. **Monitoring Key Chemical Indicators:** Tracking volatile acidity and sulphur compounds during vinification to understand their correlation with sensory scores.
3. **Evaluating Machine Learning Generalization:** Determining whether standard classification models truly learn generalizable chemical patterns or partially rely on memorized duplicate observations.

---

## 2. Dataset Lineage, Provenance & Duplicate Audit

- **Canonical Repository:** UCI Machine Learning Repository (Dataset ID: 186)
- **Direct UCI Download URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`
- **Primary Literature Reference:**  
  P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, Elsevier, 47(4):547-553, 2009.
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **SHA-256 Checksum:** `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e`
- **File Size:** 84,199 bytes.
- **Dimensions:** 1,599 observations $\times$ 12 columns (11 continuous physicochemical measurements and 1 integer sensory target).
- **Missing Values:** Exactly 0 missing values across all cells.

### Forensic Duplicate & Boundary Leakage Audit
An audit was conducted using `outputs/tables/duplicate_leakage_audit.csv`:
- **Exact Duplicate Rows:** 240 records share identical values across all 11 features and the sensory quality score.
- **Unique Records:** 1,359 unique observations.
- **Conflicting Labels:** 0 instances where identical feature vectors have different quality scores.
- **Boundary Leakage:** In a standard 80/20 stratified split (seed 42), **77 of the 320 test samples (24.06%) are exact copies of a sample in the training set**.
- **Methodological Handling:** Rather than silently dropping duplicates or reporting only the inflated benchmark, we report both:
  1. The **Standard Split Benchmark** (1,599 rows) to maintain comparability with published literature.
  2. The **Deduplicated Split Benchmark** (1,359 rows) to provide an honest, uninflated evaluation of out-of-sample generalization.

---

## 3. Physicochemical Feature Profiling & Descriptive Statistics

Each observation contains 11 laboratory-measured chemical parameters:

| Feature Name | Mean | Std Dev | Min | Median | Max | Skewness | Kurtosis | Chemical Role in Wine |
|--------------|------|---------|-----|--------|-----|----------|----------|-----------------------|
| `fixed acidity` | 8.320 | 1.741 | 4.60 | 7.90 | 15.90 | 0.983 | 1.132 | Tartaric acid foundation ($g/dm^3$) |
| `volatile acidity` | 0.528 | 0.179 | 0.12 | 0.52 | 1.58 | 0.672 | 1.226 | Acetic acid ($g/dm^3$) |
| `citric acid` | 0.271 | 0.195 | 0.00 | 0.26 | 1.00 | 0.318 | -0.789 | Fresh organic acid ($g/dm^3$) |
| `residual sugar` | 2.539 | 1.410 | 0.90 | 2.20 | 15.50 | 4.541 | 28.618 | Unfermented sugars ($g/dm^3$) |
| `chlorides` | 0.087 | 0.047 | 0.012 | 0.079 | 0.611 | 5.680 | 41.716 | Mineral salts ($g/dm^3$) |
| `free sulfur dioxide` | 15.875 | 10.460 | 1.00 | 14.00 | 72.00 | 1.251 | 2.024 | Active molecular and bisulphite preservative ($mg/dm^3$) |
| `total sulfur dioxide` | 46.468 | 32.895 | 6.00 | 38.00 | 289.00 | 1.516 | 3.810 | Combined free and bound $\text{SO}_2$ ($mg/dm^3$) |
| `density` | 0.9967 | 0.0019 | 0.9901 | 0.9968 | 1.0037 | 0.071 | 0.934 | Specific gravity ($g/cm^3$) |
| `pH` | 3.311 | 0.154 | 2.74 | 3.31 | 4.01 | 0.194 | 0.807 | Negative log hydrogen ion concentration |
| `sulphates` | 0.658 | 0.170 | 0.33 | 0.62 | 2.00 | 2.429 | 11.720 | Potassium sulphate additive ($g/dm^3$) |
| `alcohol` | 10.423 | 1.066 | 8.40 | 10.20 | 14.90 | 0.861 | 0.200 | Ethanol percentage by volume (% vol) |
| `quality` | 5.636 | 0.808 | 3.00 | 6.00 | 8.00 | 0.218 | 0.297 | Median human taster score (scale 0–10) |

*(Generated and exported to `outputs/tables/physicochemical_descriptive_stats.csv`).*

---

## 4. Class Imbalance Diagnostic & Target Formulation

### 4.1 Raw Sensory Quality Distribution
Sensory ratings in the red wine dataset exhibit an extreme central concentration:
- **Score 3:** 10 samples (0.63%)
- **Score 4:** 53 samples (3.31%)
- **Score 5:** 681 samples (42.59%)
- **Score 6:** 638 samples (39.90%)
- **Score 7:** 199 samples (12.45%)
- **Score 8:** 18 samples (1.13%)
- **Total:** 1,599 samples

Exactly **82.49%** of observations inhabit ratings 5 and 6.

### 4.2 Why 3-Tier Multi-Class Grouping Fails
Grouping wine quality into three tiers (Low: 3–4, Medium: 5–6, High: 7–8) appears intuitive, but fails in practice:
- On a 20% test split ($N = 320$), the Low tier contains only 13 samples (4.06%).
- An unweighted Random Forest achieves:
  - **Low Tier (3–4):** Precision = 0.00, Recall = 0.00, F1 = 0.00 (all 13 samples misclassified as Medium).
  - **Medium Tier (5–6):** Recall = 0.96.
  - **High Tier (7–8):** Recall = 0.51, F1 = 0.58.
- Overall accuracy is deceptively high (**85.94%**), yet the system completely fails to detect rare low-quality instances.

### 4.3 Analytical Justification for Binary Cutoff ($\ge 7$)
We engineer the binary target `good_wine`:
$$\text{good\_wine} = \begin{cases} 1 & \text{if } \text{quality} \ge 7 \quad (\text{Higher-Rated}, N = 217, 13.57\%) \\ 0 & \text{if } \text{quality} < 7 \quad (\text{Typical / Lower-Rated}, N = 1,382, 86.43\%) \end{cases}$$

This cutoff separates the top 13.57% higher-rated samples from the 86.43% majority, creating a well-defined binary classification problem for evaluation.

---

## 5. Pairwise Correlations & Feature Associations

Pearson correlation coefficients with the target `quality` score reveal notable statistical relationships:

| Feature Name | Pearson Correlation ($r$) | Direction | Statistical Interpretation |
|--------------|:-------------------------:|:---------:|----------------------------|
| `alcohol` | **+0.4762** | Positive | Strongest positive linear association with quality scores in this sample. |
| `sulphates` | **+0.2514** | Positive | Positive correlation with sensory scores. |
| `citric acid` | **+0.2264** | Positive | Positively associated with perceived quality. |
| `fixed acidity` | +0.1241 | Positive | Weak positive association. |
| `residual sugar` | +0.0137 | Neutral | Virtually unassociated with quality ratings in this dry red wine sample. |
| `free sulfur dioxide` | -0.0507 | Slight Neg | Weak negative association. |
| `pH` | -0.0577 | Slight Neg | Weak negative association. |
| `chlorides` | -0.1289 | Negative | Moderate negative correlation. |
| `density` | -0.1749 | Negative | Inversely related to alcohol ($r = -0.496$). |
| `total sulfur dioxide` | -0.1851 | Negative | Negative association with sensory scores. |
| `volatile acidity` | **-0.3906** | Negative | Strongest negative linear association with quality scores. |

*Interpretability Caveat:* These values reflect empirical correlations within this specific dataset and do not prove causal biological mechanisms.

---

## 6. Leakage-Free Preprocessing & Model Architecture

### 6.1 Data Splitting Strategy
- **Partition:** 80% Training ($N = 1,279$) and 20% Held-Out Testing ($N = 320$).
- **Stratification:** Stratified sampling on `good_wine` ensures the 13.57% minority class prevalence is preserved precisely across splits:
  - Training Split: 1,105 Typical (86.40%) vs 174 Higher-Rated (13.60%).
  - Testing Split: 277 Typical (86.56%) vs 43 Higher-Rated (13.44%).
- **Random State:** Seed fixed to `42` for deterministic reproducibility.

### 6.2 Leakage-Safe Feature Standardization
To prevent data leakage, `StandardScaler` is wrapped inside Scikit-Learn `Pipeline` objects. The scaler is fit strictly on training folds and never sees held-out test data during parameter estimation.

---

## 7. Multi-Model Performance Benchmarks

### 7.1 Standard Benchmark Split ($N = 320$ Test Samples, 24.06% Duplicate Overlap)

| Model Architecture | Accuracy | Balanced Acc | Precision (Class 1) | Recall (Class 1) | F1-Score | Macro F1 | ROC-AUC | PR-AUC | 5-Fold CV ROC-AUC |
|--------------------|:--------:|:------------:|:-------------------:|:----------------:|:--------:|:--------:|:-------:|:------:|:-----------------:|
| **Dummy (most_frequent)** | 86.56% | 50.00% | 0.00% | 0.00% | 0.0000 | 0.4640 | *N/A (Constant)* | *N/A (Constant)* | *N/A* |
| **Random Forest** | **94.06%** | **79.87%** | **92.86%** | **60.47%** | **0.7324** | **0.8495** | **0.9523** | **0.8535** | **$0.894 \pm 0.020$** |
| **SGD Classifier (Balanced)** | 83.75% | 81.77% | 44.16% | 79.07% | 0.5667 | 0.7333 | 0.8929 | 0.5536 | $0.806 \pm 0.020$ |
| **Support Vector (SVC, Balanced)** | 84.69% | 81.33% | 45.83% | 76.74% | 0.5739 | 0.7403 | 0.9015 | 0.6165 | $0.875 \pm 0.017$ |

*Baseline Metric Behavior Note:* `DummyClassifier(strategy='most_frequent')` predicts class 0 for all instances. On 320 test samples (277 negative, 43 positive), it achieves $277/320 = 86.56\%$ accuracy and $50.00\%$ balanced accuracy. Because all output scores are identical, positive-class precision, recall, and F1 are 0.0, and ROC-AUC is mathematically undefined (represented as N/A).

### 7.2 Deduplicated Dataset Split ($N = 272$ Test Samples, 0% Duplicate Overlap)

| Model Architecture | Accuracy | Balanced Acc | Precision (Class 1) | Recall (Class 1) | F1-Score | Macro F1 | ROC-AUC | PR-AUC | 5-Fold CV ROC-AUC |
|--------------------|:--------:|:------------:|:-------------------:|:----------------:|:--------:|:--------:|:-------:|:------:|:-----------------:|
| **Dummy (most_frequent)** | 86.40% | 50.00% | 0.00% | 0.00% | 0.0000 | 0.4635 | *N/A* | *N/A* | *N/A* |
| **Random Forest** | 89.34% | 66.50% | 72.22% | 35.14% | 0.4727 | 0.7067 | 0.8749 | 0.6018 | $0.869 \pm 0.031$ |
| **SGD Classifier (Balanced)** | 71.69% | 77.92% | 30.77% | 86.49% | 0.4539 | 0.6314 | 0.8252 | 0.4455 | $0.835 \pm 0.038$ |
| **Support Vector (SVC, Balanced)** | 80.51% | 79.61% | 39.19% | 78.38% | 0.5225 | 0.7001 | 0.8839 | 0.5818 | $0.867 \pm 0.030$ |

### 7.3 Methodological Comparison: Standard vs. Deduplicated Generalization
- On the standard split, Random Forest achieves 94.06% accuracy and 0.9523 ROC-AUC, boosted by memorizing the 24.1% cross-boundary duplicates.
- On the deduplicated split, with zero cross-boundary leakage, Random Forest achieves 89.34% accuracy and 0.8749 ROC-AUC.
- Support Vector Classifier (SVC) demonstrates more robust generalization across deduplication, achieving 80.51% accuracy and 0.8839 ROC-AUC.

---

## 8. Feature Importance Analysis

Feature importances were evaluated using both **Mean Decrease in Impurity (Gini)** and **Permutation Importance** on the held-out test split (20 shuffle iterations, scoring `roc_auc`):

| Rank | Physicochemical Feature | Gini Importance (MDI) | Permutation Importance (Mean $\Delta \text{AUC}$) | Permutation Std Dev |
|:---:|------------------------|:---------------------:|:-------------------------------------------------:|:-------------------:|
| 1 | **`alcohol`** | **0.1705 (17.1%)** | **0.0748** | $\pm 0.0144$ |
| 2 | **`sulphates`** | **0.1130 (11.3%)** | **0.0541** | $\pm 0.0093$ |
| 3 | **`volatile acidity`** | **0.1028 (10.3%)** | **0.0358** | $\pm 0.0043$ |
| 4 | `fixed acidity` | 0.0754 (7.5%) | 0.0151 | $\pm 0.0036$ |
| 5 | `total sulfur dioxide` | 0.0806 (8.1%) | 0.0130 | $\pm 0.0068$ |
| 6 | `density` | 0.0992 (9.9%) | 0.0082 | $\pm 0.0075$ |
| 7 | `free sulfur dioxide` | 0.0613 (6.1%) | 0.0079 | $\pm 0.0018$ |
| 8 | `chlorides` | 0.0734 (7.3%) | 0.0060 | $\pm 0.0046$ |
| 9 | `residual sugar` | 0.0690 (6.9%) | 0.0059 | $\pm 0.0035$ |
| 10 | `pH` | 0.0625 (6.3%) | 0.0049 | $\pm 0.0028$ |
| 11 | `citric acid` | 0.0921 (9.2%) | 0.0031 | $\pm 0.0041$ |

*(Generated and exported to `outputs/tables/random_forest_feature_importance.csv`).*

---

## 9. Forensic Misclassification Error Audit

An individual audit of all 19 test errors (`outputs/tables/misclassification_analysis.csv`) on the standard split revealed:

### 9.1 The False Positives ($N = 2$)
Only two lower-rated wines were misclassified as higher-rated:
- **Sample 347:** Alcohol = 12.0%, Citric Acid = 0.67, Sulphates = 0.93. Actual Quality = 6. Predicted Probability = 0.515.
- **Sample 544:** Alcohol = 8.4%, Citric Acid = 0.74, Sulphates = 0.79. Actual Quality = 6. Predicted Probability = 0.510.
- **Root Cause:** Both wines had actual quality scores of **6** (immediately adjacent to the 7 cutoff) and possessed chemical attributes near the decision boundary.

### 9.2 The False Negatives ($N = 17$)
Seventeen higher-rated wines were classified as lower-rated:
- **The Low-Alcohol Masking Effect:** 13 of the 17 missed wines (76.5%) had alcohol levels between **9.2% and 10.5%** (e.g., Sample 243 with 9.2% alcohol, predicted probability 0.11; Sample 339 with 9.8% alcohol, predicted probability 0.17). Human judges rewarded these lighter-bodied wines for balance, but the model relied heavily on alcohol as a primary predictive proxy and demoted them.

---

## 10. Quality Assurance & Automated Unit Test Verification

An automated test suite was constructed in `test_wine_quality.py`. All 12 unit tests execute in **0.17 seconds** with a 100% pass rate:

```
2026-10-21 01:15:39 [INFO] Loaded dataset from .../data/winequality-red.csv with shape (1599, 12)
2026-10-21 01:15:39 [INFO] Dataset validation: Total Nulls = 0, Duplicates = 240
2026-10-21 01:15:39 [INFO] Binary target 'good_wine' created: 0 = 1382 (86.43%), 1 = 217 (13.57%)
............
----------------------------------------------------------------------
Ran 12 tests in 0.168s

OK
```

---

## 11. Artifacts & Deliverables Summary

| Deliverable Path | Description | Verification State |
|------------------|-------------|:------------------:|
| `data/winequality-red.csv` | Canonical UCI Red Wine dataset (1,599 rows, 12 cols, SHA-256 verified) | Verified |
| `data/winequality-white.csv` | Canonical UCI White Wine dataset (4,898 rows, 12 cols) | Verified |
| `outputs/figures/fig1_wine_distributions.png` | 11-panel physicochemical feature distribution plot | Verified |
| `outputs/figures/fig2_quality_distribution.png` | Raw score distribution vs engineered binary target | Verified |
| `outputs/figures/fig3_correlation_heatmap.png` | Full Pearson correlation matrix and quality correlates | Verified |
| `outputs/figures/fig4_confusion_matrices.png` | Side-by-side test confusion matrices | Verified |
| `outputs/figures/fig5_roc_pr_curves.png` | ROC and Precision-Recall curves with honest reference lines | Verified |
| `outputs/figures/fig6_feature_importance.png` | Gini Impurity vs Held-Out Permutation Importance | Verified |
| `outputs/tables/wine_dataset_summary.csv` | High-level dataset metadata & summary metrics | Verified |
| `outputs/tables/physicochemical_descriptive_stats.csv` | Comprehensive descriptive statistics table | Verified |
| `outputs/tables/duplicate_leakage_audit.csv` | Audit of 240 duplicates and 77 cross-boundary rows | Verified |
| `outputs/tables/model_performance_comparison.csv` | Multi-model benchmark performance table (standard split) | Verified |
| `outputs/tables/deduplicated_model_comparison.csv` | Duplicate-aware benchmark table (1,359 unique rows) | Verified |
| `outputs/tables/random_forest_feature_importance.csv` | MDI and permutation feature importance values | Verified |
| `outputs/tables/misclassification_analysis.csv` | Detailed case-by-case audit of held-out test errors | Verified |
| `models/wine_quality_random_forest_pipeline.joblib` | Serialized Random Forest pipeline artifact | Verified |
| `models/wine_quality_sgd_classifier_pipeline.joblib` | Serialized SGD Classifier pipeline artifact | Verified |
| `models/wine_quality_support_vector_svc_pipeline.joblib` | Serialized Support Vector Classifier pipeline artifact | Verified |
| `wine_quality_prediction.py` | Standalone reproducible pipeline script with CLI | Verified |
| `test_wine_quality.py` | Comprehensive 12-test automated unit test suite | Verified |
| `Wine_Quality_Prediction.ipynb` | Fully executed Jupyter Notebook with embedded outputs (1.04 MB) | Verified |
| `README.md` | Standalone comprehensive project documentation | Verified |
| `requirements.txt` | Pinned package dependencies | Verified |
| `docs/Reports/Week_02/Task_6_Wine_Quality_Report.md` | Formal report (this document) | Verified |
| `docs/Architecture/Week_02/Task_6_Wine_Quality_Architecture.md` | Complete engineering architecture specification | Verified |
| `docs/Video_Demo/Week_02/Task_6_Demo_Script.md` | Time-coded video demonstration script | Verified |
| `docs/Video_Demo/Week_02/title_card_task6.png` | 1080p high-resolution static video title card | Verified |
