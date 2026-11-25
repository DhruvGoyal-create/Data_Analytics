# Advanced Analytics & ML — Task 2: Wine Quality Prediction

## Supervised Classification & Physicochemical Profiling: Random Forest, SGD & Support Vector Machines

**Track:** Data Analytics  
**Organization:** Data Analytics Portfolio (`Business-Analytics-Portfolio`)  
**Task Code:** `PRJ-06`  
**Execution Date:** October 2026  
**Candidate Name:** `Dhruv Goyal` *(Template document — replace with full legal name prior to official video recording/submission)*  
**Current Status:** `Testing Complete` (All models evaluated, 12/12 automated unit tests passing, notebook executed headlessly)

---

## 1. Project Overview & Problem Context

In sensory evaluation research, understanding how objective chemical properties correspond to human sensory judgments is a classic analytical problem. In wine evaluation, sensory panels independently score wines on integer quality scales.

This project delivers an **end-to-end machine learning classification system** built on the canonical **UCI Machine Learning Repository Wine Quality benchmark** (Cortez et al., 2009). The pipeline profiles 1,599 red Vinho Verde wines across 11 physicochemical properties, diagnoses extreme central class imbalance, conducts an explicit duplicate record and train-test boundary leakage audit, implements a leak-free Scikit-Learn `Pipeline` architecture, and evaluates four model architectures:
1. **Dummy Classifier (Baseline)** (`strategy='most_frequent'`, majority-class predictor)
2. **Random Forest Classifier** (Ensemble tree bagging with non-linear interaction modeling)
3. **Stochastic Gradient Descent (SGD) Classifier** (Log-loss convex optimization with L2 regularization)
4. **Support Vector Classifier (SVC)** (Non-linear RBF kernel projection in standardized feature space)

---

## 2. Official Task Checklist & Compliance

| # | Official Checklist Requirement | Implementation Details | Verification Status |
|---|--------------------------------|------------------------|---------------------|
| 1 | **Load Wine Quality dataset & inspect structure** | Loaded UCI red wine benchmark (1,599 rows, 12 cols, 0 nulls, 240 duplicates audited). Verified SHA-256 integrity (`4a402cf...`). | **Verified** |
| 2 | **EDA: distribution plots for all chemical features** | 11-panel distribution figure (`fig1_wine_distributions.png`) and comprehensive descriptive statistics table exported. | **Verified** |
| 3 | **Correlation heatmap** | Pearson correlation matrix (`fig3_correlation_heatmap.png`) highlighting positive correlates (alcohol, sulphates) and negative correlates (volatile acidity). | **Verified** |
| 4 | **Discuss class imbalance & modeling implications** | Detailed mathematical analysis of the 82.5% median concentration (grades 5 & 6) and failure of unweighted multi-class loss on rare grades (3, 4, 8). | **Verified** |
| 5 | **Feature engineering: bin quality scores & justify** | Formulated binary analytical target (`good_wine` $\ge 7$, 13.57% positive) and empirically demonstrated why 3-tier grouping fails for rare grades. | **Verified** |
| 6 | **Stratified train/test split** | 80/20 stratified split (1,279 train / 320 test) preserving exact 13.57% positive class proportion across splits. | **Verified** |
| 7 | **Train 3 classifiers (Random Forest, SGD, SVC)** | Trained Random Forest, SGD Classifier (`loss='log_loss'`, `class_weight='balanced'`), and SVC (RBF kernel, balanced weights). | **Verified** |
| 8 | **Evaluate each: accuracy, report, confusion matrix** | Computed Accuracy, Balanced Accuracy, Precision, Recall, F1 (binary/macro/weighted), ROC-AUC, PR-AUC, and confusion matrices (`fig4_confusion_matrices.png`). | **Verified** |
| 9 | **Feature importance chart for Random Forest** | Extracted MDI Gini importance and held-out Permutation Importance with error bars across 20 shuffle iterations (`fig6_feature_importance.png`). | **Verified** |
| 10 | **Comparison table & deployment conclusion** | Side-by-side performance tables exported for both standard and deduplicated splits; trade-off analysis documented. | **Verified** |

---

## 3. Dataset Lineage, Provenance & Duplicate Audit

- **Canonical Repository:** UCI Machine Learning Repository (Dataset ID: 186)
- **Direct UCI URL:** `https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`
- **Primary Literature Reference:**  
  P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis. *Modeling wine preferences by data mining from physicochemical properties*, Decision Support Systems, Elsevier, 47(4):547-553, 2009.
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **File SHA-256 Checksum:** `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e`
- **Dimensions:** 1,599 observations $\times$ 12 attributes (11 continuous physicochemical features, 1 integer sensory target).

### Forensic Duplicate & Boundary Leakage Audit:
- **Total Observations:** 1,599
- **Unique Observations:** 1,359
- **Exact Duplicate Rows:** 240
- **Rows in Duplicate Clusters:** 460
- **Conflicting Labels:** 0 (all duplicate feature vectors have identical sensory scores).
- **Train-Test Boundary Leakage in Standard 80/20 Split:** **77 out of 320 test samples (24.06%) are exact copies of a training sample**.
- **Implication for Generalization:** Because tree-based models easily memorize exact training points, evaluating on the standard split with 24.1% cross-boundary overlap inflates test performance. To maintain strict scientific transparency, we report both the standard benchmark and a duplicate-aware evaluation on the 1,359 unique rows.

---

## 4. Sensory Quality Target & Class Imbalance Diagnostic

### Raw Quality Score Breakdown:
- **Quality 3:** 10 samples (0.63%)
- **Quality 4:** 53 samples (3.31%)
- **Quality 5:** 681 samples (42.59%)
- **Quality 6:** 638 samples (39.90%)
- **Quality 7:** 199 samples (12.45%)
- **Quality 8:** 18 samples (1.13%)
- **Total:** 1,599 samples

### Why Unweighted Multi-Class & 3-Tier Formulations Fail:
1. **Central Clustering:** Over **82.49%** of wines reside in scores 5 and 6. The extreme tails (3, 4, and 8) account for less than 5% combined.
2. **Empirical Failure of 3-Tier Grouping:** When binned into a 3-tier grouping (Low: 3–4, Medium: 5–6, High: 7–8):
   - A standard classifier achieves **0.00 recall and 0.00 precision on the Low tier** (all 13 test samples missed) because predicting Medium blindly yields 85.9% nominal accuracy.
3. **The Binary Formulation ($\ge 7$ vs $< 7$):**
   - **Higher-Rated Wine (1):** Scores 7–8 ($N = 217$, **13.57%**).
   - **Typical / Lower-Rated Wine (0):** Scores 3–6 ($N = 1,382$, **86.43%**).
   - *Scope Note:* In this project, $\ge 7$ serves as an analytical threshold for higher-rated samples within this specific dataset. It does not represent an official commercial certification, reserve award, or guaranteed market pricing.

---

## 5. Model Performance Benchmarks

### 5.1 Standard Benchmark Split ($N = 320$ Test Samples, 24.06% Duplicate Overlap)

| Model Architecture | Accuracy | Balanced Acc | Precision (Class 1) | Recall (Class 1) | F1-Score | Macro F1 | ROC-AUC | PR-AUC | 5-Fold CV ROC-AUC |
|--------------------|:--------:|:------------:|:-------------------:|:----------------:|:--------:|:--------:|:-------:|:------:|:-----------------:|
| **Dummy (most_frequent)** | 86.56% | 50.00% | 0.00% | 0.00% | 0.0000 | 0.4640 | *N/A (Constant)* | *N/A (Constant)* | *N/A* |
| **Random Forest** | **94.06%** | **79.87%** | **92.86%** | **60.47%** | **0.7324** | **0.8495** | **0.9523** | **0.8535** | **$0.894 \pm 0.020$** |
| **SGD Classifier (Balanced)** | 83.75% | 81.77% | 44.16% | 79.07% | 0.5667 | 0.7333 | 0.8929 | 0.5536 | $0.806 \pm 0.020$ |
| **Support Vector (SVC, Balanced)** | 84.69% | 81.33% | 45.83% | 76.74% | 0.5739 | 0.7403 | 0.9015 | 0.6165 | $0.875 \pm 0.017$ |

*Baseline Metric Behavior Note:* `DummyClassifier(strategy='most_frequent')` predicts class 0 for all instances. On 320 test samples (277 negative, 43 positive), it achieves $277/320 = 86.56\%$ accuracy and $50.00\%$ balanced accuracy. Because all output scores are identical, positive-class precision, recall, and F1 are 0.0, and ROC-AUC is mathematically undefined (represented as N/A).

### 5.2 Deduplicated Dataset Split ($N = 272$ Test Samples, 0% Duplicate Overlap)

| Model Architecture | Accuracy | Balanced Acc | Precision (Class 1) | Recall (Class 1) | F1-Score | Macro F1 | ROC-AUC | PR-AUC | 5-Fold CV ROC-AUC |
|--------------------|:--------:|:------------:|:-------------------:|:----------------:|:--------:|:--------:|:-------:|:------:|:-----------------:|
| **Dummy (most_frequent)** | 86.40% | 50.00% | 0.00% | 0.00% | 0.0000 | 0.4635 | *N/A* | *N/A* | *N/A* |
| **Random Forest** | 89.34% | 66.50% | 72.22% | 35.14% | 0.4727 | 0.7067 | 0.8749 | 0.6018 | $0.869 \pm 0.031$ |
| **SGD Classifier (Balanced)** | 71.69% | 77.92% | 30.77% | 86.49% | 0.4539 | 0.6314 | 0.8252 | 0.4455 | $0.835 \pm 0.038$ |
| **Support Vector (SVC, Balanced)** | 80.51% | 79.61% | 39.19% | 78.38% | 0.5225 | 0.7001 | 0.8839 | 0.5818 | $0.867 \pm 0.030$ |

---

## 6. Feature Importance & Predictive Associations

Feature importances represent **statistical associations within this sample**, not causal biochemical proof:

| Feature Name | Gini Importance (MDI) | Permutation Importance (ROC-AUC Drop) | Permutation Std |
|--------------|:---------------------:|:-------------------------------------:|:---------------:|
| `alcohol` | **0.1705 (17.1%)** | **0.0748** | $\pm 0.0144$ |
| `sulphates` | **0.1130 (11.3%)** | **0.0541** | $\pm 0.0093$ |
| `volatile acidity` | **0.1028 (10.3%)** | **0.0358** | $\pm 0.0043$ |
| `fixed acidity` | 0.0754 (7.5%) | 0.0151 | $\pm 0.0036$ |
| `total sulfur dioxide` | 0.0806 (8.1%) | 0.0130 | $\pm 0.0068$ |
| `density` | 0.0992 (9.9%) | 0.0082 | $\pm 0.0075$ |
| `free sulfur dioxide` | 0.0613 (6.1%) | 0.0079 | $\pm 0.0018$ |
| `chlorides` | 0.0734 (7.3%) | 0.0060 | $\pm 0.0046$ |
| `residual sugar` | 0.0690 (6.9%) | 0.0059 | $\pm 0.0035$ |
| `pH` | 0.0625 (6.3%) | 0.0049 | $\pm 0.0028$ |
| `citric acid` | 0.0921 (9.2%) | 0.0031 | $\pm 0.0041$ |

---

## 7. Forensic Error Analysis & Trade-Offs

- **The Low-Alcohol Masking Effect:** 13 of the 17 missed higher-rated wines (False Negatives on standard split) had alcohol between **9.2% and 10.5%** (well below the higher-rated class median of 11.6%). In these cases, human judges rated the wines favorably for balance, but the model—relying on the statistical correlation with alcohol—assigned them lower probabilities.
- **The False Positives (2 samples on standard split):** Both false positive samples had borderline human ratings of **6** with chemical attributes near the decision boundary (alcohol $\ge 12.0\%$, high citric acid).
- **Precision vs. Recall Trade-Off:** Random Forest provides high precision (fewer false positives), while SVC and SGD provide higher recall (fewer false negatives) at the expense of lower precision.

---

## 8. Repository File Structure

```
projects/06_Wine_Quality_Prediction/
├── data/
│   ├── winequality-red.csv                  # Canonical UCI Red Wine dataset (1,599 rows, 12 cols)
│   └── winequality-white.csv                # Canonical UCI White Wine dataset (4,898 rows, 12 cols)
├── outputs/
│   ├── figures/
│   │   ├── fig1_wine_distributions.png      # 11-feature distribution histograms + KDEs
│   │   ├── fig2_quality_distribution.png    # Raw score distribution vs binary target
│   │   ├── fig3_correlation_heatmap.png     # Full Pearson correlation heatmap
│   │   ├── fig4_confusion_matrices.png      # Side-by-side test confusion matrices
│   │   ├── fig5_roc_pr_curves.png           # ROC curves and PR curves with honest baselines
│   │   └── fig6_feature_importance.png      # Gini vs Permutation Importance comparison
│   └── tables/
│       ├── wine_dataset_summary.csv         # High-level dataset metadata & summary
│       ├── physicochemical_descriptive_stats.csv # Descriptive statistics for all features
│       ├── duplicate_leakage_audit.csv      # Audit of 240 duplicates and 77 cross-boundary rows
│       ├── model_performance_comparison.csv # Standard split benchmark performance table
│       ├── deduplicated_model_comparison.csv# Deduplicated split benchmark performance table
│       ├── random_forest_feature_importance.csv  # MDI and permutation importances
│       └── misclassification_analysis.csv   # Forensic audit of test errors
├── models/
│   ├── wine_quality_random_forest_pipeline.joblib       # Serialized RF pipeline
│   ├── wine_quality_sgd_classifier_pipeline.joblib     # Serialized SGD pipeline
│   └── wine_quality_support_vector_svc_pipeline.joblib # Serialized SVC pipeline
├── build_l2_t2_notebook.py                  # Script to assemble reproducible Jupyter Notebook
├── Wine_Quality_Prediction.ipynb            # Fully executed 21-cell Jupyter Notebook (1.04 MB)
├── wine_quality_prediction.py               # Standalone, CLI-executable modeling pipeline
├── test_wine_quality.py                     # Comprehensive 12-test automated unit test suite
├── README.md                                # Comprehensive standalone documentation
└── requirements.txt                         # Pinned Python package dependencies
```

---

## 9. Reproducibility & Execution Instructions

```bash
# 1. Run Standalone Modeling Pipeline (generates both standard and deduplicated benchmarks)
python "projects/06_Wine_Quality_Prediction/wine_quality_prediction.py"

# 2. Run Automated Unit Test Suite (12 tests)
python "projects/06_Wine_Quality_Prediction/test_wine_quality.py"

# 3. Execute Jupyter Notebook Headlessly
jupyter nbconvert --to notebook --execute --inplace "projects/06_Wine_Quality_Prediction/Wine_Quality_Prediction.ipynb"
```
