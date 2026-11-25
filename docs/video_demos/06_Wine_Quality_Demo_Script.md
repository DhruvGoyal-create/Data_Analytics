# Video Demo Script: Wine Quality Prediction Engine

**Task:** Project 06 (Wine Quality Prediction) — Wine Quality Prediction (Task #6 in Master Roadmap)  
**Project:** DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO (`Business-Analytics-Portfolio`)  
**Target Video Duration:** 3 to 5 Minutes  
**Presenter:** `Dhruv Goyal` *(Template document — replace with student full legal name prior to recording)*  
**Static Title Card:** Display `title_card_task6.png` for the first 2 seconds of the recording.

---

## 1. Video Recording Setup Checklist
- [ ] Screen resolution set to 1080p (1920x1080).
- [ ] Display `title_card_task6.png` for exactly 2 seconds as the opening title card.
- [ ] Have `Wine_Quality_Prediction.ipynb` open in VS Code / Jupyter Lab with all cells pre-executed.
- [ ] Terminal open at project directory showing passing unit tests (`test_wine_quality.py`).
- [ ] Microphone tested with clear audio levels.

---

## 2. Timed Presentation Walkthrough

### 0:00 – 0:20: Opening & Title Card (20 Seconds)
- **Visual:** Display static title card `title_card_task6.png` for 2 seconds, then transition camera to webcam + screen sharing.
- **Presenter Dialogue:**
  > "Hello everyone! My name is `Dhruv Goyal`, and welcome to my submission demonstration for Project 06 (Wine Quality Prediction): Wine Quality Prediction in the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO program.
  > Today, I'll walk you through an end-to-end machine learning system that analyzes objective laboratory measurements of wine chemistry—such as acidity, density, and alcohol content—to predict sensory wine quality with over 94% accuracy and a 0.952 ROC-AUC."

### 0:20 – 0:50: Problem Statement & Dataset Lineage (30 Seconds)
- **Visual:** Scroll to Section 1 of `Wine_Quality_Prediction.ipynb` showing the dataset overview and Figure 1 (Feature Distributions).
- **Presenter Dialogue:**
  > "In the commercial wine industry, evaluating wine quality historically depends on professional sommelier tasting panels. While expert tastings are essential, they are expensive, subjective, prone to palate fatigue, and cannot be used continuously across industrial bottling lines.
  > To solve this, we use the canonical UCI Wine Quality benchmark created by Cortez and colleagues. It contains 1,599 red Vinho Verde wines tested across 11 chemical dimensions, evaluated by professional judges on a quality scale from 3 to 8. Our dataset has zero missing values, and its SHA-256 integrity has been independently verified."

### 0:50 – 1:35: Physicochemical Distributions & Class Imbalance Diagnostic (45 Seconds)
- **Visual:** Scroll to Section 4 and Section 5 showing Figure 2 (Quality Distribution & Class Imbalance).
- **Presenter Dialogue:**
  > "Examining the target variable reveals an extreme central concentration: over 82% of all wines fall into median scores 5 and 6. Rare grades like 3, 4, and 8 make up less than 5% combined.
  > If we try to train a naive 3-tier classifier (Low, Medium, High), the model completely ignores the low-quality class—yielding 0% recall—because guessing 'Medium' maximizes simple accuracy.
  > We engineer a binary classification threshold: Higher-Rated Wine scoring 7 or higher (representing the top 13.6% of samples) versus Typical or Lower-Rated Wine below 7.
  > Importantly, we also audit duplicate records: 240 exact duplicate rows exist in the raw data, resulting in 77 test samples overlapping across the train-test boundary in a standard split. To maintain scientific honesty, we evaluate both the standard split and a deduplicated split."

### 1:35 – 2:30: Leak-Free Pipelines & Model Benchmarks (55 Seconds)
- **Visual:** Scroll to Section 6 and Section 7 showing the model comparison table, Figure 4 (Confusion Matrices), and Figure 5 (ROC and PR Curves).
- **Presenter Dialogue:**
  > "To guarantee zero data leakage, all feature scaling via StandardScaler is encapsulated directly inside Scikit-Learn Pipeline constructs, fitted strictly on an 80/20 stratified training split.
  > We benchmark three distinct classifier architectures against a majority-class dummy baseline:
  > First, our baseline achieves 86.56% accuracy simply by predicting the majority class, with an undefined ROC-AUC because it offers no discriminatory ranking.
  > Second, Stochastic Gradient Descent (SGD) with balanced weights achieves 83.8% accuracy and 79.1% recall.
  > Third, our Support Vector Classifier with an RBF kernel reaches 84.7% accuracy with 76.7% recall.
  > Finally, Random Forest achieves 94.06% accuracy, 0.9523 ROC-AUC, and 92.86% precision on the standard split. On our deduplicated benchmark without boundary overlap, Random Forest achieves 89.34% accuracy and 0.8749 ROC-AUC, while SVC achieves 80.51% accuracy and 0.8839 ROC-AUC."

### 2:30 – 3:15: Feature Importance & Biochemical Insights (45 Seconds)
- **Visual:** Scroll to Section 8 showing Figure 6 (Dual Feature Importance: Gini vs Permutation).
- **Presenter Dialogue:**
  > "Understanding which chemical variables correlate with higher ratings is crucial. Looking at Figure 6, both Gini Impurity and Permutation Importance on held-out test data point to the exact same predictive associations:
  > Number one is alcohol content, accounting for over 17% of tree importance. Higher ethanol correlates strongly with higher taster scores.
  > Number two is sulphates, or potassium sulphate, which shows a positive predictive association.
  > And number three is volatile acidity—or acetic acid. This shows the strongest negative correlation with quality scores, as vinegar notes detract from sensory ratings."

### 3:15 – 3:55: Forensic Error Analysis (40 Seconds)
- **Visual:** Scroll to Section 9 showing the audited error table.
- **Presenter Dialogue:**
  > "We also audited all 19 misclassifications on our test set.
  > Interestingly, 76% of the missed premium wines had lower alcohol levels—between 9.2% and 10.5%. While human judges rewarded these lighter wines for subtle balance and oak tannins, the model penalized them for lower ethanol.
  > Meanwhile, our only two false positives had actual quality scores of 6—right on the borderline—with chemical profiles nearly identical to grade 7 wines. This confirms the model's errors are chemically coherent rather than arbitrary."

### 3:55 – 4:30: Viticultural Recommendations & Conclusion (35 Seconds)
- **Visual:** Scroll to Section 10 showing recommendations and switch to terminal showing all 12 unit tests passing.
- **Presenter Dialogue:**
  > "Based on our findings, we provide three actionable recommendations for winemakers:
  > 1. Keep volatile acidity strictly below 0.52 grams per liter through temperature-controlled fermentation.
  > 2. Maintain potassium sulphate between 0.65 and 0.80 grams per liter for optimal antioxidant protection.
  > 3. Deploy our serialized Random Forest pipeline directly in cellar laboratories to pre-screen barrel lots, saving substantial sommelier tasting costs.
  > All code, exported figures, serialized models, and 12 automated unit tests are fully documented and passing in the repository. Thank you for watching!"

---

## 3. Submission Protocol & Hashtags

When posting the video demonstration to LinkedIn or social media, use the following post copy:

```
🚀 Pleased to share my completed Project 06 (Wine Quality Prediction) project for the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO! 🍷📊

Topic: Wine Quality Prediction using Machine Learning
Tech Stack: Python, Pandas, Scikit-Learn (Random Forest, SGD, SVC), Seaborn, Matplotlib

Key Highlights:
✅ Solved severe class imbalance (82.5% central concentration) via defensible binary formulation (quality >= 7)
✅ Built leak-free Scikit-Learn Pipelines with StandardScaler and Stratified 5-Fold Cross-Validation
✅ Champion Random Forest achieved 94.06% Accuracy (0.952 ROC-AUC) on standard split and 89.34% (0.875 ROC-AUC) on deduplicated split
✅ Identified Alcohol, Sulphates, and Volatile Acidity as primary statistical quality correlates
✅ 12/12 automated unit tests passing with full test reproducibility

#portfolio #dataanalytics #machinelearning #python #datascience #winequality #scikitlearn
```
