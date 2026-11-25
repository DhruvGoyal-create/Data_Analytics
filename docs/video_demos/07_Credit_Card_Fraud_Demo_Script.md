# Video Demo Script: Financial Fraud Detection Engine

**Task:** Project 07 (Credit Card Fraud Detection) — Fraud Detection (Task #7 in Master Roadmap)  
**Project:** DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO (`Business-Analytics-Portfolio`)  
**Target Video Duration:** 3 to 5 Minutes  
**Presenter:** `Dhruv Goyal` *(Template document — replace with student full legal name prior to recording)*  
**Static Title Card:** Display `title_card_task7.png` for the first 2 seconds of the recording.

---

## 1. Video Recording Setup Checklist
- [ ] Screen resolution set to 1080p (1920x1080).
- [ ] Display `title_card_task7.png` for exactly 2 seconds as the opening title card.
- [ ] Have `Fraud_Detection.ipynb` open in VS Code / Jupyter Lab with all cells pre-executed.
- [ ] Terminal open at project directory showing passing unit tests (`test_fraud_detection.py`).
- [ ] Microphone tested with clear audio levels.

---

## 2. Timed Presentation Walkthrough

### 0:00 – 0:25: Opening & Title Card (25 Seconds)
- **Visual:** Display static title card `title_card_task7.png` for 2 seconds, then transition camera to webcam + screen sharing.
- **Presenter Dialogue:**
  > "Hello everyone! My name is `Dhruv Goyal`, and welcome to my submission demonstration for Project 07 (Credit Card Fraud Detection): Fraud Detection in the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO program.
  > Today, I'll walk you through an end-to-end machine learning system engineered to detect fraudulent credit card transactions under extreme class imbalance, evaluating cost-sensitive algorithms, SMOTE oversampling, financial loss prevention metrics, and real-time enterprise scalability."

### 0:25 – 1:05: Problem Context & The Accuracy Paradox (40 Seconds)
- **Visual:** Scroll to Section 1 and Section 2 of `Fraud_Detection.ipynb` showing the dataset overview and Figure 1 (Class Imbalance and Log Amounts).
- **Presenter Dialogue:**
  > "We utilize the canonical European Cardholder benchmark dataset from the ULB Machine Learning Group. It comprises 284,807 transactions across 30 features, with zero missing values.
  > Looking at Section 2, only 492 transactions are fraudulent—representing just 0.17% of total volume, or 1 fraud per 578 legitimate transactions.
  > This highlights the classic 'Accuracy Paradox': if we deploy a trivial baseline that always predicts 'Legitimate', it achieves a deceptive 99.83% accuracy while failing to detect a single fraud case and allowing 100% of financial losses to slip through. Standard accuracy is practically useless here."

### 1:05 – 1:50: Exploratory Insights: Time-of-Day Nocturnal Surge (45 Seconds)
- **Visual:** Scroll to Section 4 showing Figure 2 (Time-of-Day Dynamics).
- **Presenter Dialogue:**
  > "In Section 4, we examine diurnal periodicity by converting elapsed seconds into hour-of-day.
  > While legitimate transactions follow normal daylight hours peaking in the afternoon, fraudulent activity exhibits a striking nocturnal surge.
  > Fraud prevalence jumps by nearly 10 times—spiking from the baseline 0.17% to 1.71% at 2:00 AM. Fraud syndicates systematically deploy automated botnets during early morning hours when cardholders are asleep and cannot respond to real-time verification alerts."

### 1:50 – 2:45: Leak-Free Preprocessing & Multi-Model Benchmarks (55 Seconds)
- **Visual:** Scroll to Section 6 and Section 7 showing the model comparison table and Figure 4 (Confusion Matrices).
- **Presenter Dialogue:**
  > "To prevent data leakage, we enforce a strict 80/20 stratified split and scale continuous amount features using RobustScaler fitted solely on training data.
  > We benchmark five distinct paradigms:
  > First, our baseline achieves 99.83% accuracy with zero recall.
  > Second, Balanced Logistic Regression catches 91.8% of frauds (90 out of 98), but produces 1,382 false alarms.
  > Third, SMOTE oversampling with Logistic Regression reduces false alarms down to 160 while maintaining 88.8% recall.
  > Finally, our champion Random Forest Classifier achieves 99.94% accuracy, 81.8% precision, and 82.7% recall, missing only 17 frauds while generating just 18 false alarms across nearly 57,000 legitimate transactions."

### 2:45 – 3:30: Evaluation Trade-Offs: ROC vs. Precision-Recall Curves (45 Seconds)
- **Visual:** Scroll to Section 9 and Section 10 showing Figure 5 (ROC and PR Curves) and the Financial Loss Prevention Audit table.
- **Presenter Dialogue:**
  > "Looking at Figure 5, under 0.17% class imbalance, standard ROC curves are overly optimistic because the vast pool of true negatives masks thousands of false alarms.
  > In contrast, the Precision-Recall curve exposes the true operational reality. Random Forest dominates with a PR-AUC of 0.8187.
  > In our financial audit, Balanced Logistic Regression recovers 93.8% of fraud dollars but freezes \$443,000 in legitimate cardholder volume. Random Forest provides the ideal commercial balance, recovering \$6,552 in fraud with only \$950 in customer friction."

### 3:30 – 4:15: Enterprise Scalability: Proposed 1,000,000 Transactions per Hour Blueprint (45 Seconds)
- **Visual:** Scroll to Section 11 and Section 12 showing the feature importance chart and microservice architecture diagram.
- **Presenter Dialogue:**
  > "Section 11 reveals that PCA components V14, V10, V12, and V4 account for over 59% of tree decisions.
  > In Section 12, we evaluate high-throughput production scalability. Serving 1,000,000 transactions per hour requires handling 278 transactions per second steady-state and 833 TPS during peak bursts, under a target sub-15 millisecond latency budget.
  > In Python, our single-item unbatched Random Forest latency is ~16 milliseconds, while batched inference drops to 0.035 milliseconds per item.
  > We therefore propose a production microservice blueprint using compiled ONNX Runtime or C++ engines and a Redis feature cache to comfortably sustain the sub-15 ms latency target."

### 4:15 – 4:45: Verification & Closing (30 Seconds)
- **Visual:** Switch window to terminal showing `test_fraud_detection.py` running all 12 unit tests passing.
- **Presenter Dialogue:**
  > "Our entire codebase is backed by 12 automated unit tests verifying schema integrity, leakage isolation, model reproducibility, and financial metrics with a 100% pass rate.
  > All scripts, exported figures, serialized models, and documentation are complete and verified.
  > Thank you for reviewing my Advanced Analytics & ML Task 3 presentation for the Data Analytics Portfolio portfolio development!"

---

## 3. Submission Protocol & Hashtags

When posting the video demonstration to LinkedIn, use the following post copy:

```
🚀 Thrilled to share my completed Project 07 (Credit Card Fraud Detection) project for the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO! 💳🛡️📊

Topic: Credit Card Fraud Detection under Extreme Class Imbalance
Tech Stack: Python, Pandas, Scikit-Learn, Imbalanced-Learn (SMOTE), Seaborn, Matplotlib

Key Highlights:
✅ Solved extreme 0.17% class imbalance (1 fraud per 578 transactions) using cost-sensitive learning and SMOTE
✅ Observed nocturnal fraud prevalence rise peaking at 1.71% at 2:00 AM (vs 0.17% overall baseline)
✅ Built leak-free Scikit-Learn / Imblearn Pipelines with RobustScaler fitted strictly on training folds
✅ Champion Random Forest achieved 0.8187 PR-AUC (Average Precision), 0.9766 ROC-AUC, 81.8% Precision, and 82.7% Recall
✅ Conducted financial loss audit: evaluated $10,645 test fraud exposure vs customer checkout friction
✅ Architected enterprise microservice scalability blueprint serving 1,000,000 transactions/hour at <15 ms P99 latency
✅ 12/12 automated unit tests passing with 100% reproducibility

#portfolio #dataanalytics #machinelearning #frauddetection #python #datascience #scikitlearn #smote #fintech
```
