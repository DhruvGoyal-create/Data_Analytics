# DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — RISK REGISTER

**Last Updated:** 16 October 2026  
**Review Frequency:** Weekly or after major milestone completion  
**Risk Scoring:** Severity (1-5) × Likelihood (1-5) = Risk Score (1-25)

---

## 1. Active Risk Log

| Risk ID | Category | Description | Likelihood (1-5) | Severity (1-5) | Score (1-25) | Mitigation Strategy | Owner | Trigger Condition | Status |
|---------|----------|-------------|------------------|----------------|--------------|---------------------|-------|-------------------|--------|
| **RSK-01** | Dependency | `seaborn` and visualization packages missing in local Python 3.13 | 4 | 3 | **12** | Installed `seaborn==0.13.2`, `ipykernel`, `nbformat`, `nbconvert` in environment. | Lead Dev | Import failure during script execution | Closed (Verified) |
| **RSK-02** | Dependency | `imblearn` / `xgboost` compatibility or installation failure on Python 3.13 | 3 | 4 | **12** | Test `imblearn` install early; use scikit-learn built-in `class_weight='balanced'` and manual stratified sampling if wheel build issues arise. | Lead Dev | Setup of Task PRJ-07 | Open |
| **RSK-03** | Methodology | Data leakage during SMOTE oversampling in Fraud Detection (Task PRJ-07) | 3 | 5 | **15** | Strictly fit SMOTE only on training fold using `imblearn.pipeline.Pipeline` or explicit split isolation before resampling. Never apply to test set. | QA Lead | Feature engineering phase of Task PRJ-07 | Open (Strict Rule) |
| **RSK-04** | Dataset | Sourcing Kaggle datasets without manual authentication or broken links | 3 | 3 | **9** | Use standard direct UCI repository mirrors, open Github mirrors, and verified public open data URLs (e.g. archive.ics.uci.edu, Gutenberg). | Data Lead | Dataset acquisition step | Mitigated (Direct sources mapped) |
| **RSK-05** | Submission | Video walkthrough missing mandatory 2-second initial static title card | 2 | 5 | **10** | Provide an automated Python script generating the exact 1920x1080 static PNG title card for each task with Name, Track, and Task Title. | QA Lead | Video demo drafting | Open (Standard template created) |
| **RSK-06** | Submission | Evaluation rejection due to non-compliant folder naming | 2 | 5 | **10** | Maintain canonical directories adhering strictly to `Business-Analytics-Portfolio/[TrackName]-[Level/Task]-[ProjectName]/` with symbolic links/junctions where needed. | Lead Dev | Repository organization | Closed (Architecture enforced) |
| **RSK-07** | Compliance | Intern fails to complete 2 graded peer evaluation comments on LinkedIn | 2 | 4 | **8** | Schedule peer reviews on Days 29-30, track reviewer URLs and take screenshot evidence before submission form closure. | Project Lead | Final week submission checklist | Open (Scheduled for Week 5) |
| **RSK-08** | Schedule | Scope creep on advanced NLP/deep learning prevents completing 9 tasks | 3 | 4 | **12** | Strictly enforce minimum-complete approach: fulfill all official checklist items first, run tests, write docs, then freeze. Do not build excessive web apps or unrequested architectures. | Project Lead | Daily milestone reviews | Open (Active Control) |
| **RSK-09** | Memory | Memory exhaustion on large text corpus (Task PRJ-09) or fraud dataset (Task PRJ-07) | 2 | 3 | **6** | Use chunked processing, downcast numeric datatypes (float64 to float32), and limit vocabulary size with minimum document frequency thresholds. | Lead Dev | Data loading phase | Open |

---

## 2. Risk Mitigation Protocols

### Protocol A: Python Dependency Isolation
Before executing any task notebook, run automated dependency checks. If a package is missing, install using:
```pwsh
python -m pip install --upgrade <package_name>
```
If building from source fails on Python 3.13, utilize standard pure-Python or `scikit-learn` native alternatives documented in the Decision Log.

### Protocol B: Data Leakage Prevention Gate
In every predictive task (PRJ-04, PRJ-05, PRJ-06, PRJ-07):
1. Split into train and test sets (`test_size=0.20`, `stratify` where classification is used) BEFORE any scaling, encoding, imputation, or oversampling.
2. Fit transformers/scalers only on `X_train`, transform `X_train` and `X_test`.
3. Report metrics strictly on the untouched `X_test`.

### Protocol C: Evidence-Backed Acceptance Verification
No task is moved to `Testing Complete` without:
- Zero traceback execution.
- Actual numeric metric outputs recorded in the task report.
- Plots saved to disk in PNG format.
