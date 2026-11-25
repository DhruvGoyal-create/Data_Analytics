# DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — DECISION LOG

**Date Initialized:** 16 October 2026  
**Keeper:** Data Analytics Project Lead  

---

## Decision Entries

### DEC-001: Repository Structure & Unified Documentation Architecture
- **Date:** 16 October 2026
- **Context:** The workspace contained an existing `Data Analytics` folder with `doc_work`, `LEVEL-1`, and `LEVEL-2` subfolders, while the project specification requested a root-level `docs/` with a specific 5-tier documentation taxonomy (`Strategy`, `Reports`, `Presntations`, `Video_Demo`, `Architecture`). The official Data Analytics Portfolio guidelines also mandate folder naming format `Business-Analytics-Portfolio/[TrackName]-[Level/Task]-[ProjectName]/`.
- **Options Considered:**
  1. Relocate all files to root and delete `Data Analytics`. (High risk of breaking existing paths or user files).
  2. Maintain duplicate folders in both root and `Data Analytics`. (High risk of desynchronization).
  3. Create the canonical structure under `Data Analytics/doc_work` and establish an NTFS Directory Junction at `Business-Analytics-Portfolio/doc_work`, alongside canonical project directories in `LEVEL-1` and `LEVEL-2`.
- **Decision:** Option 3.
- **Rationale:** Transparently satisfies both workspace conventions without duplicating disk storage or creating desynchronization risks. Evaluators checking either `Business-Analytics-Portfolio/doc_work` or `Business-Analytics-Portfolio/Data Analytics/doc_work` see identical up-to-date documentation.

---

### DEC-002: Python Environment & Dependency Installation Protocol
- **Date:** 16 October 2026
- **Context:** The environment is running Python 3.13.6 on Windows x64. Audit revealed core packages (`pandas`, `numpy`, `sklearn`, `scipy`, `nltk`, `textblob`, `plotly`, `matplotlib`) are installed, while `seaborn`, `wordcloud`, `imblearn`, `xgboost`, `pyspellchecker`, and `textdistance` are missing.
- **Options Considered:**
  1. Create a heavyweight virtual environment with deep learning frameworks.
  2. Use the existing Python 3.13 environment and selectively install missing lightweight packages (`seaborn`, `wordcloud`, `imblearn`, `pyspellchecker`) via `pip` / `uv`.
- **Decision:** Option 2.
- **Rationale:** Minimizes startup latency and disk bloat while keeping dependencies fully isolated and reproducible. For packages with potential C-extension build issues on Python 3.13 (e.g. `imblearn`/`xgboost`), native `scikit-learn` algorithms (e.g., `HistGradientBoostingClassifier`, `class_weight='balanced'`) serve as proven, battle-tested backups.

---

### DEC-003: Task Prioritization & Execution Ordering (The 4-Phase Strategy)
- **Date:** 16 October 2026
- **Context:** The development cycle was executed with an initial intensive 30-day focus (ending 15 Nov 2026). The minimum requirement is 3 tasks; the LOR objective is 9 tasks. Attempting all 9 tasks in parallel risks delivering 9 half-finished projects.
- **Options Considered:**
  1. Sequential execution in arbitrary numerical order (PRJ-01 to PRJ-04, then PRJ-05 to PRJ-09).
  2. Complexity-first (start with hardest tasks: Fraud Detection, Autocomplete).
  3. Phased Value-Gating:
     - Phase 1 (Min Gate): PRJ-01 (EDA Retail), PRJ-03 (Data Cleaning), PRJ-06 (Wine Quality).
     - Phase 2 (Core): PRJ-02 (Customer Segmentation), PRJ-05 (House Prices).
     - Phase 3 (Ecosystem): PRJ-04 (Sentiment Analysis), PRJ-08 (Play Store).
     - Phase 4 (LOR): PRJ-07 (Fraud Detection), PRJ-09 (Autocomplete & Autocorrect).
- **Decision:** Option 3 (Phased Value-Gating).
- **Rationale:** Phase 1 focuses on achieving the stated minimum requirement target (3 tasks from either level) by October 24. Subsequent phases progressively expand the portfolio to 5, 7, and 9 tasks to maximize completion toward the LOR objective.

---

### DEC-004: Dataset Sourcing & Reproducibility Principles
- **Date:** 16 October 2026
- **Context:** Official tasks suggest searching Kaggle. However, Kaggle API often requires individual API keys, interactive web login, or acceptance of competition agreements, which hampers automated reproducibility.
- **Options Considered:**
  1. Rely exclusively on Kaggle manual downloads.
  2. Use authoritative public repositories (UCI Machine Learning Repository, Project Gutenberg, open-access GitHub CSV mirrors) with automated downloading scripts, supplemented by curated clean offline backups in each project folder.
- **Decision:** Option 2.
- **Rationale:** Ensures that every project can be rerun end-to-end without requiring personal Kaggle API tokens or broken links.

---

### DEC-005: Video Demo & Compliance Standardization
- **Date:** 16 October 2026
- **Context:** Official rules require demo videos to begin with a strict 2-second static title card displaying (1) Full Name, (2) Assigned Track, and (3) Task Title. In addition, 2 peer evaluation comments on LinkedIn are graded prerequisites.
- **Options Considered:**
  1. Leave title card creation and video recording as an ad-hoc task at the very end.
  2. Implement an automated script to generate high-resolution standard title cards (`title_card.png`) and comprehensive walkthrough narration scripts for every task during its documentation phase.
- **Decision:** Option 2.
- **Rationale:** Eliminates the risk of video disqualification and streamlines the recording and posting process for the student.

---

### DEC-006: Dataset Selection for Task PRJ-01 (EDA on Retail Sales Data)
- **Date:** 16 October 2026
- **Context:** The official checklist requires: "Top 10 best-selling products; revenue by product category". The widely shared 1,000-row Kaggle `retail_sales_dataset.csv` contains only 3 broad product categories (`Beauty`, `Clothing`, `Electronics`) and lacks an individual item/product column, making it impossible to genuinely compute a Top 10 product ranking without fabricating artificial sub-items.
- **Options Considered:**
  1. Use the 1,000-row dataset and treat the 3 categories as products (violates the requirement to show top 10 products).
  2. Use the Istanbul Customer Shopping dataset (contains 8 categories, still fewer than 10).
  3. Use the multi-year Global Retail Store Sales dataset (originating from Microsoft AdventureWorks / FreeCodeCamp benchmark) with 113,036 rows, continuous transaction dates (2011–2016), age, gender, 3 product categories, 17 sub-categories, and 130 unique products.
- **Decision:** Option 3.
- **Rationale:** Satisfies 100% of official checklist items natively without inventing synthetic fields. Provides genuine multi-year time-series variance, authentic price-quantity elasticity, and legitimate product rankings.

---

### DEC-007: Task PRJ-01 Targeted Audit & Corrective Standard
- **Date:** 17 October 2026
- **Context:** A targeted audit of Task PRJ-01 was performed to verify provenance, licensing, deduplication definition, descriptive statistics alignment, and accounting identities.
- **Audit Findings & Actions:**
  1. *Provenance Chain:* Established full lineage: Microsoft AdventureWorks sample database -> curated & flattened by rmotr.com / Santiago Basulto for FreeCodeCamp (`ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example`) -> MIT License. Corrected earlier shorthand attribution.
  2. *Deduplication Sensitivity:* Audited 1,000 duplicate rows across all 18 fields. Net revenue impact is -\$444,236 (-0.52%) and profit impact is -\$174,861 (-0.54%). Both raw (\$85.27M) and deduplicated (\$84.83M) totals are documented.
  3. *Accounting Characterization:* Clarified that $\text{Cost} = \text{Qty} \times \text{Unit\_Cost}$ (100% exact) and $\text{Profit} = \text{Revenue} - \text{Cost}$ (100% exact) represent *internal arithmetic consistency* of the table's calculated fields rather than proof of external real-world physical transactions. Realized revenue reflects nominal list price minus transaction discounts ($\text{Discount} = \text{Qty} \times \text{Unit\_Price} - \text{Revenue}$).
  4. *Descriptive Statistics:* Confirmed all descriptive statistics match `df_clean` exactly, and corrected modal age text from 29.0 to 31.0 years (count: 3,086).
- **Decision:** Update notebook, README, and Report in place without rebuilding. Preserve all outputs and files.

---

### DEC-008: Deterministic Imputation & Domain-Driven Outlier Defense for Task PRJ-03
- **Date:** 17 October 2026
- **Context:** The benchmark dataset for Project 03 (Data Cleaning Pipeline) (`dirty_cafe_sales.csv`, 10,000 records) exhibits widespread data corruption, including placeholder strings (`'UNKNOWN'`, `'ERROR'`), missing values across items, quantities, prices, and totals, plus 259 transactions at \$25.00 flagged by the 1.5*IQR rule.
- **Key Decisions Enacted:**
  1. *Deterministic vs. Naive Imputation:* Discovered fixed menu pricing matrix (\$1.00 to \$5.00 across 8 items). Used domain menu lookup and algebraic constraint propagation ($\text{Total} \equiv \text{Quantity} \times \text{Price}$) to recover 969 items, 479 quantities, 533 prices, and 502 totals with 100% mathematical precision, rejecting naive mean/mode imputation which would break accounting identities.
  2. *Defensible Purge Boundary:* Purged only 26 records (0.26% of rows, \$54 total revenue) where two or more financial dimensions were simultaneously missing, making algebraic recovery mathematically impossible.
  3. *Domain Defense of Statistical Outliers:* Retained all 259 IQR outliers at \$25.00. Domain verification proved they represent valid purchases of 5 Salads @ \$5.00. Capping or deleting them would violate physical business reality, destroy \$6,475 in authentic sales, and corrupt arithmetic consistency ($5 \times \$5.00 \neq \$24.00$).
  4. *Categorical Transparency:* Imputed missing payment methods (31.8%) and locations (39.6%) with `'Unknown'` rather than the mode to avoid artificial channel mix distortion.

---

### DEC-009: Task PRJ-03 Quality Gate Audit & Corrective Standards
- **Date:** 17 October 2026
- **Context:** A formal Quality Gate review was conducted on Task PRJ-03 to address quantity reconstruction tolerances, date imputation distortion, provenance tracking, non-tautological accounting checks, and template title card compliance.
- **Audit Findings & Actions:**
  1. *Quantity Reconstruction:* Mandated strict integer validation within floating-point tolerance ($\text{abs}(\text{ratio} - \text{round}(\text{ratio})) < 10^{-5}$) and positive non-zero denominators. Verified that all 456 reconstructed quantities in the dataset yield exact integers in $\{1, 2, 3, 4, 5\}$. Added unit tests for fractional rejection, zero denominators, negative values, and missing inputs.
  2. *Date Handling:* Eliminated median date imputation (`2023-07-02`). Preserved all 460 unparseable/missing dates as `NaT` in memory and blank in CSV export. Documented that undated records are excluded from time-series charts but retained for overall financial analysis.
  3. *Reconstruction Provenance:* Audited and classified every repaired field into directly observed (90.3% - 95.4%), deterministically reconstructed (4.6% - 4.8%), inferred from domain assumption (4.8%), or unresolved (4.8% ambiguous items labeled explicitly).
  4. *Non-Tautological Accounting Proof:* Independently tested the 8,544 directly observed raw records where Quantity, Price, and Total Spent were all present, confirming 100.00% exact match ($0$ mismatches), which empirically validates the generative multiplicative model before algebraic solving.
  5. *Template Compliance:* Replaced generic candidate strings with `Dhruv Goyal` and explicitly flagged title cards as templates requiring customization prior to recording.

---

### DEC-010: Dataset Selection & RFM Clustering Architecture for Task PRJ-02 (Customer Segmentation)
- **Date:** 18 October 2026
- **Context:** The official checklist for Task PRJ-02 requires behavioral RFM feature engineering, descriptive statistics, StandardScaler normalization, Elbow Method K-Means clustering, 2D scatter plots, cluster persona profiling, and marketing recommendations.
- **Dataset Audit & Selection:**
  - Audited the Task 1 Retail Sales dataset (`retail_sales_data.csv`). Found: contains demographic fields (`Customer_Age`, `Customer_Gender`) but **lacks an individual customer identifier** (`CustomerID`), making entity-level RFM grouping impossible.
  - Sourced and validated the canonical **UCI Online Retail Dataset** (541,909 rows, Dr. Daqing Chen, CC BY 4.0), which contains explicit `CustomerID`, `InvoiceNo`, `InvoiceDate`, `Quantity`, and `UnitPrice`.
- **Key Technical Decisions:**
  1. *Entity Filtering:* Excluded 135,080 rows without `CustomerID` (guest checkouts) and 8,905 cancelled invoices (`InvoiceNo` starting with 'C'), retaining 397,884 clean purchases across 4,338 unique customers.
  2. *Skewness Normalization:* Handled severe positive right-skewness in Monetary (£280k max vs. £674 median) via `np.log1p()` transformation before applying `StandardScaler` to ensure Euclidean distance reflects true behavioral variance.
  3. *Cluster Count Optimization:* Evaluated $K=2$ to $K=8$ using Inertia and Silhouette scores. Selected $K=4$ at the elbow inflection point (Inertia=3,938.51) and silhouette local peak (0.3371), producing four actionable CRM personas (Champions, Regulars, Promising, At-Risk).
  4. *Pareto Discovery:* Discovered that 16.51% of customers (Champions) generate 64.87% of total portfolio revenue (£5.78M of £8.91M), directly guiding marketing retention priorities.
  5. *Silhouette Interpretation & Hypothesis Framing:* Framed the peak silhouette score of 0.3371 as a modest cluster-separation indicator reflecting continuous behavioral variance, and explicitly labeled all CRM marketing tactics as proposed business hypotheses requiring empirical A/B testing.

---

### DEC-011: Dataset Sourcing & 3-Class Sentiment Domain Isolation (Task PRJ-04)
- **Date:** 19 October 2026
- **Context:** Data Analytics Portfolio Task 4 requires building a multi-class sentiment classifier for positive, negative, and neutral text.
- **Dataset Audit:**
  - Evaluated candidate sentiment corpora (Twitter Airline Sentiment, Financial PhraseBank, Twitter Entity Sentiment).
  - Selected the canonical **Kaggle Twitter Entity Sentiment Analysis** benchmark (75,682 records, 32 global brands and tech entities).
  - Audited raw labels: Negative (22,808), Positive (21,109), Neutral (18,603), Irrelevant (13,162). Purged 686 rows with null text.
- **Decision:** Excluded the off-topic 'Irrelevant' class to isolate the core 3-class sentiment problem mandated by the official task syllabus. Retained 61,949 clean, highly balanced records (36.5% Negative, 33.8% Positive, 29.7% Neutral).

---

### DEC-012: Negation-Preserving NLP Preprocessing & Lemmatization Strategy
- **Date:** 19 October 2026
- **Context:** Standard text preprocessing often strips critical linguistic modifiers or truncates word stems destructively.
- **Decision:**
  1. *WordNet Lemmatization over Truncation:* Replaced Porter/Snowball stemming with NLTK WordNet lemmatizer to map words to true morphological roots (*"crying"* $\rightarrow$ *"cry"*, *"better"* $\rightarrow$ *"good"*).
  2. *Strict Negation Preservation:* Standard stopword lists naively strip negation particles (*"not"*, *"no"*, *"never"*, *"nor"*), which inverts sentiment valence (e.g., *"not good"* becoming *"good"*). Explicitly preserved all negation particles in the token stream.
  3. *Regex Sanitation:* Stripped URLs (`http\S+`), user mentions (`@\w+`), and non-alphanumeric noise, producing 60,478 valid preprocessed records.

---

### DEC-013: Sublinear TF-IDF Feature Engineering & Zero-Leakage Split
- **Date:** 19 October 2026
- **Context:** Feature extraction must balance computational efficiency, n-gram context capture, and dimensional sparsity.
- **Decision:**
  1. *Sublinear TF Scaling:* Configured `sublinear_tf=True` ($1 + \log(tf)$) to prevent word spamming within individual tweets from dominating document length.
  2. *N-Gram Range & Vocabulary:* Selected `ngram_range=(1, 2)` with `max_features=10000` and `min_df=3` to capture key bigram phrases (*"not good"*, *"game broken"*, *"love it"*) while pruning isolated typos.
  3. *Stratified Partitioning:* Enforced an 80/20 stratified train/test split (48,382 train / 12,096 test samples). Fitted TF-IDF strictly on the training partition and transformed the test partition, guaranteeing zero data leakage.

---

### DEC-014: Architectural Champion Selection (LinearSVC) & 5-Mode Error Audit
- **Date:** 19 October 2026
- **Context:** Compare baseline probabilistic modeling against discriminative linear classifiers and perform qualitative error analysis.
- **Empirical Results:**
  - *Multinomial Naive Bayes (Baseline):* 76.33% Accuracy | 0.7564 Macro F1 (Fit: 0.05s).
  - *Multinomial Logistic Regression:* 81.15% Accuracy | 0.8082 Macro F1 (Fit: 0.98s).
  - *Linear Support Vector Classifier (LinearSVC):* **83.97% Accuracy** | **0.8372 Macro F1** (Fit: 0.67s).
- **Decision & Error Audit:**
  1. *Champion Model:* Selected LinearSVC as the champion architecture due to its mathematical formulation finding maximum-margin hyperplanes in high-dimensional sparse TF-IDF spaces.
  2. *Serialization:* Bundled vectorizer, LinearSVC estimator, and class schema into `models/sentiment_pipeline_svc.joblib` (6.1 MB).
  3. *Qualitative Error Audit:* Audited five concrete test-set failure modes: Sarcasm/Irony (`ERR-01`), Negation Inversion (`ERR-02`), Contrastive Clauses (`ERR-03`), Factual Noise (`ERR-04`), and Domain Slang Polysemy (`ERR-05`), documenting root causes in `outputs/tables/error_analysis_cases.csv`.

---

### DEC-015: Target Formulation & Class Imbalance Diagnostic for Wine Quality (`PRJ-06`)
- **Date:** 19 October 2026
- **Context:** The UCI Red Wine Quality dataset (Cortez et al., 2009) contains sensory scores from 3 to 8 across 1,599 observations. Exactly 82.49% of all samples cluster in median scores 5 (42.59%) and 6 (39.90%), while extreme tail classes (grades 3, 4, 8) represent less than 5% combined.
- **Experimental Finding:** When grouping into a 3-tier formulation (Low: 3–4, Medium: 5–6, High: 7–8), unweighted gradient optimization completely sacrifices the Low class (Precision: 0.00, Recall: 0.00, F1: 0.00), achieving deceptively high 85.94% overall accuracy by predicting Medium indiscriminately.
- **Decision:**
  1. *Binary Target Formulation:* Formulated production target as `good_wine = (quality >= 7).astype(int)`, defining Positive as Good/Premium wine ($N = 217, 13.57\%$) and Negative as Typical/Low table wine ($N = 1,382, 86.43\%$).
  2. *Viticultural Alignment:* This cutoff maps directly to commercial certification standards and wine competition medal cutoffs (e.g., Decanter World Wine Awards), establishing an actionable quality gate for reserve tiering.
  3. *Leakage Prevention:* Applied `StandardScaler` strictly inside Scikit-Learn `Pipeline` constructs fitted on an 80/20 stratified training split (1,279 train / 320 test samples).

---

### DEC-016: Model Architecture Selection & Precision Optimization for Wine Quality (`PRJ-06`)
- **Date:** 19 October 2026
- **Context:** Benchmarked four classifier architectures on held-out test data: Stratified Dummy Baseline, Random Forest (200 trees), SGD Classifier (`loss='log_loss'`, `class_weight='balanced'`), and Support Vector Classifier (RBF kernel, balanced weights).
- **Benchmark Findings:**
  - *Dummy Baseline:* 74.69% Accuracy | 0.4707 ROC-AUC | 0.0899 F1.
  - *SGD Classifier:* 83.75% Accuracy | 0.8929 ROC-AUC | 79.07% Recall | 44.16% Precision (43 False Positives).
  - *Support Vector (SVC):* 84.69% Accuracy | 0.9015 ROC-AUC | 76.74% Recall | 45.83% Precision (39 False Positives).
  - *Random Forest (Champion):* **94.06% Accuracy** | **0.9523 ROC-AUC** | **92.86% Precision** | 60.47% Recall | **0.7324 F1** | **0.8535 PR-AUC** (Only 2 False Positives).
- **Decision:**
  1. *Champion Selection:* Selected Random Forest as the production champion model. In high-value wine distribution, mislabeling bulk table wine as premium estate reserve destroys commercial brand equity. Random Forest's 92.86% precision (vs ~45% for SGD/SVC) protects brand trust.
  2. *Feature Importance Consensus:* Both MDI Gini Importance and held-out Permutation Importance identify `alcohol` (17.1% Gini, 0.0748 ROC-AUC drop), `sulphates` (11.3% Gini, 0.0541 drop), and `volatile acidity` (10.3% Gini, 0.0358 drop) as primary biochemical drivers.
  3. *Error Diagnosis:* Audited all 19 held-out errors, identifying the *Low-Alcohol Masking Effect*: 76% of missed premium wines had alcohol between 9.2% and 10.5%, where judges rewarded tannin balance and mouthfeel nuance that bulk chemical metrics underweighted.

---

### DEC-017: California Housing Benchmark Selection & ColumnTransformer Encapsulation (`PRJ-05`)
- **Date:** 21 October 2026
- **Context:** Advanced Analytics & ML Task 1 requires an econometric regression analysis predicting house prices with linear models, handling missing values, encoding categorical variables, performing correlation analysis, and residual diagnostics.
- **Decision:**
  1. *Dataset Selection:* Selected the canonical California Housing benchmark (Pace & Barry, 1997 / 1990 U.S. Census Bureau, StatLib), comprising 20,640 census block group observations across 10 continuous and categorical attributes (SHA-256: `8A3727F4CF54AC1A327F69B1D5B4DB54C5834EA81C6E4EFC0D163300022A685E`).
  2. *Missing Data Strategy:* Verified that exactly 207 rows (1.003%) contain missing values strictly confined to `total_bedrooms`. Encapsulated imputation via `SimpleImputer(strategy='median')` fitted exclusively on training splits to prevent data leakage.
  3. *Categorical Dummy Encoding:* Handled `ocean_proximity` (5 levels) via `OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')`. Dropping the first category (`<1H OCEAN`) establishes it as the econometric baseline reference and prevents the dummy variable trap (exact collinearity with the intercept).
  4. *Pipeline Encapsulation vs Spatial Dependence:* Packaged all numerical imputation, scaling, and categorical encoding into a unified Scikit-Learn `ColumnTransformer` embedded in model pipelines, eliminating preprocessing summary-statistic leakage. Explicitly documented that while preprocessing leakage is eliminated, standard random 80/20 train/test splitting does not isolate spatial autocorrelation between geographically contiguous census tracts across splits.

---

### DEC-018: Econometric Linear Modeling, Regularization Asymptotics & Census Ceiling Bias (`PRJ-05`)
- **Date:** 21 October 2026
- **Context:** Evaluate Ordinary Least Squares (OLS) Linear Regression against regularized models (Ridge and Lasso), analyze standardized coefficients, and conduct residual diagnostics.
- **Empirical Findings:**
  1. *Baseline vs OLS (Primary Educational Model):* The `DummyRegressor(strategy='mean')` baseline yields Test RMSE = \$114,485.64 and $R^2 = -0.0002$. OLS Linear Regression achieves Test $R^2 = \mathbf{0.6254}$, Test RMSE = **\$70,059.19**, Test MAE = **\$50,670.49**, and 5-Fold Training CV-RMSE = **\$68,622.54 $\pm$ \$1,438.77** (38.8% RMSE error reduction).
  2. *Regularization Asymptotics:* Evaluated Ridge ($\alpha=1.0$) and Lasso ($\alpha=100.0$). Because $N = 16,512$ training records vastly exceeds $P = 12$ encoded features ($N/P > 1,375$), OLS is exceptionally well-conditioned with ample degrees of freedom. Consequently, Ridge ($R^2 = 0.6253$, RMSE = \$70,067.48) and Lasso ($R^2 = 0.6252$, RMSE = \$70,082.90) perform virtually identically to OLS, proving that standard OLS is already at the empirical limit of linear functional form for this feature set.
  3. *Standardized Beta Associations (Not Causal Drivers):* Because the dataset is observational cross-sectional data, coefficients reflect conditional partial associations (*ceteris paribus* within the specification), not established causal mechanisms. `median_income` exhibits the strongest positive conditional association (**+\$75,167.77** per 1 SD shift), while inland location incurs a **-\$39,786.66** conditional difference relative to `<1H OCEAN`. Geographic coordinates (`latitude` -\$54,416; `longitude` -\$53,827) reflect lower interior valuations.
  4. *Gauss-Markov Nuance & Historical Census Ceiling Bias:* 
     - Residual diagnostics revealed that a test residual mean near zero (\$1,040.85) indicates low global aggregate bias, but **does not verify Gauss-Markov assumptions** (strict exogeneity, homoscedasticity, error independence are violated in this spatial dataset).
     - Furthermore, 184 test records (4.46%) are artificially capped at \$500,001 by 1990 Census Bureau truncation. Because true market values above \$500,000 were never recorded, predictions above the ceiling cannot be empirically validated against actual transaction prices. On uncapped homes (<$500k), linear RMSE is **\$61,848.69**; on capped homes ($500k+), apparent RMSE inflates to **\$167,698.58** due to deviation from the artificial administrative ceiling.
  5. *Non-Linear Comparison Benchmark:* Benchmarked against `RandomForestRegressor(n_estimators=100, max_depth=16)`, which achieves Test $R^2 = \mathbf{0.8143}$ and Test RMSE = **\$49,323.89**, demonstrating the non-linear spatial interactions of latitude and longitude across California coastal employment hubs.

---

### DEC-019: Class Imbalance Mitigation, Cost-Sensitive Evaluation & Production Architecture for Credit Card Fraud Detection (`PRJ-07`)
- **Date:** 22 October 2026
- **Context:** Advanced Analytics & ML Task 3 requires building a fraud detection classifier on an extremely skewed financial transaction dataset (284,807 transactions, 492 fraud cases, 0.1727% fraud rate = 1:578 imbalance), analyzing EDA distributions, preventing synthetic oversampling leakage, comparing models, auditing financial trade-offs, and designing an enterprise inference architecture.
- **Key Decisions & Empirical Findings:**
  1. *Dataset Sourcing & Leakage-Free Partitioning:*
     - Sourced the canonical ULB Credit Card Fraud dataset (284,807 rows, 31 columns, SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`).
     - Established an 80/20 stratified split (`random_state=42`) yielding 227,845 training transactions (394 frauds) and 56,962 held-out test transactions (98 frauds).
     - Applied `RobustScaler` to skewed non-PCA attributes (`Time`, `Amount`), fitting strictly on training partitions to prevent test leakage.
  2. *Synthetic Oversampling (SMOTE) Encapsulation:*
     - Evaluated `SMOTE` oversampling. To eliminate test leakage, SMOTE was strictly wrapped inside an `imblearn.pipeline.Pipeline` with `sampling_strategy=0.1` (moderating the synthetic ratio to avoid noise saturation) so synthetic examples are generated exclusively from training data during model fitting and cross-validation folds.
  3. *The Accuracy Paradox & Metric Complementarity (PR-AUC and ROC-AUC):*
     - The `DummyClassifier(strategy='most_frequent')` achieves **99.83% accuracy** despite scoring 0% recall, 0% precision, and recovering \$0.00 in fraud loss.
     - On 0.1727% prevalence, ROC-AUC yields high absolute values (>0.97) because the massive true-negative pool ($N = 56,864$) dominates the False Positive Rate denominator. In contrast, Average Precision / PR-AUC omits true negatives and evaluates positive-class precision against recall directly. Both metrics provide complementary diagnostic perspectives; Average Precision is prioritized for evaluating false-alarm trade-offs under severe skew.
  4. *Model Benchmark Matrix on Held-Out Test Set ($N = 56,962$):*
     - *Unweighted Logistic Regression:* Acc 99.92%, Prec 82.89%, Rec 64.29%, PR-AUC 0.7429, TP 63, FP 13, FN 35. Detected fraud exposure: \$4,506.29 (42.33%), friction: \$902.67.
     - *Balanced Logistic Regression:* Acc 97.56%, Prec 6.11%, Rec **91.84%**, PR-AUC 0.7156, TP 90, FP 1,382, FN 8. Catches 90/98 fraud events and detects **\$9,986.32 (93.81%)** in fraud exposure, but introduces high operational friction (1,382 false alarms, \$443,862 legitimate checkout volume held).
     - *SMOTE + Logistic Regression:* Acc 99.70%, Prec 35.22%, Rec 88.78%, PR-AUC 0.7523, TP 87, FP 160, FN 11. Detects **\$8,818.43 (82.84%)** with 160 false alarms.
     - *Random Forest (Champion):* Acc **99.94%**, Prec **81.82%**, Rec **82.65%**, F1 **0.8223**, ROC-AUC **0.9766**, PR-AUC **0.8187**, TP 81, FP 18, FN 17. Detects **\$6,551.61** with only 18 false alarms and \$950.71 customer friction.
  5. *Temporal Observation (Nocturnal Rise in Proportion):*
     - Observed that fraud prevalence rises significantly in early morning hours: from the 0.17% overall baseline to **1.71% at 2:00 AM** (57 frauds / 3,328 transactions) and **1.04% at 4:00 AM** (23 frauds / 2,209 transactions).
     - In payments operations, this is commonly hypothesized to reflect automated transactions exploiting delayed cardholder verification while asleep; however, the dataset does not contain cardholder timezones, merchant locations, or attack logs to establish causation.
  6. *Scalability Engineering & Proposed Blueprint:*
     - Serving 1,000,000 transactions/hour requires a sustained throughput of 277.8 TPS and peak burst capacity of 833.3 TPS under a target <15 ms SLA.
     - Measured local Python latency: single-item DataFrame prediction is **~0.44 ms** for Logistic Regression and **~16.3 ms** for Random Forest (dropping to **~0.035 ms/item** when batched).
     - Documented that achieving sub-15 ms unbatched latency in production requires compiling models to an optimized C++/ONNX engine (or implementing gateway micro-batching), and proposed a 3-worker Kubernetes topology as an architectural design blueprint (unverified in live production).

