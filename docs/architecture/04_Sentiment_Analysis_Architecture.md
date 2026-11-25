# Technical Architecture Document: Sentiment Analysis Pipeline

**Task:** Project 04 (Sentiment Analysis) — Sentiment Analysis  
**System Name:** Multi-Class NLP Text Classifier Engine  
**Portfolio Scope:** Data Analytics Portfolio (`Business-Analytics-Portfolio`)  
**Track:** Data Analytics  
**Date:** October 2026  
**Status:** `Testing Complete`

---

## 1. System Architecture Overview

The Sentiment Analysis Pipeline provides an automated, high-throughput text processing and classification architecture. It is designed to ingest noisy social media and consumer feedback streams, normalize natural language variance, transform lexical data into high-dimensional numerical feature spaces, and execute low-latency supervised inference across three affective tiers: **Positive**, **Negative**, and **Neutral**.

```
+-----------------------------------------------------------------------------+
|                          1. INGESTION & AUDIT LAYER                         |
|  - Raw Ingestion: twitter_entity_sentiment_raw.csv (75,682 rows)            |
|  - Data Cleansing: Filter 'Irrelevant' labels & purge null texts            |
|  - Retained Domain: 61,949 Clean 3-Class Records across 32 Brand Entities   |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                      2. NLP PREPROCESSING SUBSYSTEM                         |
|  - Case Folding: Canonical lowercasing                                      |
|  - Regex Normalizer: Strip URLs (http\S+), @mentions, HTML entities         |
|  - Punctuation Filter: Keep pure alphabetic characters                      |
|  - Lemmatizer: NLTK WordNetLemmatizer (Morphological base resolution)       |
|  - Negation Filter: Protect {'not', 'no', 'never', 'nor', 'neither'}        |
|  - Output: 60,478 Valid Preprocessed Text Records                           |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                     3. FEATURE VECTORIZATION SUBSYSTEM                      |
|  - Partitioning: 80/20 Stratified Train/Test Split (Zero Data Leakage)       |
|  - Training Corpus: 48,382 Samples | Testing Corpus: 12,096 Samples         |
|  - Model: TfidfVectorizer(ngram_range=(1,2), max_features=10000)            |
|  - Scaling: Sublinear TF (1 + log(tf)) & Min Document Frequency (min_df=3)  |
|  - Vector Dimension: Sparse Matrix (N x 10,000 features)                   |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                     4. STATISTICAL MODELING SUBSYSTEM                       |
|  Model A: MultinomialNB(alpha=0.5)                                          |
|    - Baseline generative classifier assuming conditional independence       |
|  Model B: LogisticRegression(C=1.0, max_iter=1000, l2_penalty)              |
|    - Softmax discriminative model minimizing multinomial cross-entropy      |
|  Model C: LinearSVC(C=0.5, max_iter=2000, hinge_loss) [CHAMPION MODEL]     |
|    - Maximum-margin hyperplane separation in 10,000-dimensional space       |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                     5. EVALUATION & DIAGNOSTICS LAYER                       |
|  - Metric Calculations: Accuracy, Precision, Recall, Macro/Weighted F1      |
|  - Confusion Matrices: Normalized true vs. predicted class distributions    |
|  - Interpretability: Hyperplane coefficient inspection (top n-grams)        |
|  - Error Audit: Automated extraction of 5 linguistic failure modes          |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
|                     6. SERIALIZATION & DEPLOYMENT LAYER                     |
|  - Serialized Artifact: models/sentiment_pipeline_svc.joblib (6.1 MB)       |
|  - Bundled Components: TfidfVectorizer + LinearSVC Estimator + Class Schema |
|  - Production SLA: Inference latency < 2 milliseconds per tweet             |
+-----------------------------------------------------------------------------+
```

---

## 2. Component Design & Implementation Details

### 2.1 Preprocessing Engine (`clean_text`)
- **Module:** `sentiment_analysis.py`
- **Execution Complexity:** $\mathcal{O}(L)$ where $L$ is tweet character length.
- **Core Transformations:**
  1. Lowercase conversion eliminates case sensitivity.
  2. Pre-compiled regular expressions strip URLs (`https?://\S+`), handles (`@\w+`), and non-alphanumeric noise.
  3. Lemmatization maps inflectional forms (*"games"*, *"gaming"*) to lemmas (*"game"*).
  4. The custom stopword filter preserves negation particles `{'not', 'no', 'nor', 'neither', 'never', 'none'}` to protect semantic valence.

### 2.2 Feature Space Engine (`TfidfVectorizer`)
- **Vocabulary Budget:** 10,000 top n-gram tokens.
- **N-gram Range:** Unigrams and Bigrams `(1, 2)`.
- **Sublinear Frequency Scaling:** Replaces raw term frequency $tf$ with $1 + \log(tf)$ for $tf > 0$, bounding the influence of repetitive word spam.
- **Sparsity Management:** Uses SciPy `csr_matrix` (Compressed Sparse Row) representation, maintaining memory footprint under 45 MB in RAM during full corpus vectorization.

### 2.3 Champion Classifier (`LinearSVC`)
- **Optimization Objective:** Primal L2-regularized hinge loss minimization:
  $$\min_{w} \frac{1}{2} w^T w + C \sum_{i=1}^{n} \max(0, 1 - y_i w^T x_i)$$
- **Hyperparameters:** $C = 0.5$, `max_iter = 2000`, `random_state = 42`.
- **Decision Strategy:** One-vs-Rest (OvR) multi-class decomposition across the three sentiment classes.
- **Computational Benchmark:** Fit completed in 0.67 seconds across 48,382 training rows, achieving 83.97% test accuracy.

---

## 3. Real-Time Production Inference Architecture

For production environments (e.g., streaming Kafka social feeds or live CRM webhooks), the serialized artifact `sentiment_pipeline_svc.joblib` enables stateless, low-latency microservice scoring:

```python
import joblib

class SentimentClassifierService:
    def __init__(self, model_path='models/sentiment_pipeline_svc.joblib'):
        artifact = joblib.load(model_path)
        self.vectorizer = artifact['vectorizer']
        self.model = artifact['model']
        self.classes = artifact['classes']

    def predict(self, raw_text: str) -> dict:
        # 1. Clean incoming string
        cleaned = clean_text_pipeline(raw_text)
        # 2. Vectorize
        vec = self.vectorizer.transform([cleaned])
        # 3. Predict & compute margin confidence
        pred_class = self.model.predict(vec)[0]
        decision_scores = self.model.decision_function(vec)[0]
        return {
            'text': raw_text,
            'predicted_sentiment': pred_class,
            'confidence_margins': {cls: round(float(score), 4) 
                                   for cls, score in zip(self.classes, decision_scores)}
        }
```

---

## 4. Architectural Verification & Quality Gate

1. **Test Coverage:** Verified via `test_sentiment_analysis.py` (8 automated test cases passing in 0.57s).
2. **Reproducibility:** Pinned in `requirements.txt` with locked random seeds (`seed=42`).
3. **Execution Integrity:** Headless notebook execution completed cleanly with 0 tracebacks via `nbconvert`.
