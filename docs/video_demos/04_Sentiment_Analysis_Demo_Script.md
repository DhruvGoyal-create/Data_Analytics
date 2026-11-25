# Video Demo Script: Sentiment Analysis Pipeline

**Task:** Project 04 (Sentiment Analysis) — Sentiment Analysis  
**Project:** DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO (`Business-Analytics-Portfolio`)  
**Target Video Duration:** 3 to 5 Minutes  
**Presenter:** `Dhruv Goyal` *(Template document — replace with student full legal name prior to recording)*  
**Static Title Card:** Display `title_card_task4.png` for the first 2 seconds of the recording.

---

## 1. Video Recording Setup Checklist
- [ ] Screen resolution set to 1080p (1920x1080).
- [ ] Display `title_card_task4.png` for exactly 2 seconds as the opening title card.
- [ ] Have `Sentiment_Analysis_Modeling.ipynb` open in VS Code / Jupyter Lab with all cells pre-executed.
- [ ] Terminal open at project directory showing clean git status and unit test results (`test_sentiment_analysis.py`).
- [ ] Microphone tested with clear audio levels.

---

## 2. Timed Presentation Walkthrough

### 0:00 – 0:20: Opening & Title Card (20 Seconds)
- **Visual:** Display static title card `title_card_task4.png` for 2 seconds, then transition camera to webcam + screen sharing.
- **Presenter Dialogue:**
  > "Hello everyone! My name is `Dhruv Goyal`, and welcome to my submission demonstration for Project 04 (Sentiment Analysis): Sentiment Analysis in the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO program.
  > Today, I'll walk you through an end-to-end Natural Language Processing system that ingests raw social media feedback, applies advanced text preprocessing, extracts TF-IDF feature representations, and trains three distinct machine learning classifiers to predict Positive, Negative, and Neutral sentiment with up to 83.97% accuracy."

### 0:20 – 0:50: Business Problem & Dataset Overview (30 Seconds)
- **Visual:** Scroll to Section 1 of `Sentiment_Analysis_Modeling.ipynb` showing Figure 1 (Sentiment Class Distribution).
- **Presenter Dialogue:**
  > "In modern enterprise consumer tech, brands face thousands of daily customer mentions. Manually reviewing this text to catch outages or product bugs is impossible.
  > We source the Twitter Entity Sentiment benchmark comprising over 75,000 tweets across 32 major brands, including Microsoft, Google, PlayStation, and Call of Duty.
  > After auditing nulls and filtering out off-topic irrelevant posts, we isolate the core 3-class sentiment problem: 22,624 Negative tweets, 20,932 Positive tweets, and 18,393 Neutral tweets, giving us a highly balanced, robust training corpus."

### 0:50 – 1:35: NLP Preprocessing & WordClouds (45 Seconds)
- **Visual:** Scroll to Section 2 and Section 3 showing code samples and Figure 2 (Multi-Panel WordClouds).
- **Presenter Dialogue:**
  > "Social media text is filled with noise like URLs, user handles, and irregular punctuation.
  > Our preprocessing pipeline lowercases text, strips URLs and mentions via regular expressions, and applies WordNet lemmatization.
  > Critically, unlike standard naive stopword filters, our pipeline explicitly preserves negation terms—such as 'not', 'no', and 'never'—ensuring that phrases like 'not good' don't get inverted into positive sentiment.
  > Looking at our WordClouds in Figure 2, we immediately observe distinctive lexical signatures: Positive tweets feature words like 'love', 'best', 'excited', and 'dope', while Negative tweets center on 'fix', 'problem', 'glitch', 'broken', and 'terrible'."

### 1:35 – 2:35: Feature Extraction & Model Benchmarks (60 Seconds)
- **Visual:** Scroll to Section 4 and Section 5 showing the TF-IDF formula, metrics comparison table, and Figure 3 (Confusion Matrices) & Figure 4 (Performance Comparison).
- **Presenter Dialogue:**
  > "To convert text into machine learning features, we use TF-IDF with sublinear term-frequency scaling and unigram plus bigram tokens across a 10,000-dimensional vocabulary, partitioned using an 80/20 stratified split.
  > We benchmark three distinct classifier architectures:
  > First, our baseline Multinomial Naive Bayes achieves 76.33% accuracy. It's exceptionally fast at 50 milliseconds, but its conditional independence assumption struggles with complex word interactions.
  > Second, Multinomial Logistic Regression reaches 81.15% accuracy with well-calibrated probabilities.
  > Finally, our champion model—Linear Support Vector Classifier, or LinearSVC—achieves 83.97% accuracy and an 0.8372 Macro F1-score. SVM's maximum-margin hyperplane optimization is uniquely suited for high-dimensional, sparse lexical text data."

### 2:35 – 3:30: Qualitative Error Analysis & Linguistic Breakdown (55 Seconds)
- **Visual:** Scroll to Section 8 showing the 5 Audited Misclassification Cases.
- **Presenter Dialogue:**
  > "A critical requirement of this project is understanding where and why the model fails. In Section 8, we audit five concrete failure modes from the unseen test set:
  > 1. Sarcasm: In 'No fucking thank you', the word 'thank you' misled the model into predicting Positive.
  > 2. Negation and Slang: Gamer vernacular like 'this game is sick' or 'crazy entertainment' uses words that standard lexicons penalize as negative, despite carrying enthusiastic praise.
  > 3. Contrastive clauses: Sentences using 'but' combine positive praise with negative frustration, causing vector averaging to collapse into Neutral.
  > Diagnosing these cases demonstrates that modern production NLP requires ongoing domain-specific lexicon refinement."

### 3:30 – 4:00: Business Use Cases, Verification & Closing (30 Seconds)
- **Visual:** Switch to terminal, run `python -m unittest test_sentiment_analysis.py`, show all 8 tests passing in 0.57s.
- **Presenter Dialogue:**
  > "In production, this model can power three critical enterprise capabilities:
  > 1. Automated CX triage, routing severe negative complaints to Tier-2 support within 60 seconds.
  > 2. Real-time brand crisis alerts when negative sentiment spikes above normal thresholds.
  > 3. Competitor sentiment benchmarking.
  > As you can see in my terminal, all eight automated unit tests pass with 100% success, confirming data integrity, pipeline execution, and model serialization.
  > Thank you to the Data Analytics Portfolio team for this fantastic learning experience!"

---

## 3. Social Media Submission Template (LinkedIn Post)

```markdown
🚀 Exciting Milestone Update: Completed Project 04 (Sentiment Analysis) (Sentiment Analysis) of my Data Analytics portfolio development at Data Analytics Portfolio!

As part of the #Business-Analytics-Portfolio program, I developed an end-to-end Natural Language Processing (NLP) text classification system capable of analyzing customer feedback across 32 major global consumer entities and classifying sentiment into Positive, Negative, and Neutral cohorts.

📊 Project Highlights:
- Sourced and audited 75,682 customer feedback records, isolating a clean 61,949 3-class target dataset.
- Engineered a linguistically-informed text preprocessing pipeline featuring WordNet lemmatization and negation-preserving stopword filtering.
- Extracted 10,000 sublinear TF-IDF unigram and bigram features.
- Benchmarked 3 classifiers: Multinomial Naive Bayes (76.33%), Logistic Regression (81.15%), and LinearSVC (83.97% Accuracy | 0.8372 Macro F1).
- Conducted deep qualitative error analysis on 5 real-world failure modes (sarcasm, slang polysemy, contrastive conjunctions).
- Verified via an automated 8-test unit test suite and serialized model artifact.

Repository: [Insert GitHub Repo Link]
Demo Video: [Insert Video Link]

#portfolio #dataanalytics #machinelearning #nlp #sentimentanalysis #datascience #portfolio development #python
```
