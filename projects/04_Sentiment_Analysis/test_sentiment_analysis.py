"""
test_sentiment_analysis.py
Automated Unit Test Suite for Project 04 (Sentiment Analysis) — Sentiment Analysis
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO (Business-Analytics-Portfolio)

Validates data integrity, NLP cleaning functions, model artifacts, evaluation benchmarks,
figure generation, and qualitative error analysis structure.
"""

import os
import sys
import unittest
import joblib
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Add script directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sentiment_analysis import clean_text


class TestSentimentAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.raw_path = os.path.join(cls.base_dir, 'data', 'twitter_entity_sentiment_raw.csv')
        cls.clean_path = os.path.join(cls.base_dir, 'data', 'twitter_sentiment_clean.csv')
        cls.fig_dir = os.path.join(cls.base_dir, 'outputs', 'figures')
        cls.tab_dir = os.path.join(cls.base_dir, 'outputs', 'tables')
        cls.model_path = os.path.join(cls.base_dir, 'models', 'sentiment_pipeline_svc.joblib')
        
        cls.lemmatizer = WordNetLemmatizer()
        cls.stop_words = set(stopwords.words('english'))

    def test_01_datasets_exist(self):
        """Verify raw and clean datasets exist on disk and exceed reasonable byte sizes."""
        self.assertTrue(os.path.exists(self.raw_path), "Raw Twitter entity dataset missing.")
        self.assertTrue(os.path.exists(self.clean_path), "Clean 3-class sentiment dataset missing.")
        self.assertGreater(os.path.getsize(self.raw_path), 5_000_000, "Raw dataset appears truncated.")
        self.assertGreater(os.path.getsize(self.clean_path), 4_000_000, "Clean dataset appears truncated.")

    def test_02_clean_dataset_integrity(self):
        """Verify clean dataset has exactly 3 sentiment classes and zero nulls in text/sentiment."""
        df_clean = pd.read_csv(self.clean_path)
        self.assertGreater(len(df_clean), 50_000, "Expected >50,000 clean tweets.")
        expected_classes = {'Positive', 'Negative', 'Neutral'}
        actual_classes = set(df_clean['sentiment'].unique())
        self.assertEqual(actual_classes, expected_classes, f"Unexpected sentiment classes: {actual_classes}")
        self.assertEqual(df_clean['text'].isnull().sum(), 0, "Null texts found in clean dataset.")
        self.assertEqual(df_clean['sentiment'].isnull().sum(), 0, "Null sentiment labels found in clean dataset.")

    def test_03_text_cleaning_logic(self):
        """Verify regex cleaning strips handles, URLs, and punctuation while preserving negations."""
        sample_raw = "Check this out @user https://t.co/xyz123! The game was NOT bad at all #awesome"
        cleaned = clean_text(sample_raw, self.lemmatizer, self.stop_words)
        
        # Must remove @user
        self.assertNotIn('@user', cleaned)
        self.assertNotIn('user', cleaned)
        # Must remove URL
        self.assertNotIn('http', cleaned)
        self.assertNotIn('t.co', cleaned)
        # Must lowercase
        self.assertEqual(cleaned, cleaned.lower())
        # Must preserve negation 'not'
        self.assertIn('not', cleaned.split())
        # Must retain content token 'bad'
        self.assertIn('bad', cleaned.split())

    def test_04_model_artifacts_exist(self):
        """Verify best serialized model pipeline loads correctly and contains vectorizer + classifier."""
        self.assertTrue(os.path.exists(self.model_path), "Serialized model pipeline missing.")
        artifact = joblib.load(self.model_path)
        self.assertIn('vectorizer', artifact, "Vectorizer missing from serialized pipeline.")
        self.assertIn('model', artifact, "Model estimator missing from serialized pipeline.")
        self.assertIn('classes', artifact, "Class list missing from serialized pipeline.")
        self.assertEqual(len(artifact['classes']), 3, "Expected 3 classes in serialized pipeline.")

    def test_05_evaluation_metrics_validity(self):
        """Verify model evaluation table contains 3 models and accuracies exceed baseline threshold (>65%)."""
        eval_path = os.path.join(self.tab_dir, 'model_evaluation_metrics.csv')
        self.assertTrue(os.path.exists(eval_path), "Evaluation metrics table missing.")
        df_eval = pd.read_csv(eval_path)
        self.assertEqual(len(df_eval), 3, "Expected 3 models in evaluation table.")
        
        models_expected = {'Multinomial Naive Bayes', 'Logistic Regression', 'Linear Support Vector Classifier (LinearSVC)'}
        models_actual = set(df_eval['Model'].unique())
        self.assertEqual(models_actual, models_expected)
        
        for _, row in df_eval.iterrows():
            self.assertGreater(row['Accuracy'], 0.65, f"Model {row['Model']} accuracy too low ({row['Accuracy']}).")
            self.assertGreater(row['F1_Macro'], 0.65, f"Model {row['Model']} F1_Macro too low ({row['F1_Macro']}).")

    def test_06_figures_generated(self):
        """Verify all 5 required figures exist and exceed 15 KB."""
        figures = [
            'fig1_sentiment_distribution.png',
            'fig2_sentiment_wordclouds.png',
            'fig3_confusion_matrices.png',
            'fig4_model_performance_comparison.png',
            'fig5_top_sentiment_features.png'
        ]
        for fig in figures:
            p = os.path.join(self.fig_dir, fig)
            self.assertTrue(os.path.exists(p), f"Missing figure: {fig}")
            self.assertGreater(os.path.getsize(p), 15_000, f"Figure truncated or empty: {fig}")

    def test_07_tables_generated(self):
        """Verify all required CSV tables exist on disk."""
        tables = [
            'class_distribution_summary.csv',
            'model_evaluation_metrics.csv',
            'top_predictive_keywords.csv',
            'error_analysis_cases.csv',
            'inference_latency_benchmark.csv'
        ]
        for tab in tables:
            p = os.path.join(self.tab_dir, tab)
            self.assertTrue(os.path.exists(p), f"Missing table: {tab}")
            self.assertGreater(os.path.getsize(p), 50, f"Table truncated: {tab}")

    def test_08_error_analysis_content(self):
        """Verify error analysis table contains at least 5 distinct failure cases with required columns."""
        err_path = os.path.join(self.tab_dir, 'error_analysis_cases.csv')
        df_err = pd.read_csv(err_path)
        self.assertGreaterEqual(len(df_err), 5, "Expected at least 5 error analysis cases.")
        required_cols = {'Error_ID', 'Category', 'Entity', 'Tweet_Text', 'True_Sentiment', 'Predicted_Sentiment', 'Linguistic_Explanation'}
        self.assertTrue(required_cols.issubset(set(df_err.columns)), f"Missing columns in error analysis: {required_cols - set(df_err.columns)}")
        # Check that true != predicted in all error rows
        self.assertTrue((df_err['True_Sentiment'] != df_err['Predicted_Sentiment']).all(), "Found non-error row in error analysis table.")

    def test_09_no_residual_url_tokens(self):
        """Verify web artifacts and bare domain tokens like 'com' are excluded from predictive features."""
        keywords_path = os.path.join(self.tab_dir, 'top_predictive_keywords.csv')
        df_kw = pd.read_csv(keywords_path)
        top_ngrams = set(df_kw['Ngram'].tolist())
        self.assertNotIn('com', top_ngrams, "Residual URL token 'com' found in top predictive keywords.")
        self.assertNotIn('pic', top_ngrams, "Residual URL token 'pic' found in top predictive keywords.")
        
        artifact = joblib.load(self.model_path)
        vocab = artifact['vectorizer'].vocabulary_
        self.assertNotIn('com', vocab, "Residual domain token 'com' present in TF-IDF vocabulary.")

    def test_10_latency_benchmark_table(self):
        """Verify inference latency benchmark table exists and records valid sub-millisecond latencies."""
        lat_path = os.path.join(self.tab_dir, 'inference_latency_benchmark.csv')
        self.assertTrue(os.path.exists(lat_path), "Latency benchmark table missing.")
        df_lat = pd.read_csv(lat_path)
        self.assertEqual(len(df_lat), 1, "Expected 1 row of latency summary benchmarks.")
        row = df_lat.iloc[0]
        self.assertEqual(row['Sample_Count'], 1000, "Expected 1,000 latency benchmark samples.")
        self.assertGreater(row['Mean_Latency_ms'], 0.0, "Mean latency must be positive.")
        self.assertLess(row['Mean_Latency_ms'], 10.0, "Mean latency abnormally high (>10ms).")
        self.assertLessEqual(row['Median_Latency_ms'], row['P95_Latency_ms'])


if __name__ == '__main__':
    unittest.main()
