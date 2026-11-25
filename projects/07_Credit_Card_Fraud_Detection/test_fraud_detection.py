"""
test_fraud_detection.py
Automated Unit Test Suite for Project 07 (Credit Card Fraud Detection) — Fraud Detection
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO (Business-Analytics-Portfolio)

Validates data integrity, schema consistency, class imbalance properties,
leakage-free preprocessing, model artifacts, performance metrics,
financial audit calculations, and visual/tabular exports.
"""

import os
import hashlib
import unittest
import joblib
import pandas as pd
import numpy as np


class TestFraudDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_path = os.path.join(cls.base_dir, 'data', 'creditcard.csv')
        cls.fig_dir = os.path.join(cls.base_dir, 'outputs', 'figures')
        cls.tab_dir = os.path.join(cls.base_dir, 'outputs', 'tables')
        cls.model_dir = os.path.join(cls.base_dir, 'models')
        
        cls.expected_sha256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
        
        # Load sample or full data for tests
        if os.path.exists(cls.data_path):
            cls.df = pd.read_csv(cls.data_path)
        else:
            cls.df = None

    def test_01_dataset_file_exists_and_sha256(self):
        """Verify creditcard.csv exists and matches authoritative SHA-256 hash."""
        self.assertIsNotNone(self.df, "Credit card fraud dataset missing from data directory.")
        self.assertTrue(os.path.exists(self.data_path))
        
        hasher = hashlib.sha256()
        with open(self.data_path, 'rb') as f:
            while chunk := f.read(1024 * 1024):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        self.assertEqual(actual_hash, self.expected_sha256, f"SHA-256 hash mismatch: {actual_hash}")

    def test_02_dataset_schema(self):
        """Verify row count (284,807) and expected columns (Time, V1..V28, Amount, Class)."""
        self.assertEqual(self.df.shape, (284807, 31))
        expected_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        self.assertEqual(list(self.df.columns), expected_cols)

    def test_03_zero_missing_values(self):
        """Verify dataset contains zero null/missing values across all cells."""
        self.assertEqual(self.df.isnull().sum().sum(), 0, "Null values found in credit card dataset.")

    def test_04_class_imbalance_ratio(self):
        """Verify exact class counts: 284,315 legitimate (0) and 492 fraud (1) = 0.1727%."""
        class_counts = self.df['Class'].value_counts()
        self.assertEqual(int(class_counts[0]), 284315)
        self.assertEqual(int(class_counts[1]), 492)
        fraud_pct = (class_counts[1] / len(self.df)) * 100
        self.assertAlmostEqual(fraud_pct, 0.172749, places=4)

    def test_05_stratified_train_test_split(self):
        """Verify stratified 80/20 partition yields 56,962 test records with 98 fraud cases."""
        tab_perf = os.path.join(self.tab_dir, 'model_performance_comparison.csv')
        self.assertTrue(os.path.exists(tab_perf), "Performance table missing.")
        df_perf = pd.read_csv(tab_perf)
        dummy_row = df_perf[df_perf['Model'].str.contains('Dummy')].iloc[0]
        test_total = int(dummy_row['True_Negatives'] + dummy_row['False_Negatives'])
        test_fraud = int(dummy_row['False_Negatives'])
        self.assertEqual(test_total, 56962)
        self.assertEqual(test_fraud, 98)

    def test_06_model_artifacts_saved(self):
        """Verify serialized pipelines exist in models/ directory with valid file sizes."""
        models = [
            'fraud_detection_random_forest.joblib',
            'fraud_detection_logistic_regression.joblib',
            'fraud_detection_smote_lr.joblib'
        ]
        for m_name in models:
            m_path = os.path.join(self.model_dir, m_name)
            self.assertTrue(os.path.exists(m_path), f"Missing model artifact: {m_name}")
            self.assertGreater(os.path.getsize(m_path), 500, f"Model artifact too small: {m_name}")

    def test_07_model_prediction_shape_and_classes(self):
        """Verify loaded pipelines generate valid binary predictions on unseen feature vectors."""
        rf_path = os.path.join(self.model_dir, 'fraud_detection_random_forest.joblib')
        pipe = joblib.load(rf_path)
        sample_X = self.df.drop(columns=['Class']).iloc[:5]
        preds = pipe.predict(sample_X)
        self.assertEqual(len(preds), 5)
        self.assertTrue(all(p in [0, 1] for p in preds))

    def test_08_performance_table_metrics(self):
        """Verify model performance comparison table contains 5 models and Random Forest achieves PR-AUC > 0.75."""
        tab_perf = os.path.join(self.tab_dir, 'model_performance_comparison.csv')
        df_perf = pd.read_csv(tab_perf)
        self.assertEqual(len(df_perf), 5, "Expected 5 models in performance table.")
        
        rf_row = df_perf[df_perf['Model'].str.contains('Random Forest')].iloc[0]
        self.assertGreater(rf_row['PR_AUC'], 0.75, "Random Forest PR-AUC below target threshold.")
        self.assertGreater(rf_row['F1_Score'], 0.75, "Random Forest F1-Score below target threshold.")
        self.assertGreater(rf_row['Precision'], 0.75, "Random Forest Precision below target threshold.")
        self.assertGreater(rf_row['Recall'], 0.75, "Random Forest Recall below target threshold.")

    def test_09_financial_prevention_audit_metrics(self):
        """Verify financial table records $10,644.93 test fraud dollars and non-zero recovery."""
        tab_fin = os.path.join(self.tab_dir, 'financial_fraud_prevention_audit.csv')
        self.assertTrue(os.path.exists(tab_fin), "Financial audit table missing.")
        df_fin = pd.read_csv(tab_fin)
        self.assertEqual(len(df_fin), 5)
        
        total_dollars = df_fin['Total_Test_Fraud_Dollars'].iloc[0]
        self.assertAlmostEqual(total_dollars, 10644.93, places=2)
        
        # Verify Balanced LR recovered >90% of fraud dollars
        bal_lr = df_fin[df_fin['Model'].str.contains('Balanced Weights') & df_fin['Model'].str.contains('Logistic')].iloc[0]
        self.assertGreater(bal_lr['Fraud_Dollars_Recovered_Pct'], 90.0)

    def test_10_figures_generated(self):
        """Verify all 6 publication-ready figures exist and exceed 20 KB."""
        expected_figs = [
            'fig1_class_imbalance_and_amounts.png',
            'fig2_time_of_day_analysis.png',
            'fig3_correlation_heatmap.png',
            'fig4_confusion_matrices.png',
            'fig5_roc_and_pr_curves.png',
            'fig6_feature_importance.png'
        ]
        for f_name in expected_figs:
            f_path = os.path.join(self.fig_dir, f_name)
            self.assertTrue(os.path.exists(f_path), f"Missing figure artifact: {f_name}")
            self.assertGreater(os.path.getsize(f_path), 20_000, f"Figure artifact too small: {f_name}")

    def test_11_tables_generated(self):
        """Verify all 7 summary CSV tables exist and have valid rows."""
        expected_tabs = [
            'fraud_dataset_summary.csv',
            'class_imbalance_and_amount_stats.csv',
            'time_of_day_fraud_rates.csv',
            'model_performance_comparison.csv',
            'financial_fraud_prevention_audit.csv',
            'top_predictive_features.csv',
            'scalability_throughput_analysis.csv'
        ]
        for t_name in expected_tabs:
            t_path = os.path.join(self.tab_dir, t_name)
            self.assertTrue(os.path.exists(t_path), f"Missing summary table: {t_name}")
            df_t = pd.read_csv(t_path)
            self.assertGreater(len(df_t), 0, f"Table {t_name} is empty.")

    def test_12_time_of_day_peak_fraud_hour(self):
        """Verify that early morning hour 2 (2:00 AM) exhibits elevated fraud prevalence (>1.0%)."""
        tab_time = os.path.join(self.tab_dir, 'time_of_day_fraud_rates.csv')
        df_time = pd.read_csv(tab_time)
        hour_2 = df_time[df_time['Hour'] == 2].iloc[0]
        self.assertGreater(hour_2['fraud_rate_pct'], 1.0, "Hour 2 fraud surge not detected.")


if __name__ == '__main__':
    unittest.main()
