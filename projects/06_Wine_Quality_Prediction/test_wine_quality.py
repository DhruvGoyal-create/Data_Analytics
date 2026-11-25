"""
Unit Tests for Wine Quality Prediction Pipeline
=================================================
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — Project 06 (Wine Quality Prediction)
Author: Dhruv Goyal
Date: October 2026
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib

# Ensure local task modules can be imported when running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wine_quality_prediction import (
    load_and_inspect_dataset,
    create_binary_target,
    build_models,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    BINARY_TARGET
)


class TestWineQualityPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_path = os.path.join(cls.base_dir, 'data', 'winequality-red.csv')
        cls.tables_dir = os.path.join(cls.base_dir, 'outputs', 'tables')
        cls.figures_dir = os.path.join(cls.base_dir, 'outputs', 'figures')
        cls.models_dir = os.path.join(cls.base_dir, 'models')

        cls.df_raw = load_and_inspect_dataset(cls.data_path)
        cls.df_binary = create_binary_target(cls.df_raw, quality_threshold=7)

    def test_01_dataset_file_exists(self):
        """Verify the raw UCI Red Wine Quality CSV exists and has content."""
        self.assertTrue(os.path.exists(self.data_path), "winequality-red.csv not found")
        self.assertGreater(os.path.getsize(self.data_path), 50000, "Dataset file appears truncated")

    def test_02_dataset_schema_and_columns(self):
        """Verify dataset row count and all required physicochemical columns exist."""
        self.assertEqual(len(self.df_raw), 1599, "Expected 1,599 observations in Red Wine dataset")
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.df_raw.columns, f"Feature column '{col}' missing")
        self.assertIn(TARGET_COLUMN, self.df_raw.columns, "Target column 'quality' missing")

    def test_03_no_missing_values(self):
        """Verify that the benchmark dataset contains exactly zero missing/null values."""
        null_count = self.df_raw.isnull().sum().sum()
        self.assertEqual(null_count, 0, f"Expected 0 nulls, found {null_count}")

    def test_04_binary_target_creation(self):
        """Verify binary target formulation: quality >= 7 is 1, else 0."""
        self.assertIn(BINARY_TARGET, self.df_binary.columns)
        unique_vals = set(self.df_binary[BINARY_TARGET].unique())
        self.assertEqual(unique_vals, {0, 1}, "Binary target must contain only 0 and 1")

        expected_positives = (self.df_raw[TARGET_COLUMN] >= 7).sum()
        actual_positives = (self.df_binary[BINARY_TARGET] == 1).sum()
        self.assertEqual(actual_positives, expected_positives)
        self.assertEqual(actual_positives, 217, "Expected exactly 217 positive cases (quality >= 7)")

    def test_05_stratified_split_preserves_ratio(self):
        """Verify stratified split preserves the positive class proportion in train and test splits."""
        y = self.df_binary[BINARY_TARGET]
        from sklearn.model_selection import train_test_split
        _, _, y_train, y_test = train_test_split(
            self.df_binary[FEATURE_COLUMNS], y, test_size=0.20, random_state=42, stratify=y
        )
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        overall_ratio = y.mean()

        self.assertAlmostEqual(train_ratio, overall_ratio, places=2)
        self.assertAlmostEqual(test_ratio, overall_ratio, places=2)
        self.assertEqual(len(y_test), 320, "Test set must have 320 observations (20% of 1599)")
        self.assertEqual(y_test.sum(), 43, "Test set must have exactly 43 positive cases")

    def test_06_leakage_free_pipeline_scaling(self):
        """Verify that models use Pipeline architecture encapsulating StandardScaler."""
        models = build_models(random_state=42)
        for name in ['Random Forest', 'SGD Classifier', 'Support Vector (SVC)']:
            self.assertIn(name, models)
            pipe = models[name]
            self.assertEqual(pipe.steps[0][0], 'scaler', f"{name} pipeline must begin with 'scaler'")

    def test_07_model_artifacts_saved(self):
        """Verify that all serialized model pipelines exist in models directory."""
        expected_artifacts = [
            'wine_quality_random_forest_pipeline.joblib',
            'wine_quality_sgd_classifier_pipeline.joblib',
            'wine_quality_support_vector_svc_pipeline.joblib'
        ]
        for art in expected_artifacts:
            path = os.path.join(self.models_dir, art)
            self.assertTrue(os.path.exists(path), f"Artifact missing: {art}")
            self.assertGreater(os.path.getsize(path), 1000, f"Artifact appears empty: {art}")

    def test_08_model_prediction_shape_and_range(self):
        """Verify that serialized models can be loaded and generate valid binary predictions."""
        rf_path = os.path.join(self.models_dir, 'wine_quality_random_forest_pipeline.joblib')
        pipeline = joblib.load(rf_path)

        sample_X = self.df_raw[FEATURE_COLUMNS].head(10)
        preds = pipeline.predict(sample_X)
        probs = pipeline.predict_proba(sample_X)[:, 1]

        self.assertEqual(len(preds), 10)
        self.assertTrue(np.all(np.isin(preds, [0, 1])))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_09_model_performance_table_consistency(self):
        """Verify model performance comparison table has all models and Random Forest is champion."""
        comp_path = os.path.join(self.tables_dir, 'model_performance_comparison.csv')
        self.assertTrue(os.path.exists(comp_path))
        df_comp = pd.read_csv(comp_path)

        self.assertGreaterEqual(len(df_comp), 4, "Expected at least 4 models in comparison table")
        self.assertIn('Model', df_comp.columns)
        self.assertIn('Accuracy', df_comp.columns)
        self.assertIn('ROC_AUC', df_comp.columns)

        dummy_row = df_comp[df_comp['Model'].str.contains('Dummy')].iloc[0]
        self.assertAlmostEqual(dummy_row['Accuracy'], 0.8656, places=3)
        self.assertTrue(pd.isna(dummy_row['ROC_AUC']), "Dummy ROC-AUC must be NaN / undefined")

        rf_row = df_comp[df_comp['Model'] == 'Random Forest'].iloc[0]
        self.assertGreater(rf_row['Accuracy'], dummy_row['Accuracy'])
        self.assertGreater(rf_row['ROC_AUC'], 0.90, "Random Forest should achieve > 0.90 ROC-AUC")

    def test_10_feature_importance_validity(self):
        """Verify feature importance table contains all 11 features and alcohol is top predictor."""
        imp_path = os.path.join(self.tables_dir, 'random_forest_feature_importance.csv')
        self.assertTrue(os.path.exists(imp_path))
        df_imp = pd.read_csv(imp_path)

        self.assertEqual(len(df_imp), 11, "Must contain exactly 11 features")
        self.assertAlmostEqual(df_imp['Gini_Importance'].sum(), 1.0, places=2)
        top_feature = df_imp.iloc[0]['Feature']
        self.assertEqual(top_feature, 'alcohol', "Alcohol must be the highest Gini importance feature")

    def test_11_figures_generated(self):
        """Verify all 6 evaluation and EDA figures are generated with valid file sizes."""
        expected_figs = [
            'fig1_wine_distributions.png',
            'fig2_quality_distribution.png',
            'fig3_correlation_heatmap.png',
            'fig4_confusion_matrices.png',
            'fig5_roc_pr_curves.png',
            'fig6_feature_importance.png'
        ]
        for fig_name in expected_figs:
            path = os.path.join(self.figures_dir, fig_name)
            self.assertTrue(os.path.exists(path), f"Figure missing: {fig_name}")
            self.assertGreater(os.path.getsize(path), 10000, f"Figure truncated: {fig_name}")

    def test_12_duplicate_and_leakage_audit_table(self):
        audit_path = os.path.join(self.tables_dir, 'duplicate_leakage_audit.csv')
        self.assertTrue(os.path.exists(audit_path), "duplicate_leakage_audit.csv missing")
        df_audit = pd.read_csv(audit_path)
        col_name = 'Metric' if 'Metric' in df_audit.columns else 'Audit_Metric'
        self.assertIn(col_name, df_audit.columns)
        self.assertIn('Value', df_audit.columns)

        metrics_map = dict(zip(df_audit[col_name], df_audit['Value'].astype(str)))
        self.assertEqual(metrics_map['Total Raw Observations'], '1599')
        self.assertEqual(metrics_map['Exact Duplicate Rows'], '240')
        self.assertEqual(metrics_map['Test Rows Identical to a Training Row (Boundary Leakage)'], '77')


if __name__ == '__main__':
    unittest.main()
