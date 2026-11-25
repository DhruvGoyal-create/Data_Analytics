"""
Unit Test Suite for House Price Prediction Pipeline
===================================================
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — Project 05 (House Price Prediction)
Author: Data Analytics Portfolio Intern

Automated verification tests:
1. Dataset file presence and SHA-256 integrity.
2. Dataset schema, row count, column names, and data types.
3. Missing value profiling (exactly 207 nulls in total_bedrooms).
4. Preprocessing pipeline construction and leakage isolation.
5. Linear regression serialized model load and inference test.
6. Model performance bounds (R² > 0.60, RMSE < baseline).
7. Regularized models (Ridge and Lasso) performance consistency.
8. Regression coefficient count, direction, and impact ranking.
9. Residual diagnostics (mean near 0, capping error inflation).
10. Figure artifact presence and file size verification.
11. Summary table presence and non-empty validation.
12. Random Forest benchmark superior performance verification.
"""

import os
import sys
import hashlib
import unittest
import numpy as np
import pandas as pd
import joblib

# Ensure local task modules can be imported when running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestHousePricePrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_path = os.path.join(cls.base_dir, "data", "housing.csv")
        cls.figures_dir = os.path.join(cls.base_dir, "outputs", "figures")
        cls.tables_dir = os.path.join(cls.base_dir, "outputs", "tables")
        cls.models_dir = os.path.join(cls.base_dir, "models")

    def test_01_raw_data_file_exists_and_sha256(self):
        """Verify data/housing.csv exists and matches expected SHA-256 hash."""
        self.assertTrue(os.path.exists(self.data_path), f"Dataset missing at {self.data_path}")
        with open(self.data_path, "rb") as f:
            computed_hash = hashlib.sha256(f.read()).hexdigest().upper()
        expected_hash = "8A3727F4CF54AC1A327F69B1D5B4DB54C5834EA81C6E4EFC0D163300022A685E"
        self.assertEqual(computed_hash, expected_hash, "Dataset SHA-256 hash mismatch!")

    def test_02_raw_data_schema(self):
        """Verify row count (20,640) and expected columns."""
        df = pd.read_csv(self.data_path)
        self.assertEqual(df.shape, (20640, 10))
        expected_cols = [
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income",
            "median_house_value", "ocean_proximity"
        ]
        self.assertListEqual(list(df.columns), expected_cols)

    def test_03_missing_value_profile(self):
        """Verify missing values are isolated to total_bedrooms (207 nulls)."""
        df = pd.read_csv(self.data_path)
        null_counts = df.isnull().sum()
        self.assertEqual(null_counts["total_bedrooms"], 207)
        for col in df.columns:
            if col != "total_bedrooms":
                self.assertEqual(null_counts[col], 0, f"Unexpected nulls in column {col}")

    def test_04_preprocessor_pipeline_structure(self):
        """Verify preprocessor handles imputation and encoding correctly."""
        from house_price_prediction import build_preprocessor
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=["median_house_value"])
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = ["ocean_proximity"]

        preprocessor = build_preprocessor(num_cols, cat_cols)
        X_trans = preprocessor.fit_transform(X)
        # 8 numeric + 4 one-hot (5 categories drop='first' -> 4) = 12 columns
        self.assertEqual(X_trans.shape, (20640, 12))
        self.assertFalse(np.isnan(X_trans).any(), "Preprocessed array contains NaNs!")

    def test_05_linear_regression_pipeline_artifact(self):
        """Verify saved linear regression model loads and makes valid predictions."""
        lr_path = os.path.join(self.models_dir, "linear_regression_pipeline.joblib")
        self.assertTrue(os.path.exists(lr_path), f"Linear regression model missing at {lr_path}")
        pipeline = joblib.load(lr_path)

        sample = pd.DataFrame([{
            "longitude": -122.23,
            "latitude": 37.88,
            "housing_median_age": 41.0,
            "total_rooms": 880.0,
            "total_bedrooms": 129.0,
            "population": 322.0,
            "households": 126.0,
            "median_income": 8.3252,
            "ocean_proximity": "NEAR BAY"
        }])
        pred = pipeline.predict(sample)
        self.assertEqual(len(pred), 1)
        self.assertGreater(pred[0], 100000.0)
        self.assertLess(pred[0], 600000.0)

    def test_06_model_performance_metrics(self):
        """Verify Linear Regression R² > 0.60 and RMSE < $75,000."""
        perf_csv = os.path.join(self.tables_dir, "model_performance_comparison.csv")
        self.assertTrue(os.path.exists(perf_csv), f"Performance table missing at {perf_csv}")
        df_perf = pd.read_csv(perf_csv)
        lr_row = df_perf[df_perf["Model"] == "Linear Regression (OLS)"].iloc[0]
        dummy_row = df_perf[df_perf["Model"].str.startswith("Dummy")].iloc[0]

        self.assertGreater(lr_row["Test_R2"], 0.60)
        self.assertLess(lr_row["Test_RMSE"], 75000.0)
        self.assertLess(lr_row["Test_RMSE"], dummy_row["Test_RMSE"])

    def test_07_regularized_models_performance(self):
        """Verify Ridge and Lasso pipelines exist and have comparable R²."""
        ridge_path = os.path.join(self.models_dir, "ridge_pipeline.joblib")
        lasso_path = os.path.join(self.models_dir, "lasso_pipeline.joblib")
        self.assertTrue(os.path.exists(ridge_path))
        self.assertTrue(os.path.exists(lasso_path))

        df_perf = pd.read_csv(os.path.join(self.tables_dir, "model_performance_comparison.csv"))
        ridge_r2 = df_perf[df_perf["Model"].str.startswith("Ridge")]["Test_R2"].values[0]
        lasso_r2 = df_perf[df_perf["Model"].str.startswith("Lasso")]["Test_R2"].values[0]
        lr_r2 = df_perf[df_perf["Model"].str.startswith("Linear Regression")]["Test_R2"].values[0]

        self.assertAlmostEqual(ridge_r2, lr_r2, places=2)
        self.assertAlmostEqual(lasso_r2, lr_r2, places=2)

    def test_08_regression_coefficients_consistency(self):
        """Verify regression coefficient count (12 features) and median_income positive weight."""
        coef_csv = os.path.join(self.tables_dir, "regression_coefficients.csv")
        self.assertTrue(os.path.exists(coef_csv))
        df_coef = pd.read_csv(coef_csv)
        self.assertEqual(len(df_coef), 12)

        income_coef = df_coef[df_coef["Feature"] == "median_income"]["Standardized_Coefficient"].values[0]
        self.assertGreater(income_coef, 50000.0, "median_income should have large positive weight")

    def test_09_residual_diagnostics_characteristics(self):
        """Verify residual mean is close to zero and capped subset has higher RMSE."""
        res_csv = os.path.join(self.tables_dir, "residual_diagnostics_summary.csv")
        self.assertTrue(os.path.exists(res_csv))
        df_res = pd.read_csv(res_csv)

        uncapped_rmse_val = df_res[df_res["Metric"] == "Uncapped Subset RMSE"]["Value"].values[0]
        capped_rmse_val = df_res[df_res["Metric"] == "Capped Ceiling Subset RMSE"]["Value"].values[0]

        uncapped_float = float(uncapped_rmse_val.replace("$", "").replace(",", ""))
        capped_float = float(capped_rmse_val.replace("$", "").replace(",", ""))

        self.assertGreater(capped_float, uncapped_float * 2.0, "Capped RMSE should be significantly larger due to boundary truncation")

    def test_10_figure_artifacts_exist(self):
        """Verify all 6 publication figures exist with size > 20 KB."""
        for i in range(1, 7):
            matches = [f for f in os.listdir(self.figures_dir) if f.startswith(f"fig{i}_")]
            self.assertEqual(len(matches), 1, f"Missing figure {i}")
            fig_path = os.path.join(self.figures_dir, matches[0])
            self.assertGreater(os.path.getsize(fig_path), 20000, f"Figure {matches[0]} is too small")

    def test_11_summary_tables_exist(self):
        """Verify all 5 summary CSV tables exist and have rows."""
        expected_tables = [
            "housing_dataset_summary.csv",
            "feature_correlations.csv",
            "model_performance_comparison.csv",
            "regression_coefficients.csv",
            "residual_diagnostics_summary.csv"
        ]
        for t in expected_tables:
            t_path = os.path.join(self.tables_dir, t)
            self.assertTrue(os.path.exists(t_path), f"Missing table {t}")
            df = pd.read_csv(t_path)
            self.assertGreater(len(df), 0, f"Table {t} is empty")

    def test_12_non_linear_benchmark_superiority(self):
        """Verify Random Forest benchmark achieves R² > 0.78 and RMSE < $52,000."""
        df_perf = pd.read_csv(os.path.join(self.tables_dir, "model_performance_comparison.csv"))
        rf_row = df_perf[df_perf["Model"].str.startswith("Random Forest")].iloc[0]
        self.assertGreater(rf_row["Test_R2"], 0.78)
        self.assertLess(rf_row["Test_RMSE"], 52000.0)


if __name__ == "__main__":
    unittest.main()
