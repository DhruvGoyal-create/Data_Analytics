"""
test_cleaning_pipeline.py
Comprehensive Unit Test Suite for Project 03 (Data Cleaning Pipeline) - Cleaning Data
Quality Gate Verification:
- Integer reconstruction tolerance, fractional rejection, zero denominators, missing inputs
- Date validation: ISO timestamps, NaT preservation, exclusion in time-series
- Non-tautological accounting verification (directly observed vs derived)
- Immutability of raw source data and full artifact validation
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clean_cafe_sales import reconstruct_quantity_vectorized, TOLERANCE

class TestCleaningPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.raw_path = os.path.join(cls.base_dir, 'data', 'raw_dirty_cafe_sales.csv')
        cls.clean_path = os.path.join(cls.base_dir, 'data', 'cleaned_cafe_sales.csv')
        cls.fig_dir = os.path.join(cls.base_dir, 'outputs', 'figures')
        cls.tab_dir = os.path.join(cls.base_dir, 'outputs', 'tables')
        
        cls.df_raw = pd.read_csv(cls.raw_path)
        cls.df_clean = pd.read_csv(cls.clean_path)
        cls.df_unrec = pd.read_csv(os.path.join(cls.tab_dir, 'unrecoverable_transactions_audit.csv'))

    def test_01_raw_dataset_immutability(self):
        """Verify raw dirty dataset is preserved with exactly 10,000 rows and 8 columns."""
        self.assertTrue(os.path.exists(self.raw_path), "Raw dataset file missing.")
        self.assertEqual(self.df_raw.shape, (10000, 8), "Raw dataset dimensions were modified!")
        self.assertEqual(len(self.df_raw['Transaction ID'].unique()), 10000, "Raw Transaction IDs corrupted.")

    def test_02_row_reconciliation(self):
        """Verify row counts reconcile: 10,000 raw = 9,974 clean + 26 purged."""
        self.assertEqual(len(self.df_clean), 9974, "Clean dataset must retain exactly 9,974 records.")
        self.assertEqual(len(self.df_unrec), 26, "Purge audit log must record exactly 26 unrecoverable rows.")
        self.assertEqual(len(self.df_clean) + len(self.df_unrec), len(self.df_raw), "Row reconciliation mismatch.")

    def test_03_quantity_reconstruction_unit_logic(self):
        """Audit quantity reconstruction: valid integers, fractional rejection, zero denominators, missing inputs."""
        # Case A: Valid exact integer
        ts = pd.Series([10.0, 15.0, 4.0])
        p = pd.Series([2.0, 3.0, 4.0])
        res = reconstruct_quantity_vectorized(ts, p)
        np.testing.assert_array_equal(res.dropna().values, [5.0, 5.0, 1.0])
        
        # Case B: Floating-point precision within tolerance (< 1e-5)
        ts_drift = pd.Series([6.000000001, 7.999999999])
        p_drift = pd.Series([2.0, 4.0])
        res_drift = reconstruct_quantity_vectorized(ts_drift, p_drift)
        np.testing.assert_array_equal(res_drift.values, [3.0, 2.0])
        
        # Case C: Fractional result must be rejected as unresolved (NaN)
        ts_frac = pd.Series([5.0, 7.0, 11.0])
        p_frac = pd.Series([3.0, 2.0, 4.0])
        res_frac = reconstruct_quantity_vectorized(ts_frac, p_frac)
        self.assertTrue(res_frac.isnull().all(), "Fractional quantities were erroneously accepted!")
        
        # Case D: Zero denominator (Price = 0) must be rejected
        ts_zero = pd.Series([10.0])
        p_zero = pd.Series([0.0])
        res_zero = reconstruct_quantity_vectorized(ts_zero, p_zero)
        self.assertTrue(res_zero.isnull().all(), "Zero denominator was not rejected!")
        
        # Case E: Non-positive Total Spent or Price
        ts_neg = pd.Series([-10.0, 10.0])
        p_neg = pd.Series([2.0, -2.0])
        res_neg = reconstruct_quantity_vectorized(ts_neg, p_neg)
        self.assertTrue(res_neg.isnull().all(), "Negative financial values were not rejected!")
        
        # Case F: Missing inputs
        ts_nan = pd.Series([np.nan, 10.0])
        p_nan = pd.Series([2.0, np.nan])
        res_nan = reconstruct_quantity_vectorized(ts_nan, p_nan)
        self.assertTrue(res_nan.isnull().all(), "Missing inputs were not rejected!")

    def test_04_clean_dataset_quantities_positive_integers(self):
        """Verify all quantities in clean dataset are strictly positive integers."""
        self.assertTrue((self.df_clean['Quantity'] > 0).all(), "Non-positive quantities found.")
        self.assertTrue(pd.api.types.is_integer_dtype(self.df_clean['Quantity']), "Quantity is not integer dtype.")
        # Check values are in valid domain [1, 2, 3, 4, 5]
        self.assertTrue(set(self.df_clean['Quantity'].unique()).issubset({1, 2, 3, 4, 5}))

    def test_05_date_handling_and_preservation(self):
        """Verify dates are preserved as missing (460 nulls), parsed correctly when present, and not filled with median."""
        clean_dates = pd.to_datetime(self.df_clean['Transaction Date'], errors='coerce')
        # Exactly 460 dates were corrupted in raw; verify all 460 remain missing (NaT / NaN in CSV)
        self.assertEqual(clean_dates.isnull().sum(), 460, "Corrupted dates were improperly imputed!")
        # Valid dates must be in 2023 calendar year
        valid_dates = clean_dates.dropna()
        self.assertEqual(len(valid_dates), 9514, "Unexpected count of valid dates.")
        self.assertEqual(valid_dates.dt.year.unique().tolist(), [2023], "Dates exist outside 2023 operating year.")

    def test_06_non_tautological_accounting_identity(self):
        """Verify accounting identity: independent check on directly observed rows vs reconstructed rows."""
        corrupt_tokens = ['UNKNOWN', 'ERROR', 'nan', 'NaN', 'None', '?']
        q_raw_ok = ~self.df_raw['Quantity'].isin(corrupt_tokens) & self.df_raw['Quantity'].notnull()
        p_raw_ok = ~self.df_raw['Price Per Unit'].isin(corrupt_tokens) & self.df_raw['Price Per Unit'].notnull()
        ts_raw_ok = ~self.df_raw['Total Spent'].isin(corrupt_tokens) & self.df_raw['Total Spent'].notnull()
        
        all_obs_mask = q_raw_ok & p_raw_ok & ts_raw_ok
        self.assertEqual(all_obs_mask.sum(), 8544, "Expected 8,544 rows with all three financials directly observed.")
        
        # Test independent match on directly observed raw records
        df_obs = self.df_raw[all_obs_mask]
        q_obs = pd.to_numeric(df_obs['Quantity'])
        p_obs = pd.to_numeric(df_obs['Price Per Unit'])
        ts_obs = pd.to_numeric(df_obs['Total Spent'])
        
        obs_match = np.isclose(ts_obs, q_obs * p_obs, atol=TOLERANCE)
        self.assertEqual(obs_match.mean(), 1.0, "Directly observed records fail accounting identity!")
        
        # Test full clean dataset internal consistency
        clean_match = np.isclose(self.df_clean['Total Spent'], self.df_clean['Quantity'] * self.df_clean['Price Per Unit'], atol=TOLERANCE)
        self.assertEqual(clean_match.mean(), 1.0, "Clean dataset contains arithmetic calculation errors.")

    def test_07_no_unresolved_placeholder_tokens(self):
        """Verify no 'UNKNOWN', 'ERROR', or 'nan' string tokens remain in non-categorical fields."""
        forbidden = ['ERROR', 'nan', 'NaN', 'None', '?']
        for col in ['Transaction ID', 'Quantity', 'Price Per Unit', 'Total Spent']:
            matches = self.df_clean[col].astype(str).str.strip().isin(forbidden)
            self.assertEqual(matches.sum(), 0, f"Column '{col}' contains forbidden error tokens.")
            
        # For categoricals, 'Unknown' is an explicit capitalized label, never 'UNKNOWN' or 'ERROR'
        for col in ['Payment Method', 'Location']:
            self.assertEqual((self.df_clean[col] == 'ERROR').sum(), 0)
            self.assertEqual((self.df_clean[col] == 'UNKNOWN').sum(), 0)

    def test_08_unique_transaction_ids(self):
        """Verify all Transaction IDs in clean dataset are unique."""
        self.assertEqual(self.df_clean['Transaction ID'].nunique(), len(self.df_clean), "Duplicate Transaction IDs found.")

    def test_09_artifacts_generated(self):
        """Verify all required figures and tables exist on disk and are non-empty."""
        expected_figs = ['fig1_data_quality_anomalies.png', 'fig2_outlier_detection.png', 'fig3_cleaned_sales_distribution.png']
        for fig in expected_figs:
            fig_path = os.path.join(self.fig_dir, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig}")
            self.assertGreater(os.path.getsize(fig_path), 10000, f"Figure truncated: {fig}")
            
        expected_tabs = [
            'before_after_metrics.csv',
            'raw_data_quality_audit.csv',
            'canonical_menu_prices.csv',
            'outlier_analysis_summary.csv',
            'unrecoverable_transactions_audit.csv',
            'reconstruction_method_classification.csv'
        ]
        for tab in expected_tabs:
            tab_path = os.path.join(self.tab_dir, tab)
            self.assertTrue(os.path.exists(tab_path), f"Missing table: {tab}")
            self.assertGreater(os.path.getsize(tab_path), 50, f"Table truncated: {tab}")

if __name__ == '__main__':
    unittest.main()
