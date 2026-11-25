"""
test_play_store_analysis.py
===========================
Automated Unit Test Suite for PRJ-08: Google Play Store Market Analytics
Author: Dhruv Goyal (https://github.com/DhruvGoyal-create)

Validates data hygiene, schema integrity, algebraic transformations,
summary statistics, and visual artifact generation.
"""

import os
import unittest
import pandas as pd
import numpy as np

class TestGooglePlayStoreAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_dir = os.path.join(cls.base_dir, 'data')
        cls.out_dir = os.path.join(cls.base_dir, 'outputs')
        cls.fig_dir = os.path.join(cls.out_dir, 'figures')
        cls.tab_dir = os.path.join(cls.out_dir, 'tables')
        
        cls.apps_csv = os.path.join(cls.data_dir, 'googleplaystore.csv')
        cls.rev_csv = os.path.join(cls.data_dir, 'googleplaystore_user_reviews.csv')

    def test_01_raw_data_files_exist(self):
        """Test 1: Verify presence of raw input datasets."""
        self.assertTrue(os.path.exists(self.apps_csv), "googleplaystore.csv missing")
        self.assertTrue(os.path.exists(self.rev_csv), "googleplaystore_user_reviews.csv missing")
        self.assertGreater(os.path.getsize(self.apps_csv), 1_000_000)
        self.assertGreater(os.path.getsize(self.rev_csv), 5_000_000)

    def test_02_corrupt_row_10472_detected_and_handled(self):
        """Test 2: Verify detection and removal of shifted row with Rating > 5.0."""
        df_raw = pd.read_csv(self.apps_csv)
        corrupt = df_raw[df_raw['Rating'] > 5.0]
        self.assertEqual(len(corrupt), 1, "Expected exactly 1 corrupt row with Rating > 5.0 (Row 10472)")
        self.assertEqual(corrupt.iloc[0]['App'], 'Life Made WI-Fi Touchscreen Photo Frame')

    def test_03_clean_apps_record_count(self):
        """Test 3: Verify cleaned unique apps count equals 9,659."""
        from google_play_store_analysis import load_and_clean_data
        df_apps_raw, df_apps_clean, _, _, _ = load_and_clean_data(self.base_dir)
        self.assertEqual(len(df_apps_raw), 10841)
        self.assertEqual(len(df_apps_clean), 9659)
        self.assertEqual(df_apps_clean['App'].nunique(), 9659)

    def test_04_installs_and_price_parsed_as_numeric(self):
        """Test 4: Verify Installs_Numeric and Price_Numeric are parsed as non-negative numeric types."""
        from google_play_store_analysis import load_and_clean_data
        _, df_apps_clean, _, _, _ = load_and_clean_data(self.base_dir)
        self.assertTrue(pd.api.types.is_integer_dtype(df_apps_clean['Installs_Numeric']))
        self.assertTrue(pd.api.types.is_float_dtype(df_apps_clean['Price_Numeric']))
        self.assertTrue((df_apps_clean['Installs_Numeric'] >= 0).all())
        self.assertTrue((df_apps_clean['Price_Numeric'] >= 0.0).all())

    def test_05_size_mb_parsed_correctly(self):
        """Test 5: Verify Size_MB conversion parses M and k correctly to megabytes."""
        from google_play_store_analysis import parse_size_to_mb
        self.assertAlmostEqual(parse_size_to_mb('19M'), 19.0)
        self.assertAlmostEqual(parse_size_to_mb('1024k'), 1.0)
        self.assertTrue(np.isnan(parse_size_to_mb('Varies with device')))

    def test_06_clean_reviews_count(self):
        """Test 6: Verify cleaned reviews with non-null text equals 37,427."""
        from google_play_store_analysis import load_and_clean_data
        _, _, df_rev_raw, df_rev_clean, _ = load_and_clean_data(self.base_dir)
        self.assertEqual(len(df_rev_raw), 64295)
        self.assertEqual(len(df_rev_clean), 37427)
        self.assertTrue(df_rev_clean['Translated_Review'].notna().all())

    def test_07_sentiment_polarity_and_subjectivity_ranges(self):
        """Test 7: Verify sentiment polarity is in [-1, 1] and subjectivity is in [0, 1]."""
        from google_play_store_analysis import load_and_clean_data
        _, _, _, df_rev_clean, _ = load_and_clean_data(self.base_dir)
        self.assertTrue((df_rev_clean['Sentiment_Polarity'] >= -1.0).all())
        self.assertTrue((df_rev_clean['Sentiment_Polarity'] <= 1.0).all())
        self.assertTrue((df_rev_clean['Sentiment_Subjectivity'] >= 0.0).all())
        self.assertTrue((df_rev_clean['Sentiment_Subjectivity'] <= 1.0).all())

    def test_08_merged_reviews_join_integrity(self):
        """Test 8: Verify merged reviews dataset preserves all expected 33 categories."""
        from google_play_store_analysis import load_and_clean_data
        _, _, _, _, df_merged = load_and_clean_data(self.base_dir)
        self.assertEqual(len(df_merged), 35929)
        self.assertEqual(df_merged['Category'].nunique(), 33)

    def test_09_summary_tables_generated(self):
        """Test 9: Verify all 7 summary tables exist and contain records."""
        expected_tables = [
            'dataset_profile_summary.csv',
            'category_saturation_analysis.csv',
            'category_ratings_summary.csv',
            'size_install_correlation_stats.csv',
            'category_revenue_estimates.csv',
            'sentiment_classification_summary.csv',
            'category_sentiment_profiles.csv'
        ]
        for tbl in expected_tables:
            path = os.path.join(self.tab_dir, tbl)
            self.assertTrue(os.path.exists(path), f"Table {tbl} missing")
            df = pd.read_csv(path)
            self.assertGreater(len(df), 0, f"Table {tbl} is empty")

    def test_10_visual_artifacts_and_interactive_html(self):
        """Test 10: Verify all 6 publication figures and interactive HTML exist."""
        expected_figs = [
            'fig1_category_distribution.png',
            'fig2_ratings_distribution_by_category.png',
            'fig3_size_vs_installs_correlation.png',
            'fig4_pricing_and_revenue_analysis.png',
            'fig5_sentiment_distribution.png',
            'fig6_category_sentiment_breakdown.png',
            'interactive_playstore_viz.html'
        ]
        for fig in expected_figs:
            path = os.path.join(self.fig_dir, fig)
            self.assertTrue(os.path.exists(path), f"Figure {fig} missing")
            self.assertGreater(os.path.getsize(path), 10_000, f"Figure {fig} suspiciously small")

if __name__ == '__main__':
    unittest.main()
