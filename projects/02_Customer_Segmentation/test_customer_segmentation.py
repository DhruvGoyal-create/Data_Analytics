"""
test_customer_segmentation.py
Automated Unit Test Suite for Project 02 (Customer Segmentation) - Customer Segmentation
Validates RFM engineering, K-Means clustering, persona assignments, and artifact generation.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

class TestCustomerSegmentation(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.raw_path = os.path.join(cls.base_dir, 'data', 'raw_online_retail.csv')
        cls.seg_path = os.path.join(cls.base_dir, 'data', 'rfm_customer_segments.csv')
        cls.fig_dir = os.path.join(cls.base_dir, 'outputs', 'figures')
        cls.tab_dir = os.path.join(cls.base_dir, 'outputs', 'tables')
        
        cls.df_seg = pd.read_csv(cls.seg_path)
        cls.df_eval = pd.read_csv(os.path.join(cls.tab_dir, 'cluster_evaluation_metrics.csv'))
        cls.df_profiles = pd.read_csv(os.path.join(cls.tab_dir, 'cluster_persona_profiles.csv'))

    def test_01_datasets_exist(self):
        """Verify raw and segmented datasets exist on disk and exceed minimum size."""
        self.assertTrue(os.path.exists(self.raw_path), "Raw Online Retail dataset missing.")
        self.assertTrue(os.path.exists(self.seg_path), "Segmented RFM dataset missing.")
        self.assertGreater(os.path.getsize(self.seg_path), 50000, "Segmented dataset appears truncated.")

    def test_02_rfm_customer_count(self):
        """Verify exactly 4,338 unique customers are segmented."""
        self.assertEqual(len(self.df_seg), 4338, f"Expected 4,338 customers, found {len(self.df_seg)}.")
        self.assertEqual(self.df_seg['CustomerID'].nunique(), 4338, "Duplicate Customer IDs found in segmented dataset.")

    def test_03_rfm_schema_and_zero_nulls(self):
        """Verify required columns exist and contain zero null values."""
        required_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Avg_Order_Value', 'Cluster', 'Persona']
        for col in required_cols:
            self.assertIn(col, self.df_seg.columns, f"Missing column: {col}")
            self.assertEqual(self.df_seg[col].isnull().sum(), 0, f"Column '{col}' contains unexpected nulls.")

    def test_04_rfm_value_bounds(self):
        """Verify R, F, and M values adhere to positive physical boundaries."""
        self.assertTrue((self.df_seg['Recency'] > 0).all(), "Found non-positive Recency values.")
        self.assertTrue((self.df_seg['Frequency'] >= 1).all(), "Found Frequency < 1.")
        self.assertTrue((self.df_seg['Monetary'] > 0).all(), "Found non-positive Monetary values.")
        self.assertTrue((self.df_seg['Avg_Order_Value'] > 0).all(), "Found non-positive Avg Order Value.")

    def test_05_cluster_optimization_metrics(self):
        """Verify cluster evaluation metrics table records K=2 to K=8 with monotonic decreasing inertia."""
        self.assertEqual(len(self.df_eval), 7, "Expected 7 evaluation entries (K=2 to 8).")
        inertias = self.df_eval['Inertia_SSE'].tolist()
        # Inertia must decrease monotonically as K increases
        for i in range(len(inertias) - 1):
            self.assertGreater(inertias[i], inertias[i+1], "Inertia failed monotonic decrease property.")
            
        # Silhouette scores must be bounded in [-1, 1]
        silhouettes = self.df_eval['Silhouette_Score'].tolist()
        for s in silhouettes:
            self.assertTrue(-1.0 <= s <= 1.0, f"Invalid silhouette score: {s}")

    def test_06_persona_coverage_and_distribution(self):
        """Verify all 4 distinct personas are populated with reasonable customer counts."""
        expected_personas = {
            'Champions / High-Value VIPs',
            'Loyal & Consistent Regulars',
            'Recent / New Promising Customers',
            'Hibernating / Churned At-Risk Customers'
        }
        actual_personas = set(self.df_seg['Persona'].unique())
        self.assertEqual(actual_personas, expected_personas, "Mismatch in customer persona names.")
        
        # Verify each cluster has at least 500 customers (balanced clustering)
        counts = self.df_seg['Persona'].value_counts()
        for persona, count in counts.items():
            self.assertGreater(count, 500, f"Cluster '{persona}' has too few members ({count}).")

    def test_07_revenue_reconciliation(self):
        """Verify sum of segment monetary totals reconciles to overall portfolio revenue."""
        total_customer_revenue = self.df_seg['Monetary'].sum()
        profile_revenue_sum = self.df_profiles['Monetary_sum'].sum()
        self.assertAlmostEqual(total_customer_revenue, profile_revenue_sum, places=1,
                               msg="Revenue sum mismatch between segmented dataset and profile summary.")

    def test_08_artifacts_generated(self):
        """Verify all required figures and tables exist on disk and exceed minimum byte sizes."""
        expected_figs = [
            'fig1_rfm_distributions.png',
            'fig2_elbow_and_silhouette.png',
            'fig3_cluster_scatter_plots.png',
            'fig4_cluster_customer_counts.png'
        ]
        for fig in expected_figs:
            fig_path = os.path.join(self.fig_dir, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig}")
            self.assertGreater(os.path.getsize(fig_path), 10000, f"Figure truncated: {fig}")
            
        expected_tabs = [
            'rfm_summary_statistics.csv',
            'cluster_evaluation_metrics.csv',
            'cluster_persona_profiles.csv',
            'customer_segment_marketing_matrix.csv'
        ]
        for tab in expected_tabs:
            tab_path = os.path.join(self.tab_dir, tab)
            self.assertTrue(os.path.exists(tab_path), f"Missing table: {tab}")
            self.assertGreater(os.path.getsize(tab_path), 50, f"Table truncated: {tab}")

if __name__ == '__main__':
    unittest.main()
