"""
test_retail_sales_eda.py
Automated Unit Test Suite for Project 01 (Retail Sales EDA) — EDA on Retail Sales Data
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO (Business-Analytics-Portfolio)

Validates dataset existence, schema integrity, duplicate count, accounting identities,
financial reconciliation, and exported visual/tabular artifacts.
"""

import os
import unittest
import pandas as pd


class TestRetailSalesEDA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_path = os.path.join(cls.base_dir, 'data', 'retail_sales_data.csv')
        cls.fig_dir = os.path.join(cls.base_dir, 'outputs', 'figures')
        cls.tab_dir = os.path.join(cls.base_dir, 'outputs', 'tables')
        
        if os.path.exists(cls.data_path):
            cls.df_raw = pd.read_csv(cls.data_path)
            cls.df_clean = cls.df_raw.drop_duplicates()
        else:
            cls.df_raw = None
            cls.df_clean = None

    def test_01_dataset_exists_and_shape(self):
        """Verify retail_sales_data.csv exists and matches expected dimensions (113,036 x 18)."""
        self.assertIsNotNone(self.df_raw, "Retail sales dataset missing from data directory.")
        self.assertEqual(self.df_raw.shape, (113036, 18), f"Unexpected raw shape: {self.df_raw.shape}")

    def test_02_columns_and_schema(self):
        """Verify all 18 required dimension and financial metric columns are present."""
        expected_cols = [
            'Date', 'Day', 'Month', 'Year', 'Customer_Age', 'Age_Group', 'Customer_Gender',
            'Country', 'State', 'Product_Category', 'Sub_Category', 'Product',
            'Order_Quantity', 'Unit_Cost', 'Unit_Price', 'Profit', 'Cost', 'Revenue'
        ]
        self.assertEqual(list(self.df_raw.columns), expected_cols)

    def test_03_duplicate_records_and_clean_shape(self):
        """Verify exactly 1,000 exact duplicate rows exist, yielding 112,036 clean rows."""
        dup_count = self.df_raw.duplicated().sum()
        self.assertEqual(dup_count, 1000, f"Expected 1,000 duplicates, found {dup_count}")
        self.assertEqual(len(self.df_clean), 112036, f"Expected 112,036 clean rows, found {len(self.df_clean)}")

    def test_04_no_missing_values(self):
        """Verify raw and clean datasets contain exactly zero null/missing values."""
        self.assertEqual(self.df_raw.isnull().sum().sum(), 0, "Missing values found in raw dataset.")
        self.assertEqual(self.df_clean.isnull().sum().sum(), 0, "Missing values found in clean dataset.")

    def test_05_accounting_identities(self):
        """Verify arithmetic consistency: Cost = Order_Quantity * Unit_Cost and Profit = Revenue - Cost."""
        cost_match = (self.df_clean['Cost'] == self.df_clean['Order_Quantity'] * self.df_clean['Unit_Cost']).all()
        profit_match = (self.df_clean['Profit'] == self.df_clean['Revenue'] - self.df_clean['Cost']).all()
        self.assertTrue(cost_match, "Cost identity violated in cleaned records.")
        self.assertTrue(profit_match, "Profit identity violated in cleaned records.")

    def test_06_revenue_reconciliation(self):
        """Verify exact financial revenue totals: Raw $85.27M, Clean $84.83M, Difference $444.2k."""
        raw_rev = self.df_raw['Revenue'].sum()
        clean_rev = self.df_clean['Revenue'].sum()
        diff_rev = raw_rev - clean_rev
        self.assertEqual(raw_rev, 85271008.00, f"Unexpected raw revenue: {raw_rev}")
        self.assertEqual(clean_rev, 84826772.00, f"Unexpected clean revenue: {clean_rev}")
        self.assertEqual(diff_rev, 444236.00, f"Unexpected duplicate revenue difference: {diff_rev}")

    def test_07_figures_exist(self):
        """Verify all 6 publication figures exist and exceed 20 KB in size."""
        expected_figs = [
            'fig1_monthly_sales_trend.png',
            'fig2_quarterly_sales_trend.png',
            'fig3_customer_demographics.png',
            'fig4_top_products_and_categories.png',
            'fig5_correlation_heatmap.png',
            'fig6_additional_insight_profit_margins.png'
        ]
        for fig_name in expected_figs:
            fig_path = os.path.join(self.fig_dir, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig_name}")
            self.assertGreater(os.path.getsize(fig_path), 20_000, f"Figure too small: {fig_name}")

    def test_08_tables_exist(self):
        """Verify all 4 summary CSV tables exist and contain valid rows."""
        expected_tabs = [
            'correlation_matrix.csv',
            'country_performance.csv',
            'demographics_summary.csv',
            'descriptive_statistics.csv'
        ]
        for tab_name in expected_tabs:
            tab_path = os.path.join(self.tab_dir, tab_name)
            self.assertTrue(os.path.exists(tab_path), f"Missing summary table: {tab_name}")
            df_tab = pd.read_csv(tab_path)
            self.assertGreater(len(df_tab), 0, f"Table {tab_name} is empty.")


if __name__ == '__main__':
    unittest.main()
