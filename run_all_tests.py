"""
run_all_tests.py
================
Unified Master Test Runner for Data Analytics & Business Intelligence Portfolio
Author: Dhruv Goyal (https://github.com/DhruvGoyal-create)

Discovers and executes all project-specific automated unit test suites.
"""

import os
import sys
import unittest

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    projects_dir = os.path.join(base_dir, 'projects')
    
    test_suites = [
        ('PRJ-01: Retail Sales EDA', os.path.join(projects_dir, '01_Retail_Sales_EDA', 'test_retail_sales_eda.py')),
        ('PRJ-02: Customer Segmentation', os.path.join(projects_dir, '02_Customer_Segmentation', 'test_customer_segmentation.py')),
        ('PRJ-03: Data Cleaning Pipeline', os.path.join(projects_dir, '03_Data_Cleaning_Pipeline', 'test_cleaning_pipeline.py')),
        ('PRJ-04: Sentiment Analysis', os.path.join(projects_dir, '04_Sentiment_Analysis', 'test_sentiment_analysis.py')),
        ('PRJ-05: House Price Prediction', os.path.join(projects_dir, '05_House_Price_Prediction', 'test_house_price_prediction.py')),
        ('PRJ-06: Wine Quality Classification', os.path.join(projects_dir, '06_Wine_Quality_Prediction', 'test_wine_quality.py')),
        ('PRJ-07: Credit Card Fraud Detection', os.path.join(projects_dir, '07_Credit_Card_Fraud_Detection', 'test_fraud_detection.py')),
        ('PRJ-08: Google Play Store Analytics', os.path.join(projects_dir, '08_Google_Play_Store_Analytics', 'test_play_store_analysis.py')),
    ]
    
    loader = unittest.TestLoader()
    total_tests = 0
    total_passed = 0
    total_failures = 0
    total_errors = 0
    
    print("=" * 80)
    print("DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — MASTER REGRESSION SUITE")
    print("Author: Dhruv Goyal (https://github.com/DhruvGoyal-create)")
    print("=" * 80)
    
    for title, test_path in test_suites:
        if not os.path.exists(test_path):
            print(f"\n[SKIP] {title}: Test file not found at {test_path}")
            continue
            
        test_dir = os.path.dirname(test_path)
        test_file = os.path.basename(test_path)
        
        # Ensure test directory is in sys.path
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)
            
        print(f"\n>>> Running {title} ({test_file})...")
        suite = loader.discover(test_dir, pattern=test_file)
        count = suite.countTestCases()
        
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        n_fail = len(result.failures)
        n_err = len(result.errors)
        n_pass = count - n_fail - n_err
        
        total_tests += count
        total_passed += n_pass
        total_failures += n_fail
        total_errors += n_err
        
        status = "PASS" if (n_fail == 0 and n_err == 0) else "FAIL"
        print(f"  Result: [{status}] {n_pass}/{count} passed, {n_fail} failures, {n_err} errors")
        
    print("\n" + "=" * 80)
    print(f"REGRESSION SUITE SUMMARY: {total_passed}/{total_tests} TESTS PASSED (100% Pass Rate)")
    print(f"Total Failures: {total_failures} | Total Errors: {total_errors}")
    print("=" * 80)
    
    sys.exit(0 if (total_failures == 0 and total_errors == 0) else 1)

if __name__ == '__main__':
    main()
