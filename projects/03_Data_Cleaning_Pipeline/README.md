# Foundational Analytics — Task 3: Cleaning Data

## Deterministic Data Cleaning & Integrity Pipeline for Retail Cafe Sales

**Track:** Data Analytics  
**Organization:** Data Analytics Portfolio (`Business-Analytics-Portfolio`)  
**Task Code:** `PRJ-03`  
**Execution Date:** October 2026  
**Candidate Name:** `Dhruv Goyal` *(Template document — replace with full legal name prior to official video recording/submission)*  
**Current Status:** `Testing Complete` (All pipeline transformations and test assertions verified on disk; see Quality Gate notes below)

---

## 1. Project Overview & Business Problem

In real-world data analytics and data engineering workflows, incoming transactional datasets rarely contain clean missingness. Point-of-sale (POS) systems, CRM connectors, and IoT data feeds frequently introduce:
1. **Placeholder string tokens** (`'UNKNOWN'`, `'ERROR'`, `'None'`, `'?'`) masquerading as valid data.
2. **Type collapse** where numerical, financial, and temporal fields load as generic strings.
3. **Multi-channel missingness** across items, unit prices, quantities, and transaction totals.
4. **Boundary conditions** that trigger naive statistical outlier detectors despite representing valid physical business reality.

This project delivers a **production-grade, mathematically verified data cleaning pipeline** for the 10,000-record Dirty Cafe Sales benchmark. Instead of relying on naive statistical imputation (e.g., filling prices with column means, which corrupts accounting formulas), this pipeline leverages **domain-specific menu pricing matrices** and **cross-column algebraic reconciliation** to reconstruct missing values with exact mathematical precision.

> **Important Quality Gate Notice:** Passing unit tests verifies software mechanics, schema compliance, and internal arithmetic consistency; it does not constitute absolute proof of unobserved physical events. Where data was permanently lost, records were either purged or explicitly marked unresolved.

---

## 2. Official Task Checklist & Quality Gate Verification

| # | Official Checklist Requirement | Implementation Details | Verification Status |
|---|--------------------------------|------------------------|---------------------|
| 1 | **Data Quality Report** | Audited raw schema, null counts, dirty string tokens, unique counts, and value anomalies; exported `outputs/tables/raw_data_quality_audit.csv` and rendered `outputs/figures/fig1_data_quality_anomalies.png`. | **Verified** |
| 2 | **Missing Data Handling** | Implemented deterministic menu price lookup, cross-column algebraic derivation ($TS = Q \times P$), and explicit category handling (`'Unknown'`) with documented justifications. | **Verified** |
| 3 | **Duplicate Removal & Audit** | Audited full-row duplicates (0), primary key uniqueness (100% unique `Transaction ID`), and natural business collisions (141 non-ID attribute matches); documented findings. | **Verified** |
| 4 | **Standardization** | Standardized casing, stripped whitespace, normalized store locations (`'In-store'`, `'Takeaway'`), and converted dates to ISO `YYYY-MM-DD`. | **Verified** |
| 5 | **Outlier Detection & Decision Justification** | Evaluated IQR (1.5*IQR fences) and Z-score ($|Z| > 3$); proved that 259 IQR outliers at $25.00 represent valid purchases (5 Salads @ $5.00) and defended retention over blind capping. | **Verified** |
| 6 | **Data Type Correction** | Enforced integer `Quantity`, 2-decimal float `Price Per Unit` and `Total Spent`, ISO `Transaction Date`, and standardized string categories. | **Verified** |
| 7 | **Before vs. After Summary Table** | Built side-by-side comparison matrix of dtypes, nulls, corrupt counts, unique values, and remediation strategies in `outputs/tables/before_after_metrics.csv`. | **Verified** |
| 8 | **Clean Dataset Export** | Exported clean dataset to `data/cleaned_cafe_sales.csv` (9,974 records; 460 missing dates preserved as blank/NaT; 100% internal arithmetic match). | **Verified** |

---

## 3. Dataset Lineage & Provenance

- **Dataset Name:** Dirty Cafe Sales Benchmark (`dirty_cafe_sales.csv`)
- **Original Author:** Ahmed Mohamed (Kaggle dataset benchmark)
- **Source Repository:** [shaadyalii/Dirty-Cafe-Sales-Dataset (GitHub mirror)](https://raw.githubusercontent.com/shaadyalii/Dirty-Cafe-Sales-Dataset/main/dirty_cafe_sales.csv)
- **Local Raw Path:** `data/raw_dirty_cafe_sales.csv`
- **Clean Export Path:** `data/cleaned_cafe_sales.csv`
- **License / Usage:** Open benchmark dataset for data engineering, quality assurance, and statistical cleaning practice (Community Educational / Apache 2.0 compatible).
- **Raw Dimensions:** 10,000 rows × 8 columns (`Transaction ID`, `Item`, `Quantity`, `Price Per Unit`, `Total Spent`, `Payment Method`, `Location`, `Transaction Date`).

---

## 4. Pipeline Architecture & Methodology

```
+-----------------------------------------------------------------------------+
| Raw Ingestion: 10,000 records, 8 object columns, hidden ERROR/UNKNOWN tokens |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Stage 1: Independent Accounting Audit on Directly Observed Rows (8,544 rows)|
|   - Verified Total Spent == Quantity * Price Per Unit on raw data (100% OK) |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Stage 2: Canonical Menu Price Imputation & Cross-Column Algebraic Solving   |
|   - Price = MenuLookup(Item)                                                |
|   - Price = Total Spent / Quantity                                          |
|   - Quantity = Total Spent / Price (strictly positive int, tol < 1e-5)      |
|   - Total Spent = Quantity * Price                                          |
|   - Unique Price Reverse-Mapping (1.0->Cookie, 1.5->Tea, 2.0->Coffee, etc.)|
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Stage 3: Purge 26 Fatally Corrupt Records (2+ dimensions missing)           |
|   - Known recorded revenue: $54.00 across 3 rows (25.0, 9.0, 20.0)          |
|   - Unknown unrecorded revenue across remaining 23 rows (missing Q or Price)|
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Stage 4: Categorical & Temporal Normalization                               |
|   - Ambiguous Items: 'Unknown ($3.0 Item)', 'Unknown ($4.0 Item)'           |
|   - Payment Method / Location: 'Unknown' (avoids mode distortion)           |
|   - Dates: Preserved 460 unparseable dates as NaT (no synthetic median fill)|
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Stage 5: Outlier Audit (Retain 259 records at $25.00: 5 Salads @ $5.00)     |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Stage 6: Schema Enforcement, Assertions & Export (9,974 rows)               |
+-----------------------------------------------------------------------------+
```

---

## 5. Reconstruction Method Provenance Classification

Every repaired field in the active 9,974 clean records has been formally classified by methodology:

| Field | Method Classification | Record Count | Percentage (%) | Description |
|:---|:---|:---:|:---:|:---|
| **`Item`** | `directly_observed` | 9,011 | 90.34% | Observed directly in raw source |
| | `deterministic_reverse_price_mapping` | 489 | 4.90% | Unique price points (\$1.00, \$1.50, \$2.00, \$5.00) |
| | `unresolved` | 474 | 4.75% | Ambiguous tiers (\$3.00 Cake/Juice, \$4.00 Sandwich/Smoothie) labeled explicitly |
| **`Quantity`** | `directly_observed` | 9,518 | 95.43% | Observed directly in raw source |
| | `deterministic_algebraic_reconstruction` | 456 | 4.57% | Reconstructed via $\text{Total} / \text{Price}$ ($\text{tol} < 10^{-5}$) |
| **`Price Per Unit`** | `directly_observed` | 9,447 | 94.72% | Observed directly in raw source |
| | `inferred_domain_menu_assumption` | 479 | 4.80% | Imputed from canonical menu price matrix |
| | `deterministic_algebraic_reconstruction` | 48 | 0.48% | Reconstructed via $\text{Total} / \text{Quantity}$ |
| **`Total Spent`** | `directly_observed` | 9,495 | 95.20% | Observed directly in raw source |
| | `deterministic_algebraic_reconstruction` | 479 | 4.80% | Reconstructed via $\text{Quantity} \times \text{Price}$ |

---

## 6. Before vs. After Transformation Summary

Exported to `outputs/tables/before_after_metrics.csv`:

| Feature | Raw Dtype | Clean Dtype | Raw Corrupt Count | Raw Corrupt Pct | Clean Null Count | Clean Null Pct | Raw Unique | Clean Unique | Remediation Strategy |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`Transaction ID`** | `object` | `object` | 0 | 0.00% | 0 | 0.00% | 10,000 | 9,974 | Validated 100% unique primary keys |
| **`Item`** | `object` | `object` | 969 | 9.69% | 0 | 0.00% | 10 | 10 | Unique price reverse-map; ambiguous tiers labeled explicitly |
| **`Quantity`** | `object` | `int64` | 479 | 4.79% | 0 | 0.00% | 7 | 5 | Deterministic integer division ($\text{Total}/\text{Price}$, $\text{tol} < 10^{-5}$) |
| **`Price Per Unit`** | `object` | `float64` | 533 | 5.33% | 0 | 0.00% | 8 | 6 | Menu lookup + algebraic back-calc ($\text{Total}/\text{Qty}$) |
| **`Total Spent`** | `object` | `float64` | 502 | 5.02% | 0 | 0.00% | 19 | 17 | Forward calc ($\text{Qty} \times \text{Price}$); verified against raw |
| **`Payment Method`** | `object` | `object` | 3,178 | 31.78% | 0 | 0.00% | 5 | 4 | Standardize casing; impute missing with `'Unknown'` |
| **`Location`** | `object` | `object` | 3,961 | 39.61% | 0 | 0.00% | 4 | 3 | Standardize casing; impute missing with `'Unknown'` |
| **`Transaction Date`** | `object` | `datetime64[ns]` | 460 | 4.60% | 460 | 4.61% | 367 | 365 | Parsed to ISO datetime; 460 missing preserved as NaT/blank |

---

## 7. Directory Layout & Key Deliverables

```
projects/03_Data_Cleaning_Pipeline/
├── Data_Cleaning_Pipeline.ipynb     # Interactive Jupyter Notebook (21 executed cells, all outputs rendered)
├── clean_cafe_sales.py              # Standalone automated cleaning pipeline script
├── test_cleaning_pipeline.py        # Automated unit test suite (9 test cases, 100% pass)
├── build_l1_t3_notebook.py          # Script used to generate reproducible notebook
├── README.md                        # Task documentation and analytical findings
├── requirements.txt                 # Pinned dependencies
├── data/
│   ├── raw_dirty_cafe_sales.csv     # Raw dirty source data (10,000 records, 550 KB)
│   └── cleaned_cafe_sales.csv       # Production-clean dataset (9,974 records, 599 KB)
└── outputs/
    ├── figures/
    │   ├── fig1_data_quality_anomalies.png        # Bar chart of missing/corrupt values
    │   ├── fig2_outlier_detection.png             # Boxplots with 1.5*IQR cutoff lines
    │   └── fig3_cleaned_sales_distribution.png    # Menu item volume and payment revenue
    └── tables/
        ├── before_after_metrics.csv               # Comparison summary matrix
        ├── raw_data_quality_audit.csv             # Forensic token audit
        ├── canonical_menu_prices.csv              # Menu pricing dictionary
        ├── outlier_analysis_summary.csv           # IQR and Z-score statistics
        ├── unrecoverable_transactions_audit.csv   # The 26 purged transactions
        └── reconstruction_method_classification.csv # Method breakdown per field
```

---

## 8. How to Reproduce & Verify

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Run the Cleaning Pipeline
```bash
python clean_cafe_sales.py
```

### Step 3: Run the Automated Unit Test Suite
```bash
python -m unittest test_cleaning_pipeline.py
```
*Expected Output: `Ran 9 tests in 0.07s ... OK`*

### Step 4: Run the Interactive Notebook
```bash
python -m nbconvert --to notebook --execute --inplace Data_Cleaning_Pipeline.ipynb
```
