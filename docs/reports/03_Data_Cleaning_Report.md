# DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO
## Formal Task Report: Foundational Analytics — Task 3: Cleaning Data
### High-Fidelity Deterministic Data Cleaning & Integrity Assurance Pipeline

**Candidate:** `Dhruv Goyal` *(Notice: Template document; customize with actual student name prior to final submission)*  
**Track:** Data Analytics  
**Task Code:** `PRJ-03`  
**Execution Date:** October 2026  
**Status:** `Completed & Quality Gate Verified`  
**Target Repository Path:** `projects/03_Data_Cleaning_Pipeline/`  

---

## Executive Summary

Data quality is the foundational prerequisite for all downstream business intelligence, machine learning, and financial reporting. In dirty transactional environments, applying naive statistical heuristics (such as column-wise mean or median imputation) introduces severe distortion, breaks accounting identities, and manufactures invalid business records.

This report documents the design, implementation, and rigorous verification of an **end-to-end deterministic data cleaning pipeline** applied to the 10,000-record Dirty Cafe Sales benchmark. The pipeline resolves hidden error tokens, multi-channel missingness, formatting variances, and statistical edge cases.

### Key Performance Highlights:
- **Exact Quantity Reconstruction:** Reconstructed 456 missing quantity values via deterministic integer division with floating-point tolerance ($\text{tol} < 10^{-5}$); rejected any fractional ratios or zero denominators.
- **Non-Tautological Accounting Proof:** Independently validated that on the **8,544 records where Quantity, Price Per Unit, and Total Spent were all directly observed**, the accounting identity $\text{Total Spent} \equiv \text{Quantity} \times \text{Price Per Unit}$ holds for **100.00% of rows (0 mismatches)**. This proved the underlying generative model before using it for imputation.
- **Transparent Missing Date Preservation:** Preserved 460 unparseable and missing dates as `NaT` (Not-a-Time in memory) and blank in CSV exports, explicitly rejecting synthetic median date imputation to prevent fabricating historical transaction records.
- **Method Provenance Classification:** Fully categorized every field transformation as *directly observed*, *deterministically reconstructed*, *inferred from domain assumption*, or *unresolved*.
- **Defensible Purge Strategy:** Isolated and dropped only 26 irreparably corrupt records (0.26% of transactions). Exactly \$54.00 is the known recorded revenue across the 3 discarded rows that retained a `Total Spent` value; the remaining 23 discarded rows had unrecorded/unknown revenue because both Quantity and Total Spent (or Price) were missing. Retained 99.74% of transactions with verified arithmetic consistency.
- **Domain-Driven Outlier Defense:** Evaluated 259 IQR statistical outliers at \$25.00 and defended their retention, proving they represent valid purchases of 5 Salads ($5 \times \$5.00$) rather than sensor errors. Blind statistical capping would have artificially destroyed \$259.00 in revenue.

---

## 1. Problem Statement & Baseline Data Quality Audit

Real-world point-of-sale (POS) systems, legacy data migrations, and network disconnects often produce datasets containing both visible nulls and deceptive string placeholders. 

The raw dataset (`raw_dirty_cafe_sales.csv`) contains **10,000 records across 8 columns**, all initially loaded as generic `object` (string) data types.

### Forensic Data Quality Audit Summary

| Column | Raw Inferred Dtype | Native Nulls | Dirty String Tokens (`ERROR`/`UNKNOWN`) | Total Corrupted Records | Corruption Rate (%) | Unique Raw Values |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`Transaction ID`** | `object` | 0 | 0 | 0 | **0.00%** | 10,000 |
| **`Item`** | `object` | 333 | 636 | 969 | **9.69%** | 10 |
| **`Quantity`** | `object` | 138 | 341 | 479 | **4.79%** | 7 |
| **`Price Per Unit`** | `object` | 179 | 354 | 533 | **5.33%** | 8 |
| **`Total Spent`** | `object` | 173 | 329 | 502 | **5.02%** | 19 |
| **`Payment Method`** | `object` | 2,579 | 599 | 3,178 | **31.78%** | 5 |
| **`Location`** | `object` | 3,265 | 696 | 3,961 | **39.61%** | 4 |
| **`Transaction Date`** | `object` | 159 | 301 | 460 | **4.60%** | 367 |

### Primary Key Audit & Duplicate Row Analysis
- **Exact Full-Row Duplicates:** $0$ rows.
- **Duplicate Primary Keys (`Transaction ID`):** $0$ duplicate IDs ($10,000$ unique IDs).
- **Natural Attribute Collisions:** $141$ records share identical non-ID attributes (`Item`, `Quantity`, `Price`, `Total`, `Payment Method`, `Location`, `Date`). In an urban cafe serving thousands of customers across 365 operating days, two customers purchasing identical items (e.g., 1 Coffee for Takeaway with Cash on a Friday morning) is an expected business event, **not a data pipeline duplication defect**. Purging these would improperly undercount real cafe sales.

---

## 2. Deterministic Cleaning Methodology & Provenance

### 2.1 The Menu Pricing Key & Algebraic Equations
Inspection of uncorrupted records revealed an underlying discrete pricing matrix:

| Menu Item | Price Per Unit | Category | Reverse-Lookup Status |
|:---|:---:|:---:|:---|
| **Cookie** | \$1.00 | Bakery | Unique price point ($1.00 \to \text{Cookie}$) |
| **Tea** | \$1.50 | Beverage | Unique price point ($1.50 \to \text{Tea}$) |
| **Coffee** | \$2.00 | Beverage | Unique price point ($2.00 \to \text{Coffee}$) |
| **Cake** | \$3.00 | Bakery | Shared tier with Juice |
| **Juice** | \$3.00 | Beverage | Shared tier with Cake |
| **Sandwich** | \$4.00 | Food / Meal | Shared tier with Smoothie |
| **Smoothie** | \$4.00 | Beverage | Shared tier with Sandwich |
| **Salad** | \$5.00 | Food / Meal | Unique price point ($5.00 \to \text{Salad}$) |

### 2.2 Reconstruction Method Provenance Classification
Every repaired field in the active 9,974 clean records has been formally audited:

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

### 2.3 Purge Rationale for the 26 Unrecoverable Records
Following algebraic reconstruction, exactly 26 records (0.26%) remained unresolvable:
- In these records, **two or more core dimensions were corrupted simultaneously** (e.g., both `Quantity` and `Total Spent` were missing, or `Item`, `Price`, and `Total Spent` were missing).
- Without at least two equations in $\text{Total Spent} = \text{Quantity} \times \text{Price Per Unit}$, the mathematical system is underdetermined ($1$ equation, $2$ unknowns).
- Imputing with mean or mode would invent synthetic transactions and corrupt downstream margin analysis.
- **Revenue Reconciliation of Discarded Rows:** Exactly \$54.00 is the *known recorded revenue* across the 3 discarded rows that retained a non-null `Total Spent` value (Row 10: \$25.0, Row 16: \$9.0, Row 24: \$20.0). The remaining 23 discarded rows had *unrecorded/unknown revenue* because both `Quantity` and `Total Spent` (or Price) were missing. Because quantities in this cafe range from 1 to 5 and prices range from \$1.00 to \$5.00, true unrecorded revenue cannot be determined without pure speculation. Their small proportion (0.23% of transactions) ensures minimal impact on analytical conclusions.
- **Action:** Dropped 26 rows. Saved full audit trace to `outputs/tables/unrecoverable_transactions_audit.csv`.

### 2.4 Handling Categorical & Date Attributes
1. **Ambiguous Items ($3.00 and $4.00 tiers):**
   - 247 records have price \$3.00 (Cake vs. Juice) and 227 records have price \$4.00 (Sandwich vs. Smoothie).
   - Rather than arbitrarily guessing customer dietary preference, they are transparently labeled as `'Unknown ($3.0 Item)'` and `'Unknown ($4.0 Item)'`. All financial calculations remain 100% exact.
2. **Payment Method & Location:**
   - Both features exhibit high corruption rates (31.8% and 39.6%).
   - Imputing with the mode (`Cash` or `Takeaway`) would artificially bias channel analytics and mislead operations.
   - Both were imputed with `'Unknown'` to preserve overall sales volume without fabricating behavioral facts.
3. **Transaction Date:**
   - 460 unparseable dates were preserved as `NaT` (Not-a-Time in memory) and blank in CSV exports.
   - Synthetic median date imputation was rejected because it would artificially inject 460 transactions into a single arbitrarily chosen summer day (`2023-07-02`), distorting day-of-week and monthly trends.

---

## 3. Statistical Outlier Audit & Domain Justification

Standard statistical packages flag outliers using the Interquartile Range (IQR) rule:
$$\text{Tukey's Fences}: [Q_1 - 1.5 \times \text{IQR}, \; Q_3 + 1.5 \times \text{IQR}]$$

### Empirical Distribution Findings:
| Metric | Min | Max | Mean | Std Dev | Q1 (25%) | Q3 (75%) | IQR | 1.5*IQR Upper Cutoff | IQR Outlier Count | Z-Score Outlier Count ($|Z|>3$) | Decision |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`Quantity`** | 1 | 5 | 3.03 | 1.42 | 2.0 | 4.0 | 2.0 | 7.0 | 0 | 0 | None (Within bounds) |
| **`Price Per Unit`** | \$1.00 | \$5.00 | \$2.95 | \$1.28 | \$2.00 | \$4.00 | \$2.00 | \$7.00 | 0 | 0 | None (Within bounds) |
| **`Total Spent`** | \$1.00 | \$25.00 | \$8.92 | \$6.01 | \$4.00 | \$12.00 | \$8.00 | **\$24.00** | **259** | 0 | **Retained (Valid Business Edge)** |

### Analytical Defense of Retaining the 259 Outliers:
1. **Physical Reality:** Every one of the 259 transactions at \$25.00 represents a valid purchase of **5 Salads at \$5.00 each** ($5 \times \$5.00 = \$25.00$).
2. **Arithmetic Consistency:** Capping these records to \$24.00 would break the identity:
   $$5 \times \$5.00 = \$25.00 \neq \$24.00$$
   This would force downstream analytics to report either a fake price of \$4.80 or a fractional quantity of 4.8 salads!
3. **Revenue Impact:** Trimming these records would purge **\$6,475.00** of authentic, high-margin revenue from the business report.
4. **Conclusion:** Statistical outlier boundaries must always yield to verifiable physical business realities. The 259 records were retained in full.

---

## 4. Before-and-After Transformation Matrix

| Feature | Raw Dtype | Clean Dtype | Raw Corrupt Count | Raw Corrupt % | Clean Null Count | Clean Null % | Raw Unique | Clean Unique | Remediation Strategy |
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

## 5. Artifacts and Generated Evidence

All deliverables have been generated on disk, programmatically verified, and tested:

1. **Jupyter Notebook:** `projects/03_Data_Cleaning_Pipeline/Data_Cleaning_Pipeline.ipynb` (363 KB, executed headlessly via `nbconvert`, 21 cells, all plots embedded).
2. **Clean Dataset:** `projects/03_Data_Cleaning_Pipeline/data/cleaned_cafe_sales.csv` (9,974 records, 599 KB).
3. **Standalone Cleaning Script:** `projects/03_Data_Cleaning_Pipeline/clean_cafe_sales.py`.
4. **Unit Test Suite:** `projects/03_Data_Cleaning_Pipeline/test_cleaning_pipeline.py` (9 tests passing in 0.07s).
5. **High-Resolution Figures:**
   - Figure 1: `outputs/figures/fig1_data_quality_anomalies.png`
   - Figure 2: `outputs/figures/fig2_outlier_detection.png`
   - Figure 3: `outputs/figures/fig3_cleaned_sales_distribution.png`
6. **Data Quality Audit Tables:**
   - `outputs/tables/before_after_metrics.csv`
   - `outputs/tables/raw_data_quality_audit.csv`
   - `outputs/tables/canonical_menu_prices.csv`
   - `outputs/tables/outlier_analysis_summary.csv`
   - `outputs/tables/unrecoverable_transactions_audit.csv`
   - `outputs/tables/reconstruction_method_classification.csv`
