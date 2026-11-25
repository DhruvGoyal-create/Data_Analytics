# Foundational Analytics — Task 2: Customer Segmentation

## Behavioral RFM Modeling & K-Means Clustering on Retail Transactions

**Track:** Data Analytics  
**Organization:** Data Analytics Portfolio (`Business-Analytics-Portfolio`)  
**Task Code:** `PRJ-02`  
**Execution Date:** October 2026  
**Candidate Name:** `Dhruv Goyal` *(Template document — replace with full legal name prior to official video recording/submission)*  
**Current Status:** `Testing Complete` (All pipeline transformations and test assertions verified on disk)

---

## 1. Project Overview & Business Problem

Customer segmentation is the analytical discipline of dividing a company's customer base into distinct, actionable cohorts sharing similar behavioral traits. While demographic analysis (e.g., age, gender) provides static descriptors, it fails to explain purchasing cadence, monetary value, or churn risk.

This project implements an **end-to-end behavioral segmentation pipeline** utilizing the classical **Recency, Frequency, Monetary (RFM)** framework combined with unsupervised **K-Means clustering**. Using 541,909 raw transactional records from a UK-based online gift retailer, the pipeline engineers customer-level behavioral profiles, normalizes extreme right-skewed spending distributions, rigorously identifies the optimal cluster count ($K=4$) via the Elbow Method and Silhouette Analysis, profiles distinct customer personas, and formulates data-driven CRM marketing strategies.

---

## 2. Official Task Checklist & Compliance

| # | Official Checklist Requirement | Implementation Details | Verification Status |
|---|--------------------------------|------------------------|---------------------|
| 1 | **Load & Inspect Structure** | Ingested raw dataset, audited schema, handled 135,080 unassigned transactions (guest checkouts), and filtered cancellations/returns (`InvoiceNo` starting with 'C'). | **Verified** |
| 2 | **Descriptive Statistics** | Calculated average purchase value (£480.97 mean, £301.83 median), purchase frequency (4.27 mean, 2.0 median), and total customer spend (£2,054.27 mean, £674.48 median). | **Verified** |
| 3 | **RFM Feature Engineering** | Engineered Recency (days since last purchase relative to snapshot date `2011-12-10`), Frequency (distinct completed invoices), and Monetary (gross customer spend) for 4,338 customers. | **Verified** |
| 4 | **Data Normalization** | Addressed extreme right-skewness via `np.log1p()` transformation and standardized features using `StandardScaler` ($\mu=0, \sigma=1$). | **Verified** |
| 5 | **K-Means Clustering & Elbow Method** | Evaluated cluster counts $K=2$ to $K=8$ using Inertia (SSE) and Silhouette coefficients; selected $K=4$ based on elbow inflection and silhouette local maximum (0.3371, indicating modest cluster separation across continuous spend distributions). | **Verified** |
| 6 | **2D Cluster Visualizations** | Rendered multi-panel scatter plots across key feature combinations (Recency vs. Monetary, Frequency vs. Monetary on log-log scales). | **Verified** |
| 7 | **Cluster Profiling & Personas** | Computed mean and median behavioral benchmarks per cluster and defined 4 actionable customer personas: Champions, Loyal Regulars, Recent Promising, and At-Risk Hibernating. | **Verified** |
| 8 | **Customer Distribution Chart** | Visualized customer counts and population percentages across all 4 segments in `outputs/figures/fig4_cluster_customer_counts.png`. | **Verified** |
| 9 | **Strategic Marketing Recommendations** | Formulated proposed CRM marketing playbooks, retention tactics, and revenue-maximizing hypotheses for each segment (to be validated via A/B testing). | **Verified** |

---

## 3. Dataset Lineage & Provenance

- **Dataset Name:** Online Retail Dataset (`raw_online_retail.csv`)
- **Original Source:** UCI Machine Learning Repository (Donated by Dr. Daqing Chen, London South Bank University)
- **Local Raw Path:** `data/raw_online_retail.csv` (43.95 MB, 541,909 records)
- **Segmented Export Path:** `data/rfm_customer_segments.csv` (4,338 customers)
- **License / Terms:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Temporal Coverage:** 1 December 2010 to 9 December 2011.
- **Transactional Schema:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.

---

## 4. Methodology & Analytical Architecture

```
+-----------------------------------------------------------------------------+
| Raw Transaction Logs: 541,909 records x 8 columns                           |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Data Cleaning & Filtering:                                                  |
|   - Exclude 135,080 rows without CustomerID (guest checkouts)               |
|   - Exclude 8,905 cancelled invoices (InvoiceNo prefix 'C')                 |
|   - Exclude non-positive quantities (returns) and zero/negative prices      |
|   -> Retained 397,884 clean, completed transactions across 4,338 customers  |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| RFM Feature Engineering:                                                    |
|   - Reference Date: 2011-12-10 12:50:00 (Max Date + 1 Day)                  |
|   - Recency = (Reference Date - Max Customer InvoiceDate).days              |
|   - Frequency = nunique(InvoiceNo)                                          |
|   - Monetary = sum(Quantity * UnitPrice)                                    |
|   - Avg Order Value = Monetary / Frequency                                  |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Feature Normalization:                                                      |
|   - Log Transformation: log1p(Recency, Frequency, Monetary)                 |
|   - Standardization: StandardScaler() -> Zero Mean, Unit Variance           |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Optimization & Unsupervised Clustering:                                     |
|   - Elbow Method: SSE Inertia from K=2 to 8 (inflection at K=4)             |
|   - Silhouette Analysis: Peak stability at K=4 (Score: 0.3371, modest)      |
|   - Model Fitting: KMeans(n_clusters=4, random_state=42, n_init=10)         |
+-----------------------------------------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------+
| Persona Mapping & Proposed Strategic Marketing Hypotheses                   |
+-----------------------------------------------------------------------------+
```

---

## 5. Cluster Persona Profile Benchmarks

Exported to `outputs/tables/cluster_persona_profiles.csv`:

| Persona Name | Customer Count | Customer Share (%) | Recency Mean (Days) | Frequency Mean (Orders) | Monetary Mean (£) | Total Revenue (£) | Revenue Share (%) | Avg Order Value (£) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Champions / High-Value VIPs** | 716 | 16.51% | 12.1 | 13.7 | £8,074.27 | £5,781,175.08 | **64.87%** | £623.51 |
| **Loyal & Consistent Regulars** | 1,173 | 27.04% | 71.1 | 4.1 | £1,802.83 | £2,114,718.42 | **23.73%** | £577.42 |
| **Recent / New Promising Customers** | 837 | 19.29% | 18.1 | 2.1 | £551.82 | £461,872.95 | **5.18%** | £282.11 |
| **Hibernating / Churned At-Risk** | 1,612 | 37.16% | 182.5 | 1.3 | £343.45 | £553,641.45 | **6.21%** | £284.42 |

### Key Business Insights:
1. **Classic Pareto Distribution:** Just **16.51% of customers (Champions)** generate **64.87% of total revenue** (£5.78M of £8.91M). Maintaining relationships with this cohort is the organization's single most critical commercial priority.
2. **Growth Engine (Regulars):** The 1,173 Loyal Regulars contribute 23.73% of revenue. Targeted upselling can elevate high-frequency members into Champions.
3. **High Dormancy Risk:** Over 37% of the customer base (1,612 customers) has been inactive for over 6 months (mean recency 182.5 days). Re-engagement or list pruning is necessary to protect CRM deliverability.

---

## 6. Proposed Marketing Hypotheses per Customer Persona

> [!NOTE]
> All marketing tactics outlined below represent **proposed business hypotheses** derived from behavioral segmentation. They do not constitute guaranteed revenue drivers and must be subjected to randomized controlled A/B testing (measuring incremental conversion, lift, and churn rates against holdout control groups) prior to enterprise-wide rollout.

Exported to `outputs/tables/customer_segment_marketing_matrix.csv`:

| Persona | Core Characteristics | Primary CRM Objective | Proposed Marketing Strategy (To Test) |
|:---|:---|:---|:---|
| **Champions / High-Value VIPs** | Ultra-recent (12 days), frequent repeat buyers (14 orders), highest spend (£8,074 mean). | Retention, brand advocacy, white-glove loyalty. | Dedicated VIP account manager, exclusive previews of new seasonal lines, private reward tiers, personalized handwritten thank-you notes. |
| **Loyal & Consistent Regulars** | Steady cadence (71 days), moderate-to-high spend (£1,803 mean), multiple orders (4 orders). | Cross-selling, basket expansion, tier migration. | Automated replenishment reminders based on product life-cycle, category cross-sell discounts (e.g., "Complete your collection"), multi-pack bundle savings. |
| **Recent / New Promising** | Transacted very recently (18 days) but only 1–2 orders; modest basket size (£552 mean). | Fast second purchase conversion, habit formation. | Multi-stage onboarding drip email sequence, product usage tutorials, 15% discount on second order placed within 30 days of first transaction. |
| **Hibernating / Churned At-Risk** | Long inactivity (>6 months), low order frequency (1.3 orders), lowest spend (£343 mean). | Reactivation of viable buyers; cost reduction on dormant contacts. | "We miss you" win-back campaign with compelling discount code (\$20 off \$50+), clearance catalog access; unsubscribe non-responders after 90 days. |

---

## 7. Directory Layout & Artifacts

```
projects/02_Customer_Segmentation/
├── Customer_Segmentation.ipynb      # Fully executed notebook (1.33 MB, 18 cells, all outputs embedded)
├── customer_segmentation.py         # Standalone reproducible execution script
├── test_customer_segmentation.py    # Automated unit test suite (8 test cases, 100% pass)
├── build_l1_t2_notebook.py          # Script used to generate reproducible notebook
├── README.md                        # Task documentation and analytical findings
├── requirements.txt                 # Pinned dependencies
├── data/
│   ├── raw_online_retail.csv        # Raw source dataset (541,909 rows, 43.95 MB)
│   └── rfm_customer_segments.csv    # Segmented dataset with RFM metrics and assigned personas (4,338 rows)
└── outputs/
    ├── figures/
    │   ├── fig1_rfm_distributions.png       # Raw vs. Log-transformed RFM distributions
    │   ├── fig2_elbow_and_silhouette.png    # Elbow curve and silhouette score plots across K=2..8
    │   ├── fig3_cluster_scatter_plots.png   # 2D scatter plots (Recency vs Monetary, Frequency vs Monetary)
    │   └── fig4_cluster_customer_counts.png # Customer count per cluster bar chart
    └── tables/
        ├── rfm_summary_statistics.csv       # Overall descriptive statistics
        ├── cluster_evaluation_metrics.csv   # SSE and silhouette scores across K=2..8
        ├── cluster_persona_profiles.csv     # Mean/median metrics and revenue shares per persona
        └── customer_segment_marketing_matrix.csv # Marketing action playbooks
```

---

## 8. How to Reproduce & Verify

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Run the Segmentation Pipeline
```bash
python customer_segmentation.py
```

### Step 3: Run the Automated Unit Test Suite
```bash
python -m unittest test_customer_segmentation.py
```
*Expected Output: `Ran 8 tests in 0.03s ... OK`*

### Step 4: Run the Interactive Notebook Headlessly
```bash
python -m nbconvert --to notebook --execute --inplace Customer_Segmentation.ipynb
```
