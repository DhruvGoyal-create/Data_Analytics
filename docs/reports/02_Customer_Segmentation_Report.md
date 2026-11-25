# DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO
## Formal Task Report: Foundational Analytics — Task 2: Customer Segmentation
### Behavioral RFM Feature Engineering & K-Means Unsupervised Clustering on Online Retail Transactions

**Candidate:** `Dhruv Goyal` *(Notice: Template document; customize with actual student name prior to final submission)*  
**Track:** Data Analytics  
**Task Code:** `PRJ-02`  
**Execution Date:** October 2026  
**Status:** `Completed & Verified`  
**Target Repository Path:** `projects/02_Customer_Segmentation/`  

---

## Executive Summary

Customer segmentation is the cornerstone of modern CRM strategy, enabling businesses to replace generic mass marketing with targeted, personalized engagements that maximize customer lifetime value (CLV) and reduce churn.

This report documents the design, execution, and analytical findings of a **reproducible customer segmentation pipeline** built on 541,909 raw transactional records from the **UCI Online Retail Dataset**. Leveraging the classical **Recency, Frequency, Monetary (RFM)** framework, the pipeline transforms noisy invoice-level transactions into structured customer behavioral vectors, normalizes extreme positive right-skewness, applies unsupervised **K-Means clustering** optimized via the Elbow Method and Silhouette Analysis, and discovers four distinct, operational customer cohorts.

### Key Performance Highlights:
- **Comprehensive Data Filtering:** Isolated and filtered 135,080 unassigned transactions (guest checkouts) and 8,905 cancelled invoices, yielding **397,884 clean transactions across 4,338 unique customers**.
- **Portfolio Revenue Footprint:** The segmented customer base accounts for **£8,911,407.90 in gross commercial revenue** between December 2010 and December 2011.
- **Pareto Concentration Discovered:** Exactly **16.51% of customers (716 Champions)** generate **64.87% of total revenue** (£5.78M).
- **Cluster Count Optimization ($K=4$):** Evaluated $K=2$ to $K=8$. While $K=2$ yields mathematical separation (silhouette 0.4329), it oversimplifies customers into binary active/inactive groups. $K=4$ provides the optimal trade-off with an elbow inflection point, a silhouette local peak of **0.3371**, and four distinct, actionable marketing personas.
- **Actionable Strategic Matrix:** Formulated tailored retention, upselling, onboarding, and win-back playbooks for all four segments.

---

## 1. Problem Statement & Dataset Provenance

### Business Problem
A UK-based online non-store gift retailer serves thousands of individual consumers and small wholesale clients. Treating all customers uniformly leads to wasted promotional spend (e.g., offering discounts to VIPs who would purchase anyway) and missed retention opportunities (e.g., failing to re-engage once-frequent buyers before they permanently defect).

### Dataset Provenance & Lineage
- **Dataset Title:** Online Retail Dataset
- **Donor:** Dr. Daqing Chen, Director of Public Analytics, School of Engineering, London South Bank University
- **Original Archive:** UCI Machine Learning Repository (Online Retail Archive)
- **Direct Download URL:** `https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Local Path:** `data/raw_online_retail.csv` (43.95 MB, 541,909 rows × 8 columns)
- **Time Horizon:** 1 December 2010 to 9 December 2011 (373 calendar days)

---

## 2. Transactional Data Quality Cleaning & Exact Row Reconciliation

Raw transactional logs contain noise, non-customer checkouts, and administrative records that must be systematically filtered before customer entity aggregation.

### Exact Transaction Reconciliation Breakdown:
1. **Total Raw Ingested Records:** **541,909 rows (100.00%)**
2. **Missing CustomerID (Guest Checkouts):** **135,080 rows (24.93%)**
   - *Handling Justification:* Customer segmentation clusters *identifiable human entities*. Aggregating unassigned transactions into an artificial "Guest" customer would create an artificial hyper-cluster that corrupts distance calculations.
3. **Identified Cancellations (`InvoiceNo` prefix 'C'):** **8,905 rows (1.64%)**
   - *Handling Justification:* Order cancellations and returns with negative quantities must be removed to isolate completed, gross revenue-generating purchase behavior.
4. **Identified Non-Positive Unit Price:** **40 rows (0.01%)**
   - *Handling Justification:* Transactions with zero or negative price represent administrative corrections or data errors.
5. **Clean Completed Purchases Retained:** **397,884 rows (73.42%)**
   - *Customer Coverage:* Exactly **4,338 unique customers** across 373 operating days.
   - *Exact Arithmetic Balance:* $135,080 + 8,905 + 40 + 397,884 = 541,909$ rows (100.00% exact reconciliation).

---

## 3. Behavioral RFM Feature Engineering & Descriptive Statistics

The transactional data was aggregated to the customer level using the snapshot reference date of `2011-12-10 12:50:00` (one day after the final recorded invoice):

$$\text{Recency}_i = \text{Date}_{\text{ref}} - \max(\text{InvoiceDate}_i) \quad [\text{Days}]$$
$$\text{Frequency}_i = \text{CountDistinct}(\text{InvoiceNo}_i) \quad [\text{Invoices}]$$
$$\text{Monetary}_i = \sum_{j \in i} (\text{Quantity}_j \times \text{UnitPrice}_j) \quad [\text{GBP } £]$$
$$\text{Average Order Value}_i = \frac{\text{Monetary}_i}{\text{Frequency}_i} \quad [\text{GBP } £]$$

### RFM Summary Statistics (4,338 Customers)

| Metric | Mean | Std Dev | Min | 25% (Q1) | Median | 75% (Q3) | Max |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recency (Days)** | 92.54 | 100.01 | 1 | 18 | 51 | 142 | 374 |
| **Frequency (Orders)** | 4.27 | 7.70 | 1 | 1 | 2 | 5 | 209 |
| **Monetary (£)** | £2,054.27 | £8,989.23 | £3.75 | £307.41 | £674.48 | £1,661.74 | £280,206.02 |
| **Avg Order Value (£)** | £480.97 | £1,805.51 | £3.45 | £153.64 | £301.83 | £487.61 | £84,236.25 |

### Management of Skewness:
As shown in Figure 1, Frequency and Monetary exhibit extreme positive right-skewness (Monetary kurtosis > 200). Applying K-Means directly on un-transformed Euclidean space causes a handful of wholesale buyers (£280k spend) to monopolize individual clusters. Applying the log transformation:
$$x_{\text{transformed}} = \log(1 + x)$$
yields bell-shaped, quasi-normal feature distributions, which are subsequently standardized using `StandardScaler` to ensure zero mean and unit variance across all three axes.

---

## 4. K-Means Optimization: Elbow Method & Silhouette Analysis

We evaluated cluster counts from $K=2$ to $K=8$ using both Inertia (Sum of Squared Errors) and the Silhouette Coefficient:

| $K$ Clusters | Inertia (SSE) | Silhouette Score | Analytical Assessment |
|:---:|:---:|:---:|:---|
| **$K=2$** | 6,481.23 | 0.4329 | Binary split (Active vs. Inactive); insufficient operational granularity |
| **$K=3$** | 4,867.85 | 0.3365 | Broad separation; collapses regulars and new customers |
| **$K=4$** | **3,938.51** | **0.3371** | **Optimal Selection: Elbow inflection + local silhouette maximum** |
| **$K=5$** | 3,295.98 | 0.3161 | Fragmented clusters; lower cohesion |
| **$K=6$** | 2,855.01 | 0.3133 | Diminishing returns in variance explanation |
| **$K=7$** | 2,548.91 | 0.3100 | Marginal improvement in SSE; over-segmentation |
| **$K=8$** | 2,336.78 | 0.3008 | Overfitting; poor cluster interpretability |

**Selection Rationale:** While $K=2$ produces the highest silhouette score (0.4329), it represents a crude binary split (Active vs. Inactive) that fails the business requirement of separating high-value champions from newly acquired customers. $K=4$ sits at the prominent elbow inflection point, achieves a local silhouette peak of **0.3371**, and generates four operationally meaningful marketing cohorts. Crucially, a silhouette score of 0.3371 indicates **modest cluster separation**, which is expected in real-world retail behavioral data where customer spend and order frequency follow a continuous spectrum rather than naturally isolated, disjoint clusters. $K=4$ represents the optimal pragmatic balance between mathematical structure and CRM actionability.

---

## 5. Customer Persona Benchmarks

Exported to `outputs/tables/cluster_persona_profiles.csv`:

| Persona Name | Customer Count | Share (%) | Recency Mean | Frequency Mean | Monetary Mean | Total Spend (£) | Revenue Share (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Champions / High-Value VIPs** | 716 | 16.51% | 12.1 days | 13.7 orders | £8,074.27 | £5,781,175.08 | **64.87%** |
| **Loyal & Consistent Regulars** | 1,173 | 27.04% | 71.1 days | 4.1 orders | £1,802.83 | £2,114,718.42 | **23.73%** |
| **Recent / New Promising** | 837 | 19.29% | 18.1 days | 2.1 orders | £551.82 | £461,872.95 | **5.18%** |
| **Hibernating / Churned At-Risk** | 1,612 | 37.16% | 182.5 days | 1.3 orders | £343.45 | £553,641.45 | **6.21%** |

### Persona Characterizations:
1. **Champions / High-Value VIPs:** Bought within the last 2 weeks, average 14 orders across the year, and account for £8,074 mean spend. They are the financial foundation of the business.
2. **Loyal & Consistent Regulars:** Dependable repeat shoppers with ~4 orders and £1,803 spend. They represent the primary pool for account expansion.
3. **Recent / New Promising Customers:** Bought recently (mean 18 days) but have only 1–2 transactions. They have shown purchase intent but have not yet developed an entrenched buying habit.
4. **Hibernating / Churned At-Risk Customers:** Have not transacted in over 6 months (mean 182.5 days), with single low-value orders (£343 mean). They represent dormant churn risk.

---

## 6. Proposed Marketing Hypotheses & Experimental Playbooks

> [!NOTE]
> All marketing tactics outlined below represent **proposed business hypotheses** derived from behavioral segmentation. They do not constitute guaranteed revenue drivers and must be subjected to randomized controlled A/B testing (measuring incremental conversion, lift, and churn rates against holdout control groups) prior to enterprise-wide rollout.

Exported to `outputs/tables/customer_segment_marketing_matrix.csv`:

### 1. Champions / High-Value VIPs (Protect & Delight)
- **Primary Objective:** Maximize retention, cultivate brand advocacy, and protect against competitor poaching.
- **Proposed Campaign Tactics (To Test):**
  - Dedicated VIP support line and personal account manager.
  - Exclusive early access to seasonal gift collections 48 hours prior to public launch.
  - Surprise-and-delight gifting (e.g., high-value free product sample on order milestones).
  - Invite-only loyalty tier with complimentary premium packaging and expedited shipping.

### 2. Loyal & Consistent Regulars (Grow & Expand)
- **Primary Objective:** Increase average order value (AOV) and purchase frequency to graduate members into Champions.
- **Proposed Campaign Tactics (To Test):**
  - Automated predictive replenishment reminders based on historical purchasing intervals.
  - Bundle cross-sell recommendations ("Frequently bought together" discount tiers).
  - Spend-threshold rewards (e.g., "Spend £50 more to unlock Gold VIP status").

### 3. Recent / New Promising Customers (Nurture & Activate)
- **Primary Objective:** Secure the second and third purchases rapidly to build habituation.
- **Proposed Campaign Tactics (To Test):**
  - Multi-stage onboarding welcome sequence highlighting brand guarantees and bestselling lines.
  - Time-sensitive incentive: 15% discount code valid for 30 days post-initial purchase.
  - Post-purchase satisfaction surveys and review requests with small credit rewards.

### 4. Hibernating / Churned At-Risk Customers (Win-Back or Sunset)
- **Primary Objective:** Reactivate recoverable buyers while pruning permanent inactives to reduce marketing cost.
- **Proposed Campaign Tactics (To Test):**
  - Multi-channel "We miss you" win-back campaign featuring an aggressive discount (£20 voucher on £50+ orders).
  - "Tell us what went wrong" feedback surveys to uncover root causes of churn.
  - Automated sunset policy: contacts failing to open or click across a 90-day win-back sequence are unsubscribed to preserve email domain reputation.

---

## 7. Limitations & Methodological Caveats

1. **Unidentified Guest Checkouts:** 24.9% of transactions lacked a `CustomerID`. If guest shoppers behave differently from registered buyers (e.g., higher single-purchase rates), their exclusion could slightly overestimate repeat customer proportions.
2. **Holiday Seasonality:** The dataset begins and ends during the UK holiday shopping peak (December). Customers classified as "Hibernating" in summer months may simply be annual seasonal gift buyers who purchase exclusively in Q4.
3. **B2B vs. B2C Heterogeneity:** The dataset contains both bulk corporate gift buyers and individual shoppers. Future iterations should implement a B2B classifier prior to RFM segmentation to further refine cluster boundaries.
4. **Modest Cluster Separation & Continuous Behavioral Overlap:** The silhouette score of 0.3371 confirms moderate cluster cohesion with continuous overlap along the behavioral continuum. Customers do not occupy discrete, isolated islands; rather, boundaries are heuristic operational partitions. Marketing playbooks must therefore be treated as testable hypotheses rather than guaranteed revenue levers.

---

## 8. Artifacts & Deliverables Summary

- **Interactive Executed Notebook:** `projects/02_Customer_Segmentation/Customer_Segmentation.ipynb` (1.33 MB, 18 cells, all outputs rendered).
- **Segmented Dataset Export:** `projects/02_Customer_Segmentation/data/rfm_customer_segments.csv` (4,338 customers).
- **Automated Unit Tests:** `projects/02_Customer_Segmentation/test_customer_segmentation.py` (8 tests passing in 0.03s).
- **Figures on Disk:**
  - Figure 1: `fig1_rfm_distributions.png`
  - Figure 2: `fig2_elbow_and_silhouette.png`
  - Figure 3: `fig3_cluster_scatter_plots.png`
  - Figure 4: `fig4_cluster_customer_counts.png`
- **Tables on Disk:**
  - `rfm_summary_statistics.csv`
  - `cluster_evaluation_metrics.csv`
  - `cluster_persona_profiles.csv`
  - `customer_segment_marketing_matrix.csv`
