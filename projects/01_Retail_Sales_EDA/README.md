# Foundational Analytics — Task 1: Exploratory Data Analysis (EDA) on Retail Sales Data

**Portfolio:** Data Analytics & Business Intelligence Portfolio
**Track:** Data Analytics  
**Execution Window:** 16 October 2026 – 15 November 2026  
**Project Directory:** `projects/01_Retail_Sales_EDA/`  
**Notebook Deliverable:** `EDA_Retail_Sales.ipynb`  

---

## 1. Project Overview & Objective
This project conducts a comprehensive, production-grade Exploratory Data Analysis (EDA) on an enterprise multi-year retail sales dataset (113,036 records spanning 2011 to 2016). The objective is to uncover underlying purchasing patterns, seasonality, demographic dynamics, and product catalog performance, translating quantitative findings into evidence-backed business recommendations.

---

## 2. Official Checklist Fulfillment

| Official Requirement | Implementation Detail | Status |
|:---|:---|:---:|
| **Initial Inspection** | Loaded 113,036 rows x 18 cols; verified dtypes; 0 missing values detected. | **Verified** |
| **Descriptive Statistics** | Mean, median, mode, and std dev calculated for 7 numerical attributes. | **Verified** |
| **Time Series Analysis** | Monthly and quarterly sales trendlines plotted and annotated across 2011–2016. | **Verified** |
| **Customer Demographics** | Age group distribution and gender revenue share analyzed and cross-tabulated. | **Verified** |
| **Product Analysis** | Top 10 products by units sold and revenue identified; category revenue bar charts. | **Verified** |
| **Correlation Heatmap** | Annotated Seaborn correlation matrix evaluating price-quantity and cost-revenue dynamics. | **Verified** |
| **Additional Visualization** | Geographic gross profit margin efficiency and cross-category margin spreads. | **Verified** |
| **Markdown Observations** | Detailed analytical commentary placed immediately below every visualization. | **Verified** |
| **Business Recommendations** | Three targeted, actionable commercial initiatives with projected KPIs. | **Verified** |

---

## 3. Dataset Provenance, Licensing & Schema
- **Primary Source:** FreeCodeCamp open-source curriculum repository: [`ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example`](https://github.com/ine-rmotr-curriculum/FreeCodeCamp-Pandas-Real-Life-Example) (File: `data/sales_data.csv`).
- **Curriculum Author:** Santiago Basulto / rmotr.com for FreeCodeCamp.
- **Underlying Upstream Origin:** Adapted and flattened from Microsoft's public sample database `AdventureWorks` (capturing bicycle, accessory, and clothing retail transactions).
- **License:** MIT License (Permissive open educational and portfolio use).
- **Temporal Span:** January 1, 2011 – July 31, 2016 (5.5 years of continuous commercial operations).
- **Dimensions:** 113,036 transactions across 6 countries (United States, Australia, United Kingdom, Germany, France, Canada).
- **Catalog Breadth:** 3 major categories (Accessories, Bikes, Clothing), 17 sub-categories, and 130 unique products.
- **Data Quality & Deduplication Sensitivity:**
  - 0 missing values across all 18 attributes.
  - Raw records: 113,036 rows (Total Revenue: \$85,271,008; Total Profit: \$32,221,100).
  - Strict duplicates: 1,000 rows (0.88%). Deduplication yields 112,036 clean rows (Total Revenue: \$84,826,772; Total Profit: \$32,046,239; net revenue difference: -\$444,236 or -0.52%).
  - Both raw and deduplicated totals are reported to provide full analytical transparency.
- **Arithmetic Consistency:** Reconciled internal arithmetic identities: $\text{Cost} = \text{Order\_Quantity} \times \text{Unit\_Cost}$ (100% match) and $\text{Profit} = \text{Revenue} - \text{Cost}$ (100% match). $\text{Unit\_Price}$ represents nominal list price while $\text{Revenue}$ represents net realized sales after transaction discounts ($\text{Discount} = \text{Quantity} \times \text{Unit\_Price} - \text{Revenue}$; mean discount: \$87.63). This demonstrates internal arithmetic consistency of table fields rather than proof of external real-world physical events.

---

## 4. Directory Structure

```
Task_1-EDA_on_retail_sales_data/
├── data/
│   └── retail_sales_data.csv          # Raw transaction log (113,036 rows)
├── outputs/
│   ├── figures/                       # High-resolution (300 DPI) exported charts
│   │   ├── fig1_monthly_sales_trend.png
│   │   ├── fig2_quarterly_sales_trend.png
│   │   ├── fig3_customer_demographics.png
│   │   ├── fig4_top_products_and_categories.png
│   │   ├── fig5_correlation_heatmap.png
│   │   └── fig6_additional_insight_profit_margins.png
│   └── tables/                        # Exported statistical summary CSVs
│       ├── descriptive_statistics.csv
│       ├── demographics_summary.csv
│       ├── correlation_matrix.csv
│       └── country_performance.csv
├── EDA_Retail_Sales.ipynb             # Master executable Jupyter Notebook
├── README.md                          # Project documentation and execution guide
└── requirements.txt                   # Reproducible Python dependencies
```

---

## 5. Environment Setup & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13.6 Windows x64)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Notebook via Jupyter or Command Line
To run interactively:
```bash
jupyter notebook EDA_Retail_Sales.ipynb
```

To execute headlessly and regenerate all outputs:
```bash
python -m nbconvert --to notebook --execute --inplace EDA_Retail_Sales.ipynb
```

---

## 6. Summary of Key Analytical Findings
1. **Catalog Dichotomy (Volume vs. Revenue):** Bicycles generate **73.0% (\\$61.7M)** of enterprise revenue from only 23% of transactions. Accessories dominate transaction frequency (55%) and unit sales (top volume items: *Water Bottle 30 oz.* with 232k units, *Patch Kit* with 157k units) but generate only 17.9% of revenue.
2. **Seasonal Demand Surges:** Sales velocity experiences two predictable annual peaks: a secondary surge in **May/June (Q2)** corresponding to the start of spring outdoor activities, and an all-time surge in **December (Q4)** peaking at \\$4.14M monthly.
3. **Core Demographic Engine:** Customers aged **35–64 (Adults)** generate **56.7% (\\$47.9M)** of total revenue with the highest Average Order Value (\\$790). Spending is evenly split between male (50.8%) and female (49.2%) shoppers.
4. **Margin Consistency:** Gross margins remain stable at **39.5%–41.5%** across all six geographic markets, with Accessories delivering the highest median margin (44.2%) compared to Bikes (36.8%–39.0%).

---

## 7. Actionable Business Recommendations
1. **Automated Point-of-Sale Accessory Attachment:** Implement bundled cross-selling at online checkout (e.g., offering helmet, bottle cage, and repair kit with every bike purchase) to lift bike order margin by an estimated 3.2%.
2. **Supply Chain Procurement Alignment:** Shift bulk bicycle container purchase orders 90 days earlier (August/September for Q4; February/March for Q2) to prevent stockouts on top revenue SKUs (`Mountain-200 Black`, `Road-150 Red`).
3. **Targeted Mature Adult VIP Program:** Launch a tailored loyalty program for the 35–64 professional demographic featuring seasonal bike servicing and premium carbon-frame upgrades to expand annual purchase frequency from 1.4 to 2.1 orders.
