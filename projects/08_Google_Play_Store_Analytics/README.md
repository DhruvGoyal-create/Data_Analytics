# Project 08: Google Play Store Market Analytics & Consumer Sentiment Engine

**Project Code:** `PRJ-08`  
**Domain:** Mobile Application Marketplace Intelligence & NLP  
**Author:** Dhruv Goyal ([GitHub Profile](https://github.com/DhruvGoyal-create))  
**Specialization:** Data Analytics, Business Intelligence & Econometric Analysis  
**Repository:** `Secondary A2-Bussiness Analytics`  
**Test Suite:** `test_play_store_analysis.py` (10 / 10 Tests PASS)  

---

## 1. Executive Summary & Objective

The Google Play Store functions as one of the world's largest digital software marketplaces, hosting millions of competing applications across diverse business categories. This project conducts an exhaustive, empirical market analysis across **10,841 application listings** and **64,295 consumer review records**, systematically diagnosing market saturation, pricing elasticity, technical packaging dynamics (binary file size in MB), user rating distributions, and qualitative customer sentiment polarity.

The primary objective is to replace intuitive app publishing guesses with reproducible quantitative benchmarks, providing actionable go-to-market strategies for software developers, product managers, and digital investors.

---

## 2. Methodology & Data Engineering Architecture

The analysis is driven by `google_play_store_analysis.py` and structured as a 5-stage automated data pipeline:

1. **Ingestion & Data Hygiene:**
   - Detects and handles corrupted record **Row 10472** (`Life Made WI-Fi Touchscreen Photo Frame`), where a missing category caused field shifting and an impossible rating of `19.0` (valid range: 1.0 to 5.0).
   - Cleans numeric columns by stripping non-numeric artifacts: removes `+` and `,` from `Installs` (converted to `int64`), removes `$` from `Price` (converted to `float64`), and standardizes missing values.
   - Parses alphanumeric application size representations (e.g., `19M` $\to$ 19.0 MB, `1024k` $\to$ 1.0 MB), while preserving `"Varies with device"` as explicit `NaN` to prevent imputation distortion.
   - De-duplicates multi-snapshot application listings by sorting on `Reviews` descending and selecting the latest, most authoritative snapshot per unique package name, reducing 10,841 raw rows to **9,659 clean unique applications**.

2. **Cross-Entity Merging & Sentiment Cleaning:**
   - Cleans 64,295 user reviews by isolating 37,427 non-null text records with valid TextBlob polarity $[-1.0, +1.0]$ and subjectivity $[0.0, 1.0]$.
   - Inner-joins review sentiment data with application metadata on `App`, yielding **35,929 high-fidelity merged observations** spanning 33 categories.

3. **Statistical Analysis & Econometrics:**
   - Computes parametric and non-parametric statistics (mean, median, interquartile range, standard deviation) across ratings and reviews.
   - Evaluates Spearman rank correlation ($\rho$) between application file size (MB) and install volume to measure the adoption penalty of bloated APKs.
   - Computes minimum qualified upfront revenue lower bounds ($\text{Est\_Revenue} = \text{Installs} \times \text{Price}$).

4. **Visual Analytics & Interactive Deliverables:**
   - Generates 6 publication-grade charts (300 DPI) using Seaborn and Matplotlib.
   - Compiles an interactive 4-dimensional HTML market landscape (`interactive_playstore_viz.html`) powered by Plotly CDN.

---

## 3. Key Analytical Findings & Business Insights

| Category Metric | Leading / Noteworthy Categories | Quantitative Findings & Strategic Takeaways |
|:---|:---|:---|
| **Market Saturation** | `FAMILY` (1,832 apps), `GAME` (959 apps), `TOOLS` (827 apps) | Top 3 categories account for 37.5% of all catalog offerings, signaling high acquisition friction and steep advertising costs. |
| **User Rating Benchmark** | Global Median: **4.30** / 5.0 (Mean: 4.17) | Positively skewed distribution with tight dispersion ($IQR = 0.50$). Ratings below 4.0 fall into the bottom quartile and experience severe organic ranking degradation. |
| **Size vs. Downloads** | Game ($\rho = +0.31$, $p < 10^{-15}$)<br>Tools ($\rho = -0.04$, $p = 0.28$) | In gaming, higher file size correlates positively with downloads (reflecting richer 3D assets); in utility/tools, users penalize heavy packages. |
| **Revenue Generation** | Free: **92.2%** of Catalog<br>Paid: **7.8%** of Catalog | Upfront purchase monetization is heavily concentrated. Enterprise categories (`BUSINESS`, `MEDICAL`, `FINANCE`) support healthy single-purchase pricing ($2.99–$29.99), whereas casual consumer apps require freemium or in-app purchase architectures. |
| **Consumer Sentiment** | Positive: **64.1%**<br>Neutral: **13.8%**<br>Negative: **22.1%** | Positive sentiment dominates general lifestyle and entertainment apps, while financial and utility apps exhibit higher negative sentiment density due to transactional friction and authentication bugs. |

---

## 4. Directory Structure

```
08_Google_Play_Store_Analytics/
├── data/
│   ├── googleplaystore.csv                  # Raw apps catalog (10,841 rows)
│   └── googleplaystore_user_reviews.csv     # Raw user reviews (64,295 rows)
├── outputs/
│   ├── figures/                             # 300 DPI publication charts & interactive HTML
│   │   ├── fig1_category_distribution.png
│   │   ├── fig2_ratings_distribution_by_category.png
│   │   ├── fig3_size_vs_installs_correlation.png
│   │   ├── fig4_pricing_and_revenue_analysis.png
│   │   ├── fig5_sentiment_distribution.png
│   │   ├── fig6_category_sentiment_breakdown.png
│   │   └── interactive_playstore_viz.html
│   └── tables/                              # Statistical summary CSV exports (7 tables)
│       ├── category_ratings_summary.csv
│       ├── category_revenue_estimates.csv
│       ├── category_saturation_analysis.csv
│       ├── category_sentiment_profiles.csv
│       ├── dataset_profile_summary.csv
│       ├── sentiment_classification_summary.csv
│       └── size_install_correlation_stats.csv
├── google_play_store_analysis.py            # Master end-to-end analytical pipeline
├── test_play_store_analysis.py              # Automated unit test suite (10/10 PASS)
├── requirements.txt                         # Pinned dependencies
└── README.md                                # Project documentation
```

---

## 5. Verification & Testing

To execute the project-level test suite:
```bash
python test_play_store_analysis.py
```

Expected result:
```
Ran 10 tests in ~2.1s
OK (10/10 tests passed)
```
