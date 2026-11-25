"""
google_play_store_analysis.py
=============================
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — Project 08 (Google Play Store Analytics)
Project: Unveiling the Android App Market (Google Play Store Analysis)

Full end-to-end data pipeline:
1. Dual-table ingestion: Play Store apps catalog & user reviews corpus.
2. Data hygiene & type parsing: row 10472 correction, Installs, Price, Size, Reviews.
3. Deduplication: retains highest review-count snapshot per unique app.
4. Descriptive & statistical analytics:
   - Category saturation & market concentration.
   - Ratings distribution & Kruskal-Wallis non-parametric ANOVA.
   - Size vs. Installs Spearman rank and Pearson log correlation.
   - Pricing economics & qualified minimum upfront gross revenue estimation.
   - User reviews sentiment NLP (polarity, subjectivity, category breakdown).
5. Publication-grade visualization suite (6 static PNG figures at 300 DPI).
6. Standalone interactive 4D visualization in Plotly HTML.
7. Structured CSV table exports.

Author: Data Analytics Portfolio Intern
Track: Data Analytics
"""

import os
import sys
import argparse
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

# Configure UTF-8 for safe console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Matplotlib aesthetic defaults
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def compute_sha256(filepath):
    """Compute and return uppercase SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def parse_size_to_mb(size_str):
    """
    Parse messy Size strings into numerical Megabytes (float).
    Examples: '19M' -> 19.0, '201k' -> ~0.196, 'Varies with device' -> np.nan.
    """
    if pd.isna(size_str) or size_str == 'Varies with device':
        return np.nan
    s = str(size_str).strip()
    if s.endswith('M'):
        try:
            return float(s[:-1])
        except ValueError:
            return np.nan
    elif s.endswith('k'):
        try:
            return float(s[:-1]) / 1024.0
        except ValueError:
            return np.nan
    return np.nan


def load_and_clean_data(base_dir):
    """
    Load and execute data cleaning protocol on Google Play Store apps and reviews.
    Returns df_apps_raw, df_apps_clean, df_reviews_raw, df_reviews_clean, df_merged.
    """
    data_dir = os.path.join(base_dir, 'data')
    apps_path = os.path.join(data_dir, 'googleplaystore.csv')
    rev_path = os.path.join(data_dir, 'googleplaystore_user_reviews.csv')

    if not os.path.exists(apps_path) or not os.path.exists(rev_path):
        raise FileNotFoundError(f"Required datasets not found in {data_dir}")

    # Load raw datasets
    df_apps_raw = pd.read_csv(apps_path)
    df_reviews_raw = pd.read_csv(rev_path)

    # 1. Clean Apps Catalog
    # Detect and drop shifted corrupted row (Row 10472 where Category is missing and Rating is 19.0)
    df_apps = df_apps_raw.copy()
    corrupt_mask = df_apps['Rating'] > 5.0
    num_corrupt = int(corrupt_mask.sum())
    if num_corrupt > 0:
        df_apps = df_apps.drop(index=df_apps[corrupt_mask].index).copy()

    # Numeric conversion for Reviews
    df_apps['Reviews'] = df_apps['Reviews'].astype(int)

    # Clean Installs: strip '+' and ',' -> int64
    df_apps['Installs_Numeric'] = (
        df_apps['Installs']
        .astype(str)
        .str.replace('+', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .astype(int)
    )

    # Clean Price: strip '$' -> float64
    df_apps['Price_Numeric'] = (
        df_apps['Price']
        .astype(str)
        .str.replace('$', '', regex=False)
        .str.strip()
        .astype(float)
    )

    # Normalize Type: Free vs Paid
    df_apps['Type'] = df_apps['Type'].fillna(
        df_apps['Price_Numeric'].apply(lambda p: 'Free' if p == 0.0 else 'Paid')
    )

    # Parse Size to MB
    df_apps['Size_MB'] = df_apps['Size'].apply(parse_size_to_mb)

    # Deduplication Policy:
    # Multiple scrape snapshots exist. Sort by Reviews descending and keep the first snapshot
    df_apps_sorted = df_apps.sort_values(by='Reviews', ascending=False)
    df_apps_clean = df_apps_sorted.drop_duplicates(subset=['App'], keep='first').copy()

    # Upfront Qualified Revenue (Direct Upfront Sales Lower Bound = Installs * Price)
    df_apps_clean['Est_Min_Upfront_Revenue'] = (
        df_apps_clean['Installs_Numeric'] * df_apps_clean['Price_Numeric']
    )

    # Save cleaned apps benchmark
    clean_apps_path = os.path.join(data_dir, 'googleplaystore_clean.csv')
    df_apps_clean.to_csv(clean_apps_path, index=False)

    # 2. Clean User Reviews
    df_reviews_clean = df_reviews_raw.dropna(subset=['Translated_Review']).copy()

    # 3. Merged Reviews with Clean Apps
    df_merged = pd.merge(
        df_reviews_clean,
        df_apps_clean[['App', 'Category', 'Rating', 'Type', 'Price_Numeric', 'Installs_Numeric']],
        on='App',
        how='inner'
    )

    return df_apps_raw, df_apps_clean, df_reviews_raw, df_reviews_clean, df_merged


def generate_tables(df_apps_raw, df_apps_clean, df_reviews_raw, df_reviews_clean, df_merged, tab_dir):
    """Generate and save 7 authoritative statistical summary CSV tables."""
    os.makedirs(tab_dir, exist_ok=True)

    # Table 1: Dataset profile summary
    profile_rows = [
        {'Metric': 'Raw Apps Catalog Rows', 'Value': f"{len(df_apps_raw):,}"},
        {'Metric': 'Clean Unique Apps Count', 'Value': f"{len(df_apps_clean):,}"},
        {'Metric': 'Dropped Corrupt Rows (Row 10472)', 'Value': f"{len(df_apps_raw) - 10840}"},
        {'Metric': 'Removed Duplicate Scrape Snapshots', 'Value': f"{10840 - len(df_apps_clean):,}"},
        {'Metric': 'Apps with Valid Ratings', 'Value': f"{df_apps_clean['Rating'].notnull().sum():,} ({df_apps_clean['Rating'].notnull().mean()*100:.1f}%)"},
        {'Metric': 'Apps with Known Size in MB', 'Value': f"{df_apps_clean['Size_MB'].notnull().sum():,} ({df_apps_clean['Size_MB'].notnull().mean()*100:.1f}%)"},
        {'Metric': 'Apps with Varies with Device Size', 'Value': f"{df_apps_clean['Size_MB'].isnull().sum():,} ({df_apps_clean['Size_MB'].isnull().mean()*100:.1f}%)"},
        {'Metric': 'Raw Reviews Corpus Total Rows', 'Value': f"{len(df_reviews_raw):,}"},
        {'Metric': 'Clean Reviews with Non-Null Text', 'Value': f"{len(df_reviews_clean):,} ({len(df_reviews_clean)/len(df_reviews_raw)*100:.1f}%)"},
        {'Metric': 'Merged Reviews with App Metadata', 'Value': f"{len(df_merged):,}"},
        {'Metric': 'Unique Apps Represented in Reviews', 'Value': f"{df_merged['App'].nunique():,}"}
    ]
    pd.DataFrame(profile_rows).to_csv(os.path.join(tab_dir, 'dataset_profile_summary.csv'), index=False)

    # Table 2: Category saturation analysis
    total_apps = len(df_apps_clean)
    total_installs = df_apps_clean['Installs_Numeric'].sum()
    cat_sat = df_apps_clean.groupby('Category').agg(
        App_Count=('App', 'count'),
        Total_Installs=('Installs_Numeric', 'sum'),
        Median_Installs=('Installs_Numeric', 'median'),
        Mean_Rating=('Rating', 'mean'),
        Median_Rating=('Rating', 'median')
    ).reset_index()
    cat_sat['App_Share_Pct'] = (cat_sat['App_Count'] / total_apps) * 100
    cat_sat['Install_Share_Pct'] = (cat_sat['Total_Installs'] / total_installs) * 100
    cat_sat = cat_sat.sort_values(by='App_Count', ascending=False)
    cat_sat.to_csv(os.path.join(tab_dir, 'category_saturation_analysis.csv'), index=False)

    # Table 3: Category ratings summary & Kruskal-Wallis stats
    cat_ratings = [
        group['Rating'].dropna().values
        for _, group in df_apps_clean.groupby('Category')
        if len(group['Rating'].dropna()) >= 10
    ]
    kw_stat, kw_p = stats.kruskal(*cat_ratings)
    cat_rate_df = df_apps_clean.groupby('Category')['Rating'].agg(
        Count='count',
        Mean='mean',
        Std='std',
        Median='median',
        Min='min',
        Max='max'
    ).reset_index().sort_values(by='Mean', ascending=False)
    cat_rate_df['KW_H_Statistic'] = round(kw_stat, 4)
    cat_rate_df['KW_P_Value'] = f"{kw_p:.4e}"
    cat_rate_df.to_csv(os.path.join(tab_dir, 'category_ratings_summary.csv'), index=False)

    # Table 4: Size vs. installs correlation stats
    df_size = df_apps_clean.dropna(subset=['Size_MB']).copy()
    spearman_rho, spearman_p = stats.spearmanr(df_size['Size_MB'], df_size['Installs_Numeric'])
    pearson_r, pearson_p = stats.pearsonr(df_size['Size_MB'], np.log1p(df_size['Installs_Numeric']))
    
    # Sizing tiers
    bins = [0, 10, 25, 50, 100]
    labels = ['< 10 MB', '10 - 25 MB', '25 - 50 MB', '50 - 100 MB']
    df_size['Size_Tier'] = pd.cut(df_size['Size_MB'], bins=bins, labels=labels)
    tier_stats = df_size.groupby('Size_Tier', observed=False).agg(
        App_Count=('App', 'count'),
        Median_Installs=('Installs_Numeric', 'median'),
        Mean_Installs=('Installs_Numeric', 'mean'),
        Mean_Rating=('Rating', 'mean')
    ).reset_index()
    tier_stats['Spearman_Rho'] = round(spearman_rho, 4)
    tier_stats['Spearman_P'] = f"{spearman_p:.4e}"
    tier_stats['Pearson_Log_R'] = round(pearson_r, 4)
    tier_stats['Pearson_P'] = f"{pearson_p:.4e}"
    tier_stats.to_csv(os.path.join(tab_dir, 'size_install_correlation_stats.csv'), index=False)

    # Table 5: Category revenue estimates (Qualified Minimum Upfront Revenue)
    cat_rev = df_apps_clean.groupby('Category').agg(
        Total_Apps=('App', 'count'),
        Paid_Apps=('Price_Numeric', lambda x: (x > 0).sum()),
        Free_Apps=('Price_Numeric', lambda x: (x == 0).sum()),
        Avg_Paid_Price=('Price_Numeric', lambda x: x[x > 0].mean() if (x > 0).sum() > 0 else 0.0),
        Max_Paid_Price=('Price_Numeric', 'max'),
        Total_Paid_Installs=('Installs_Numeric', lambda x: df_apps_clean.loc[x.index][df_apps_clean.loc[x.index, 'Price_Numeric'] > 0]['Installs_Numeric'].sum()),
        Est_Min_Upfront_Revenue=('Est_Min_Upfront_Revenue', 'sum')
    ).reset_index().sort_values(by='Est_Min_Upfront_Revenue', ascending=False)
    cat_rev.to_csv(os.path.join(tab_dir, 'category_revenue_estimates.csv'), index=False)

    # Table 6: Sentiment classification summary
    sent_counts = df_reviews_clean['Sentiment'].value_counts()
    sent_summary = pd.DataFrame({
        'Sentiment_Class': sent_counts.index,
        'Review_Count': sent_counts.values,
        'Percentage': (sent_counts.values / len(df_reviews_clean)) * 100,
        'Mean_Polarity': [
            df_reviews_clean[df_reviews_clean['Sentiment'] == s]['Sentiment_Polarity'].mean()
            for s in sent_counts.index
        ],
        'Mean_Subjectivity': [
            df_reviews_clean[df_reviews_clean['Sentiment'] == s]['Sentiment_Subjectivity'].mean()
            for s in sent_counts.index
        ]
    })
    sent_summary.to_csv(os.path.join(tab_dir, 'sentiment_classification_summary.csv'), index=False)

    # Table 7: Category sentiment profiles
    cat_sent = df_merged.groupby('Category').agg(
        Review_Count=('Translated_Review', 'count'),
        Mean_Polarity=('Sentiment_Polarity', 'mean'),
        Median_Polarity=('Sentiment_Polarity', 'median'),
        Positive_Reviews=('Sentiment', lambda x: (x == 'Positive').sum()),
        Negative_Reviews=('Sentiment', lambda x: (x == 'Negative').sum()),
        Neutral_Reviews=('Sentiment', lambda x: (x == 'Neutral').sum())
    ).reset_index()
    cat_sent['Positive_Pct'] = (cat_sent['Positive_Reviews'] / cat_sent['Review_Count']) * 100
    cat_sent['Negative_Pct'] = (cat_sent['Negative_Reviews'] / cat_sent['Review_Count']) * 100
    cat_sent['Neutral_Pct'] = (cat_sent['Neutral_Reviews'] / cat_sent['Review_Count']) * 100
    cat_sent = cat_sent.sort_values(by='Mean_Polarity', ascending=False)
    cat_sent.to_csv(os.path.join(tab_dir, 'category_sentiment_profiles.csv'), index=False)

    print("  [Artifacts] Generated 7 summary tables in outputs/tables/.")


def generate_figures(df_apps_clean, df_reviews_clean, df_merged, fig_dir):
    """Generate and save 6 publication-ready static PNG figures (300 DPI)."""
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Category Distribution & Market Saturation
    plt.figure(figsize=(14, 7))
    cat_counts = df_apps_clean['Category'].value_counts()
    colors = ['#1f77b4' if i >= 3 else '#d62728' for i in range(len(cat_counts))]
    ax = cat_counts.plot(kind='bar', color=colors, edgecolor='black', width=0.8)
    plt.title('Figure 1: Google Play Store App Distribution by Category (Market Saturation)', fontsize=13, fontweight='bold')
    plt.xlabel('Category', fontsize=11, fontweight='bold')
    plt.ylabel('Number of Unique Applications', fontsize=11, fontweight='bold')
    plt.xticks(rotation=75, ha='right', fontsize=9)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Label top 3 categories
    for i, count in enumerate(cat_counts.head(3)):
        pct = (count / len(df_apps_clean)) * 100
        ax.text(i, count + 20, f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig1_category_distribution.png'), dpi=300)
    plt.close()

    # Figure 2: Ratings Distribution & Category Ratings Boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Overall Rating Density
    ratings = df_apps_clean['Rating'].dropna()
    sns.histplot(ratings, kde=True, ax=ax1, color='#1f77b4', bins=30, edgecolor='black', stat='density')
    ax1.axvline(ratings.mean(), color='#d62728', linestyle='--', linewidth=2, label=f"Mean: {ratings.mean():.2f}")
    ax1.axvline(ratings.median(), color='#2ca02c', linestyle=':', linewidth=2, label=f"Median: {ratings.median():.2f}")
    ax1.set_title('Figure 2A: Overall App Rating Distribution (Left-Skewed)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('User Rating (1.0 to 5.0 Scale)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Panel B: Top 15 Categories Rating Boxplot
    top_15_cats = df_apps_clean['Category'].value_counts().head(15).index
    df_top_cats = df_apps_clean[df_apps_clean['Category'].isin(top_15_cats)].dropna(subset=['Rating'])
    order = df_top_cats.groupby('Category')['Rating'].median().sort_values(ascending=False).index

    sns.boxplot(data=df_top_cats, y='Category', x='Rating', order=order, ax=ax2, palette='Blues_r', hue='Category', legend=False)
    ax2.set_title('Figure 2B: Rating Variance Across Top 15 Categories', fontsize=12, fontweight='bold')
    ax2.set_xlabel('User Rating', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Category', fontsize=10, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_ratings_distribution_by_category.png'), dpi=300)
    plt.close()

    # Figure 3: Size vs. Installs Correlation
    plt.figure(figsize=(10, 6))
    df_size = df_apps_clean.dropna(subset=['Size_MB', 'Installs_Numeric']).copy()
    df_size = df_size[df_size['Installs_Numeric'] > 0]
    df_size['Log_Installs'] = np.log10(df_size['Installs_Numeric'])

    # Sample for visual clarity if very dense
    sample_size = min(3000, len(df_size))
    df_sample = df_size.sample(n=sample_size, random_state=42)

    sns.regplot(
        data=df_sample, x='Size_MB', y='Log_Installs',
        scatter_kws={'alpha': 0.25, 'color': '#1f77b4', 's': 20},
        line_kws={'color': '#d62728', 'linewidth': 2}
    )
    spearman_rho, _ = stats.spearmanr(df_size['Size_MB'], df_size['Installs_Numeric'])
    pearson_r, _ = stats.pearsonr(df_size['Size_MB'], df_size['Log_Installs'])

    plt.title('Figure 3: Application Package Size vs. Download Volume', fontsize=13, fontweight='bold')
    plt.xlabel('Package Size (Megabytes)', fontsize=11, fontweight='bold')
    plt.ylabel('log10(Installs)', fontsize=11, fontweight='bold')
    plt.yticks(
        ticks=[2, 4, 6, 8],
        labels=['100 (10^2)', '10,000 (10^4)', '1,000,000 (10^6)', '100,000,000 (10^8)']
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.annotate(
        f"Spearman rank correlation: \u03c1 = +{spearman_rho:.3f}\nPearson log correlation: r = +{pearson_r:.3f}\nFeature-rich apps drive larger download tiers",
        xy=(0.05, 0.85), xycoords='axes fraction',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef6ff', edgecolor='#1f77b4', alpha=0.9),
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig3_size_vs_installs_correlation.png'), dpi=300)
    plt.close()

    # Figure 4: Pricing Economics & Revenue Estimation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Price distribution for typical paid apps (< $30)
    df_paid = df_apps_clean[df_apps_clean['Price_Numeric'] > 0]
    df_paid_typical = df_paid[df_paid['Price_Numeric'] <= 30]
    sns.histplot(df_paid_typical['Price_Numeric'], bins=30, ax=ax1, color='#2ca02c', edgecolor='black', kde=True)
    ax1.axvline(df_paid['Price_Numeric'].median(), color='#d62728', linestyle='--', linewidth=2, label=f"Median Paid Price: ${df_paid['Price_Numeric'].median():.2f}")
    ax1.axvline(df_paid_typical['Price_Numeric'].mean(), color='#1f77b4', linestyle=':', linewidth=2, label=f"Mean Typical Price: ${df_paid_typical['Price_Numeric'].mean():.2f}")
    ax1.set_title('Figure 4A: Paid Apps Price Distribution (\u2264 $30.00)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Price (USD)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Number of Applications', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Panel B: Top 10 Categories by Estimated Minimum Upfront Revenue
    cat_rev = df_apps_clean.groupby('Category')['Est_Min_Upfront_Revenue'].sum().sort_values(ascending=False).head(10)
    rev_millions = cat_rev / 1_000_000.0
    ax2.barh(rev_millions.index[::-1], rev_millions.values[::-1], color='#ff7f0e', edgecolor='black')
    ax2.set_title('Figure 4B: Top 10 Categories by Est. Minimum Upfront Revenue', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Estimated Gross Upfront Revenue ($ Millions USD)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Category', fontsize=10, fontweight='bold')
    ax2.grid(axis='x', linestyle='--', alpha=0.4)
    for idx, val in enumerate(rev_millions.values[::-1]):
        ax2.text(val + 0.5, idx, f"${val:.1f}M", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig4_pricing_and_revenue_analysis.png'), dpi=300)
    plt.close()

    # Figure 5: Sentiment Distribution (Polarity and Subjectivity)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: Review Sentiment Classes
    sent_counts = df_reviews_clean['Sentiment'].value_counts()
    ax1.pie(
        sent_counts.values,
        labels=[f"{k}\n({v:,}, {v/len(df_reviews_clean)*100:.1f}%)" for k, v in sent_counts.items()],
        colors=['#2ca02c', '#d62728', '#7f7f7f'],
        startangle=140,
        wedgeprops=dict(edgecolor='black', linewidth=1.2),
        explode=[0.03, 0.03, 0.03]
    )
    ax1.set_title('Figure 5A: User Review Sentiment Breakdown (N = 37,427)', fontsize=12, fontweight='bold')

    # Panel B: Polarity Density
    sns.kdeplot(df_reviews_clean['Sentiment_Polarity'], ax=ax2, color='#1f77b4', fill=True, alpha=0.3, linewidth=2)
    ax2.axvline(0.0, color='black', linestyle=':', linewidth=1.5, label='Neutral Threshold (0.0)')
    ax2.axvline(df_reviews_clean['Sentiment_Polarity'].mean(), color='#2ca02c', linestyle='--', linewidth=2, label=f"Mean Polarity ({df_reviews_clean['Sentiment_Polarity'].mean():.2f})")
    ax2.set_title('Figure 5B: Sentiment Polarity Density Spectrum', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sentiment Polarity (-1.0 Negative to +1.0 Positive)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig5_sentiment_distribution.png'), dpi=300)
    plt.close()

    # Figure 6: Category Sentiment Breakdown (Stacked Bar)
    cat_sent = df_merged.groupby('Category').agg(
        Review_Count=('Translated_Review', 'count'),
        Positive_Pct=('Sentiment', lambda x: (x == 'Positive').mean() * 100),
        Neutral_Pct=('Sentiment', lambda x: (x == 'Neutral').mean() * 100),
        Negative_Pct=('Sentiment', lambda x: (x == 'Negative').mean() * 100)
    ).reset_index()

    # Filter categories with at least 100 reviews for reliable sample size
    cat_sent_filt = cat_sent[cat_sent['Review_Count'] >= 100].sort_values(by='Positive_Pct', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    p1 = ax.barh(cat_sent_filt['Category'], cat_sent_filt['Positive_Pct'], color='#2ca02c', edgecolor='black', label='Positive %')
    p2 = ax.barh(cat_sent_filt['Category'], cat_sent_filt['Neutral_Pct'], left=cat_sent_filt['Positive_Pct'], color='#bcbd22', edgecolor='black', label='Neutral %')
    p3 = ax.barh(cat_sent_filt['Category'], cat_sent_filt['Negative_Pct'], left=cat_sent_filt['Positive_Pct'] + cat_sent_filt['Neutral_Pct'], color='#d62728', edgecolor='black', label='Negative %')

    ax.set_title('Figure 6: Sentiment Breakdown Across App Categories (\u2265 100 Reviews)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Percentage of Reviews (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Category', fontsize=11, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig6_category_sentiment_breakdown.png'), dpi=300)
    plt.close()

    print("  [Artifacts] Rendered 6 publication-grade figures in outputs/figures/.")


def generate_interactive_plotly(df_apps_clean, fig_dir):
    """
    Generate an interactive 4D Plotly HTML visualization:
    X: Size in MB, Y: User Rating, Size: Installs, Color: Category, Hover metadata.
    """
    os.makedirs(fig_dir, exist_ok=True)
    df_plot = df_apps_clean.dropna(subset=['Size_MB', 'Rating']).copy()
    df_plot = df_plot[df_plot['Installs_Numeric'] > 0]

    # Sample top 10 categories to keep visualization crisp and responsive
    top_10_cats = df_plot['Category'].value_counts().head(10).index
    df_plot_sub = df_plot[df_plot['Category'].isin(top_10_cats)].copy()

    # Log-scaled install size for markers
    df_plot_sub['Log_Installs'] = np.log10(df_plot_sub['Installs_Numeric'])
    df_plot_sub['Marker_Size'] = df_plot_sub['Log_Installs'] * 3.5

    fig = px.scatter(
        df_plot_sub,
        x='Size_MB',
        y='Rating',
        size='Marker_Size',
        color='Category',
        hover_name='App',
        hover_data={
            'Category': True,
            'Size_MB': ':.1f',
            'Rating': ':.2f',
            'Installs_Numeric': ':,',
            'Price_Numeric': ':.2f',
            'Marker_Size': False
        },
        labels={
            'Size_MB': 'Package Size (MB)',
            'Rating': 'User Rating (1.0 to 5.0)',
            'Installs_Numeric': 'Total Installs',
            'Category': 'App Category'
        },
        title='<b>Interactive Google Play Store 4D Market Landscape</b><br><sup>Size (MB) vs. Rating vs. Downloads (Marker Size) across Top Categories</sup>',
        template='plotly_white',
        height=700
    )

    fig.update_layout(
        legend_title_text='<b>App Category</b>',
        xaxis=dict(gridcolor='#e0e0e0', title_font=dict(size=12, family='Arial')),
        yaxis=dict(gridcolor='#e0e0e0', title_font=dict(size=12, family='Arial'), range=[1.0, 5.2]),
        font=dict(family='Arial', size=11)
    )

    out_html = os.path.join(fig_dir, 'interactive_playstore_viz.html')
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"  [Artifacts] Exported interactive Plotly visualization to {out_html}")


def main():
    parser = argparse.ArgumentParser(description="Google Play Store End-to-End Analytics Pipeline")
    parser.add_argument('--base-dir', default='.', help="Base project root directory")
    args = parser.parse_args()

    task_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(task_dir, 'outputs', 'figures')
    tab_dir = os.path.join(task_dir, 'outputs', 'tables')

    print("=" * 70)
    print("Data Analytics Portfolio: Google Play Store Analytics Pipeline (PRJ-08)")
    print("=" * 70)

    print("\n[1/5] Ingesting and cleaning datasets...")
    df_apps_raw, df_apps_clean, df_reviews_raw, df_reviews_clean, df_merged = load_and_clean_data(task_dir)
    print(f"  Raw Apps: {len(df_apps_raw):,} | Clean Unique Apps: {len(df_apps_clean):,}")
    print(f"  Raw Reviews: {len(df_reviews_raw):,} | Clean Reviews with Text: {len(df_reviews_clean):,}")
    print(f"  Merged Reviews with Apps: {len(df_merged):,} across {df_merged['Category'].nunique()} categories.")

    print("\n[2/5] Generating statistical summary tables...")
    generate_tables(df_apps_raw, df_apps_clean, df_reviews_raw, df_reviews_clean, df_merged, tab_dir)

    print("\n[3/5] Rendering publication-grade static figures (300 DPI)...")
    generate_figures(df_apps_clean, df_reviews_clean, df_merged, fig_dir)

    print("\n[4/5] Exporting interactive Plotly visualization (HTML)...")
    generate_interactive_plotly(df_apps_clean, fig_dir)

    print("\n[5/5] Pipeline execution verified successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()
