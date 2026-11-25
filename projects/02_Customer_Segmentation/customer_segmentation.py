"""
customer_segmentation.py
Reproducible Customer Segmentation & RFM K-Means Clustering Pipeline
Project 02 (Customer Segmentation) - Customer Segmentation (Data Analytics Portfolio portfolio development)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

def run_segmentation_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.join(base_dir, 'data', 'raw_online_retail.csv')
    segmented_path = os.path.join(base_dir, 'data', 'rfm_customer_segments.csv')
    fig_dir = os.path.join(base_dir, 'outputs', 'figures')
    tab_dir = os.path.join(base_dir, 'outputs', 'tables')
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    
    print("[1/6] Ingesting raw Online Retail dataset...")
    df_raw = pd.read_csv(raw_path, encoding='latin1')
    print(f"Raw shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
    
    # ---------------------------------------------------------
    # 1. Data Cleaning & Transaction Filtering
    # ---------------------------------------------------------
    print("[2/6] Cleaning transactional data and filtering valid purchases...")
    # Drop records without CustomerID (guest checkouts cannot be assigned to customer entities)
    n_missing_cust = df_raw['CustomerID'].isnull().sum()
    print(f"Transactions missing CustomerID: {n_missing_cust:,} ({n_missing_cust/len(df_raw)*100:.2f}%)")
    
    # Valid completed purchases: non-null CustomerID, positive quantity & unit price, non-cancelled invoice
    valid_mask = (
        df_raw['CustomerID'].notnull() &
        (df_raw['Quantity'] > 0) &
        (df_raw['UnitPrice'] > 0) &
        (~df_raw['InvoiceNo'].astype(str).str.startswith('C'))
    )
    df_clean = df_raw[valid_mask].copy()
    
    # Parse timestamps
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int).astype(str)
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    print(f"Clean valid transactions retained: {len(df_clean):,} ({len(df_clean)/len(df_raw)*100:.2f}%)")
    print(f"Unique customers: {df_clean['CustomerID'].nunique():,}")
    print(f"Transaction date range: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")

    # ---------------------------------------------------------
    # 2. RFM Feature Engineering
    # ---------------------------------------------------------
    print("[3/6] Engineering Recency, Frequency, and Monetary (RFM) features...")
    ref_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda d: (ref_date - d.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    })
    
    # Additional behavioral descriptors
    rfm['Avg_Order_Value'] = (rfm['Monetary'] / rfm['Frequency']).round(2)
    rfm['Recency'] = rfm['Recency'].astype(int)
    rfm['Frequency'] = rfm['Frequency'].astype(int)
    rfm['Monetary'] = rfm['Monetary'].round(2)
    
    # Descriptive statistics table
    rfm_desc = rfm[['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value']].describe().round(2)
    rfm_desc.to_csv(os.path.join(tab_dir, 'rfm_summary_statistics.csv'))
    print("Saved rfm_summary_statistics.csv")
    
    # Figure 1: RFM Distributions (Raw vs Log)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    metrics = ['Recency', 'Frequency', 'Monetary']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, col in enumerate(metrics):
        sns.histplot(rfm[col], kde=True, ax=axes[0, i], color=colors[i], bins=30)
        axes[0, i].set_title(f'Raw {col} Distribution', fontweight='bold')
        axes[0, i].set_ylabel('Customer Count')
        
        # Log-transformed
        sns.histplot(np.log1p(rfm[col]), kde=True, ax=axes[1, i], color=colors[i], bins=30)
        axes[1, i].set_title(f'Log-Transformed log1p({col})', fontweight='bold')
        axes[1, i].set_ylabel('Customer Count')
        
    fig.suptitle('Figure 1: RFM Behavioral Metric Distributions — Raw Skewness vs. Log Transformation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig1_rfm_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Figure 1.")

    # ---------------------------------------------------------
    # 3. Normalization & Clustering Optimization (Elbow & Silhouette)
    # ---------------------------------------------------------
    print("[4/6] Normalizing features and evaluating optimal cluster count K...")
    # Log transform to manage extreme skewness
    rfm_log = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # StandardScaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    k_range = range(2, 9)
    inertias = []
    silhouettes = []
    
    eval_records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        inertia = km.inertia_
        sil = silhouette_score(rfm_scaled, labels)
        inertias.append(inertia)
        silhouettes.append(sil)
        eval_records.append({
            'K_Clusters': k,
            'Inertia_SSE': round(inertia, 2),
            'Silhouette_Score': round(sil, 4)
        })
        
    df_eval = pd.DataFrame(eval_records)
    df_eval.to_csv(os.path.join(tab_dir, 'cluster_evaluation_metrics.csv'), index=False)
    print("Saved cluster_evaluation_metrics.csv")
    
    # Figure 2: Elbow & Silhouette Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = '#2980b9'
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (Sum of Squared Errors)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(k_range, inertias, marker='o', color=color, linewidth=2.5, label='Inertia (SSE)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axvline(x=4, color='#e74c3c', linestyle='--', linewidth=1.5, label='Optimal K=4 Selected')
    
    ax2 = ax1.twinx()
    color2 = '#27ae60'
    ax2.set_ylabel('Silhouette Score', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(k_range, silhouettes, marker='s', color=color2, linewidth=2.5, linestyle=':', label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines = line1 + line2 + [ax1.get_lines()[-1]]
    labels_plot = [l.get_label() for l in lines]
    ax1.legend(lines, labels_plot, loc='upper right', frameon=True)
    
    plt.title('Figure 2: K-Means Optimization — Elbow Curve and Silhouette Coefficient (K=2 to 8)', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_elbow_and_silhouette.png'), dpi=300)
    plt.close()
    print("Generated Figure 2.")

    # ---------------------------------------------------------
    # 4. K-Means Model Fitting (K=4) & Persona Mapping
    # ---------------------------------------------------------
    print("[5/6] Fitting K-Means model (K=4) and assigning customer personas...")
    kmeans_opt = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans_opt.fit_predict(rfm_scaled)
    
    # Map cluster index to business persona based on empirical centroid characteristics
    # Calculate centroids to map dynamically
    centroids = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Identify clusters:
    # Champions: Lowest Recency + Highest Monetary & Frequency
    # At-Risk: Highest Recency + Low Frequency & Monetary
    # Regulars: Mid-low Recency + Healthy Frequency & Monetary
    # Promising: Low Recency + Low Frequency (recently transacted, 1-2 orders)
    
    champ_id = centroids['Monetary'].idxmax()
    at_risk_id = centroids['Recency'].idxmax()
    remaining = [c for c in centroids.index if c not in [champ_id, at_risk_id]]
    
    # Between the remaining two: the one with lower recency is New/Promising, other is Loyal Regulars
    if centroids.loc[remaining[0], 'Recency'] < centroids.loc[remaining[1], 'Recency']:
        promising_id = remaining[0]
        loyal_id = remaining[1]
    else:
        promising_id = remaining[1]
        loyal_id = remaining[0]
        
    persona_mapping = {
        champ_id: 'Champions / High-Value VIPs',
        loyal_id: 'Loyal & Consistent Regulars',
        promising_id: 'Recent / New Promising Customers',
        at_risk_id: 'Hibernating / Churned At-Risk Customers'
    }
    
    rfm['Persona'] = rfm['Cluster'].map(persona_mapping)
    
    # Export segmented dataset
    rfm.to_csv(segmented_path)
    print(f"Exported segmented customer dataset to: {segmented_path}")
    
    # Cluster Persona Profiles Table
    profiles = rfm.groupby('Persona').agg({
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median', 'sum'],
        'Avg_Order_Value': ['mean', 'median'],
        'Cluster': 'count'
    }).rename(columns={'count': 'Customer_Count'})
    
    # Flatten multi-index
    profiles.columns = ['_'.join(c).strip() for c in profiles.columns.values]
    total_rev = rfm['Monetary'].sum()
    profiles['Revenue_Share_Pct'] = (profiles['Monetary_sum'] / total_rev * 100).round(2)
    profiles['Customer_Share_Pct'] = (profiles['Cluster_Customer_Count'] / len(rfm) * 100).round(2)
    profiles = profiles.round(2)
    profiles.to_csv(os.path.join(tab_dir, 'cluster_persona_profiles.csv'))
    print("Saved cluster_persona_profiles.csv")
    
    # Strategic Marketing Recommendations Matrix (Proposed Hypotheses)
    marketing_matrix = pd.DataFrame([
        {
            'Persona': 'Champions / High-Value VIPs',
            'Customer_Count': profiles.loc['Champions / High-Value VIPs', 'Cluster_Customer_Count'],
            'Revenue_Share': f"{profiles.loc['Champions / High-Value VIPs', 'Revenue_Share_Pct']}%",
            'Characteristics': 'Ultra-recent, high frequency, highest spending',
            'Key_Objective': 'Retain and reward loyalty; prevent defection to competitors',
            'Proposed_Marketing_Strategy': 'VIP concierge support, exclusive reward previews, early product access, personalized appreciation gifts'
        },
        {
            'Persona': 'Loyal & Consistent Regulars',
            'Customer_Count': profiles.loc['Loyal & Consistent Regulars', 'Cluster_Customer_Count'],
            'Revenue_Share': f"{profiles.loc['Loyal & Consistent Regulars', 'Revenue_Share_Pct']}%",
            'Characteristics': 'Steady purchase history, moderate-to-high spending, consistent cadence',
            'Key_Objective': 'Upsell and cross-sell into higher-tier spending; elevate to VIP status',
            'Proposed_Marketing_Strategy': 'Multi-item bundle discounts, automated replenishment triggers, category cross-sell recommendations'
        },
        {
            'Persona': 'Recent / New Promising Customers',
            'Customer_Count': profiles.loc['Recent / New Promising Customers', 'Cluster_Customer_Count'],
            'Revenue_Share': f"{profiles.loc['Recent / New Promising Customers', 'Revenue_Share_Pct']}%",
            'Characteristics': 'Transacted very recently but have only 1-2 orders; low-to-moderate spend',
            'Key_Objective': 'Cultivate habit and secure second/third purchases quickly',
            'Proposed_Marketing_Strategy': 'Welcome onboarding drip campaign, post-purchase satisfaction surveys, 15% discount on second purchase within 30 days'
        },
        {
            'Persona': 'Hibernating / Churned At-Risk Customers',
            'Customer_Count': profiles.loc['Hibernating / Churned At-Risk Customers', 'Cluster_Customer_Count'],
            'Revenue_Share': f"{profiles.loc['Hibernating / Churned At-Risk Customers', 'Revenue_Share_Pct']}%",
            'Characteristics': 'Dormant (>6 months inactivity), low frequency, one-time purchasers',
            'Key_Objective': 'Win back reachable buyers; prune unengaged contacts to reduce marketing spend',
            'Proposed_Marketing_Strategy': 'High-incentive comeback discounts ("We miss you - $20 off"), clearance sale alerts, automated feedback inquiry'
        }
    ])
    marketing_matrix.to_csv(os.path.join(tab_dir, 'customer_segment_marketing_matrix.csv'), index=False)
    print("Saved customer_segment_marketing_matrix.csv")

    # ---------------------------------------------------------
    # 5. Visualizations: Scatter Plots & Segment Counts
    # ---------------------------------------------------------
    print("[6/6] Rendering cluster visualizations...")
    # Figure 3: 2D Scatter Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    palette = {'Champions / High-Value VIPs': '#e74c3c',
               'Loyal & Consistent Regulars': '#2ecc71',
               'Recent / New Promising Customers': '#3498db',
               'Hibernating / Churned At-Risk Customers': '#95a5a6'}
    
    # Plot 1: Recency vs Monetary (Log Scale)
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Persona', palette=palette, alpha=0.7, ax=axes[0], s=35)
    axes[0].set_yscale('log')
    axes[0].set_title('Recency vs. Monetary Value (Log Scale)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Recency (Days Since Last Order)', fontsize=11)
    axes[0].set_ylabel('Monetary Value (£, Log Scale)', fontsize=11)
    axes[0].legend(title='Customer Persona', loc='upper right', fontsize=9)
    
    # Plot 2: Frequency vs Monetary (Log Scale)
    sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='Persona', palette=palette, alpha=0.7, ax=axes[1], s=35)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title('Frequency vs. Monetary Value (Log-Log Scale)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Frequency (Distinct Invoices, Log Scale)', fontsize=11)
    axes[1].set_ylabel('Monetary Value (£, Log Scale)', fontsize=11)
    axes[1].legend(title='Customer Persona', loc='upper left', fontsize=9)
    
    fig.suptitle('Figure 3: 2-Dimensional Cluster Projections Across Key Behavioral Combinations', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig3_cluster_scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Figure 3.")

    # Figure 4: Customer Count per Cluster Bar Chart
    plt.figure(figsize=(10, 5))
    counts = rfm['Persona'].value_counts()
    bars = plt.barh(counts.index, counts.values, color=['#95a5a6', '#2ecc71', '#3498db', '#e74c3c'], edgecolor='black', alpha=0.85)
    plt.title('Figure 4: Customer Distribution per Segment (K=4)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Number of Customers', fontsize=11, fontweight='bold')
    plt.xlim(0, counts.max() * 1.15)
    for bar in bars:
        w = bar.get_width()
        pct = (w / len(rfm)) * 100
        plt.text(w + 25, bar.get_y() + bar.get_height()/2, f"{int(w):,} ({pct:.1f}%)", va='center', ha='left', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig4_cluster_customer_counts.png'), dpi=300)
    plt.close()
    print("Generated Figure 4.")

    print(f"\nALL CUSTOMER SEGMENTATION PIPELINE STAGES COMPLETED.")
    print(f"Segmented {len(rfm):,} customers across 4 distinct personas.")

if __name__ == '__main__':
    run_segmentation_pipeline()
