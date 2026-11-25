"""
Wine Quality Prediction Pipeline
=================================
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — Project 06 (Wine Quality Prediction)
Author: Dhruv Goyal
Date: October 2026

Description:
    End-to-end data processing, exploratory analysis, class-imbalance diagnostic,
    duplicate record and train-test boundary leakage audit, leakage-free machine
    learning pipelines, and multi-model benchmark evaluation for predicting wine
    sensory quality ratings from physicochemical properties on the canonical
    UCI Machine Learning Repository Wine Quality benchmark (Cortez et al., 2009).
"""

import os
import sys
import argparse
import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance

# Set plotting style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('WineQualityPipeline')


FEATURE_COLUMNS = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol'
]
TARGET_COLUMN = 'quality'
BINARY_TARGET = 'good_wine'


def load_and_inspect_dataset(data_path: str) -> pd.DataFrame:
    """Load wine quality dataset from disk and perform structural verification."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    df = pd.read_csv(data_path, sep=';')
    logger.info("Loaded dataset from %s with shape %s", data_path, df.shape)

    missing_cols = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    null_count = df.isnull().sum().sum()
    dup_count = df.duplicated().sum()
    logger.info("Dataset validation: Total Nulls = %d, Duplicates = %d", null_count, dup_count)
    return df


def generate_exploratory_tables(df: pd.DataFrame, tables_dir: str) -> Dict[str, pd.DataFrame]:
    """Calculate and export dataset profiling and descriptive statistics."""
    os.makedirs(tables_dir, exist_ok=True)

    summary_data = {
        'Metric': [
            'Total Observations',
            'Unique Observations',
            'Duplicate Records',
            'Physicochemical Features',
            'Target Feature',
            'Missing Values',
            'Min Quality Score',
            'Max Quality Score',
            'Mean Quality Score',
            'Median Quality Score',
            'Std Dev Quality Score'
        ],
        'Value': [
            str(len(df)),
            str(len(df.drop_duplicates())),
            str(df.duplicated().sum()),
            str(len(FEATURE_COLUMNS)),
            TARGET_COLUMN,
            str(df.isnull().sum().sum()),
            str(df[TARGET_COLUMN].min()),
            str(df[TARGET_COLUMN].max()),
            f"{df[TARGET_COLUMN].mean():.3f}",
            f"{df[TARGET_COLUMN].median():.1f}",
            f"{df[TARGET_COLUMN].std():.3f}"
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(tables_dir, 'wine_dataset_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    stats_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T
    stats_df['median'] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].median()
    stats_df['skewness'] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].skew()
    stats_df['kurtosis'] = df[FEATURE_COLUMNS + [TARGET_COLUMN]].kurtosis()
    stats_path = os.path.join(tables_dir, 'physicochemical_descriptive_stats.csv')
    stats_df.to_csv(stats_path)

    logger.info("Exported summary tables to %s", tables_dir)
    return {'summary': summary_df, 'stats': stats_df}


def audit_duplicates_and_boundary_leakage(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tables_dir: str
) -> pd.DataFrame:
    """Audit duplicate records and quantify train-test boundary leakage."""
    os.makedirs(tables_dir, exist_ok=True)

    total_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())
    unique_rows = total_rows - exact_duplicates
    duplicate_cluster_rows = int(df.duplicated(keep=False).sum())

    # Check for conflicting targets on identical feature vectors
    feat_dups = df[df.duplicated(subset=FEATURE_COLUMNS, keep=False)]
    conflicting_groups = 0
    if len(feat_dups) > 0:
        conflicting_groups = int((feat_dups.groupby(FEATURE_COLUMNS)[TARGET_COLUMN].nunique() > 1).sum())

    # Train/Test boundary overlap audit
    train_tuples = set(tuple(x) for x in X_train[FEATURE_COLUMNS].values)
    test_tuples = [tuple(x) for x in X_test[FEATURE_COLUMNS].values]

    cross_boundary_test_rows = sum(1 for t in test_tuples if t in train_tuples)
    cross_boundary_pct = (cross_boundary_test_rows / len(X_test)) * 100.0

    audit_data = {
        'Audit_Metric': [
            'Total Raw Observations',
            'Unique Observations',
            'Exact Duplicate Rows',
            'Rows Belonging to Duplicate Groups',
            'Conflicting Target Labels in Duplicate Groups',
            'Total Test Split Observations',
            'Test Rows Identical to a Training Row (Boundary Leakage)',
            'Boundary Leakage Rate in Test Set (%)',
            'Generalization Risk Assessment'
        ],
        'Value': [
            str(total_rows),
            str(unique_rows),
            str(exact_duplicates),
            str(duplicate_cluster_rows),
            str(conflicting_groups),
            str(len(X_test)),
            str(cross_boundary_test_rows),
            f"{cross_boundary_pct:.2f}%",
            "High memorization risk on standard split; duplicate-aware evaluation required"
        ]
    }
    audit_df = pd.DataFrame(audit_data)
    audit_path = os.path.join(tables_dir, 'duplicate_leakage_audit.csv')
    audit_df.to_csv(audit_path, index=False)
    logger.info("Saved duplicate leakage audit table to %s", audit_path)
    return audit_df


def create_binary_target(df: pd.DataFrame, quality_threshold: int = 7) -> pd.DataFrame:
    """Create binary target where quality >= threshold is classified as Higher-Rated (1)."""
    df_clean = df.copy()
    df_clean[BINARY_TARGET] = (df_clean[TARGET_COLUMN] >= quality_threshold).astype(int)
    counts = df_clean[BINARY_TARGET].value_counts().to_dict()
    logger.info(
        "Binary target '%s' created (Threshold >= %d): 0 (Score < %d) = %d (%.2f%%), 1 (Score >= %d) = %d (%.2f%%)",
        BINARY_TARGET,
        quality_threshold,
        quality_threshold,
        counts.get(0, 0),
        100.0 * counts.get(0, 0) / len(df_clean),
        quality_threshold,
        counts.get(1, 0),
        100.0 * counts.get(1, 0) / len(df_clean)
    )
    return df_clean


def plot_distributions(df: pd.DataFrame, figures_dir: str) -> str:
    """Plot distribution histograms and KDE curves for all 11 physicochemical features."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLUMNS):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#2b5c8f', edgecolor='white', alpha=0.6)
        axes[i].set_title(col.title(), fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Count')

    fig.delaxes(axes[11])
    plt.suptitle('Physicochemical Feature Distributions (UCI Red Wine Quality)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, 'fig1_wine_distributions.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure: %s", out_path)
    return out_path


def plot_quality_and_class_imbalance(df: pd.DataFrame, figures_dir: str) -> str:
    """Plot raw quality score distribution vs engineered binary target distribution."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    raw_counts = df[TARGET_COLUMN].value_counts().sort_index()
    sns.barplot(x=raw_counts.index, y=raw_counts.values, ax=axes[0], color='#2b5c8f', edgecolor='black')
    axes[0].set_title('Raw Sensory Quality Score Distribution (3 to 8)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Sensory Quality Score')
    axes[0].set_ylabel('Wine Sample Count')
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f'{int(height)}\n({height/len(df)*100:.1f}%)',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    binary_counts = df[BINARY_TARGET].value_counts().sort_index()
    labels = ['Score < 7 (Typical / Low)', 'Score >= 7 (Higher-Rated)']
    colors = ['#4a7bb0', '#d95f02']
    sns.barplot(x=labels, y=binary_counts.values, ax=axes[1], palette=colors, edgecolor='black')
    axes[1].set_title('Engineered Binary Target Distribution (Threshold >= 7)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Analytical Target Tier')
    axes[1].set_ylabel('Wine Sample Count')
    for p in axes[1].patches:
        height = p.get_height()
        axes[1].annotate(f'{int(height)}\n({height/len(df)*100:.1f}%)',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')

    plt.suptitle('Class Distribution and Imbalance Diagnostic', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, 'fig2_quality_distribution.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure: %s", out_path)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, figures_dir: str) -> str:
    """Plot pairwise Pearson correlation heatmap across all physicochemical features and quality."""
    os.makedirs(figures_dir, exist_ok=True)
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.8,
        vmin=-0.8,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .8, 'label': 'Pearson Correlation Coefficient (r)'},
        ax=ax
    )
    plt.title('Physicochemical Correlation Matrix & Sensory Score Association', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, 'fig3_correlation_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure: %s", out_path)
    return out_path


def build_models(random_state: int = 42) -> Dict[str, Any]:
    """Build baseline (most_frequent) and 3 official classifiers wrapped in leakage-free pipelines."""
    models = {
        'Dummy (most_frequent)': DummyClassifier(strategy='most_frequent'),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1))
        ]),
        'SGD Classifier': Pipeline([
            ('scaler', StandardScaler()),
            ('sgd', SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=1e-4,
                max_iter=2000,
                random_state=random_state,
                class_weight='balanced'
            ))
        ]),
        'Support Vector (SVC)': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=random_state,
                class_weight='balanced'
            ))
        ])
    }
    return models


def evaluate_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tables_dir: str,
    models_dir: str,
    output_filename: str = 'model_performance_comparison.csv'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train models, evaluate on held-out test split and 5-fold CV, and save artifacts."""
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    rows = []
    trained_artifacts = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        logger.info("Training and evaluating: %s", name)

        if 'Dummy' in name:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = None

            acc = float(accuracy_score(y_test, y_pred))
            bacc = float(balanced_accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
            weighted_f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            rows.append({
                'Model': name,
                'Accuracy': round(acc, 4),
                'Balanced_Accuracy': round(bacc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1_Score': round(f1, 4),
                'Macro_F1': round(macro_f1, 4),
                'Weighted_F1': round(weighted_f1, 4),
                'ROC_AUC': np.nan,
                'PR_AUC': np.nan,
                'CV_ROC_AUC_Mean': np.nan,
                'CV_ROC_AUC_Std': np.nan,
                'TP': int(tp),
                'FP': int(fp),
                'TN': int(tn),
                'FN': int(fn)
            })
            trained_artifacts[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_prob': None,
                'cm': cm
            }
        else:
            # 5-fold CV strictly on training data
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # Train on full training split
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                dec = model.decision_function(X_test)
                y_prob = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
            else:
                y_prob = np.zeros_like(y_pred, dtype=float)

            acc = float(accuracy_score(y_test, y_pred))
            bacc = float(balanced_accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
            weighted_f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None and len(np.unique(y_prob)) > 1 else np.nan
            ap = float(average_precision_score(y_test, y_prob)) if y_prob is not None and len(np.unique(y_prob)) > 1 else np.nan

            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            rows.append({
                'Model': name,
                'Accuracy': round(acc, 4),
                'Balanced_Accuracy': round(bacc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1_Score': round(f1, 4),
                'Macro_F1': round(macro_f1, 4),
                'Weighted_F1': round(weighted_f1, 4),
                'ROC_AUC': round(auc, 4),
                'PR_AUC': round(ap, 4),
                'CV_ROC_AUC_Mean': round(cv_mean, 4),
                'CV_ROC_AUC_Std': round(cv_std, 4),
                'TP': int(tp),
                'FP': int(fp),
                'TN': int(tn),
                'FN': int(fn)
            })

            trained_artifacts[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'cm': cm
            }

            # Save serialized model artifact
            slug = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            save_path = os.path.join(models_dir, f'wine_quality_{slug}_pipeline.joblib')
            joblib.dump(model, save_path)
            logger.info("Saved serialized model pipeline: %s", save_path)

    comparison_df = pd.DataFrame(rows)
    comp_path = os.path.join(tables_dir, output_filename)
    comparison_df.to_csv(comp_path, index=False)
    logger.info("Saved model comparison table to %s", comp_path)

    return comparison_df, trained_artifacts


def plot_confusion_matrices(trained_artifacts: Dict[str, Any], figures_dir: str) -> str:
    """Plot side-by-side confusion matrices for Random Forest, SGD, and SVC."""
    os.makedirs(figures_dir, exist_ok=True)
    models_to_plot = ['Random Forest', 'SGD Classifier', 'Support Vector (SVC)']

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for i, name in enumerate(models_to_plot):
        data = trained_artifacts[name]
        cm = data['cm']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        annot = np.array([
            [f"{cm[0, 0]}\n({cm_norm[0, 0]*100:.1f}%)", f"{cm[0, 1]}\n({cm_norm[0, 1]*100:.1f}%)"],
            [f"{cm[1, 0]}\n({cm_norm[1, 0]*100:.1f}%)", f"{cm[1, 1]}\n({cm_norm[1, 1]*100:.1f}%)"]
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            cmap='Blues',
            cbar=False,
            ax=axes[i],
            xticklabels=['Score <7', 'Score >=7'],
            yticklabels=['Score <7', 'Score >=7']
        )
        axes[i].set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')

    plt.suptitle('Held-Out Test Set Confusion Matrices (N = 320, Standard Split)', fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, 'fig4_confusion_matrices.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure: %s", out_path)
    return out_path


def plot_roc_pr_curves(trained_artifacts: Dict[str, Any], y_test: pd.Series, figures_dir: str) -> str:
    """Plot ROC and Precision-Recall curves with honest reference lines."""
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = {
        'Random Forest': '#2b5c8f',
        'SGD Classifier': '#d95f02',
        'Support Vector (SVC)': '#7570b3'
    }

    # 1. ROC Curves
    for name, data in trained_artifacts.items():
        if 'Dummy' in name:
            continue
        y_prob = data.get('y_prob')
        if y_prob is not None and len(np.unique(y_prob)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colors.get(name, '#333333'), lw=2.2)

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Chance Baseline (AUC = 0.50)')
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].legend(loc='lower right', frameon=True)

    # 2. Precision-Recall Curves
    baseline_ap = y_test.mean()
    axes[1].axhline(y=baseline_ap, color='k', linestyle='--', lw=1.5, alpha=0.7, label=f"Uninformative Prevalence Baseline ({baseline_ap*100:.1f}%)")

    for name, data in trained_artifacts.items():
        if 'Dummy' in name:
            continue
        y_prob = data.get('y_prob')
        if y_prob is not None and len(np.unique(y_prob)) > 1:
            prec, rec, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {ap:.3f})", color=colors.get(name, '#333333'), lw=2.2)

    axes[1].set_title('Precision-Recall (PR) Curves', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc='upper right', frameon=True)

    plt.suptitle('Diagnostic Discrimination Curves on Held-Out Test Set (Standard Split)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, 'fig5_roc_pr_curves.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure: %s", out_path)
    return out_path


def analyze_feature_importance(
    rf_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tables_dir: str,
    figures_dir: str
) -> pd.DataFrame:
    """Extract and plot Random Forest MDI Gini Importance and Permutation Importance."""
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    rf_model = rf_pipeline.named_steps['rf']
    gini_imp = rf_model.feature_importances_

    perm = permutation_importance(rf_pipeline, X_test, y_test, n_repeats=20, random_state=42, scoring='roc_auc')

    imp_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Gini_Importance': gini_imp,
        'Permutation_Importance_Mean': perm.importances_mean,
        'Permutation_Importance_Std': perm.importances_std
    }).sort_values(by='Gini_Importance', ascending=False).reset_index(drop=True)

    out_csv = os.path.join(tables_dir, 'random_forest_feature_importance.csv')
    imp_df.to_csv(out_csv, index=False)
    logger.info("Saved feature importance table to %s", out_csv)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.barplot(data=imp_df, x='Gini_Importance', y='Feature', ax=axes[0], color='#2b5c8f', edgecolor='black')
    axes[0].set_title('Random Forest MDI Gini Importance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Mean Decrease in Impurity')
    axes[0].set_ylabel('')

    perm_sorted = imp_df.sort_values(by='Permutation_Importance_Mean', ascending=False)
    axes[1].barh(
        perm_sorted['Feature'],
        perm_sorted['Permutation_Importance_Mean'],
        xerr=perm_sorted['Permutation_Importance_Std'],
        color='#4a7bb0',
        edgecolor='black',
        capsize=3
    )
    axes[1].invert_yaxis()
    axes[1].set_title('Held-Out Test Permutation Importance (ROC-AUC Drop)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mean ROC-AUC Drop When Shuffled')
    axes[1].set_ylabel('')

    plt.suptitle('Physicochemical Predictive Association Diagnostic (Random Forest)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_fig = os.path.join(figures_dir, 'fig6_feature_importance.png')
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure: %s", out_fig)

    return imp_df


def perform_misclassification_analysis(
    rf_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    raw_df: pd.DataFrame,
    tables_dir: str
) -> pd.DataFrame:
    """Audit false positives and false negatives against class median values."""
    os.makedirs(tables_dir, exist_ok=True)
    y_pred = rf_pipeline.predict(X_test)
    y_prob = rf_pipeline.predict_proba(X_test)[:, 1]

    test_indices = X_test.index
    analysis_df = X_test.copy()
    analysis_df['actual_quality'] = raw_df.loc[test_indices, TARGET_COLUMN]
    analysis_df['actual_good_wine'] = y_test.values
    analysis_df['predicted_good_wine'] = y_pred
    analysis_df['predicted_prob_good'] = np.round(y_prob, 4)

    def outcome(row):
        if row['actual_good_wine'] == 1 and row['predicted_good_wine'] == 1:
            return 'True Positive'
        elif row['actual_good_wine'] == 0 and row['predicted_good_wine'] == 0:
            return 'True Negative'
        elif row['actual_good_wine'] == 0 and row['predicted_good_wine'] == 1:
            return 'False Positive'
        else:
            return 'False Negative'

    analysis_df['prediction_outcome'] = analysis_df.apply(outcome, axis=1)

    errors_df = analysis_df[analysis_df['prediction_outcome'].isin(['False Positive', 'False Negative'])].copy()
    errors_df = errors_df.sort_values(by='predicted_prob_good', ascending=False)

    out_csv = os.path.join(tables_dir, 'misclassification_analysis.csv')
    errors_df.to_csv(out_csv, index=True)
    logger.info("Exported %d misclassification records to %s", len(errors_df), out_csv)
    return errors_df


def run_pipeline(
    data_path: str,
    output_dir: str,
    random_state: int = 42
) -> Dict[str, Any]:
    """Execute complete end-to-end wine quality analysis and modeling pipeline."""
    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')
    models_dir = os.path.join(os.path.dirname(output_dir), 'models') if 'outputs' in output_dir else os.path.join(output_dir, 'models')

    logger.info("Starting Wine Quality Prediction Pipeline...")
    raw_df = load_and_inspect_dataset(data_path)

    # 1. Summary tables
    tables = generate_exploratory_tables(raw_df, tables_dir)

    # 2. Binary target engineering (Analytical threshold >= 7)
    df = create_binary_target(raw_df, quality_threshold=7)

    # 3. Visualizations
    fig1 = plot_distributions(df, figures_dir)
    fig2 = plot_quality_and_class_imbalance(df, figures_dir)
    fig3 = plot_correlation_heatmap(df, figures_dir)

    # 4. Standard Stratified Split (80/20)
    X = df[FEATURE_COLUMNS]
    y = df[BINARY_TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    logger.info("Standard Stratified Split: X_train=%s, X_test=%s", X_train.shape, X_test.shape)

    # 5. Duplicate and Boundary Leakage Audit
    audit_df = audit_duplicates_and_boundary_leakage(df, X_train, X_test, y_train, y_test, tables_dir)

    # 6. Evaluate on Standard Benchmark Split
    models_standard = build_models(random_state=random_state)
    comparison_df, artifacts = evaluate_models(
        models_standard, X_train, y_train, X_test, y_test, tables_dir, models_dir,
        output_filename='model_performance_comparison.csv'
    )

    # 7. Evaluate on Deduplicated Dataset (Duplicate-Aware Evaluation)
    df_dedup = df.drop_duplicates().copy()
    X_dedup = df_dedup[FEATURE_COLUMNS]
    y_dedup = df_dedup[BINARY_TARGET]
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_dedup, y_dedup, test_size=0.20, random_state=random_state, stratify=y_dedup
    )
    models_dedup = build_models(random_state=random_state)
    dedup_comp_df, _ = evaluate_models(
        models_dedup, X_train_d, y_train_d, X_test_d, y_test_d, tables_dir, models_dir,
        output_filename='deduplicated_model_comparison.csv'
    )
    logger.info("Completed duplicate-aware evaluation on %d unique rows", len(df_dedup))

    # 8. Evaluation figures
    fig4 = plot_confusion_matrices(artifacts, figures_dir)
    fig5 = plot_roc_pr_curves(artifacts, y_test, figures_dir)

    # 9. Feature importance
    rf_pipeline = artifacts['Random Forest']['model']
    imp_df = analyze_feature_importance(rf_pipeline, X_test, y_test, tables_dir, figures_dir)

    # 10. Misclassification analysis
    errors_df = perform_misclassification_analysis(rf_pipeline, X_test, y_test, raw_df, tables_dir)

    logger.info("Pipeline execution completed successfully!")
    return {
        'raw_df': raw_df,
        'df': df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'audit_df': audit_df,
        'comparison_df': comparison_df,
        'dedup_comp_df': dedup_comp_df,
        'artifacts': artifacts,
        'imp_df': imp_df,
        'errors_df': errors_df
    }


def main():
    parser = argparse.ArgumentParser(description="Run Wine Quality Prediction Pipeline.")
    base_default = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        '--data-path',
        type=str,
        default=os.path.join(base_default, 'data', 'winequality-red.csv'),
        help='Path to winequality-red.csv dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=os.path.join(base_default, 'outputs'),
        help='Directory to store exported figures and tables'
    )
    args = parser.parse_args()
    run_pipeline(args.data_path, args.output_dir)


if __name__ == '__main__':
    main()
