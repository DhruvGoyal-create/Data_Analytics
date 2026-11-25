"""
Predicting House Prices with Linear Regression — Pipeline & Diagnostics
========================================================================
DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO — Project 05 (House Price Prediction)
Author: Data Analytics Portfolio Intern
Dataset: California Housing Dataset (StatLib / 1990 U.S. Census)
Target: median_house_value (Continuous numeric)

This module implements an end-to-end regression pipeline:
1. Data loading, schema validation, and missing value auditing.
2. Exploratory Data Analysis (target distribution, correlations, feature distributions).
3. Leakage-free preprocessing via ColumnTransformer and Pipeline:
   - Median imputation + StandardScaler for numeric features
   - OneHotEncoder (drop='first') for categorical features
4. Model training: Dummy Baseline, Linear Regression (OLS), Ridge, Lasso, and Random Forest.
5. 5-Fold Cross-Validation and comprehensive test set evaluation (MAE, MSE, RMSE, R²).
6. Residual diagnostics (homoscedasticity, normality, census cap boundary effects).
7. Feature coefficient analysis (standardized beta weights and directional impacts).
8. Publication-ready figure and summary table generation.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configure matplotlib styling
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_and_validate_data(data_path: str) -> pd.DataFrame:
    """Load California housing dataset and perform structural validation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Housing dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset successfully from %s. Shape: %s", data_path, df.shape)

    expected_cols = [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income",
        "median_house_value", "ocean_proximity"
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' missing from dataset.")

    return df


def generate_eda_figures(df: pd.DataFrame, figures_dir: str, tables_dir: str) -> pd.DataFrame:
    """Generate exploratory data analysis charts and correlation tables."""
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Dataset summary table
    summary_stats = df.describe(include="all").T
    summary_stats["null_count"] = df.isnull().sum()
    summary_stats["null_percent"] = (df.isnull().sum() / len(df)) * 100.0
    summary_csv = os.path.join(tables_dir, "housing_dataset_summary.csv")
    summary_stats.to_csv(summary_csv)
    logger.info("Exported dataset summary to %s", summary_csv)

    # 2. Figure 1: Target Variable Distribution (median_house_value)
    target = df["median_house_value"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(target, kde=True, ax=ax1, color="#1f77b4", bins=50)
    ax1.set_title("Distribution of Median House Value (USD)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Median House Value ($)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.axvline(target.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: ${target.mean():,.0f}")
    ax1.axvline(target.median(), color="green", linestyle="-.", linewidth=1.5, label=f"Median: ${target.median():,.0f}")
    ax1.legend(loc="upper right", frameon=True)

    # Inset / bar showing ceiling cap effect at $500,001
    cap_count = (target >= 500000).sum()
    cap_pct = (cap_count / len(target)) * 100.0
    ax1.text(
        0.05, 0.80,
        f"Skewness: {stats.skew(target):.2f}\nKurtosis: {stats.kurtosis(target):.2f}\nCapped ($500k+): {cap_count} ({cap_pct:.1f}%)",
        transform=ax1.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9)
    )

    # Boxplot by ocean proximity
    order = df.groupby("ocean_proximity")["median_house_value"].median().sort_values(ascending=False).index.tolist()
    sns.boxplot(x="ocean_proximity", y="median_house_value", hue="ocean_proximity", data=df, order=order, ax=ax2, palette="Blues_r", legend=False)
    ax2.set_title("Median House Value by Ocean Proximity", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Ocean Proximity Category", fontsize=11)
    ax2.set_ylabel("Median House Value ($)", fontsize=11)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=15)

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "fig1_target_distribution.png")
    plt.savefig(fig1_path)
    plt.close()
    logger.info("Saved Figure 1 to %s", fig1_path)

    # 3. Figure 2: Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # Save correlation table
    target_corr = corr_matrix["median_house_value"].sort_values(ascending=False).reset_index()
    target_corr.columns = ["Feature", "Pearson_Correlation"]
    corr_csv = os.path.join(tables_dir, "feature_correlations.csv")
    target_corr.to_csv(corr_csv, index=False)
    logger.info("Exported feature correlations to %s", corr_csv)

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
    )
    plt.title("Correlation Matrix of California Housing Attributes", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "fig2_correlation_heatmap.png")
    plt.savefig(fig2_path)
    plt.close()
    logger.info("Saved Figure 2 to %s", fig2_path)

    return target_corr


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Construct leakage-free ColumnTransformer pipeline."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    return preprocessor


def train_and_evaluate_models(
    df: pd.DataFrame,
    tables_dir: str,
    figures_dir: str,
    models_dir: str,
    random_state: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train models, evaluate performance metrics, generate residual diagnostics, and save pipelines."""
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    X = df.drop(columns=["median_house_value"])
    y = df["median_house_value"]

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    logger.info("Partitioned dataset: %d train records, %d test records (80/20 split)", len(X_train), len(X_test))

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = ["ocean_proximity"]

    preprocessor = build_preprocessor(num_features, cat_features)

    # Define model dictionary
    models = {
        "Dummy (Mean Baseline)": DummyRegressor(strategy="mean"),
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (alpha=1.0)": Ridge(alpha=1.0, random_state=random_state),
        "Lasso Regression (alpha=100.0)": Lasso(alpha=100.0, random_state=random_state),
        "Random Forest (n=100, depth=16)": RandomForestRegressor(n_estimators=100, max_depth=16, random_state=random_state, n_jobs=-1)
    }

    results = []
    trained_pipelines = {}
    test_predictions = {}

    for name, model in models.items():
        if name.startswith("Dummy"):
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            cv_mean = np.nan
            cv_std = np.nan
        else:
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", model)
            ])
            # 5-Fold Cross-Validation on training data only
            cv_scores = cross_val_score(
                pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
            )
            cv_mean = -cv_scores.mean()
            cv_std = cv_scores.std()

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)

        trained_pipelines[name] = pipe
        test_predictions[name] = y_pred

        results.append({
            "Model": name,
            "Test_MAE": round(mae, 2),
            "Test_MSE": round(mse, 2),
            "Test_RMSE": round(rmse, 2),
            "Test_R2": round(r2, 4),
            "CV_RMSE_Mean": round(cv_mean, 2) if not np.isnan(cv_mean) else "N/A",
            "CV_RMSE_Std": round(cv_std, 2) if not np.isnan(cv_std) else "N/A"
        })
        logger.info(
            "Model [%s] -> Test MAE: $%.2f, Test RMSE: $%.2f, R²: %.4f, CV-RMSE: %s",
            name, mae, rmse, r2, f"${cv_mean:,.2f} +/- ${cv_std:,.2f}" if not np.isnan(cv_mean) else "N/A"
        )

    results_df = pd.DataFrame(results)
    perf_csv = os.path.join(tables_dir, "model_performance_comparison.csv")
    results_df.to_csv(perf_csv, index=False)
    logger.info("Exported model performance comparison to %s", perf_csv)

    # Save primary pipelines
    joblib.dump(trained_pipelines["Linear Regression (OLS)"], os.path.join(models_dir, "linear_regression_pipeline.joblib"))
    joblib.dump(trained_pipelines["Ridge Regression (alpha=1.0)"], os.path.join(models_dir, "ridge_pipeline.joblib"))
    joblib.dump(trained_pipelines["Lasso Regression (alpha=100.0)"], os.path.join(models_dir, "lasso_pipeline.joblib"))
    joblib.dump(trained_pipelines["Random Forest (n=100, depth=16)"], os.path.join(models_dir, "random_forest_pipeline.joblib"))
    logger.info("Serialized trained model pipelines into %s", models_dir)

    # 4. Coefficient extraction from Linear Regression
    lr_pipe = trained_pipelines["Linear Regression (OLS)"]
    lr_model = lr_pipe.named_steps["regressor"]
    lr_prep = lr_pipe.named_steps["preprocessor"]

    cat_encoder = lr_prep.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + cat_feature_names

    coef_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Standardized_Coefficient": lr_model.coef_,
        "Absolute_Weight": np.abs(lr_model.coef_)
    }).sort_values(by="Absolute_Weight", ascending=False)
    coef_df["Impact_Rank"] = range(1, len(coef_df) + 1)
    coef_df["Direction"] = coef_df["Standardized_Coefficient"].apply(lambda w: "Positive (+)" if w > 0 else "Negative (-)")

    coef_csv = os.path.join(tables_dir, "regression_coefficients.csv")
    coef_df.to_csv(coef_csv, index=False)
    logger.info("Exported standardized coefficients to %s (Intercept: $%.2f)", coef_csv, lr_model.intercept_)

    # 5. Figure 3: Actual vs. Predicted Prices
    y_pred_lr = test_predictions["Linear Regression (OLS)"]
    r2_lr = results_df.loc[results_df["Model"] == "Linear Regression (OLS)", "Test_R2"].values[0]
    rmse_lr = results_df.loc[results_df["Model"] == "Linear Regression (OLS)", "Test_RMSE"].values[0]

    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, y_pred_lr, alpha=0.25, color="#1f77b4", edgecolor="none", s=25, label="Test Observations")
    min_val = min(y_test.min(), y_pred_lr.min())
    max_val = max(y_test.max(), y_pred_lr.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", linewidth=2.0, label="Ideal Fit (y = x)")

    plt.title("Actual vs. Predicted House Prices — Linear Regression", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Actual Median House Value ($)", fontsize=11)
    plt.ylabel("Predicted Median House Value ($)", fontsize=11)
    plt.xlim(0, 550000)
    plt.ylim(-50000, 550000)
    plt.legend(loc="upper left", frameon=True)

    plt.text(
        0.65, 0.15,
        f"Test R²: {r2_lr:.4f}\nTest RMSE: ${rmse_lr:,.0f}\nIntercept: ${lr_model.intercept_:,.0f}",
        transform=plt.gca().transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9)
    )
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "fig3_actual_vs_predicted.png")
    plt.savefig(fig3_path)
    plt.close()
    logger.info("Saved Figure 3 to %s", fig3_path)

    # 6. Figure 4: Residual Diagnostics
    residuals_lr = y_test - y_pred_lr

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 11))

    # Ax1: Residuals vs Predicted (Heteroscedasticity check)
    ax1.scatter(y_pred_lr, residuals_lr, alpha=0.25, color="#2ca02c", edgecolor="none", s=22)
    ax1.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    ax1.set_title("Residuals vs. Fitted Values (Heteroscedasticity Check)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Fitted Values ($)", fontsize=10)
    ax1.set_ylabel("Residuals ($: Actual - Predicted)", fontsize=10)

    # Ax2: Residuals Distribution
    sns.histplot(residuals_lr, kde=True, ax=ax2, color="#9467bd", bins=50)
    ax2.axvline(0, color="crimson", linestyle="--", linewidth=1.5)
    ax2.set_title("Distribution of Residual Errors", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Residual ($)", fontsize=10)
    ax2.set_ylabel("Density / Count", fontsize=10)
    ax2.text(
        0.05, 0.85,
        f"Mean: ${residuals_lr.mean():,.0f}\nMedian: ${np.median(residuals_lr):,.0f}\nSkewness: {stats.skew(residuals_lr):.2f}\nKurtosis: {stats.kurtosis(residuals_lr):.2f}",
        transform=ax2.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.9)
    )

    # Ax3: Normal Q-Q Plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals_lr, dist="norm")
    ax3.plot(osm, osr, "o", color="#17becf", alpha=0.3, markersize=3.5)
    ax3.plot(osm, slope * np.array(osm) + intercept, "r-", linewidth=1.8, label=f"Fit (R = {r:.3f})")
    ax3.set_title("Normal Q-Q Plot of Residuals", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax3.set_ylabel("Standardized Residuals", fontsize=10)
    ax3.legend(loc="upper left", frameon=True)

    # Ax4: Residual Boxplot for Capped vs Uncapped Target Samples
    capped_mask = y_test >= 500000
    res_df = pd.DataFrame({
        "Subset": np.where(capped_mask, "Capped ($500k Ceiling, N=184)", "Uncapped (<$500k, N=3944)"),
        "Absolute_Error": np.abs(residuals_lr)
    })
    sns.boxplot(x="Subset", y="Absolute_Error", hue="Subset", data=res_df, ax=ax4, palette="Set2", legend=False)
    ax4.set_title("Absolute Error: Uncapped vs. Capped Ceiling Samples", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Sample Category", fontsize=10)
    ax4.set_ylabel("Absolute Error ($)", fontsize=10)

    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "fig4_residual_diagnostics.png")
    plt.savefig(fig4_path)
    plt.close()
    logger.info("Saved Figure 4 to %s", fig4_path)

    # Residual summary table
    uncapped_rmse = np.sqrt(mean_squared_error(y_test[~capped_mask], y_pred_lr[~capped_mask]))
    capped_rmse = np.sqrt(mean_squared_error(y_test[capped_mask], y_pred_lr[capped_mask]))
    uncapped_mae = mean_absolute_error(y_test[~capped_mask], y_pred_lr[~capped_mask])
    capped_mae = mean_absolute_error(y_test[capped_mask], y_pred_lr[capped_mask])

    res_diag_df = pd.DataFrame([
        {"Metric": "Overall Mean Residual", "Value": f"${residuals_lr.mean():,.2f}"},
        {"Metric": "Overall Median Residual", "Value": f"${np.median(residuals_lr):,.2f}"},
        {"Metric": "Overall Residual Std Dev", "Value": f"${residuals_lr.std():,.2f}"},
        {"Metric": "Residual Skewness", "Value": f"{stats.skew(residuals_lr):.4f}"},
        {"Metric": "Residual Kurtosis (Excess)", "Value": f"{stats.kurtosis(residuals_lr):.4f}"},
        {"Metric": "Uncapped Subset Count (<$500k)", "Value": f"{(~capped_mask).sum()} (95.54%)"},
        {"Metric": "Uncapped Subset MAE", "Value": f"${uncapped_mae:,.2f}"},
        {"Metric": "Uncapped Subset RMSE", "Value": f"${uncapped_rmse:,.2f}"},
        {"Metric": "Capped Ceiling Subset Count ($500k+)", "Value": f"{capped_mask.sum()} (4.46%)"},
        {"Metric": "Capped Ceiling Subset MAE", "Value": f"${capped_mae:,.2f}"},
        {"Metric": "Capped Ceiling Subset RMSE", "Value": f"${capped_rmse:,.2f}"},
    ])
    res_diag_csv = os.path.join(tables_dir, "residual_diagnostics_summary.csv")
    res_diag_df.to_csv(res_diag_csv, index=False)
    logger.info("Exported residual diagnostics summary to %s", res_diag_csv)

    # 7. Figure 5: Standardized Coefficient Analysis
    plt.figure(figsize=(10, 7))
    plot_coef = coef_df.sort_values(by="Standardized_Coefficient", ascending=True)
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in plot_coef["Standardized_Coefficient"]]
    bars = plt.barh(plot_coef["Feature"], plot_coef["Standardized_Coefficient"], color=colors, alpha=0.85, edgecolor="#333", linewidth=0.7)
    plt.axvline(0, color="black", linestyle="-", linewidth=1.0)
    plt.title("Standardized Linear Regression Coefficients (Conditional Associations)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Conditional USD Difference per 1 SD Feature Shift (ceteris paribus)", fontsize=11)
    plt.ylabel("Housing Feature / Indicator", fontsize=11)

    # Annotate bar values
    for bar in bars:
        val = bar.get_width()
        offset = 2000 if val >= 0 else -14000
        plt.text(val + offset, bar.get_y() + bar.get_height() / 2, f"${val:,.0f}", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "fig5_coefficient_analysis.png")
    plt.savefig(fig5_path)
    plt.close()
    logger.info("Saved Figure 5 to %s", fig5_path)

    # 8. Figure 6: Model Performance Benchmark Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    benchmark_df = results_df.copy()
    model_labels = [
        "Dummy (Mean)", "Linear Reg (Primary)", "Ridge (a=1.0)", "Lasso (a=100)", "Random Forest (Comp)"
    ]

    # Subplot 1: Test RMSE (Lower is better)
    rmse_bars = ax1.bar(model_labels, benchmark_df["Test_RMSE"], color=["#7f7f7f", "#1f77b4", "#aec7e8", "#ff7f0e", "#2ca02c"], edgecolor="#333")
    ax1.set_title("Test RMSE Comparison (Lower is Better)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Root Mean Squared Error ($)", fontsize=11)
    ax1.set_xticks(range(len(model_labels)))
    ax1.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=9.5)
    for bar in rmse_bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1500, f"${h:,.0f}", ha="center", fontsize=8.5, fontweight="bold")

    # Subplot 2: Test R² (Higher is better)
    r2_bars = ax2.bar(model_labels, benchmark_df["Test_R2"], color=["#7f7f7f", "#1f77b4", "#aec7e8", "#ff7f0e", "#2ca02c"], edgecolor="#333")
    ax2.set_title("Test R² Score Comparison (Higher is Better)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("R² Coefficient of Determination", fontsize=11)
    ax2.set_ylim(-0.05, 1.0)
    ax2.set_xticks(range(len(model_labels)))
    ax2.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=9.5)
    for bar in r2_bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.4f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    fig6_path = os.path.join(figures_dir, "fig6_model_performance_comparison.png")
    plt.savefig(fig6_path)
    plt.close()
    logger.info("Saved Figure 6 to %s", fig6_path)

    return trained_pipelines, results_df, coef_df


def main():
    """Main CLI entrypoint for running the house price prediction pipeline."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_default = os.path.join(base_dir, "data", "housing.csv")
    figures_default = os.path.join(base_dir, "outputs", "figures")
    tables_default = os.path.join(base_dir, "outputs", "tables")
    models_default = os.path.join(base_dir, "models")

    parser = argparse.ArgumentParser(description="Predicting House Prices with Linear Regression Pipeline")
    parser.add_argument("--data-path", type=str, default=data_default, help="Path to input housing CSV")
    parser.add_argument("--figures-dir", type=str, default=figures_default, help="Directory to save figures")
    parser.add_argument("--tables-dir", type=str, default=tables_default, help="Directory to save CSV tables")
    parser.add_argument("--models-dir", type=str, default=models_default, help="Directory to save model artifacts")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed")

    args = parser.parse_args()

    logger.info("Starting House Price Prediction pipeline...")
    df = load_and_validate_data(args.data_path)
    generate_eda_figures(df, args.figures_dir, args.tables_dir)
    train_and_evaluate_models(df, args.tables_dir, args.figures_dir, args.models_dir, args.random_state)
    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
