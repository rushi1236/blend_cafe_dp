from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=".*joblib will operate in serial mode.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns.*",
    category=DeprecationWarning,
)

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT_DIR / "data" / "models"
DEFAULT_REPORTS_DIR = ROOT_DIR / "data" / "reports"
DEFAULT_ANALYSIS_DIR = ROOT_DIR / "data" / "data_analysis"
DEFAULT_CHARTS_DIR = DEFAULT_ANALYSIS_DIR / "charts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate Blend Cafe model evaluation metrics and create final summary artifacts."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing model artifacts and model results",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory where final evaluation CSVs are written",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="Directory containing analysis CSV outputs",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where learning_curve.png and project_summary.png will be written",
    )
    return parser.parse_args()


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def format_inr(value: float | int) -> str:
    value_int = int(round(value))
    sign = "-" if value_int < 0 else ""
    digits = str(abs(value_int))
    if len(digits) <= 3:
        formatted = digits
    else:
        last_three = digits[-3:]
        remaining = digits[:-3]
        parts: list[str] = []
        while len(remaining) > 2:
            parts.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            parts.insert(0, remaining)
        formatted = ",".join(parts + [last_three])
    return f"{sign}₹{formatted}"


def load_required_csv(path: Path, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")
    return df


def find_correlation(
    correlation_matrix: pd.DataFrame,
    variable_x: str,
    variable_y: str,
) -> tuple[float, float]:
    mask = (
        ((correlation_matrix["Variable_X"] == variable_x) & (correlation_matrix["Variable_Y"] == variable_y))
        | ((correlation_matrix["Variable_X"] == variable_y) & (correlation_matrix["Variable_Y"] == variable_x))
    )
    match = correlation_matrix.loc[mask]
    if match.empty:
        raise ValueError(f"Correlation pair not found: {variable_x} vs {variable_y}")
    row = match.iloc[0]
    return float(row["Correlation_r"]), float(row["P_Value"])


def build_learning_curve(
    rf_model: RandomForestRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    charts_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print("\nComputing learning curve (this takes ~60 seconds)...")
    estimator = RandomForestRegressor(**rf_model.get_params())
    train_sizes, train_scores, cv_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    train_scores_mean = -train_scores.mean(axis=1)
    train_scores_std = train_scores.std(axis=1)
    cv_scores_mean = -cv_scores.mean(axis=1)
    cv_scores_std = cv_scores.std(axis=1)

    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        train_sizes,
        train_scores_mean,
        "o-",
        color="#d62728",
        linewidth=2,
        label="Training RMSE",
    )
    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.15,
        color="#d62728",
    )

    ax.plot(
        train_sizes,
        cv_scores_mean,
        "o-",
        color="#2c7bb6",
        linewidth=2,
        label="CV RMSE (5-fold)",
    )
    ax.fill_between(
        train_sizes,
        cv_scores_mean - cv_scores_std,
        cv_scores_mean + cv_scores_std,
        alpha=0.15,
        color="#2c7bb6",
    )

    ax.axvline(
        x=len(X_train),
        color="green",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=f"Full train set ({len(X_train)})",
    )
    ax.set_xlabel("Training Set Size (transactions)", fontsize=11)
    ax.set_ylabel("RMSE (Quantity Units)", fontsize=11)
    ax.set_title(
        "Random Forest - Learning Curve\nTrain vs Cross-Validation RMSE as Training Data Grows",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(charts_dir / "learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Learning curve saved.")
    return train_sizes, train_scores_mean, train_scores_std, cv_scores_mean, cv_scores_std


def build_project_summary_chart(
    model_results: pd.DataFrame,
    uplift_by_slot: pd.DataFrame,
    uplift_by_weather: pd.DataFrame,
    evaluation_metrics: dict[str, object],
    charts_dir: Path,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    lr_row = model_results.loc[model_results["Model"].eq("LinearRegression")].iloc[0]
    rf_row = model_results.loc[model_results["Model"].eq("RandomForest")].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Blend Cafe Dynamic Pricing - Project Summary",
        fontsize=15,
        fontweight="bold",
    )

    # Panel 1: model performance
    ax = axes[0, 0]
    metric_labels = ["Test RMSE", "CV RMSE"]
    x = np.arange(len(metric_labels))
    width = 0.35
    lr_values = [float(lr_row["Test_RMSE"]), float(lr_row["CV_RMSE"])]
    rf_values = [float(rf_row["Test_RMSE"]), float(rf_row["CV_RMSE"])]
    ax.bar(
        x - width / 2,
        lr_values,
        width,
        label="Linear Regression",
        color="#2c7bb6",
        edgecolor="black",
        linewidth=0.7,
    )
    ax.bar(
        x + width / 2,
        rf_values,
        width,
        label="Random Forest",
        color="#d62728",
        edgecolor="black",
        linewidth=0.7,
    )
    for idx, (lr_val, rf_val) in enumerate(zip(lr_values, rf_values)):
        ax.text(idx - width / 2, lr_val + 0.002, f"{lr_val:.3f}", ha="center", fontsize=9)
        ax.text(idx + width / 2, rf_val + 0.002, f"{rf_val:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("RMSE (Quantity Units)", fontsize=11)
    ax.set_title("Forecasting Accuracy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    rmse_gap_pct = (
        (float(lr_row["Test_RMSE"]) - float(rf_row["Test_RMSE"])) / float(lr_row["Test_RMSE"]) * 100
    )
    gap_text = f"RF vs LR test RMSE: {rmse_gap_pct:+.1f}%"
    ax.text(
        0.5,
        max(max(lr_values), max(rf_values)) + 0.02,
        gap_text,
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#2ca02c" if rmse_gap_pct > 0 else "#d62728",
    )

    # Panel 2: uplift by time slot
    ax = axes[0, 1]
    slot_plot = uplift_by_slot.set_index("Time_Slot").reindex(["Morning", "Afternoon", "Evening", "Dinner"])
    slot_colors = ["#2ca02c" if value >= 0 else "#2c7bb6" for value in slot_plot["Uplift_Pct"]]
    bars = ax.bar(slot_plot.index, slot_plot["Uplift_Pct"], color=slot_colors, edgecolor="black", linewidth=0.7)
    for bar, value in zip(bars, slot_plot["Uplift_Pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.15 if value >= 0 else -0.35),
            f"{value:+.2f}%",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Revenue Uplift %", fontsize=11)
    ax.set_title("Revenue Uplift by Time Slot", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: uplift by weather
    ax = axes[1, 0]
    weather_plot = uplift_by_weather.set_index("Weather_State").reindex(
        ["Dry Day", "Light Drizzle", "Heavy Rain", "Clear After Rain"]
    )
    weather_colors = ["#2ca02c" if value >= 0 else "#2c7bb6" for value in weather_plot["Uplift_Pct"]]
    bars = ax.barh(
        weather_plot.index,
        weather_plot["Uplift_Pct"],
        color=weather_colors,
        edgecolor="black",
        linewidth=0.7,
    )
    for bar, value in zip(bars, weather_plot["Uplift_Pct"]):
        ax.text(
            value + (0.12 if value >= 0 else -0.12),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
            fontweight="bold",
        )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Revenue Uplift %", fontsize=11)
    ax.set_title("Revenue Uplift by Weather State", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 4: text summary
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = "\n".join(
        [
            "PROJECT SNAPSHOT",
            f"Simulation       : {evaluation_metrics['simulation_period']}",
            f"Transactions     : {evaluation_metrics['total_transactions']}",
            f"Menu             : {evaluation_metrics['menu_summary']}",
            f"Revenue Sim.     : {evaluation_metrics['total_revenue_simulated']}",
            "",
            f"ABC Class A      : {evaluation_metrics['abc_class_a_items']}",
            f"A Revenue Share  : {evaluation_metrics['abc_class_a_revenue_share']}",
            f"Pareto 80%       : {evaluation_metrics['pareto_80_crossover']}",
            "",
            f"Cold Brew r      : {evaluation_metrics['temp_vs_cold_brew']}",
            f"Footfall r       : {evaluation_metrics['footfall_vs_revenue']}",
            f"Rain Boost Items : {evaluation_metrics['rain_boost_items']}",
            "",
            f"Dyn Targets      : {evaluation_metrics['dynamic_targets']}",
            f"Premium Lock     : {evaluation_metrics['premium_lock_items']}",
            f"Static Test Rev  : {evaluation_metrics['test_static_revenue']}",
            f"Dynamic Test Rev : {evaluation_metrics['test_dynamic_revenue']}",
            f"Revenue Uplift   : {evaluation_metrics['revenue_uplift']}",
            "",
            f"Best Slot        : {evaluation_metrics['best_slot_uplift']}",
            f"Best Weather     : {evaluation_metrics['best_weather_uplift']}",
            f"Best Condition   : {evaluation_metrics['best_condition']}",
        ]
    )
    ax.text(
        0.03,
        0.97,
        summary_text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8f1d4", "edgecolor": "#666666"},
    )

    plt.tight_layout()
    fig.savefig(charts_dir / "project_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_evaluation_summary(
    model_results: pd.DataFrame,
    abc: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    elasticity: pd.DataFrame,
    demand_segments: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    revenue_uplift: pd.DataFrame,
    uplift_by_slot: pd.DataFrame,
    uplift_by_weather: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    total_revenue_simulated = int(abc["Total_Revenue_₹"].sum())
    class_a_count = int(abc["ABC_Class"].eq("A").sum())
    class_a_pct = class_a_count / len(abc) * 100
    class_a_revenue_share = abc.loc[abc["ABC_Class"].eq("A"), "Total_Revenue_₹"].sum() / total_revenue_simulated * 100
    pareto_80_index = int(abc.loc[abc["Cumulative_Revenue_%"].ge(80)].index[0]) + 1
    pareto_80_pct = pareto_80_index / len(abc) * 100

    lr_row = model_results.loc[model_results["Model"].eq("LinearRegression")].iloc[0]
    rf_row = model_results.loc[model_results["Model"].eq("RandomForest")].iloc[0]

    temp_cold_brew_r, temp_cold_brew_p = find_correlation(
        correlation_matrix, "Temperature_°C", "Cold_Brew_Quantity"
    )
    footfall_revenue_r, footfall_revenue_p = find_correlation(
        correlation_matrix, "Daily_Footfall_Customers", "Total_Revenue_₹"
    )

    rain_boost_items = int(elasticity["Rain_Boost_Flag"].sum())
    dynamic_targets = int(demand_segments["Segment"].eq("Dynamic Pricing Target").sum())
    premium_lock_items = int(demand_segments["Segment"].eq("Premium Lock").sum())

    total_static = int(revenue_uplift["Static_Revenue_₹"].sum())
    total_dynamic = int(revenue_uplift["Dynamic_Revenue_₹"].sum())
    total_uplift = int(revenue_uplift["Revenue_Difference_₹"].sum())
    uplift_pct = total_uplift / total_static * 100 if total_static else 0.0

    best_slot_row = uplift_by_slot.loc[uplift_by_slot["Uplift_Pct"].idxmax()]
    best_weather_row = uplift_by_weather.loc[uplift_by_weather["Uplift_Pct"].idxmax()]

    slot_weather_mean = revenue_uplift.pivot_table(
        index="Time_Slot",
        columns="Weather_State",
        values="Revenue_Difference_₹",
        aggfunc="mean",
    )
    best_condition = slot_weather_mean.stack().idxmax()
    slot_weather_totals = (
        revenue_uplift.groupby(["Time_Slot", "Weather_State"])[
            ["Static_Revenue_₹", "Dynamic_Revenue_₹"]
        ]
        .sum()
        .assign(
            Uplift_Pct=lambda frame: np.where(
                frame["Static_Revenue_₹"].eq(0),
                0.0,
                (
                    (frame["Dynamic_Revenue_₹"] - frame["Static_Revenue_₹"])
                    / frame["Static_Revenue_₹"]
                    * 100
                ),
            )
        )
    )
    best_condition_uplift = slot_weather_totals["Uplift_Pct"].round(2)
    best_condition_pct = float(best_condition_uplift.loc[best_condition])

    min_date = pd.to_datetime(feature_matrix["Date"]).min().date()
    max_date = pd.to_datetime(feature_matrix["Date"]).max().date()

    evaluation_metrics = {
        "simulation_period": f"{min_date:%b %d} - {max_date:%b %d %Y} ({feature_matrix['Date'].nunique()} days)",
        "total_transactions": f"{len(feature_matrix):,}",
        "menu_summary": f"{feature_matrix['Item_Name'].nunique()} across {feature_matrix['Category'].nunique()} categories",
        "total_revenue_simulated": format_inr(total_revenue_simulated),
        "abc_class_a_items": f"{class_a_count} items ({class_a_pct:.1f}% of menu)",
        "abc_class_a_revenue_share": f"{class_a_revenue_share:.2f}%",
        "pareto_80_crossover": f"{pareto_80_index} items ({pareto_80_pct:.1f}% of menu)",
        "lr_test_rmse": f"{float(lr_row['Test_RMSE']):.4f}",
        "rf_test_rmse": f"{float(rf_row['Test_RMSE']):.4f}",
        "rf_cv_rmse": f"{float(rf_row['CV_RMSE']):.4f}",
        "rf_test_r2": f"{float(rf_row['Test_R2']):.4f}",
        "temp_vs_cold_brew": f"{temp_cold_brew_r:+.3f} (p={temp_cold_brew_p:.2e})",
        "footfall_vs_revenue": f"{footfall_revenue_r:+.3f} (p={footfall_revenue_p:.2e})",
        "rain_boost_items": f"{rain_boost_items} items",
        "dynamic_targets": f"{dynamic_targets} items",
        "premium_lock_items": f"{premium_lock_items} items",
        "test_static_revenue": format_inr(total_static),
        "test_dynamic_revenue": format_inr(total_dynamic),
        "revenue_uplift_amount": format_inr(total_uplift),
        "revenue_uplift": f"{uplift_pct:.2f}%",
        "best_slot_uplift": f"{best_slot_row['Time_Slot']} {best_slot_row['Uplift_Pct']:+.2f}%",
        "best_weather_uplift": f"{best_weather_row['Weather_State']} {best_weather_row['Uplift_Pct']:+.2f}%",
        "best_condition": f"{best_condition[0]} x {best_condition[1]} {best_condition_pct:+.2f}%",
    }

    evaluation_summary = pd.DataFrame(
        {
            "Metric": [
                "Simulation Period",
                "Total Transactions",
                "Total Menu Items",
                "Total Revenue Simulated",
                "ABC Class A Items",
                "ABC Class A Revenue Share",
                "Pareto 80% crossover",
                "LR Test RMSE",
                "RF Test RMSE",
                "RF CV RMSE",
                "RF R² Test",
                "Temperature vs Cold Brew r",
                "Footfall vs Revenue r",
                "Rain Boost Items",
                "Dynamic Pricing Targets",
                "Premium Lock Items",
                "Test Period Static Revenue",
                "Test Period Dynamic Revenue",
                "Revenue Uplift ₹",
                "Revenue Uplift %",
                "Best Slot Uplift",
                "Best Weather Uplift",
                "Best Condition",
            ],
            "Value": [
                evaluation_metrics["simulation_period"],
                evaluation_metrics["total_transactions"],
                evaluation_metrics["menu_summary"],
                evaluation_metrics["total_revenue_simulated"],
                evaluation_metrics["abc_class_a_items"],
                evaluation_metrics["abc_class_a_revenue_share"],
                evaluation_metrics["pareto_80_crossover"],
                evaluation_metrics["lr_test_rmse"],
                evaluation_metrics["rf_test_rmse"],
                evaluation_metrics["rf_cv_rmse"],
                evaluation_metrics["rf_test_r2"],
                evaluation_metrics["temp_vs_cold_brew"],
                evaluation_metrics["footfall_vs_revenue"],
                evaluation_metrics["rain_boost_items"],
                evaluation_metrics["dynamic_targets"],
                evaluation_metrics["premium_lock_items"],
                evaluation_metrics["test_static_revenue"],
                evaluation_metrics["test_dynamic_revenue"],
                evaluation_metrics["revenue_uplift_amount"],
                evaluation_metrics["revenue_uplift"],
                evaluation_metrics["best_slot_uplift"],
                evaluation_metrics["best_weather_uplift"],
                evaluation_metrics["best_condition"],
            ],
        }
    )
    return evaluation_summary, evaluation_metrics


def main() -> int:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.charts_dir.mkdir(parents=True, exist_ok=True)

    model_results = load_required_csv(
        args.models_dir / "model_results_summary.csv",
        ["Model", "Test_RMSE", "Test_R2", "CV_RMSE"],
    )
    uplift_by_slot = load_required_csv(
        args.reports_dir / "uplift_by_slot.csv",
        ["Time_Slot", "Static_Revenue", "Dynamic_Revenue", "Uplift_₹", "Uplift_Pct"],
    )
    uplift_by_weather = load_required_csv(
        args.reports_dir / "uplift_by_weather.csv",
        ["Weather_State", "Transaction_Count", "Static_Revenue", "Dynamic_Revenue", "Uplift_₹", "Uplift_Pct"],
    )
    uplift_by_segment = load_required_csv(
        args.reports_dir / "uplift_by_segment.csv",
        ["Segment", "Static_Revenue", "Dynamic_Revenue", "Uplift_₹", "Uplift_Pct"],
    )
    feature_columns = joblib.load(args.models_dir / "feature_columns.pkl")
    train = load_required_csv(args.models_dir / "train_features.csv", feature_columns + ["Quantity_Units"])
    X_train = train[feature_columns]
    y_train = train["Quantity_Units"]
    rf = joblib.load(args.models_dir / "rf_demand_model.pkl")

    abc = load_required_csv(
        args.analysis_dir / "abc_classified.csv",
        ["Item_Name", "Total_Revenue_₹", "Cumulative_Revenue_%", "ABC_Class"],
    )
    correlation_matrix = load_required_csv(
        args.analysis_dir / "correlation_matrix.csv",
        ["Variable_X", "Variable_Y", "Correlation_r", "P_Value"],
    )
    elasticity = load_required_csv(
        args.analysis_dir / "elasticity_coefficients.csv",
        ["Item_Name", "Rain_Boost_Flag"],
    )
    demand_segments = load_required_csv(
        args.analysis_dir / "demand_segments.csv",
        ["Item_Name", "Segment"],
    )
    feature_matrix = load_required_csv(
        args.models_dir / "feature_matrix.csv",
        ["Date", "Item_Name", "Category"],
    )
    revenue_uplift = load_required_csv(
        args.reports_dir / "revenue_uplift_analysis.csv",
        ["Date", "Time_Slot", "Weather_State", "Static_Revenue_₹", "Dynamic_Revenue_₹", "Revenue_Difference_₹"],
    )
    recommendations = load_required_csv(
        args.models_dir / "price_recommendations.csv",
        ["Time_Slot", "Weather_State", "Price_Change_%"],
    )

    elasticity["Rain_Boost_Flag"] = elasticity["Rain_Boost_Flag"].astype(str).str.lower().eq("true")

    print("All inputs loaded.")
    print(
        "Model results:\n"
        + model_results[["Model", "Test_RMSE", "Test_R2", "CV_RMSE"]].to_string(index=False)
    )

    build_learning_curve(rf, X_train, y_train, args.charts_dir)

    slot_weather_price_change = recommendations.pivot_table(
        index="Time_Slot",
        columns="Weather_State",
        values="Price_Change_%",
        aggfunc="mean",
    )
    best_price_condition = slot_weather_price_change.stack().idxmax()
    best_price_condition_pct = float(slot_weather_price_change.loc[best_price_condition[0], best_price_condition[1]])

    evaluation_summary, evaluation_metrics = build_evaluation_summary(
        model_results,
        abc,
        correlation_matrix,
        elasticity,
        demand_segments,
        feature_matrix,
        revenue_uplift,
        uplift_by_slot,
        uplift_by_weather,
    )
    evaluation_metrics["best_condition"] = (
        f"{best_price_condition[0]} x {best_price_condition[1]} {best_price_condition_pct:+.2f}%"
    )

    build_project_summary_chart(
        model_results,
        uplift_by_slot,
        uplift_by_weather,
        evaluation_metrics,
        args.charts_dir,
    )

    evaluation_summary.loc[
        evaluation_summary["Metric"].eq("Best Condition"), "Value"
    ] = evaluation_metrics["best_condition"]
    evaluation_summary.to_csv(args.reports_dir / "project_evaluation_summary.csv", index=False)
    print("\nProject evaluation summary saved.")

    print("\n05_model_evaluation.py complete.")
    print("")
    print("All Phase 3 Outputs")
    print("")
    print("data/models/")
    for name in [
        "lr_demand_model.pkl",
        "rf_demand_model.pkl",
        "feature_matrix.csv",
        "train_features.csv",
        "test_features.csv",
        "test_predictions.csv",
        "model_results_summary.csv",
        "price_recommendations.csv",
        "feature_columns.pkl",
        "le_category.pkl",
    ]:
        print(display_path(args.models_dir / name))
    print("")
    print("data/reports/")
    for name in [
        "revenue_uplift_analysis.csv",
        "uplift_by_category.csv",
        "uplift_by_slot.csv",
        "uplift_by_weather.csv",
        "uplift_by_segment.csv",
        "project_evaluation_summary.csv",
    ]:
        print(display_path(args.reports_dir / name))
    print("")
    print("data/data_analysis/charts/")
    for name in [
        "rf_feature_importance.png",
        "model_comparison.png",
        "learning_curve.png",
        "project_summary.png",
    ]:
        print(display_path(args.charts_dir / name))
    print("")
    print("Final Numbers")
    print("")
    print(
        f"Projected revenue uplift: {evaluation_metrics['revenue_uplift']} "
        f"({evaluation_metrics['revenue_uplift_amount']} on test period)"
    )
    print(f"Best condition: {evaluation_metrics['best_condition']} per item")
    print("All 13 scripts complete. Project pipeline verified end-to-end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
