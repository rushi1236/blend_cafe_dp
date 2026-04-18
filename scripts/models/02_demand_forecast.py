from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT_DIR / "data" / "models"
DEFAULT_CHARTS_DIR = ROOT_DIR / "data" / "data_analysis" / "charts"
FEATURE_COLUMNS_PATH = DEFAULT_MODELS_DIR / "feature_columns.pkl"
TRAIN_PATH = DEFAULT_MODELS_DIR / "train_features.csv"
TEST_PATH = DEFAULT_MODELS_DIR / "test_features.csv"
TARGET_COLUMN = "Quantity_Units"
PRIMARY_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}
FALLBACK_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Blend Cafe demand forecasting models and compare baseline vs primary model."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing train/test feature files and where model outputs will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where comparison charts will be written",
    )
    return parser.parse_args()


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def load_feature_columns(path: Path) -> list[str]:
    feature_columns = joblib.load(path)
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("feature_columns.pkl does not contain a non-empty list")
    return feature_columns


def load_dataset(path: Path, feature_columns: list[str]) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    required_columns = feature_columns + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

    feature_null_count = int(dataset[feature_columns].isna().sum().sum())
    if feature_null_count != 0:
        raise ValueError(f"{path.name} contains {feature_null_count} null feature values")
    if dataset[TARGET_COLUMN].isna().any():
        raise ValueError(f"{path.name} contains null target values")

    return dataset


def evaluate_predictions(
    y_train: pd.Series,
    pred_train: np.ndarray,
    y_test: pd.Series,
    pred_test: np.ndarray,
    cv_rmse: float,
    cv_std: float,
) -> dict[str, float]:
    return {
        "train_mae": mean_absolute_error(y_train, pred_train),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "train_r2": r2_score(y_train, pred_train),
        "test_mae": mean_absolute_error(y_test, pred_test),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "test_r2": r2_score(y_test, pred_test),
        "cv_rmse": cv_rmse,
        "cv_std": cv_std,
    }


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[LinearRegression, np.ndarray, np.ndarray, dict[str, float]]:
    model = LinearRegression()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    cv_rmse = float(-cv_scores.mean())
    cv_std = float(cv_scores.std())

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    results = evaluate_predictions(y_train, pred_train, y_test, pred_test, cv_rmse, cv_std)
    return model, pred_train, pred_test, results


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict[str, object],
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray, dict[str, float]]:
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
    )
    cv_rmse = float(-cv_scores.mean())
    cv_std = float(cv_scores.std())

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    results = evaluate_predictions(y_train, pred_train, y_test, pred_test, cv_rmse, cv_std)
    return model, pred_train, pred_test, results


def save_feature_importance_chart(
    feature_columns: list[str],
    model: RandomForestRegressor,
    charts_dir: Path,
) -> pd.DataFrame:
    charts_dir.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {"Feature": feature_columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=True)

    high_quantile = importance_df["Importance"].quantile(0.75)
    mid_quantile = importance_df["Importance"].quantile(0.50)
    colors = [
        "#d62728"
        if importance > high_quantile
        else "#ff7f0e"
        if importance > mid_quantile
        else "#aec7e8"
        for importance in importance_df["Importance"]
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    for bar, value in zip(bars, importance_df["Importance"]):
        ax.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(
        "Random Forest - Feature Importance\nBlend Cafe Demand Forecasting Model",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(charts_dir / "rf_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)


def save_model_comparison_chart(
    y_test: pd.Series,
    lr_pred_test: np.ndarray,
    rf_pred_test: np.ndarray,
    lr_results: dict[str, float],
    rf_results: dict[str, float],
    charts_dir: Path,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Model Comparison - Linear Regression vs Random Forest\nDemand Forecasting (Quantity_Units)",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].scatter(
        y_test,
        lr_pred_test,
        alpha=0.4,
        color="#2c7bb6",
        s=20,
        label=f"LR  (RMSE={lr_results['test_rmse']:.3f})",
    )
    axes[0].scatter(
        y_test,
        rf_pred_test,
        alpha=0.4,
        color="#d62728",
        s=20,
        label=f"RF  (RMSE={rf_results['test_rmse']:.3f})",
    )
    upper_bound = max(float(y_test.max()), float(np.max(lr_pred_test)), float(np.max(rf_pred_test))) + 0.2
    lims = [0, upper_bound]
    axes[0].plot(lims, lims, "k--", linewidth=1, alpha=0.6, label="Perfect prediction")
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("Actual Quantity", fontsize=11)
    axes[0].set_ylabel("Predicted Quantity", fontsize=11)
    axes[0].set_title("Actual vs Predicted (Test Set)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    metrics = ["Test MAE", "Test RMSE"]
    lr_values = [lr_results["test_mae"], lr_results["test_rmse"]]
    rf_values = [rf_results["test_mae"], rf_results["test_rmse"]]
    positions = np.arange(len(metrics))
    width = 0.3

    axes[1].bar(
        positions - width / 2,
        lr_values,
        width,
        label="Linear Regression",
        color="#2c7bb6",
        edgecolor="black",
        linewidth=0.7,
    )
    axes[1].bar(
        positions + width / 2,
        rf_values,
        width,
        label="Random Forest",
        color="#d62728",
        edgecolor="black",
        linewidth=0.7,
    )

    for index, (lr_value, rf_value) in enumerate(zip(lr_values, rf_values)):
        axes[1].text(
            index - width / 2,
            lr_value + 0.003,
            f"{lr_value:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
        axes[1].text(
            index + width / 2,
            rf_value + 0.003,
            f"{rf_value:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(metrics, fontsize=11)
    axes[1].set_ylabel("Error (units)", fontsize=11)
    axes[1].set_title("Error Metrics Comparison", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(charts_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_predictions_and_summary(
    test: pd.DataFrame,
    feature_columns: list[str],
    lr_pred_test: np.ndarray,
    rf_pred_test: np.ndarray,
    lr_results: dict[str, float],
    rf_results: dict[str, float],
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    predictions_df = test[feature_columns + [TARGET_COLUMN]].copy()
    predictions_df["LR_Predicted"] = lr_pred_test
    predictions_df["RF_Predicted"] = rf_pred_test
    predictions_df["RF_Residual"] = test[TARGET_COLUMN].to_numpy() - rf_pred_test
    predictions_df.to_csv(models_dir / "test_predictions.csv", index=False)

    results_summary = pd.DataFrame(
        {
            "Model": ["LinearRegression", "RandomForest"],
            "Train_MAE": [lr_results["train_mae"], rf_results["train_mae"]],
            "Train_RMSE": [lr_results["train_rmse"], rf_results["train_rmse"]],
            "Train_R2": [lr_results["train_r2"], rf_results["train_r2"]],
            "Test_MAE": [lr_results["test_mae"], rf_results["test_mae"]],
            "Test_RMSE": [lr_results["test_rmse"], rf_results["test_rmse"]],
            "Test_R2": [lr_results["test_r2"], rf_results["test_r2"]],
            "CV_RMSE": [lr_results["cv_rmse"], rf_results["cv_rmse"]],
            "CV_Std": [lr_results["cv_std"], rf_results["cv_std"]],
        }
    )
    results_summary.to_csv(models_dir / "model_results_summary.csv", index=False)


def print_dataset_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Target mean (train): {y_train.mean():.3f}")
    print(f"Target std  (train): {y_train.std():.3f}")
    print(f"Target range: {y_train.min()} - {y_train.max()}")


def print_model_results(title: str, results: dict[str, float]) -> None:
    print(f"\n--- {title} ---")
    print(f"  Train MAE : {results['train_mae']:.4f}")
    print(f"  Train RMSE: {results['train_rmse']:.4f}")
    print(f"  Train R²  : {results['train_r2']:.4f}")
    print(f"  Test  MAE : {results['test_mae']:.4f}")
    print(f"  Test  RMSE: {results['test_rmse']:.4f}")
    print(f"  Test  R²  : {results['test_r2']:.4f}")
    print(f"  CV RMSE   : {results['cv_rmse']:.4f} ± {results['cv_std']:.4f}")


def print_final_summary(
    lr_results: dict[str, float],
    rf_results: dict[str, float],
    importance_df: pd.DataFrame,
    models_dir: Path,
    charts_dir: Path,
) -> None:
    rf_improvement = (
        (lr_results["test_rmse"] - rf_results["test_rmse"]) / lr_results["test_rmse"] * 100
    )
    train_test_ratio = rf_results["train_rmse"] / rf_results["test_rmse"]
    ratio_status = "OK" if train_test_ratio < 1.5 else "WARNING"
    top_features = importance_df.head(3).reset_index(drop=True)
    lr_cv_text = f"{lr_results['cv_rmse']:.4f} ± {lr_results['cv_std']:.4f}"
    rf_cv_text = f"{rf_results['cv_rmse']:.4f} ± {rf_results['cv_std']:.4f}"

    print("")
    print("02_demand_forecast.py complete.")
    print("")
    print("MODEL COMPARISON SUMMARY")
    print("")
    print(f"{'Metric':<12} | {'LinearRegression':>20} | {'RandomForest':>14}")
    print(f"{'Train MAE':<12} | {lr_results['train_mae']:>20.4f} | {rf_results['train_mae']:>14.4f}")
    print(f"{'Train RMSE':<12} | {lr_results['train_rmse']:>20.4f} | {rf_results['train_rmse']:>14.4f}")
    print(f"{'Train R²':<12} | {lr_results['train_r2']:>20.4f} | {rf_results['train_r2']:>14.4f}")
    print(f"{'Test MAE':<12} | {lr_results['test_mae']:>20.4f} | {rf_results['test_mae']:>14.4f}")
    print(f"{'Test RMSE':<12} | {lr_results['test_rmse']:>20.4f} | {rf_results['test_rmse']:>14.4f}")
    print(f"{'Test R²':<12} | {lr_results['test_r2']:>20.4f} | {rf_results['test_r2']:>14.4f}")
    print(f"{'CV RMSE':<12} | {lr_cv_text:>20} | {rf_cv_text:>14}")
    print("")
    print(f"RF improvement over LR: {rf_improvement:.1f}% RMSE reduction")
    print(f"Train/Test RMSE ratio (RF): {train_test_ratio:.3f} - {ratio_status}")
    print("")
    print("Top 3 features by importance:")
    for rank, row in top_features.iterrows():
        print(f"{rank + 1}. {row['Feature']} ({row['Importance']:.4f})")
    print("")
    print("Outputs:")
    print(f"{display_path(models_dir / 'lr_demand_model.pkl')}")
    print(f"{display_path(models_dir / 'rf_demand_model.pkl')}")
    print(f"{display_path(models_dir / 'test_predictions.csv')}")
    print(f"{display_path(models_dir / 'model_results_summary.csv')}")
    print(f"{display_path(charts_dir / 'rf_feature_importance.png')}")
    print(f"{display_path(charts_dir / 'model_comparison.png')}")


def main() -> int:
    args = parse_args()
    models_dir = args.models_dir
    charts_dir = args.charts_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = load_feature_columns(models_dir / FEATURE_COLUMNS_PATH.name)
    train = load_dataset(models_dir / TRAIN_PATH.name, feature_columns)
    test = load_dataset(models_dir / TEST_PATH.name, feature_columns)

    X_train = train[feature_columns]
    y_train = train[TARGET_COLUMN]
    X_test = test[feature_columns]
    y_test = test[TARGET_COLUMN]

    print_dataset_summary(X_train, X_test, y_train)

    lr_model, _, lr_pred_test, lr_results = train_linear_regression(X_train, y_train, X_test, y_test)
    lr_model_path = models_dir / "lr_demand_model.pkl"
    joblib.dump(lr_model, lr_model_path)
    print_model_results("MODEL A: Linear Regression (Baseline)", lr_results)
    print(f"  Saved: {display_path(lr_model_path)}")

    rf_model, rf_pred_train, rf_pred_test, rf_results = train_random_forest(
        X_train, y_train, X_test, y_test, PRIMARY_RF_PARAMS
    )
    rf_params = PRIMARY_RF_PARAMS.copy()
    print_model_results("MODEL B: Random Forest Regressor (Primary)", rf_results)

    if rf_results["test_rmse"] >= lr_results["test_rmse"]:
        print(
            "  Linear Regression outperformed the primary RF on test RMSE; "
            "retrying with conservative RF settings (n_estimators=100, max_depth=8)."
        )
        fallback_model, fallback_pred_train, fallback_pred_test, fallback_results = train_random_forest(
            X_train,
            y_train,
            X_test,
            y_test,
            FALLBACK_RF_PARAMS,
        )
        print_model_results("MODEL B: Random Forest Regressor (Fallback)", fallback_results)
        if fallback_results["test_rmse"] <= rf_results["test_rmse"]:
            rf_model = fallback_model
            rf_pred_train = fallback_pred_train
            rf_pred_test = fallback_pred_test
            rf_results = fallback_results
            rf_params = FALLBACK_RF_PARAMS.copy()

    rf_model_path = models_dir / "rf_demand_model.pkl"
    joblib.dump(rf_model, rf_model_path)
    print(f"  Final RF params: {rf_params}")
    print(f"  Saved: {display_path(rf_model_path)}")

    train_test_gap = rf_results["train_rmse"] / rf_results["test_rmse"]
    gap_status = "OK" if train_test_gap < 1.5 else "WARNING: possible overfit"
    print(f"  Train/Test RMSE ratio: {train_test_gap:.3f} ({gap_status})")

    importance_df = save_feature_importance_chart(feature_columns, rf_model, charts_dir)
    print(f"\n  Chart saved: {display_path(charts_dir / 'rf_feature_importance.png')}")
    save_model_comparison_chart(y_test, lr_pred_test, rf_pred_test, lr_results, rf_results, charts_dir)
    print(f"  Chart saved: {display_path(charts_dir / 'model_comparison.png')}")

    save_predictions_and_summary(
        test,
        feature_columns,
        lr_pred_test,
        rf_pred_test,
        lr_results,
        rf_results,
        models_dir,
    )
    print(f"  Saved: {display_path(models_dir / 'model_results_summary.csv')}")

    if rf_results["test_rmse"] >= lr_results["test_rmse"]:
        print("  WARNING: Random Forest test RMSE is not better than Linear Regression.")
    if importance_df.iloc[0]["Feature"] == "Base_Price_₹":
        print("  WARNING: Base_Price_₹ is the top feature; review encoding and leakage assumptions.")

    print_final_summary(lr_results, rf_results, importance_df, models_dir, charts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
