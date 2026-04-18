from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DAILY_PATH = ROOT_DIR / "data" / "data_analysis" / "daily_aggregated.csv"
DEFAULT_PROCESSED_PATH = ROOT_DIR / "data" / "data_analysis" / "processed_transactions.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "data_analysis"
DEFAULT_CHARTS_DIR = DEFAULT_OUTPUT_DIR / "charts"

HOT_BEVERAGE_CATEGORIES = {"Hot Beverages", "Hot Brews", "Hot Chocolate"}
SIGNIFICANCE_LEVEL = 0.05

REQUIRED_DAILY_COLUMNS = [
    "Date",
    "Temperature_°C",
    "Cold_Brew_Quantity",
    "Hot_Beverage_Quantity",
    "Daily_Footfall_Customers",
    "Total_Revenue_₹",
    "Is_Weekend_Flag",
    "Premium_Item_Revenue_₹",
]

REQUIRED_PROCESSED_COLUMNS = [
    "Date",
    "Category",
    "Quantity_Units",
    "Price_Tier",
    "Revenue_₹",
    "Daily_Footfall_Customers",
    "Is_Weekend",
]

PRIORITY_PAIRS = [
    {
        "x": "Temperature_°C",
        "y": "Cold_Brew_Quantity",
        "expected_direction": "Strong positive",
        "expected_range": (0.70, 0.78),
    },
    {
        "x": "Temperature_°C",
        "y": "Hot_Beverage_Quantity",
        "expected_direction": "Strong negative",
        "expected_range": (-0.70, -0.60),
    },
    {
        "x": "Daily_Footfall_Customers",
        "y": "Total_Revenue_₹",
        "expected_direction": "Strong positive",
        "expected_range": (0.78, 0.85),
    },
    {
        "x": "Is_Weekend_Flag",
        "y": "Premium_Item_Revenue_₹",
        "expected_direction": "Moderate positive",
        "expected_range": None,
    },
    {
        "x": "Daily_Footfall_Customers",
        "y": "Premium_Item_Revenue_₹",
        "expected_direction": "Moderate positive",
        "expected_range": None,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Pearson correlation analysis for Blend Cafe daily metrics."
    )
    parser.add_argument(
        "--daily-path",
        type=Path,
        default=DEFAULT_DAILY_PATH,
        help="Path to daily_aggregated.csv",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help="Path to processed_transactions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where correlation_matrix.csv will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where correlation heatmap PNG will be written",
    )
    return parser.parse_args()


def load_daily_aggregated(path: Path) -> pd.DataFrame:
    daily = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_DAILY_COLUMNS if column not in daily.columns]
    if missing_columns:
        raise ValueError(f"daily_aggregated.csv is missing required columns: {missing_columns}")

    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="raise")
    numeric_columns = daily.select_dtypes(include=np.number).columns.tolist()
    for column in numeric_columns:
        daily[column] = pd.to_numeric(daily[column], errors="raise")

    return daily.sort_values("Date").reset_index(drop=True)


def load_processed_transactions(path: Path) -> pd.DataFrame:
    processed = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_PROCESSED_COLUMNS if column not in processed.columns]
    if missing_columns:
        raise ValueError(
            f"processed_transactions.csv is missing required columns: {missing_columns}"
        )

    processed = processed.copy()
    processed["Date"] = pd.to_datetime(processed["Date"], errors="raise")
    processed["Quantity_Units"] = pd.to_numeric(processed["Quantity_Units"], errors="raise").astype(int)
    processed["Revenue_₹"] = pd.to_numeric(processed["Revenue_₹"], errors="raise")
    processed["Daily_Footfall_Customers"] = pd.to_numeric(
        processed["Daily_Footfall_Customers"], errors="raise"
    ).astype(int)
    processed["Category"] = processed["Category"].astype("string").str.strip()
    processed["Price_Tier"] = processed["Price_Tier"].astype("string").str.strip()
    processed["Is_Weekend"] = processed["Is_Weekend"].astype("string").str.strip()

    return processed.sort_values(["Date"]).reset_index(drop=True)


def build_processed_daily_validation(processed: pd.DataFrame) -> pd.DataFrame:
    validation = (
        processed.groupby("Date", as_index=False)
        .agg(
            **{
                "Daily_Footfall_Customers": ("Daily_Footfall_Customers", "first"),
                "Total_Revenue_₹": ("Revenue_₹", "sum"),
                "Is_Weekend_Flag": ("Is_Weekend", lambda values: int(values.iloc[0] == "Yes")),
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    enrichments = [
        quantity_series(
            processed.loc[processed["Category"].eq("Cold Brews")],
            "Cold_Brew_Quantity",
        ),
        quantity_series(
            processed.loc[processed["Category"].isin(HOT_BEVERAGE_CATEGORIES)],
            "Hot_Beverage_Quantity",
        ),
        revenue_series(
            processed.loc[processed["Price_Tier"].eq("Premium")],
            "Premium_Item_Revenue_₹",
        ),
    ]

    validation = validation.set_index("Date")
    for series in enrichments:
        validation = validation.join(series, how="left")

    for column in [
        "Cold_Brew_Quantity",
        "Hot_Beverage_Quantity",
        "Premium_Item_Revenue_₹",
    ]:
        validation[column] = validation[column].fillna(0).astype(int)

    return validation.reset_index()


def quantity_series(frame: pd.DataFrame, output_name: str) -> pd.Series:
    return frame.groupby("Date")["Quantity_Units"].sum().rename(output_name)


def revenue_series(frame: pd.DataFrame, output_name: str) -> pd.Series:
    return frame.groupby("Date")["Revenue_₹"].sum().rename(output_name)


def validate_daily_contract(daily: pd.DataFrame, processed: pd.DataFrame) -> None:
    processed_daily = build_processed_daily_validation(processed)
    comparison = daily.merge(
        processed_daily,
        on="Date",
        how="inner",
        suffixes=("_daily", "_processed"),
        validate="one_to_one",
    )

    if len(comparison) != len(daily):
        raise ValueError("daily_aggregated.csv and processed_transactions.csv do not align by Date")

    columns_to_validate = [
        "Daily_Footfall_Customers",
        "Total_Revenue_₹",
        "Is_Weekend_Flag",
        "Cold_Brew_Quantity",
        "Hot_Beverage_Quantity",
        "Premium_Item_Revenue_₹",
    ]
    for column in columns_to_validate:
        daily_values = comparison[f"{column}_daily"].to_numpy()
        processed_values = comparison[f"{column}_processed"].to_numpy()
        if not np.array_equal(daily_values, processed_values):
            raise ValueError(
                f"{column} in daily_aggregated.csv does not match processed_transactions.csv"
            )


def build_pairwise_correlation_table(daily: pd.DataFrame) -> pd.DataFrame:
    numeric_df = daily.select_dtypes(include=np.number).copy()
    records: list[dict[str, object]] = []

    for left, right in combinations(numeric_df.columns, 2):
        pair_data = numeric_df[[left, right]].dropna()
        if pair_data[left].nunique() < 2 or pair_data[right].nunique() < 2:
            records.append(
                {
                    "Variable_X": left,
                    "Variable_Y": right,
                    "Correlation_r": np.nan,
                    "P_Value": np.nan,
                    "Significant_0.05": False,
                    "Absolute_r": np.nan,
                }
            )
            continue

        r_value, p_value = pearsonr(pair_data[left], pair_data[right])
        records.append(
            {
                "Variable_X": left,
                "Variable_Y": right,
                "Correlation_r": round(float(r_value), 6),
                "P_Value": float(p_value),
                "Significant_0.05": bool(p_value < SIGNIFICANCE_LEVEL),
                "Absolute_r": round(abs(float(r_value)), 6),
            }
        )

    correlation_table = pd.DataFrame(records).sort_values(
        ["Absolute_r", "Variable_X", "Variable_Y"], ascending=[False, True, True]
    )
    return correlation_table.reset_index(drop=True)


def build_priority_results(correlation_table: pd.DataFrame) -> list[dict[str, object]]:
    indexed = correlation_table.set_index(["Variable_X", "Variable_Y"])
    results: list[dict[str, object]] = []

    for pair in PRIORITY_PAIRS:
        key = (pair["x"], pair["y"])
        reverse_key = (pair["y"], pair["x"])
        if key in indexed.index:
            selected = indexed.loc[key]
        elif reverse_key in indexed.index:
            selected = indexed.loc[reverse_key]
        else:
            raise ValueError(f"Priority pair missing from correlation table: {key}")

        result = selected.to_dict()
        result["Variable_X"] = pair["x"]
        result["Variable_Y"] = pair["y"]
        result["Expected_Direction"] = pair["expected_direction"]
        result["Expected_Range"] = pair["expected_range"]
        results.append(result)

    return results


def build_verification_warnings(priority_results: list[dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    for result in priority_results:
        r_value = float(result["Correlation_r"])
        p_value = float(result["P_Value"])
        expected_range = result["Expected_Range"]
        pair_label = f"{result['Variable_X']} vs {result['Variable_Y']}"

        if p_value >= SIGNIFICANCE_LEVEL:
            warnings.append(
                f"{pair_label} is not significant at p < 0.05 (r={r_value:.3f}, p={p_value:.6f})."
            )
            continue

        if expected_range is None:
            continue

        lower, upper = expected_range
        if not (lower <= r_value <= upper):
            warnings.append(
                f"{pair_label} fell outside the narrative range {lower:.2f} to {upper:.2f} "
                f"(actual r={r_value:.3f}, p={p_value:.6f})."
            )

    return warnings


def save_correlation_table(correlation_table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    correlation_table.to_csv(output_dir / "correlation_matrix.csv", index=False)


def save_correlation_heatmap(daily: pd.DataFrame, charts_dir: Path) -> None:
    numeric_df = daily.select_dtypes(include=np.number)
    correlation_matrix = numeric_df.corr(method="pearson")
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        vmin=-1,
        vmax=1,
        center=0,
        square=False,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix (Daily Aggregated Numeric Features)", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(charts_dir / "heatmap_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    output_dir: Path,
    charts_dir: Path,
    priority_results: list[dict[str, object]],
    warnings: list[str],
) -> None:
    print("04_correlation_analysis.py complete.")
    print("Significant priority correlations:")

    significant_results = [result for result in priority_results if result["P_Value"] < SIGNIFICANCE_LEVEL]
    if significant_results:
        for result in significant_results:
            print(
                f"- {result['Variable_X']} vs {result['Variable_Y']}: "
                f"r={float(result['Correlation_r']):+.3f}, p={float(result['P_Value']):.6f}"
            )
    else:
        print("- None at p < 0.05")

    nonsignificant_results = [
        result for result in priority_results if result["P_Value"] >= SIGNIFICANCE_LEVEL
    ]
    if nonsignificant_results:
        print("Non-significant priority pairs:")
        for result in nonsignificant_results:
            print(
                f"- {result['Variable_X']} vs {result['Variable_Y']}: "
                f"r={float(result['Correlation_r']):+.3f}, p={float(result['P_Value']):.6f}"
            )

    if warnings:
        print("Verification warnings:")
        for warning in warnings:
            print(f"- {warning}")

    print("Files saved:")
    print(display_path(output_dir / "correlation_matrix.csv"))
    print(display_path(charts_dir / "heatmap_correlation_matrix.png"))


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def main() -> int:
    args = parse_args()
    sns.set_theme(style="white")

    daily = load_daily_aggregated(args.daily_path)
    processed = load_processed_transactions(args.processed_path)
    validate_daily_contract(daily, processed)

    correlation_table = build_pairwise_correlation_table(daily)
    priority_results = build_priority_results(correlation_table)
    warnings = build_verification_warnings(priority_results)

    save_correlation_table(correlation_table, args.output_dir)
    save_correlation_heatmap(daily, args.charts_dir)
    print_summary(args.output_dir, args.charts_dir, priority_results, warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
