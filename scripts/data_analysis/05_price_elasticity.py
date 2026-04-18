from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PIVOT_PATH = ROOT_DIR / "data" / "data_analysis" / "pivot_quantity_weather_category.csv"
DEFAULT_PROCESSED_PATH = ROOT_DIR / "data" / "data_analysis" / "processed_transactions.csv"
DEFAULT_ABC_PATH = ROOT_DIR / "data" / "data_analysis" / "abc_classified.csv"
DEFAULT_WORKBOOK_PATH = ROOT_DIR / "data" / "obs" / "BlendCafe_DynamicPricing_Data.xlsx"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "data_analysis"
DEFAULT_CHARTS_DIR = DEFAULT_OUTPUT_DIR / "charts"

WEATHER_ORDER = ["Dry Day", "Light Drizzle", "Heavy Rain", "Clear After Rain"]
ITEM_MASTER_COLUMNS = ["Item_Name", "Category", "Price_Tier", "Base_Price_₹"]
ELASTICITY_CLASS_COLORS = {
    "Highly Elastic": "#d62728",
    "Moderately Elastic": "#ff7f0e",
    "Inelastic": "#2ca02c",
    "Fixed Demand": "#aec7e8",
}
IMPLIED_PRICE_SWING_PCT = 0.15
TOTAL_PRICE_SWING_RANGE = IMPLIED_PRICE_SWING_PCT * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute condition-based demand elasticity for Blend Cafe items."
    )
    parser.add_argument(
        "--pivot-path",
        type=Path,
        default=DEFAULT_PIVOT_PATH,
        help="Path to pivot_quantity_weather_category.csv",
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help="Path to processed_transactions.csv",
    )
    parser.add_argument(
        "--abc-path",
        type=Path,
        default=DEFAULT_ABC_PATH,
        help="Path to abc_classified.csv",
    )
    parser.add_argument(
        "--workbook-path",
        type=Path,
        default=DEFAULT_WORKBOOK_PATH,
        help="Path to BlendCafe_DynamicPricing_Data.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where elasticity_coefficients.csv will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where elasticity_distribution.png will be written",
    )
    return parser.parse_args()


def load_category_weather_pivot(path: Path) -> pd.DataFrame:
    pivot = pd.read_csv(path, index_col=0)
    pivot.index = pivot.index.astype("string").str.strip()
    pivot.columns = [str(column).strip() for column in pivot.columns]
    missing_columns = [column for column in WEATHER_ORDER if column not in pivot.columns]
    if missing_columns:
        raise ValueError(
            f"pivot_quantity_weather_category.csv is missing weather columns: {missing_columns}"
        )

    pivot = pivot.reindex(columns=WEATHER_ORDER).fillna(0).astype(int)
    return pivot


def load_processed_transactions(path: Path) -> pd.DataFrame:
    processed = pd.read_csv(path)
    required_columns = [
        "Item_Name",
        "Category",
        "Weather",
        "Quantity_Units",
        "ABC_Class",
        "Base_Price_₹",
        "Price_Tier",
    ]
    missing_columns = [column for column in required_columns if column not in processed.columns]
    if missing_columns:
        raise ValueError(
            f"processed_transactions.csv is missing required columns: {missing_columns}"
        )

    processed = processed.copy()
    for column in ["Item_Name", "Category", "Weather", "ABC_Class", "Price_Tier"]:
        processed[column] = processed[column].astype("string").str.strip()
        if processed[column].isna().any() or processed[column].eq("").any():
            raise ValueError(f"{column} contains null or blank values")

    processed["Quantity_Units"] = pd.to_numeric(
        processed["Quantity_Units"], errors="raise"
    ).astype(int)
    processed["Base_Price_₹"] = pd.to_numeric(processed["Base_Price_₹"], errors="raise").astype(int)

    return processed


def load_abc_summary(path: Path) -> pd.DataFrame:
    abc = pd.read_csv(path)
    required_columns = ["Item_Name", "ABC_Class"]
    missing_columns = [column for column in required_columns if column not in abc.columns]
    if missing_columns:
        raise ValueError(f"abc_classified.csv is missing required columns: {missing_columns}")

    abc = abc[required_columns].copy()
    abc["Item_Name"] = abc["Item_Name"].astype("string").str.strip()
    abc["ABC_Class"] = abc["ABC_Class"].astype("string").str.strip()
    return abc


def load_item_master(path: Path) -> pd.DataFrame:
    item_master = pd.read_excel(path, sheet_name="Item_Master", usecols=ITEM_MASTER_COLUMNS)
    item_master = item_master.copy()
    for column in ["Item_Name", "Category", "Price_Tier"]:
        item_master[column] = item_master[column].astype("string").str.strip()
    item_master["Base_Price_₹"] = pd.to_numeric(
        item_master["Base_Price_₹"], errors="raise"
    ).astype(int)

    if item_master["Item_Name"].duplicated().any():
        raise ValueError("Item_Master contains duplicate Item_Name values")

    return item_master


def validate_weather_pivot(category_weather_pivot: pd.DataFrame, processed: pd.DataFrame) -> None:
    processed_pivot = processed.pivot_table(
        index="Category",
        columns="Weather",
        values="Quantity_Units",
        aggfunc="sum",
        fill_value=0,
    )
    processed_pivot = processed_pivot.reindex(columns=WEATHER_ORDER, fill_value=0).astype(int)
    processed_pivot = processed_pivot.loc[
        processed_pivot.sum(axis=1).sort_values(ascending=False).index
    ]

    if not category_weather_pivot.equals(processed_pivot):
        raise ValueError(
            "pivot_quantity_weather_category.csv does not match processed_transactions.csv"
        )


def validate_item_contracts(
    processed: pd.DataFrame, item_master: pd.DataFrame, abc_summary: pd.DataFrame
) -> None:
    if processed["Item_Name"].nunique() != 200:
        raise ValueError("Expected 200 unique items in processed_transactions.csv")
    if len(item_master) != 200:
        raise ValueError("Expected 200 rows in Item_Master")
    if len(abc_summary) != 200:
        raise ValueError("Expected 200 rows in abc_classified.csv")

    merged = processed.merge(
        item_master,
        on="Item_Name",
        how="left",
        suffixes=("_processed", "_master"),
    )
    if merged["Category_master"].isna().any():
        raise ValueError("Some processed items are missing from Item_Master")

    for column in ["Category", "Price_Tier", "Base_Price_₹"]:
        mismatch_mask = merged[f"{column}_processed"] != merged[f"{column}_master"]
        if mismatch_mask.any():
            raise ValueError(f"{column} mismatch between processed_transactions.csv and Item_Master")

    abc_join = processed[["Item_Name", "ABC_Class"]].drop_duplicates().merge(
        abc_summary, on="Item_Name", how="inner", suffixes=("_processed", "_abc")
    )
    if len(abc_join) != 200:
        raise ValueError("ABC summary does not align with processed transactions for all items")
    if abc_join["ABC_Class_processed"].ne(abc_join["ABC_Class_abc"]).any():
        raise ValueError("ABC_Class mismatch between processed_transactions.csv and abc_classified.csv")


def build_item_weather_matrix(processed: pd.DataFrame) -> pd.DataFrame:
    item_weather = (
        processed.groupby(["Item_Name", "Weather"])["Quantity_Units"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=WEATHER_ORDER, fill_value=0)
        .astype(int)
    )
    return item_weather


def build_elasticity_dataframe(
    processed: pd.DataFrame, item_master: pd.DataFrame, abc_summary: pd.DataFrame
) -> pd.DataFrame:
    item_weather = build_item_weather_matrix(processed)

    dry_qty = item_weather["Dry Day"]
    heavy_rain_qty = item_weather["Heavy Rain"]
    quantity_baseline = heavy_rain_qty.replace(0, 0.5)
    quantity_swing_pct = (dry_qty - heavy_rain_qty).abs() / quantity_baseline
    elasticity_coefficient = quantity_swing_pct / TOTAL_PRICE_SWING_RANGE

    demand_std = item_weather.std(axis=1)
    demand_mean = item_weather.mean(axis=1)
    demand_cv = demand_std / demand_mean.replace(0, 0.001)
    clear_rain_multiplier = item_weather["Clear After Rain"] / demand_mean.replace(0, 0.001)
    peak_weather = item_weather.idxmax(axis=1)
    rain_boost_flag = heavy_rain_qty > dry_qty

    base_frame = (
        item_master.set_index("Item_Name")
        .join(abc_summary.set_index("Item_Name")[["ABC_Class"]], how="inner")
        .join(item_weather.rename(columns=weather_qty_columns()), how="inner")
    )

    elasticity_df = base_frame.assign(
        Elasticity_Coefficient=elasticity_coefficient.round(4),
        Elasticity_Class=lambda frame: frame["Elasticity_Coefficient"].apply(classify_elasticity),
        Demand_CV=demand_cv.round(4),
        Rain_Boost_Flag=rain_boost_flag,
        Clear_Rain_Multiplier=clear_rain_multiplier.round(4),
        Peak_Weather_State=peak_weather,
    ).reset_index()

    output_columns = [
        "Item_Name",
        "Category",
        "Price_Tier",
        "Base_Price_₹",
        "ABC_Class",
        "Qty_Dry_Day",
        "Qty_Light_Drizzle",
        "Qty_Heavy_Rain",
        "Qty_Clear_After_Rain",
        "Elasticity_Coefficient",
        "Elasticity_Class",
        "Demand_CV",
        "Rain_Boost_Flag",
        "Clear_Rain_Multiplier",
        "Peak_Weather_State",
    ]
    return elasticity_df[output_columns].sort_values(
        ["Elasticity_Coefficient", "Item_Name"], ascending=[False, True]
    ).reset_index(drop=True)


def weather_qty_columns() -> dict[str, str]:
    return {
        "Dry Day": "Qty_Dry_Day",
        "Light Drizzle": "Qty_Light_Drizzle",
        "Heavy Rain": "Qty_Heavy_Rain",
        "Clear After Rain": "Qty_Clear_After_Rain",
    }


def classify_elasticity(value: float) -> str:
    if value >= 2.0:
        return "Highly Elastic"
    if value >= 1.0:
        return "Moderately Elastic"
    if value >= 0.3:
        return "Inelastic"
    return "Fixed Demand"


def save_elasticity_csv(elasticity_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    elasticity_df.to_csv(output_dir / "elasticity_coefficients.csv", index=False)


def save_elasticity_chart(elasticity_df: pd.DataFrame, charts_dir: Path) -> None:
    top30 = elasticity_df.nlargest(30, "Elasticity_Coefficient").sort_values(
        "Elasticity_Coefficient", ascending=True
    )
    colors = [ELASTICITY_CLASS_COLORS[label] for label in top30["Elasticity_Class"]]

    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(top30["Item_Name"], top30["Elasticity_Coefficient"], color=colors)

    for bar, value in zip(bars, top30["Elasticity_Coefficient"]):
        ax.text(
            value + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    legend_handles = [
        Patch(facecolor=color, label=label) for label, color in ELASTICITY_CLASS_COLORS.items()
    ]
    ax.legend(handles=legend_handles, title="Elasticity Class", loc="lower right")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Unit Elasticity")
    ax.axvline(2.0, color="black", linestyle=":", linewidth=1.2, label="Highly Elastic Threshold")
    ax.set_title("Top 30 Items by Demand Elasticity Coefficient", pad=12)
    ax.set_xlabel("Elasticity_Coefficient")
    ax.set_ylabel("Item_Name")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_xlim(0, max(top30["Elasticity_Coefficient"].max() * 1.15, 2.5))
    plt.tight_layout()
    fig.savefig(charts_dir / "elasticity_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(elasticity_df: pd.DataFrame, output_dir: Path, charts_dir: Path) -> None:
    class_counts = elasticity_df["Elasticity_Class"].value_counts()
    print("05_price_elasticity.py complete.")
    print("")
    print(f"Items analysed: {len(elasticity_df)}")
    print(f"Highly Elastic:      {class_counts.get('Highly Elastic', 0)} items")
    print(f"Moderately Elastic:  {class_counts.get('Moderately Elastic', 0)} items")
    print(f"Inelastic:           {class_counts.get('Inelastic', 0)} items")
    print(f"Fixed Demand:        {class_counts.get('Fixed Demand', 0)} items")
    print("")
    print(
        "Rain boost items (demand increases in Heavy Rain): "
        f"{int(elasticity_df['Rain_Boost_Flag'].sum())} items"
    )
    print("")
    print(f"Output: {display_path(output_dir / 'elasticity_coefficients.csv')}")
    print(f"Chart:  {display_path(charts_dir / 'elasticity_distribution.png')}")


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def main() -> int:
    args = parse_args()

    category_weather_pivot = load_category_weather_pivot(args.pivot_path)
    processed = load_processed_transactions(args.processed_path)
    abc_summary = load_abc_summary(args.abc_path)
    item_master = load_item_master(args.workbook_path)

    validate_weather_pivot(category_weather_pivot, processed)
    validate_item_contracts(processed, item_master, abc_summary)

    elasticity_df = build_elasticity_dataframe(processed, item_master, abc_summary)
    save_elasticity_csv(elasticity_df, args.output_dir)
    save_elasticity_chart(elasticity_df, args.charts_dir)
    print_summary(elasticity_df, args.output_dir, args.charts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
