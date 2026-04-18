from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ELASTICITY_PATH = ROOT_DIR / "data" / "data_analysis" / "elasticity_coefficients.csv"
DEFAULT_PROCESSED_PATH = ROOT_DIR / "data" / "data_analysis" / "processed_transactions.csv"
DEFAULT_ABC_PATH = ROOT_DIR / "data" / "data_analysis" / "abc_classified.csv"
DEFAULT_PIVOT_AOV_PATH = ROOT_DIR / "data" / "data_analysis" / "pivot_aov_slot_day.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "data_analysis"
DEFAULT_CHARTS_DIR = DEFAULT_OUTPUT_DIR / "charts"

SEGMENT_ORDER = [
    "Dynamic Pricing Target",
    "Premium Lock",
    "Promotional Item",
    "Review Item",
]
SEGMENT_COLORS = {
    "Dynamic Pricing Target": "#d62728",
    "Premium Lock": "#2ca02c",
    "Promotional Item": "#ff7f0e",
    "Review Item": "#7f7f7f",
}
ABC_COLORS = {"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c"}
SIZE_MAP = {"A": 120, "B": 60, "C": 25}
ELASTICITY_THRESHOLD = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment Blend Cafe items into a demand-elasticity 2x2 matrix."
    )
    parser.add_argument(
        "--elasticity-path",
        type=Path,
        default=DEFAULT_ELASTICITY_PATH,
        help="Path to elasticity_coefficients.csv",
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
        "--pivot-aov-path",
        type=Path,
        default=DEFAULT_PIVOT_AOV_PATH,
        help="Path to pivot_aov_slot_day.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where demand_segments.csv will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where segmentation charts will be written",
    )
    return parser.parse_args()


def load_elasticity(path: Path) -> pd.DataFrame:
    elasticity = pd.read_csv(path)
    required_columns = [
        "Item_Name",
        "Category",
        "Price_Tier",
        "Base_Price_₹",
        "ABC_Class",
        "Elasticity_Coefficient",
        "Elasticity_Class",
        "Rain_Boost_Flag",
    ]
    missing_columns = [column for column in required_columns if column not in elasticity.columns]
    if missing_columns:
        raise ValueError(
            f"elasticity_coefficients.csv is missing required columns: {missing_columns}"
        )

    elasticity = elasticity.copy()
    for column in ["Item_Name", "Category", "Price_Tier", "ABC_Class", "Elasticity_Class"]:
        elasticity[column] = elasticity[column].astype("string").str.strip()
    elasticity["Base_Price_₹"] = pd.to_numeric(
        elasticity["Base_Price_₹"], errors="raise"
    ).astype(int)
    elasticity["Elasticity_Coefficient"] = pd.to_numeric(
        elasticity["Elasticity_Coefficient"], errors="raise"
    )

    if len(elasticity) != 200:
        raise ValueError(f"Expected 200 rows in elasticity_coefficients.csv, found {len(elasticity)}")
    if elasticity["Item_Name"].duplicated().any():
        raise ValueError("elasticity_coefficients.csv contains duplicate Item_Name values")

    return elasticity


def load_transactions(path: Path) -> pd.DataFrame:
    transactions = pd.read_csv(path)
    required_columns = ["Item_Name", "Date", "Quantity_Units", "Time_Slot"]
    missing_columns = [column for column in required_columns if column not in transactions.columns]
    if missing_columns:
        raise ValueError(
            f"processed_transactions.csv is missing required columns: {missing_columns}"
        )

    transactions = transactions.copy()
    transactions["Date"] = pd.to_datetime(transactions["Date"], errors="raise")
    transactions["Item_Name"] = transactions["Item_Name"].astype("string").str.strip()
    transactions["Time_Slot"] = transactions["Time_Slot"].astype("string").str.strip()
    transactions["Quantity_Units"] = pd.to_numeric(
        transactions["Quantity_Units"], errors="raise"
    ).astype(int)

    if len(transactions) != 3818:
        raise ValueError(f"Expected 3818 transaction rows, found {len(transactions)}")

    return transactions


def load_abc(path: Path) -> pd.DataFrame:
    abc = pd.read_csv(path)
    required_columns = ["Item_Name", "Total_Revenue_₹", "Transaction_Count"]
    missing_columns = [column for column in required_columns if column not in abc.columns]
    if missing_columns:
        raise ValueError(f"abc_classified.csv is missing required columns: {missing_columns}")

    abc = abc.copy()
    abc["Item_Name"] = abc["Item_Name"].astype("string").str.strip()
    abc["Total_Revenue_₹"] = pd.to_numeric(abc["Total_Revenue_₹"], errors="raise")
    abc["Transaction_Count"] = pd.to_numeric(abc["Transaction_Count"], errors="raise").astype(int)
    return abc


def load_pivot_aov(path: Path) -> list[str]:
    pivot_aov = pd.read_csv(path, index_col=0)
    slot_order = pivot_aov.index.astype("string").str.strip().tolist()
    if not slot_order:
        raise ValueError("pivot_aov_slot_day.csv does not contain any time slot rows")
    return slot_order


def compute_demand_level(transactions: pd.DataFrame) -> pd.DataFrame:
    item_daily = (
        transactions.groupby(["Item_Name", "Date"])["Quantity_Units"].sum().reset_index()
    )
    demand_level = (
        item_daily.groupby("Item_Name")["Quantity_Units"]
        .median()
        .reset_index()
        .rename(columns={"Quantity_Units": "Median_Daily_Qty"})
    )
    return demand_level


def compute_peak_slot(transactions: pd.DataFrame, slot_order: list[str]) -> pd.DataFrame:
    slot_rank = {slot: index for index, slot in enumerate(slot_order)}
    item_slot = (
        transactions.groupby(["Item_Name", "Time_Slot"])["Quantity_Units"].sum().reset_index()
    )
    item_slot["Slot_Order"] = item_slot["Time_Slot"].map(slot_rank).fillna(len(slot_rank))
    peak_slot = (
        item_slot.sort_values(
            ["Item_Name", "Quantity_Units", "Slot_Order"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset="Item_Name", keep="first")
        [["Item_Name", "Time_Slot"]]
        .rename(columns={"Time_Slot": "Peak_Time_Slot"})
    )
    return peak_slot


def merge_segmentation_frame(
    elasticity: pd.DataFrame,
    demand_level: pd.DataFrame,
    peak_slot: pd.DataFrame,
    abc: pd.DataFrame,
) -> pd.DataFrame:
    df = elasticity[
        [
            "Item_Name",
            "Category",
            "Price_Tier",
            "Base_Price_₹",
            "ABC_Class",
            "Elasticity_Coefficient",
            "Elasticity_Class",
            "Rain_Boost_Flag",
        ]
    ].copy()
    df = df.merge(demand_level, on="Item_Name", how="left")
    df = df.merge(peak_slot, on="Item_Name", how="left")
    df = df.merge(abc[["Item_Name", "Total_Revenue_₹", "Transaction_Count"]], on="Item_Name", how="left")

    if len(df) != 200:
        raise ValueError(f"Merged segmentation dataframe should have 200 rows, found {len(df)}")
    if df["Median_Daily_Qty"].isna().any():
        raise ValueError("Median_Daily_Qty contains null values after merge")
    if df["Peak_Time_Slot"].isna().any():
        raise ValueError("Peak_Time_Slot contains null values after merge")
    if df["Total_Revenue_₹"].isna().any() or df["Transaction_Count"].isna().any():
        raise ValueError("ABC metrics contain null values after merge")

    return df


def assign_demand_classes(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    demand_threshold = float(df["Median_Daily_Qty"].median())
    ordered = df.sort_values(
        ["Median_Daily_Qty", "Total_Revenue_₹", "Item_Name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    half_point = len(ordered) // 2
    ordered["Demand_Class"] = np.where(ordered.index < half_point, "High Demand", "Low Demand")
    return ordered, demand_threshold


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Elasticity_Group"] = np.where(
        df["Elasticity_Coefficient"] >= ELASTICITY_THRESHOLD,
        "High Elasticity",
        "Low Elasticity",
    )

    def assign_segment(row: pd.Series) -> str:
        if row["Demand_Class"] == "High Demand" and row["Elasticity_Group"] == "High Elasticity":
            return "Dynamic Pricing Target"
        if row["Demand_Class"] == "High Demand" and row["Elasticity_Group"] == "Low Elasticity":
            return "Premium Lock"
        if row["Demand_Class"] == "Low Demand" and row["Elasticity_Group"] == "High Elasticity":
            return "Promotional Item"
        return "Review Item"

    df["Segment"] = df.apply(assign_segment, axis=1)
    return df


def save_segments_csv(df: pd.DataFrame, output_dir: Path) -> None:
    output_cols = [
        "Item_Name",
        "Category",
        "Price_Tier",
        "Base_Price_₹",
        "ABC_Class",
        "Total_Revenue_₹",
        "Transaction_Count",
        "Median_Daily_Qty",
        "Demand_Class",
        "Elasticity_Coefficient",
        "Elasticity_Class",
        "Elasticity_Group",
        "Segment",
        "Peak_Time_Slot",
        "Rain_Boost_Flag",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    df[output_cols].sort_values(
        ["Segment", "ABC_Class", "Total_Revenue_₹"],
        ascending=[True, True, False],
    ).to_csv(output_dir / "demand_segments.csv", index=False)


def save_segmentation_scatter(df: pd.DataFrame, demand_threshold: float, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 10))

    for segment in SEGMENT_ORDER:
        group = df.loc[df["Segment"].eq(segment)]
        if group.empty:
            continue
        sizes = group["ABC_Class"].map(SIZE_MAP).fillna(25)
        ax.scatter(
            group["Median_Daily_Qty"],
            group["Elasticity_Coefficient"],
            c=SEGMENT_COLORS[segment],
            s=sizes,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            label=segment,
            zorder=3,
        )

    ax.axvline(x=demand_threshold, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axhline(y=ELASTICITY_THRESHOLD, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("Demand Level — Median Daily Quantity Sold", fontsize=12)
    ax.set_ylabel("Elasticity Coefficient", fontsize=12)
    ax.set_title("Demand Segmentation 2×2 — Blend Café Items", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Segment", loc="upper right")

    ax.text(demand_threshold + 0.03, ELASTICITY_THRESHOLD + 0.05, "Dynamic Pricing Target", fontsize=10)
    ax.text(demand_threshold + 0.03, max(df["Elasticity_Coefficient"].max() * 0.25, 0.25), "Premium Lock", fontsize=10)
    ax.text(max(df["Median_Daily_Qty"].min() - 0.1, 0), ELASTICITY_THRESHOLD + 0.05, "Promotional Item", fontsize=10)
    ax.text(max(df["Median_Daily_Qty"].min() - 0.1, 0), max(df["Elasticity_Coefficient"].max() * 0.25, 0.25), "Review Item", fontsize=10)

    plt.tight_layout()
    fig.savefig(charts_dir / "demand_segmentation_2x2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_segment_abc_breakdown(df: pd.DataFrame, charts_dir: Path) -> None:
    segment_abc = (
        df.groupby(["Segment", "ABC_Class"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SEGMENT_ORDER, columns=["A", "B", "C"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(segment_abc))
    x_labels = segment_abc.index.tolist()

    for abc_class in ["A", "B", "C"]:
        values = segment_abc[abc_class].to_numpy()
        bars = ax.bar(
            x_labels,
            values,
            bottom=bottom,
            color=ABC_COLORS[abc_class],
            edgecolor="black",
            linewidth=0.5,
            label=f"ABC {abc_class}",
        )
        for bar, value, base in zip(bars, values, bottom):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                )
        bottom += values

    ax.set_xlabel("Segment", fontsize=11)
    ax.set_ylabel("Item Count", fontsize=11)
    ax.set_title("ABC Breakdown Within Demand Segments", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(title="ABC Class", loc="upper right")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(charts_dir / "segment_abc_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def print_summary(df: pd.DataFrame, demand_threshold: float, output_dir: Path, charts_dir: Path) -> None:
    segment_counts = df["Segment"].value_counts()
    print("08_demand_segmentation.py complete.")
    print(f"Items segmented: {len(df)}")
    print(f"Demand threshold: {demand_threshold:.1f} units/day (median split)")
    print(f"Elasticity threshold: {ELASTICITY_THRESHOLD:.1f} (unit business rule)")
    print("")
    print("Segment distribution:")
    for segment in SEGMENT_ORDER:
        print(f"{segment} : {int(segment_counts.get(segment, 0))} items")
    print("")
    print(f"Output: {display_path(output_dir / 'demand_segments.csv')}")
    print(f"Charts: {display_path(charts_dir / 'demand_segmentation_2x2.png')}")
    print(f"{display_path(charts_dir / 'segment_abc_breakdown.png')}")


def main() -> int:
    args = parse_args()

    elasticity = load_elasticity(args.elasticity_path)
    transactions = load_transactions(args.processed_path)
    abc = load_abc(args.abc_path)
    slot_order = load_pivot_aov(args.pivot_aov_path)

    demand_level = compute_demand_level(transactions)
    peak_slot = compute_peak_slot(transactions, slot_order)
    df = merge_segmentation_frame(elasticity, demand_level, peak_slot, abc)
    df, demand_threshold = assign_demand_classes(df)
    df = assign_segments(df)

    save_segments_csv(df, args.output_dir)
    save_segmentation_scatter(df, demand_threshold, args.charts_dir)
    save_segment_abc_breakdown(df, args.charts_dir)
    print_summary(df, demand_threshold, args.output_dir, args.charts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
