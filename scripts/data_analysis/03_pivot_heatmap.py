from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "data_analysis" / "processed_transactions.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "data_analysis"
DEFAULT_CHARTS_DIR = DEFAULT_OUTPUT_DIR / "charts"

REQUIRED_COLUMNS = [
    "Time_Slot",
    "Day_of_Week",
    "Revenue_₹",
    "Weather",
    "Category",
    "Quantity_Units",
    "Price_Tier",
]

SLOT_ORDER = ["Morning", "Afternoon", "Evening", "Dinner"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER_ORDER = ["Dry Day", "Light Drizzle", "Heavy Rain", "Clear After Rain"]

PRICE_TIER_COLORS = {
    "Budget": "#2ca25f",
    "Mid-range": "#f28e2b",
    "Premium": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Blend Cafe pivot CSVs and heatmap/bar chart outputs."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to processed_transactions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where pivot CSVs will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where chart PNGs will be written",
    )
    return parser.parse_args()


def load_transactions(input_path: Path) -> pd.DataFrame:
    transactions = pd.read_csv(input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in transactions.columns]
    if missing_columns:
        raise ValueError(
            f"processed_transactions.csv is missing required columns: {missing_columns}"
        )

    transactions = transactions.copy()
    for column in ["Time_Slot", "Day_of_Week", "Weather", "Category", "Price_Tier"]:
        transactions[column] = transactions[column].astype("string").str.strip()
        if transactions[column].isna().any() or transactions[column].eq("").any():
            raise ValueError(f"{column} contains null or blank values")

    transactions["Revenue_₹"] = pd.to_numeric(transactions["Revenue_₹"], errors="raise")
    transactions["Quantity_Units"] = pd.to_numeric(
        transactions["Quantity_Units"], errors="raise"
    ).astype(int)

    return transactions


def build_revenue_pivot(transactions: pd.DataFrame) -> pd.DataFrame:
    pivot = transactions.pivot_table(
        index="Time_Slot",
        columns="Day_of_Week",
        values="Revenue_₹",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.reindex(index=SLOT_ORDER, columns=DAY_ORDER, fill_value=0)
    return pivot.astype(int)


def build_weather_category_pivot(transactions: pd.DataFrame) -> pd.DataFrame:
    pivot = transactions.pivot_table(
        index="Category",
        columns="Weather",
        values="Quantity_Units",
        aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.reindex(columns=WEATHER_ORDER, fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    return pivot.astype(int)


def build_aov_pivot(transactions: pd.DataFrame) -> pd.DataFrame:
    pivot = transactions.pivot_table(
        index="Time_Slot",
        columns="Day_of_Week",
        values="Revenue_₹",
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=SLOT_ORDER, columns=DAY_ORDER).fillna(0).round(0)
    return pivot.astype(int)


def format_currency(value: float) -> str:
    return f"₹{value:,.0f}"


def currency_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.map(format_currency)


def save_heatmap(
    frame: pd.DataFrame,
    title: str,
    cmap: str,
    output_path: Path,
    figsize: tuple[float, float],
    annotate: bool,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        frame,
        cmap=cmap,
        annot=currency_annotations(frame) if annotate else False,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel(frame.columns.name or "")
    ax.set_ylabel(frame.index.name or "")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dominant_price_tier_by_category(transactions: pd.DataFrame) -> pd.Series:
    category_tier_revenue = (
        transactions.groupby(["Category", "Price_Tier"], as_index=False)["Revenue_₹"]
        .sum()
        .sort_values(["Category", "Revenue_₹", "Price_Tier"], ascending=[True, False, True])
    )
    dominant_tier = category_tier_revenue.drop_duplicates("Category").set_index("Category")[
        "Price_Tier"
    ]
    return dominant_tier


def save_category_revenue_barchart(transactions: pd.DataFrame, output_path: Path) -> None:
    category_revenue = (
        transactions.groupby("Category")["Revenue_₹"].sum().sort_values(ascending=True)
    )
    dominant_tier = dominant_price_tier_by_category(transactions)
    bar_colors = [PRICE_TIER_COLORS[dominant_tier[category]] for category in category_revenue.index]

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(category_revenue.index, category_revenue.values, color=bar_colors)

    xmax = float(category_revenue.max())
    label_offset = xmax * 0.015
    for bar, value in zip(bars, category_revenue.values):
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            format_currency(value),
            va="center",
            ha="left",
            fontsize=9,
        )

    legend_handles = [
        Patch(facecolor=color, label=price_tier) for price_tier, color in PRICE_TIER_COLORS.items()
    ]
    ax.legend(handles=legend_handles, title="Dominant Price Tier", loc="lower right")
    ax.set_title("Total Revenue by Menu Category (May–July 2024)", pad=12)
    ax.set_xlabel("Revenue (₹)")
    ax.set_ylabel("Category")
    ax.set_xlim(0, xmax * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_pivots(
    revenue_pivot: pd.DataFrame,
    weather_category_pivot: pd.DataFrame,
    aov_pivot: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    revenue_pivot.to_csv(output_dir / "pivot_revenue_slot_day.csv")
    weather_category_pivot.to_csv(output_dir / "pivot_quantity_weather_category.csv")
    aov_pivot.to_csv(output_dir / "pivot_aov_slot_day.csv")


def print_completion_message(output_dir: Path, charts_dir: Path) -> None:
    output_dir_display = display_path(output_dir)
    charts_dir_display = display_path(charts_dir)
    print("03_pivot_heatmap.py complete.")
    print("Pivots saved:")
    print(output_dir_display / "pivot_revenue_slot_day.csv")
    print(output_dir_display / "pivot_quantity_weather_category.csv")
    print(output_dir_display / "pivot_aov_slot_day.csv")
    print("")
    print("Charts saved:")
    print(charts_dir_display / "heatmap_revenue_slot_day.png")
    print(charts_dir_display / "heatmap_quantity_weather_category.png")
    print(charts_dir_display / "heatmap_aov_slot_day.png")
    print(charts_dir_display / "barchart_category_revenue.png")


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def main() -> int:
    args = parse_args()
    sns.set_theme(style="white")

    transactions = load_transactions(args.input_path)
    revenue_pivot = build_revenue_pivot(transactions)
    weather_category_pivot = build_weather_category_pivot(transactions)
    aov_pivot = build_aov_pivot(transactions)

    write_pivots(
        revenue_pivot=revenue_pivot,
        weather_category_pivot=weather_category_pivot,
        aov_pivot=aov_pivot,
        output_dir=args.output_dir,
    )

    args.charts_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap(
        frame=revenue_pivot,
        title="Total Revenue (₹) by Time Slot × Day of Week",
        cmap="YlOrRd",
        output_path=args.charts_dir / "heatmap_revenue_slot_day.png",
        figsize=(12, 5),
        annotate=True,
    )
    save_heatmap(
        frame=weather_category_pivot,
        title="Total Quantity Sold by Category × Weather State",
        cmap="Blues",
        output_path=args.charts_dir / "heatmap_quantity_weather_category.png",
        figsize=(10, 12),
        annotate=False,
    )
    save_heatmap(
        frame=aov_pivot,
        title="Average Order Value (₹) by Time Slot × Day of Week",
        cmap="YlGn",
        output_path=args.charts_dir / "heatmap_aov_slot_day.png",
        figsize=(12, 5),
        annotate=True,
    )
    save_category_revenue_barchart(
        transactions=transactions,
        output_path=args.charts_dir / "barchart_category_revenue.png",
    )

    print_completion_message(output_dir=args.output_dir, charts_dir=args.charts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
