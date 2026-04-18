from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ABC_PATH = ROOT_DIR / "data" / "data_analysis" / "abc_classified.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "data_analysis"
DEFAULT_CHARTS_DIR = DEFAULT_OUTPUT_DIR / "charts"

REQUIRED_COLUMNS = [
    "Item_Name",
    "Total_Revenue_₹",
    "Cumulative_Revenue_%",
    "ABC_Class",
]
ABC_ORDER = ["A", "B", "C"]
ABC_COLORS = {"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Pareto validation outputs from Blend Cafe ABC analysis."
    )
    parser.add_argument(
        "--abc-path",
        type=Path,
        default=DEFAULT_ABC_PATH,
        help="Path to abc_classified.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where pareto_validation.csv will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where pareto_analysis.png will be written",
    )
    return parser.parse_args()


def load_abc_summary(path: Path) -> pd.DataFrame:
    abc = pd.read_csv(path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in abc.columns]
    if missing_columns:
        raise ValueError(f"abc_classified.csv is missing required columns: {missing_columns}")

    abc = abc.copy()
    abc["Item_Name"] = abc["Item_Name"].astype("string").str.strip()
    abc["ABC_Class"] = abc["ABC_Class"].astype("string").str.strip()
    abc["Total_Revenue_₹"] = pd.to_numeric(abc["Total_Revenue_₹"], errors="raise")
    abc["Cumulative_Revenue_%"] = pd.to_numeric(
        abc["Cumulative_Revenue_%"], errors="raise"
    )

    abc = abc.sort_values(["Total_Revenue_₹", "Item_Name"], ascending=[False, True]).reset_index(
        drop=True
    )
    abc["Item_Rank"] = range(1, len(abc) + 1)
    abc["Item_Pct"] = abc["Item_Rank"] / len(abc) * 100
    return abc


def build_abc_summary(abc: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    total_revenue = float(abc["Total_Revenue_₹"].sum())
    total_items = int(len(abc))

    abc_summary = (
        abc.groupby("ABC_Class", as_index=False)
        .agg(
            **{
                "Item_Count": ("Item_Name", "count"),
                "Revenue_₹": ("Total_Revenue_₹", "sum"),
            }
        )
        .set_index("ABC_Class")
        .reindex(ABC_ORDER)
        .reset_index()
    )
    abc_summary["Item_Pct"] = (abc_summary["Item_Count"] / total_items * 100).round(2)
    abc_summary["Revenue_Pct"] = (abc_summary["Revenue_₹"] / total_revenue * 100).round(2)
    return abc_summary, total_revenue, total_items


def find_crossover(abc: pd.DataFrame) -> pd.Series:
    crossover_rows = abc.loc[abc["Cumulative_Revenue_%"] >= 80]
    if crossover_rows.empty:
        raise ValueError("Pareto crossover could not be found because cumulative revenue never reaches 80%")
    return crossover_rows.iloc[0]


def save_pareto_validation(abc_summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = abc_summary.rename(columns={"Revenue_₹": "Revenue_₹"})[
        ["ABC_Class", "Item_Count", "Item_Pct", "Revenue_₹", "Revenue_Pct"]
    ]
    validation.to_csv(output_dir / "pareto_validation.csv", index=False)


def save_pareto_chart(
    abc: pd.DataFrame,
    abc_summary: pd.DataFrame,
    crossover: pd.Series,
    charts_dir: Path,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    crossover_item_count = int(crossover["Item_Rank"])
    crossover_item_pct = float(crossover["Item_Pct"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(
        abc["Item_Pct"],
        abc["Cumulative_Revenue_%"],
        color="#2c7bb6",
        linewidth=2.5,
        label="Cumulative Revenue %",
    )
    ax1.fill_between(
        abc["Item_Pct"],
        abc["Cumulative_Revenue_%"],
        alpha=0.15,
        color="#2c7bb6",
    )
    ax1.axhline(
        y=80,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="80% Revenue Threshold",
    )
    ax1.axvline(
        x=20,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="20% of Items (Classic Pareto)",
    )
    ax1.axvline(
        x=crossover_item_pct,
        color="green",
        linestyle="-.",
        linewidth=1.5,
        label=f"Actual: {crossover_item_pct:.1f}% of items",
    )
    ax1.annotate(
        f"{crossover_item_count} items\n({crossover_item_pct:.1f}%)\nreach 80% revenue",
        xy=(crossover_item_pct, 80),
        xytext=(min(crossover_item_pct + 5, 88), 65),
        arrowprops={"arrowstyle": "->", "color": "black"},
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightyellow", "edgecolor": "gray"},
    )
    ax1.set_xlabel("Cumulative % of Menu Items (200 total)", fontsize=11)
    ax1.set_ylabel("Cumulative % of Total Revenue", fontsize=11)
    ax1.set_title("Pareto Curve — Revenue Concentration\n(Blend Café, May–July 2024)", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)

    bars = ax2.bar(
        abc_summary["ABC_Class"],
        abc_summary["Revenue_Pct"],
        color=[ABC_COLORS[label] for label in abc_summary["ABC_Class"]],
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    for bar, row in zip(bars, abc_summary.to_dict("records")):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{row['Revenue_Pct']:.1f}% revenue\n{int(row['Item_Count'])} items ({row['Item_Pct']:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_xlabel("ABC Class", fontsize=11)
    ax2.set_ylabel("% of Total Revenue", fontsize=11)
    ax2.set_title(
        "Revenue by ABC Class\n(A = top 70%, B = next 20%, C = bottom 10%)",
        fontsize=12,
    )
    ax2.set_ylim(0, 85)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Pareto Analysis — Blend Café Menu Revenue Distribution",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(charts_dir / "pareto_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_inr(value: float | int) -> str:
    sign = "-" if value < 0 else ""
    value_int = int(round(abs(value)))
    digits = str(value_int)
    if len(digits) <= 3:
        return f"{sign}₹{digits}"

    last_three = digits[-3:]
    remaining = digits[:-3]
    parts: list[str] = []
    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.insert(0, remaining)
    grouped = ",".join(parts + [last_three])
    return f"{sign}₹{grouped}"


def classic_pareto_statement(crossover_item_pct: float) -> str:
    if crossover_item_pct <= 20.0:
        return "True — revenue is concentrated in a small hero-item set"
    return "False — Blend Café revenue is broadly distributed"


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def print_summary(
    abc_summary: pd.DataFrame,
    total_items: int,
    total_revenue: float,
    crossover: pd.Series,
    output_dir: Path,
    charts_dir: Path,
) -> None:
    crossover_item_count = int(crossover["Item_Rank"])
    crossover_item_pct = float(crossover["Item_Pct"])

    print("06_pareto_analysis.py complete.")
    print(f"  Total items analysed: {total_items}")
    print(f"  Total revenue: {format_inr(total_revenue)}")
    print("")
    print("  ABC Summary:")
    for row in abc_summary.to_dict("records"):
        print(
            f"    Class {row['ABC_Class']}: {int(row['Item_Count'])} items "
            f"({row['Item_Pct']:.1f}% of menu) → {row['Revenue_Pct']:.1f}% of revenue"
        )
    print("")
    print(
        f"  Pareto finding: 80% of revenue reached at item #{crossover_item_count} "
        f"({crossover_item_pct:.1f}% of menu)"
    )
    print(f"  Classic 80/20 holds: {classic_pareto_statement(crossover_item_pct)}")
    print("")
    print(f"  Output: {display_path(output_dir / 'pareto_validation.csv')}")
    print(f"  Chart:  {display_path(charts_dir / 'pareto_analysis.png')}")


def main() -> int:
    args = parse_args()
    abc = load_abc_summary(args.abc_path)
    abc_summary, total_revenue, total_items = build_abc_summary(abc)
    crossover = find_crossover(abc)

    save_pareto_validation(abc_summary, args.output_dir)
    save_pareto_chart(abc, abc_summary, crossover, args.charts_dir)
    print_summary(abc_summary, total_items, total_revenue, crossover, args.output_dir, args.charts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
