from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "data_analysis" / "processed_transactions.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "data_analysis"

REQUIRED_COLUMNS = [
    "Transaction_ID",
    "Item_Name",
    "Category",
    "Price_Tier",
    "Revenue_₹",
]

SUMMARY_COLUMNS = [
    "Item_Name",
    "Category",
    "Price_Tier",
    "Total_Revenue_₹",
    "Transaction_Count",
    "Cumulative_Revenue_%",
    "ABC_Class",
]

A_THRESHOLD = 70.0
B_THRESHOLD = 90.0

EXPECTED_CLASS_RANGES = {
    "A": {"item_count": (35, 42), "revenue_pct": (66.0, 70.0)},
    "B": {"item_count": (40, 50), "revenue_pct": (20.0, 24.0)},
    "C": {"item_count": (108, 125), "revenue_pct": (8.0, 12.0)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ABC analysis on processed Blend Cafe transactions and append "
            "ABC_Class back onto the transaction-level dataset."
        )
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
        help="Directory where abc_classified.csv will be written",
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
    if "ABC_Class" in transactions.columns:
        transactions = transactions.drop(columns=["ABC_Class"])

    for column in ["Item_Name", "Category", "Price_Tier"]:
        transactions[column] = transactions[column].astype("string").str.strip()
        if transactions[column].isna().any() or transactions[column].eq("").any():
            raise ValueError(f"{column} contains null or blank values")

    transactions["Transaction_ID"] = pd.to_numeric(
        transactions["Transaction_ID"], errors="raise"
    ).astype(int)
    transactions["Revenue_₹"] = pd.to_numeric(transactions["Revenue_₹"], errors="raise")

    if transactions["Revenue_₹"].lt(0).any():
        raise ValueError("Revenue_₹ cannot be negative")

    return transactions


def validate_item_metadata(transactions: pd.DataFrame) -> None:
    item_consistency = transactions.groupby("Item_Name").agg(
        **{
            "Category_Count": ("Category", "nunique"),
            "Price_Tier_Count": ("Price_Tier", "nunique"),
        }
    )

    inconsistent_items = item_consistency.loc[
        (item_consistency["Category_Count"] != 1)
        | (item_consistency["Price_Tier_Count"] != 1)
    ]
    if not inconsistent_items.empty:
        raise ValueError(
            "Item metadata must be stable per Item_Name. "
            f"Found inconsistent items:\n{inconsistent_items.head(10)}"
        )


def build_abc_summary(transactions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    summary = (
        transactions.groupby("Item_Name", as_index=False)
        .agg(
            **{
                "Category": ("Category", "first"),
                "Price_Tier": ("Price_Tier", "first"),
                "Total_Revenue_₹": ("Revenue_₹", "sum"),
                "Transaction_Count": ("Transaction_ID", "count"),
            }
        )
        .sort_values(["Total_Revenue_₹", "Item_Name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    total_revenue = float(summary["Total_Revenue_₹"].sum())
    cumulative_revenue_pct = summary["Total_Revenue_₹"].cumsum() / total_revenue * 100

    summary["Cumulative_Revenue_%"] = cumulative_revenue_pct.round(2)
    summary["ABC_Class"] = np.select(
        [cumulative_revenue_pct <= A_THRESHOLD, cumulative_revenue_pct <= B_THRESHOLD],
        ["A", "B"],
        default="C",
    )

    return summary[SUMMARY_COLUMNS], total_revenue


def append_abc_class(
    transactions: pd.DataFrame, abc_summary: pd.DataFrame
) -> pd.DataFrame:
    merged = (
        transactions.reset_index(names="_row_order")
        .merge(
            abc_summary[["Item_Name", "ABC_Class"]],
            on="Item_Name",
            how="left",
            validate="many_to_one",
        )
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )

    if len(merged) != len(transactions):
        raise ValueError("Transaction row count changed after ABC left join")
    if merged["ABC_Class"].isna().any():
        raise ValueError("ABC_Class contains NaN values after left join on Item_Name")

    return merged


def build_class_summary(abc_summary: pd.DataFrame, total_revenue: float) -> pd.DataFrame:
    class_summary = (
        abc_summary.groupby("ABC_Class", as_index=False)
        .agg(
            **{
                "Item_Count": ("Item_Name", "count"),
                "Revenue_₹": ("Total_Revenue_₹", "sum"),
            }
        )
        .sort_values("ABC_Class")
        .reset_index(drop=True)
    )
    class_summary["Revenue_%"] = (class_summary["Revenue_₹"] / total_revenue * 100).round(2)
    return class_summary


def build_class_boundaries(abc_summary: pd.DataFrame) -> list[dict[str, float | int | str]]:
    boundaries: list[dict[str, float | int | str]] = []
    ranked = abc_summary.reset_index(drop=True).copy()
    ranked["Item_Rank"] = ranked.index + 1

    cumulative_start = 0.0
    for class_name in ["A", "B", "C"]:
        class_rows = ranked.loc[ranked["ABC_Class"].eq(class_name)]
        if class_rows.empty:
            continue

        boundaries.append(
            {
                "ABC_Class": class_name,
                "Item_Start": int(class_rows["Item_Rank"].min()),
                "Item_End": int(class_rows["Item_Rank"].max()),
                "Cumulative_Start_%": cumulative_start,
                "Cumulative_End_%": float(class_rows["Cumulative_Revenue_%"].iloc[-1]),
            }
        )
        cumulative_start = float(class_rows["Cumulative_Revenue_%"].iloc[-1])

    return boundaries


def build_expectation_warnings(class_summary: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for row in class_summary.to_dict("records"):
        expected = EXPECTED_CLASS_RANGES.get(row["ABC_Class"])
        if expected is None:
            continue

        item_low, item_high = expected["item_count"]
        revenue_low, revenue_high = expected["revenue_pct"]
        item_in_range = item_low <= row["Item_Count"] <= item_high
        revenue_in_range = revenue_low <= row["Revenue_%"] <= revenue_high

        if item_in_range and revenue_in_range:
            continue

        warnings.append(
            (
                f"Class {row['ABC_Class']} actual split differs from narrative expectation: "
                f"items={row['Item_Count']} vs expected {item_low}-{item_high}, "
                f"revenue={row['Revenue_%']}% vs expected {revenue_low:.1f}-{revenue_high:.1f}%."
            )
        )
    return warnings


def write_outputs(
    transactions_with_abc: pd.DataFrame,
    abc_summary: pd.DataFrame,
    output_dir: Path,
    processed_transactions_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    abc_summary.to_csv(output_dir / "abc_classified.csv", index=False)
    transactions_with_abc.to_csv(processed_transactions_path, index=False)


def print_summary(
    input_path: Path,
    output_dir: Path,
    transactions: pd.DataFrame,
    abc_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    class_boundaries: list[dict[str, float | int | str]],
    warnings: list[str],
) -> None:
    print("\nBlend Cafe ABC analysis completed.\n")
    print(f"Input file             : {input_path}")
    print(f"Transactions processed : {len(transactions):,}")
    print(f"Unique items ranked    : {len(abc_summary):,}")
    print(f"ABC summary written    : {output_dir / 'abc_classified.csv'}")
    print(f"Transactions updated   : {input_path}")

    print("\nClass distribution")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(class_summary.to_string(index=False))

    print("\nActual class boundaries")
    for boundary in class_boundaries:
        print(
            f"- Class {boundary['ABC_Class']}: items {boundary['Item_Start']}"
            f"–{boundary['Item_End']} (cumulative revenue "
            f"{format_pct(float(boundary['Cumulative_Start_%']))}% → "
            f"{format_pct(float(boundary['Cumulative_End_%']))}%)"
        )

    if warnings:
        print("\nVerification warnings")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nVerification against expected narrative ranges passed.")


def main() -> int:
    args = parse_args()
    transactions = load_transactions(args.input_path)
    validate_item_metadata(transactions)

    abc_summary, total_revenue = build_abc_summary(transactions)
    transactions_with_abc = append_abc_class(transactions, abc_summary)
    class_summary = build_class_summary(abc_summary, total_revenue)
    class_boundaries = build_class_boundaries(abc_summary)
    warnings = build_expectation_warnings(class_summary)

    write_outputs(
        transactions_with_abc=transactions_with_abc,
        abc_summary=abc_summary,
        output_dir=args.output_dir,
        processed_transactions_path=args.input_path,
    )
    print_summary(
        input_path=args.input_path,
        output_dir=args.output_dir,
        transactions=transactions,
        abc_summary=abc_summary,
        class_summary=class_summary,
        class_boundaries=class_boundaries,
        warnings=warnings,
    )
    return 0


def format_pct(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
