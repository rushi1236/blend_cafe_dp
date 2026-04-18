from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RECOMMENDATIONS_PATH = ROOT_DIR / "data" / "models" / "price_recommendations.csv"
DEFAULT_FEATURE_MATRIX_PATH = ROOT_DIR / "data" / "models" / "feature_matrix.csv"
DEFAULT_SEGMENTS_PATH = ROOT_DIR / "data" / "data_analysis" / "demand_segments.csv"
DEFAULT_REPORTS_DIR = ROOT_DIR / "data" / "reports"
DEFAULT_CHARTS_DIR = ROOT_DIR / "data" / "data_analysis" / "charts"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_DECODE = {index: day for index, day in enumerate(DAY_ORDER)}
SLOT_COLUMNS = {
    "Slot_Morning": "Morning",
    "Slot_Afternoon": "Afternoon",
    "Slot_Evening": "Evening",
    "Slot_Dinner": "Dinner",
}
WEATHER_COLUMNS = {
    "Weather_Dry_Day": "Dry Day",
    "Weather_Light_Drizzle": "Light Drizzle",
    "Weather_Heavy_Rain": "Heavy Rain",
    "Weather_Clear_After_Rain": "Clear After Rain",
}
SLOT_ORDER = ["Morning", "Afternoon", "Evening", "Dinner"]
WEATHER_ORDER = ["Dry Day", "Light Drizzle", "Heavy Rain", "Clear After Rain"]
SEGMENT_ORDER = [
    "Dynamic Pricing Target",
    "Premium Lock",
    "Promotional Item",
    "Review Item",
]
WEEKEND_DAYS = {"Friday", "Saturday", "Sunday"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare static vs dynamic pricing revenue on the Blend Cafe test period."
    )
    parser.add_argument(
        "--recommendations-path",
        type=Path,
        default=DEFAULT_RECOMMENDATIONS_PATH,
        help="Path to price_recommendations.csv",
    )
    parser.add_argument(
        "--feature-matrix-path",
        type=Path,
        default=DEFAULT_FEATURE_MATRIX_PATH,
        help="Path to feature_matrix.csv",
    )
    parser.add_argument(
        "--segments-path",
        type=Path,
        default=DEFAULT_SEGMENTS_PATH,
        help="Path to demand_segments.csv",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory where uplift reports will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where uplift charts will be written",
    )
    return parser.parse_args()


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def clean_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = df[column].astype("string").str.strip()
        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(f"{column} contains null or blank values")
    return df


def load_feature_matrix(path: Path) -> pd.DataFrame:
    feature_matrix = pd.read_csv(path)
    required_columns = [
        "Transaction_ID",
        "Date",
        "Item_Name",
        "Category",
        "ABC_Class",
        "Segment",
        "Day_of_Week_encoded",
        "Is_Weekend_encoded",
        "Base_Price_₹",
        "Quantity_Units",
        "Split",
        *SLOT_COLUMNS.keys(),
        *WEATHER_COLUMNS.keys(),
    ]
    missing_columns = [column for column in required_columns if column not in feature_matrix.columns]
    if missing_columns:
        raise ValueError(f"feature_matrix.csv is missing required columns: {missing_columns}")

    feature_matrix = clean_string_columns(
        feature_matrix,
        ["Item_Name", "Category", "ABC_Class", "Segment", "Split"],
    )
    feature_matrix["Date"] = pd.to_datetime(feature_matrix["Date"], errors="raise")
    for column in [
        "Day_of_Week_encoded",
        "Is_Weekend_encoded",
        "Base_Price_₹",
        "Quantity_Units",
        *SLOT_COLUMNS.keys(),
        *WEATHER_COLUMNS.keys(),
    ]:
        feature_matrix[column] = pd.to_numeric(feature_matrix[column], errors="raise").astype(int)

    return feature_matrix


def load_recommendations(path: Path) -> pd.DataFrame:
    recommendations = pd.read_csv(path)
    required_columns = [
        "Item_Name",
        "Time_Slot",
        "Day_Type",
        "Weather_State",
        "Recommended_Price_₹",
        "Price_Change_%",
        "Direction",
        "Demand_Verdict",
        "Review_Flag",
    ]
    missing_columns = [column for column in required_columns if column not in recommendations.columns]
    if missing_columns:
        raise ValueError(f"price_recommendations.csv is missing required columns: {missing_columns}")

    recommendations = clean_string_columns(
        recommendations,
        [
            "Item_Name",
            "Time_Slot",
            "Day_Type",
            "Weather_State",
            "Direction",
            "Demand_Verdict",
            "Review_Flag",
        ],
    )
    recommendations["Recommended_Price_₹"] = pd.to_numeric(
        recommendations["Recommended_Price_₹"], errors="raise"
    ).astype(int)
    recommendations["Price_Change_%"] = pd.to_numeric(
        recommendations["Price_Change_%"], errors="raise"
    )

    duplicate_mask = recommendations.duplicated(
        subset=["Item_Name", "Time_Slot", "Day_Type", "Weather_State"]
    )
    if duplicate_mask.any():
        raise ValueError("price_recommendations.csv contains duplicate join-key rows")

    return recommendations


def load_segments(path: Path) -> pd.DataFrame:
    segments = pd.read_csv(path)
    required_columns = ["Item_Name", "Segment", "ABC_Class", "Category"]
    missing_columns = [column for column in required_columns if column not in segments.columns]
    if missing_columns:
        raise ValueError(f"demand_segments.csv is missing required columns: {missing_columns}")

    segments = clean_string_columns(segments, required_columns)
    if segments["Item_Name"].duplicated().any():
        raise ValueError("demand_segments.csv contains duplicate Item_Name values")
    return segments[required_columns].copy()


def decode_one_hot(df: pd.DataFrame, column_map: dict[str, str], output_column: str) -> pd.DataFrame:
    df = df.copy()
    active_counts = df[list(column_map.keys())].sum(axis=1)
    if not active_counts.eq(1).all():
        raise ValueError(f"{output_column} one-hot columns must sum to exactly 1 per row")

    label_order = list(column_map.values())
    argmax_positions = df[list(column_map.keys())].to_numpy().argmax(axis=1)
    df[output_column] = [label_order[position] for position in argmax_positions]
    return df


def prepare_test_transactions(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    test_df = feature_matrix.loc[feature_matrix["Split"].eq("test")].copy()
    if test_df.empty:
        raise ValueError("feature_matrix.csv does not contain any test rows")

    test_df["Day_of_Week"] = test_df["Day_of_Week_encoded"].map(DAY_DECODE)
    if test_df["Day_of_Week"].isna().any():
        raise ValueError("Day_of_Week_encoded contains values outside 0-6")

    test_df["Day_Type"] = np.where(
        test_df["Day_of_Week"].isin(WEEKEND_DAYS), "Weekend", "Weekday"
    )
    test_df = decode_one_hot(test_df, SLOT_COLUMNS, "Time_Slot")
    test_df = decode_one_hot(test_df, WEATHER_COLUMNS, "Weather_State")
    return test_df


def join_recommendations(
    test_df: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    rec_lookup = recommendations[
        [
            "Item_Name",
            "Time_Slot",
            "Day_Type",
            "Weather_State",
            "Recommended_Price_₹",
            "Price_Change_%",
            "Direction",
            "Demand_Verdict",
            "Review_Flag",
        ]
    ].copy()

    test_enriched = test_df.merge(
        rec_lookup,
        on=["Item_Name", "Time_Slot", "Day_Type", "Weather_State"],
        how="left",
    )

    null_rec = int(test_enriched["Recommended_Price_₹"].isna().sum())
    if null_rec > 0:
        missing = test_enriched.loc[
            test_enriched["Recommended_Price_₹"].isna(),
            ["Item_Name", "Time_Slot", "Day_Type", "Weather_State"],
        ]
        print("Sample unmatched:")
        print(missing.head(10).to_string(index=False))
        test_enriched["Recommended_Price_₹"] = test_enriched["Recommended_Price_₹"].fillna(
            test_enriched["Base_Price_₹"]
        )
        test_enriched["Price_Change_%"] = test_enriched["Price_Change_%"].fillna(0.0)
        test_enriched["Direction"] = test_enriched["Direction"].fillna("HOLD")
        test_enriched["Demand_Verdict"] = test_enriched["Demand_Verdict"].fillna("NORMAL")
        test_enriched["Review_Flag"] = test_enriched["Review_Flag"].fillna("OK")

    test_enriched["Recommended_Price_₹"] = pd.to_numeric(
        test_enriched["Recommended_Price_₹"], errors="raise"
    ).astype(int)
    test_enriched["Price_Change_%"] = pd.to_numeric(
        test_enriched["Price_Change_%"], errors="raise"
    )
    return test_enriched, null_rec


def validate_segment_alignment(test_enriched: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    segment_lookup = segments.rename(
        columns={
            "Segment": "Segment_Lookup",
            "ABC_Class": "ABC_Class_Lookup",
            "Category": "Category_Lookup",
        }
    )
    merged = test_enriched.merge(segment_lookup, on="Item_Name", how="left")
    if merged["Segment_Lookup"].isna().any():
        raise ValueError("Some test transactions are missing segment lookup values")

    for column in ["Segment", "ABC_Class", "Category"]:
        mismatch_mask = merged[column] != merged[f"{column}_Lookup"]
        if mismatch_mask.any():
            raise ValueError(f"{column} mismatch between feature_matrix.csv and demand_segments.csv")

    merged = merged.drop(columns=["Segment_Lookup", "ABC_Class_Lookup", "Category_Lookup"])
    return merged


def compute_revenue_columns(test_enriched: pd.DataFrame) -> pd.DataFrame:
    test_enriched = test_enriched.copy()
    test_enriched["Static_Revenue_₹"] = (
        test_enriched["Base_Price_₹"] * test_enriched["Quantity_Units"]
    )
    test_enriched["Dynamic_Revenue_₹"] = (
        test_enriched["Recommended_Price_₹"] * test_enriched["Quantity_Units"]
    )
    test_enriched["Revenue_Difference_₹"] = (
        test_enriched["Dynamic_Revenue_₹"] - test_enriched["Static_Revenue_₹"]
    )
    return test_enriched


def uplift_percent(dynamic_revenue: pd.Series, static_revenue: pd.Series) -> pd.Series:
    return np.where(
        static_revenue.eq(0),
        0.0,
        ((dynamic_revenue - static_revenue) / static_revenue * 100).round(2),
    )


def build_breakdowns(
    test_enriched: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    category_breakdown = (
        test_enriched.groupby("Category")
        .agg(
            Static_Revenue=("Static_Revenue_₹", "sum"),
            Dynamic_Revenue=("Dynamic_Revenue_₹", "sum"),
            Transaction_Count=("Transaction_ID", "count"),
        )
        .assign(
            **{
                "Uplift_₹": lambda frame: frame["Dynamic_Revenue"] - frame["Static_Revenue"],
                "Uplift_Pct": lambda frame: uplift_percent(
                    frame["Dynamic_Revenue"], frame["Static_Revenue"]
                ),
            }
        )
        .sort_values("Uplift_₹", ascending=False)
        .reset_index()
    )

    slot_breakdown = (
        test_enriched.groupby("Time_Slot")
        .agg(
            Static_Revenue=("Static_Revenue_₹", "sum"),
            Dynamic_Revenue=("Dynamic_Revenue_₹", "sum"),
        )
        .assign(
            **{
                "Uplift_₹": lambda frame: frame["Dynamic_Revenue"] - frame["Static_Revenue"],
                "Uplift_Pct": lambda frame: uplift_percent(
                    frame["Dynamic_Revenue"], frame["Static_Revenue"]
                ),
            }
        )
        .reindex(SLOT_ORDER)
        .reset_index()
    )

    weather_breakdown = (
        test_enriched.groupby("Weather_State")
        .agg(
            Transaction_Count=("Transaction_ID", "count"),
            Static_Revenue=("Static_Revenue_₹", "sum"),
            Dynamic_Revenue=("Dynamic_Revenue_₹", "sum"),
        )
        .assign(
            **{
                "Uplift_₹": lambda frame: frame["Dynamic_Revenue"] - frame["Static_Revenue"],
                "Uplift_Pct": lambda frame: uplift_percent(
                    frame["Dynamic_Revenue"], frame["Static_Revenue"]
                ),
            }
        )
        .reindex(WEATHER_ORDER)
        .reset_index()
    )

    segment_breakdown = (
        test_enriched.groupby("Segment")
        .agg(
            Static_Revenue=("Static_Revenue_₹", "sum"),
            Dynamic_Revenue=("Dynamic_Revenue_₹", "sum"),
        )
        .assign(
            **{
                "Uplift_₹": lambda frame: frame["Dynamic_Revenue"] - frame["Static_Revenue"],
                "Uplift_Pct": lambda frame: uplift_percent(
                    frame["Dynamic_Revenue"], frame["Static_Revenue"]
                ),
            }
        )
        .reindex(SEGMENT_ORDER)
        .reset_index()
    )

    return category_breakdown, slot_breakdown, weather_breakdown, segment_breakdown


def save_reports(
    test_enriched: pd.DataFrame,
    category_breakdown: pd.DataFrame,
    slot_breakdown: pd.DataFrame,
    weather_breakdown: pd.DataFrame,
    segment_breakdown: pd.DataFrame,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_enriched[
        [
            "Transaction_ID",
            "Date",
            "Item_Name",
            "Category",
            "Time_Slot",
            "Day_Type",
            "Weather_State",
            "Base_Price_₹",
            "Recommended_Price_₹",
            "Quantity_Units",
            "Static_Revenue_₹",
            "Dynamic_Revenue_₹",
            "Revenue_Difference_₹",
            "Direction",
            "Demand_Verdict",
            "Review_Flag",
        ]
    ].to_csv(reports_dir / "revenue_uplift_analysis.csv", index=False)

    category_breakdown.to_csv(reports_dir / "uplift_by_category.csv", index=False)
    slot_breakdown.to_csv(reports_dir / "uplift_by_slot.csv", index=False)
    weather_breakdown.to_csv(reports_dir / "uplift_by_weather.csv", index=False)
    segment_breakdown.to_csv(reports_dir / "uplift_by_segment.csv", index=False)


def save_waterfall_chart(
    category_breakdown: pd.DataFrame,
    slot_breakdown: pd.DataFrame,
    charts_dir: Path,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    top_categories = category_breakdown.nlargest(8, "Uplift_₹")
    bottom_categories = category_breakdown.nsmallest(2, "Uplift_₹")
    waterfall_data = (
        pd.concat([top_categories, bottom_categories], ignore_index=True)
        .drop_duplicates(subset="Category")
        .sort_values("Uplift_₹")
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Revenue Uplift Analysis - Dynamic vs Static Pricing\n"
        "Test Period: July 16-31, 2024 (Blend Cafe)",
        fontsize=13,
        fontweight="bold",
    )

    colors = ["#d62728" if value > 0 else "#2c7bb6" for value in waterfall_data["Uplift_₹"]]
    bars = axes[0].barh(
        waterfall_data["Category"],
        waterfall_data["Uplift_₹"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, value in zip(bars, waterfall_data["Uplift_₹"]):
        axes[0].text(
            value + (50 if value >= 0 else -50),
            bar.get_y() + bar.get_height() / 2,
            f"₹{value:,.0f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
        )

    axes[0].axvline(x=0, color="black", linewidth=1)
    axes[0].set_xlabel("Revenue Uplift (₹)", fontsize=11)
    axes[0].set_title("Uplift by Category\n(Top 8 positive, Bottom 2 negative)", fontsize=11)
    axes[0].grid(True, alpha=0.3, axis="x")

    slot_plot = slot_breakdown.set_index("Time_Slot").reindex(SLOT_ORDER)
    x_positions = np.arange(len(SLOT_ORDER))
    width = 0.35
    axes[1].bar(
        x_positions - width / 2,
        slot_plot["Static_Revenue"] / 1000,
        width,
        label="Static Pricing",
        color="#2c7bb6",
        edgecolor="black",
        linewidth=0.7,
    )
    axes[1].bar(
        x_positions + width / 2,
        slot_plot["Dynamic_Revenue"] / 1000,
        width,
        label="Dynamic Pricing",
        color="#d62728",
        edgecolor="black",
        linewidth=0.7,
    )

    for index, slot in enumerate(SLOT_ORDER):
        uplift_pct = float(slot_plot.loc[slot, "Uplift_Pct"])
        max_height = max(
            float(slot_plot.loc[slot, "Static_Revenue"]),
            float(slot_plot.loc[slot, "Dynamic_Revenue"]),
        )
        axes[1].text(
            index,
            max_height / 1000 + 0.3,
            f"+{uplift_pct:.1f}%" if uplift_pct >= 0 else f"{uplift_pct:.1f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#2ca02c" if uplift_pct >= 0 else "#d62728",
        )

    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(SLOT_ORDER, fontsize=10)
    axes[1].set_ylabel("Revenue (₹ thousands)", fontsize=11)
    axes[1].set_title("Static vs Dynamic Revenue by Time Slot\n(Test Period - Jul 16-31)", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(charts_dir / "revenue_uplift_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()

    feature_matrix = load_feature_matrix(args.feature_matrix_path)
    recommendations = load_recommendations(args.recommendations_path)
    segments = load_segments(args.segments_path)

    print(f"Feature matrix rows: {len(feature_matrix)}")
    print(f"Recommendations rows: {len(recommendations)}")

    test_df = prepare_test_transactions(feature_matrix)
    print(f"Test period transactions: {len(test_df)}")
    print(
        f"Test period: {test_df['Date'].min().date()} -> {test_df['Date'].max().date()}"
    )

    print("\nDay type distribution in test set:")
    print(test_df["Day_Type"].value_counts().to_string())

    test_enriched, null_rec = join_recommendations(test_df, recommendations)
    print(f"\nTest transactions with no recommendation match: {null_rec}")

    test_enriched = validate_segment_alignment(test_enriched, segments)
    test_enriched = compute_revenue_columns(test_enriched)

    total_static = int(test_enriched["Static_Revenue_₹"].sum())
    total_dynamic = int(test_enriched["Dynamic_Revenue_₹"].sum())
    total_uplift = int(total_dynamic - total_static)
    uplift_pct = (total_uplift / total_static) * 100 if total_static else 0.0

    print("\n" + "=" * 55)
    print("  HEADLINE RESULT")
    print("=" * 55)
    print("  Test period: Jul 16 - Jul 31 (16 days)")
    print(f"  Transactions analysed: {len(test_enriched):,}")
    print(f"  Static  revenue: ₹{total_static:>10,.0f}")
    print(f"  Dynamic revenue: ₹{total_dynamic:>10,.0f}")
    print(f"  Uplift  amount:  ₹{total_uplift:>10,.0f}")
    print(f"  Uplift  %:       {uplift_pct:>10.2f}%")
    print("=" * 55)

    category_breakdown, slot_breakdown, weather_breakdown, segment_breakdown = build_breakdowns(
        test_enriched
    )

    print("\nTop 5 categories by absolute uplift:")
    print(
        category_breakdown[
            ["Category", "Static_Revenue", "Dynamic_Revenue", "Uplift_₹", "Uplift_Pct"]
        ]
        .head(5)
        .to_string(index=False)
    )

    print("\nUplift by time slot:")
    print(
        slot_breakdown[
            ["Time_Slot", "Static_Revenue", "Dynamic_Revenue", "Uplift_₹", "Uplift_Pct"]
        ].to_string(index=False)
    )

    print("\nUplift by weather state:")
    print(
        weather_breakdown[
            ["Weather_State", "Transaction_Count", "Static_Revenue", "Uplift_₹", "Uplift_Pct"]
        ].to_string(index=False)
    )

    print("\nUplift by pricing segment:")
    print(
        segment_breakdown[
            ["Segment", "Static_Revenue", "Dynamic_Revenue", "Uplift_₹", "Uplift_Pct"]
        ].to_string(index=False)
    )

    save_reports(
        test_enriched,
        category_breakdown,
        slot_breakdown,
        weather_breakdown,
        segment_breakdown,
        args.reports_dir,
    )
    save_waterfall_chart(category_breakdown, slot_breakdown, args.charts_dir)

    print("\n04_revenue_comparison.py complete.")
    print("")
    print("=" * 55)
    print("HEADLINE RESULT")
    print("=" * 55)
    print("Test period:          Jul 16 - Jul 31 (16 days)")
    print(f"Transactions:         {len(test_enriched)}")
    print(f"Static revenue:       ₹{total_static:,.0f}")
    print(f"Dynamic revenue:      ₹{total_dynamic:,.0f}")
    print(f"Uplift amount:        ₹{total_uplift:,.0f}")
    print(f"Uplift %:             {uplift_pct:.2f}%")
    print("")
    print("Uplift by time slot:")
    for _, row in slot_breakdown.iterrows():
        print(f"  {row['Time_Slot']:<10}: {row['Uplift_Pct']:+.2f}%")
    print("")
    print("Uplift by weather state:")
    for _, row in weather_breakdown.iterrows():
        print(f"  {row['Weather_State']:<16}: {row['Uplift_Pct']:+.2f}%")
    print("")
    print("Uplift by segment:")
    for _, row in segment_breakdown.iterrows():
        print(f"  {row['Segment']:<22}: {row['Uplift_Pct']:+.2f}%")
    print("")
    print(f"Outputs saved to {display_path(args.reports_dir)}:")
    print(f"  {display_path(args.reports_dir / 'revenue_uplift_analysis.csv')} ({len(test_enriched)} transaction rows)")
    print(f"  {display_path(args.reports_dir / 'uplift_by_category.csv')}")
    print(f"  {display_path(args.reports_dir / 'uplift_by_slot.csv')}")
    print(f"  {display_path(args.reports_dir / 'uplift_by_weather.csv')}")
    print(f"  {display_path(args.reports_dir / 'uplift_by_segment.csv')}")
    print("Chart:")
    print(f"  {display_path(args.charts_dir / 'revenue_uplift_waterfall.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
