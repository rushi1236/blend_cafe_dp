from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_PATH = ROOT_DIR / "data" / "data_analysis" / "processed_transactions.csv"
DEFAULT_SEGMENTS_PATH = ROOT_DIR / "data" / "data_analysis" / "demand_segments.csv"
DEFAULT_PIVOT_REVENUE_PATH = ROOT_DIR / "data" / "data_analysis" / "pivot_revenue_slot_day.csv"
DEFAULT_ELASTICITY_PATH = ROOT_DIR / "data" / "data_analysis" / "elasticity_coefficients.csv"
DEFAULT_WORKBOOK_PATH = ROOT_DIR / "data" / "obs" / "BlendCafe_DynamicPricing_Data.xlsx"
DEFAULT_MODELS_DIR = ROOT_DIR / "data" / "models"
DEFAULT_CHARTS_DIR = ROOT_DIR / "data" / "data_analysis" / "charts"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIME_SLOT_ORDER = ["Morning", "Afternoon", "Evening", "Dinner"]
WEATHER_ORDER = ["Dry Day", "Light Drizzle", "Heavy Rain", "Clear After Rain"]
DAY_TYPES = {
    "Weekday": ["Monday", "Tuesday", "Wednesday", "Thursday"],
    "Weekend": ["Friday", "Saturday", "Sunday"],
}
WEEKDAY_REP_DAY = "Wednesday"
WEEKEND_REP_DAY = "Saturday"
MAX_INCREASE_PCT = 0.15
MAX_DECREASE_PCT = 0.12
PRICE_FLOOR_PCT = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate traceable dynamic price recommendations for Blend Cafe."
    )
    parser.add_argument(
        "--processed-path",
        type=Path,
        default=DEFAULT_PROCESSED_PATH,
        help="Path to processed_transactions.csv",
    )
    parser.add_argument(
        "--segments-path",
        type=Path,
        default=DEFAULT_SEGMENTS_PATH,
        help="Path to demand_segments.csv",
    )
    parser.add_argument(
        "--pivot-revenue-path",
        type=Path,
        default=DEFAULT_PIVOT_REVENUE_PATH,
        help="Path to pivot_revenue_slot_day.csv",
    )
    parser.add_argument(
        "--elasticity-path",
        type=Path,
        default=DEFAULT_ELASTICITY_PATH,
        help="Path to elasticity_coefficients.csv",
    )
    parser.add_argument(
        "--workbook-path",
        type=Path,
        default=DEFAULT_WORKBOOK_PATH,
        help="Path to BlendCafe_DynamicPricing_Data.xlsx",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory where price_recommendations.csv will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where recommender charts will be written",
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


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unexpected boolean-like value: {value}")


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = [
        "Date",
        "Day_of_Week",
        "Time_Slot",
        "Weather",
        "Item_Name",
        "Daily_Footfall_Customers",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"processed_transactions.csv is missing required columns: {missing_columns}")

    df = clean_string_columns(df, ["Day_of_Week", "Time_Slot", "Weather", "Item_Name"])
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="raise")
    df["Daily_Footfall_Customers"] = pd.to_numeric(
        df["Daily_Footfall_Customers"], errors="raise"
    ).astype(int)

    if len(df) != 3818:
        raise ValueError(f"Expected 3818 transaction rows, found {len(df)}")
    return df


def load_segments(path: Path) -> pd.DataFrame:
    segments = pd.read_csv(path)
    required_columns = [
        "Item_Name",
        "ABC_Class",
        "Category",
        "Price_Tier",
        "Segment",
        "Base_Price_₹",
        "Elasticity_Coefficient",
        "Rain_Boost_Flag",
    ]
    missing_columns = [column for column in required_columns if column not in segments.columns]
    if missing_columns:
        raise ValueError(f"demand_segments.csv is missing required columns: {missing_columns}")

    segments = clean_string_columns(
        segments,
        ["Item_Name", "ABC_Class", "Category", "Price_Tier", "Segment"],
    )
    segments["Base_Price_₹"] = pd.to_numeric(segments["Base_Price_₹"], errors="raise").astype(int)
    segments["Elasticity_Coefficient"] = pd.to_numeric(
        segments["Elasticity_Coefficient"], errors="raise"
    )
    segments["Rain_Boost_Flag"] = segments["Rain_Boost_Flag"].apply(parse_bool)

    if len(segments) != 200:
        raise ValueError(f"Expected 200 rows in demand_segments.csv, found {len(segments)}")
    if segments["Item_Name"].duplicated().any():
        raise ValueError("demand_segments.csv contains duplicate Item_Name values")
    return segments


def load_pivot_revenue(path: Path) -> pd.DataFrame:
    pivot = pd.read_csv(path, index_col="Time_Slot")
    pivot.index = pivot.index.astype("string").str.strip()
    pivot.columns = [str(column).strip() for column in pivot.columns]
    missing_days = [day for day in DAY_ORDER if day not in pivot.columns]
    if missing_days:
        raise ValueError(f"pivot_revenue_slot_day.csv is missing day columns: {missing_days}")
    pivot = pivot.reindex(index=TIME_SLOT_ORDER, columns=DAY_ORDER)
    pivot = pivot.apply(pd.to_numeric, errors="raise")
    return pivot


def load_elasticity(path: Path) -> pd.DataFrame:
    elasticity = pd.read_csv(path)
    required_columns = [
        "Item_Name",
        "Base_Price_₹",
        "Elasticity_Coefficient",
        "Rain_Boost_Flag",
    ]
    missing_columns = [column for column in required_columns if column not in elasticity.columns]
    if missing_columns:
        raise ValueError(
            f"elasticity_coefficients.csv is missing required columns: {missing_columns}"
        )

    elasticity = clean_string_columns(elasticity, ["Item_Name"])
    elasticity["Base_Price_₹"] = pd.to_numeric(
        elasticity["Base_Price_₹"], errors="raise"
    ).astype(int)
    elasticity["Elasticity_Coefficient"] = pd.to_numeric(
        elasticity["Elasticity_Coefficient"], errors="raise"
    )
    elasticity["Rain_Boost_Flag"] = elasticity["Rain_Boost_Flag"].apply(parse_bool)

    if len(elasticity) != 200:
        raise ValueError(
            f"Expected 200 rows in elasticity_coefficients.csv, found {len(elasticity)}"
        )
    if elasticity["Item_Name"].duplicated().any():
        raise ValueError("elasticity_coefficients.csv contains duplicate Item_Name values")
    return elasticity


def load_item_master(path: Path) -> pd.DataFrame:
    item_master = pd.read_excel(
        path,
        sheet_name="Item_Master",
        usecols=["Item_Name", "Category", "Base_Price_₹", "Price_Tier"],
    )
    item_master = clean_string_columns(item_master, ["Item_Name", "Category", "Price_Tier"])
    item_master["Base_Price_₹"] = pd.to_numeric(
        item_master["Base_Price_₹"], errors="raise"
    ).astype(int)

    if len(item_master) != 200:
        raise ValueError(f"Expected 200 rows in Item_Master, found {len(item_master)}")
    if item_master["Item_Name"].duplicated().any():
        raise ValueError("Item_Master contains duplicate Item_Name values")
    return item_master


def validate_contracts(
    segments: pd.DataFrame,
    elasticity: pd.DataFrame,
    item_master: pd.DataFrame,
) -> None:
    segment_item_names = set(segments["Item_Name"])
    elasticity_item_names = set(elasticity["Item_Name"])
    master_item_names = set(item_master["Item_Name"])

    if segment_item_names != elasticity_item_names:
        raise ValueError("Item_Name mismatch between demand_segments.csv and elasticity_coefficients.csv")
    if segment_item_names != master_item_names:
        raise ValueError("Item_Name mismatch between demand_segments.csv and Item_Master")

    merged = segments.merge(
        elasticity[["Item_Name", "Base_Price_₹", "Elasticity_Coefficient", "Rain_Boost_Flag"]],
        on="Item_Name",
        how="inner",
        suffixes=("_segment", "_elasticity"),
    )
    if len(merged) != 200:
        raise ValueError("Segment to elasticity validation merge did not produce 200 rows")

    for column in ["Base_Price_₹", "Elasticity_Coefficient", "Rain_Boost_Flag"]:
        mismatch_mask = merged[f"{column}_segment"] != merged[f"{column}_elasticity"]
        if mismatch_mask.any():
            raise ValueError(f"{column} mismatch between demand_segments.csv and elasticity_coefficients.csv")

    master_join = segments.merge(
        item_master,
        on="Item_Name",
        how="inner",
        suffixes=("_segment", "_master"),
    )
    if len(master_join) != 200:
        raise ValueError("Segment to Item_Master validation merge did not produce 200 rows")

    for column in ["Category", "Base_Price_₹", "Price_Tier"]:
        mismatch_mask = master_join[f"{column}_segment"] != master_join[f"{column}_master"]
        if mismatch_mask.any():
            raise ValueError(f"{column} mismatch between demand_segments.csv and Item_Master")


def build_footfall_baseline(df: pd.DataFrame) -> dict[str, float]:
    daily_footfall = (
        df.groupby(["Date", "Day_of_Week"])["Daily_Footfall_Customers"].first().reset_index()
    )
    baseline = (
        daily_footfall.groupby("Day_of_Week")["Daily_Footfall_Customers"].median().to_dict()
    )
    return baseline


def build_slot_day_revenue_lookup(
    pivot_revenue: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], float, float]:
    slot_day_revenue: dict[tuple[str, str], float] = {}
    for slot in pivot_revenue.index:
        for day in pivot_revenue.columns:
            slot_day_revenue[(str(slot), str(day))] = float(pivot_revenue.loc[slot, day])

    all_revenues = list(slot_day_revenue.values())
    revenue_p25 = float(np.percentile(all_revenues, 25))
    revenue_p75 = float(np.percentile(all_revenues, 75))
    return slot_day_revenue, revenue_p25, revenue_p75


def footfall_signal(day_of_week: str, actual_footfall: float, baseline_dict: dict[str, float]) -> str:
    baseline = baseline_dict.get(day_of_week, 70.0)
    ratio = actual_footfall / baseline
    if ratio >= 1.15:
        return "HIGH"
    if ratio <= 0.85:
        return "LOW"
    return "NORMAL"


def weather_signal(weather_state: str, rain_boost_flag: bool) -> str:
    if weather_state == "Clear After Rain":
        return "HIGH"
    if weather_state == "Heavy Rain":
        return "HIGH" if rain_boost_flag else "LOW"
    return "NORMAL"


def slot_day_signal(
    time_slot: str,
    day_of_week: str,
    revenue_lookup: dict[tuple[str, str], float],
    p25: float,
    p75: float,
) -> str:
    revenue = revenue_lookup.get((time_slot, day_of_week), 0.0)
    if revenue >= p75:
        return "HIGH"
    if revenue <= p25:
        return "LOW"
    return "NORMAL"


def demand_verdict(signal_1: str, signal_2: str, signal_3: str) -> str:
    signals = [signal_1, signal_2, signal_3]
    if signals.count("HIGH") >= 2:
        return "HIGH"
    if signals.count("LOW") >= 2:
        return "LOW"
    return "NORMAL"


def round_to_nearest_five(value: float) -> int:
    return int(np.round(value / 5.0) * 5)


def round_up_to_nearest_five(value: float) -> int:
    return int(np.ceil(value / 5.0) * 5)


def round_down_to_nearest_five(value: float) -> int:
    return int(np.floor(value / 5.0) * 5)


def compute_recommended_price(
    base_price: int,
    demand_condition: str,
    elasticity_coeff: float,
    price_tier: str,
    segment: str,
) -> tuple[int, float, str, str]:
    _ = segment
    base_price_float = float(base_price)
    floor_price = base_price_float * PRICE_FLOOR_PCT
    ceiling_price = base_price_float * (1 + MAX_INCREASE_PCT)
    floor_applied = False

    if demand_condition == "HIGH":
        raw_increase = min(elasticity_coeff / 25.0, MAX_INCREASE_PCT)
        recommended = base_price_float * (1 + raw_increase)
        direction = "INCREASE"
    elif demand_condition == "LOW":
        raw_decrease = min(elasticity_coeff / 30.0, MAX_DECREASE_PCT)
        recommended = base_price_float * (1 - raw_decrease)
        direction = "DECREASE"
    else:
        recommended = base_price_float
        direction = "HOLD"

    recommended = min(recommended, ceiling_price)
    if recommended < floor_price:
        recommended = floor_price
        floor_applied = True

    recommended_int = round_to_nearest_five(recommended)
    if recommended_int > ceiling_price:
        recommended_int = round_down_to_nearest_five(ceiling_price)
    if recommended_int < floor_price:
        recommended_int = round_up_to_nearest_five(floor_price)
        floor_applied = True

    if recommended_int == base_price:
        direction = "HOLD"
    elif demand_condition == "LOW" and floor_applied and recommended_int < base_price:
        direction = "DECREASE_FLOORED"

    price_change_pct = ((recommended_int - base_price_float) / base_price_float) * 100
    review_flag = "REVIEW_REQUIRED" if price_tier == "Premium" else "OK"
    return recommended_int, price_change_pct, direction, review_flag


def generate_recommendations(
    segments: pd.DataFrame,
    footfall_baseline: dict[str, float],
    revenue_lookup: dict[tuple[str, str], float],
    revenue_p25: float,
    revenue_p75: float,
) -> pd.DataFrame:
    weekday_footfall = float(
        np.median([footfall_baseline[day] for day in DAY_TYPES["Weekday"]])
    )
    weekend_footfall = float(
        np.median([footfall_baseline[day] for day in DAY_TYPES["Weekend"]])
    )

    records: list[dict[str, object]] = []
    for _, item in segments.iterrows():
        item_name = str(item["Item_Name"])
        base_price = int(item["Base_Price_₹"])
        price_tier = str(item["Price_Tier"])
        elasticity_coeff = float(item["Elasticity_Coefficient"])
        segment_label = str(item["Segment"])
        rain_boost = bool(item["Rain_Boost_Flag"])
        abc_class = str(item["ABC_Class"])
        category = str(item["Category"])

        for slot in TIME_SLOT_ORDER:
            for day_type, day_list in DAY_TYPES.items():
                _ = day_list
                rep_day = WEEKEND_REP_DAY if day_type == "Weekend" else WEEKDAY_REP_DAY
                footfall = weekend_footfall if day_type == "Weekend" else weekday_footfall

                for weather in WEATHER_ORDER:
                    signal_1 = footfall_signal(rep_day, footfall, footfall_baseline)
                    signal_2 = weather_signal(weather, rain_boost)
                    signal_3 = slot_day_signal(slot, rep_day, revenue_lookup, revenue_p25, revenue_p75)
                    verdict = demand_verdict(signal_1, signal_2, signal_3)

                    rec_price, change_pct, direction, review_flag = compute_recommended_price(
                        base_price,
                        verdict,
                        elasticity_coeff,
                        price_tier,
                        segment_label,
                    )

                    records.append(
                        {
                            "Item_Name": item_name,
                            "ABC_Class": abc_class,
                            "Category": category,
                            "Price_Tier": price_tier,
                            "Segment": segment_label,
                            "Base_Price_₹": base_price,
                            "Time_Slot": slot,
                            "Day_Type": day_type,
                            "Weather_State": weather,
                            "Signal_Footfall": signal_1,
                            "Signal_Weather": signal_2,
                            "Signal_SlotDay": signal_3,
                            "Demand_Verdict": verdict,
                            "Recommended_Price_₹": rec_price,
                            "Price_Change_%": round(change_pct, 2),
                            "Direction": direction,
                            "Review_Flag": review_flag,
                            "Elasticity_Coefficient": round(elasticity_coeff, 4),
                            "Rain_Boost_Flag": rain_boost,
                        }
                    )

    recommendations = pd.DataFrame(records)
    return recommendations


def run_sanity_checks(recommendations: pd.DataFrame) -> tuple[int, int, int]:
    floor_violations = recommendations[
        recommendations["Recommended_Price_₹"] < recommendations["Base_Price_₹"] * PRICE_FLOOR_PCT
    ]
    ceiling_violations = recommendations[
        recommendations["Price_Change_%"] > MAX_INCREASE_PCT * 100 + 0.1
    ]
    premium_unflagged = recommendations[
        (recommendations["Price_Tier"] == "Premium")
        & (recommendations["Review_Flag"] == "OK")
    ]
    return len(floor_violations), len(ceiling_violations), len(premium_unflagged)


def save_recommendation_charts(recommendations: pd.DataFrame, charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Price Recommendations - Distribution Analysis\nBlend Cafe Dynamic Pricing Engine",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].hist(
        recommendations["Price_Change_%"],
        bins=30,
        color="#2c7bb6",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.8,
    )
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Price Change %", fontsize=11)
    axes[0].set_ylabel("Number of Recommendations", fontsize=11)
    axes[0].set_title("Distribution of Price Changes\n(All Items x All Conditions)", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    seg_change = recommendations.groupby("Segment")["Price_Change_%"].mean().sort_values()
    colors = ["#d62728" if value > 0 else "#2c7bb6" for value in seg_change.values]
    bars = axes[1].barh(
        seg_change.index,
        seg_change.values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, value in zip(bars, seg_change.values):
        x_text = value + 0.1 if value >= 0 else value - 0.1
        axes[1].text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            fontweight="bold",
        )

    axes[1].axvline(x=0, color="black", linewidth=1)
    axes[1].set_xlabel("Mean Price Change %", fontsize=11)
    axes[1].set_title("Mean Price Change by Pricing Segment", fontsize=11)
    axes[1].grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(charts_dir / "price_change_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    heatmap_data = recommendations.pivot_table(
        index="Time_Slot",
        columns="Weather_State",
        values="Price_Change_%",
        aggfunc="mean",
    )
    heatmap_data = heatmap_data.reindex(index=TIME_SLOT_ORDER, columns=WEATHER_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto", vmin=-8, vmax=8)
    plt.colorbar(image, ax=ax, label="Mean Price Change %")

    for row_index, slot in enumerate(TIME_SLOT_ORDER):
        for col_index, weather in enumerate(WEATHER_ORDER):
            value = float(heatmap_data.loc[slot, weather])
            ax.text(
                col_index,
                row_index,
                f"{value:+.1f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="black" if abs(value) < 5 else "white",
            )

    ax.set_xticks(range(len(WEATHER_ORDER)))
    ax.set_xticklabels(WEATHER_ORDER, fontsize=10)
    ax.set_yticks(range(len(TIME_SLOT_ORDER)))
    ax.set_yticklabels(TIME_SLOT_ORDER, fontsize=10)
    ax.set_title(
        "Mean Recommended Price Change (%) by Slot x Weather\nGreen = increase, Red = decrease",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(charts_dir / "price_heatmap_slot_weather.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_item_name(available_items: set[str], patterns: list[str]) -> str:
    for pattern in patterns:
        for item_name in sorted(available_items):
            if pattern.lower() in item_name.lower():
                return item_name
    raise ValueError(f"None of the patterns matched available items: {patterns}")


def sample_recommendation_lines(recommendations: pd.DataFrame) -> list[str]:
    available_items = set(recommendations["Item_Name"].astype(str))
    sample_specs = [
        ("Masala Chai", ["Masala Chai"], "Heavy Rain", "Dinner", "Weekend"),
        ("Iced Café Latte", ["Iced Café Latte", "Iced Latte", "Café Latte"], "Heavy Rain", "Afternoon", "Weekday"),
        ("Ramen Bowl", ["Soy Ramen Bowl", "Kaopoon Ramen Bowl", "Tonkatsu Ramen Bowl"], "Dry Day", "Afternoon", "Weekday"),
        ("Cappuccino", ["Cappuccino"], "Heavy Rain", "Morning", "Weekday"),
        ("Fondue", ["Jalapeño Cheese Fondue", "Burnt Garlic Cheese Fondue"], "Clear After Rain", "Dinner", "Weekend"),
    ]

    lines: list[str] = []
    for label, patterns, weather, slot, day_type in sample_specs:
        item_name = find_item_name(available_items, patterns)
        match = recommendations[
            (recommendations["Item_Name"] == item_name)
            & (recommendations["Weather_State"] == weather)
            & (recommendations["Time_Slot"] == slot)
            & (recommendations["Day_Type"] == day_type)
        ]
        if match.empty:
            continue

        row = match.iloc[0]
        if row["Direction"] == "HOLD":
            action_text = "HOLD"
        else:
            action_text = f"{row['Price_Change_%']:+.1f}%"
        if row["Review_Flag"] == "REVIEW_REQUIRED":
            action_text = f"{action_text}, REVIEW_REQUIRED"
        lines.append(
            f"{item_name} | {weather} | {slot} | {day_type} -> "
            f"₹{int(row['Recommended_Price_₹'])} ({action_text})"
        )
    return lines


def main() -> int:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.charts_dir.mkdir(parents=True, exist_ok=True)

    df = load_transactions(args.processed_path)
    segments = load_segments(args.segments_path)
    pivot_revenue = load_pivot_revenue(args.pivot_revenue_path)
    elasticity = load_elasticity(args.elasticity_path)
    item_master = load_item_master(args.workbook_path)
    validate_contracts(segments, elasticity, item_master)

    print(f"Transactions: {len(df)}")
    print(f"Segments: {len(segments)}")
    print(f"Items in master: {len(item_master)}")

    footfall_baseline = build_footfall_baseline(df)
    print("\nFootfall baselines by day:")
    for day in DAY_ORDER:
        print(f"  {day:<12}: {footfall_baseline[day]:.0f} customers")

    revenue_lookup, revenue_p25, revenue_p75 = build_slot_day_revenue_lookup(pivot_revenue)
    print("\nSlot-day revenue percentiles:")
    print(f"  25th percentile (trough threshold): ₹{revenue_p25:,.0f}")
    print(f"  75th percentile (peak threshold):   ₹{revenue_p75:,.0f}")

    total_combinations = len(segments) * len(TIME_SLOT_ORDER) * len(DAY_TYPES) * len(WEATHER_ORDER)
    print("\nBuilding recommendation grid...")
    print(f"  Items: {len(segments)}")
    print(f"  Slots: {len(TIME_SLOT_ORDER)}")
    print(f"  Day types: {len(DAY_TYPES)}")
    print(f"  Weather states: {len(WEATHER_ORDER)}")
    print(f"  Total combinations: {total_combinations}")

    recommendations = generate_recommendations(
        segments,
        footfall_baseline,
        revenue_lookup,
        revenue_p25,
        revenue_p75,
    )
    print(f"\nTotal recommendations generated: {len(recommendations)}")

    floor_violations, ceiling_violations, premium_unflagged = run_sanity_checks(recommendations)
    print("\n-- Sanity Checks ----------------------------------------")
    print(f"Floor violations (must be 0): {floor_violations}")
    print(f"Ceiling violations (must be 0): {ceiling_violations}")
    print(f"Premium items without review flag (must be 0): {premium_unflagged}")
    if floor_violations != 0 or ceiling_violations != 0 or premium_unflagged != 0:
        raise ValueError("Sanity checks failed; stop and debug before proceeding")

    direction_counts = recommendations["Direction"].value_counts()
    verdict_counts = recommendations["Demand_Verdict"].value_counts()
    print("\nDirection breakdown:")
    print(direction_counts.to_string())
    print("\nDemand verdict breakdown:")
    print(verdict_counts.to_string())

    recommendations = recommendations.sort_values(
        ["ABC_Class", "Segment", "Base_Price_₹"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    recommendations_path = args.models_dir / "price_recommendations.csv"
    recommendations.to_csv(recommendations_path, index=False)
    print(f"\nSaved: {display_path(recommendations_path)}")
    print(f"Shape: {recommendations.shape}")

    save_recommendation_charts(recommendations, args.charts_dir)

    print("")
    print("03_price_recommender.py complete.")
    print("")
    print("Inputs")
    print(f"Items: {len(segments)}")
    print(f"Time slots: {len(TIME_SLOT_ORDER)}")
    print(f"Day types: {len(DAY_TYPES)} (Weekday / Weekend)")
    print(f"Weather states: {len(WEATHER_ORDER)}")
    print(f"Total combinations: {len(recommendations):,}")
    print("")
    print("Sanity Checks")
    print(f"Floor violations: {floor_violations}")
    print(f"Ceiling violations: {ceiling_violations}")
    print(f"Premium items unflagged: {premium_unflagged}")
    print("")
    print("Demand Verdict Distribution")
    for verdict in ["HIGH", "NORMAL", "LOW"]:
        print(f"{verdict}: {int(verdict_counts.get(verdict, 0)):,} recommendations")
    print("")
    print("Direction Distribution")
    for direction in ["INCREASE", "HOLD", "DECREASE", "DECREASE_FLOORED"]:
        print(f"{direction}: {int(direction_counts.get(direction, 0)):,}")
    print("")
    print("Sample Recommendations (key items)")
    for line in sample_recommendation_lines(recommendations):
        print(line)
    print("")
    print("Outputs")
    print(f"{display_path(recommendations_path)} ({len(recommendations):,} rows)")
    print(f"{display_path(args.charts_dir / 'price_change_distribution.png')}")
    print(f"{display_path(args.charts_dir / 'price_heatmap_slot_weather.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
