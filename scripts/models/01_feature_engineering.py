from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_PATH = ROOT_DIR / "data" / "data_analysis" / "processed_transactions.csv"
DEFAULT_SEGMENTS_PATH = ROOT_DIR / "data" / "data_analysis" / "demand_segments.csv"
DEFAULT_WORKBOOK_PATH = ROOT_DIR / "data" / "obs" / "BlendCafe_DynamicPricing_Data.xlsx"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "models"

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = ["May", "June", "July"]
ACADEMIC_ORDER = ["Summer Vacation", "Pre-Semester", "New Semester"]
PRICE_TIER_ORDER = ["Budget", "Mid-range", "Premium"]
TIME_SLOT_ORDER = ["Morning", "Afternoon", "Evening", "Dinner"]
WEATHER_ORDER = ["Dry Day", "Light Drizzle", "Heavy Rain", "Clear After Rain"]
CUSTOMER_SEGMENT_ORDER = ["College", "Professional", "Couple-Group", "Family"]
PRICING_SEGMENT_ORDER = [
    "Dynamic Pricing Target",
    "Premium Lock",
    "Promotional Item",
    "Review Item",
]
ITEM_MASTER_COLUMNS = [
    "Item_Name",
    "Category",
    "Base_Price_₹",
    "Price_Tier",
    "Is_Vegetarian",
]
TARGET_COLUMN = "Quantity_Units"
FEATURE_COLUMNS = [
    "Month_encoded",
    "Day_of_Week_encoded",
    "Is_Weekend_encoded",
    "Order_Minute",
    "Academic_Period_encoded",
    "Slot_Morning",
    "Slot_Afternoon",
    "Slot_Evening",
    "Slot_Dinner",
    "Weather_Dry_Day",
    "Weather_Light_Drizzle",
    "Weather_Heavy_Rain",
    "Weather_Clear_After_Rain",
    "Temperature_°C",
    "Daily_Footfall_Customers",
    "Seg_College",
    "Seg_Professional",
    "Seg_Couple_Group",
    "Seg_Family",
    "Category_encoded",
    "Price_Tier_encoded",
    "Base_Price_₹",
    "Is_Vegetarian_encoded",
    "Elasticity_Coefficient",
    "Pricing_Segment_encoded",
]
METADATA_COLUMNS = [
    "Transaction_ID",
    "Date",
    "Item_Name",
    "Category",
    "Price_Tier",
    "ABC_Class",
    "Segment",
]
SPLIT_DATE = pd.Timestamp("2024-07-16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Blend Cafe transaction data into a model-ready feature matrix."
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
        "--workbook-path",
        type=Path,
        default=DEFAULT_WORKBOOK_PATH,
        help="Path to BlendCafe_DynamicPricing_Data.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model-ready CSVs and encoders will be written",
    )
    return parser.parse_args()


def clean_string_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in columns:
        df[column] = df[column].astype("string").str.strip()
        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(f"{column} contains null or blank values")
    return df


def load_transactions(path: Path) -> pd.DataFrame:
    transactions = pd.read_csv(path)
    required_columns = [
        "Transaction_ID",
        "Date",
        "Month",
        "Day_of_Week",
        "Is_Weekend",
        "Time_of_Order",
        "Time_Slot",
        "Weather",
        "Temperature_°C",
        "Academic_Period",
        "Customer_Segment",
        "Item_Name",
        "Category",
        "Is_Vegetarian",
        "Price_Tier",
        "Base_Price_₹",
        "Quantity_Units",
        "Daily_Footfall_Customers",
        "ABC_Class",
    ]
    missing_columns = [column for column in required_columns if column not in transactions.columns]
    if missing_columns:
        raise ValueError(
            f"processed_transactions.csv is missing required columns: {missing_columns}"
        )

    transactions = clean_string_columns(
        transactions,
        [
            "Month",
            "Day_of_Week",
            "Is_Weekend",
            "Time_of_Order",
            "Time_Slot",
            "Weather",
            "Academic_Period",
            "Customer_Segment",
            "Item_Name",
            "Category",
            "Is_Vegetarian",
            "Price_Tier",
            "ABC_Class",
        ],
    )
    transactions["Date"] = pd.to_datetime(
        transactions["Date"], format="mixed", dayfirst=True, errors="raise"
    )
    for column in ["Temperature_°C", "Base_Price_₹", "Quantity_Units", "Daily_Footfall_Customers"]:
        transactions[column] = pd.to_numeric(transactions[column], errors="raise")

    transactions["Base_Price_₹"] = transactions["Base_Price_₹"].astype(int)
    transactions["Quantity_Units"] = transactions["Quantity_Units"].astype(int)
    transactions["Daily_Footfall_Customers"] = transactions["Daily_Footfall_Customers"].astype(int)

    if len(transactions) != 3818:
        raise ValueError(f"Expected 3818 transaction rows, found {len(transactions)}")
    if transactions["Transaction_ID"].duplicated().any():
        raise ValueError("processed_transactions.csv contains duplicate Transaction_ID values")
    if transactions["Item_Name"].nunique() != 200:
        raise ValueError(
            f"Expected 200 unique items in processed_transactions.csv, found {transactions['Item_Name'].nunique()}"
        )

    return transactions


def load_segments(path: Path) -> pd.DataFrame:
    segments = pd.read_csv(path)
    required_columns = [
        "Item_Name",
        "Segment",
        "Elasticity_Coefficient",
        "Demand_Class",
        "Rain_Boost_Flag",
        "Peak_Time_Slot",
    ]
    missing_columns = [column for column in required_columns if column not in segments.columns]
    if missing_columns:
        raise ValueError(f"demand_segments.csv is missing required columns: {missing_columns}")

    segments = clean_string_columns(
        segments,
        ["Item_Name", "Segment", "Demand_Class", "Peak_Time_Slot"],
    )
    segments["Elasticity_Coefficient"] = pd.to_numeric(
        segments["Elasticity_Coefficient"], errors="raise"
    )

    if len(segments) != 200:
        raise ValueError(f"Expected 200 rows in demand_segments.csv, found {len(segments)}")
    if segments["Item_Name"].duplicated().any():
        raise ValueError("demand_segments.csv contains duplicate Item_Name values")

    return segments[required_columns].copy()


def load_item_master(path: Path) -> pd.DataFrame:
    item_master = pd.read_excel(
        path,
        sheet_name="Item_Master",
        usecols=ITEM_MASTER_COLUMNS,
    )
    item_master = clean_string_columns(
        item_master, ["Item_Name", "Category", "Price_Tier", "Is_Vegetarian"]
    )
    item_master["Base_Price_₹"] = pd.to_numeric(
        item_master["Base_Price_₹"], errors="raise"
    ).astype(int)

    if len(item_master) != 200:
        raise ValueError(f"Expected 200 rows in Item_Master, found {len(item_master)}")
    if item_master["Item_Name"].duplicated().any():
        raise ValueError("Item_Master contains duplicate Item_Name values")

    return item_master


def validate_ordered_values(
    series: pd.Series,
    expected_values: list[str],
    column_name: str,
) -> None:
    observed_values = sorted(series.dropna().astype(str).unique().tolist())
    if sorted(expected_values) != observed_values:
        raise ValueError(
            f"{column_name} values do not match the expected project contract. "
            f"Expected {sorted(expected_values)}, found {observed_values}"
        )


def validate_item_master_alignment(
    transactions: pd.DataFrame,
    item_master: pd.DataFrame,
) -> None:
    static_columns = ["Item_Name", "Category", "Price_Tier", "Base_Price_₹", "Is_Vegetarian"]
    tx_static = transactions[static_columns].drop_duplicates().copy()
    if len(tx_static) != 200:
        raise ValueError(
            "processed_transactions.csv does not resolve to 200 unique item-level static rows"
        )

    merged = tx_static.merge(
        item_master,
        on="Item_Name",
        how="left",
        suffixes=("_processed", "_master"),
    )
    if merged["Category_master"].isna().any():
        raise ValueError("Some transaction items are missing from Item_Master")

    for column in ["Category", "Price_Tier", "Base_Price_₹", "Is_Vegetarian"]:
        mismatch_mask = merged[f"{column}_processed"] != merged[f"{column}_master"]
        if mismatch_mask.any():
            raise ValueError(
                f"{column} mismatch between processed_transactions.csv and Item_Master"
            )


def validate_project_contract(transactions: pd.DataFrame, segments: pd.DataFrame) -> None:
    validate_ordered_values(transactions["Day_of_Week"], DAY_ORDER, "Day_of_Week")
    validate_ordered_values(transactions["Month"], MONTH_ORDER, "Month")
    validate_ordered_values(transactions["Academic_Period"], ACADEMIC_ORDER, "Academic_Period")
    validate_ordered_values(transactions["Price_Tier"], PRICE_TIER_ORDER, "Price_Tier")
    validate_ordered_values(transactions["Time_Slot"], TIME_SLOT_ORDER, "Time_Slot")
    validate_ordered_values(transactions["Weather"], WEATHER_ORDER, "Weather")
    validate_ordered_values(
        transactions["Customer_Segment"],
        CUSTOMER_SEGMENT_ORDER,
        "Customer_Segment",
    )
    validate_ordered_values(transactions["Is_Vegetarian"], ["No", "Yes"], "Is_Vegetarian")
    validate_ordered_values(transactions["Is_Weekend"], ["No", "Yes"], "Is_Weekend")
    validate_ordered_values(segments["Segment"], PRICING_SEGMENT_ORDER, "Segment")


def merge_segments(transactions: pd.DataFrame, segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    merged = transactions.merge(segments, on="Item_Name", how="left")
    null_check = merged[["Segment", "Elasticity_Coefficient"]].isna().sum()
    if len(merged) != len(transactions):
        raise ValueError("Segment merge changed the transaction row count")
    if null_check.any():
        raise ValueError(f"Nulls after segment join:\n{null_check.to_string()}")
    return merged, null_check


def encode_ordered(series: pd.Series, ordered_values: list[str], column_name: str) -> pd.Series:
    mapping = {value: index for index, value in enumerate(ordered_values)}
    encoded = series.map(mapping)
    if encoded.isna().any():
        invalid_values = sorted(series.loc[encoded.isna()].astype(str).unique().tolist())
        raise ValueError(f"{column_name} contains unexpected values: {invalid_values}")
    return encoded.astype(int)


def sanitize_dummy_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def build_dummies(series: pd.Series, categories: list[str], prefix: str) -> pd.DataFrame:
    categorical = pd.Categorical(series, categories=categories)
    dummies = pd.get_dummies(categorical, prefix=prefix, dtype=int)
    dummies.columns = [sanitize_dummy_name(str(column)) for column in dummies.columns]
    return dummies


def time_to_minutes(value: str) -> int:
    parts = str(value).split(":")
    if len(parts) != 2:
        raise ValueError(f"Unexpected Time_of_Order value: {value}")
    hour, minute = parts
    if not hour.isdigit() or not minute.isdigit():
        raise ValueError(f"Unexpected Time_of_Order value: {value}")
    total_minutes = int(hour) * 60 + int(minute)
    if total_minutes < 0 or total_minutes > 1439:
        raise ValueError(f"Time_of_Order is out of range: {value}")
    return total_minutes


def engineer_features(df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, LabelEncoder, int]:
    df = df.copy()
    df["Day_of_Week_encoded"] = encode_ordered(df["Day_of_Week"], DAY_ORDER, "Day_of_Week")
    df["Month_encoded"] = encode_ordered(df["Month"], MONTH_ORDER, "Month")
    df["Academic_Period_encoded"] = encode_ordered(
        df["Academic_Period"], ACADEMIC_ORDER, "Academic_Period"
    )
    df["Price_Tier_encoded"] = encode_ordered(df["Price_Tier"], PRICE_TIER_ORDER, "Price_Tier")
    df["Pricing_Segment_encoded"] = encode_ordered(
        df["Segment"], PRICING_SEGMENT_ORDER, "Segment"
    )
    df["Is_Vegetarian_encoded"] = df["Is_Vegetarian"].eq("Yes").astype(int)
    df["Is_Weekend_encoded"] = df["Is_Weekend"].eq("Yes").astype(int)
    df["Order_Minute"] = df["Time_of_Order"].apply(time_to_minutes).astype(int)

    df = pd.concat(
        [
            df,
            build_dummies(df["Time_Slot"], TIME_SLOT_ORDER, "Slot"),
            build_dummies(df["Weather"], WEATHER_ORDER, "Weather"),
            build_dummies(df["Customer_Segment"], CUSTOMER_SEGMENT_ORDER, "Seg"),
        ],
        axis=1,
    )

    label_encoder = LabelEncoder()
    df["Category_encoded"] = label_encoder.fit_transform(df["Category"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, output_dir / "le_category.pkl")

    missing_feature_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_feature_columns:
        raise ValueError(f"Missing feature columns: {missing_feature_columns}")

    null_feature_count = int(df[FEATURE_COLUMNS].isna().sum().sum())
    if null_feature_count != 0:
        raise ValueError(f"Feature matrix contains {null_feature_count} null feature values")

    return df, label_encoder, null_feature_count


def save_outputs(
    df: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["Date", "Transaction_ID"]).reset_index(drop=True)

    feature_matrix = df[METADATA_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    feature_matrix["Split"] = np.where(feature_matrix["Date"] < SPLIT_DATE, "train", "test")

    train_df = df.loc[df["Date"] < SPLIT_DATE, FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    test_df = df.loc[df["Date"] >= SPLIT_DATE, FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(output_dir / "feature_matrix.csv", index=False)
    train_df.to_csv(output_dir / "train_features.csv", index=False)
    test_df.to_csv(output_dir / "test_features.csv", index=False)
    joblib.dump(FEATURE_COLUMNS, output_dir / "feature_columns.pkl")

    return feature_matrix, train_df, test_df


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def print_summary(
    input_rows: int,
    null_check: pd.Series,
    missing_feature_columns: list[str],
    null_feature_count: int,
    feature_matrix: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    train_dates = feature_matrix.loc[feature_matrix["Split"].eq("train"), "Date"]
    test_dates = feature_matrix.loc[feature_matrix["Split"].eq("test"), "Date"]
    train_start = train_dates.min().date()
    train_end = train_dates.max().date()
    test_start = test_dates.min().date()
    test_end = test_dates.max().date()

    print("01_feature_engineering.py complete.")
    print("")
    print(f"  Input transactions: {input_rows:,}")
    print(f"  Features engineered: {len(FEATURE_COLUMNS)}")
    print(f"  Target variable: {TARGET_COLUMN}")
    print("")
    print("  Encoding summary:")
    print("    Day_of_Week     -> label encoded (0-6)")
    print("    Month           -> label encoded (0-2)")
    print("    Academic_Period -> label encoded (0-2)")
    print("    Price_Tier      -> label encoded (0-2)")
    print("    Time_Slot       -> one-hot (4 columns)")
    print("    Weather         -> one-hot (4 columns)")
    print("    Customer_Segment-> one-hot (4 columns)")
    print("    Category        -> label encoded (0-23), encoder saved")
    print("    Is_Vegetarian   -> binary (0/1)")
    print("    Is_Weekend      -> binary (0/1)")
    print("    Time_of_Order   -> minute of day (0-1439)")
    print("")
    print(f"  Nulls after segment join:")
    print(null_check.to_string())
    print("")
    print(f"  Chronological split (split date: {SPLIT_DATE.date()}):")
    print(
        f"    Train: {len(train_df):,} rows - {train_start:%b %d} -> {train_end:%b %d}"
    )
    print(
        f"    Test:  {len(test_df):,} rows - {test_start:%b %d} -> {test_end:%b %d}"
    )
    print(
        f"    Train/Test split: {len(train_df) / input_rows * 100:.1f}% / "
        f"{len(test_df) / input_rows * 100:.1f}%"
    )
    print("")
    print(f"  Missing feature columns: {missing_feature_columns}")
    print(f"  Feature-matrix null count: {null_feature_count}")
    print("")
    print("  Outputs:")
    print(f"    {display_path(output_dir / 'feature_matrix.csv')}      (full - all rows)")
    print(f"    {display_path(output_dir / 'train_features.csv')}      (train only)")
    print(f"    {display_path(output_dir / 'test_features.csv')}       (test only)")
    print(f"    {display_path(output_dir / 'feature_columns.pkl')}     (column list)")
    print(f"    {display_path(output_dir / 'le_category.pkl')}         (category encoder)")


def main() -> int:
    args = parse_args()

    transactions = load_transactions(args.processed_path)
    segments = load_segments(args.segments_path)
    item_master = load_item_master(args.workbook_path)

    validate_project_contract(transactions, segments)
    validate_item_master_alignment(transactions, item_master)

    df, null_check = merge_segments(transactions, segments)
    df, _, null_feature_count = engineer_features(df, args.output_dir)
    feature_matrix, train_df, test_df = save_outputs(df, args.output_dir)

    missing_feature_columns = [column for column in FEATURE_COLUMNS if column not in feature_matrix.columns]
    print_summary(
        input_rows=len(transactions),
        null_check=null_check,
        missing_feature_columns=missing_feature_columns,
        null_feature_count=null_feature_count,
        feature_matrix=feature_matrix,
        train_df=train_df,
        test_df=test_df,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
