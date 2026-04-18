from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import kurtosis, skew


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = ROOT_DIR / "data" / "obs" / "BlendCafe_DynamicPricing_Data.xlsx"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "data_analysis"

SOURCE_DATE_FORMAT = "%d-%b-%Y"
OUTPUT_DATE_FORMAT = "%Y-%m-%d"
SOURCE_TIME_FORMAT = "%H:%M"

EXPECTED_SHEETS = {"README", "Raw_Transactions", "Daily_Summary", "Item_Master"}

RAW_COLUMNS = [
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
    "Table_No",
    "Order_Type",
    "Item_Name",
    "Category",
    "Is_Vegetarian",
    "Price_Tier",
    "Base_Price_₹",
    "Quantity_Units",
    "Daily_Footfall_Customers",
    "Revenue_₹",
]

DAILY_SUMMARY_COLUMNS = [
    "Date",
    "Month",
    "Day_of_Week",
    "Is_Weekend",
    "Weather",
    "Temperature_°C",
    "Academic_Period",
    "Daily_Footfall_Customers",
    "Total_Transactions",
    "Total_Quantity_Units",
    "Total_Revenue_₹",
    "Avg_Order_Value_₹",
    "Avg_Items_Per_Customer",
    "Pct_Vegetarian_Orders_%",
    "Pct_Premium_Orders_%",
    "Top_Selling_Category",
    "Out_of_Stock_Items",
]

ITEM_MASTER_COLUMNS = [
    "Item_Name",
    "Category",
    "Base_Price_₹",
    "Price_Tier",
    "Is_Vegetarian",
    "Available_Slots",
    "Primary_Segments",
    "Weather_Sensitive",
]

MONTH_VALUES = {"May", "June", "July"}
DAY_VALUES = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}
YES_NO_VALUES = {"Yes", "No"}
TIME_SLOT_VALUES = {"Morning", "Afternoon", "Evening", "Dinner"}
WEATHER_VALUES = {"Dry Day", "Light Drizzle", "Heavy Rain", "Clear After Rain"}
ACADEMIC_PERIOD_VALUES = {"Summer Vacation", "Pre-Semester", "New Semester"}
CUSTOMER_SEGMENT_VALUES = {"College", "Professional", "Couple-Group", "Family"}
ORDER_TYPE_VALUES = {"Dine-in", "Takeaway"}
PRICE_TIER_VALUES = {"Budget", "Mid-range", "Premium"}
HOT_BEVERAGE_CATEGORIES = {"Hot Beverages", "Hot Brews", "Hot Chocolate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load Blend Cafe Excel data, validate key assumptions, and write "
            "clean CSVs for the analysis pipeline."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to BlendCafe_DynamicPricing_Data.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where processed CSV outputs will be written",
    )
    return parser.parse_args()


def expect_columns(frame: pd.DataFrame, expected: list[str], label: str) -> None:
    missing = [column for column in expected if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected]
    if missing or extra:
        raise ValueError(
            f"{label} schema mismatch. Missing={missing or 'None'}; "
            f"Unexpected={extra or 'None'}"
        )


def strip_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    object_columns = cleaned.select_dtypes(include="object").columns
    for column in object_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()
        cleaned.loc[cleaned[column].eq("nan"), column] = pd.NA
    return cleaned


def validate_allowed_values(
    frame: pd.DataFrame, column: str, allowed_values: set[str]
) -> None:
    observed = set(frame[column].dropna().unique())
    unexpected = sorted(observed - allowed_values)
    if unexpected:
        raise ValueError(f"{column} contains unexpected values: {unexpected}")


def clean_transactions(raw_transactions: pd.DataFrame) -> pd.DataFrame:
    expect_columns(raw_transactions, RAW_COLUMNS, "Raw_Transactions")
    transactions = strip_object_columns(raw_transactions[RAW_COLUMNS])

    transactions["Date"] = pd.to_datetime(
        transactions["Date"], format=SOURCE_DATE_FORMAT, errors="raise"
    )
    transactions["Time_of_Order"] = pd.to_datetime(
        transactions["Time_of_Order"], format=SOURCE_TIME_FORMAT, errors="raise"
    ).dt.strftime(SOURCE_TIME_FORMAT)
    transactions["Temperature_°C"] = pd.to_numeric(
        transactions["Temperature_°C"], errors="raise"
    ).round(1)

    integer_columns = [
        "Transaction_ID",
        "Table_No",
        "Base_Price_₹",
        "Quantity_Units",
        "Daily_Footfall_Customers",
        "Revenue_₹",
    ]
    for column in integer_columns:
        transactions[column] = pd.to_numeric(transactions[column], errors="raise").astype(
            int
        )

    validate_allowed_values(transactions, "Month", MONTH_VALUES)
    validate_allowed_values(transactions, "Day_of_Week", DAY_VALUES)
    validate_allowed_values(transactions, "Is_Weekend", YES_NO_VALUES)
    validate_allowed_values(transactions, "Time_Slot", TIME_SLOT_VALUES)
    validate_allowed_values(transactions, "Weather", WEATHER_VALUES)
    validate_allowed_values(transactions, "Academic_Period", ACADEMIC_PERIOD_VALUES)
    validate_allowed_values(transactions, "Customer_Segment", CUSTOMER_SEGMENT_VALUES)
    validate_allowed_values(transactions, "Order_Type", ORDER_TYPE_VALUES)
    validate_allowed_values(transactions, "Is_Vegetarian", YES_NO_VALUES)
    validate_allowed_values(transactions, "Price_Tier", PRICE_TIER_VALUES)

    return transactions.sort_values(["Date", "Transaction_ID"]).reset_index(drop=True)


def clean_daily_summary(daily_summary: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    expect_columns(daily_summary, DAILY_SUMMARY_COLUMNS, "Daily_Summary")
    summary = strip_object_columns(daily_summary[DAILY_SUMMARY_COLUMNS])

    parsed_dates = pd.to_datetime(summary["Date"], format=SOURCE_DATE_FORMAT, errors="coerce")
    removed_rows = int(parsed_dates.isna().sum())
    summary = summary.loc[parsed_dates.notna()].copy()
    summary["Date"] = parsed_dates.loc[parsed_dates.notna()]

    integer_columns = [
        "Daily_Footfall_Customers",
        "Total_Transactions",
        "Total_Quantity_Units",
        "Total_Revenue_₹",
    ]
    float_columns = [
        "Temperature_°C",
        "Avg_Order_Value_₹",
        "Avg_Items_Per_Customer",
        "Pct_Vegetarian_Orders_%",
        "Pct_Premium_Orders_%",
    ]

    for column in integer_columns:
        summary[column] = pd.to_numeric(summary[column], errors="raise").astype(int)
    for column in float_columns:
        summary[column] = pd.to_numeric(summary[column], errors="raise").round(2)

    summary["Out_of_Stock_Items"] = summary["Out_of_Stock_Items"].fillna("")

    validate_allowed_values(summary, "Month", MONTH_VALUES)
    validate_allowed_values(summary, "Day_of_Week", DAY_VALUES)
    validate_allowed_values(summary, "Is_Weekend", YES_NO_VALUES)
    validate_allowed_values(summary, "Weather", WEATHER_VALUES)
    validate_allowed_values(summary, "Academic_Period", ACADEMIC_PERIOD_VALUES)

    return summary.sort_values("Date").reset_index(drop=True), removed_rows


def clean_item_master(item_master: pd.DataFrame) -> pd.DataFrame:
    expect_columns(item_master, ITEM_MASTER_COLUMNS, "Item_Master")
    items = strip_object_columns(item_master[ITEM_MASTER_COLUMNS])
    items["Base_Price_₹"] = pd.to_numeric(items["Base_Price_₹"], errors="raise").astype(int)

    validate_allowed_values(items, "Price_Tier", PRICE_TIER_VALUES)
    validate_allowed_values(items, "Is_Vegetarian", YES_NO_VALUES)

    return items.sort_values(["Category", "Item_Name"]).reset_index(drop=True)


def validate_transaction_contract(
    transactions: pd.DataFrame, daily_summary: pd.DataFrame, item_master: pd.DataFrame
) -> None:
    if len(transactions) != 3818:
        raise ValueError(f"Expected 3818 transaction rows, found {len(transactions)}")
    if transactions["Transaction_ID"].duplicated().any():
        raise ValueError("Transaction_ID must be unique in Raw_Transactions")
    if transactions["Revenue_₹"].ne(
        transactions["Base_Price_₹"] * transactions["Quantity_Units"]
    ).any():
        raise ValueError("Revenue_₹ must equal Base_Price_₹ * Quantity_Units")
    if transactions["Date"].nunique() != 92:
        raise ValueError("Expected 92 distinct transaction dates")
    if transactions["Date"].min() != pd.Timestamp("2024-05-01") or transactions[
        "Date"
    ].max() != pd.Timestamp("2024-07-31"):
        raise ValueError("Transaction date range must be 2024-05-01 to 2024-07-31")

    if len(item_master) != 200:
        raise ValueError(f"Expected 200 item master rows, found {len(item_master)}")
    if item_master["Item_Name"].duplicated().any():
        raise ValueError("Item_Master must contain unique Item_Name values")
    if item_master["Category"].nunique() != 24:
        raise ValueError("Expected 24 unique menu categories in Item_Master")

    merged = transactions.merge(item_master, on="Item_Name", how="left", suffixes=("", "_master"))
    if merged["Category_master"].isna().any():
        missing_items = sorted(merged.loc[merged["Category_master"].isna(), "Item_Name"].unique())
        raise ValueError(f"Transactions missing in Item_Master: {missing_items}")

    for column in ["Category", "Base_Price_₹", "Price_Tier", "Is_Vegetarian"]:
        mismatch_mask = merged[column] != merged[f"{column}_master"]
        if mismatch_mask.any():
            sample = merged.loc[mismatch_mask, ["Item_Name", column, f"{column}_master"]].head(5)
            raise ValueError(f"{column} mismatch between Raw_Transactions and Item_Master:\n{sample}")

    allowed_slots = item_master.set_index("Item_Name")["Available_Slots"].to_dict()
    invalid_slot_mask = transactions.apply(
        lambda row: row["Time_Slot"]
        not in {slot.strip() for slot in allowed_slots[row["Item_Name"]].split("|")},
        axis=1,
    )
    if invalid_slot_mask.any():
        sample = transactions.loc[invalid_slot_mask, ["Item_Name", "Time_Slot"]].head(5)
        raise ValueError(f"Found transactions outside Item_Master slot availability:\n{sample}")

    if len(daily_summary) != 92:
        raise ValueError(f"Expected 92 daily summary rows after cleanup, found {len(daily_summary)}")


def validate_price_tiers(frame: pd.DataFrame, label: str) -> None:
    budget_mask = frame["Base_Price_₹"] < 200
    mid_range_mask = frame["Base_Price_₹"].between(200, 400, inclusive="both")
    premium_mask = frame["Base_Price_₹"] > 400

    invalid_rows = frame.loc[
        (budget_mask & frame["Price_Tier"].ne("Budget"))
        | (mid_range_mask & frame["Price_Tier"].ne("Mid-range"))
        | (premium_mask & frame["Price_Tier"].ne("Premium")),
        ["Item_Name", "Base_Price_₹", "Price_Tier"],
    ]
    if not invalid_rows.empty:
        raise ValueError(f"{label} contains invalid price tier assignments:\n{invalid_rows.head(5)}")


def compute_daily_metrics(transactions: pd.DataFrame) -> pd.DataFrame:
    daily_base = (
        transactions.groupby("Date", as_index=False)
        .agg(
            **{
                "Month": ("Month", "first"),
                "Day_of_Week": ("Day_of_Week", "first"),
                "Is_Weekend": ("Is_Weekend", "first"),
                "Weather": ("Weather", "first"),
                "Temperature_°C": ("Temperature_°C", "first"),
                "Academic_Period": ("Academic_Period", "first"),
                "Daily_Footfall_Customers": ("Daily_Footfall_Customers", "first"),
                "Total_Transactions": ("Transaction_ID", "count"),
                "Total_Quantity_Units": ("Quantity_Units", "sum"),
                "Total_Revenue_₹": ("Revenue_₹", "sum"),
                "Avg_Order_Value_₹": ("Revenue_₹", "mean"),
            }
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    daily_base["Avg_Order_Value_₹"] = daily_base["Avg_Order_Value_₹"].round(2)

    enrichments = [
        quantity_series(
            transactions.loc[transactions["Category"].eq("Cold Brews")],
            "Cold_Brew_Quantity",
        ),
        quantity_series(
            transactions.loc[transactions["Category"].isin(HOT_BEVERAGE_CATEGORIES)],
            "Hot_Beverage_Quantity",
        ),
        revenue_series(
            transactions.loc[transactions["Price_Tier"].eq("Premium")],
            "Premium_Item_Revenue_₹",
        ),
        revenue_series(
            transactions.loc[transactions["Is_Vegetarian"].eq("Yes")],
            "Vegetarian_Item_Revenue_₹",
        ),
        count_series(
            transactions.loc[transactions["Order_Type"].eq("Takeaway")],
            "Takeaway_Transactions",
        ),
        count_series(
            transactions.loc[transactions["Order_Type"].eq("Dine-in")],
            "Dine_In_Transactions",
        ),
        unique_series(transactions, "Item_Name", "Unique_Items_Sold"),
        unique_series(transactions, "Category", "Unique_Categories_Sold"),
    ]

    daily_metrics = daily_base.set_index("Date")
    for series in enrichments:
        daily_metrics = daily_metrics.join(series, how="left")

    integer_fill_columns = [
        "Cold_Brew_Quantity",
        "Hot_Beverage_Quantity",
        "Premium_Item_Revenue_₹",
        "Vegetarian_Item_Revenue_₹",
        "Takeaway_Transactions",
        "Dine_In_Transactions",
        "Unique_Items_Sold",
        "Unique_Categories_Sold",
    ]
    daily_metrics[integer_fill_columns] = daily_metrics[integer_fill_columns].fillna(0).astype(int)
    daily_metrics["Is_Weekend_Flag"] = daily_metrics["Is_Weekend"].eq("Yes").astype(int)

    return daily_metrics.reset_index()


def quantity_series(frame: pd.DataFrame, output_name: str) -> pd.Series:
    return frame.groupby("Date")["Quantity_Units"].sum().rename(output_name)


def revenue_series(frame: pd.DataFrame, output_name: str) -> pd.Series:
    return frame.groupby("Date")["Revenue_₹"].sum().rename(output_name)


def count_series(frame: pd.DataFrame, output_name: str) -> pd.Series:
    return frame.groupby("Date")["Transaction_ID"].count().rename(output_name)


def unique_series(frame: pd.DataFrame, column: str, output_name: str) -> pd.Series:
    return frame.groupby("Date")[column].nunique().rename(output_name)


def build_daily_aggregated(
    daily_summary: pd.DataFrame, computed_daily_metrics: pd.DataFrame
) -> pd.DataFrame:
    merged = daily_summary.merge(
        computed_daily_metrics,
        on=[
            "Date",
            "Month",
            "Day_of_Week",
            "Is_Weekend",
            "Weather",
            "Temperature_°C",
            "Academic_Period",
            "Daily_Footfall_Customers",
            "Total_Transactions",
            "Total_Quantity_Units",
            "Total_Revenue_₹",
            "Avg_Order_Value_₹",
        ],
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 92:
        raise ValueError(
            "Daily summary did not align with raw transactions on core metrics after validation"
        )

    for label in ["Pct_Vegetarian_Orders_%", "Pct_Premium_Orders_%"]:
        if merged[label].isna().any():
            raise ValueError(f"{label} contains missing values in daily summary")

    merged["Out_of_Stock_Flag"] = merged["Out_of_Stock_Items"].ne("").astype(int)
    return merged.sort_values("Date").reset_index(drop=True)


def format_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["Date"] = output["Date"].dt.strftime(OUTPUT_DATE_FORMAT)
    return output


def print_project_summary(
    input_path: Path,
    transactions: pd.DataFrame,
    daily_aggregated: pd.DataFrame,
    item_master: pd.DataFrame,
    removed_summary_rows: int,
) -> None:
    overall_revenue_summary = transactions["Revenue_₹"].describe().round(2)
    overall_quantity_summary = transactions["Quantity_Units"].describe().round(2)

    category_summary = (
        transactions.groupby("Category")["Revenue_₹"]
        .agg(
            **{
                "Transactions": "count",
                "Total_Revenue_₹": "sum",
                "Mean_Revenue_₹": "mean",
                "Median_Revenue_₹": "median",
                "Std_Revenue_₹": "std",
            }
        )
        .assign(
            Skewness=lambda frame: transactions.groupby("Category")["Revenue_₹"].apply(
                lambda values: round(float(skew(values, bias=False)), 3)
            ),
            Kurtosis=lambda frame: transactions.groupby("Category")["Revenue_₹"].apply(
                lambda values: round(float(kurtosis(values, fisher=True, bias=False)), 3)
            ),
        )
        .round({"Mean_Revenue_₹": 2, "Median_Revenue_₹": 2, "Std_Revenue_₹": 2})
        .sort_values("Total_Revenue_₹", ascending=False)
    )

    print("\nBlend Cafe data loader completed.\n")
    print(f"Input workbook          : {input_path}")
    print(f"Transactions loaded     : {len(transactions):,}")
    print(f"Daily rows retained     : {len(daily_aggregated):,}")
    print(f"Daily rows removed      : {removed_summary_rows:,} subtotal/blank rows")
    print(f"Item master rows        : {len(item_master):,}")
    print(f"Unique menu items       : {transactions['Item_Name'].nunique():,}")
    print(f"Unique categories       : {transactions['Category'].nunique():,}")
    print(
        "Date range             : "
        f"{transactions['Date'].min().date()} to {transactions['Date'].max().date()}"
    )
    print(f"Total simulated revenue : Rs {transactions['Revenue_₹'].sum():,}")
    print(
        f"Out-of-stock days       : {daily_aggregated['Out_of_Stock_Flag'].sum():,} "
        "days"
    )

    print("\nOverall revenue summary")
    print(overall_revenue_summary.to_string())

    print("\nOverall quantity summary")
    print(overall_quantity_summary.to_string())

    print("\nCategory revenue statistics")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(category_summary.to_string())


def write_outputs(
    transactions: pd.DataFrame, daily_aggregated: pd.DataFrame, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    format_date_column(transactions).to_csv(
        output_dir / "processed_transactions.csv", index=False
    )
    format_date_column(daily_aggregated).to_csv(
        output_dir / "daily_aggregated.csv", index=False
    )


def main() -> int:
    args = parse_args()
    workbook = pd.read_excel(args.input_path, sheet_name=None)

    missing_sheets = EXPECTED_SHEETS - set(workbook)
    if missing_sheets:
        raise ValueError(f"Workbook is missing expected sheets: {sorted(missing_sheets)}")

    transactions = clean_transactions(workbook["Raw_Transactions"])
    daily_summary, removed_summary_rows = clean_daily_summary(workbook["Daily_Summary"])
    item_master = clean_item_master(workbook["Item_Master"])

    validate_price_tiers(transactions, "Raw_Transactions")
    validate_price_tiers(item_master, "Item_Master")
    validate_transaction_contract(transactions, daily_summary, item_master)

    computed_daily_metrics = compute_daily_metrics(transactions)
    daily_aggregated = build_daily_aggregated(daily_summary, computed_daily_metrics)

    write_outputs(transactions, daily_aggregated, args.output_dir)
    print_project_summary(
        input_path=args.input_path,
        transactions=transactions,
        daily_aggregated=daily_aggregated,
        item_master=item_master,
        removed_summary_rows=removed_summary_rows,
    )

    print(f"\nWrote {args.output_dir / 'processed_transactions.csv'}")
    print(f"Wrote {args.output_dir / 'daily_aggregated.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
