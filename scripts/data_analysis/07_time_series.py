from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DAILY_PATH = ROOT_DIR / "data" / "data_analysis" / "daily_aggregated.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "data_analysis"
DEFAULT_CHARTS_DIR = DEFAULT_OUTPUT_DIR / "charts"
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run additive weekly time series decomposition on Blend Cafe daily revenue."
    )
    parser.add_argument(
        "--daily-path",
        type=Path,
        default=DEFAULT_DAILY_PATH,
        help="Path to daily_aggregated.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where time_series_decomposed.csv will be written",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Directory where time series charts will be written",
    )
    return parser.parse_args()


def load_daily_revenue(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    daily = pd.read_csv(path)
    required_columns = ["Date", "Total_Revenue_₹"]
    missing_columns = [column for column in required_columns if column not in daily.columns]
    if missing_columns:
        raise ValueError(f"daily_aggregated.csv is missing required columns: {missing_columns}")

    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="raise")
    daily["Total_Revenue_₹"] = pd.to_numeric(daily["Total_Revenue_₹"], errors="raise")
    daily = daily.sort_values("Date").set_index("Date")

    revenue = daily["Total_Revenue_₹"].asfreq("D")
    missing_dates = pd.date_range(revenue.index[0], revenue.index[-1], freq="D").difference(
        revenue.index
    )
    if len(missing_dates) > 0:
        raise ValueError(f"Missing dates detected in daily revenue series: {missing_dates.tolist()}")

    if len(revenue) != 92:
        raise ValueError(f"Expected 92 daily observations, found {len(revenue)}")

    return daily, revenue


def run_decomposition(revenue: pd.Series):
    return seasonal_decompose(
        revenue,
        model="additive",
        period=7,
        extrapolate_trend="freq",
    )


def build_decomposed_dataframe(revenue: pd.Series, decomposition) -> pd.DataFrame:
    decomposed_df = pd.DataFrame(
        {
            "Date": revenue.index,
            "Observed": revenue.values,
            "Trend": decomposition.trend.values,
            "Seasonal": decomposition.seasonal.values,
            "Residual": decomposition.resid.values,
        }
    )
    decomposed_df["Month"] = decomposed_df["Date"].dt.strftime("%B")
    decomposed_df["Day_of_Week"] = decomposed_df["Date"].dt.day_name()
    return decomposed_df


def save_decomposed_csv(decomposed_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decomposed_df.to_csv(output_dir / "time_series_decomposed.csv", index=False)


def save_decomposition_chart(
    revenue: pd.Series,
    trend: pd.Series,
    seasonal: pd.Series,
    residual: pd.Series,
    charts_dir: Path,
) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "Time Series Decomposition — Daily Revenue\n(Blend Café, May–July 2024)",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(revenue.index, revenue.values, color="#2c7bb6", linewidth=1.2)
    axes[0].set_ylabel("Observed (₹)", fontsize=10)
    axes[0].set_title("Observed Daily Revenue", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(trend.index, trend.values, color="#d62728", linewidth=2)
    axes[1].set_ylabel("Trend (₹)", fontsize=10)
    axes[1].set_title("Trend Component (7-day moving average basis)", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    for month_start, month_end, color, label in [
        ("2024-05-01", "2024-05-31", "#fff3cd", "May (Summer)"),
        ("2024-06-01", "2024-06-30", "#d1ecf1", "June (Monsoon onset)"),
        ("2024-07-01", "2024-07-31", "#d4edda", "July (College returns)"),
    ]:
        axes[1].axvspan(
            pd.Timestamp(month_start),
            pd.Timestamp(month_end),
            alpha=0.3,
            color=color,
            label=label,
        )
    axes[1].legend(fontsize=8, loc="upper right")

    axes[2].plot(seasonal.index, seasonal.values, color="#ff7f0e", linewidth=1.2)
    axes[2].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Seasonal (₹)", fontsize=10)
    axes[2].set_title("Seasonal Component (Weekly Pattern — Mon trough, Sat peak)", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    residual_colors = [
        "#d62728" if pd.notna(value) and value < 0 else "#2ca02c" for value in residual.values
    ]
    axes[3].bar(
        residual.index,
        residual.values,
        color=residual_colors,
        width=0.8,
        alpha=0.7,
    )
    axes[3].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    axes[3].set_ylabel("Residual (₹)", fontsize=10)
    axes[3].set_title("Residual Component (Unexplained — weather events, OOS days)", fontsize=10)
    axes[3].grid(True, alpha=0.3)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[3].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(charts_dir / "time_series_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_weekly_seasonality_chart(decomposed_df: pd.DataFrame, charts_dir: Path) -> pd.Series:
    seasonal_by_day = (
        decomposed_df.groupby("Day_of_Week")["Seasonal"].mean().reindex(DAY_ORDER)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d62728" if value > 0 else "#2c7bb6" for value in seasonal_by_day.values]
    bars = ax.bar(
        DAY_ORDER,
        seasonal_by_day.values,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.6,
    )
    ax.axhline(y=0, color="black", linewidth=1)

    for bar, value in zip(bars, seasonal_by_day.values):
        offset = 200 if value > 0 else -500
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            format_inr(value),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Seasonal Effect (₹ above/below trend)", fontsize=11)
    ax.set_title(
        "Average Weekly Seasonality — How Much Each Day Adds/Removes vs Trend\n"
        "(Blend Café, May–July 2024)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(charts_dir / "weekly_seasonality_pattern.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return seasonal_by_day


def monthly_trend_direction(trend: pd.Series) -> tuple[str, str]:
    monthly_trend = trend.groupby(trend.index.strftime("%Y-%m")).mean()
    may_to_june = direction_label(monthly_trend.loc["2024-05"], monthly_trend.loc["2024-06"])
    june_to_july = direction_label(monthly_trend.loc["2024-06"], monthly_trend.loc["2024-07"])
    return may_to_june, june_to_july


def direction_label(start: float, end: float) -> str:
    if end > start:
        return "UP"
    if end < start:
        return "DOWN"
    return "FLAT"


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


def display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(ROOT_DIR)
    except ValueError:
        return path.resolve()


def print_summary(
    revenue: pd.Series,
    trend: pd.Series,
    residual: pd.Series,
    seasonal_by_day: pd.Series,
    output_dir: Path,
    charts_dir: Path,
) -> None:
    missing_dates = pd.date_range(revenue.index[0], revenue.index[-1], freq="D").difference(
        revenue.index
    )
    may_to_june, june_to_july = monthly_trend_direction(trend)
    strongest_day = seasonal_by_day.idxmax()
    weakest_day = seasonal_by_day.idxmin()
    residual_std = residual.std()
    largest_negative_date = residual.idxmin().date()
    largest_positive_date = residual.idxmax().date()

    print("07_time_series.py complete.")
    print(
        f"  Date range: {revenue.index[0].date()} to {revenue.index[-1].date()} "
        f"({len(revenue)} days)"
    )
    if len(missing_dates) == 0:
        print("  Missing dates: none")
    else:
        print(f"  Missing dates: {missing_dates.tolist()}")
    print("")
    print("  Decomposition results:")
    print(
        f"    Trend range: {format_inr(trend.min())} (lowest) to "
        f"{format_inr(trend.max())} (highest)"
    )
    print(
        f"    Trend direction: May → June = {may_to_june} | June → July = {june_to_july}"
    )
    print(
        f"    Strongest seasonal day: {strongest_day} "
        f"({format_inr(seasonal_by_day.max())} above trend)"
    )
    print(
        f"    Weakest seasonal day:   {weakest_day} "
        f"({format_inr(seasonal_by_day.min())} below trend)"
    )
    print(f"    Residual std dev: {format_inr(residual_std)} (unexplained daily variation)")
    print(f"    Largest negative residual: {largest_negative_date}")
    print(f"    Largest positive residual: {largest_positive_date}")
    print("")
    print(f"  Output: {display_path(output_dir / 'time_series_decomposed.csv')}")
    print(f"  Charts: {display_path(charts_dir / 'time_series_decomposition.png')}")
    print(f"          {display_path(charts_dir / 'weekly_seasonality_pattern.png')}")


def main() -> int:
    args = parse_args()
    _daily, revenue = load_daily_revenue(args.daily_path)
    decomposition = run_decomposition(revenue)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    decomposed_df = build_decomposed_dataframe(revenue, decomposition)
    save_decomposed_csv(decomposed_df, args.output_dir)
    save_decomposition_chart(revenue, trend, seasonal, residual, args.charts_dir)
    seasonal_by_day = save_weekly_seasonality_chart(decomposed_df, args.charts_dir)
    print_summary(revenue, trend, residual, seasonal_by_day, args.output_dir, args.charts_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
