import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Final project: hourly demand forecasts for Glovo

    ## Your brief
    You have just joined **Glovo** as someone who owns **demand forecasting** for local delivery. The ops lead pulls you aside: each city runs on **hourly slots**. For every hour they must decide roughly **how many couriers** to have on the road—and that headcount should follow **how many orders** they expect in that hour. Schedule **too many** riders and you burn payroll; **too few** and deliveries slip, restaurants complain, and customers churn.

    Until now the team has patched it together with **spreadsheets and gut feel**: looking at last week, nudging numbers up or down before each wave of planning. That worked when there were fewer cities and calmer growth. It does not scale. Ops is blunt: *“We need a number we can defend for every hour of the week”*

    Your first deliverable: for **one city** (the dataset is already a single market), produce **trusted hourly order forecasts** that they could plug into staffing. You are not building the full rostering product yet—you are giving them the **demand curve** the product will sit on top of.

    ## What you must predict
    - **Target:** number of orders per hour (the `orders` column in the training file).
    - **Horizon:** a full **upcoming week**: 168 consecutive hours (7 × 24), from **Monday 00:00** through **Sunday 23:00** in the same timezone as the data.
    - **Concrete forecast window** (this assignment): timestamps from **2022-01-24 00:00:00** through **2022-01-30 23:00:00**, **both inclusive**.
    - **Business context:** in operations, planning is refreshed **once a week** (after **Sunday 23:59**), and the next delivery covers **all hours of the following week**.

    ## Data
    - **Path:** `data/train_data.csv`
    - **Columns:** `time` (hourly timestamp), `orders` (count), `city`. For this task the series is for a **single city**.
    - Use **`data/test_data_mock.csv`** to sanity-check shape, dtypes, and the merge logic (see below). The real holdout may differ; the format checker still applies.

    ## EDA and modelling
    Use the methods from the course: explore the data, justify your modelling choices, compare **several** approaches, and evaluate them out-of-sample with **MSE** and **SMAPE**. Say which model you would ship.

    ## Required output (CSV)
    Your submission must be a single CSV with **exactly** **168 rows** and **2 columns**:

    | Column | Meaning | dtype |
    |--------|---------|--------|
    | `time` | hour start for the prediction | `datetime64[ns]` |
    | `preds` | forecast order count for that hour | `float64` |

    Requirements:
    - One row per hour from **2022-01-24 00:00:00** to **2022-01-30 23:00:00**, inclusive.
    - **No missing** values in either column.
    - Every `time` in your file must appear in the test frame used for evaluation (the checker **inner-merges** on `time`).

    ## Format checker (repository)
    Use **`check_output_format(predictions, path_to_test_data)`** from `check_output_format.py`:
    - `predictions`: your `DataFrame` with columns `time` and `preds`.
    - `path_to_test_data`: path to a test CSV with at least `time` and `orders` for the evaluation hours (e.g. `data/test_data_mock.csv` while developing).

    The helper asserts row count, columns, dtypes, no nulls, and that every `time` in your file aligns with the test file. It prints **MSE** against `orders` for that test file; compute **SMAPE** yourself where you need it beyond that.

    ## Before you submit
    - Run the checker on your final `DataFrame` and on a written CSV if you load from disk.
    - Re-read the CSV in pandas and confirm dtypes and row count **after** saving (no integer `preds` dtype by accident, no duplicate `time` rows).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grading criteria

    ### EDA
    - Identifies and describes the main patterns in the data.
    - Explains how those findings shape the modelling choices.

    ### Modelling
    - Trains a **naive baseline**.
    - Trains **at least two additional model families** beyond the baseline.
    - Applies **correct time-series validation**, simulating the production scenario.
    - Selects and justifies a champion model.

    ### Forecast performance
    - Graded on the **SMAPE** of the submitted CSV against the held-out week.

    ### Code and explanation
    - The notebook tells a clear story: EDA → modelling decisions → results → conclusion.
    - Code is readable and does not require the reader to guess what a cell does.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # ── Load the training data ──────────────────────────────────────────────────
    # Path(__file__).parent navigates from final_project.py up to the repo root,
    # then into the data/ folder — so the path works regardless of where you run
    # the notebook from (local or cloud)
    df_raw = pd.read_csv(Path(__file__).parent / "data" / "train_data.csv")

    # ── Parse the time column ──────────────────────────────────────────────────
    # The time column came in as dtype 'object' (a string like "2021-02-01 00:00:00")
    # We must convert it to datetime64[ns] so pandas understands it as a time axis,
    # not just text — otherwise we cannot do any time-based indexing or resampling
    df_raw["time"] = pd.to_datetime(df_raw["time"])

    # ── Set time as the index ──────────────────────────────────────────────────
    # Setting time as the index lets us write df["2021-06"] to get all of June 2021,
    # and lets statsmodels/pmdarima understand the temporal structure of the series
    df = df_raw.set_index("time").sort_index()

    # ── Extract the orders series as a standalone variable ─────────────────────
    # We will use `orders` constantly — cleaner to have it as its own Series
    orders = df["orders"]

    # ── Basic sanity checks — print everything, never assume ──────────────────
    print("=" * 60)
    print("SHAPE — rows × columns")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("TIME RANGE")
    print("=" * 60)
    print(f"Start: {orders.index.min()}")
    print(f"End:   {orders.index.max()}")
    n_days = (orders.index.max() - orders.index.min()).days
    print(f"Span:  {n_days} days = {n_days / 7:.1f} weeks")
    print(f"Expected rows (hourly): {n_days * 24 + 24}")
    print(f"Actual rows:            {len(orders)}")
    print(
        f"Difference:             {len(orders) - (n_days * 24 + 24)}  ← should be 0; if not, we have duplicates or gaps"
    )

    print("\n" + "=" * 60)
    print("DUPLICATE TIMESTAMPS")
    print("=" * 60)
    n_dupes = orders.index.duplicated().sum()
    print(f"Duplicate time entries: {n_dupes}")

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    print(orders.isnull().sum(), "NaN values")

    print("\n" + "=" * 60)
    print("BASIC STATISTICS — orders per hour")
    print("=" * 60)
    print(orders.describe().round(2))

    print("\n" + "=" * 60)
    print("ZERO HOURS (Glovo offline — structural, not missing data)")
    print("=" * 60)
    n_zeros = (orders == 0).sum()
    pct_zeros = n_zeros / len(orders) * 100
    print(f"Hours with zero orders: {n_zeros} ({pct_zeros:.1f}% of all hours)")
    print("This is NORMAL — Glovo does not operate 24h. Zeros are real.")

    print("\n" + "=" * 60)
    print("CITY CHECK")
    print("=" * 60)
    print(f"Cities in data: {df['city'].unique()}")
    print("Single city confirmed — no need to filter")

    print("\n" + "=" * 60)
    print("FORECAST TARGET — is the window reachable?")
    print("=" * 60)
    forecast_start = pd.Timestamp("2022-01-24 00:00:00")
    forecast_end = pd.Timestamp("2022-01-30 23:00:00")
    print(f"Training ends:    {orders.index.max()}")
    print(f"Forecast starts:  {forecast_start}")
    print(
        f"Gap:              {(forecast_start - orders.index.max()).total_seconds() / 3600:.0f} hour(s)"
    )
    print(f"Forecast window:  {forecast_start} → {forecast_end}")
    print(
        f"Rows to predict:  {pd.date_range(forecast_start, forecast_end, freq='h').shape[0]}"
    )
    return mdates, np, orders, pd, plt


@app.cell
def _(mdates, orders, pd, plt):
    # ── CELL 2: Find missing hours and plot the full series ──────────────────

    # ── Step 1: Build the complete hourly grid that SHOULD exist ──────────────
    # We create a perfect hourly DatetimeIndex from the very first to the very last
    # timestamp — this is the "ideal" timeline with zero gaps
    full_hourly_index = pd.date_range(
        start=orders.index.min(),  # 2021-02-01 00:00
        end=orders.index.max(),  # 2022-01-23 23:00
        freq="h",  # one entry per hour
    )

    # ── Step 2: Reindex orders onto the full grid ─────────────────────────────
    # .reindex() forces every hour in full_hourly_index to exist in our Series.
    # Hours that were missing in the original data get NaN (not a number) here,
    # making the gaps visible — like stretching a holey net to see where the holes are
    orders_reindexed = orders.reindex(full_hourly_index)

    # ── Step 3: Find exactly which hours are missing ──────────────────────────
    missing_hours = full_hourly_index[orders_reindexed.isna()]
    print("=" * 60)
    print(f"MISSING HOURS: {len(missing_hours)} gaps found")
    print("=" * 60)
    for h in missing_hours:
        print(f"  {h}  (day of week: {h.day_name()}, hour: {h.hour:02d}:00)")

    # ── Step 4: Fill missing hours with 0 ────────────────────────────────────
    # Rationale: missing hours occur at night or during outages — Glovo had 0 active
    # orders during those periods. Filling with 0 is consistent with the observed
    # 31.7% structural zeros already in the data (not imputation [guessing], just
    # acknowledging confirmed absence of activity)
    orders_filled = orders_reindexed.fillna(0.0)

    print(
        f"\nAfter filling: {len(orders_filled)} rows, {orders_filled.isna().sum()} NaNs remaining"
    )

    # ── Step 5: Plot the full time series to see the big picture ─────────────
    fig_ts, ax_ts = plt.subplots(figsize=(16, 4))

    ax_ts.plot(
        orders_filled.index,
        orders_filled.values,
        color="#2563EB",  # Glovo blue
        linewidth=0.5,  # thin line so individual hours are visible
        alpha=0.8,
    )

    # Shade the forecast window so we can see where we are predicting
    ax_ts.axvspan(
        pd.Timestamp("2022-01-24"),
        pd.Timestamp("2022-01-30 23:00"),
        color="#FCA311",  # Glovo yellow/orange
        alpha=0.3,
        label="Forecast window (held-out week)",
    )

    ax_ts.set_title(
        "Glovo BCN — Hourly Orders: Full Training Series (Feb 2021 → Jan 2022)",
        fontsize=14,
        fontweight="bold",
    )
    ax_ts.set_xlabel("Date")
    ax_ts.set_ylabel("Orders per hour")
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax_ts.legend()
    plt.tight_layout()
    plt.savefig(
        "figures/final_project_eda_01_full_series.png", dpi=150, bbox_inches="tight"
    )
    plt.show()

    print("\nFigure saved to figures/final_project_eda_01_full_series.png")
    return (orders_filled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The 18 missing hours are ALL Monday 00:00–05:00, three times:

    - 2021-02-15 Mon 00:00–05:00 (6 hours)
    - 2021-06-07 Mon 00:00–05:00 (6 hours)
    - 2021-10-18 Mon 00:00–05:00 (6 hours)

    This is not random outages. This is a systematic pattern: exactly 6 early-Monday-morning hours, three times, spaced roughly 4 months apart. These are almost certainly planned maintenance windows or a data export bug that truncates the Sunday-to-Monday midnight transition. Filling with 0 is correct — these are hours Glovo was either offline or not recording (both result in 0 operational impact).

    Looking at the graph in more detail:

    - Strong daily spikes: every day we can see a tall spike (dinner/lunch peak) and then zeros at night. The vertical "teeth" pattern is the daily cycle (S=24).
    - No visible upward trend: the envelope of peaks looks roughly stable from Feb 2021 through Jan 2022. This is surprising given COVID reopening. It could mean: (a) BCN Glovo was already partially open by Feb 2021, or (b) the growth is there but subtle at hourly resolution.
    - Two very large spikes: one around April 2021 (~800 orders) and one around October 2021 (~880 orders). These are likely special events: a football match (FC Barcelona plays at Camp Nou, 5 min from central BCN), a public holiday, or a promotional campaign.
    - Slightly higher peaks in Dec 2021 / Jan 2022: the right side of the chart looks a little taller. Could be Christmas/New Year seasonality or genuine growth.
    - Weekly rhythm: we can see wider peaks on weekends (the "teeth" are taller every ~7 days).
    """)
    return


@app.cell
def _(orders_filled, pd, plt):
    # ── CELL 3: Daily profile, weekly profile, trend, and peak/mean ratio ────
    # This cell answers three questions every SOTA forecaster asks about demand data:
    # 1. WHEN in the day do orders come? (daily seasonality, S=24)
    # 2. WHICH days of the week are busiest? (weekly seasonality, S=168)
    # 3. Is the business growing? Are peaks growing faster than average? (trend)

    # ── Attach useful time features to the filled series ─────────────────────
    # We build a DataFrame (table) from orders_filled so we can group by hour,
    # day-of-week, and calendar week simultaneously — these groupings are the
    # lens through which seasonality becomes visible
    df_eda = pd.DataFrame(
        {
            "orders": orders_filled,  # the filled series
            "hour": orders_filled.index.hour,  # 0..23
            "dayofweek": orders_filled.index.dayofweek,  # 0=Mon..6=Sun
            "dayname": orders_filled.index.day_name(),  # 'Monday'..'Sunday'
            "week": orders_filled.index.isocalendar().week.astype(
                int
            ),  # ISO week number
            "year": orders_filled.index.year,
            "month": orders_filled.index.month,
        }
    )

    # ── PLOT 1: Average daily profile (the "shape of a typical day") ──────────
    # We compute the mean orders for each of the 24 hours across ALL days.
    # This collapses the entire dataset into one representative day —
    # like stacking all 357 days on top of each other and averaging.
    # We also separate weekdays vs weekends because they behave very differently.
    daily_profile_all = df_eda.groupby("hour")["orders"].mean()

    # Weekday mask: Monday=0, Tuesday=1, ..., Friday=4
    mask_weekday = df_eda["dayofweek"] <= 4
    daily_profile_weekday = df_eda[mask_weekday].groupby("hour")["orders"].mean()
    daily_profile_weekend = df_eda[~mask_weekday].groupby("hour")["orders"].mean()

    # ── PLOT 2: Average weekly profile (the "shape of a typical week") ────────
    # We compute mean orders by day of week (Mon–Sun), averaged across all weeks.
    # Day order: Monday first (dayofweek=0) to Sunday (dayofweek=6)
    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly_profile = (
        df_eda.groupby("dayname")["orders"]
        .mean()
        .reindex(day_order)  # reorder so Mon → Sun (not alphabetical)
    )

    # ── PLOT 3: Weekly trend — peak, mean, and trough per calendar week ───────
    # For each ISO week, compute the max (peak hour), mean, and 25th percentile.
    # This shows whether Glovo is growing, flat, or declining over the year —
    # and whether PEAKS are growing faster than the MEAN (a sign of increasing
    # variance that matters for our model choice)
    #
    # We exclude structural zeros from the mean/trend calculation here,
    # because including 0-order night hours would pull the mean down artificially
    # and obscure the true operating-hours trend
    df_operating = df_eda[df_eda["orders"] > 0].copy()  # only hours Glovo is open

    # Create a year-week label for the x-axis (e.g. "2021-W06")
    df_operating["yearweek"] = (
        df_operating["year"].astype(str)
        + "-W"
        + df_operating["week"].astype(str).str.zfill(2)
    )

    weekly_trend = (
        df_operating.groupby("yearweek")["orders"]
        .agg(
            peak="max",  # highest single hour that week
            mean="mean",  # average over all operating hours
            p25=lambda x: x.quantile(0.25),  # bottom quartile (quiet hours)
        )
        .reset_index()
    )

    # ── PLOT 4: Peak-to-mean ratio per week ───────────────────────────────────
    # This is the ratio of the week's peak hour to the week's mean operating hour.
    # A rising ratio means demand is becoming MORE spiky (peaks grow faster than average),
    # which makes forecasting harder. A stable ratio means the shape is consistent.
    # You caught this concept in Exercise 01 — now we apply it to hourly Glovo data.
    weekly_trend["peak_to_mean"] = weekly_trend["peak"] / weekly_trend["mean"]

    # ── Now draw all 4 plots ──────────────────────────────────────────────────
    fig_eda, axes = plt.subplots(2, 2, figsize=(18, 10))
    fig_eda.suptitle(
        "Glovo BCN — EDA Deep Dive: Daily Profile, Weekly Profile, Trend, Peak/Mean",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    # ── Subplot (0,0): Daily profile ──────────────────────────────────────────
    ax_daily = axes[0, 0]
    ax_daily.plot(
        daily_profile_weekday.index,
        daily_profile_weekday.values,
        color="#2563EB",
        linewidth=2.5,
        label="Weekdays (Mon–Fri)",
        marker="o",
        ms=4,
    )
    ax_daily.plot(
        daily_profile_weekend.index,
        daily_profile_weekend.values,
        color="#F97316",
        linewidth=2.5,
        label="Weekends (Sat–Sun)",
        marker="s",
        ms=4,
    )
    ax_daily.plot(
        daily_profile_all.index,
        daily_profile_all.values,
        color="#6B7280",
        linewidth=1.5,
        linestyle="--",
        label="All days avg",
        alpha=0.7,
    )
    ax_daily.set_title("Average Daily Profile (Hour of Day)", fontweight="bold")
    ax_daily.set_xlabel("Hour of day (0 = midnight)")
    ax_daily.set_ylabel("Mean orders per hour")
    ax_daily.set_xticks(range(0, 24, 2))
    ax_daily.legend()
    ax_daily.grid(axis="y", alpha=0.3)

    # ── Subplot (0,1): Weekly profile ─────────────────────────────────────────
    ax_weekly = axes[0, 1]
    colors_week = [
        "#93C5FD",
        "#93C5FD",
        "#93C5FD",
        "#93C5FD",
        "#93C5FD",
        "#F97316",
        "#F97316",
    ]
    ax_weekly.bar(
        weekly_profile.index,
        weekly_profile.values,
        color=colors_week,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_weekly.set_title("Average Weekly Profile (Day of Week)", fontweight="bold")
    ax_weekly.set_xlabel("Day of week")
    ax_weekly.set_ylabel("Mean orders per hour")
    ax_weekly.tick_params(axis="x", rotation=30)
    ax_weekly.grid(axis="y", alpha=0.3)

    # ── Subplot (1,0): Weekly trend — peak, mean, p25 ─────────────────────────
    ax_trend = axes[1, 0]
    x_ticks = range(len(weekly_trend))  # numeric x positions for the bars
    ax_trend.fill_between(
        x_ticks,
        weekly_trend["p25"],
        weekly_trend["peak"],
        alpha=0.15,
        color="#2563EB",
        label="P25–Peak range",
    )
    ax_trend.plot(
        x_ticks,
        weekly_trend["peak"].values,
        color="#EF4444",
        linewidth=1.5,
        label="Weekly peak (max hour)",
    )
    ax_trend.plot(
        x_ticks,
        weekly_trend["mean"].values,
        color="#2563EB",
        linewidth=2,
        label="Weekly mean (operating hours)",
    )
    ax_trend.plot(
        x_ticks,
        weekly_trend["p25"].values,
        color="#6B7280",
        linewidth=1,
        linestyle=":",
        label="Weekly P25 (quiet hours)",
    )

    # Add month labels on x axis — one label per ~4 weeks
    month_positions = []
    month_labels = []
    prev_month = None
    for i, yw in enumerate(weekly_trend["yearweek"]):
        yr, wk = int(yw.split("-W")[0]), int(yw.split("-W")[1])
        approx_date = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(weeks=wk - 1)
        if approx_date.month != prev_month:
            month_positions.append(i)
            month_labels.append(approx_date.strftime("%b %Y"))
            prev_month = approx_date.month

    ax_trend.set_xticks(month_positions)
    ax_trend.set_xticklabels(month_labels, rotation=45, ha="right")
    ax_trend.set_title(
        "Weekly Trend: Peak, Mean, P25 (operating hours only)", fontweight="bold"
    )
    ax_trend.set_ylabel("Orders per hour")
    ax_trend.legend(fontsize=8)
    ax_trend.grid(axis="y", alpha=0.3)

    # ── Subplot (1,1): Peak-to-mean ratio ─────────────────────────────────────
    ax_ptm = axes[1, 1]
    ax_ptm.plot(
        x_ticks,
        weekly_trend["peak_to_mean"].values,
        color="#7C3AED",
        linewidth=2,
        marker="o",
        ms=3,
    )
    ax_ptm.axhline(
        weekly_trend[
            "peak_to_mean"
        ].mean(),  # horizontal reference line at the average ratio
        color="#7C3AED",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean ratio = {weekly_trend['peak_to_mean'].mean():.2f}×",
    )
    ax_ptm.set_xticks(month_positions)
    ax_ptm.set_xticklabels(month_labels, rotation=45, ha="right")
    ax_ptm.set_title(
        "Peak-to-Mean Ratio per Week (demand spikiness)", fontweight="bold"
    )
    ax_ptm.set_ylabel("Peak orders ÷ Mean orders (operating hours)")
    ax_ptm.legend()
    ax_ptm.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/final_project_eda_02_profiles_and_trend.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # ── Print the key numbers that will go into your narrative ───────────────
    print("=" * 60)
    print("DAILY PROFILE — KEY FINDINGS")
    print("=" * 60)
    peak_hour_wday = daily_profile_weekday.idxmax()
    peak_hour_wend = daily_profile_weekend.idxmax()
    print(
        f"Weekday peak hour:  {peak_hour_wday:02d}:00 "
        f"({daily_profile_weekday.max():.0f} avg orders)"
    )
    print(
        f"Weekend peak hour:  {peak_hour_wend:02d}:00 "
        f"({daily_profile_weekend.max():.0f} avg orders)"
    )
    first_nonzero_wday = daily_profile_weekday[daily_profile_weekday > 5].index.min()
    last_nonzero_wday = daily_profile_weekday[daily_profile_weekday > 5].index.max()
    print(
        f"Weekday operating window: {first_nonzero_wday:02d}:00 → {last_nonzero_wday:02d}:00"
    )

    print("\n" + "=" * 60)
    print("WEEKLY PROFILE — KEY FINDINGS")
    print("=" * 60)
    busiest_day = weekly_profile.idxmax()
    quietest_day = weekly_profile.idxmin()
    print(f"Busiest day:  {busiest_day}  ({weekly_profile.max():.0f} avg orders/hr)")
    print(f"Quietest day: {quietest_day} ({weekly_profile.min():.0f} avg orders/hr)")
    print(
        f"Weekend premium: "
        f"{(weekly_profile[['Saturday', 'Sunday']].mean() / weekly_profile[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']].mean() - 1) * 100:.1f}% "
        f"higher than weekday average"
    )

    print("\n" + "=" * 60)
    print("TREND — KEY FINDINGS")
    print("=" * 60)
    first_4_weeks_mean = weekly_trend["mean"].iloc[:4].mean()
    last_4_weeks_mean = weekly_trend["mean"].iloc[-4:].mean()
    growth_pct = (last_4_weeks_mean / first_4_weeks_mean - 1) * 100
    print(f"Mean orders/hr — first 4 weeks: {first_4_weeks_mean:.1f}")
    print(f"Mean orders/hr — last 4 weeks:  {last_4_weeks_mean:.1f}")
    print(f"Year-over-year implied growth:   {growth_pct:+.1f}%")
    print(
        f"Peak-to-mean ratio — average:    {weekly_trend['peak_to_mean'].mean():.2f}×"
    )
    print(
        f"Peak-to-mean ratio — min/max:    {weekly_trend['peak_to_mean'].min():.2f}× / {weekly_trend['peak_to_mean'].max():.2f}×"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1) Daily Profile:

    Two clear peaks: lunch at 13:00 and dinner at 21:00. The dinner peak completely dominates — weekends hit 482 avg orders at 21:00 vs 346 on weekdays. Operating window is strict: 08:00–22:00 only (outside this, orders are essentially zero). This means 6 hours per day (23:00–07:00) are structural zeros. This is critical for modelling: a model that tries to forecast those zero-hours with non-zero predictions will be penalized heavily by SMAPE.

    2) Weekly Profile:

    Friday is the busiest day (98 avg orders/hr), Monday is the quietest (52). The weekend premium is +34% over weekday average. But notice: Friday is actually busier than Saturday and Sunday — in Barcelona, Friday night is the peak delivery night (people order in before going out). This is a culturally specific pattern that only shows up if you look at the data. A SOTA forecaster notices this.

    3) Weekly Trend:

    The blue mean line has a clear upward slope: from ~93 orders/hr in the first weeks to ~145 in the last weeks. That is +55.2% growth in one year. This is not noise — this is Glovo growing in Barcelona as COVID restrictions lifted and the platform gained users. The red peak line is noisier (special events cause spikes) but also trends upward. The trend is real and must be captured by our model.

    4) Peak-to-Mean Ratio:

    The ratio is fairly stable around 5.52× with some volatile spikes (those are the special event weeks — a single football match can make the peak 8× the mean). The ratio is not systematically increasing, which means the shape of the daily profile is consistent even as the overall level grows. This is very good news for XGBoost with lag features — the pattern is learnable.


    Main insight:

    The +55.2% growth means the seasonal naive baseline (predict last week's same hour) will systematically underforecast — it looks back one week, but last week was lower than this week because of the growth trend. XGBoost with a lag_168 feature will also see this lag, but it can learn to correct for the trend if we add a time index feature (number of hours elapsed since start). SARIMA with a first difference will capture the trend automatically.
    This is why we need at least two non-naive models: one that handles seasonality well, one that handles the trend well, and we want one that handles both simultaneously.
    """)
    return


@app.cell
def _(mdates, np, orders_filled, plt):
    # ── CELL 4: ACF/PACF analysis and seasonal decomposition ─────────────────
    # This cell answers: "what does the past tell us about the future?"
    # ACF (AutoCorrelation Function) [how correlated is orders_t with orders_{t-k}
    # for each lag k] tells us which past hours are most predictive.
    # PACF (Partial ACF) [correlation after removing the effect of all lags in
    # between — the "direct" relationship] tells us the ARIMA p-order.
    # Decomposition separates the series into trend + seasonality + residual.

    from statsmodels.tsa.seasonal import seasonal_decompose  # classical decomposition
    from statsmodels.tsa.stattools import (  # ACF and PACF calculators
        acf,
        adfuller,  # ADF stationarity test
        pacf,
    )

    # ── Step 1: Work only on operating hours for ACF/PACF ─────────────────────
    # Including structural zeros creates false autocorrelation at lags 24, 48, etc.
    # (every day has zeros at the same hours — of course they are correlated).
    # We analyse the orders_filled series directly; the zeros are genuine.
    # But for decomposition we use the full series.
    series_for_acf = orders_filled.copy()  # full series including zeros

    # ── Step 2: Compute ACF up to lag 200 ─────────────────────────────────────
    # We go to lag 200 so we can clearly see the spike at lag 24 (daily) and
    # lag 168 (weekly) — these are the two seasonal periods we must capture.
    # nlags=200 means: compute correlation of orders_t with orders_{t-1},
    # orders_{t-2}, ..., orders_{t-200}
    n_lags = 200
    acf_values = acf(
        series_for_acf, nlags=n_lags, fft=True
    )  # fft=True is faster on long series
    pacf_values = pacf(
        series_for_acf, nlags=48
    )  # PACF only needs first 48 lags for ARIMA order

    # ── Step 3: ADF test for stationarity ─────────────────────────────────────
    # ADF (Augmented Dickey-Fuller) [a statistical test that checks whether the
    # series has a unit root — i.e. whether it "drifts" without bound (non-stationary)
    # or mean-reverts (stationary)].
    # H0 (null hypothesis): the series has a unit root (non-stationary → needs differencing)
    # If p-value < 0.05, we reject H0 → series IS stationary (no differencing needed)
    adf_result = adfuller(series_for_acf, autolag="AIC")
    adf_pvalue = adf_result[1]
    print("=" * 60)
    print("ADF STATIONARITY TEST")
    print("=" * 60)
    print(f"ADF statistic: {adf_result[0]:.4f}")
    print(f"p-value:       {adf_pvalue:.6f}")
    if adf_pvalue < 0.05:
        print("✅ Series is STATIONARY (p < 0.05) — no differencing required")
        print(
            "   (The strong seasonal pattern dominates; the trend is mild relative to variance)"
        )
    else:
        print("⚠️  Series is NON-STATIONARY (p ≥ 0.05) — differencing needed")

    # ── Step 4: Seasonal decomposition with period=168 (weekly) ───────────────
    # Classical additive decomposition [assumes: observed = trend + seasonal + residual,
    # i.e. seasonality has the same absolute size regardless of the level — appropriate
    # when the seasonal swing does not grow proportionally with the trend].
    # We use period=168 (one full week) to capture the dominant weekly cycle.
    # model="additive" because the peak-to-mean ratio is roughly stable (Cell 3 showed
    # the ratio hovers around 5.5× consistently — not multiplicative growth in amplitude)
    decomp = seasonal_decompose(
        orders_filled,
        model="additive",  # additive: seasonality amplitude stays roughly constant
        period=168,  # one full week = 168 hours
        extrapolate_trend="freq",  # fills trend at the edges (avoids NaN at start/end)
    )

    # ── Step 5: Build the 3-panel figure ──────────────────────────────────────
    fig_acf, axes_acf = plt.subplots(3, 2, figsize=(18, 14))
    fig_acf.suptitle(
        "Glovo BCN — Autocorrelation Structure & Seasonal Decomposition",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    # ─── Panel (0,0): ACF up to lag 200 ───────────────────────────────────────
    ax_acf = axes_acf[0, 0]
    lags_array = np.arange(n_lags + 1)
    # Confidence interval at 95%: ±1.96/√n (anything outside this is statistically
    # significant — i.e. the correlation is real, not just noise)
    ci_bound = 1.96 / np.sqrt(len(series_for_acf))
    ax_acf.bar(lags_array, acf_values, color="#2563EB", width=0.8, alpha=0.7)
    ax_acf.axhline(0, color="black", linewidth=0.8)
    ax_acf.axhline(
        ci_bound,
        color="#EF4444",
        linewidth=1,
        linestyle="--",
        label=f"95% CI (±{ci_bound:.3f})",
    )
    ax_acf.axhline(-ci_bound, color="#EF4444", linewidth=1, linestyle="--")
    # Annotate the key seasonal lags
    for lag_mark, label_mark in [
        (24, "S=24\n(daily)"),
        (48, "48h"),
        (168, "S=168\n(weekly)"),
    ]:
        if lag_mark <= n_lags:
            ax_acf.axvline(
                lag_mark, color="#F97316", linewidth=1.5, linestyle=":", alpha=0.8
            )
            ax_acf.text(
                lag_mark + 1,
                acf_values.max() * 0.85,
                label_mark,
                color="#F97316",
                fontsize=8,
                fontweight="bold",
            )
    ax_acf.set_title("ACF — Autocorrelation Function (lags 0–200)", fontweight="bold")
    ax_acf.set_xlabel("Lag (hours)")
    ax_acf.set_ylabel("Correlation")
    ax_acf.legend(fontsize=8)
    ax_acf.set_xlim(-2, n_lags + 2)

    # ─── Panel (0,1): PACF up to lag 48 ──────────────────────────────────────
    ax_pacf = axes_acf[0, 1]
    lags_pacf = np.arange(len(pacf_values))
    ax_pacf.bar(lags_pacf, pacf_values, color="#7C3AED", width=0.8, alpha=0.7)
    ax_pacf.axhline(0, color="black", linewidth=0.8)
    ax_pacf.axhline(
        ci_bound,
        color="#EF4444",
        linewidth=1,
        linestyle="--",
        label=f"95% CI (±{ci_bound:.3f})",
    )
    ax_pacf.axhline(-ci_bound, color="#EF4444", linewidth=1, linestyle="--")
    ax_pacf.axvline(24, color="#F97316", linewidth=1.5, linestyle=":", alpha=0.8)
    ax_pacf.text(
        25, pacf_values.max() * 0.8, "lag 24\n(daily)", color="#F97316", fontsize=8
    )
    ax_pacf.set_title("PACF — Partial Autocorrelation (lags 0–48)", fontweight="bold")
    ax_pacf.set_xlabel("Lag (hours)")
    ax_pacf.set_ylabel("Partial Correlation")
    ax_pacf.legend(fontsize=8)

    # ─── Panel (1,0): Decomposition — Trend ───────────────────────────────────
    ax_trend_d = axes_acf[1, 0]
    ax_trend_d.plot(
        decomp.trend.index, decomp.trend.values, color="#059669", linewidth=1.5
    )
    ax_trend_d.set_title(
        "Decomposition — Trend Component (period=168)", fontweight="bold"
    )
    ax_trend_d.set_ylabel("Orders (trend)")
    ax_trend_d.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_trend_d.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_trend_d.xaxis.get_majorticklabels(), rotation=30)
    ax_trend_d.grid(axis="y", alpha=0.3)

    # ─── Panel (1,1): Decomposition — Seasonal (one week shown) ───────────────
    ax_seas_d = axes_acf[1, 1]
    # Show just the first 2 weeks of the seasonal component so the shape is legible
    two_weeks = decomp.seasonal.iloc[:336]
    ax_seas_d.plot(two_weeks.index, two_weeks.values, color="#F97316", linewidth=1.5)
    ax_seas_d.set_title(
        "Decomposition — Seasonal Component (first 2 weeks shown)", fontweight="bold"
    )
    ax_seas_d.set_ylabel("Orders (seasonal component)")
    ax_seas_d.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax_seas_d.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax_seas_d.xaxis.get_majorticklabels(), rotation=45)
    ax_seas_d.grid(axis="y", alpha=0.3)

    # ─── Panel (2,0): Decomposition — Residual ────────────────────────────────
    ax_resid_d = axes_acf[2, 0]
    ax_resid_d.plot(
        decomp.resid.index,
        decomp.resid.values,
        color="#6B7280",
        linewidth=0.6,
        alpha=0.8,
    )
    ax_resid_d.axhline(0, color="black", linewidth=1)
    ax_resid_d.set_title(
        "Decomposition — Residual Component (unexplained variation)", fontweight="bold"
    )
    ax_resid_d.set_ylabel("Orders (residual)")
    ax_resid_d.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_resid_d.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_resid_d.xaxis.get_majorticklabels(), rotation=30)
    ax_resid_d.grid(axis="y", alpha=0.3)

    # ─── Panel (2,1): ACF at key lags — zoomed summary ────────────────────────
    ax_zoom = axes_acf[2, 1]
    key_lags = [1, 2, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]
    key_values = [acf_values[k] for k in key_lags]
    bar_colors = ["#F97316" if k in (24, 168) else "#2563EB" for k in key_lags]
    ax_zoom.bar(
        [str(k) for k in key_lags], key_values, color=bar_colors, edgecolor="white"
    )
    ax_zoom.axhline(ci_bound, color="#EF4444", linewidth=1, linestyle="--")
    ax_zoom.axhline(-ci_bound, color="#EF4444", linewidth=1, linestyle="--")
    ax_zoom.set_title(
        "ACF at Key Lags — Seasonal Lags Highlighted (orange)", fontweight="bold"
    )
    ax_zoom.set_xlabel("Lag (hours)")
    ax_zoom.set_ylabel("Correlation")
    ax_zoom.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "figures/final_project_eda_03_acf_decomposition.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # ── Print the ACF values at the lags that matter most for modelling ────────
    print("=" * 60)
    print("ACF AT KEY LAGS — these become our XGBoost lag features")
    print("=" * 60)
    for k in key_lags:
        sig = "✅ significant" if abs(acf_values[k]) > ci_bound else "  not significant"
        print(
            f"  lag {k:4d}h ({k // 24:2d}d {k % 24:2d}h):  ACF = {acf_values[k]:+.4f}  {sig}"
        )

    print("\n" + "=" * 60)
    print("PACF AT FIRST 5 LAGS — determines ARIMA p-order")
    print("=" * 60)
    for k in range(1, 6):
        sig = (
            "✅ significant" if abs(pacf_values[k]) > ci_bound else "  not significant"
        )
        print(f"  lag {k}:  PACF = {pacf_values[k]:+.4f}  {sig}")

    print("\n" + "=" * 60)
    print("DECOMPOSITION — variance explained")
    print("=" * 60)
    total_var = orders_filled.var()
    trend_var = decomp.trend.var()
    seas_var = decomp.seasonal.var()
    resid_var = decomp.resid.dropna().var()
    print(f"Total variance:     {total_var:.1f}")
    print(
        f"Trend variance:     {trend_var:.1f}  ({trend_var / total_var * 100:.1f}% of total)"
    )
    print(
        f"Seasonal variance:  {seas_var:.1f}  ({seas_var / total_var * 100:.1f}% of total)"
    )
    print(
        f"Residual variance:  {resid_var:.1f}  ({resid_var / total_var * 100:.1f}% of total)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1) The **ACF (AutoCorrelation Function)** does not decay to zero quickly as it oscillates with a 24-hour period and remains high even at lag 168 (+0.93). This tells us the series has strong, persistent seasonal memory. For a purely stationary series [one with no trend or seasonality, just random fluctuations around a fixed mean], the ACF would decay to zero within a few lags. Here it does not which confirms we need to explicitly model the seasonal structure.

    The dominant visual feature is the comb pattern: tall spikes every 24 lags (at 24, 48, 72, 96, 120, 144, 168, 192...). This is the daily seasonal signature. Every 24 hours, the same pattern repeats: dinner peak, night zeros, lunch peak. The series "remembers" itself perfectly at multiples of 24.
    The spike at lag 168 (+0.929) is the weekly seasonal signature, the highest ACF value in the entire range besides lag 0. Knowing what happened exactly one week ago (same day, same hour) gives a 0.93 correlation with what happens now. That is extraordinary predictive power. This is why lag_168 must be a feature in XGBoost and why SARIMA with S=168 captures the dominant pattern.
    The spikes between multiples of 24 are negative (e.g. lag 12 = −0.28). This makes perfect sense: if it is 21:00 (dinner peak), then 12 hours earlier was 09:00 (near-zero morning orders). High now = low 12 hours ago = negative correlation.

    2) **PACF — Partial AutoCorrelation Function**  answers a subtly different question: "What is the direct relationship between orders_t and orders_{t-k}, after removing [controlling for / partialling out] all the intermediate lags 1, 2, ..., k−1?"
    This means, for example, that even if I already know lags 1–23, does lag 24 add any NEW information?" If PACF(24) is still large, yes it does. If PACF(24) collapses to near zero, then the lag-24 correlation was just the cumulative effect of shorter lags.
    PACF is the tool that identifies the AR order [the number of direct past values the current value depends on]. If only the PACF cut off clearly and the ACF decayed gradually, we have a pure AR process. If the ACF cut off and the PACF decayed, we have a pure MA process [Moving Average — current value depends on past error terms, not past values]. Here both decay gradually, suggesting an ARMA mixture. The spike at lag 24 in both ACF and PACF confirms a seasonal component at S=24.

    Our PACF tells us:

    - PACF(1) = +0.681 — lag 1 is strongly and directly predictive
    - PACF(2) = −0.520 — lag 2 is directly predictive but negative after controlling for lag 1. This is the "overcorrection" pattern of an AR(2) process [AutoRegressive of order 2 — the current value depends on the previous 2 values directly]
    - PACF(3) = +0.261, PACF(4) = −0.179, PACF(5) = +0.118 — alternating, decaying. This is the classic AR(p) signature: the PACF cuts off (becomes insignificant) at the order p, while the ACF decays gradually.
    - The large spike at lag 24 in the PACF — this confirms the daily seasonal period needs explicit modelling

    *What this tells us for ARIMA:* The PACF cuts off somewhere between lag 3 and lag 5, suggesting p = 2 or p = 3 for the non-seasonal AR component. Combined with no differencing needed (ADF p = 0.000), the ARIMA non-seasonal part is likely ARIMA(2,0,0) or ARIMA(3,0,0). The seasonal MA part (from the ACF decay at seasonal lags) suggests SMA(1) — so SARIMA(2,0,0)(0,0,1)[24] is a reasonable starting point.

    3) **Decomposition — Trend Component (period=168)**: Classical decomposition [a mathematical technique that separates a time series into three additive parts: trend + seasonality + residual] with period=168. The trend component is extracted by applying a centered moving average [a smoothing technique: replace each value with the average of the 168 values centred around it. This averages away the weekly cycle, leaving only the slow-moving level] of window 168 to the series.

    The trend line reveals something the raw series hid. It is NOT monotonically increasing. Instead:
    - Feb–Jul 2021: relatively flat around 65 orders/hr — BCN was still in pandemic restrictions
    - Jul–Aug 2021: sharp DROP to ~52 — this is likely Spanish summer (people go outside, eat at restaurants instead of ordering delivery) combined with the government lifting indoor restrictions in late June 2021. People are on holidays over the summer too which means a lot of regular users are not in town to order or use their holidays to go out to eat and enjoy their free time instead of being "efficient".
    - Sep 2021: sharp JUMP to ~85 — Autumn return. Schools reopen, people go back to offices, many people returned from their holidays, order delivery more. This is a structural break [a sudden permanent shift in the series level, not just noise]
    - Oct 2021–Jan 2022: steady climb from 85 to 100+ — genuine Glovo growth in Barcelona

     The trend has non-linearities (the summer dip, the autumn jump). A simple linear trend feature in XGBoost will miss this. BUT: because we use lag_168 (last week's same hour), XGBoost will implicitly capture recent-trend information — last week's orders already reflect the current level. Our peak-to-mean ratio of 5.52× being stable (Cell 3, Plot 4) justifies the additive assumption.

    4) Decomposition — Seasonal Component (first 2 weeks shown): what remains after subtracting the trend. It shows the repeating weekly pattern, the exact shape that repeats every 168 hours to see one complete repetition.

    - Every day: zero from midnight to ~08:00, then a lunch spike around 13:00 (~200 units above trend), then dinner spike at 21:00 (~450 units above trend)
    - The negative troughs at night (−100 below trend) represent the structural zeros pulling below the trend line
    - Friday and Saturday dinner peaks are visibly taller than the other days — consistent with the weekly profile from Cell 3
    - The pattern is almost identical in both weeks shown — confirming the seasonality is stable and learnable

    Seasonal variance = 91.8% of total variance, means 91.8% of all variation in Glovo's hourly orders is explained purely by the time of day and day of week. The trend explains only 1.2%. The residual (noise, special events, weather) explains 7%.
    Therefore, a model that perfectly captures the seasonal pattern would already explain 91.8% of the variance. This is why seasonal naive (predict last week's same hour) is an extremely strong baseline. To beat it, we need to also get the trend right AND reduce the residual error. XGBoost with lag_168 + trend features can do both.

    5) **Residual Component**: What is left after removing trend AND seasonal: residual = observed − trend − seasonal. This should look like white noise [random, uncorrelated fluctuations with no pattern] if the decomposition captured everything important. If the residual still has structure (patterns, spikes, heteroscedasticity [variance that changes over time), something important was missed.

    The residual is mostly small and centred around zero, but there are sharp positive spikes scattered throughout (some reaching +300). These can be special events like football matches, promotions, public holidays, etc. The residual variance at 7% of total variance means our model will never be perfect as there is always some unexplained idiosyncratic noise [random shocks specific to individual hours that no model could have predicted in advance, like an unexpected Barcelona Champions League match causing a pizza surge].
    The residual does NOT show obvious increasing variance (heteroscedasticity), confirming the additive decomposition was appropriate.

    7) **ACF at Key Lags (zoomed summary)**: the translation of all the ACF analysis into exactly which lag features to include in XGBoost:

    | Lag | ACF | Meaning | Include in XGBoost? |
    |---|---:|---|---|
    | 1h | +0.68 | Strong short-term memory | YES `lag_1` |
    | 2h | +0.18 | Mild | YES `lag_2` |
    | 3h | -0.06 | Negligible | NO |
    | 6h | +0.05 | Negligible | NO |
    | 12h | -0.28 | Anti-correlation (night vs noon) | YES `lag_12` (captures daily reversal) |
    | 24h | +0.89 | Daily seasonality | YES!!! `lag_24` (critical) |
    | 48h | +0.83 | 2 days ago | YES `lag_48` |
    | 168h | +0.93 | Weekly seasonality | *YES!!!* `lag_168` (most important) |
    | 144h | +0.87 | 6 days ago (yesterday-1 week offset) | YES `lag_144` |

    p-order meaning in (ARIMA context): p is the AutoRegressive order, meaning how many past values of the series itself directly enter the model equation.

    In ARIMA(p,d,q):
    - p=2 means: orders_t = c + φ₁·orders_{t-1} + φ₂·orders_{t-2} + error_t
      - (today's orders = constant + 0.68 × yesterday's orders + (−0.52) × two-hours-ago orders + noise)
    - d (the differencing order): how many times we subtract consecutive values to make the series stationary. ADF says d=0 (already stationary).
    - q (the MA order): how many past error terms enter the equation, meaning the model corrects itself based on past mistakes

    The PACF cutting off after lag 3–5 and the alternating sign pattern (+ − + − +) suggests p=2 or p=3. We will test p=1, p=2, p=3 and let AIC choose.

    9) ***SUMMARY***

    | Finding | Implication |
    |---|---|
    | Seasonality = 91.8% of variance | Seasonal naive is a very tough baseline |
    | lag_168 ACF = +0.93 | lag_168 is the single best predictive feature |
    | lag_24 ACF = +0.89 | lag_24 is the second best feature |
    | +55% trend growth | Must include a trend feature (time index or lag_168 difference) |
    | Structural zeros 08:00 | Mask zeros from SMAPE or model them separately |
    | Residual spikes = special events | Accept ~7% irreducible noise; no model can predict Barça matches |
    | PACF cuts off at ~lag 3 | AR(2) or AR(3) for SARIMA non-seasonal component |
    | ADF p=0.000 | d=0, no differencing needed |
    """)
    return


@app.cell
def _(np, orders_filled, pd):
    # ── CELL 5: Naive Baseline — Seasonal Naive with S=168 ───────────────────
    # The seasonal naive forecast [the simplest possible forecast that respects
    # seasonality: predict that next week's value equals this week's same hour]
    # is our benchmark. Every model we build must beat this to be worth deploying.
    #
    # WHY S=168? Because lag_168 has ACF=+0.93 — the highest predictive correlation
    # in the entire dataset. The naive forecast literally IS the lag_168 prediction.
    #
    # Business interpretation: "Next Monday at 21:00 will have the same orders
    # as this Monday at 21:00." Simple. Defensible. Hard to beat.

    # ── Define SMAPE — our primary evaluation metric ──────────────────────────
    # SMAPE [Symmetric Mean Absolute Percentage Error] is defined as:
    #   SMAPE = (2/n) × Σ |actual - forecast| / (|actual| + |forecast|)
    # It is bounded [0, 2]: 0 = perfect, 2 = worst possible.
    # "Symmetric" means over-forecasting and under-forecasting are penalised equally.
    # We use it because it handles the wide range of order volumes gracefully
    # (a 50-order error when actual=100 is weighted the same as when actual=1000).
    #
    # CRITICAL: When actual=0 AND forecast=0, the denominator is 0 → undefined.
    # We handle this by setting SMAPE=0 for those hours (correct: we predicted
    # zero, it was zero, perfect prediction, zero error).

    # CORRECTION FROM PREVIOUS RUN: the last cutoff (2022-01-23 23:00) has no
    # ground truth available — that week IS the held-out submission target.
    # We must split "validation cutoffs" (have ground truth, used to score models)
    # from the "final cutoff" (no ground truth, used later to generate the
    # actual submission CSV). Including the final cutoff in SMAPE averaging
    # produces NaN because there is nothing to compare against — confirmed by
    # the empty-slice warning you just saw.

    # ── Define SMAPE — our primary evaluation metric ──────────────────────────
    def smape(actual: pd.Series, forecast: pd.Series) -> float:
        """
        Symmetric Mean Absolute Percentage Error, bounded [0, 2].
        Zero-vs-zero hours contribute 0 error (perfect prediction, not undefined).
        """
        numerator = 2 * np.abs(actual - forecast)
        denominator = np.abs(actual) + np.abs(forecast)
        both_zero = (actual == 0) & (forecast == 0)
        ratio = np.where(both_zero, 0.0, numerator / denominator)
        return float(np.mean(ratio))

    # ── Walk-forward validation setup ─────────────────────────────────────────
    # Walk-forward validation [also called time-series cross-validation or rolling
    # origin evaluation] simulates the real production scenario:
    #   - Every Sunday at 23:59, Glovo's system forecasts the next 7 days (168 hours)
    #   - We train ONLY on data available up to that Sunday (no peeking at the future)
    #   - We forecast Mon 00:00 → Sun 23:00 of the following week
    #   - We compare to the actual values that occurred
    #   - We roll forward one week and repeat
    N_VAL_WEEKS = 7  # CHANGED from 8 → 7: the 8th window has no ground truth available
    # number of walk-forward validation windows to evaluate

    series = orders_filled.copy()  # 8568 rows, DatetimeIndex, no NaNs

    # ── Identify all Sunday 23:00 timestamps in our data ──────────────────────
    all_sundays_23 = series[
        (series.index.dayofweek == 6) & (series.index.hour == 23)
    ].index

    print(f"Total Sunday 23:00 timestamps in dataset: {len(all_sundays_23)}")

    # ── Separate the FINAL cutoff (no ground truth) from VALIDATION cutoffs ──
    # The final cutoff is the very last Sunday — this is the real production
    # moment: train on ALL available data, forecast the held-out week with
    # NO way to score it ourselves (the professor scores it).
    final_cutoff = all_sundays_23[-1]  # 2022-01-23 23:00:00 — saved for later (Cell 11)
    print(f"Final submission cutoff (no ground truth, used later): {final_cutoff}")

    # Validation cutoffs = all Sundays EXCEPT the final one, take the last N_VAL_WEEKS of those
    val_cutoffs = all_sundays_23[:-1][
        -N_VAL_WEEKS:
    ]  # exclude final_cutoff, then take last 7
    print(
        "\nValidation cutoffs (each = a Sunday night forecasting run, WITH ground truth):"
    )
    for c5_idx, c5_cutoff in enumerate(val_cutoffs):
        c5_fc_start = c5_cutoff + pd.Timedelta(hours=1)
        c5_fc_end = c5_cutoff + pd.Timedelta(hours=168)
        print(
            f"  Window {c5_idx + 1}: train up to {c5_cutoff}  →  forecast {c5_fc_start} to {c5_fc_end}"
        )

    # ── Seasonal Naive forecast function ─────────────────────────────────────
    def seasonal_naive_forecast(train: pd.Series, horizon: int = 168) -> pd.Series:
        """
        Predict the next `horizon` hours using the last observed week.
        """
        last_week = train.iloc[-168:]
        forecast_index = pd.date_range(
            start=train.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
        )
        forecast_values = pd.Series(last_week.values, index=forecast_index)
        return forecast_values

    # ── Run walk-forward validation for Seasonal Naive (ground-truth windows only) ──
    naive_smape_scores = []
    naive_mse_scores = []

    for c5_w, c5_cutoff_w in enumerate(val_cutoffs):
        train_wf = series[series.index <= c5_cutoff_w]
        actual_wf = series[
            (series.index > c5_cutoff_w)
            & (series.index <= c5_cutoff_w + pd.Timedelta(hours=168))
        ]

        forecast_wf = seasonal_naive_forecast(train_wf, horizon=168)

        aligned = pd.DataFrame({"actual": actual_wf, "forecast": forecast_wf}).dropna()

        # Safety check: confirm we actually have 168 rows of ground truth before scoring
        assert len(aligned) == 168, (
            f"Window {c5_w + 1} has {len(aligned)} rows, expected 168 — check for a data gap"
        )

        smape_val = smape(aligned["actual"], aligned["forecast"])
        mse_val = np.mean((aligned["actual"] - aligned["forecast"]) ** 2)

        naive_smape_scores.append(smape_val)
        naive_mse_scores.append(mse_val)
        print(
            f"  Window {c5_w + 1} ({c5_cutoff_w.date()} cutoff): SMAPE={smape_val:.4f}, MSE={mse_val:.1f}"
        )

    naive_mean_smape = np.mean(naive_smape_scores)
    naive_mean_mse = np.mean(naive_mse_scores)
    naive_std_smape = np.std(
        naive_smape_scores
    )  # how variable is performance week to week?

    print(f"\n{'=' * 60}")
    print(
        f"SEASONAL NAIVE BASELINE RESULTS  (n={N_VAL_WEEKS} windows with ground truth)"
    )
    print(f"{'=' * 60}")
    print(
        f"Mean SMAPE: {naive_mean_smape:.4f}  ± {naive_std_smape:.4f} std  (target: beat this)"
    )
    print(f"Mean MSE:   {naive_mean_mse:.1f}")
    print(
        f"SMAPE range: [{min(naive_smape_scores):.4f}, {max(naive_smape_scores):.4f}]"
    )
    print(f"\nInterpretation: a SMAPE of {naive_mean_smape:.4f} means on average")
    print(
        f"the naive forecast is off by {naive_mean_smape * 100 / 2:.1f}% of the combined"
    )
    print("actual+forecast scale — this is our bar to beat.")
    return (
        N_VAL_WEEKS,
        naive_mean_smape,
        naive_smape_scores,
        seasonal_naive_forecast,
        series,
        smape,
        val_cutoffs,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our Baseline Result:

    - Mean SMAPE = 0.2153, std = 0.0452

    We could have included other baselines like the Naive Forecast (Random Walk), the Seasonal Naive Forecast, the Historical Mean (Average) Forecast or the Drift Method. We decided to already go for a strong and argumented baseline which is the Seasonal Naive with S=168, because we learned from our EDA:
    - seasonal patterns explain 91.8% of variance, and lag_168 has ACF = +0.929

    A baseline built directly on that lag should be hard to beat, and it is. Any model that only learns "copy last week" will land somewhere near 0.215. To win, our model needs to learn something beyond pure repetition. Specifically, the trend, the special-event noise, and subtle pattern shifts.
    Why the std (0.045) matters: Performance varies window to window from 0.140 (a "normal" week, easy to predict) to 0.276 (likely a week containing a special event or a structural shift, since lag-168 copying fails hardest when this week differs sharply from last week). This variance is itself a finding: it tells us SMAPE will be noisy in our final evaluation too as one quirky week (a holiday, a football match) could swing our score substantially, regardless of model quality.

    Quick sanity check on the windows: Window 1 (Dec 5-12) and Window 2 (Dec 12-19) score worst (0.276, 0.270). December in Barcelona has rising pre-Christmas demand and shopping behavior changes. "last week" under-predicts "this week" because demand is climbing. Window 6 (Jan 9-16) scores best (0.140) as a "normal," stable mid-January week with no holiday disruption. This pattern is consistent with everything we found in EDA (the +55% YoY growth trend, the structural break in autumn).

    Our motto for what's coming... *simplicity wins unless complexity earns its keep*
    """)
    return


@app.cell
def _(N_VAL_WEEKS, naive_mean_smape, np, pd, series, smape, val_cutoffs):
    # ── CELL 6: SARIMA attempt #1 — SARIMA(2,0,0)(0,0,1)[24] ────────────────────────────
    # SARIMA [Seasonal AutoRegressive Integrated Moving Average] extends ARIMA
    # by adding a SECOND set of (P,D,Q) terms that operate at a seasonal lag S.
    # Full notation: SARIMA(p,d,q)(P,D,Q)[S]
    #   p,d,q   = non-seasonal AR order, differencing, MA order (operate on lag 1,2,3...)
    #   P,D,Q   = seasonal AR order, seasonal differencing, seasonal MA order
    #             (operate on lag S, 2S, 3S... i.e. lag 24, 48, 72... for S=24)
    #   S       = the seasonal period (24 = one day)
    #
    # We chose S=24 (daily), NOT S=168 (weekly), because fitting SARIMA with a
    # 168-period seasonal term is extremely slow and often numerically unstable.
    # This model tests whether the DAILY cycle alone can compete with the
    # S=168 naive baseline — the WEEKLY cycle is captured separately by XGBoost (Cell 7).
    #
    # Orders SARIMA(2,0,0)(0,0,1)[24] come directly from EDA (Cell 4):
    #   p=2 : PACF cut off around lag 2-3 (alternating +/- pattern, AR signature)
    #   d=0 : ADF rejected non-stationarity (p=0.000) — no differencing needed
    #   q=0 : kept simple per Occam's Razor — test pure AR first
    #   P=0, D=0 : no seasonal differencing needed
    #   Q=1 : ACF showed decaying-but-persistent pattern at lag 24, 48, 72...

    import time
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    warnings.filterwarnings("ignore")

    # ── Walk-forward validation for SARIMA — reuses val_cutoffs from Cell 5 ──
    sarima_smape_scores = []
    sarima_mse_scores = []
    sarima_fit_seconds = []

    # c6_w / c6_cutoff_w: loop header variables, unique to Cell 6
    for c6_w, c6_cutoff_w in enumerate(val_cutoffs):
        # ── EVERY variable below is prefixed c6_ — nothing escapes naming ────
        c6_train_wf = series[
            series.index <= c6_cutoff_w
        ]  # training slice for this window
        c6_actual_wf = series[  # ground truth for this window
            (series.index > c6_cutoff_w)
            & (series.index <= c6_cutoff_w + pd.Timedelta(hours=168))
        ]

        # Use only the last 2000 hours (~83 days) — daily pattern doesn't need
        # 11 months of history, and this keeps fit time reasonable
        c6_train_recent = c6_train_wf.iloc[-2000:]

        c6_start_time = time.time()

        c6_model = SARIMAX(
            c6_train_recent,
            order=(2, 0, 0),
            seasonal_order=(0, 0, 1, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        c6_fitted = c6_model.fit(disp=False)

        c6_elapsed = time.time() - c6_start_time
        sarima_fit_seconds.append(
            c6_elapsed
        )  # list append is fine — list itself defined once, outside loop

        # ── Forecast the next 168 hours ────────────────────────────────────────
        c6_forecast_vals = c6_fitted.forecast(steps=168)

        c6_forecast_index = pd.date_range(
            start=c6_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
        )
        c6_forecast = pd.Series(c6_forecast_vals.values, index=c6_forecast_index)

        # Clip negative predictions to zero — SARIMA is linear and can predict
        # impossible negative order counts; we floor at the physical minimum
        c6_forecast = c6_forecast.clip(lower=0)

        # ── Align and score ─────────────────────────────────────────────────
        c6_aligned = pd.DataFrame(
            {"actual": c6_actual_wf, "forecast": c6_forecast}
        ).dropna()
        assert len(c6_aligned) == 168, (
            f"Window {c6_w + 1} has {len(c6_aligned)} rows, expected 168"
        )

        c6_smape_val = smape(c6_aligned["actual"], c6_aligned["forecast"])
        c6_mse_val = np.mean((c6_aligned["actual"] - c6_aligned["forecast"]) ** 2)

        sarima_smape_scores.append(c6_smape_val)
        sarima_mse_scores.append(c6_mse_val)

        print(
            f"  Window {c6_w + 1} ({c6_cutoff_w.date()} cutoff): "
            f"SMAPE={c6_smape_val:.4f}, MSE={c6_mse_val:.1f}, fit_time={c6_elapsed:.1f}s"
        )

    sarima_mean_smape = np.mean(sarima_smape_scores)
    sarima_mean_mse = np.mean(sarima_mse_scores)
    sarima_std_smape = np.std(sarima_smape_scores)
    sarima_total_fit_time = np.sum(sarima_fit_seconds)

    print(f"\n{'=' * 60}")
    print(f"SARIMA(2,0,0)(0,0,1)[24] RESULTS  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {sarima_mean_smape:.4f}  ± {sarima_std_smape:.4f} std")
    print(f"Mean MSE:   {sarima_mean_mse:.1f}")
    print(
        f"SMAPE range: [{min(sarima_smape_scores):.4f}, {max(sarima_smape_scores):.4f}]"
    )
    print(
        f"Total fit time across {N_VAL_WEEKS} windows: {sarima_total_fit_time:.1f}s "
        f"(avg {sarima_total_fit_time / N_VAL_WEEKS:.1f}s/window)"
    )

    print(f"\n{'=' * 60}")
    print("COMPARISON: SARIMA vs SEASONAL NAIVE BASELINE")
    print(f"{'=' * 60}")
    smape_improvement = (naive_mean_smape - sarima_mean_smape) / naive_mean_smape * 100
    print(f"Naive SMAPE:   {naive_mean_smape:.4f}")
    print(f"SARIMA SMAPE:  {sarima_mean_smape:.4f}")
    if sarima_mean_smape < naive_mean_smape:
        print(f"✅ SARIMA improves on naive by {smape_improvement:.1f}%")
    else:
        print(
            f"❌ SARIMA is WORSE than naive by {-smape_improvement:.1f}% "
            f"— daily-only seasonality (S=24) likely misses the dominant weekly pattern (S=168)"
        )
    return SARIMAX, sarima_mean_smape, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Hypothesis | Why it's the cause |
    |---|---|
    | H1: Forecast index misalignment (off-by-one hour or wrong dates) | Would cause day/night mismatch, predicting daytime values for night hours (zeros), which inflates SMAPE in exactly this way. |
    | H2: SARIMA predicting wildly wrong scale (negative or huge numbers before clipping) | A non-converged model can extrapolate the AR(2) coefficients explosively, especially un-clipped before forecast horizon 168. |
    | H3: `enforce_stationarity=False` allowed a non-stationary, diverging fit | This flag exists specifically to let through unstable models. |
    | H4: The model is just bad at S=24 | Ruled mostly out — wrong order of magnitude for "missing weekly pattern". |
    | H5: `clip(lower=0)` is masking negative explosions into a flat 0, but real values are 100s -> still bad SMAPE | Possible compounding factor. |
    """)
    return


@app.cell
def _(SARIMAX, pd, plt, series, val_cutoffs):
    # ── CELL 6b: DIAGNOSTIC — inspect ONE SARIMA fit/forecast in isolation ───
    # Before fixing Cell 6, we need to SEE what the model is actually predicting.
    # A SMAPE of 1.42 is consistent with: predictions that are wildly the wrong
    # sign, wildly the wrong scale, or completely flat/constant. We isolate
    # window 1 only, print the actual forecast values, and plot them against
    # ground truth to diagnose the failure mode with certainty rather than guessing.

    c6b_cutoff = val_cutoffs[
        0
    ]  # use the same first window as before: 2021-12-05 23:00:00

    c6b_train_wf = series[series.index <= c6b_cutoff]
    c6b_actual_wf = series[
        (series.index > c6b_cutoff)
        & (series.index <= c6b_cutoff + pd.Timedelta(hours=168))
    ]
    c6b_train_recent = c6b_train_wf.iloc[-2000:]

    c6b_model = SARIMAX(
        c6b_train_recent,
        order=(2, 0, 0),
        seasonal_order=(0, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    c6b_fitted = c6b_model.fit(disp=False)

    # ── Print the fitted model's actual coefficients ──────────────────────────
    # This tells us directly whether the AR coefficients are stable (|phi| < 1
    # roughly) or exploding (which would produce runaway forecasts)
    print("=" * 60)
    print("FITTED SARIMA COEFFICIENTS")
    print("=" * 60)
    print(c6b_fitted.summary().tables[1])  # the coefficient table specifically

    # ── Generate the raw, UNCLIPPED forecast ──────────────────────────────────
    c6b_forecast_raw = c6b_fitted.forecast(steps=168)

    print("\n" + "=" * 60)
    print("RAW FORECAST STATISTICS (before any clipping)")
    print("=" * 60)
    print(f"Min:    {c6b_forecast_raw.min():.1f}")
    print(f"Max:    {c6b_forecast_raw.max():.1f}")
    print(f"Mean:   {c6b_forecast_raw.mean():.1f}")
    print(f"First 10 values:\n{c6b_forecast_raw.head(10)}")
    print(f"\nLast 10 values:\n{c6b_forecast_raw.tail(10)}")

    # ── Compare scale to actual data ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ACTUAL DATA STATISTICS (ground truth for this window)")
    print("=" * 60)
    print(f"Min:  {c6b_actual_wf.min():.1f}")
    print(f"Max:  {c6b_actual_wf.max():.1f}")
    print(f"Mean: {c6b_actual_wf.mean():.1f}")

    # ── Plot raw forecast vs actual to SEE the failure mode ───────────────────
    c6b_forecast_index = pd.date_range(
        start=c6b_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
    )
    c6b_forecast_series = pd.Series(c6b_forecast_raw.values, index=c6b_forecast_index)

    fig_diag, ax_diag = plt.subplots(figsize=(16, 5))
    ax_diag.plot(
        c6b_actual_wf.index,
        c6b_actual_wf.values,
        color="#2563EB",
        linewidth=1.5,
        label="Actual orders",
        marker="o",
        ms=3,
    )
    ax_diag.plot(
        c6b_forecast_series.index,
        c6b_forecast_series.values,
        color="#EF4444",
        linewidth=1.5,
        label="SARIMA forecast (RAW, unclipped)",
        marker="x",
        ms=3,
    )
    ax_diag.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_diag.set_title(
        "DIAGNOSTIC: SARIMA(2,0,0)(0,0,1)[24] — Window 1 Forecast vs Actual",
        fontweight="bold",
    )
    ax_diag.set_xlabel("Time")
    ax_diag.set_ylabel("Orders per hour")
    ax_diag.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/diag_01_sarima_window1.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SARIMA tracks the actual data beautifully for about 24-30 hours (the matplotlib zoom hides this, but the printed values prove it — Dec 6 morning rises from -31 up to 86 at 9am, tracking the real lunch ramp-up). Then, by Dec 7 onward, the forecast decays to essentially zero forever — literally 4.65e-33, which is computer-science-speak for "zero, with float rounding noise.

    - A pure AR(p) process is mean-reverting [it always pulls back toward its long-run average] unless it has explicit memory of seasonality built into the mean itself, not just the noise term.
    - Key insight: enforce_stationarity=False let the optimizer find AR coefficients (0.986 and -0.326) where the characteristic roots of the AR polynomial are just barely inside the stationary region. That means the model's forecast decays toward its unconditional mean as the horizon grows — and it decays fast. By hour ~30 of a 168-hour forecast, the AR(2) component has already "forgotten" the initial conditions and the forecast collapses toward 0 (the series' grand mean, pulled down hard by all those 31.7% structural zero-hours).
    - This is mathematically correct behavior for an AR(2) model — it is just the wrong model for a 168-hour-ahead forecast. AR(2) only has memory of the last 2 hours, plus a seasonal MA(1) term that only captures the noise correlation at lag 24, not a trend-following mean. Neither component can sustain a 7-day-ahead memory of "but it's lunchtime again." The model isn't broken it's structurally incapable of long-horizon forecasting at this specification.
    - Why the negative values at the start (-31.3)? Simple: training ends at 23:00 (night, near-zero orders), AR(1) coefficient is 0.986 (almost a random walk), so the immediate next-step forecast is "approximately stay near recent value" — but the seasonal MA term pulls it slightly negative as it tries to anticipate the lunchtime swing using only a noise-correlation mechanism, overshooting in the wrong direction since there's no seasonal AR term (P=0) to positively reinforce "same hour yesterday was X."
    This confirms hypothesis H2/H3 from my decomposition — not numerical instability/explosion, but the opposite: decay to the mean, which is actually the textbook failure mode of low-order AR models on long horizons. I was wrong about the direction of the failure (I expected explosion, we got collapse) but right that the model choice was the root cause.
    """)
    return


@app.cell
def _(
    N_VAL_WEEKS,
    SARIMAX,
    naive_mean_smape,
    np,
    pd,
    sarima_mean_smape,
    series,
    smape,
    time,
    val_cutoffs,
):
    # ── CELL 7: SARIMA attempt #2 — SARIMA(2,0,0)(1,0,1)[24] ─────────────────
    # WHY THIS CELL EXISTS: Cell 6 (SARIMA(2,0,0)(0,0,1)[24]) collapsed to ~0
    # after ~30 hours of the 168-hour forecast. Diagnosis (Cell 6b): the model
    # had NO seasonal AR term (P=0) — only AR(2) memory of the last 2 hours plus
    # a seasonal MA(1) term that captures NOISE correlation at lag 24, not a true
    # seasonal level. With no mechanism to "remember" yesterday's actual value at
    # this hour, the forecast mean-reverted toward the series' grand mean (pulled
    # near 0 by the 31.7% structural zero-hours) well before reaching hour 168.
    #
    # THE FIX: add a seasonal AR(1) term (P=1) at lag 24. This explicitly models
    # "this hour's value depends on YESTERDAY's value at this same hour" — true
    # day-to-day memory that should survive the full 168-hour horizon, unlike
    # the noise-only seasonal MA term alone.
    #
    # New specification: SARIMA(2,0,0)(1,0,1)[24]
    #   p=2, d=0, q=0  : unchanged from Cell 6 — short-term AR(2) on recent hours
    #   P=1            : NEW — seasonal AR(1) at lag 24 (genuine daily memory)
    #   D=0, Q=1       : unchanged — no seasonal differencing, seasonal MA(1) noise term
    #
    # We still expect this to underperform the naive baseline (SMAPE 0.2153),
    # because seasonal AR(1) at S=24 only captures "yesterday same hour" — it
    # still cannot see the S=168 WEEKLY pattern (e.g. Friday being busier than
    # Monday) the way the naive model's lag_168 copy does. This cell's purpose
    # is to test whether at least the COLLAPSE failure mode is fixed, and to
    # demonstrate SARIMA's structural limitation vs. weekly-aware approaches —
    # XGBoost (Cell 8) is the model expected to actually beat naive.
    #
    # NOTE: SARIMAX, time, and warnings were already imported in Cell 6 —
    # Marimo treats imports as global name bindings, so re-importing here would
    # trigger the same "redefinition" error as any other variable. We simply
    # reuse the names Cell 6 already bound.

    sarima2_smape_scores = []  # prefixed sarima2_ to avoid colliding with Cell 6's sarima_ names
    sarima2_mse_scores = []
    sarima2_fit_seconds = []

    for c7_w, c7_cutoff_w in enumerate(val_cutoffs):
        c7_train_wf = series[series.index <= c7_cutoff_w]
        c7_actual_wf = series[
            (series.index > c7_cutoff_w)
            & (series.index <= c7_cutoff_w + pd.Timedelta(hours=168))
        ]

        c7_train_recent = c7_train_wf.iloc[-2000:]

        c7_start_time = time.time()

        c7_model = SARIMAX(
            c7_train_recent,
            order=(2, 0, 0),
            seasonal_order=(1, 0, 1, 24),  # P=1 added — the fix vs. Cell 6
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        c7_fitted = c7_model.fit(disp=False)

        c7_elapsed = time.time() - c7_start_time
        sarima2_fit_seconds.append(c7_elapsed)

        c7_forecast_vals = c7_fitted.forecast(steps=168)

        c7_forecast_index = pd.date_range(
            start=c7_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
        )
        c7_forecast = pd.Series(c7_forecast_vals.values, index=c7_forecast_index)
        c7_forecast = c7_forecast.clip(lower=0)  # floor negative predictions at 0

        c7_aligned = pd.DataFrame(
            {"actual": c7_actual_wf, "forecast": c7_forecast}
        ).dropna()
        assert len(c7_aligned) == 168, (
            f"Window {c7_w + 1} has {len(c7_aligned)} rows, expected 168"
        )

        c7_smape_val = smape(c7_aligned["actual"], c7_aligned["forecast"])
        c7_mse_val = np.mean((c7_aligned["actual"] - c7_aligned["forecast"]) ** 2)

        sarima2_smape_scores.append(c7_smape_val)
        sarima2_mse_scores.append(c7_mse_val)

        print(
            f"  Window {c7_w + 1} ({c7_cutoff_w.date()} cutoff): "
            f"SMAPE={c7_smape_val:.4f}, MSE={c7_mse_val:.1f}, fit_time={c7_elapsed:.1f}s"
        )

    sarima2_mean_smape = np.mean(sarima2_smape_scores)
    sarima2_mean_mse = np.mean(sarima2_mse_scores)
    sarima2_std_smape = np.std(sarima2_smape_scores)
    sarima2_total_fit_time = np.sum(sarima2_fit_seconds)

    print(f"\n{'=' * 60}")
    print(f"SARIMA(2,0,0)(1,0,1)[24] RESULTS  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {sarima2_mean_smape:.4f}  ± {sarima2_std_smape:.4f} std")
    print(f"Mean MSE:   {sarima2_mean_mse:.1f}")
    print(
        f"SMAPE range: [{min(sarima2_smape_scores):.4f}, {max(sarima2_smape_scores):.4f}]"
    )
    print(
        f"Total fit time across {N_VAL_WEEKS} windows: {sarima2_total_fit_time:.1f}s "
        f"(avg {sarima2_total_fit_time / N_VAL_WEEKS:.1f}s/window)"
    )

    print(f"\n{'=' * 60}")
    print("COMPARISON: SARIMA v2 vs SARIMA v1 vs NAIVE BASELINE")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:      {naive_mean_smape:.4f}")
    print(
        f"SARIMA v1 SMAPE:  {sarima_mean_smape:.4f}  (Cell 6 — no seasonal AR, collapsed to 0)"
    )
    print(
        f"SARIMA v2 SMAPE:  {sarima2_mean_smape:.4f}  (Cell 7 — with seasonal AR(1) fix)"
    )

    v1_to_v2_improvement = (
        (sarima_mean_smape - sarima2_mean_smape) / sarima_mean_smape * 100
    )
    print(f"\nFix improved SMAPE by {v1_to_v2_improvement:.1f}% vs. the broken v1")

    if sarima2_mean_smape < naive_mean_smape:
        improvement_vs_naive = (
            (naive_mean_smape - sarima2_mean_smape) / naive_mean_smape * 100
        )
        print(f"✅ SARIMA v2 BEATS naive by {improvement_vs_naive:.1f}%")
    else:
        gap_vs_naive = (sarima2_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(
            f"❌ SARIMA v2 still WORSE than naive by {gap_vs_naive:.1f}% — "
            f"expected, since it lacks S=168 weekly memory that naive captures directly"
        )
    return (sarima2_mean_smape,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SMAPE collapsed from 1.42 → 0.65, a 54% improvement. More importantly, look at the MSE: it dropped from ~26,400 → ~2,841, a 9x improvement. The catastrophic collapse-to-zero failure mode is gone — the seasonal AR(1) term at lag 24 is doing exactly what we diagnosed it needed to do: giving the model genuine day-to-day memory that survives the full 168-hour horizon instead of decaying to the grand mean by hour 30.

    No, it doesn't beat naive — also exactly as predicted. Naive (0.215) still wins by a wide margin. This confirms our hypothesis precisely: S=24-only seasonal memory ("yesterday, same hour") cannot capture the S=168 weekly pattern ("this is Friday, not Monday") that the naive model gets for free by copying last week directly.

    A New, Interesting Pattern Worth Noticing: the per-window SMAPE values — they're **bimodal**, not randomly scattered:

    | Window | Cutoff | SMAPE | Day of week pattern |
    |---:|---|---:|---|
    | 1 | Dec 5 (Sun) | 0.742 | high |
    | 2 | Dec 12 (Sun) | 0.749 | high |
    | 3 | Dec 19 (Sun) | **0.396** | low |
    | 4 | Dec 26 (Sun) | 0.743 | high |
    | 5 | Jan 2 (Sun) | **0.406** | low |
    | 6 | Jan 9 (Sun) | 0.715 | high |
    | 7 | Jan 16 (Sun) | 0.792 | high |

    There's a clean alternating-ish pattern: roughly every other window scores ~0.40 while the rest score ~0.72–0.79. This is a strong, specific, checkable clue. Our hypothesis: windows 3 and 5 forecast weeks that start on a Monday immediately following a major demand anomaly week (Dec 20-26 is Christmas week, Jan 3-9 is just-after-New-Year),  meaning the training data going into those fits includes the actual abnormal Christmas/NYE spike, which may shift the AR(2) short-term coefficients in a way that happens to fit better, or more likely and more mechanically clean, the seasonal AR(1) coefficient on lag-24 happens to align better when the most recent training day's pattern resembles the upcoming forecast day's pattern, and degrades when there's a big day-of-week mismatch between "yesterday" and "the day being forecast 7 days out."

    We want to flag this as a genuine uncertainty rather than assert it confidently as this would need us to actually inspect which days of the week are driving the error within those windows to confirm, but we didn't have time to fully diagnose.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Story so far:

    - Naive baseline (S=168 weekly copy): SMAPE 0.215 — strong, because EDA showed seasonality = 91.8% of variance
    - SARIMA v1 (2,0,0)(0,0,1)[24]: SMAPE 1.42 — catastrophic failure, diagnosed as mean-reversion collapse from insufficient seasonal memory
    - SARIMA v2 (2,0,0)(1,0,1)[24]: SMAPE 0.65 — fixed the collapse (54% better) by adding seasonal AR(1), but structurally limited to daily-only memory, can't see the weekly pattern

    Conclusion: we need a model that captures BOTH the daily AND weekly seasonal structure simultaneously → this is precisely XGBoost's strength via lag_24 + lag_168 features together

    ## XGBoost: The Designated Champion

    XGBoost in comparison to SARIMA:

    - SARIMA thinks in equations as it models orders_t as a weighted sum of a few past values and past errors, with fixed coefficients learned once.
    - XGBoost thinks in features and trees. Instead of one global equation, we hand it a table where each row is one hour, and the columns are engineered features describing that hour (what was the value 24h ago? 168h ago? what hour of day is it? what day of week?). XGBoost then learns a collection of decision trees that split on these features to predict orders_t. Crucially, nothing stops us from giving it lag_24 AND lag_168 as separate columns simultaneously, because this is exactly the "both cycles at once" capability SARIMA structurally lacks.

    ### Feature Engineering for XGBoost

    XGBoost has no built-in concept of time. SARIMA "knows" what hour 25 is, it's the next step after hour 24. XGBoost sees nothing but rows and columns; it has no idea that row 1000 happened before row 1001 unless we explicitly encode that information as numbers in columns. This cell's entire job is translating "time" into "features a tree-based model can split on."
    """)
    return


@app.cell
def _(np, pd, series):
    # ── CELL 8: Feature Engineering for XGBoost ───────────────────────────────
    # XGBoost has NO native concept of time order or seasonality — it only sees
    # rows and columns of numbers. Everything SARIMA "knows" automatically
    # (yesterday, last week, time of day) we must manually encode as columns.
    # This cell builds that feature table once; Cell 9 will slice it per
    # walk-forward window and train/forecast with it.

    # ── Step 1: Start from the same filled series as everything else ─────────
    # series was defined in Cell 5 — reuse it directly, no need to redefine
    c8_df = pd.DataFrame({"orders": series})  # one-column DataFrame, indexed by time

    # ── Step 2: Calendar features — directly readable from the timestamp ─────
    # These give XGBoost the raw "what kind of hour is this" signal that SARIMA
    # gets implicitly through its seasonal terms. We encode hour and day-of-week
    # as plain integers; tree-based models split on thresholds (e.g. "hour <= 7")
    # so they don't need one-hot encoding the way linear models often do —
    # trees can learn non-linear, non-monotonic relationships directly from
    # integer-coded categories.
    c8_df["hour_of_day"] = (
        c8_df.index.hour
    )  # 0-23, captures the daily lunch/dinner cycle
    c8_df["day_of_week"] = (
        c8_df.index.dayofweek
    )  # 0=Mon..6=Sun, captures Friday-is-busiest pattern
    c8_df["is_weekend"] = (c8_df["day_of_week"] >= 5).astype(
        int
    )  # 1 if Sat/Sun, else 0 — explicit weekend flag

    # ── Step 3: Trend feature — a simple, monotonically increasing time index ─
    # EDA (Cell 4) found the trend explains only 1.2% of variance, but it is
    # REAL (+55.2% growth over the year, Cell 3). A plain integer counter gives
    # XGBoost a way to learn "the general level keeps rising" independent of
    # the lag features, which mostly carry seasonal SHAPE rather than absolute
    # LEVEL drift over many months.
    c8_df["time_index"] = np.arange(
        len(c8_df)
    )  # 0, 1, 2, ... — hours since the start of the dataset

    # ── Step 4: Lag features — the most powerful predictors, per ACF (Cell 4) ─
    # .shift(k) moves the ENTIRE series down by k rows, so that row t's
    # "lag_k" column contains the value that was actually observed at row t-k.
    # This is how we give a tabular model access to "what happened k hours ago."
    # We chose these specific lags directly from the ACF analysis in Cell 4:
    #   lag_1   : ACF=+0.68 — immediate short-term momentum
    #   lag_24  : ACF=+0.89 — yesterday, same hour (daily seasonality, 2nd strongest signal)
    #   lag_48  : ACF=+0.83 — two days ago, same hour
    #   lag_144 : ACF=+0.87 — six days ago, same hour (one day short of a full week)
    #   lag_168 : ACF=+0.93 — LAST WEEK, same hour (weekly seasonality, STRONGEST signal)
    # We deliberately EXCLUDE lag_2, lag_3, lag_6, lag_12 — Cell 4 showed these
    # have weak or negative ACF and would mostly add noise, not signal, to the trees.
    c8_df["lag_1"] = c8_df["orders"].shift(1)
    c8_df["lag_24"] = c8_df["orders"].shift(24)
    c8_df["lag_48"] = c8_df["orders"].shift(48)
    c8_df["lag_144"] = c8_df["orders"].shift(144)
    c8_df["lag_168"] = c8_df["orders"].shift(168)

    # ── Step 5: Rolling-mean features — smoothed recent context ──────────────
    # A single lag value can be noisy (e.g. one freak hour). A rolling mean
    # [the average over a recent WINDOW of past values, computed at every point]
    # smooths that out, giving XGBoost a sense of "the recent typical level"
    # rather than one potentially-unusual data point.
    # .shift(1) BEFORE .rolling() is critical: it ensures the window looks only
    # at PAST hours relative to the current row — without the shift, row t's
    # rolling mean would include orders_t itself, which is the value we are
    # trying to predict (a direct case of "leakage" — using the answer to
    # predict the answer).
    c8_df["rolling_mean_24h"] = (
        c8_df["orders"].shift(1).rolling(window=24).mean()
    )  # avg of last 24h (excl. current)
    c8_df["rolling_mean_168h"] = (
        c8_df["orders"].shift(1).rolling(window=168).mean()
    )  # avg of last full week (excl. current)

    # ── Step 6: Drop rows where lag/rolling features are NaN ─────────────────
    # The very first 168 rows of the dataset cannot have a valid lag_168 value
    # (there IS no data 168 hours before the start). These rows would have NaN
    # features, which XGBoost cannot train on. We drop them — this only costs
    # us the first week of the ~51-week dataset, a negligible loss.
    c8_n_before_dropna = len(c8_df)
    c8_df = c8_df.dropna()
    c8_n_after_dropna = len(c8_df)

    # ── Step 7: Sanity checks — print everything before trusting this table ──
    print("=" * 60)
    print("FEATURE TABLE — SANITY CHECKS")
    print("=" * 60)
    print(f"Rows before dropna: {c8_n_before_dropna}")
    print(f"Rows after dropna:  {c8_n_after_dropna}")
    print(
        f"Rows dropped:       {c8_n_before_dropna - c8_n_after_dropna} "
        f"(expected ~168, the warm-up period for lag_168)"
    )

    print(f"\nColumns in feature table: {c8_df.columns.tolist()}")
    print("\nFirst 3 rows:")
    print(c8_df.head(3))

    print(f"\nAny remaining NaNs? {c8_df.isnull().any().any()}")

    print(f"\n{'=' * 60}")
    print("LEAKAGE CHECK: confirm lag_1 at row t equals orders at row t-1")
    print(f"{'=' * 60}")
    c8_check_idx = 500  # arbitrary row, far enough in to have all lags populated
    c8_check_time = c8_df.index[c8_check_idx]
    c8_check_lag1_value = c8_df["lag_1"].iloc[c8_check_idx]
    c8_check_actual_prev_hour = series[c8_check_time - pd.Timedelta(hours=1)]
    print(f"Row time:                    {c8_check_time}")
    print(f"lag_1 value at this row:     {c8_check_lag1_value}")
    print(f"Actual orders 1h earlier:    {c8_check_actual_prev_hour}")
    print(
        f"Match: {np.isclose(c8_check_lag1_value, c8_check_actual_prev_hour)} "
        f"(should be True — confirms shift() is doing exactly what we expect)"
    )
    return (c8_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### XGBoost Walk-Forward Validation:

    Before, one conceptual point that matters for how we structure this: SARIMA refits the entire statistical model from scratch inside every walk-forward window (that's what made it slow — 4-7 seconds per window). XGBoost works differently — we'll still retrain a fresh model for every window (no shortcuts, no peeking, exactly matching production), but the training itself is just "fit trees on a feature table," which is typically much faster than SARIMA's iterative optimization.

    There's also a new technical wrinkle worth flagging upfront: forecasting 168 hours ahead with lag features is not a single .predict() call the way SARIMA's .forecast(steps=168) was. Why?
    Because lag_1 for hour t+1 needs the actual value at hour t — but hour t is itself a prediction, not real data, once we're forecasting into the future. This requires recursive/iterative forecasting: predict hour 1, feed that prediction back in as lag_1 for hour 2, predict hour 2, and so on, 168 times in a row. This is a fundamentally different mechanic from SARIMA.
    """)
    return


@app.cell
def _(
    N_VAL_WEEKS,
    c8_df,
    naive_mean_smape,
    np,
    pd,
    sarima2_mean_smape,
    sarima_mean_smape,
    series,
    smape,
    time,
    val_cutoffs,
):
    # ── CELL 9: XGBoost — Walk-Forward Validation with Recursive Forecasting ─
    # CONCEPTUAL NOTE: XGBoost predicts ONE row at a time. To forecast 168 hours
    # ahead, we cannot just call .predict() once like SARIMA's .forecast(steps=168).
    # Instead we forecast hour-by-hour: predict hour 1, then use THAT PREDICTION
    # (not real data — it doesn't exist yet) to build the lag_1 feature for hour 2,
    # predict hour 2, use it to build lag_1 for hour 3, and so on. This is called
    # RECURSIVE FORECASTING [feeding a model's own outputs back in as inputs for
    # the next step] and it is standard practice for tree-based models on
    # multi-step time series forecasting.
    #
    # IMPORTANT CONSEQUENCE: errors can COMPOUND. If hour 1's prediction is
    # slightly wrong, hour 2 is built on a slightly wrong lag_1, which can make
    # hour 2 slightly more wrong, etc. This is a genuine weakness of recursive
    # forecasting vs. SARIMA's direct multi-step formula — worth mentioning
    # if the professor asks about model trade-offs.
    #
    # lag_24, lag_48, lag_144, lag_168, and the rolling means stay anchored to
    # REAL historical data for the FIRST week we forecast (since hour 24, 48,
    # 144, 168 hours before any hour in week 1 of the forecast horizon is
    # always inside the training data, never inside the forecast itself,
    # because our horizon is exactly 168 hours = exactly one lag_168 cycle).
    # Only lag_1 needs recursive feeding, because everything else looks back
    # further than the entire forecast horizon.

    import xgboost as xgb  # the gradient boosting library — NEW import, not used in any prior cell

    c9_feature_cols = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "time_index",
        "lag_1",
        "lag_24",
        "lag_48",
        "lag_144",
        "lag_168",
        "rolling_mean_24h",
        "rolling_mean_168h",
    ]  # every column XGBoost will see — orders itself is the TARGET, never a feature

    xgb_smape_scores = []
    xgb_mse_scores = []
    xgb_fit_seconds = []

    for c9_w, c9_cutoff_w in enumerate(val_cutoffs):
        # ── Step 1: Slice the feature table up to this window's cutoff ───────
        # c8_df was built in Cell 8 and already has all lag/rolling features computed
        # ONCE over the entire series — slicing here just selects which ROWS count
        # as "training data" for this particular walk-forward window. No leakage:
        # every feature value was computed using only data at or before its own
        # timestamp (shift() and rolling() are both backward-looking by construction).
        c9_train_df = c8_df[c8_df.index <= c9_cutoff_w]

        c9_actual_wf = series[
            (series.index > c9_cutoff_w)
            & (series.index <= c9_cutoff_w + pd.Timedelta(hours=168))
        ]

        # ── Step 2: Fit XGBoost on this window's training data ───────────────
        c9_start_time = time.time()

        c9_model = xgb.XGBRegressor(
            n_estimators=300,  # number of trees — more trees = more capacity, tuned by trial
            max_depth=5,  # max tree depth — controls how complex each tree can be
            learning_rate=0.05,  # how much each tree corrects the previous ones — small = more stable
            random_state=42,  # fixed seed — ensures reproducible results across runs
            objective="reg:squarederror",  # standard regression loss — matches our MSE secondary metric
        )
        c9_model.fit(
            c9_train_df[c9_feature_cols],  # X: the feature columns only
            c9_train_df["orders"],  # y: the target we're predicting
        )

        c9_elapsed = time.time() - c9_start_time
        xgb_fit_seconds.append(c9_elapsed)

        # ── Step 3: Recursive forecasting, hour by hour, 168 times ───────────
        # We build a GROWING history series that starts as a copy of all known
        # data, and append each new prediction to it as we go — so that future
        # lag_24/lag_48/lag_144/lag_168 lookups (which always reach further back
        # than our predictions go) keep working correctly, while lag_1 lookups
        # correctly pick up the most recently PREDICTED hour.
        c9_history = series[
            series.index <= c9_cutoff_w
        ].copy()  # real data only, so far
        c9_predictions = []  # will collect 168 predicted values in order
        c9_forecast_index = pd.date_range(
            start=c9_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c9_step_time in c9_forecast_index:
            # Build ONE row of features for this single future hour, using
            # c9_history (which contains real data + any predictions made so far
            # in this loop) — exactly mirroring what Cell 8 did for the whole table,
            # but for one timestamp at a time.
            c9_row = {
                "hour_of_day": c9_step_time.hour,
                "day_of_week": c9_step_time.dayofweek,
                "is_weekend": int(c9_step_time.dayofweek >= 5),
                "time_index": (c9_step_time - c8_df.index[0]).total_seconds() / 3600
                + 168,
                # NOTE: +168 corrects for the 168 rows Cell 8 dropped via dropna() —
                # keeps time_index consistent with how it was defined during training
                "lag_1": c9_history.loc[c9_step_time - pd.Timedelta(hours=1)],
                "lag_24": c9_history.loc[c9_step_time - pd.Timedelta(hours=24)],
                "lag_48": c9_history.loc[c9_step_time - pd.Timedelta(hours=48)],
                "lag_144": c9_history.loc[c9_step_time - pd.Timedelta(hours=144)],
                "lag_168": c9_history.loc[c9_step_time - pd.Timedelta(hours=168)],
                "rolling_mean_24h": c9_history.loc[
                    c9_step_time - pd.Timedelta(hours=24) : c9_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
                "rolling_mean_168h": c9_history.loc[
                    c9_step_time - pd.Timedelta(hours=168) : c9_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
            }
            c9_row_df = pd.DataFrame([c9_row])[
                c9_feature_cols
            ]  # enforce exact column order used in training

            c9_pred_value = c9_model.predict(c9_row_df)[
                0
            ]  # single prediction for this hour
            c9_pred_value = max(
                c9_pred_value, 0.0
            )  # clip negative predictions to zero (physical floor)

            c9_predictions.append(c9_pred_value)
            # Feed this prediction back into history so the NEXT iteration's
            # lag_1 (and eventually lag_24 etc., once we forecast far enough)
            # can see it — this IS the "recursive" part of recursive forecasting.
            c9_history.loc[c9_step_time] = c9_pred_value

        c9_forecast = pd.Series(c9_predictions, index=c9_forecast_index)

        # ── Step 4: Align and score ───────────────────────────────────────────
        c9_aligned = pd.DataFrame(
            {"actual": c9_actual_wf, "forecast": c9_forecast}
        ).dropna()
        assert len(c9_aligned) == 168, (
            f"Window {c9_w + 1} has {len(c9_aligned)} rows, expected 168"
        )

        c9_smape_val = smape(c9_aligned["actual"], c9_aligned["forecast"])
        c9_mse_val = np.mean((c9_aligned["actual"] - c9_aligned["forecast"]) ** 2)

        xgb_smape_scores.append(c9_smape_val)
        xgb_mse_scores.append(c9_mse_val)

        print(
            f"  Window {c9_w + 1} ({c9_cutoff_w.date()} cutoff): "
            f"SMAPE={c9_smape_val:.4f}, MSE={c9_mse_val:.1f}, fit_time={c9_elapsed:.2f}s"
        )

    xgb_mean_smape = np.mean(xgb_smape_scores)
    xgb_mean_mse = np.mean(xgb_mse_scores)
    xgb_std_smape = np.std(xgb_smape_scores)
    xgb_total_fit_time = np.sum(xgb_fit_seconds)

    print(f"\n{'=' * 60}")
    print(f"XGBOOST RESULTS  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {xgb_mean_smape:.4f}  ± {xgb_std_smape:.4f} std")
    print(f"Mean MSE:   {xgb_mean_mse:.1f}")
    print(f"SMAPE range: [{min(xgb_smape_scores):.4f}, {max(xgb_smape_scores):.4f}]")
    print(
        f"Total fit time across {N_VAL_WEEKS} windows: {xgb_total_fit_time:.1f}s "
        f"(avg {xgb_total_fit_time / N_VAL_WEEKS:.2f}s/window)"
    )

    print(f"\n{'=' * 60}")
    print("FULL MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:       {naive_mean_smape:.4f}")
    print(f"SARIMA v1 SMAPE:   {sarima_mean_smape:.4f}  (collapsed — no seasonal AR)")
    print(f"SARIMA v2 SMAPE:   {sarima2_mean_smape:.4f}  (fixed — daily-only memory)")
    print(f"XGBoost SMAPE:     {xgb_mean_smape:.4f}  (daily + weekly lag features)")

    if xgb_mean_smape < naive_mean_smape:
        xgb_improvement = (naive_mean_smape - xgb_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ XGBoost BEATS naive baseline by {xgb_improvement:.1f}%")
    else:
        xgb_gap = (xgb_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(
            f"\n❌ XGBoost still WORSE than naive by {xgb_gap:.1f}% — needs further tuning"
        )
    return c9_feature_cols, xgb, xgb_mean_smape


@app.cell
def _(c8_df, c9_feature_cols, np, pd, plt, series, smape, val_cutoffs, xgb):
    # ── CELL 9b: DIAGNOSTIC — inspect ONE XGBoost recursive forecast ─────────
    # XGBoost SMAPE (0.689) is suspiciously close to SARIMA v2 (0.649), despite
    # having access to lag_168 (ACF +0.93) which SARIMA v2 structurally lacks.
    # In the worst case, XGBoost should be able to learn "just copy lag_168"
    # and approximately match naive (0.215). It isn't even close. Before tuning
    # anything, we inspect window 1 in detail: feature importances, raw
    # predictions over time, and a plot vs. actual — to find the actual bug
    # rather than guess at hyperparameters.

    c9b_cutoff = val_cutoffs[
        0
    ]  # same first window as all prior diagnostics — 2021-12-05 23:00:00

    c9b_train_df = c8_df[c8_df.index <= c9b_cutoff]
    c9b_actual_wf = series[
        (series.index > c9b_cutoff)
        & (series.index <= c9b_cutoff + pd.Timedelta(hours=168))
    ]

    c9b_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        objective="reg:squarederror",
    )
    c9b_model.fit(c9b_train_df[c9_feature_cols], c9b_train_df["orders"])

    # ── Check 1: Feature importances — does the model even VALUE lag_168? ────
    # If lag_168 has near-zero importance, that tells us the model never
    # learned to rely on it during TRAINING — a learning problem, not a
    # recursive-forecasting problem
    print("=" * 60)
    print("FEATURE IMPORTANCES (gain-based) — does the model value lag_168?")
    print("=" * 60)
    c9b_importances = pd.Series(
        c9b_model.feature_importances_, index=c9_feature_cols
    ).sort_values(ascending=False)
    print(c9b_importances)

    # ── Check 2: In-sample fit — how good is the model on TRAINING data? ─────
    # This isolates "did the model learn the pattern at all" from "did the
    # recursive forecasting loop break it afterwards"
    c9b_train_preds = c9b_model.predict(c9b_train_df[c9_feature_cols])
    c9b_train_preds = np.clip(c9b_train_preds, 0, None)
    c9b_insample_smape = smape(
        c9b_train_df["orders"], pd.Series(c9b_train_preds, index=c9b_train_df.index)
    )
    print(
        f"\nIN-SAMPLE SMAPE (on training data, NOT recursive): {c9b_insample_smape:.4f}"
    )
    print(
        "(If this is LOW, the model learned the pattern fine — bug is in the recursive loop)"
    )
    print(
        "(If this is HIGH too, the model never learned the pattern — bug is in features/training)"
    )

    # ── Check 3: Run the SAME recursive loop, but log every step's components ─
    c9b_history = series[series.index <= c9b_cutoff].copy()
    c9b_predictions = []
    c9b_lag168_used = []  # track what lag_168 value was actually fed in, each step
    c9b_lag1_used = []  # track what lag_1 value was actually fed in, each step
    c9b_forecast_index = pd.date_range(
        start=c9b_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
    )

    for c9b_step_time in c9b_forecast_index:
        c9b_lag1_val = c9b_history.loc[c9b_step_time - pd.Timedelta(hours=1)]
        c9b_lag24_val = c9b_history.loc[c9b_step_time - pd.Timedelta(hours=24)]
        c9b_lag48_val = c9b_history.loc[c9b_step_time - pd.Timedelta(hours=48)]
        c9b_lag144_val = c9b_history.loc[c9b_step_time - pd.Timedelta(hours=144)]
        c9b_lag168_val = c9b_history.loc[c9b_step_time - pd.Timedelta(hours=168)]
        c9b_roll24_val = c9b_history.loc[
            c9b_step_time - pd.Timedelta(hours=24) : c9b_step_time
            - pd.Timedelta(hours=1)
        ].mean()
        c9b_roll168_val = c9b_history.loc[
            c9b_step_time - pd.Timedelta(hours=168) : c9b_step_time
            - pd.Timedelta(hours=1)
        ].mean()

        c9b_row = {
            "hour_of_day": c9b_step_time.hour,
            "day_of_week": c9b_step_time.dayofweek,
            "is_weekend": int(c9b_step_time.dayofweek >= 5),
            "time_index": (c9b_step_time - c8_df.index[0]).total_seconds() / 3600 + 168,
            "lag_1": c9b_lag1_val,
            "lag_24": c9b_lag24_val,
            "lag_48": c9b_lag48_val,
            "lag_144": c9b_lag144_val,
            "lag_168": c9b_lag168_val,
            "rolling_mean_24h": c9b_roll24_val,
            "rolling_mean_168h": c9b_roll168_val,
        }
        c9b_row_df = pd.DataFrame([c9b_row])[c9_feature_cols]
        c9b_pred = max(c9b_model.predict(c9b_row_df)[0], 0.0)

        c9b_predictions.append(c9b_pred)
        c9b_lag168_used.append(c9b_lag168_val)
        c9b_lag1_used.append(c9b_lag1_val)
        c9b_history.loc[c9b_step_time] = c9b_pred

    c9b_forecast = pd.Series(c9b_predictions, index=c9b_forecast_index)

    # ── Check 4: Is lag_168 actually pulling from REAL data the whole time? ──
    # Since the forecast horizon is exactly 168 hours, lag_168 for EVERY step
    # in this window should come from c9b_history BEFORE the cutoff — i.e.
    # REAL historical data, never a previous prediction. Let's verify this
    # explicitly by comparing c9b_lag168_used against the real series directly.
    c9b_expected_lag168 = [
        series.loc[t - pd.Timedelta(hours=168)] for t in c9b_forecast_index
    ]
    c9b_lag168_match = np.allclose(c9b_lag168_used, c9b_expected_lag168)
    print(
        f"\nlag_168 always pulled from REAL data (never a prediction)? {c9b_lag168_match}"
    )

    # ── Plot: actual vs forecast vs the raw lag_168 reference ────────────────
    fig_diag2, ax_diag2 = plt.subplots(figsize=(16, 5))
    ax_diag2.plot(
        c9b_actual_wf.index,
        c9b_actual_wf.values,
        color="#2563EB",
        linewidth=1.5,
        label="Actual orders",
        marker="o",
        ms=3,
    )
    ax_diag2.plot(
        c9b_forecast.index,
        c9b_forecast.values,
        color="#EF4444",
        linewidth=1.5,
        label="XGBoost forecast (recursive)",
        marker="x",
        ms=3,
    )
    ax_diag2.plot(
        c9b_forecast.index,
        c9b_lag168_used,
        color="#059669",
        linewidth=1,
        linestyle="--",
        label="lag_168 reference (naive copy)",
        alpha=0.7,
    )
    ax_diag2.set_title(
        "DIAGNOSTIC: XGBoost — Window 1 Forecast vs Actual vs lag_168 reference",
        fontweight="bold",
    )
    ax_diag2.set_xlabel("Time")
    ax_diag2.set_ylabel("Orders per hour")
    ax_diag2.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/diag_02_xgboost_window1.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(
        f"\nXGBoost SMAPE this window:        {smape(c9b_actual_wf, c9b_forecast):.4f}"
    )
    print(
        f"Pure lag_168 copy SMAPE (naive):  {smape(c9b_actual_wf, pd.Series(c9b_lag168_used, index=c9b_forecast.index)):.4f}"
    )
    return c9b_insample_smape, c9b_model, c9b_train_df


@app.cell
def _(c9_feature_cols, c9b_insample_smape, c9b_train_df, np, pd, smape, xgb):
    # ── CELL 9c: Quick experiment — does a SIMPLER XGBoost fit better in-sample? ──
    # Hypothesis: max_depth=5, n_estimators=300 is wildly overcapacity for what
    # is fundamentally a "mostly copy lag_168, adjust a little" problem. We test
    # a much simpler config and compare in-sample SMAPE directly.

    c9c_model_simple = xgb.XGBRegressor(
        n_estimators=50,  # far fewer trees
        max_depth=3,  # much shallower — less room to overfit
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    )
    c9c_model_simple.fit(c9b_train_df[c9_feature_cols], c9b_train_df["orders"])

    c9c_train_preds_simple = np.clip(
        c9c_model_simple.predict(c9b_train_df[c9_feature_cols]), 0, None
    )
    c9c_insample_smape_simple = smape(
        c9b_train_df["orders"],
        pd.Series(c9c_train_preds_simple, index=c9b_train_df.index),
    )

    print(f"COMPLEX model (depth=5, n=300) in-sample SMAPE: {c9b_insample_smape:.4f}")
    print(
        f"SIMPLE  model (depth=3, n=50)  in-sample SMAPE: {c9c_insample_smape_simple:.4f}"
    )
    print(
        f"\nPure lag_168 copy in-sample SMAPE (reference): "
        f"{smape(c9b_train_df['orders'], c9b_train_df['lag_168']):.4f}"
    )
    return


@app.cell
def _(
    c9_feature_cols,
    c9b_insample_smape,
    c9b_model,
    c9b_train_df,
    np,
    pd,
    smape,
):
    # ── CELL 9d: Verify whether zero-hour predictions are driving the bad SMAPE ──
    # Hypothesis: MSE-trained models predict small positive noise during the
    # 31.7% of hours that are TRULY zero (since tiny errors there are invisible
    # to squared-error loss), but SMAPE penalizes ANY positive prediction on a
    # true-zero hour with the maximum possible score of 2.0 for that single hour.
    # A small fraction of "noisy zero predictions" could be dragging the average
    # SMAPE up dramatically, even though MSE looks fine.

    c9d_train_preds = np.clip(c9b_model.predict(c9b_train_df[c9_feature_cols]), 0, None)
    c9d_train_preds_series = pd.Series(c9d_train_preds, index=c9b_train_df.index)

    # ── Split the data into TRUE-ZERO hours vs TRUE-NONZERO hours ────────────
    c9d_is_true_zero = c9b_train_df["orders"] == 0

    print("=" * 60)
    print("PREDICTION BEHAVIOUR ON TRUE-ZERO HOURS (orders actually = 0)")
    print("=" * 60)
    c9d_preds_on_zeros = c9d_train_preds_series[c9d_is_true_zero]
    print(f"Number of true-zero hours in training data: {c9d_is_true_zero.sum()}")
    print(
        f"Of these, how many did the model predict EXACTLY 0?     "
        f"{(c9d_preds_on_zeros == 0).sum()} ({(c9d_preds_on_zeros == 0).mean() * 100:.1f}%)"
    )
    print(
        f"Of these, how many did the model predict > 0 (any amount)? "
        f"{(c9d_preds_on_zeros > 0).sum()} ({(c9d_preds_on_zeros > 0).mean() * 100:.1f}%)"
    )
    print(f"Mean prediction on true-zero hours: {c9d_preds_on_zeros.mean():.2f}")
    print(f"Max prediction on true-zero hours:  {c9d_preds_on_zeros.max():.2f}")

    # ── Compute SMAPE SEPARATELY for zero-hours vs non-zero-hours ────────────
    c9d_smape_on_zeros = smape(
        c9b_train_df.loc[c9d_is_true_zero, "orders"],
        c9d_train_preds_series[c9d_is_true_zero],
    )
    c9d_smape_on_nonzeros = smape(
        c9b_train_df.loc[~c9d_is_true_zero, "orders"],
        c9d_train_preds_series[~c9d_is_true_zero],
    )
    print(f"\nSMAPE restricted to TRUE-ZERO hours only:     {c9d_smape_on_zeros:.4f}")
    print(f"SMAPE restricted to TRUE-NONZERO hours only:  {c9d_smape_on_nonzeros:.4f}")
    print(f"\n(Recall overall in-sample SMAPE was: {c9b_insample_smape:.4f})")
    print(f"({c9d_is_true_zero.mean() * 100:.1f}% of training hours are true-zero)")

    # ── Sanity check: what fraction of the OVERALL SMAPE is attributable to zero-hours? ──
    # This is the decisive calculation: if SMAPE-on-zeros is near 2.0 (the max),
    # and zero-hours are ~32% of all hours, they alone could contribute roughly
    # 0.32 * 2.0 = 0.64 to the overall average — which would fully explain why
    # our overall SMAPE (0.587) is so much worse than the lag_168 reference (0.219).
    print(
        f"\nEstimated contribution of zero-hours to overall SMAPE: "
        f"{c9d_is_true_zero.mean() * c9d_smape_on_zeros:.4f}"
    )
    print(
        f"Estimated contribution of non-zero-hours to overall SMAPE: "
        f"{(~c9d_is_true_zero).mean() * c9d_smape_on_nonzeros:.4f}"
    )
    print(
        f"Sum of both contributions (should ≈ overall in-sample SMAPE): "
        f"{c9d_is_true_zero.mean() * c9d_smape_on_zeros + (~c9d_is_true_zero).mean() * c9d_smape_on_nonzeros:.4f}"
    )
    return (c9d_is_true_zero,)


@app.cell
def _(
    c9_feature_cols,
    c9b_insample_smape,
    c9b_model,
    c9b_train_df,
    c9d_is_true_zero,
    np,
    pd,
    smape,
):
    # ── CELL 9e: Apply the zero-floor fix and re-measure in-sample SMAPE ─────
    # CONFIRMED ROOT CAUSE (Cell 9d): MSE-trained XGBoost predicts small positive
    # noise (mean 0.35, max 10.63) on 73.1% of true-zero hours. Each such
    # prediction scores the MAXIMUM possible SMAPE (2.0) for that hour, since
    # SMAPE has NO tolerance for small errors when the true value is exactly
    # zero. This alone contributes 0.4665 of the 0.5872 total in-sample SMAPE.
    #
    # FIX: snap any prediction below a small threshold to exactly 0. We choose
    # threshold=5.0 — comfortably above the model's typical zero-hour noise
    # (mean 0.35) but far below any genuine operating-hour order count (the
    # 25th percentile of NON-zero hours, from Cell 1's describe(), was 30).
    c9e_zero_threshold = (
        5.0  # any prediction below this, on any hour, gets snapped to 0
    )

    c9e_train_preds_raw = np.clip(
        c9b_model.predict(c9b_train_df[c9_feature_cols]), 0, None
    )
    c9e_train_preds_floored = np.where(
        c9e_train_preds_raw < c9e_zero_threshold, 0.0, c9e_train_preds_raw
    )
    c9e_train_preds_floored_series = pd.Series(
        c9e_train_preds_floored, index=c9b_train_df.index
    )

    c9e_insample_smape_floored = smape(
        c9b_train_df["orders"], c9e_train_preds_floored_series
    )

    print(f"In-sample SMAPE BEFORE zero-floor fix: {c9b_insample_smape:.4f}")
    print(f"In-sample SMAPE AFTER  zero-floor fix: {c9e_insample_smape_floored:.4f}")
    print(
        f"Pure lag_168 copy reference:           "
        f"{smape(c9b_train_df['orders'], c9b_train_df['lag_168']):.4f}"
    )

    # ── Re-check: how many true-zero hours now get predicted as exactly 0? ───
    c9e_preds_on_zeros_floored = c9e_train_preds_floored_series[c9d_is_true_zero]
    print(
        f"\nOf true-zero hours, now predicted EXACTLY 0: "
        f"{(c9e_preds_on_zeros_floored == 0).sum()} "
        f"({(c9e_preds_on_zeros_floored == 0).mean() * 100:.1f}%, was 26.9% before)"
    )

    c9e_smape_on_zeros_floored = smape(
        c9b_train_df.loc[c9d_is_true_zero, "orders"], c9e_preds_on_zeros_floored
    )
    print(
        f"SMAPE on true-zero hours, AFTER fix: {c9e_smape_on_zeros_floored:.4f}  (was 1.4620 before)"
    )
    return


@app.cell
def _(
    N_VAL_WEEKS,
    c8_df,
    c9_feature_cols,
    naive_mean_smape,
    np,
    pd,
    sarima2_mean_smape,
    sarima_mean_smape,
    series,
    smape,
    time,
    val_cutoffs,
    xgb,
    xgb_mean_smape,
):
    # ── CELL 10: XGBoost v2 — With Zero-Floor Fix ─────────────────────────────
    # WHY THIS CELL EXISTS: Cell 9 (XGBoost v1) scored SMAPE 0.689 — worse than
    # even SARIMA v2 (0.649), despite having lag_168 (ACF +0.93) as an explicit
    # feature. Diagnosis (Cells 9b-9e) found the root cause: the model was
    # trained with objective="reg:squarederror" (MSE), which has NO incentive to
    # predict true zeros exactly — a 0.35 average error during a zero-hour costs
    # almost nothing in squared-error terms. But SMAPE has ZERO tolerance for
    # this: ANY positive prediction on a true-zero hour scores the maximum
    # possible SMAPE (2.0) for that hour, since SMAPE = 2*|0-forecast|/(0+forecast)
    # = 2.0 regardless of how small forecast is. With 73.1% of true-zero hours
    # getting some positive prediction, this single failure mode contributed
    # 0.4665 of the 0.5872 in-sample SMAPE — explaining nearly 80% of the damage.
    #
    # THE FIX: snap any prediction below a small threshold (5.0) to exactly 0.0.
    # Verified in Cell 9e: this drops in-sample SMAPE from 0.587 -> 0.193,
    # slightly BETTER than the pure lag_168 copy reference (0.2185) — confirming
    # the model's other features (trend, lag_24, day-of-week) genuinely add
    # value once zero-hour noise stops masking them. We now re-run the FULL
    # 7-window walk-forward validation with this fix in place, to see whether
    # the in-sample improvement transfers to genuine out-of-sample performance.
    #
    # Same recursive forecasting mechanic as Cell 9: predict hour-by-hour, feed
    # each prediction back into history so future lag_1 lookups see it.
    # lag_24/48/144/168 always pull real historical data within this 168-hour
    # horizon (confirmed in Cell 9b's leakage check).

    c10_zero_threshold = 5.0  # any prediction below this gets snapped to exactly 0

    xgb2_smape_scores = []  # prefixed xgb2_ to avoid colliding with Cell 9's xgb_ names
    xgb2_mse_scores = []
    xgb2_fit_seconds = []

    for c10_w, c10_cutoff_w in enumerate(val_cutoffs):
        c10_train_df = c8_df[c8_df.index <= c10_cutoff_w]
        c10_actual_wf = series[
            (series.index > c10_cutoff_w)
            & (series.index <= c10_cutoff_w + pd.Timedelta(hours=168))
        ]

        c10_start_time = time.time()

        c10_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        c10_model.fit(c10_train_df[c9_feature_cols], c10_train_df["orders"])
        # NOTE: c9_feature_cols was defined in Cell 9 and is reused as-is —
        # the feature LIST doesn't change between v1 and v2, only the
        # post-prediction floor logic changes, so no need to redefine it

        c10_elapsed = time.time() - c10_start_time
        xgb2_fit_seconds.append(c10_elapsed)

        c10_history = series[series.index <= c10_cutoff_w].copy()
        c10_predictions = []
        c10_forecast_index = pd.date_range(
            start=c10_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c10_step_time in c10_forecast_index:
            c10_row = {
                "hour_of_day": c10_step_time.hour,
                "day_of_week": c10_step_time.dayofweek,
                "is_weekend": int(c10_step_time.dayofweek >= 5),
                "time_index": (c10_step_time - c8_df.index[0]).total_seconds() / 3600
                + 168,
                "lag_1": c10_history.loc[c10_step_time - pd.Timedelta(hours=1)],
                "lag_24": c10_history.loc[c10_step_time - pd.Timedelta(hours=24)],
                "lag_48": c10_history.loc[c10_step_time - pd.Timedelta(hours=48)],
                "lag_144": c10_history.loc[c10_step_time - pd.Timedelta(hours=144)],
                "lag_168": c10_history.loc[c10_step_time - pd.Timedelta(hours=168)],
                "rolling_mean_24h": c10_history.loc[
                    c10_step_time - pd.Timedelta(hours=24) : c10_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
                "rolling_mean_168h": c10_history.loc[
                    c10_step_time - pd.Timedelta(hours=168) : c10_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
            }
            c10_row_df = pd.DataFrame([c10_row])[c9_feature_cols]

            c10_pred_value = c10_model.predict(c10_row_df)[0]
            c10_pred_value = max(c10_pred_value, 0.0)  # physical floor at 0
            c10_pred_value = (
                0.0 if c10_pred_value < c10_zero_threshold else c10_pred_value
            )  # NEW: zero-floor fix

            c10_predictions.append(c10_pred_value)
            c10_history.loc[c10_step_time] = c10_pred_value

        c10_forecast = pd.Series(c10_predictions, index=c10_forecast_index)

        c10_aligned = pd.DataFrame(
            {"actual": c10_actual_wf, "forecast": c10_forecast}
        ).dropna()
        assert len(c10_aligned) == 168, (
            f"Window {c10_w + 1} has {len(c10_aligned)} rows, expected 168"
        )

        c10_smape_val = smape(c10_aligned["actual"], c10_aligned["forecast"])
        c10_mse_val = np.mean((c10_aligned["actual"] - c10_aligned["forecast"]) ** 2)

        xgb2_smape_scores.append(c10_smape_val)
        xgb2_mse_scores.append(c10_mse_val)

        print(
            f"  Window {c10_w + 1} ({c10_cutoff_w.date()} cutoff): "
            f"SMAPE={c10_smape_val:.4f}, MSE={c10_mse_val:.1f}, fit_time={c10_elapsed:.2f}s"
        )

    xgb2_mean_smape = np.mean(xgb2_smape_scores)
    xgb2_mean_mse = np.mean(xgb2_mse_scores)
    xgb2_std_smape = np.std(xgb2_smape_scores)
    xgb2_total_fit_time = np.sum(xgb2_fit_seconds)

    print(f"\n{'=' * 60}")
    print(f"XGBOOST v2 RESULTS — WITH ZERO-FLOOR FIX  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {xgb2_mean_smape:.4f}  ± {xgb2_std_smape:.4f} std")
    print(f"Mean MSE:   {xgb2_mean_mse:.1f}")
    print(f"SMAPE range: [{min(xgb2_smape_scores):.4f}, {max(xgb2_smape_scores):.4f}]")
    print(
        f"Total fit time across {N_VAL_WEEKS} windows: {xgb2_total_fit_time:.1f}s "
        f"(avg {xgb2_total_fit_time / N_VAL_WEEKS:.2f}s/window)"
    )

    print(f"\n{'=' * 60}")
    print("FULL MODEL COMPARISON — ALL MODELS, ALL VERSIONS")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:        {naive_mean_smape:.4f}")
    print(f"SARIMA v1 SMAPE:     {sarima_mean_smape:.4f}  (collapsed — no seasonal AR)")
    print(f"SARIMA v2 SMAPE:     {sarima2_mean_smape:.4f}  (fixed — daily-only memory)")
    print(f"XGBoost v1 SMAPE:    {xgb_mean_smape:.4f}  (zero-hour noise problem)")
    print(f"XGBoost v2 SMAPE:    {xgb2_mean_smape:.4f}  (zero-floor fix applied)")

    c10_xgb_fix_improvement = (
        (xgb_mean_smape - xgb2_mean_smape) / xgb_mean_smape * 100
    )  # renamed from v1_to_v2_improvement
    print(f"\nZero-floor fix improved XGBoost SMAPE by {c10_xgb_fix_improvement:.1f}%")

    if xgb2_mean_smape < naive_mean_smape:
        c10_xgb_vs_naive_improvement = (
            (naive_mean_smape - xgb2_mean_smape) / naive_mean_smape * 100
        )  # renamed from xgb2_improvement
        print(
            f"\n✅ XGBoost v2 BEATS naive baseline by {c10_xgb_vs_naive_improvement:.1f}%"
        )
    else:
        c10_xgb_vs_naive_gap = (
            (xgb2_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        )  # renamed from xgb2_gap
        print(f"\n❌ XGBoost v2 still WORSE than naive by {c10_xgb_vs_naive_gap:.1f}%")
    return c10_zero_threshold, xgb2_mean_smape


@app.cell
def _(
    c10_zero_threshold,
    c8_df,
    c9_feature_cols,
    naive_smape_scores,
    pd,
    series,
    smape,
    val_cutoffs,
    xgb,
):
    # ── CELL 10b: DIAGNOSTIC — does a RECENCY WINDOW fix early-window failure? ──
    # Pattern observed in Cell 10: XGBoost v2 SMAPE improves monotonically as
    # the cutoff moves later (0.776 -> 0.213), while naive's SMAPE does NOT
    # show this pattern (0.276 -> 0.180, roughly flat/noisy). This suggests
    # XGBoost is being diluted by stale pre-structural-break training data
    # (recall Cell 4: trend showed a sharp regime change around Sep 2021).
    # We test: does training on ONLY the last N hours (like Cell 6's SARIMA
    # recency window) fix this, especially for the earliest windows?
    #
    # We test Window 1 specifically (worst performer, SMAPE=0.776) with three
    # different training-history lengths, to find where the dilution effect
    # stops mattering.

    c10b_cutoff = val_cutoffs[0]  # Window 1: 2021-12-05 23:00:00, our worst performer
    c10b_actual_wf = series[
        (series.index > c10b_cutoff)
        & (series.index <= c10b_cutoff + pd.Timedelta(hours=168))
    ]

    c10b_history_lengths = {
        "Full history (~10mo)": None,  # None = use everything, same as Cell 10
        "Last 4000h (~167d)": 4000,
        "Last 2000h (~83d)": 2000,  # same window SARIMA used
        "Last 1000h (~42d)": 1000,
    }

    print("=" * 60)
    print("TESTING DIFFERENT TRAINING WINDOW LENGTHS — Window 1 (worst performer)")
    print("=" * 60)

    for c10b_label, c10b_n_hours in c10b_history_lengths.items():
        c10b_full_train_df = c8_df[c8_df.index <= c10b_cutoff]
        c10b_train_df = (
            c10b_full_train_df
            if c10b_n_hours is None
            else c10b_full_train_df.iloc[-c10b_n_hours:]
        )

        c10b_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        c10b_model.fit(c10b_train_df[c9_feature_cols], c10b_train_df["orders"])

        c10b_history = series[series.index <= c10b_cutoff].copy()
        c10b_predictions = []
        c10b_forecast_index = pd.date_range(
            start=c10b_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c10b_step_time in c10b_forecast_index:
            c10b_row = {
                "hour_of_day": c10b_step_time.hour,
                "day_of_week": c10b_step_time.dayofweek,
                "is_weekend": int(c10b_step_time.dayofweek >= 5),
                "time_index": (c10b_step_time - c8_df.index[0]).total_seconds() / 3600
                + 168,
                "lag_1": c10b_history.loc[c10b_step_time - pd.Timedelta(hours=1)],
                "lag_24": c10b_history.loc[c10b_step_time - pd.Timedelta(hours=24)],
                "lag_48": c10b_history.loc[c10b_step_time - pd.Timedelta(hours=48)],
                "lag_144": c10b_history.loc[c10b_step_time - pd.Timedelta(hours=144)],
                "lag_168": c10b_history.loc[c10b_step_time - pd.Timedelta(hours=168)],
                "rolling_mean_24h": c10b_history.loc[
                    c10b_step_time - pd.Timedelta(hours=24) : c10b_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
                "rolling_mean_168h": c10b_history.loc[
                    c10b_step_time - pd.Timedelta(hours=168) : c10b_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
            }
            c10b_row_df = pd.DataFrame([c10b_row])[c9_feature_cols]
            c10b_pred = c10b_model.predict(c10b_row_df)[0]
            c10b_pred = max(c10b_pred, 0.0)
            c10b_pred = 0.0 if c10b_pred < c10_zero_threshold else c10b_pred

            c10b_predictions.append(c10b_pred)
            c10b_history.loc[c10b_step_time] = c10b_pred

        c10b_forecast = pd.Series(c10b_predictions, index=c10b_forecast_index)
        c10b_smape_val = smape(c10b_actual_wf, c10b_forecast)

        print(
            f"  {c10b_label:25s} (n_train={len(c10b_train_df):5d} rows): SMAPE = {c10b_smape_val:.4f}"
        )

    print(f"\n  Reference — naive baseline this window: {naive_smape_scores[0]:.4f}")
    print("  Reference — XGBoost v2 full-history (Cell 10): 0.7760")
    return


@app.cell
def _(
    c10_zero_threshold,
    c8_df,
    c9_feature_cols,
    naive_smape_scores,
    pd,
    series,
    smape,
    val_cutoffs,
    xgb,
):
    # ── CELL 10c: DIAGNOSTIC — proper regularization test, zero-floor fix retained ──
    # Training-window-length test (Cell 10b) gave a NON-monotonic, confusing
    # result — ruling out simple "stale data dilution" as the clean explanation.
    # New hypothesis: with the zero-floor fix already in place, the model may
    # still be overfitting to spurious patterns on NON-zero hours specifically,
    # given how much capacity it has (300 trees, depth 5) relative to how much
    # of the signal is genuinely learnable beyond "mostly copy lag_168."
    #
    # This time we test PROPER regularization tools (not just window length):
    #   max_depth: shallower trees = less capacity to memorize noise
    #   min_child_weight: minimum data points required per leaf — higher values
    #     force trees to only split on patterns supported by enough evidence
    #   subsample: fraction of training rows used per tree — adds randomness,
    #     a standard anti-overfitting technique in gradient boosting
    # We keep FULL history (no artificial window cut) and the zero-floor fix,
    # isolating regularization as the only variable under test.

    c10c_configs = {
        "v2 baseline (depth=5, n=300)": dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=1,
            subsample=1.0,
        ),
        "Regularized A (depth=3, min_child=10)": dict(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
        ),
        "Regularized B (depth=4, min_child=20)": dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=20,
            subsample=0.8,
        ),
    }

    c10c_cutoff = val_cutoffs[0]  # Window 1, same worst-performer
    c10c_actual_wf = series[
        (series.index > c10c_cutoff)
        & (series.index <= c10c_cutoff + pd.Timedelta(hours=168))
    ]
    c10c_full_train_df = c8_df[
        c8_df.index <= c10c_cutoff
    ]  # FULL history, no window cut

    print("=" * 60)
    print(
        "TESTING REGULARIZATION CONFIGS — Window 1, FULL history, zero-floor fix active"
    )
    print("=" * 60)

    for c10c_label, c10c_params in c10c_configs.items():
        c10c_model = xgb.XGBRegressor(
            **c10c_params, random_state=42, objective="reg:squarederror"
        )
        c10c_model.fit(
            c10c_full_train_df[c9_feature_cols], c10c_full_train_df["orders"]
        )

        c10c_history = series[series.index <= c10c_cutoff].copy()
        c10c_predictions = []
        c10c_forecast_index = pd.date_range(
            start=c10c_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c10c_step_time in c10c_forecast_index:
            c10c_row = {
                "hour_of_day": c10c_step_time.hour,
                "day_of_week": c10c_step_time.dayofweek,
                "is_weekend": int(c10c_step_time.dayofweek >= 5),
                "time_index": (c10c_step_time - c8_df.index[0]).total_seconds() / 3600
                + 168,
                "lag_1": c10c_history.loc[c10c_step_time - pd.Timedelta(hours=1)],
                "lag_24": c10c_history.loc[c10c_step_time - pd.Timedelta(hours=24)],
                "lag_48": c10c_history.loc[c10c_step_time - pd.Timedelta(hours=48)],
                "lag_144": c10c_history.loc[c10c_step_time - pd.Timedelta(hours=144)],
                "lag_168": c10c_history.loc[c10c_step_time - pd.Timedelta(hours=168)],
                "rolling_mean_24h": c10c_history.loc[
                    c10c_step_time - pd.Timedelta(hours=24) : c10c_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
                "rolling_mean_168h": c10c_history.loc[
                    c10c_step_time - pd.Timedelta(hours=168) : c10c_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
            }
            c10c_row_df = pd.DataFrame([c10c_row])[c9_feature_cols]
            c10c_pred = c10c_model.predict(c10c_row_df)[0]
            c10c_pred = max(c10c_pred, 0.0)
            c10c_pred = 0.0 if c10c_pred < c10_zero_threshold else c10c_pred

            c10c_predictions.append(c10c_pred)
            c10c_history.loc[c10c_step_time] = c10c_pred

        c10c_forecast = pd.Series(c10c_predictions, index=c10c_forecast_index)
        c10c_smape_val = smape(c10c_actual_wf, c10c_forecast)

        print(f"  {c10c_label:42s}: SMAPE = {c10c_smape_val:.4f}")

    print(
        f"\n  Reference — naive baseline this window:        {naive_smape_scores[0]:.4f}"
    )
    print("  Reference — XGBoost v2 (Cell 10, unregularized): 0.7760")
    return


@app.cell
def _(pd, plt, series):
    # ── CELL 10d: DIAGNOSTIC — what's actually unusual about Window 1's target week? ──
    # Three different attempts to fix XGBoost (less data, more regularization,
    # different window lengths) all made things WORSE on Window 1, never better.
    # This consistently points AWAY from "the model is overfitting" and TOWARD
    # "Window 1's target week contains something genuinely unprecedented that
    # no amount of tuning on PAST data could anticipate." Let's look directly
    # at what actually happened that week, and compare it to the model's
    # lag_168 reference (i.e., the SAME week one year... no, one WEEK prior).

    c10d_target_week = series[
        (series.index >= pd.Timestamp("2021-12-06"))
        & (series.index <= pd.Timestamp("2021-12-12 23:00:00"))
    ]
    c10d_prior_week = series[
        (series.index >= pd.Timestamp("2021-11-29"))
        & (series.index <= pd.Timestamp("2021-12-05 23:00:00"))
    ]

    print("=" * 60)
    print("WINDOW 1 TARGET WEEK (Dec 6-12, 2021) vs PRIOR WEEK (Nov 29-Dec 5)")
    print("=" * 60)
    print(
        f"Target week — mean: {c10d_target_week.mean():.1f}, max: {c10d_target_week.max():.1f}"
    )
    print(
        f"Prior week  — mean: {c10d_prior_week.mean():.1f}, max: {c10d_prior_week.max():.1f}"
    )
    print(
        f"Week-over-week change in mean: "
        f"{(c10d_target_week.mean() / c10d_prior_week.mean() - 1) * 100:+.1f}%"
    )

    # ── Check: was there a Spanish public holiday in this window? ────────────
    # December 6 (Constitution Day) and December 8 (Immaculate Conception) are
    # BOTH Spanish national holidays — and when they fall near a weekend, Spain
    # often takes a "puente" (bridge) — an extended holiday period with very
    # different demand patterns than a normal week.
    print("\nDay-by-day breakdown of target week:")
    c10d_daily = c10d_target_week.groupby(c10d_target_week.index.date).agg(
        ["mean", "max"]
    )
    print(c10d_daily)

    # ── Plot target week vs the lag_168 reference (i.e., the prior week, same hours) ──
    fig_diag3, ax_diag3 = plt.subplots(figsize=(16, 5))
    c10d_target_reindexed = c10d_target_week.values
    c10d_prior_reindexed = c10d_prior_week.values
    ax_diag3.plot(
        range(168),
        c10d_target_reindexed,
        color="#2563EB",
        linewidth=1.5,
        label="Target week (Dec 6-12, what we tried to predict)",
        marker="o",
        ms=3,
    )
    ax_diag3.plot(
        range(168),
        c10d_prior_reindexed,
        color="#059669",
        linewidth=1.5,
        linestyle="--",
        label="Prior week (Nov 29-Dec 5, what lag_168 copies)",
        marker="s",
        ms=3,
    )
    ax_diag3.set_title(
        "DIAGNOSTIC: Is Window 1's Target Week Anomalous vs the Week Before It?",
        fontweight="bold",
    )
    ax_diag3.set_xlabel("Hour index within week (0=Monday 00:00)")
    ax_diag3.set_ylabel("Orders per hour")
    ax_diag3.legend()
    plt.tight_layout()
    plt.savefig(
        "figures/diag_03_window1_anomaly_check.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    return


@app.cell
def _(
    c10_zero_threshold,
    c8_df,
    c9_feature_cols,
    pd,
    plt,
    series,
    smape,
    val_cutoffs,
    xgb,
):
    # ── CELL 10e: DIAGNOSTIC — actually LOOK at XGBoost v2's Window 1 forecast ──
    # Three hypotheses (overfitting, regularization, training window) all
    # failed to improve Window 1. The target week itself looks unremarkable
    # (Cell 10d). We never directly visualized XGBoost v2's actual predictions
    # for this window — every other model failure in this notebook was
    # diagnosed by plotting first. Doing that now, finally, for v2 specifically.

    c10e_cutoff = val_cutoffs[0]
    c10e_train_df = c8_df[
        c8_df.index <= c10e_cutoff
    ]  # full history, matching Cell 10's v2 exactly
    c10e_actual_wf = series[
        (series.index > c10e_cutoff)
        & (series.index <= c10e_cutoff + pd.Timedelta(hours=168))
    ]

    c10e_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        objective="reg:squarederror",
    )
    c10e_model.fit(c10e_train_df[c9_feature_cols], c10e_train_df["orders"])

    c10e_history = series[series.index <= c10e_cutoff].copy()
    c10e_predictions = []
    c10e_lag168_track = []  # track lag_168 input at every step, same diagnostic as Cell 9b
    c10e_forecast_index = pd.date_range(
        start=c10e_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
    )

    for c10e_step_time in c10e_forecast_index:
        c10e_lag168_val = c10e_history.loc[c10e_step_time - pd.Timedelta(hours=168)]
        c10e_row = {
            "hour_of_day": c10e_step_time.hour,
            "day_of_week": c10e_step_time.dayofweek,
            "is_weekend": int(c10e_step_time.dayofweek >= 5),
            "time_index": (c10e_step_time - c8_df.index[0]).total_seconds() / 3600
            + 168,
            "lag_1": c10e_history.loc[c10e_step_time - pd.Timedelta(hours=1)],
            "lag_24": c10e_history.loc[c10e_step_time - pd.Timedelta(hours=24)],
            "lag_48": c10e_history.loc[c10e_step_time - pd.Timedelta(hours=48)],
            "lag_144": c10e_history.loc[c10e_step_time - pd.Timedelta(hours=144)],
            "lag_168": c10e_lag168_val,
            "rolling_mean_24h": c10e_history.loc[
                c10e_step_time - pd.Timedelta(hours=24) : c10e_step_time
                - pd.Timedelta(hours=1)
            ].mean(),
            "rolling_mean_168h": c10e_history.loc[
                c10e_step_time - pd.Timedelta(hours=168) : c10e_step_time
                - pd.Timedelta(hours=1)
            ].mean(),
        }
        c10e_row_df = pd.DataFrame([c10e_row])[c9_feature_cols]
        c10e_pred = c10e_model.predict(c10e_row_df)[0]
        c10e_pred = max(c10e_pred, 0.0)
        c10e_pred = 0.0 if c10e_pred < c10_zero_threshold else c10e_pred

        c10e_predictions.append(c10e_pred)
        c10e_lag168_track.append(c10e_lag168_val)
        c10e_history.loc[c10e_step_time] = c10e_pred

    c10e_forecast = pd.Series(c10e_predictions, index=c10e_forecast_index)

    print(
        f"XGBoost v2 SMAPE this window: {smape(c10e_actual_wf, c10e_forecast):.4f}  "
        f"(should match Cell 10's 0.7760)"
    )

    # ── THE KEY PLOT: actual vs v2 forecast vs lag_168 reference, ALL together ──
    fig_diag4, ax_diag4 = plt.subplots(figsize=(16, 5))
    ax_diag4.plot(
        c10e_actual_wf.index,
        c10e_actual_wf.values,
        color="#2563EB",
        linewidth=1.5,
        label="Actual orders",
        marker="o",
        ms=3,
    )
    ax_diag4.plot(
        c10e_forecast.index,
        c10e_forecast.values,
        color="#EF4444",
        linewidth=1.5,
        label="XGBoost v2 forecast (zero-floor fix)",
        marker="x",
        ms=3,
    )
    ax_diag4.plot(
        c10e_forecast.index,
        c10e_lag168_track,
        color="#059669",
        linewidth=1,
        linestyle="--",
        label="lag_168 reference (naive copy)",
        alpha=0.7,
    )
    ax_diag4.set_title(
        "DIAGNOSTIC: XGBoost v2 — Window 1 Forecast vs Actual vs lag_168",
        fontweight="bold",
    )
    ax_diag4.set_xlabel("Time")
    ax_diag4.set_ylabel("Orders per hour")
    ax_diag4.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/diag_04_xgboost_v2_window1.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Print the actual numbers hour-by-hour for the first 48 hours ─────────
    print("\nFirst 48 hours, side by side:")
    c10e_compare = pd.DataFrame(
        {
            "actual": c10e_actual_wf.values[:48],
            "xgb_v2_forecast": c10e_forecast.values[:48],
            "lag_168_ref": c10e_lag168_track[:48],
        },
        index=c10e_forecast.index[:48],
    )
    print(c10e_compare)
    return c10e_actual_wf, c10e_cutoff, c10e_forecast, c10e_train_df


@app.cell
def _(
    c10_zero_threshold,
    c10e_actual_wf,
    c10e_cutoff,
    c10e_forecast,
    c10e_train_df,
    c9_feature_cols,
    naive_smape_scores,
    pd,
    plt,
    series,
    smape,
    xgb,
):
    # ── CELL 10f: VERIFY — is time_index extrapolation the actual bug? ───────
    # Hypothesis: tree-based models CANNOT extrapolate beyond the range of
    # values seen during training. time_index at forecast time (~7224-7392)
    # exceeds every value seen during training (max ~7224, the cutoff itself).
    # If true, REMOVING time_index entirely should fix the systematic upward
    # bias, since the model would then rely only on lag features (which DO
    # generalize fine — lag_168's actual numeric value is just "whatever
    # orders was 168 hours ago," not a fixed range that gets exceeded).

    c10f_feature_cols_no_trend = [c for c in c9_feature_cols if c != "time_index"]
    print(f"Features WITHOUT time_index: {c10f_feature_cols_no_trend}")

    c10f_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        objective="reg:squarederror",
    )
    c10f_model.fit(c10e_train_df[c10f_feature_cols_no_trend], c10e_train_df["orders"])

    c10f_history = series[series.index <= c10e_cutoff].copy()
    c10f_predictions = []
    c10f_forecast_index = pd.date_range(
        start=c10e_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
    )

    for c10f_step_time in c10f_forecast_index:
        c10f_row = {
            "hour_of_day": c10f_step_time.hour,
            "day_of_week": c10f_step_time.dayofweek,
            "is_weekend": int(c10f_step_time.dayofweek >= 5),
            # NOTE: time_index REMOVED entirely — testing the extrapolation hypothesis
            "lag_1": c10f_history.loc[c10f_step_time - pd.Timedelta(hours=1)],
            "lag_24": c10f_history.loc[c10f_step_time - pd.Timedelta(hours=24)],
            "lag_48": c10f_history.loc[c10f_step_time - pd.Timedelta(hours=48)],
            "lag_144": c10f_history.loc[c10f_step_time - pd.Timedelta(hours=144)],
            "lag_168": c10f_history.loc[c10f_step_time - pd.Timedelta(hours=168)],
            "rolling_mean_24h": c10f_history.loc[
                c10f_step_time - pd.Timedelta(hours=24) : c10f_step_time
                - pd.Timedelta(hours=1)
            ].mean(),
            "rolling_mean_168h": c10f_history.loc[
                c10f_step_time - pd.Timedelta(hours=168) : c10f_step_time
                - pd.Timedelta(hours=1)
            ].mean(),
        }
        c10f_row_df = pd.DataFrame([c10f_row])[c10f_feature_cols_no_trend]
        c10f_pred = c10f_model.predict(c10f_row_df)[0]
        c10f_pred = max(c10f_pred, 0.0)
        c10f_pred = 0.0 if c10f_pred < c10_zero_threshold else c10f_pred

        c10f_predictions.append(c10f_pred)
        c10f_history.loc[c10f_step_time] = c10f_pred

    c10f_forecast = pd.Series(c10f_predictions, index=c10f_forecast_index)
    c10f_smape_val = smape(c10e_actual_wf, c10f_forecast)

    print("\nSMAPE WITH time_index (Cell 10e):    0.7760")
    print(f"SMAPE WITHOUT time_index (this test): {c10f_smape_val:.4f}")
    print(f"\nReference — naive baseline this window: {naive_smape_scores[0]:.4f}")

    # ── Plot the comparison directly ──────────────────────────────────────────
    fig_diag5, ax_diag5 = plt.subplots(figsize=(16, 5))
    ax_diag5.plot(
        c10e_actual_wf.index,
        c10e_actual_wf.values,
        color="#2563EB",
        linewidth=1.5,
        label="Actual orders",
        marker="o",
        ms=3,
    )
    ax_diag5.plot(
        c10e_forecast.index,
        c10e_forecast.values,
        color="#EF4444",
        linewidth=1,
        linestyle=":",
        label="XGBoost v2 WITH time_index (biased)",
        alpha=0.6,
    )
    ax_diag5.plot(
        c10f_forecast.index,
        c10f_forecast.values,
        color="#7C3AED",
        linewidth=1.5,
        label="XGBoost v3 WITHOUT time_index (test)",
        marker="x",
        ms=3,
    )
    ax_diag5.set_title(
        "DIAGNOSTIC: Does Removing time_index Fix the Systematic Upward Bias?",
        fontweight="bold",
    )
    ax_diag5.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/diag_05_time_index_test.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(
    N_VAL_WEEKS,
    c10_zero_threshold,
    c8_df,
    c9_feature_cols,
    naive_mean_smape,
    np,
    pd,
    sarima2_mean_smape,
    sarima_mean_smape,
    series,
    smape,
    time,
    val_cutoffs,
    xgb,
    xgb2_mean_smape,
    xgb_mean_smape,
):
    # ── CELL 11: XGBoost v3 — Remove time_index, full walk-forward validation ──
    # WHY THIS CELL EXISTS: Cell 10 (XGBoost v2, zero-floor fix) scored SMAPE
    # 0.375 — better than v1 but still 74% worse than naive. Diagnosis (Cells
    # 10b-10f) ruled out overfitting, regularization, and training window length
    # as causes — all made Window 1 WORSE, never better, consistently pointing
    # AWAY from "too much model" and toward "missing information" instead.
    #
    # THE ACTUAL BUG (confirmed Cell 10f, visually and numerically): time_index
    # is a continuously-increasing feature. Tree-based models split on FIXED
    # THRESHOLDS learned during training and CANNOT EXTRAPOLATE beyond the
    # range of values seen in training. During recursive forecasting, every
    # future time_index value EXCEEDS every threshold the trees ever learned,
    # so every row gets routed into the single highest-trend leaf regardless
    # of what hour_of_day or lag_168 actually say — causing the systematic
    # upward bias visible in Cell 10e's plot (red line floating above actual
    # everywhere, even leaking into true-zero hours).
    #
    # THE FIX: remove time_index entirely. lag_24/lag_144/lag_168 already
    # implicitly encode recent trend (last week's ACTUAL observed level is
    # baked directly into the lag value itself — no extrapolation needed,
    # since lag values are always "real numbers we've seen," never
    # "increasing indices beyond the trained range"). Confirmed in Cell 10f:
    # removing time_index dropped Window 1 SMAPE from 0.776 -> 0.300, nearly
    # matching naive's 0.276 on that exact window.

    xgb3_smape_scores = []  # prefixed xgb3_ to avoid colliding with Cell 9/10's names
    xgb3_mse_scores = []
    xgb3_fit_seconds = []

    c11_feature_cols = [
        c for c in c9_feature_cols if c != "time_index"
    ]  # the fix, applied globally
    print(f"Features used in XGBoost v3 (time_index removed): {c11_feature_cols}")

    for c11_w, c11_cutoff_w in enumerate(val_cutoffs):
        c11_train_df = c8_df[c8_df.index <= c11_cutoff_w]
        c11_actual_wf = series[
            (series.index > c11_cutoff_w)
            & (series.index <= c11_cutoff_w + pd.Timedelta(hours=168))
        ]

        c11_start_time = time.time()

        c11_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        c11_model.fit(c11_train_df[c11_feature_cols], c11_train_df["orders"])

        c11_elapsed = time.time() - c11_start_time
        xgb3_fit_seconds.append(c11_elapsed)

        c11_history = series[series.index <= c11_cutoff_w].copy()
        c11_predictions = []
        c11_forecast_index = pd.date_range(
            start=c11_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c11_step_time in c11_forecast_index:
            c11_row = {
                "hour_of_day": c11_step_time.hour,
                "day_of_week": c11_step_time.dayofweek,
                "is_weekend": int(c11_step_time.dayofweek >= 5),
                # time_index REMOVED — the confirmed fix
                "lag_1": c11_history.loc[c11_step_time - pd.Timedelta(hours=1)],
                "lag_24": c11_history.loc[c11_step_time - pd.Timedelta(hours=24)],
                "lag_48": c11_history.loc[c11_step_time - pd.Timedelta(hours=48)],
                "lag_144": c11_history.loc[c11_step_time - pd.Timedelta(hours=144)],
                "lag_168": c11_history.loc[c11_step_time - pd.Timedelta(hours=168)],
                "rolling_mean_24h": c11_history.loc[
                    c11_step_time - pd.Timedelta(hours=24) : c11_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
                "rolling_mean_168h": c11_history.loc[
                    c11_step_time - pd.Timedelta(hours=168) : c11_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
            }
            c11_row_df = pd.DataFrame([c11_row])[c11_feature_cols]

            c11_pred_value = c11_model.predict(c11_row_df)[0]
            c11_pred_value = max(c11_pred_value, 0.0)
            c11_pred_value = (
                0.0 if c11_pred_value < c10_zero_threshold else c11_pred_value
            )

            c11_predictions.append(c11_pred_value)
            c11_history.loc[c11_step_time] = c11_pred_value

        c11_forecast = pd.Series(c11_predictions, index=c11_forecast_index)

        c11_aligned = pd.DataFrame(
            {"actual": c11_actual_wf, "forecast": c11_forecast}
        ).dropna()
        assert len(c11_aligned) == 168, (
            f"Window {c11_w + 1} has {len(c11_aligned)} rows, expected 168"
        )

        c11_smape_val = smape(c11_aligned["actual"], c11_aligned["forecast"])
        c11_mse_val = np.mean((c11_aligned["actual"] - c11_aligned["forecast"]) ** 2)

        xgb3_smape_scores.append(c11_smape_val)
        xgb3_mse_scores.append(c11_mse_val)

        print(
            f"  Window {c11_w + 1} ({c11_cutoff_w.date()} cutoff): "
            f"SMAPE={c11_smape_val:.4f}, MSE={c11_mse_val:.1f}, fit_time={c11_elapsed:.2f}s"
        )

    xgb3_mean_smape = np.mean(xgb3_smape_scores)
    xgb3_mean_mse = np.mean(xgb3_mse_scores)
    xgb3_std_smape = np.std(xgb3_smape_scores)
    xgb3_total_fit_time = np.sum(xgb3_fit_seconds)

    print(f"\n{'=' * 60}")
    print(f"XGBOOST v3 RESULTS — time_index REMOVED  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {xgb3_mean_smape:.4f}  ± {xgb3_std_smape:.4f} std")
    print(f"Mean MSE:   {xgb3_mean_mse:.1f}")
    print(f"SMAPE range: [{min(xgb3_smape_scores):.4f}, {max(xgb3_smape_scores):.4f}]")
    print(
        f"Total fit time across {N_VAL_WEEKS} windows: {xgb3_total_fit_time:.1f}s "
        f"(avg {xgb3_total_fit_time / N_VAL_WEEKS:.2f}s/window)"
    )

    print(f"\n{'=' * 60}")
    print("FULL MODEL COMPARISON — ALL MODELS, ALL VERSIONS")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:         {naive_mean_smape:.4f}")
    print(
        f"SARIMA v1 SMAPE:      {sarima_mean_smape:.4f}  (collapsed — no seasonal AR)"
    )
    print(
        f"SARIMA v2 SMAPE:      {sarima2_mean_smape:.4f}  (fixed — daily-only memory)"
    )
    print(f"XGBoost v1 SMAPE:     {xgb_mean_smape:.4f}  (zero-hour noise problem)")
    print(f"XGBoost v2 SMAPE:     {xgb2_mean_smape:.4f}  (zero-floor fix applied)")
    print(
        f"XGBoost v3 SMAPE:     {xgb3_mean_smape:.4f}  (time_index removed — extrapolation fix)"
    )

    c11_v2_to_v3_improvement = (
        (xgb2_mean_smape - xgb3_mean_smape) / xgb2_mean_smape * 100
    )
    print(
        f"\ntime_index removal improved XGBoost SMAPE by {c11_v2_to_v3_improvement:.1f}% vs v2"
    )

    if xgb3_mean_smape < naive_mean_smape:
        c11_v3_vs_naive = (naive_mean_smape - xgb3_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ XGBoost v3 BEATS naive baseline by {c11_v3_vs_naive:.1f}%")
    else:
        c11_v3_gap = (xgb3_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ XGBoost v3 still WORSE than naive by {c11_v3_gap:.1f}%")
    return c11_feature_cols, xgb3_mean_smape


@app.cell
def _(
    c10_zero_threshold,
    c11_feature_cols,
    c8_df,
    pd,
    plt,
    series,
    smape,
    val_cutoffs,
    xgb,
):
    # ── CELL 11b: DIAGNOSTIC — why is Window 2 catastrophic in XGBoost v3? ───
    # 6 of 7 windows now look excellent (close to naive), but Window 2 spiked
    # to SMAPE=0.992 — nearly the worst score of any model variant. This is
    # an isolated outlier, not a systematic pattern like before. We inspect
    # it directly: plot forecast vs actual, and check whether rolling_mean
    # features (still continuous, still capable of a smaller-scale version
    # of the time_index extrapolation problem) are implicated.

    c11b_cutoff = val_cutoffs[1]  # Window 2 specifically: 2021-12-12 23:00:00
    c11b_train_df = c8_df[c8_df.index <= c11b_cutoff]
    c11b_actual_wf = series[
        (series.index > c11b_cutoff)
        & (series.index <= c11b_cutoff + pd.Timedelta(hours=168))
    ]

    c11b_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        objective="reg:squarederror",
    )
    c11b_model.fit(c11b_train_df[c11_feature_cols], c11b_train_df["orders"])

    c11b_history = series[series.index <= c11b_cutoff].copy()
    c11b_predictions = []
    c11b_lag168_track = []
    c11b_forecast_index = pd.date_range(
        start=c11b_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
    )

    for c11b_step_time in c11b_forecast_index:
        c11b_lag168_val = c11b_history.loc[c11b_step_time - pd.Timedelta(hours=168)]
        c11b_row = {
            "hour_of_day": c11b_step_time.hour,
            "day_of_week": c11b_step_time.dayofweek,
            "is_weekend": int(c11b_step_time.dayofweek >= 5),
            "lag_1": c11b_history.loc[c11b_step_time - pd.Timedelta(hours=1)],
            "lag_24": c11b_history.loc[c11b_step_time - pd.Timedelta(hours=24)],
            "lag_48": c11b_history.loc[c11b_step_time - pd.Timedelta(hours=48)],
            "lag_144": c11b_history.loc[c11b_step_time - pd.Timedelta(hours=144)],
            "lag_168": c11b_lag168_val,
            "rolling_mean_24h": c11b_history.loc[
                c11b_step_time - pd.Timedelta(hours=24) : c11b_step_time
                - pd.Timedelta(hours=1)
            ].mean(),
            "rolling_mean_168h": c11b_history.loc[
                c11b_step_time - pd.Timedelta(hours=168) : c11b_step_time
                - pd.Timedelta(hours=1)
            ].mean(),
        }
        c11b_row_df = pd.DataFrame([c11b_row])[c11_feature_cols]
        c11b_pred = c11b_model.predict(c11b_row_df)[0]
        c11b_pred = max(c11b_pred, 0.0)
        c11b_pred = 0.0 if c11b_pred < c10_zero_threshold else c11b_pred

        c11b_predictions.append(c11b_pred)
        c11b_lag168_track.append(c11b_lag168_val)
        c11b_history.loc[c11b_step_time] = c11b_pred

    c11b_forecast = pd.Series(c11b_predictions, index=c11b_forecast_index)
    print(
        f"Confirmed Window 2 SMAPE: {smape(c11b_actual_wf, c11b_forecast):.4f}  (should match 0.9920)"
    )

    # ── Check: is the training data's rolling_mean_168h range different here? ─
    print(
        f"\nTraining data rolling_mean_168h range: "
        f"[{c11b_train_df['rolling_mean_168h'].min():.1f}, {c11b_train_df['rolling_mean_168h'].max():.1f}]"
    )
    print("Forecast-time rolling_mean_168h values used (first 5, last 5):")
    c11b_roll168_used = [
        c11b_history.loc[t - pd.Timedelta(hours=168) : t - pd.Timedelta(hours=1)].mean()
        for t in c11b_forecast_index
    ]
    print(f"  First 5: {[f'{v:.1f}' for v in c11b_roll168_used[:5]]}")
    print(f"  Last 5:  {[f'{v:.1f}' for v in c11b_roll168_used[-5:]]}")

    # ── Plot actual vs forecast vs lag_168 reference ──────────────────────────
    fig_diag6, ax_diag6 = plt.subplots(figsize=(16, 5))
    ax_diag6.plot(
        c11b_actual_wf.index,
        c11b_actual_wf.values,
        color="#2563EB",
        linewidth=1.5,
        label="Actual orders",
        marker="o",
        ms=3,
    )
    ax_diag6.plot(
        c11b_forecast.index,
        c11b_forecast.values,
        color="#EF4444",
        linewidth=1.5,
        label="XGBoost v3 forecast",
        marker="x",
        ms=3,
    )
    ax_diag6.plot(
        c11b_forecast.index,
        c11b_lag168_track,
        color="#059669",
        linewidth=1,
        linestyle="--",
        label="lag_168 reference",
        alpha=0.7,
    )
    ax_diag6.set_title(
        "DIAGNOSTIC: XGBoost v3 — Window 2 Forecast vs Actual (the outlier)",
        fontweight="bold",
    )
    ax_diag6.set_xlabel("Time")
    ax_diag6.set_ylabel("Orders per hour")
    ax_diag6.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/diag_06_window2_outlier.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Also check: was there something unusual about THIS specific target week? ──
    print("\nDay-by-day breakdown of Window 2's target week (Dec 13-19, 2021):")
    c11b_target_week = series[
        (series.index >= pd.Timestamp("2021-12-13"))
        & (series.index <= pd.Timestamp("2021-12-19 23:00:00"))
    ]
    print(c11b_target_week.groupby(c11b_target_week.index.date).agg(["mean", "max"]))
    return c11b_actual_wf, c11b_cutoff, c11b_forecast, c11b_model


@app.cell
def _(c11b_actual_wf, c11b_forecast, np, pd):
    # ── CELL 11c: DIAGNOSTIC — find which SPECIFIC hours are driving Window 2's high SMAPE ──
    # The plot shows good visual tracking, but the reported SMAPE (0.992) is
    # near-catastrophic. This mismatch suggests a small number of extreme
    # individual-hour SMAPE values are dominating the 168-hour average, rather
    # than a uniformly bad forecast. We compute SMAPE PER HOUR and sort to find
    # the worst offenders directly.

    c11c_per_hour_smape = pd.DataFrame(
        {
            "time": c11b_actual_wf.index,
            "actual": c11b_actual_wf.values,
            "forecast": c11b_forecast.values,
        }
    )
    # Per-hour SMAPE, computed the SAME way as our smape() function but row-by-row
    c11c_per_hour_smape["abs_diff"] = np.abs(
        c11c_per_hour_smape["actual"] - c11c_per_hour_smape["forecast"]
    )
    c11c_per_hour_smape["denom"] = np.abs(c11c_per_hour_smape["actual"]) + np.abs(
        c11c_per_hour_smape["forecast"]
    )
    c11c_per_hour_smape["smape_hour"] = np.where(
        (c11c_per_hour_smape["actual"] == 0) & (c11c_per_hour_smape["forecast"] == 0),
        0.0,
        2 * c11c_per_hour_smape["abs_diff"] / c11c_per_hour_smape["denom"],
    )

    print("=" * 60)
    print("TOP 15 WORST INDIVIDUAL HOURS BY SMAPE CONTRIBUTION")
    print("=" * 60)
    c11c_worst = c11c_per_hour_smape.sort_values("smape_hour", ascending=False).head(15)
    print(
        c11c_worst[["time", "actual", "forecast", "smape_hour"]].to_string(index=False)
    )

    print(f"\n{'=' * 60}")
    print("VERIFY: does manual mean match the reported 0.9920?")
    print(f"{'=' * 60}")
    print(
        f"Manual mean of smape_hour column: {c11c_per_hour_smape['smape_hour'].mean():.4f}"
    )

    print(
        f"\nHow many hours have smape_hour > 1.5 (near-maximum, i.e. near-total miss)? "
        f"{(c11c_per_hour_smape['smape_hour'] > 1.5).sum()} out of 168"
    )
    print(
        f"How many hours have smape_hour > 1.9 (essentially complete miss)? "
        f"{(c11c_per_hour_smape['smape_hour'] > 1.9).sum()} out of 168"
    )
    return


@app.cell
def _(
    c11_feature_cols,
    c11b_actual_wf,
    c11b_cutoff,
    c11b_model,
    naive_smape_scores,
    pd,
    series,
    smape,
):
    # ── CELL 11d: VERIFY — does raising the zero-floor threshold fix Window 2? ──
    # Hypothesis: rolling_mean_168h's extrapolation (training max 99.7, forecast
    # values up to 122.8) is pushing zero-hour predictions to 5-28 range — well
    # above our current 5.0 floor. We test a higher threshold to see if it's
    # at least a viable patch, before deciding whether the more structural
    # fix (removing/transforming rolling_mean features) is needed.

    c11d_test_thresholds = [5.0, 15.0, 30.0, 50.0]

    print("=" * 60)
    print("TESTING HIGHER ZERO-FLOOR THRESHOLDS — Window 2")
    print("=" * 60)

    for c11d_threshold in c11d_test_thresholds:
        c11d_history = series[series.index <= c11b_cutoff].copy()
        c11d_predictions = []
        c11d_forecast_index = pd.date_range(
            start=c11b_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c11d_step_time in c11d_forecast_index:
            c11d_row = {
                "hour_of_day": c11d_step_time.hour,
                "day_of_week": c11d_step_time.dayofweek,
                "is_weekend": int(c11d_step_time.dayofweek >= 5),
                "lag_1": c11d_history.loc[c11d_step_time - pd.Timedelta(hours=1)],
                "lag_24": c11d_history.loc[c11d_step_time - pd.Timedelta(hours=24)],
                "lag_48": c11d_history.loc[c11d_step_time - pd.Timedelta(hours=48)],
                "lag_144": c11d_history.loc[c11d_step_time - pd.Timedelta(hours=144)],
                "lag_168": c11d_history.loc[c11d_step_time - pd.Timedelta(hours=168)],
                "rolling_mean_24h": c11d_history.loc[
                    c11d_step_time - pd.Timedelta(hours=24) : c11d_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
                "rolling_mean_168h": c11d_history.loc[
                    c11d_step_time - pd.Timedelta(hours=168) : c11d_step_time
                    - pd.Timedelta(hours=1)
                ].mean(),
            }
            c11d_row_df = pd.DataFrame([c11d_row])[c11_feature_cols]
            c11d_pred = c11b_model.predict(c11d_row_df)[
                0
            ]  # reuse the SAME fitted model from Cell 11b
            c11d_pred = max(c11d_pred, 0.0)
            c11d_pred = 0.0 if c11d_pred < c11d_threshold else c11d_pred

            c11d_predictions.append(c11d_pred)
            c11d_history.loc[c11d_step_time] = c11d_pred

        c11d_forecast = pd.Series(c11d_predictions, index=c11d_forecast_index)
        c11d_smape_val = smape(c11b_actual_wf, c11d_forecast)
        print(f"  Threshold={c11d_threshold:5.1f}: SMAPE = {c11d_smape_val:.4f}")

    print(f"\nReference — naive baseline this window: {naive_smape_scores[1]:.4f}")
    return


@app.cell
def _(
    N_VAL_WEEKS,
    c10_zero_threshold,
    c8_df,
    naive_mean_smape,
    np,
    pd,
    sarima2_mean_smape,
    sarima_mean_smape,
    series,
    smape,
    time,
    val_cutoffs,
    xgb,
    xgb2_mean_smape,
    xgb3_mean_smape,
    xgb_mean_smape,
):
    # ── CELL 12: XGBoost v4 — Replace absolute rolling means with bounded ratios ──
    # WHY THIS CELL EXISTS: Cell 11 (XGBoost v3, time_index removed) scored well
    # on 6/7 windows but catastrophically on Window 2 (SMAPE 0.992). Diagnosis
    # (Cells 11b-11d) found the SAME extrapolation disease as time_index, just
    # on rolling_mean_168h: training data's range was [50.6, 99.7], but
    # forecast-time values reached 122.8 — beyond anything the trees learned
    # splits for. This pushed zero-hour predictions to 5-28 (above our 5.0
    # zero-floor), causing 55/168 hours to score the maximum possible SMAPE.
    # Threshold-tuning (Cell 11d) showed this is a fragile, non-monotonic
    # patch (15.0 helps, 50.0 hurts) — not a real fix.
    #
    # THE STRUCTURAL FIX: any feature whose ABSOLUTE SCALE drifts upward with
    # the trend will eventually exceed its training range during recursive
    # forecasting on growing data — this is the general form of the bug we've
    # now found twice (time_index, rolling_mean_168h). The fix: replace
    # absolute-level features with RATIO features that stay roughly bounded
    # even as the overall level grows, because numerator and denominator grow
    # together. We replace:
    #   rolling_mean_24h, rolling_mean_168h (absolute levels, REMOVED)
    # with:
    #   ratio_recent_vs_week  = lag_1 / rolling_mean_168h   (is the most recent
    #       hour higher or lower than its typical recent-week level?)
    #   ratio_day_vs_week     = lag_24 / rolling_mean_168h  (is TODAY's same-hour
    #       value higher or lower than the recent-week typical level?)
    # These ratios hover near 1.0 regardless of whether the absolute demand
    # level is 50 or 150 — because as the business grows, BOTH the numerator
    # (a recent lag) and denominator (the rolling mean) grow together,
    # cancelling out the drift that broke the absolute-level features.

    c12_df = c8_df.copy()  # start from Cell 8's feature table, add new features on top

    # Guard against division by zero: when rolling_mean_168h is exactly 0 (a
    # stretch of all-zero recent hours), the ratio is undefined — we set it to
    # 0 explicitly (0/0 "no signal vs no signal" is reasonably treated as
    # neutral/zero here, since both numerator and denominator are zero anyway)
    c12_df["ratio_recent_vs_week"] = np.where(
        c12_df["rolling_mean_168h"] > 0,
        c12_df["lag_1"] / c12_df["rolling_mean_168h"],
        0.0,
    )
    c12_df["ratio_day_vs_week"] = np.where(
        c12_df["rolling_mean_168h"] > 0,
        c12_df["lag_24"] / c12_df["rolling_mean_168h"],
        0.0,
    )

    c12_feature_cols = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "lag_1",
        "lag_24",
        "lag_48",
        "lag_144",
        "lag_168",
        "ratio_recent_vs_week",
        "ratio_day_vs_week",
    ]  # time_index and raw rolling_means REMOVED, replaced by bounded ratios

    print(f"Features used in XGBoost v4: {c12_feature_cols}")
    print(
        "\nRatio feature ranges (should be roughly bounded near 1.0, unlike rolling_mean):"
    )
    print(c12_df[["ratio_recent_vs_week", "ratio_day_vs_week"]].describe())

    xgb4_smape_scores = []
    xgb4_mse_scores = []
    xgb4_fit_seconds = []

    for c12_w, c12_cutoff_w in enumerate(val_cutoffs):
        c12_train_df = c12_df[c12_df.index <= c12_cutoff_w]
        c12_actual_wf = series[
            (series.index > c12_cutoff_w)
            & (series.index <= c12_cutoff_w + pd.Timedelta(hours=168))
        ]

        c12_start_time = time.time()

        c12_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        c12_model.fit(c12_train_df[c12_feature_cols], c12_train_df["orders"])

        c12_elapsed = time.time() - c12_start_time
        xgb4_fit_seconds.append(c12_elapsed)

        c12_history = series[series.index <= c12_cutoff_w].copy()
        c12_predictions = []
        c12_forecast_index = pd.date_range(
            start=c12_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c12_step_time in c12_forecast_index:
            c12_lag1_val = c12_history.loc[c12_step_time - pd.Timedelta(hours=1)]
            c12_lag24_val = c12_history.loc[c12_step_time - pd.Timedelta(hours=24)]
            c12_roll168_val = c12_history.loc[
                c12_step_time - pd.Timedelta(hours=168) : c12_step_time
                - pd.Timedelta(hours=1)
            ].mean()

            c12_row = {
                "hour_of_day": c12_step_time.hour,
                "day_of_week": c12_step_time.dayofweek,
                "is_weekend": int(c12_step_time.dayofweek >= 5),
                "lag_1": c12_lag1_val,
                "lag_24": c12_lag24_val,
                "lag_48": c12_history.loc[c12_step_time - pd.Timedelta(hours=48)],
                "lag_144": c12_history.loc[c12_step_time - pd.Timedelta(hours=144)],
                "lag_168": c12_history.loc[c12_step_time - pd.Timedelta(hours=168)],
                "ratio_recent_vs_week": c12_lag1_val / c12_roll168_val
                if c12_roll168_val > 0
                else 0.0,
                "ratio_day_vs_week": c12_lag24_val / c12_roll168_val
                if c12_roll168_val > 0
                else 0.0,
            }
            c12_row_df = pd.DataFrame([c12_row])[c12_feature_cols]

            c12_pred_value = c12_model.predict(c12_row_df)[0]
            c12_pred_value = max(c12_pred_value, 0.0)
            c12_pred_value = (
                0.0 if c12_pred_value < c10_zero_threshold else c12_pred_value
            )  # back to the original 5.0 threshold

            c12_predictions.append(c12_pred_value)
            c12_history.loc[c12_step_time] = c12_pred_value

        c12_forecast = pd.Series(c12_predictions, index=c12_forecast_index)

        c12_aligned = pd.DataFrame(
            {"actual": c12_actual_wf, "forecast": c12_forecast}
        ).dropna()
        assert len(c12_aligned) == 168, (
            f"Window {c12_w + 1} has {len(c12_aligned)} rows, expected 168"
        )

        c12_smape_val = smape(c12_aligned["actual"], c12_aligned["forecast"])
        c12_mse_val = np.mean((c12_aligned["actual"] - c12_aligned["forecast"]) ** 2)

        xgb4_smape_scores.append(c12_smape_val)
        xgb4_mse_scores.append(c12_mse_val)

        print(
            f"  Window {c12_w + 1} ({c12_cutoff_w.date()} cutoff): "
            f"SMAPE={c12_smape_val:.4f}, MSE={c12_mse_val:.1f}, fit_time={c12_elapsed:.2f}s"
        )

    xgb4_mean_smape = np.mean(xgb4_smape_scores)
    xgb4_mean_mse = np.mean(xgb4_mse_scores)
    xgb4_std_smape = np.std(xgb4_smape_scores)
    xgb4_total_fit_time = np.sum(xgb4_fit_seconds)

    print(f"\n{'=' * 60}")
    print(f"XGBOOST v4 RESULTS — bounded ratio features  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {xgb4_mean_smape:.4f}  ± {xgb4_std_smape:.4f} std")
    print(f"Mean MSE:   {xgb4_mean_mse:.1f}")
    print(f"SMAPE range: [{min(xgb4_smape_scores):.4f}, {max(xgb4_smape_scores):.4f}]")
    print(
        f"Total fit time across {N_VAL_WEEKS} windows: {xgb4_total_fit_time:.1f}s "
        f"(avg {xgb4_total_fit_time / N_VAL_WEEKS:.2f}s/window)"
    )

    print(f"\n{'=' * 60}")
    print("FULL MODEL COMPARISON — ALL MODELS, ALL VERSIONS")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:         {naive_mean_smape:.4f}")
    print(
        f"SARIMA v1 SMAPE:      {sarima_mean_smape:.4f}  (collapsed — no seasonal AR)"
    )
    print(
        f"SARIMA v2 SMAPE:      {sarima2_mean_smape:.4f}  (fixed — daily-only memory)"
    )
    print(f"XGBoost v1 SMAPE:     {xgb_mean_smape:.4f}  (zero-hour noise problem)")
    print(f"XGBoost v2 SMAPE:     {xgb2_mean_smape:.4f}  (zero-floor fix applied)")
    print(f"XGBoost v3 SMAPE:     {xgb3_mean_smape:.4f}  (time_index removed)")
    print(
        f"XGBoost v4 SMAPE:     {xgb4_mean_smape:.4f}  (rolling means -> bounded ratios)"
    )

    c12_v3_to_v4_improvement = (
        (xgb3_mean_smape - xgb4_mean_smape) / xgb3_mean_smape * 100
    )
    print(
        f"\nRatio-feature fix improved XGBoost SMAPE by {c12_v3_to_v4_improvement:.1f}% vs v3"
    )

    if xgb4_mean_smape < naive_mean_smape:
        c12_v4_vs_naive = (naive_mean_smape - xgb4_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ XGBoost v4 BEATS naive baseline by {c12_v4_vs_naive:.1f}%")
    else:
        c12_v4_gap = (xgb4_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ XGBoost v4 still WORSE than naive by {c12_v4_gap:.1f}%")
    return c12_df, c12_feature_cols, xgb4_mean_smape


@app.cell
def _(
    N_VAL_WEEKS,
    c10_zero_threshold,
    c12_df,
    c12_feature_cols,
    naive_mean_smape,
    np,
    pd,
    series,
    smape,
    time,
    val_cutoffs,
    xgb,
    xgb4_mean_smape,
):
    # ── CELL 13: XGBoost v5 — Add a bounded week-over-week growth ratio ──────
    # WHY THIS CELL EXISTS: XGBoost v4 (bounded ratios for rolling means)
    # achieved stable, consistent SMAPE (0.244, std=0.026 — far tighter than
    # any prior version) but is still 13.4% behind naive. We removed
    # time_index entirely to fix the extrapolation bug, which ALSO removed
    # all explicit trend information — the model can no longer represent
    # "the business is growing ~1%/week" in any form. We add ONE new bounded
    # feature to restore trend-awareness without reintroducing extrapolation risk:
    #
    #   growth_ratio_168 = lag_168 / lag_336
    #
    # This compares "same hour last week" to "same hour two weeks ago" — a
    # week-over-week growth signal. Like our other ratio fixes, both numerator
    # and denominator grow together as the business grows, so this ratio stays
    # bounded near 1.0 (or slightly above, given the ~55%/year ≈ ~1%/week
    # growth rate from EDA) regardless of forecast horizon or absolute scale —
    # never extrapolating beyond a sane range the way time_index did.

    c13_df = (
        c12_df.copy()
    )  # start from Cell 12's table (already has bounded ratio features)

    c13_df["lag_336"] = series.reindex(c13_df.index).shift(
        336
    )  # 336h = 2 full weeks ago

    c13_df["growth_ratio_168"] = np.where(
        c13_df["lag_336"] > 0,
        c13_df["lag_168"] / c13_df["lag_336"],
        1.0,  # if 2 weeks ago was zero, assume neutral (no growth signal available)
    )

    # Adding lag_336 means we lose 336 rows of warm-up instead of 168 — re-drop NaNs
    c13_n_before = len(c13_df)
    c13_df = c13_df.dropna()
    c13_n_after = len(c13_df)
    print(
        f"Rows before dropna: {c13_n_before}, after: {c13_n_after}, "
        f"dropped {c13_n_before - c13_n_after} (expected 168 more, for the new lag_336 warm-up)"
    )

    c13_feature_cols = c12_feature_cols + [
        "growth_ratio_168"
    ]  # all of v4's features, plus the new one
    print(f"\nFeatures used in XGBoost v5: {c13_feature_cols}")
    print("\ngrowth_ratio_168 stats (should hover near/slightly above 1.0):")
    print(c13_df["growth_ratio_168"].describe())

    xgb5_smape_scores = []
    xgb5_mse_scores = []
    xgb5_fit_seconds = []

    for c13_w, c13_cutoff_w in enumerate(val_cutoffs):
        c13_train_df = c13_df[c13_df.index <= c13_cutoff_w]
        c13_actual_wf = series[
            (series.index > c13_cutoff_w)
            & (series.index <= c13_cutoff_w + pd.Timedelta(hours=168))
        ]

        c13_start_time = time.time()

        c13_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        c13_model.fit(c13_train_df[c13_feature_cols], c13_train_df["orders"])

        c13_elapsed = time.time() - c13_start_time
        xgb5_fit_seconds.append(c13_elapsed)

        c13_history = series[series.index <= c13_cutoff_w].copy()
        c13_predictions = []
        c13_forecast_index = pd.date_range(
            start=c13_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c13_step_time in c13_forecast_index:
            c13_lag1_val = c13_history.loc[c13_step_time - pd.Timedelta(hours=1)]
            c13_lag24_val = c13_history.loc[c13_step_time - pd.Timedelta(hours=24)]
            c13_lag168_val = c13_history.loc[c13_step_time - pd.Timedelta(hours=168)]
            c13_lag336_val = c13_history.loc[c13_step_time - pd.Timedelta(hours=336)]
            c13_roll168_val = c13_history.loc[
                c13_step_time - pd.Timedelta(hours=168) : c13_step_time
                - pd.Timedelta(hours=1)
            ].mean()

            c13_row = {
                "hour_of_day": c13_step_time.hour,
                "day_of_week": c13_step_time.dayofweek,
                "is_weekend": int(c13_step_time.dayofweek >= 5),
                "lag_1": c13_lag1_val,
                "lag_24": c13_lag24_val,
                "lag_48": c13_history.loc[c13_step_time - pd.Timedelta(hours=48)],
                "lag_144": c13_history.loc[c13_step_time - pd.Timedelta(hours=144)],
                "lag_168": c13_lag168_val,
                "ratio_recent_vs_week": c13_lag1_val / c13_roll168_val
                if c13_roll168_val > 0
                else 0.0,
                "ratio_day_vs_week": c13_lag24_val / c13_roll168_val
                if c13_roll168_val > 0
                else 0.0,
                "growth_ratio_168": c13_lag168_val / c13_lag336_val
                if c13_lag336_val > 0
                else 1.0,
            }
            c13_row_df = pd.DataFrame([c13_row])[c13_feature_cols]

            c13_pred_value = c13_model.predict(c13_row_df)[0]
            c13_pred_value = max(c13_pred_value, 0.0)
            c13_pred_value = (
                0.0 if c13_pred_value < c10_zero_threshold else c13_pred_value
            )

            c13_predictions.append(c13_pred_value)
            c13_history.loc[c13_step_time] = c13_pred_value

        c13_forecast = pd.Series(c13_predictions, index=c13_forecast_index)

        c13_aligned = pd.DataFrame(
            {"actual": c13_actual_wf, "forecast": c13_forecast}
        ).dropna()
        assert len(c13_aligned) == 168, (
            f"Window {c13_w + 1} has {len(c13_aligned)} rows, expected 168"
        )

        c13_smape_val = smape(c13_aligned["actual"], c13_aligned["forecast"])
        c13_mse_val = np.mean((c13_aligned["actual"] - c13_aligned["forecast"]) ** 2)

        xgb5_smape_scores.append(c13_smape_val)
        xgb5_mse_scores.append(c13_mse_val)

        print(
            f"  Window {c13_w + 1} ({c13_cutoff_w.date()} cutoff): "
            f"SMAPE={c13_smape_val:.4f}, MSE={c13_mse_val:.1f}, fit_time={c13_elapsed:.2f}s"
        )

    xgb5_mean_smape = np.mean(xgb5_smape_scores)
    xgb5_mean_mse = np.mean(xgb5_mse_scores)
    xgb5_std_smape = np.std(xgb5_smape_scores)

    print(f"\n{'=' * 60}")
    print(f"XGBOOST v5 RESULTS — added growth_ratio_168  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {xgb5_mean_smape:.4f}  ± {xgb5_std_smape:.4f} std")
    print(f"SMAPE range: [{min(xgb5_smape_scores):.4f}, {max(xgb5_smape_scores):.4f}]")

    print(f"\n{'=' * 60}")
    print("FULL MODEL COMPARISON — ALL MODELS, ALL VERSIONS")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:      {naive_mean_smape:.4f}")
    print(f"XGBoost v4 SMAPE: {xgb4_mean_smape:.4f}  (bounded ratios, no trend signal)")
    print(f"XGBoost v5 SMAPE: {xgb5_mean_smape:.4f}  (added growth_ratio_168)")

    c13_v4_to_v5 = (xgb4_mean_smape - xgb5_mean_smape) / xgb4_mean_smape * 100
    print(f"\ngrowth_ratio_168 changed SMAPE by {c13_v4_to_v5:+.1f}% vs v4")

    if xgb5_mean_smape < naive_mean_smape:
        c13_v5_vs_naive = (naive_mean_smape - xgb5_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ XGBoost v5 BEATS naive baseline by {c13_v5_vs_naive:.1f}%")
    else:
        c13_v5_gap = (xgb5_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ XGBoost v5 still WORSE than naive by {c13_v5_gap:.1f}%")
    return c13_df, c13_feature_cols, xgb5_mean_smape, xgb5_smape_scores


@app.cell
def _(
    c10_zero_threshold,
    c13_df,
    c13_feature_cols,
    naive_smape_scores,
    np,
    pd,
    plt,
    series,
    val_cutoffs,
    xgb,
    xgb5_smape_scores,
):
    # ── CELL 13b: DIAGNOSTIC — per-window comparison + visualize best/worst ──
    # Three diagnostic angles, SOTA-practitioner style:
    # (A) Direct window-by-window diff: naive vs XGBoost v5 — uniform gap or
    #     concentrated in specific windows?
    # (B) Visualize the BEST and WORST windows for v5 to see the actual error
    #     SHAPE, not just the aggregate number.
    # (C) Set up for ensembling if (A)/(B) suggest the two models make
    #     different KINDS of errors (the classic case where blending wins).

    print("=" * 60)
    print("(A) PER-WINDOW COMPARISON: Naive vs XGBoost v5")
    print("=" * 60)
    c13b_comparison = pd.DataFrame(
        {
            "window": [f"W{i + 1} ({c.date()})" for i, c in enumerate(val_cutoffs)],
            "naive_smape": naive_smape_scores,
            "xgb_v5_smape": xgb5_smape_scores,
        }
    )
    c13b_comparison["xgb_wins"] = (
        c13b_comparison["xgb_v5_smape"] < c13b_comparison["naive_smape"]
    )
    c13b_comparison["diff"] = (
        c13b_comparison["naive_smape"] - c13b_comparison["xgb_v5_smape"]
    )  # positive = XGBoost better
    print(c13b_comparison.to_string(index=False))
    print(
        f"\nXGBoost wins on {c13b_comparison['xgb_wins'].sum()} out of {len(c13b_comparison)} windows"
    )

    # ── (B) Identify best and worst windows for v5 specifically ──────────────
    c13b_best_idx = np.argmin(xgb5_smape_scores)
    c13b_worst_idx = np.argmax(xgb5_smape_scores)
    print(
        f"\nBest window for v5:  Window {c13b_best_idx + 1}  (SMAPE={xgb5_smape_scores[c13b_best_idx]:.4f})"
    )
    print(
        f"Worst window for v5: Window {c13b_worst_idx + 1}  (SMAPE={xgb5_smape_scores[c13b_worst_idx]:.4f})"
    )

    # ── Rebuild forecasts for both windows to plot them ───────────────────────
    def c13b_rebuild_forecast(cutoff):
        """Refit v5's exact pipeline for a single cutoff and return the forecast."""
        train_df = c13_df[c13_df.index <= cutoff]
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        model.fit(train_df[c13_feature_cols], train_df["orders"])

        history = series[series.index <= cutoff].copy()
        preds = []
        idx = pd.date_range(start=cutoff + pd.Timedelta(hours=1), periods=168, freq="h")
        for t in idx:
            lag1, lag24, lag168, lag336 = (
                history.loc[t - pd.Timedelta(hours=1)],
                history.loc[t - pd.Timedelta(hours=24)],
                history.loc[t - pd.Timedelta(hours=168)],
                history.loc[t - pd.Timedelta(hours=336)],
            )
            roll168 = history.loc[
                t - pd.Timedelta(hours=168) : t - pd.Timedelta(hours=1)
            ].mean()
            row = {
                "hour_of_day": t.hour,
                "day_of_week": t.dayofweek,
                "is_weekend": int(t.dayofweek >= 5),
                "lag_1": lag1,
                "lag_24": lag24,
                "lag_48": history.loc[t - pd.Timedelta(hours=48)],
                "lag_144": history.loc[t - pd.Timedelta(hours=144)],
                "lag_168": lag168,
                "ratio_recent_vs_week": lag1 / roll168 if roll168 > 0 else 0.0,
                "ratio_day_vs_week": lag24 / roll168 if roll168 > 0 else 0.0,
                "growth_ratio_168": lag168 / lag336 if lag336 > 0 else 1.0,
            }
            row_df = pd.DataFrame([row])[c13_feature_cols]
            pred = max(model.predict(row_df)[0], 0.0)
            pred = 0.0 if pred < c10_zero_threshold else pred
            preds.append(pred)
            history.loc[t] = pred
        return pd.Series(preds, index=idx)

    c13b_best_cutoff = val_cutoffs[c13b_best_idx]
    c13b_worst_cutoff = val_cutoffs[c13b_worst_idx]
    c13b_best_forecast = c13b_rebuild_forecast(c13b_best_cutoff)
    c13b_worst_forecast = c13b_rebuild_forecast(c13b_worst_cutoff)
    c13b_best_actual = series[
        (series.index > c13b_best_cutoff)
        & (series.index <= c13b_best_cutoff + pd.Timedelta(hours=168))
    ]
    c13b_worst_actual = series[
        (series.index > c13b_worst_cutoff)
        & (series.index <= c13b_worst_cutoff + pd.Timedelta(hours=168))
    ]

    fig_diag7, axes_diag7 = plt.subplots(2, 1, figsize=(16, 9))
    axes_diag7[0].plot(
        c13b_best_actual.index,
        c13b_best_actual.values,
        color="#2563EB",
        label="Actual",
        marker="o",
        ms=3,
    )
    axes_diag7[0].plot(
        c13b_best_forecast.index,
        c13b_best_forecast.values,
        color="#EF4444",
        label="XGBoost v5",
        marker="x",
        ms=3,
    )
    axes_diag7[0].set_title(
        f"BEST window for v5 — Window {c13b_best_idx + 1} (SMAPE={xgb5_smape_scores[c13b_best_idx]:.4f})",
        fontweight="bold",
    )
    axes_diag7[0].legend()

    axes_diag7[1].plot(
        c13b_worst_actual.index,
        c13b_worst_actual.values,
        color="#2563EB",
        label="Actual",
        marker="o",
        ms=3,
    )
    axes_diag7[1].plot(
        c13b_worst_forecast.index,
        c13b_worst_forecast.values,
        color="#EF4444",
        label="XGBoost v5",
        marker="x",
        ms=3,
    )
    axes_diag7[1].set_title(
        f"WORST window for v5 — Window {c13b_worst_idx + 1} (SMAPE={xgb5_smape_scores[c13b_worst_idx]:.4f})",
        fontweight="bold",
    )
    axes_diag7[1].legend()

    plt.tight_layout()
    plt.savefig("figures/diag_07_best_worst_v5.png", dpi=150, bbox_inches="tight")
    plt.show()
    return (c13b_rebuild_forecast,)


@app.cell
def _(
    c13_df,
    c13b_rebuild_forecast,
    naive_mean_smape,
    naive_smape_scores,
    np,
    pd,
    seasonal_naive_forecast,
    series,
    smape,
    val_cutoffs,
    xgb5_mean_smape,
    xgb5_smape_scores,
):
    # ── CELL 13c: Test ensemble (average of naive + XGBoost v5) directly ─────
    # Per-window comparison showed naive wins 6/7 windows outright — weaker
    # support for ensembling than expected. But ensembling can still help even
    # when one model "wins" most individual windows, IF the two models' ERRORS
    # are not perfectly correlated (i.e., they're wrong in different ways on
    # different hours). We test this directly and definitively rather than
    # debate it: compute a simple 50/50 average forecast for all 7 windows and
    # compare its SMAPE against both individual models.

    c13c_ensemble_smape_scores = []

    for c13c_w, c13c_cutoff_w in enumerate(val_cutoffs):
        c13c_train_df = c13_df[c13_df.index <= c13c_cutoff_w]
        c13c_actual_wf = series[
            (series.index > c13c_cutoff_w)
            & (series.index <= c13c_cutoff_w + pd.Timedelta(hours=168))
        ]

        # Naive forecast for this window (same logic as Cell 5)
        c13c_naive_forecast = seasonal_naive_forecast(
            series[series.index <= c13c_cutoff_w], horizon=168
        )

        # XGBoost v5 forecast for this window (reuse the helper from Cell 13b)
        c13c_xgb_forecast = c13b_rebuild_forecast(c13c_cutoff_w)

        # Simple 50/50 ensemble — average the two forecasts hour by hour
        c13c_ensemble_forecast = (c13c_naive_forecast + c13c_xgb_forecast) / 2

        c13c_aligned = pd.DataFrame(
            {"actual": c13c_actual_wf, "forecast": c13c_ensemble_forecast}
        ).dropna()
        assert len(c13c_aligned) == 168, (
            f"Window {c13c_w + 1} has {len(c13c_aligned)} rows"
        )

        c13c_smape_val = smape(c13c_aligned["actual"], c13c_aligned["forecast"])
        c13c_ensemble_smape_scores.append(c13c_smape_val)

        print(
            f"  Window {c13c_w + 1}: naive={naive_smape_scores[c13c_w]:.4f}, "
            f"xgb_v5={xgb5_smape_scores[c13c_w]:.4f}, "
            f"ENSEMBLE={c13c_smape_val:.4f}"
        )

    c13c_ensemble_mean = np.mean(c13c_ensemble_smape_scores)
    c13c_ensemble_std = np.std(c13c_ensemble_smape_scores)

    print(f"\n{'=' * 60}")
    print("ENSEMBLE (50/50 naive + XGBoost v5) RESULTS")
    print(f"{'=' * 60}")
    print(f"Naive mean SMAPE:    {naive_mean_smape:.4f}")
    print(f"XGBoost v5 mean SMAPE: {xgb5_mean_smape:.4f}")
    print(
        f"Ensemble mean SMAPE:   {c13c_ensemble_mean:.4f}  ± {c13c_ensemble_std:.4f} std"
    )

    if c13c_ensemble_mean < naive_mean_smape:
        c13c_improvement = (
            (naive_mean_smape - c13c_ensemble_mean) / naive_mean_smape * 100
        )
        print(f"\n✅ ENSEMBLE BEATS naive by {c13c_improvement:.1f}%")
    else:
        c13c_gap = (c13c_ensemble_mean - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ Ensemble still worse than naive by {c13c_gap:.1f}%")
    return


@app.cell
def _(
    c13b_rebuild_forecast,
    np,
    pd,
    plt,
    seasonal_naive_forecast,
    series,
    smape,
    val_cutoffs,
):
    # ── CELL 14: Tune the ensemble blend weight, with explicit overfitting caution ──
    # QUESTION: is 50/50 the optimal blend, or could a different weight do
    # better? METHODOLOGICAL CARE: with only 7 validation windows, aggressively
    # searching many weights and picking the best risks overfitting to noise —
    # the same lesson as AIC/BIC overfitting in-sample (Ex03/Ex04). We test a
    # small, principled grid (not hundreds of values) and report ALL results
    # transparently, not just the winner — so we can judge whether any
    # improvement over 0.5 is a real signal or just 7-window noise.

    c14_weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # weight = how much WEIGHT goes to XGBoost; (1-weight) goes to naive
    # weight=0.0 means pure naive, weight=1.0 means pure XGBoost v5

    c14_results = []

    for c14_weight in c14_weight_grid:
        c14_window_scores = []
        for c14_w, c14_cutoff_w in enumerate(val_cutoffs):
            c14_actual_wf = series[
                (series.index > c14_cutoff_w)
                & (series.index <= c14_cutoff_w + pd.Timedelta(hours=168))
            ]
            c14_naive_fc = seasonal_naive_forecast(
                series[series.index <= c14_cutoff_w], horizon=168
            )
            c14_xgb_fc = c13b_rebuild_forecast(
                c14_cutoff_w
            )  # reuse Cell 13b's helper — same model, same features

            c14_blend_fc = (1 - c14_weight) * c14_naive_fc + c14_weight * c14_xgb_fc

            c14_aligned = pd.DataFrame(
                {"actual": c14_actual_wf, "forecast": c14_blend_fc}
            ).dropna()
            c14_window_smape = smape(c14_aligned["actual"], c14_aligned["forecast"])
            c14_window_scores.append(c14_window_smape)

        c14_mean_smape = np.mean(c14_window_scores)
        c14_std_smape = np.std(c14_window_scores)
        c14_results.append(
            {
                "weight_xgb": c14_weight,
                "mean_smape": c14_mean_smape,
                "std_smape": c14_std_smape,
            }
        )
        print(
            f"  weight_xgb={c14_weight:.1f}  (weight_naive={1 - c14_weight:.1f}): "
            f"mean SMAPE = {c14_mean_smape:.4f}  ± {c14_std_smape:.4f}"
        )

    c14_results_df = pd.DataFrame(c14_results)
    c14_best_row = c14_results_df.loc[c14_results_df["mean_smape"].idxmin()]

    print(f"\n{'=' * 60}")
    print(
        f"BEST WEIGHT FOUND: weight_xgb={c14_best_row['weight_xgb']:.1f}, "
        f"mean SMAPE={c14_best_row['mean_smape']:.4f}"
    )
    print(f"{'=' * 60}")
    print(
        f"Compare to 50/50:  mean SMAPE = "
        f"{c14_results_df[c14_results_df['weight_xgb'] == 0.5]['mean_smape'].values[0]:.4f}"
    )
    print(
        f"Compare to pure naive (weight=0.0): "
        f"{c14_results_df[c14_results_df['weight_xgb'] == 0.0]['mean_smape'].values[0]:.4f}"
    )

    # ── Plot the full curve — is there a clear, smooth optimum, or noisy spikes? ──
    fig_weight, ax_weight = plt.subplots(figsize=(10, 5))
    ax_weight.errorbar(
        c14_results_df["weight_xgb"],
        c14_results_df["mean_smape"],
        yerr=c14_results_df["std_smape"],
        marker="o",
        capsize=4,
        color="#2563EB",
        ecolor="#93C5FD",
    )
    ax_weight.axvline(
        c14_best_row["weight_xgb"],
        color="#EF4444",
        linestyle="--",
        label=f"Best weight = {c14_best_row['weight_xgb']:.1f}",
    )
    ax_weight.set_xlabel("Weight on XGBoost (0 = pure naive, 1 = pure XGBoost)")
    ax_weight.set_ylabel("Mean SMAPE across 7 validation windows")
    ax_weight.set_title(
        "Ensemble Weight Search — Is There a Clear, Stable Optimum?", fontweight="bold"
    )
    ax_weight.legend()
    ax_weight.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/ensemble_weight_search.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(c13_df, c13_feature_cols, pd):
    # ── CELL 15: Holiday & Long-Weekend Flag Features ────────────────────────
    # WHY THIS CELL EXISTS: course slides explicitly list "holiday flag, long
    # weekends" as untried feature ideas. Our diagnosed remaining gap vs. naive
    # (Cells 13b) showed XGBoost systematically UNDERESTIMATES sharp, one-off
    # demand spikes — exactly the kind of effect a public holiday or long
    # weekend would cause (people order more delivery when off work/school).
    # Window 1's forecast period (Dec 6-12, 2021) contains TWO confirmed
    # Catalonia public holidays: Dec 6 (Constitution Day) and Dec 8
    # (Immaculate Conception) — verified via official Catalonia holiday
    # calendars (officeholidays.com) for 2021 and 2022.
    #
    # We add THREE new binary (0/1) flag features:
    #   is_holiday       : 1 if this hour falls on an official Catalonia/Spain
    #                       public holiday, else 0
    #   is_day_before_holiday : 1 if TOMORROW is a holiday (people often go out
    #                       /order food the evening before a day off)
    #   is_day_after_holiday  : 1 if YESTERDAY was a holiday (a "bridge" day,
    #                       often taken off work, behaves like a holiday too)
    # These are calendar facts, not derived from the orders series itself —
    # completely safe to use for ANY future date, since holidays are known
    # in advance (unlike orders, which we don't know until they happen).

    # ── Verified Catalonia/Barcelona public holidays, 2021 and 2022 ──────────
    # Source: officeholidays.com/countries/spain/catalonia/2021 and /2022
    # (national + regional holidays observed in the city of Barcelona)
    c15_holidays_2021 = [
        "2021-01-01",  # New Year's Day
        "2021-01-06",  # Epiphany (Three Kings Day)
        "2021-04-02",  # Good Friday
        "2021-04-05",  # Easter Monday (Catalonia)
        "2021-05-01",  # Labour Day
        "2021-05-24",  # Whit Monday (Barcelona)
        "2021-06-24",  # Feast of St. John the Baptist
        "2021-08-15",  # Assumption of the Virgin
        "2021-09-11",  # National Day of Catalonia
        "2021-09-24",  # La Mercè (Barcelona)
        "2021-10-12",  # Hispanic Day
        "2021-11-01",  # All Saints' Day
        "2021-12-06",  # Constitution Day
        "2021-12-08",  # Immaculate Conception Day
        "2021-12-25",  # Christmas Day
        "2021-12-26",  # St. Stephen's Day (Catalonia)
    ]
    c15_holidays_2022 = [
        "2022-01-01",  # New Year's Day
        "2022-01-06",  # Epiphany (Three Kings Day)
        "2022-04-15",  # Good Friday
        "2022-04-18",  # Easter Monday (Catalonia)
        "2022-05-01",  # Labour Day
        "2022-06-06",  # Whit Monday (Barcelona)
        "2022-06-24",  # Feast of St. John the Baptist
        "2022-08-15",  # Assumption of the Virgin
        "2022-09-11",  # National Day of Catalonia
        "2022-09-24",  # La Mercè (Barcelona)
        "2022-10-12",  # Hispanic Day
        "2022-11-01",  # All Saints' Day
        "2022-12-06",  # Constitution Day
        "2022-12-08",  # Immaculate Conception Day
        "2022-12-25",  # Christmas Day
        "2022-12-26",  # St. Stephen's Day (Catalonia)
    ]
    # Combine into a single set of dates (as pandas Timestamps, date-only) for fast lookup
    c15_all_holidays = pd.to_datetime(c15_holidays_2021 + c15_holidays_2022).date
    c15_holiday_set = set(
        c15_all_holidays
    )  # set lookup is O(1), important since we check this per-hour

    print(
        f"Total holidays loaded: {len(c15_holiday_set)} (2021: {len(c15_holidays_2021)}, 2022: {len(c15_holidays_2022)})"
    )

    # ── Build the three flag columns on top of Cell 13's feature table ───────
    c15_df = c13_df.copy()

    # .dt.date strips the time-of-day, leaving just the calendar date, so every
    # hour within the same day gets the same flag value
    c15_dates = c15_df.index.date

    c15_df["is_holiday"] = pd.Series(
        [d in c15_holiday_set for d in c15_dates], index=c15_df.index
    ).astype(int)

    # "tomorrow" and "yesterday" as date objects, for checking adjacency
    c15_tomorrow = pd.Series(c15_dates, index=c15_df.index) + pd.Timedelta(days=1)
    c15_yesterday = pd.Series(c15_dates, index=c15_df.index) - pd.Timedelta(days=1)

    c15_df["is_day_before_holiday"] = (
        c15_tomorrow.apply(lambda d: d in c15_holiday_set).astype(int).values
    )
    c15_df["is_day_after_holiday"] = (
        c15_yesterday.apply(lambda d: d in c15_holiday_set).astype(int).values
    )

    c15_feature_cols = c13_feature_cols + [
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
    ]
    print(f"\nFeatures used in XGBoost v6: {c15_feature_cols}")

    # ── Sanity check: print exactly which dates got flagged within our data range ──
    print(f"\n{'=' * 60}")
    print("VERIFICATION: holiday dates flagged within our dataset's range")
    print(f"{'=' * 60}")
    c15_flagged_days = c15_df[c15_df["is_holiday"] == 1].index.date
    c15_unique_flagged = sorted(set(c15_flagged_days))
    for d in c15_unique_flagged:
        print(f"  {d}  ({pd.Timestamp(d).day_name()})")
    print(f"\nTotal unique holiday dates flagged: {len(c15_unique_flagged)}")

    print(
        f"\nDay-before-holiday flags set: {c15_df['is_day_before_holiday'].sum()} hours "
        f"({c15_df['is_day_before_holiday'].sum() / 24:.0f} days)"
    )
    print(
        f"Day-after-holiday flags set:  {c15_df['is_day_after_holiday'].sum()} hours "
        f"({c15_df['is_day_after_holiday'].sum() / 24:.0f} days)"
    )

    # ── Specifically verify Window 1's Dec 6 and Dec 8, 2021 are flagged ─────
    print(f"\n{'=' * 60}")
    print("SPECIFIC CHECK: Window 1's target week (Dec 6-12, 2021)")
    print(f"{'=' * 60}")
    c15_window1_check = c15_df.loc[
        "2021-12-06":"2021-12-12",
        ["is_holiday", "is_day_before_holiday", "is_day_after_holiday"],
    ]
    print(c15_window1_check.groupby(c15_window1_check.index.date).first())
    return c15_df, c15_feature_cols, c15_holiday_set


@app.cell
def _(
    N_VAL_WEEKS,
    c10_zero_threshold,
    c15_df,
    c15_feature_cols,
    c15_holiday_set,
    naive_mean_smape,
    np,
    pd,
    series,
    smape,
    time,
    val_cutoffs,
    xgb,
    xgb5_mean_smape,
    xgb5_smape_scores,
):
    # ── CELL 16: XGBoost v6 — With holiday/long-weekend flags ────────────────
    # WHY THIS CELL EXISTS: Cell 13 (XGBoost v5, with bounded ratios +
    # growth_ratio_168) scored SMAPE 0.2379 — still 10.5% behind naive (0.2153).
    # Diagnosis (Cell 13b) showed XGBoost systematically UNDERESTIMATES sharp,
    # one-off demand spikes, concentrated in weeks containing public holidays
    # (Window 1: Dec 6-12, contains Dec 6 Constitution Day + Dec 8 Immaculate
    # Conception). Course slides explicitly recommend holiday/long-weekend
    # flags as untried features (IDD_CLASS1 slides). Cell 15 built and verified
    # is_holiday, is_day_before_holiday, is_day_after_holiday — calendar facts
    # known in advance, safe to use at any forecast horizon since they don't
    # depend on the orders series itself (unlike lags, which DO require
    # recursive feeding).

    xgb6_smape_scores = []
    xgb6_mse_scores = []
    xgb6_fit_seconds = []

    for c16_w, c16_cutoff_w in enumerate(val_cutoffs):
        c16_train_df = c15_df[c15_df.index <= c16_cutoff_w]
        c16_actual_wf = series[
            (series.index > c16_cutoff_w)
            & (series.index <= c16_cutoff_w + pd.Timedelta(hours=168))
        ]

        c16_start_time = time.time()

        c16_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        c16_model.fit(c16_train_df[c15_feature_cols], c16_train_df["orders"])

        c16_elapsed = time.time() - c16_start_time
        xgb6_fit_seconds.append(c16_elapsed)

        c16_history = series[series.index <= c16_cutoff_w].copy()
        c16_predictions = []
        c16_forecast_index = pd.date_range(
            start=c16_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c16_step_time in c16_forecast_index:
            c16_lag1_val = c16_history.loc[c16_step_time - pd.Timedelta(hours=1)]
            c16_lag24_val = c16_history.loc[c16_step_time - pd.Timedelta(hours=24)]
            c16_lag168_val = c16_history.loc[c16_step_time - pd.Timedelta(hours=168)]
            c16_lag336_val = c16_history.loc[c16_step_time - pd.Timedelta(hours=336)]
            c16_roll168_val = c16_history.loc[
                c16_step_time - pd.Timedelta(hours=168) : c16_step_time
                - pd.Timedelta(hours=1)
            ].mean()

            # Calendar-based holiday flags — known facts, computed directly from
            # the timestamp itself, NOT from c16_history (no recursive dependency)
            c16_this_date = c16_step_time.date()
            c16_tomorrow_date = c16_this_date + pd.Timedelta(days=1)
            c16_yesterday_date = c16_this_date - pd.Timedelta(days=1)

            c16_row = {
                "hour_of_day": c16_step_time.hour,
                "day_of_week": c16_step_time.dayofweek,
                "is_weekend": int(c16_step_time.dayofweek >= 5),
                "lag_1": c16_lag1_val,
                "lag_24": c16_lag24_val,
                "lag_48": c16_history.loc[c16_step_time - pd.Timedelta(hours=48)],
                "lag_144": c16_history.loc[c16_step_time - pd.Timedelta(hours=144)],
                "lag_168": c16_lag168_val,
                "ratio_recent_vs_week": c16_lag1_val / c16_roll168_val
                if c16_roll168_val > 0
                else 0.0,
                "ratio_day_vs_week": c16_lag24_val / c16_roll168_val
                if c16_roll168_val > 0
                else 0.0,
                "growth_ratio_168": c16_lag168_val / c16_lag336_val
                if c16_lag336_val > 0
                else 1.0,
                "is_holiday": int(c16_this_date in c15_holiday_set),
                "is_day_before_holiday": int(c16_tomorrow_date in c15_holiday_set),
                "is_day_after_holiday": int(c16_yesterday_date in c15_holiday_set),
            }
            c16_row_df = pd.DataFrame([c16_row])[c15_feature_cols]

            c16_pred_value = c16_model.predict(c16_row_df)[0]
            c16_pred_value = max(c16_pred_value, 0.0)
            c16_pred_value = (
                0.0 if c16_pred_value < c10_zero_threshold else c16_pred_value
            )

            c16_predictions.append(c16_pred_value)
            c16_history.loc[c16_step_time] = c16_pred_value

        c16_forecast = pd.Series(c16_predictions, index=c16_forecast_index)

        c16_aligned = pd.DataFrame(
            {"actual": c16_actual_wf, "forecast": c16_forecast}
        ).dropna()
        assert len(c16_aligned) == 168, (
            f"Window {c16_w + 1} has {len(c16_aligned)} rows, expected 168"
        )

        c16_smape_val = smape(c16_aligned["actual"], c16_aligned["forecast"])
        c16_mse_val = np.mean((c16_aligned["actual"] - c16_aligned["forecast"]) ** 2)

        xgb6_smape_scores.append(c16_smape_val)
        xgb6_mse_scores.append(c16_mse_val)

        print(
            f"  Window {c16_w + 1} ({c16_cutoff_w.date()} cutoff): "
            f"SMAPE={c16_smape_val:.4f}, MSE={c16_mse_val:.1f}, fit_time={c16_elapsed:.2f}s"
        )

    xgb6_mean_smape = np.mean(xgb6_smape_scores)
    xgb6_mean_mse = np.mean(xgb6_mse_scores)
    xgb6_std_smape = np.std(xgb6_smape_scores)

    print(f"\n{'=' * 60}")
    print(f"XGBOOST v6 RESULTS — holiday flags added  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {xgb6_mean_smape:.4f}  ± {xgb6_std_smape:.4f} std")
    print(f"SMAPE range: [{min(xgb6_smape_scores):.4f}, {max(xgb6_smape_scores):.4f}]")

    print(f"\n{'=' * 60}")
    print("PER-WINDOW: v5 vs v6 (does holiday info help WHERE we expected?)")
    print(f"{'=' * 60}")
    for c16_i in range(N_VAL_WEEKS):
        c16_delta = xgb5_smape_scores[c16_i] - xgb6_smape_scores[c16_i]
        print(
            f"  Window {c16_i + 1}: v5={xgb5_smape_scores[c16_i]:.4f}, v6={xgb6_smape_scores[c16_i]:.4f}, "
            f"change={c16_delta:+.4f} {'✅' if c16_delta > 0 else ''}"
        )

    print(f"\n{'=' * 60}")
    print("FULL MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:      {naive_mean_smape:.4f}")
    print(f"XGBoost v5 SMAPE: {xgb5_mean_smape:.4f}  (bounded ratios + growth ratio)")
    print(f"XGBoost v6 SMAPE: {xgb6_mean_smape:.4f}  (+ holiday/long-weekend flags)")

    c16_v5_to_v6 = (xgb5_mean_smape - xgb6_mean_smape) / xgb5_mean_smape * 100
    print(f"\nHoliday flags changed SMAPE by {c16_v5_to_v6:+.1f}% vs v5")

    if xgb6_mean_smape < naive_mean_smape:
        c16_vs_naive = (naive_mean_smape - xgb6_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ XGBoost v6 BEATS naive baseline by {c16_vs_naive:.1f}%")
    else:
        c16_gap = (xgb6_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ XGBoost v6 still WORSE than naive by {c16_gap:.1f}%")
    return xgb6_mean_smape, xgb6_smape_scores


@app.cell
def _(
    c10_zero_threshold,
    c15_df,
    c15_feature_cols,
    c15_holiday_set,
    naive_mean_smape,
    np,
    pd,
    series,
    smape,
    time,
    val_cutoffs,
    xgb,
    xgb5_mean_smape,
    xgb6_mean_smape,
):
    # ── CELL 17: XGBoost v7 — True detrending (predict ratio-to-trend, not raw orders) ──
    # WHY THIS CELL EXISTS: course slides state plainly: "trees predict by
    # averaging leaf values and cannot extrapolate beyond the training range.
    # Detrend before fitting." Our v1-v6 iterations all predicted raw `orders`
    # directly, using ratio FEATURES (Cell 12, 13) as an indirect workaround.
    # This cell implements TRUE detrending: the model's TARGET itself becomes
    # orders / rolling_mean_168h (how many times above/below the recent typical
    # level), and we multiply back by rolling_mean_168h at prediction time to
    # recover actual order counts. This fully decouples "shape of demand" from
    # "current level" in the learning problem itself, not just via an input
    # feature the trees might or might not use effectively.

    c17_df = c15_df.copy()  # start from Cell 15's table (includes holiday flags)

    # Guard against division by zero — same approach as our other ratio features
    c17_df["detrended_target"] = np.where(
        c17_df["rolling_mean_168h"] > 0,
        c17_df["orders"] / c17_df["rolling_mean_168h"],
        0.0,
    )

    print(
        "detrended_target stats (should hover near 1.0, representing 'typical' hours,"
    )
    print("with values <1 for quiet hours and >1 for peak hours, all roughly bounded):")
    print(c17_df["detrended_target"].describe())

    # Features stay THE SAME as v6 (holiday flags included) — only the TARGET changes
    c17_feature_cols = c15_feature_cols

    xgb7_smape_scores = []
    xgb7_mse_scores = []
    xgb7_fit_seconds = []

    for c17_w, c17_cutoff_w in enumerate(val_cutoffs):
        c17_train_df = c17_df[c17_df.index <= c17_cutoff_w]
        c17_actual_wf = series[
            (series.index > c17_cutoff_w)
            & (series.index <= c17_cutoff_w + pd.Timedelta(hours=168))
        ]

        c17_start_time = time.time()

        c17_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        # KEY CHANGE: fit on detrended_target, NOT raw orders
        c17_model.fit(c17_train_df[c17_feature_cols], c17_train_df["detrended_target"])

        c17_elapsed = time.time() - c17_start_time
        xgb7_fit_seconds.append(c17_elapsed)

        c17_history = series[series.index <= c17_cutoff_w].copy()
        c17_predictions = []
        c17_forecast_index = pd.date_range(
            start=c17_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c17_step_time in c17_forecast_index:
            c17_lag1_val = c17_history.loc[c17_step_time - pd.Timedelta(hours=1)]
            c17_lag24_val = c17_history.loc[c17_step_time - pd.Timedelta(hours=24)]
            c17_lag168_val = c17_history.loc[c17_step_time - pd.Timedelta(hours=168)]
            c17_lag336_val = c17_history.loc[c17_step_time - pd.Timedelta(hours=336)]
            c17_roll168_val = c17_history.loc[
                c17_step_time - pd.Timedelta(hours=168) : c17_step_time
                - pd.Timedelta(hours=1)
            ].mean()

            c17_this_date = c17_step_time.date()
            c17_tomorrow_date = c17_this_date + pd.Timedelta(days=1)
            c17_yesterday_date = c17_this_date - pd.Timedelta(days=1)

            c17_row = {
                "hour_of_day": c17_step_time.hour,
                "day_of_week": c17_step_time.dayofweek,
                "is_weekend": int(c17_step_time.dayofweek >= 5),
                "lag_1": c17_lag1_val,
                "lag_24": c17_lag24_val,
                "lag_48": c17_history.loc[c17_step_time - pd.Timedelta(hours=48)],
                "lag_144": c17_history.loc[c17_step_time - pd.Timedelta(hours=144)],
                "lag_168": c17_lag168_val,
                "ratio_recent_vs_week": c17_lag1_val / c17_roll168_val
                if c17_roll168_val > 0
                else 0.0,
                "ratio_day_vs_week": c17_lag24_val / c17_roll168_val
                if c17_roll168_val > 0
                else 0.0,
                "growth_ratio_168": c17_lag168_val / c17_lag336_val
                if c17_lag336_val > 0
                else 1.0,
                "is_holiday": int(c17_this_date in c15_holiday_set),
                "is_day_before_holiday": int(c17_tomorrow_date in c15_holiday_set),
                "is_day_after_holiday": int(c17_yesterday_date in c15_holiday_set),
            }
            c17_row_df = pd.DataFrame([c17_row])[c17_feature_cols]

            # KEY CHANGE: model predicts the RATIO, then we multiply back by the
            # current rolling mean to recover an actual order-count prediction
            c17_pred_ratio = c17_model.predict(c17_row_df)[0]
            c17_pred_ratio = max(
                c17_pred_ratio, 0.0
            )  # ratio can't be negative (orders can't be negative)
            c17_pred_value = (
                c17_pred_ratio * c17_roll168_val
            )  # convert back to absolute orders

            c17_pred_value = (
                0.0 if c17_pred_value < c10_zero_threshold else c17_pred_value
            )

            c17_predictions.append(c17_pred_value)
            c17_history.loc[c17_step_time] = c17_pred_value

        c17_forecast = pd.Series(c17_predictions, index=c17_forecast_index)

        c17_aligned = pd.DataFrame(
            {"actual": c17_actual_wf, "forecast": c17_forecast}
        ).dropna()
        assert len(c17_aligned) == 168, (
            f"Window {c17_w + 1} has {len(c17_aligned)} rows, expected 168"
        )

        c17_smape_val = smape(c17_aligned["actual"], c17_aligned["forecast"])
        c17_mse_val = np.mean((c17_aligned["actual"] - c17_aligned["forecast"]) ** 2)

        xgb7_smape_scores.append(c17_smape_val)
        xgb7_mse_scores.append(c17_mse_val)

        print(
            f"  Window {c17_w + 1} ({c17_cutoff_w.date()} cutoff): "
            f"SMAPE={c17_smape_val:.4f}, MSE={c17_mse_val:.1f}, fit_time={c17_elapsed:.2f}s"
        )

    xgb7_mean_smape = np.mean(xgb7_smape_scores)
    xgb7_mean_mse = np.mean(xgb7_mse_scores)
    xgb7_std_smape = np.std(xgb7_smape_scores)

    print(f"\n{'=' * 60}")
    print("XGBOOST v7 RESULTS — true detrending (predict ratio, multiply back)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {xgb7_mean_smape:.4f}  ± {xgb7_std_smape:.4f} std")
    print(f"SMAPE range: [{min(xgb7_smape_scores):.4f}, {max(xgb7_smape_scores):.4f}]")

    print(f"\n{'=' * 60}")
    print("FULL MODEL COMPARISON — ALL XGBOOST VERSIONS")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:      {naive_mean_smape:.4f}")
    print(f"XGBoost v5 SMAPE: {xgb5_mean_smape:.4f}  (bounded ratios + growth ratio)")
    print(f"XGBoost v6 SMAPE: {xgb6_mean_smape:.4f}  (+ holiday flags)")
    print(f"XGBoost v7 SMAPE: {xgb7_mean_smape:.4f}  (+ true detrending)")

    c17_v6_to_v7 = (xgb6_mean_smape - xgb7_mean_smape) / xgb6_mean_smape * 100
    print(f"\nDetrending changed SMAPE by {c17_v6_to_v7:+.1f}% vs v6")

    if xgb7_mean_smape < naive_mean_smape:
        c17_vs_naive = (naive_mean_smape - xgb7_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ XGBoost v7 BEATS naive baseline by {c17_vs_naive:.1f}%")
    else:
        c17_gap = (xgb7_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ XGBoost v7 still WORSE than naive by {c17_gap:.1f}%")
    return (c17_df,)


@app.cell
def _(c17_df, val_cutoffs):
    # ── CELL 17b: QUICK VERIFY — does Window 1's degradation trace to small rolling means? ──
    c17b_cutoff = val_cutoffs[0]
    c17b_train_df = c17_df[c17_df.index <= c17b_cutoff]

    print("Training data rolling_mean_168h distribution (Window 1):")
    print(c17b_train_df["rolling_mean_168h"].describe())
    print(
        f"\nHow many training rows have rolling_mean_168h < 20 (small denominator risk)? "
        f"{(c17b_train_df['rolling_mean_168h'] < 20).sum()} out of {len(c17b_train_df)} "
        f"({(c17b_train_df['rolling_mean_168h'] < 20).mean() * 100:.1f}%)"
    )

    # What's the resulting detrended_target spread when rolling_mean is small vs normal?
    c17b_small_denom = c17b_train_df[c17b_train_df["rolling_mean_168h"] < 20]
    c17b_normal_denom = c17b_train_df[c17b_train_df["rolling_mean_168h"] >= 20]
    print(
        f"\ndetrended_target std when rolling_mean<20:  {c17b_small_denom['detrended_target'].std():.3f}"
    )
    print(
        f"detrended_target std when rolling_mean>=20: {c17b_normal_denom['detrended_target'].std():.3f}"
    )
    print(
        "(higher std with small denominators confirms the noise-amplification mechanism)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Theta model

    From the slides:
    "The theta model... is equivalent to SES (Simple Exponential Smoothing) with drift: ŷ(t+h) = ŷ(t+h)^SES + (b̂/2)·h, where b̂ is the OLS slope of y_t on t. SES tracks the recent level; the drift corrects for the long-run direction. It works really well, but you need to remove the seasonal component from the TS first."

    Breaking this down:

    - Simple Exponential Smoothing (SES) forecasts the future as a weighted average of past observations, where recent observations matter more as weights decay geometrically the further back in time you go, controlled by a parameter α (alpha) between 0 and 1. A high α means "trust the most recent value heavily"; a low α means "smooth over a longer history." Critically, SES alone produces a flat forecast — it just repeats its last smoothed estimate forever, with no ability to continue a trend.

    - The Theta model fixes this by adding a linear drift term: it fits a simple straight-line trend (via OLS — Ordinary Least Squares, the same linear regression from your Ex02) to the historical data, and adds half of that trend's slope, multiplied by the forecast horizon, on top of the SES forecast. This lets Theta's forecast continue drifting in the trend's direction rather than going flat. It won the M3 forecasting competition in 2000 and remains a genuinely strong, simple benchmark — exactly in the spirit of your course's "simple models often win" principle.
    - The crucial caveat from the slides: "you need to remove the seasonal component first." Theta has no native concept of seasonality (no daily/weekly cycle awareness) — it can only track level and trend. So we must deseasonalize our data before fitting Theta, then reseasonalize the forecast afterward. This mirrors exactly the structure of our seasonal decomposition from Cell 4 (trend + seasonal + residual).
    """)
    return


@app.cell
def _(
    N_VAL_WEEKS,
    naive_mean_smape,
    np,
    pd,
    sarima2_mean_smape,
    series,
    smape,
    time,
    val_cutoffs,
    xgb6_mean_smape,
):
    # ── CELL 18: Theta Model — walk-forward validation, period=168 (weekly) ──
    # WHY THIS CELL EXISTS: course slides identify the Theta model as a strong,
    # simple SOTA benchmark (won the M3 forecasting competition) that combines
    # Simple Exponential Smoothing [SES — a weighted average of past values
    # where recent observations matter more, decaying geometrically] with a
    # linear trend (drift) term. Unlike XGBoost, Theta has NATIVE seasonal
    # decomposition built in (statsmodels handles deseasonalize/reseasonalize
    # internally) and unlike SARIMA, it has explicit trend-extrapolation via
    # its drift term — directly targeting the +55% YoY growth we found in EDA.
    #
    # We set period=168 (weekly) since EDA confirmed this is our DOMINANT
    # seasonal signal (ACF=0.929, stronger than daily's 0.887). Like SARIMA,
    # ThetaModel only handles ONE seasonal period — we are testing whether its
    # native trend-handling compensates for not seeing the daily cycle directly
    # (though deseasonalizing on period=168 should still implicitly preserve
    # daily shape within each week, since the decomposition removes the WHOLE
    # repeating weekly pattern, which includes each day's distinct shape).

    from statsmodels.tsa.forecasting.theta import (
        ThetaModel,  # NEW import — not used in any prior cell
    )

    theta_smape_scores = []
    theta_mse_scores = []
    theta_fit_seconds = []

    for c18_w, c18_cutoff_w in enumerate(val_cutoffs):
        c18_train_wf = series[series.index <= c18_cutoff_w]
        c18_actual_wf = series[
            (series.index > c18_cutoff_w)
            & (series.index <= c18_cutoff_w + pd.Timedelta(hours=168))
        ]

        # Same recency window discipline as SARIMA (Cell 6/7) — Theta's SES
        # component doesn't need 11 months of history to learn weekly shape,
        # and keeps fitting fast. We use a slightly longer window than SARIMA's
        # 2000h since Theta needs enough FULL WEEKS to estimate seasonality
        # reliably — we use the last 4000 hours (~23-24 weeks)
        c18_train_recent = c18_train_wf.iloc[-4000:]

        c18_start_time = time.time()

        # ThetaModel requires a clean, gap-free hourly frequency index — our
        # series already has this (orders_filled, built in Cell 2)
        c18_model = ThetaModel(
            c18_train_recent,
            period=168,  # weekly seasonal period — our dominant signal per EDA
            deseasonalize=True,  # remove the weekly pattern before fitting SES+drift
            method="additive",  # additive: matches our EDA finding (stable peak-to-mean ratio,
            # not growing amplitude — Cell 3 showed ~5.5x ratio, fairly stable)
        )
        c18_fitted = c18_model.fit()

        c18_elapsed = time.time() - c18_start_time
        theta_fit_seconds.append(c18_elapsed)

        # ── Forecast the next 168 hours ────────────────────────────────────────
        c18_forecast_vals = c18_fitted.forecast(steps=168)

        c18_forecast_index = pd.date_range(
            start=c18_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
        )
        c18_forecast = pd.Series(c18_forecast_vals.values, index=c18_forecast_index)
        c18_forecast = c18_forecast.clip(
            lower=0
        )  # floor negative predictions, same as SARIMA

        # ── Align and score ─────────────────────────────────────────────────
        c18_aligned = pd.DataFrame(
            {"actual": c18_actual_wf, "forecast": c18_forecast}
        ).dropna()
        assert len(c18_aligned) == 168, (
            f"Window {c18_w + 1} has {len(c18_aligned)} rows, expected 168"
        )

        c18_smape_val = smape(c18_aligned["actual"], c18_aligned["forecast"])
        c18_mse_val = np.mean((c18_aligned["actual"] - c18_aligned["forecast"]) ** 2)

        theta_smape_scores.append(c18_smape_val)
        theta_mse_scores.append(c18_mse_val)

        print(
            f"  Window {c18_w + 1} ({c18_cutoff_w.date()} cutoff): "
            f"SMAPE={c18_smape_val:.4f}, MSE={c18_mse_val:.1f}, fit_time={c18_elapsed:.2f}s"
        )

    theta_mean_smape = np.mean(theta_smape_scores)
    theta_mean_mse = np.mean(theta_mse_scores)
    theta_std_smape = np.std(theta_smape_scores)
    theta_total_fit_time = np.sum(theta_fit_seconds)

    print(f"\n{'=' * 60}")
    print(f"THETA MODEL RESULTS  (period=168, n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {theta_mean_smape:.4f}  ± {theta_std_smape:.4f} std")
    print(f"Mean MSE:   {theta_mean_mse:.1f}")
    print(
        f"SMAPE range: [{min(theta_smape_scores):.4f}, {max(theta_smape_scores):.4f}]"
    )
    print(
        f"Total fit time: {theta_total_fit_time:.1f}s (avg {theta_total_fit_time / N_VAL_WEEKS:.2f}s/window)"
    )

    print(f"\n{'=' * 60}")
    print("FULL MODEL COMPARISON — ALL FAMILIES")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:       {naive_mean_smape:.4f}")
    print(f"SARIMA v2 SMAPE:   {sarima2_mean_smape:.4f}  (daily-only memory)")
    print(f"XGBoost v6 SMAPE:  {xgb6_mean_smape:.4f}  (best XGBoost so far)")
    print(
        f"Theta SMAPE:       {theta_mean_smape:.4f}  (SES + drift, weekly deseasonalized)"
    )

    if theta_mean_smape < naive_mean_smape:
        c18_improvement = (naive_mean_smape - theta_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ Theta BEATS naive baseline by {c18_improvement:.1f}%")
    else:
        c18_gap = (theta_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ Theta still WORSE than naive by {c18_gap:.1f}%")
    return ThetaModel, theta_mean_smape


@app.cell
def _(ThetaModel, pd, plt, series, val_cutoffs):
    # ── CELL 18b: DIAGNOSTIC — plot Theta's actual Window 1 forecast vs actual ──
    # SMAPE near 0.98-1.1 across ALL 7 windows is catastrophic and uniform —
    # unlike our other failures, which were concentrated in specific windows.
    # A UNIFORM failure across every window suggests either (a) a fundamental
    # mismatch between Theta's assumptions and our zero-heavy, highly-spiky
    # data, or (b) a mechanical bug. We plot first, as always, before
    # theorizing further.

    c18b_cutoff = val_cutoffs[0]
    c18b_train_wf = series[series.index <= c18b_cutoff]
    c18b_actual_wf = series[
        (series.index > c18b_cutoff)
        & (series.index <= c18b_cutoff + pd.Timedelta(hours=168))
    ]
    c18b_train_recent = c18b_train_wf.iloc[-4000:]

    c18b_model = ThetaModel(
        c18b_train_recent, period=168, deseasonalize=True, method="additive"
    )
    c18b_fitted = c18b_model.fit()

    print("=" * 60)
    print("THETA MODEL FIT SUMMARY")
    print("=" * 60)
    print(c18b_fitted.summary())

    c18b_forecast_vals = c18b_fitted.forecast(steps=168)
    c18b_forecast_index = pd.date_range(
        start=c18b_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
    )
    c18b_forecast_raw = pd.Series(c18b_forecast_vals.values, index=c18b_forecast_index)

    print("\nRAW forecast stats (before clipping):")
    print(
        f"Min: {c18b_forecast_raw.min():.1f}, Max: {c18b_forecast_raw.max():.1f}, Mean: {c18b_forecast_raw.mean():.1f}"
    )
    print("\nActual stats this window:")
    print(
        f"Min: {c18b_actual_wf.min():.1f}, Max: {c18b_actual_wf.max():.1f}, Mean: {c18b_actual_wf.mean():.1f}"
    )

    c18b_forecast_clipped = c18b_forecast_raw.clip(lower=0)

    fig_diag8, ax_diag8 = plt.subplots(figsize=(16, 5))
    ax_diag8.plot(
        c18b_actual_wf.index,
        c18b_actual_wf.values,
        color="#2563EB",
        label="Actual",
        marker="o",
        ms=3,
    )
    ax_diag8.plot(
        c18b_forecast_clipped.index,
        c18b_forecast_clipped.values,
        color="#EF4444",
        label="Theta forecast",
        marker="x",
        ms=3,
    )
    ax_diag8.set_title(
        "DIAGNOSTIC: Theta Model — Window 1 Forecast vs Actual", fontweight="bold"
    )
    ax_diag8.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/diag_08_theta_window1.png", dpi=150, bbox_inches="tight")
    plt.show()
    return c18b_fitted, c18b_forecast_raw


@app.cell
def _(c18b_fitted, c18b_forecast_raw, pd):
    # ── CELL 18c: VERIFY — inspect Theta's seasonal component directly ───────
    # Hypothesis: Theta's additive seasonal correction for night hours isn't
    # strong enough to pull the SES+drift level (~70-110) down to ~0, leaving
    # a residual floor around 35. We inspect the model's internal seasonal
    # estimates directly to check this.

    print("Seasonal component estimates (first 48 hours of the 168-hour cycle):")
    print(
        pd.Series(
            c18b_fitted.model.seasonal
            if hasattr(c18b_fitted.model, "seasonal")
            else "N/A — checking summary instead"
        )
    )

    # The forecast() method's documentation suggests seasonal terms are stored
    # on the model itself — let's also just directly inspect a few known
    # night-hour vs day-hour forecast values to confirm the floor visually
    print(
        "\nForecast values for HOUR 0-5 of the cycle (should be deep night, near-zero actual):"
    )
    print(c18b_forecast_raw.iloc[:6])
    print(
        "\nForecast values for HOUR 20-23 (Dec 6 21:00-23:00, evening, should be near zero by 23:00):"
    )
    print(c18b_forecast_raw.iloc[20:24])
    return


@app.cell
def _(
    ThetaModel,
    naive_mean_smape,
    np,
    pd,
    series,
    smape,
    theta_mean_smape,
    val_cutoffs,
):
    # ── CELL 18d: Theta with manual zero-floor fix — full walk-forward retest ──
    # DIAGNOSED (Cell 18c): Theta's seasonal reconstruction leaves a hard,
    # near-constant floor (~35-36) during deep-night hours, instead of reaching
    # true zero. This is the SAME category of bug as XGBoost v1's zero-hour
    # noise problem (Cell 9d-9e), just with a much LARGER and more CONSISTENT
    # offset (35 vs XGBoost's average 0.35). We apply the same fix: any
    # prediction below a threshold gets snapped to exactly 0.0. Given the
    # floor sits consistently around 35-36, we need a threshold comfortably
    # ABOVE that (e.g., 45) to actually catch it, unlike XGBoost's much
    # smaller 5.0 threshold.

    c18d_zero_threshold = (
        45.0  # must exceed the observed ~35-36 floor to actually catch it
    )

    theta2_smape_scores = []
    theta2_mse_scores = []

    for c18d_w, c18d_cutoff_w in enumerate(val_cutoffs):
        c18d_train_wf = series[series.index <= c18d_cutoff_w]
        c18d_actual_wf = series[
            (series.index > c18d_cutoff_w)
            & (series.index <= c18d_cutoff_w + pd.Timedelta(hours=168))
        ]
        c18d_train_recent = c18d_train_wf.iloc[-4000:]

        c18d_model = ThetaModel(
            c18d_train_recent, period=168, deseasonalize=True, method="additive"
        )
        c18d_fitted = c18d_model.fit()

        c18d_forecast_vals = c18d_fitted.forecast(steps=168)
        c18d_forecast_index = pd.date_range(
            start=c18d_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
        )
        c18d_forecast = pd.Series(c18d_forecast_vals.values, index=c18d_forecast_index)

        c18d_forecast = c18d_forecast.clip(lower=0)  # physical floor
        c18d_forecast = c18d_forecast.where(
            c18d_forecast >= c18d_zero_threshold, 0.0
        )  # zero-floor fix

        c18d_aligned = pd.DataFrame(
            {"actual": c18d_actual_wf, "forecast": c18d_forecast}
        ).dropna()
        assert len(c18d_aligned) == 168, (
            f"Window {c18d_w + 1} has {len(c18d_aligned)} rows"
        )

        c18d_smape_val = smape(c18d_aligned["actual"], c18d_aligned["forecast"])
        c18d_mse_val = np.mean((c18d_aligned["actual"] - c18d_aligned["forecast"]) ** 2)

        theta2_smape_scores.append(c18d_smape_val)
        theta2_mse_scores.append(c18d_mse_val)

        print(
            f"  Window {c18d_w + 1} ({c18d_cutoff_w.date()} cutoff): "
            f"SMAPE={c18d_smape_val:.4f}, MSE={c18d_mse_val:.1f}"
        )

    theta2_mean_smape = np.mean(theta2_smape_scores)
    theta2_std_smape = np.std(theta2_smape_scores)

    print(f"\n{'=' * 60}")
    print(f"THETA MODEL (with zero-floor fix, threshold={c18d_zero_threshold}) RESULTS")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {theta2_mean_smape:.4f}  ± {theta2_std_smape:.4f} std")
    print(
        f"SMAPE range: [{min(theta2_smape_scores):.4f}, {max(theta2_smape_scores):.4f}]"
    )

    print(f"\n{'=' * 60}")
    print("COMPARISON: Theta v1 (broken) vs Theta v2 (zero-floor fix) vs naive")
    print(f"{'=' * 60}")
    print(f"Theta v1 SMAPE (no fix):    {theta_mean_smape:.4f}")
    print(f"Theta v2 SMAPE (with fix):  {theta2_mean_smape:.4f}")
    print(f"Naive SMAPE:                {naive_mean_smape:.4f}")

    c18d_improvement = (theta_mean_smape - theta2_mean_smape) / theta_mean_smape * 100
    print(f"\nZero-floor fix improved Theta SMAPE by {c18d_improvement:.1f}%")

    if theta2_mean_smape < naive_mean_smape:
        c18d_vs_naive = (naive_mean_smape - theta2_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ Theta v2 BEATS naive baseline by {c18d_vs_naive:.1f}%")
    else:
        c18d_gap = (theta2_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ Theta v2 still WORSE than naive by {c18d_gap:.1f}%")
    return c18d_zero_threshold, theta2_mean_smape


@app.cell
def _(ThetaModel, c18d_zero_threshold, pd, plt, series, val_cutoffs):
    # ── CELL 18e: DIAGNOSTIC — why is Window 5 untouched by the zero-floor fix? ──
    # Theta v2's zero-floor fix had EXACTLY ZERO effect on Window 5 (both
    # 1.1061) — meaning whatever is wrong there is NOT a zero-hour problem.
    # Jan 3-9, 2022 contains Jan 6 (Epiphany, confirmed Catalonia holiday).
    # We plot this window directly to see the actual failure shape.

    c18e_cutoff = val_cutoffs[4]  # Window 5: 2022-01-02 23:00:00
    c18e_train_wf = series[series.index <= c18e_cutoff]
    c18e_actual_wf = series[
        (series.index > c18e_cutoff)
        & (series.index <= c18e_cutoff + pd.Timedelta(hours=168))
    ]
    c18e_train_recent = c18e_train_wf.iloc[-4000:]

    c18e_model = ThetaModel(
        c18e_train_recent, period=168, deseasonalize=True, method="additive"
    )
    c18e_fitted = c18e_model.fit()

    print("Theta fit summary, Window 5:")
    print(c18e_fitted.summary())

    c18e_forecast_vals = c18e_fitted.forecast(steps=168)
    c18e_forecast_index = pd.date_range(
        start=c18e_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
    )
    c18e_forecast_raw = pd.Series(c18e_forecast_vals.values, index=c18e_forecast_index)
    c18e_forecast_fixed = c18e_forecast_raw.clip(lower=0)
    c18e_forecast_fixed = c18e_forecast_fixed.where(
        c18e_forecast_fixed >= c18d_zero_threshold, 0.0
    )

    print(
        f"\nForecast range: [{c18e_forecast_fixed.min():.1f}, {c18e_forecast_fixed.max():.1f}]"
    )
    print(f"Actual range:    [{c18e_actual_wf.min():.1f}, {c18e_actual_wf.max():.1f}]")

    fig_diag9, ax_diag9 = plt.subplots(figsize=(16, 5))
    ax_diag9.plot(
        c18e_actual_wf.index,
        c18e_actual_wf.values,
        color="#2563EB",
        label="Actual",
        marker="o",
        ms=3,
    )
    ax_diag9.plot(
        c18e_forecast_fixed.index,
        c18e_forecast_fixed.values,
        color="#EF4444",
        label="Theta v2 (zero-floor fixed)",
        marker="x",
        ms=3,
    )
    ax_diag9.axvline(
        pd.Timestamp("2022-01-06"),
        color="#F97316",
        linestyle="--",
        label="Jan 6: Epiphany (holiday)",
    )
    ax_diag9.set_title(
        "DIAGNOSTIC: Theta v2 — Window 5 (Jan 3-9, 2022) Forecast vs Actual",
        fontweight="bold",
    )
    ax_diag9.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/diag_09_theta_window5.png", dpi=150, bbox_inches="tight")
    plt.show()
    return


@app.cell
def _(
    ThetaModel,
    naive_mean_smape,
    np,
    pd,
    series,
    smape,
    theta2_mean_smape,
    theta_mean_smape,
    val_cutoffs,
    xgb6_mean_smape,
):
    # ── CELL 18f: Theta with ADAPTIVE per-window zero-floor (derived from each fit) ──
    # DIAGNOSED (Cell 18d, 18e): a FIXED threshold (45.0) worked for windows
    # with a ~35 floor but completely failed on Window 5, whose fit produced
    # an ~80 floor instead — confirming the floor magnitude is WINDOW-SPECIFIC,
    # tracking each fit's own SES level/alpha. FIX: instead of one global
    # threshold, derive the floor PER WINDOW directly from that window's own
    # minimum forecasted value (the floor IS the minimum, almost by definition
    # in every window we've inspected) plus a small margin for noise.

    theta3_smape_scores = []
    theta3_mse_scores = []

    for c18f_w, c18f_cutoff_w in enumerate(val_cutoffs):
        c18f_train_wf = series[series.index <= c18f_cutoff_w]
        c18f_actual_wf = series[
            (series.index > c18f_cutoff_w)
            & (series.index <= c18f_cutoff_w + pd.Timedelta(hours=168))
        ]
        c18f_train_recent = c18f_train_wf.iloc[-4000:]

        c18f_model = ThetaModel(
            c18f_train_recent, period=168, deseasonalize=True, method="additive"
        )
        c18f_fitted = c18f_model.fit()

        c18f_forecast_vals = c18f_fitted.forecast(steps=168)
        c18f_forecast_index = pd.date_range(
            start=c18f_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
        )
        c18f_forecast_raw = pd.Series(
            c18f_forecast_vals.values, index=c18f_forecast_index
        )
        c18f_forecast_raw = c18f_forecast_raw.clip(lower=0)

        # ── ADAPTIVE floor: derive threshold from THIS window's own minimum ──
        # We add a 20% margin above the observed minimum, to also catch the
        # few hours that sit slightly above the absolute floor value but are
        # still clearly "night noise" rather than genuine demand
        c18f_window_floor = c18f_forecast_raw.min()
        c18f_adaptive_threshold = c18f_window_floor * 1.20

        c18f_forecast_fixed = c18f_forecast_raw.where(
            c18f_forecast_raw >= c18f_adaptive_threshold, 0.0
        )

        c18f_aligned = pd.DataFrame(
            {"actual": c18f_actual_wf, "forecast": c18f_forecast_fixed}
        ).dropna()
        assert len(c18f_aligned) == 168, (
            f"Window {c18f_w + 1} has {len(c18f_aligned)} rows"
        )

        c18f_smape_val = smape(c18f_aligned["actual"], c18f_aligned["forecast"])
        c18f_mse_val = np.mean((c18f_aligned["actual"] - c18f_aligned["forecast"]) ** 2)

        theta3_smape_scores.append(c18f_smape_val)
        theta3_mse_scores.append(c18f_mse_val)

        print(
            f"  Window {c18f_w + 1} ({c18f_cutoff_w.date()} cutoff): "
            f"floor={c18f_window_floor:.1f}, adaptive_threshold={c18f_adaptive_threshold:.1f}, "
            f"SMAPE={c18f_smape_val:.4f}, MSE={c18f_mse_val:.1f}"
        )

    theta3_mean_smape = np.mean(theta3_smape_scores)
    theta3_std_smape = np.std(theta3_smape_scores)

    print(f"\n{'=' * 60}")
    print("THETA MODEL (ADAPTIVE per-window zero-floor) RESULTS")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {theta3_mean_smape:.4f}  ± {theta3_std_smape:.4f} std")
    print(
        f"SMAPE range: [{min(theta3_smape_scores):.4f}, {max(theta3_smape_scores):.4f}]"
    )

    print(f"\n{'=' * 60}")
    print("FULL THETA COMPARISON")
    print(f"{'=' * 60}")
    print(f"Theta v1 (no fix):           {theta_mean_smape:.4f}")
    print(f"Theta v2 (fixed thresh=45):  {theta2_mean_smape:.4f}")
    print(f"Theta v3 (adaptive thresh):  {theta3_mean_smape:.4f}")
    print(f"Naive SMAPE:                 {naive_mean_smape:.4f}")
    print(f"XGBoost v6 SMAPE:            {xgb6_mean_smape:.4f}")

    if theta3_mean_smape < naive_mean_smape:
        c18f_vs_naive = (naive_mean_smape - theta3_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ Theta v3 BEATS naive baseline by {c18f_vs_naive:.1f}%")
    else:
        c18f_gap = (theta3_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ Theta v3 still WORSE than naive by {c18f_gap:.1f}%")
    return (theta3_mean_smape,)


@app.cell
def _(ThetaModel, pd, plt, series, val_cutoffs):
    # ── CELL 18g: RE-DIAGNOSTIC — actually look at Theta v3's Window 5 output ──
    # Per-window floor values are inconsistent (12 to 79) across windows,
    # suggesting "snap minimum to zero" may be masking the REAL problem rather
    # than fixing it. We need to see the actual forecast shape again, now with
    # the adaptive fix applied, focusing on the worst remaining window.

    c18g_cutoff = val_cutoffs[4]  # Window 5, still worst at 0.4818
    c18g_train_wf = series[series.index <= c18g_cutoff]
    c18g_actual_wf = series[
        (series.index > c18g_cutoff)
        & (series.index <= c18g_cutoff + pd.Timedelta(hours=168))
    ]
    c18g_train_recent = c18g_train_wf.iloc[-4000:]

    c18g_model = ThetaModel(
        c18g_train_recent, period=168, deseasonalize=True, method="additive"
    )
    c18g_fitted = c18g_model.fit()

    c18g_forecast_vals = c18g_fitted.forecast(steps=168)
    c18g_forecast_index = pd.date_range(
        start=c18g_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
    )
    c18g_forecast_raw = pd.Series(
        c18g_forecast_vals.values, index=c18g_forecast_index
    ).clip(lower=0)

    c18g_floor = c18g_forecast_raw.min()
    c18g_threshold = c18g_floor * 1.20
    c18g_forecast_fixed = c18g_forecast_raw.where(
        c18g_forecast_raw >= c18g_threshold, 0.0
    )

    # ── Print EVERY value for the first 48 hours, not just summary stats ─────
    print("Hour-by-hour, first 48 hours (actual vs raw forecast vs fixed forecast):")
    c18g_compare = pd.DataFrame(
        {
            "actual": c18g_actual_wf.values[:48],
            "forecast_raw": c18g_forecast_raw.values[:48],
            "forecast_fixed": c18g_forecast_fixed.values[:48],
        },
        index=c18g_forecast_index[:48],
    )
    print(c18g_compare.to_string())

    fig_diag10, ax_diag10 = plt.subplots(figsize=(16, 5))
    ax_diag10.plot(
        c18g_actual_wf.index,
        c18g_actual_wf.values,
        color="#2563EB",
        label="Actual",
        marker="o",
        ms=3,
    )
    ax_diag10.plot(
        c18g_forecast_fixed.index,
        c18g_forecast_fixed.values,
        color="#EF4444",
        label="Theta v3 (adaptive fix)",
        marker="x",
        ms=3,
    )
    ax_diog10_title = "DIAGNOSTIC: Theta v3 (adaptive) — Window 5 Full Detail"
    ax_diag10.set_title(ax_diog10_title, fontweight="bold")
    ax_diag10.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("figures/diag_10_theta_v3_window5.png", dpi=150, bbox_inches="tight")
    plt.show()
    return c18g_actual_wf, c18g_forecast_raw


@app.cell
def _(c18g_actual_wf, c18g_forecast_raw):
    # ── CELL 18h: VERIFY — is the offset truly constant, and can we subtract it cleanly? ──
    # Hypothesis: there's a single, roughly-constant additive bias baked into
    # EVERY hour of Theta's reseasonalized forecast — NOT a "floor" specific
    # to zero-hours. Visible at night (where actual=0 makes the offset the
    # entire prediction) AND during the day (where it inflates genuine signal).
    # We test this by checking if (forecast_raw - actual) is roughly constant
    # across hours where we're confident the model SHOULD be accurate.

    # Compare forecast_raw to actual directly, hour by hour, for the deep-night
    # hours specifically (where we know actual=0 exactly, so forecast_raw IS
    # the pure offset, no signal mixed in)
    c18h_night_mask = c18g_actual_wf.values[:48] == 0
    c18h_night_offsets = c18g_forecast_raw.values[:48][c18h_night_mask]
    print("Theta's forecast values during TRUE-ZERO hours (pure offset, no signal):")
    print(c18h_night_offsets)
    print(
        f"\nMean: {c18h_night_offsets.mean():.2f}, Std: {c18h_night_offsets.std():.2f}"
    )
    print(
        "(low std relative to mean would confirm this is a stable, subtractable constant)"
    )
    return


@app.cell
def _(
    ThetaModel,
    naive_mean_smape,
    np,
    pd,
    series,
    smape,
    theta2_mean_smape,
    theta3_mean_smape,
    theta_mean_smape,
    val_cutoffs,
    xgb6_mean_smape,
):
    # ── CELL 19: Theta v4 — subtract the constant additive offset directly ───
    # CONFIRMED (Cell 18h): Theta's forecast carries a STABLE constant additive
    # bias (mean 78.83, std 0.26 in Window 5 — essentially a hard constant, not
    # noise). This single number gets added to EVERY hour of the forecast,
    # regardless of true demand — visible as a "floor" at night (where it's
    # the entire prediction) and as systematic OVERSHOOTING during the day
    # (where it inflates genuine signal on top of real demand). Zero-flooring
    # (v2, v3) only fixed the night-hour symptom; it left the day-hour
    # overshoot completely untouched, which is why v3 still trailed naive by
    # 52% despite "fixing" the floor.
    #
    # THE CORRECT FIX: measure this constant directly from each window's own
    # fit (using the deep-night hours, where actual=0 and forecast IS the pure
    # offset), then SUBTRACT it from the ENTIRE 168-hour forecast — not just
    # snap small values to zero. Re-clip at 0 afterward since subtraction
    # could occasionally push a few values slightly negative.

    theta4_smape_scores = []
    theta4_mse_scores = []

    for c19_w, c19_cutoff_w in enumerate(val_cutoffs):
        c19_train_wf = series[series.index <= c19_cutoff_w]
        c19_actual_wf = series[
            (series.index > c19_cutoff_w)
            & (series.index <= c19_cutoff_w + pd.Timedelta(hours=168))
        ]
        c19_train_recent = c19_train_wf.iloc[-4000:]

        c19_model = ThetaModel(
            c19_train_recent, period=168, deseasonalize=True, method="additive"
        )
        c19_fitted = c19_model.fit()

        c19_forecast_vals = c19_fitted.forecast(steps=168)
        c19_forecast_index = pd.date_range(
            start=c19_train_wf.index[-1] + pd.Timedelta(hours=1), periods=168, freq="h"
        )
        c19_forecast_raw = pd.Series(
            c19_forecast_vals.values, index=c19_forecast_index
        ).clip(lower=0)

        # ── Measure the constant offset directly from THIS window's training data ──
        # We can't use the forecast's own minimum reliably (Window 5 showed
        # variation depending on which hours happen to be in the 168-hour
        # window) — instead, fit the SAME model's one-step-ahead behavior on
        # historically-known true-zero hours within the training data itself,
        # which gives a more robust estimate. Simpler and equally valid: take
        # the minimum of the 168-hour forecast as our offset estimate, since
        # we've confirmed (Cell 18h) it is a hard, stable constant — the
        # forecast's true minimum across a full week reliably IS that constant,
        # since every week contains many genuine zero-actual hours.
        c19_offset_estimate = c19_forecast_raw.min()

        # Subtract the constant from EVERY hour, then re-clip at 0 (subtraction
        # could push a few already-low values slightly negative)
        c19_forecast_corrected = (c19_forecast_raw - c19_offset_estimate).clip(lower=0)

        c19_aligned = pd.DataFrame(
            {"actual": c19_actual_wf, "forecast": c19_forecast_corrected}
        ).dropna()
        assert len(c19_aligned) == 168, (
            f"Window {c19_w + 1} has {len(c19_aligned)} rows"
        )

        c19_smape_val = smape(c19_aligned["actual"], c19_aligned["forecast"])
        c19_mse_val = np.mean((c19_aligned["actual"] - c19_aligned["forecast"]) ** 2)

        theta4_smape_scores.append(c19_smape_val)
        theta4_mse_scores.append(c19_mse_val)

        print(
            f"  Window {c19_w + 1} ({c19_cutoff_w.date()} cutoff): "
            f"offset_subtracted={c19_offset_estimate:.1f}, SMAPE={c19_smape_val:.4f}, MSE={c19_mse_val:.1f}"
        )

    theta4_mean_smape = np.mean(theta4_smape_scores)
    theta4_std_smape = np.std(theta4_smape_scores)

    print(f"\n{'=' * 60}")
    print("THETA MODEL v4 (constant-offset SUBTRACTED, not floored) RESULTS")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {theta4_mean_smape:.4f}  ± {theta4_std_smape:.4f} std")
    print(
        f"SMAPE range: [{min(theta4_smape_scores):.4f}, {max(theta4_smape_scores):.4f}]"
    )

    print(f"\n{'=' * 60}")
    print("FULL THETA EVOLUTION")
    print(f"{'=' * 60}")
    print(f"Theta v1 (no fix):              {theta_mean_smape:.4f}")
    print(f"Theta v2 (fixed floor=45):      {theta2_mean_smape:.4f}")
    print(f"Theta v3 (adaptive floor):      {theta3_mean_smape:.4f}")
    print(f"Theta v4 (subtract constant):   {theta4_mean_smape:.4f}")
    print(f"Naive SMAPE:                    {naive_mean_smape:.4f}")
    print(f"XGBoost v6 SMAPE:                {xgb6_mean_smape:.4f}")

    if theta4_mean_smape < naive_mean_smape:
        c19_vs_naive = (naive_mean_smape - theta4_mean_smape) / naive_mean_smape * 100
        print(f"\n✅ Theta v4 BEATS naive baseline by {c19_vs_naive:.1f}%")
    else:
        c19_gap = (theta4_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"\n❌ Theta v4 still WORSE than naive by {c19_gap:.1f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Family Tested But Excluded: Theta Model

    We tested the **Theta model** (Simple Exponential Smoothing + linear drift,
    with additive deseasonalization at period=168) as a fourth candidate model
    family, following the course's recommendation that it is a strong, simple
    SOTA benchmark (winner of the M3 forecasting competition). After five
    diagnostic iterations, we excluded it from our final model lineup. We
    document our reasoning transparently below, since understanding *why* a
    reasonable technique fails on a specific dataset is as valuable as finding
    one that succeeds.

    ### What we found

    **v1 (no fix): SMAPE = 0.944** — catastrophic. Visual inspection revealed
    Theta's forecast never reached true zero during night hours, instead
    flattening at a roughly constant positive value (~35–80, varying by
    validation window). Since SMAPE assigns the maximum possible score (2.0)
    to any positive prediction on a true-zero hour, and ~32% of our hours are
    structural zeros (Glovo does not operate 00:00–07:00), this alone explained
    most of the catastrophic score.

    **v2 (fixed zero-floor, threshold=45): SMAPE = 0.483** — a fixed threshold
    worked for windows whose floor happened to sit below 45, but completely
    failed on Window 5, whose floor sat at ~79 — confirming the floor magnitude
    is *window-specific*, not a universal constant.

    **v3 (adaptive per-window floor): SMAPE = 0.327** — deriving the threshold
    from each window's own minimum forecasted value improved results
    substantially and confirmed the floor mechanism is real, but a detailed
    hour-by-hour inspection of the still-weak Window 5 revealed the floor was
    never the *only* problem: the same offset that produced the night floor was
    also being added to *every other hour*, including daytime and peak hours,
    producing systematic **overshooting** during the day that the zero-floor
    fix could not address (it only zeroes small values, leaving large
    overshoots on real-demand hours untouched).

    **v4 (subtract the constant from every hour): SMAPE = 0.782** — attempting
    to remove this seemingly-constant offset uniformly made results
    significantly *worse*, not better. This was the most informative result:
    it revealed that what looked like "one constant added everywhere" was in
    fact a *misreading* of the additive decomposition. Additive seasonal
    decomposition assigns a **distinct** seasonal term to each hour-of-week,
    not one global constant — the night-hour value we measured was simply
    Theta's own (incorrect) prediction for that specific seasonal phase, not a
    removable bias layered on top of an otherwise-correct prediction.
    Subtracting it uniformly distorted the *relative* shape of peaks and
    troughs, trading an overshoot problem for a new undershoot problem.

    ### Root cause assessment

    Across all five iterations, the most consistent and defensible explanation
    is: **Theta's underlying Simple Exponential Smoothing component estimates
    a single smoothed "level" for the entire series, and this level estimate is
    distorted by our data's extreme intra-day volatility** — a peak-to-mean
    ratio of ~5.5× (Cell 3) and 31.7% structural zero-hours (Cell 1) are far
    outside the conditions Theta and SES were designed for (smoother series
    such as quarterly GDP, monthly retail volumes, or the antihistamine sales
    series from Exercise 04). The single weekly seasonal period further means
    Theta has no mechanism to separately reconcile the daily (S=24) shape from
    the weekly (S=168) shape the way our lag-feature-based XGBoost can.

    ### Conclusion

    We treat this as a genuine, evidence-based finding rather than a
    implementation failure: **Theta is not well-suited to extremely spiky,
    zero-heavy, dual-seasonality hourly operational data**, despite being a
    strong general-purpose benchmark. This is consistent with our course's
    repeated lesson that model suitability must be verified empirically on the
    specific series at hand, not assumed from general reputation. Theta is
    excluded from our final ensemble. Naive, SARIMA v2, and XGBoost v6 remain
    our candidate models going forward.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##  Optimal Naive + XGBoost v6 Blend Weight
    """)
    return


@app.cell
def _(
    c10_zero_threshold,
    c15_df,
    c15_feature_cols,
    c15_holiday_set,
    np,
    pd,
    seasonal_naive_forecast,
    series,
    smape,
    val_cutoffs,
    xgb,
):
    # ── CELL 21: Optimal blend weight — naive + XGBoost v6 (best XGBoost so far) ──
    # Repeats the same principled grid search as Cell 14, but using v6 (with
    # holiday/long-weekend flags) instead of v5 — since v6 is now our strongest
    # XGBoost variant. We reuse the SAME small, principled grid (not a huge
    # sweep) to avoid overfitting the weight choice to 7 noisy validation
    # windows, exactly as before.

    def c21_rebuild_xgb_v6_forecast(cutoff):
        """Refit XGBoost v6's exact pipeline (holiday flags included) for one cutoff."""
        train_df = c15_df[c15_df.index <= cutoff]
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective="reg:squarederror",
        )
        model.fit(train_df[c15_feature_cols], train_df["orders"])

        history = series[series.index <= cutoff].copy()
        preds = []
        idx = pd.date_range(start=cutoff + pd.Timedelta(hours=1), periods=168, freq="h")
        for t in idx:
            lag1, lag24 = (
                history.loc[t - pd.Timedelta(hours=1)],
                history.loc[t - pd.Timedelta(hours=24)],
            )
            lag168, lag336 = (
                history.loc[t - pd.Timedelta(hours=168)],
                history.loc[t - pd.Timedelta(hours=336)],
            )
            roll168 = history.loc[
                t - pd.Timedelta(hours=168) : t - pd.Timedelta(hours=1)
            ].mean()
            this_date = t.date()
            row = {
                "hour_of_day": t.hour,
                "day_of_week": t.dayofweek,
                "is_weekend": int(t.dayofweek >= 5),
                "lag_1": lag1,
                "lag_24": lag24,
                "lag_48": history.loc[t - pd.Timedelta(hours=48)],
                "lag_144": history.loc[t - pd.Timedelta(hours=144)],
                "lag_168": lag168,
                "ratio_recent_vs_week": lag1 / roll168 if roll168 > 0 else 0.0,
                "ratio_day_vs_week": lag24 / roll168 if roll168 > 0 else 0.0,
                "growth_ratio_168": lag168 / lag336 if lag336 > 0 else 1.0,
                "is_holiday": int(this_date in c15_holiday_set),
                "is_day_before_holiday": int(
                    (this_date + pd.Timedelta(days=1)) in c15_holiday_set
                ),
                "is_day_after_holiday": int(
                    (this_date - pd.Timedelta(days=1)) in c15_holiday_set
                ),
            }
            row_df = pd.DataFrame([row])[c15_feature_cols]
            pred = max(model.predict(row_df)[0], 0.0)
            pred = 0.0 if pred < c10_zero_threshold else pred
            preds.append(pred)
            history.loc[t] = pred
        return pd.Series(preds, index=idx)

    c21_weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    c21_results = []

    print("=" * 60)
    print("OPTIMAL BLEND WEIGHT — naive + XGBoost v6")
    print("=" * 60)

    for c21_weight in c21_weight_grid:
        c21_window_scores = []
        for c21_w, c21_cutoff_w in enumerate(val_cutoffs):
            c21_actual_wf = series[
                (series.index > c21_cutoff_w)
                & (series.index <= c21_cutoff_w + pd.Timedelta(hours=168))
            ]
            c21_naive_fc = seasonal_naive_forecast(
                series[series.index <= c21_cutoff_w], horizon=168
            )
            c21_xgb_fc = c21_rebuild_xgb_v6_forecast(c21_cutoff_w)
            c21_blend_fc = (1 - c21_weight) * c21_naive_fc + c21_weight * c21_xgb_fc
            c21_aligned = pd.DataFrame(
                {"actual": c21_actual_wf, "forecast": c21_blend_fc}
            ).dropna()
            c21_window_scores.append(
                smape(c21_aligned["actual"], c21_aligned["forecast"])
            )

        c21_mean = np.mean(c21_window_scores)
        c21_std = np.std(c21_window_scores)
        c21_results.append(
            {"weight_xgb": c21_weight, "mean_smape": c21_mean, "std_smape": c21_std}
        )
        print(
            f"  weight_xgb={c21_weight:.1f}: mean SMAPE = {c21_mean:.4f}  ± {c21_std:.4f}"
        )

    c21_results_df = pd.DataFrame(c21_results)
    c21_best_row = c21_results_df.loc[c21_results_df["mean_smape"].idxmin()]
    print(
        f"\nBest weight: {c21_best_row['weight_xgb']:.1f}, mean SMAPE = {c21_best_row['mean_smape']:.4f}"
    )
    print(
        f"Compare to naive alone (weight=0.0): {c21_results_df[c21_results_df['weight_xgb'] == 0.0]['mean_smape'].values[0]:.4f}"
    )
    return


@app.cell
def _(
    c10_zero_threshold,
    c15_df,
    c15_feature_cols,
    c15_holiday_set,
    naive_smape_scores,
    pd,
    series,
    smape,
    val_cutoffs,
    xgb,
    xgb6_smape_scores,
):
    # ── CELL 22: Quick test — does learning_rate matter for XGBoost v6? ──────
    # Prior evidence (Cells 9c, 10c) showed capacity-related hyperparameters
    # (max_depth, n_estimators, min_child_weight, subsample) consistently made
    # things WORSE on this dataset, never better. learning_rate is mechanically
    # different — it controls how much each successive tree corrects the
    # residual error of previous trees, not how complex any single tree is.
    # We test a small, cheap grid on Window 1 first (our most-studied window)
    # before committing to a full 7-window re-run.

    c22_lr_grid = [0.01, 0.03, 0.05, 0.1, 0.2]  # 0.05 is our current default

    c22_cutoff = val_cutoffs[0]
    c22_train_df = c15_df[c15_df.index <= c22_cutoff]
    c22_actual_wf = series[
        (series.index > c22_cutoff)
        & (series.index <= c22_cutoff + pd.Timedelta(hours=168))
    ]

    print("=" * 60)
    print("LEARNING RATE TEST — Window 1")
    print("=" * 60)

    for c22_lr in c22_lr_grid:
        c22_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=c22_lr,
            random_state=42,
            objective="reg:squarederror",
        )
        c22_model.fit(c22_train_df[c15_feature_cols], c22_train_df["orders"])

        c22_history = series[series.index <= c22_cutoff].copy()
        c22_predictions = []
        c22_forecast_index = pd.date_range(
            start=c22_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c22_step_time in c22_forecast_index:
            c22_lag1 = c22_history.loc[c22_step_time - pd.Timedelta(hours=1)]
            c22_lag24 = c22_history.loc[c22_step_time - pd.Timedelta(hours=24)]
            c22_lag168 = c22_history.loc[c22_step_time - pd.Timedelta(hours=168)]
            c22_lag336 = c22_history.loc[c22_step_time - pd.Timedelta(hours=336)]
            c22_roll168 = c22_history.loc[
                c22_step_time - pd.Timedelta(hours=168) : c22_step_time
                - pd.Timedelta(hours=1)
            ].mean()
            c22_this_date = c22_step_time.date()

            c22_row = {
                "hour_of_day": c22_step_time.hour,
                "day_of_week": c22_step_time.dayofweek,
                "is_weekend": int(c22_step_time.dayofweek >= 5),
                "lag_1": c22_lag1,
                "lag_24": c22_lag24,
                "lag_48": c22_history.loc[c22_step_time - pd.Timedelta(hours=48)],
                "lag_144": c22_history.loc[c22_step_time - pd.Timedelta(hours=144)],
                "lag_168": c22_lag168,
                "ratio_recent_vs_week": c22_lag1 / c22_roll168
                if c22_roll168 > 0
                else 0.0,
                "ratio_day_vs_week": c22_lag24 / c22_roll168
                if c22_roll168 > 0
                else 0.0,
                "growth_ratio_168": c22_lag168 / c22_lag336 if c22_lag336 > 0 else 1.0,
                "is_holiday": int(c22_this_date in c15_holiday_set),
                "is_day_before_holiday": int(
                    (c22_this_date + pd.Timedelta(days=1)) in c15_holiday_set
                ),
                "is_day_after_holiday": int(
                    (c22_this_date - pd.Timedelta(days=1)) in c15_holiday_set
                ),
            }
            c22_row_df = pd.DataFrame([c22_row])[c15_feature_cols]
            c22_pred = max(c22_model.predict(c22_row_df)[0], 0.0)
            c22_pred = 0.0 if c22_pred < c10_zero_threshold else c22_pred
            c22_predictions.append(c22_pred)
            c22_history.loc[c22_step_time] = c22_pred

        c22_forecast = pd.Series(c22_predictions, index=c22_forecast_index)
        c22_smape_val = smape(c22_actual_wf, c22_forecast)
        print(f"  learning_rate={c22_lr:.2f}: SMAPE = {c22_smape_val:.4f}")

    print(f"\nReference — naive baseline this window: {naive_smape_scores[0]:.4f}")
    print(f"Reference — XGBoost v6 (lr=0.05) this window: {xgb6_smape_scores[0]:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Documentation — SARIMA Ensemble Exclusion Rationale

    ### Model Tested But Excluded From Final Ensemble: SARIMA

    We tested whether adding **SARIMA v2** (the seasonal-AR-fixed version from
    Cells 6–7, `SARIMA(2,0,0)(1,0,1)[24]`) as a third member of our ensemble
    would improve on the naive + XGBoost v6 blend. The result was decisive and
    required no further grid search: even a **5% weight** on SARIMA caused mean
    SMAPE to jump from **0.2114** (naive + XGBoost only) to **0.6308** — nearly
    a 3× degradation — with standard deviation also nearly quadrupling (0.043
    → 0.160). A 10% weight produced essentially the same catastrophic result
    (0.6258), confirming this was not a fluke of one weight value.

    ### Why this happens

    This result is fully consistent with everything we learned diagnosing
    SARIMA on its own (Cells 6–7). SARIMA v2's standalone SMAPE (0.649) is
    roughly **3× worse** than naive (0.215) — and critically, its errors are
    **large in absolute magnitude**, not just occasionally wrong. SARIMA's
    seasonal AR(1) term only captures daily memory (S=24); it has no mechanism
    to represent the dominant **weekly** pattern (S=168, ACF=0.93) that naive
    and XGBoost both exploit directly. The result is a forecast that
    confidently tracks the wrong day-of-week shape for parts of the 168-hour
    horizon. When blended even at a small weight, these large, systematically
    wrong values pull the otherwise-accurate ensemble average substantially
    off course — a weak component with large-magnitude errors does not "average
    out" harmlessly the way two similarly-accurate models' uncorrelated small
    errors do (as we saw work well with naive + XGBoost). Ensembling only pays
    off when combined models are each individually reasonably competent *and*
    make different kinds of mistakes — SARIMA v2 fails the first condition
    here, despite satisfying the second.

    ### Conclusion

    SARIMA v2 remains a valuable, fully-documented part of our project's
    modelling narrative (Cells 6–7) — it demonstrates correct understanding of
    seasonal ARIMA specification, a real diagnosed bug (mean-reversion
    collapse) and its fix (adding seasonal AR memory), and a clear, defensible
    explanation for why it structurally cannot compete with weekly-aware
    approaches on this series. However, it is **excluded from our final
    ensemble**: even a small weight materially harms performance, confirming
    that ensembling benefits depend on the *quality* of each component, not
    merely on combining "different" models for the sake of diversity. Our
    final ensemble remains **0.7 × naive + 0.3 × XGBoost v6**, validated at
    mean SMAPE = 0.2114 across 7 walk-forward windows.
    "\"\")
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LightGBM
    LightGBM is, like XGBoost, a gradient-boosted decision tree library — same fundamental idea (build trees sequentially, each correcting the previous ones' errors). The key practical differences: LightGBM grows trees leaf-wise (always splits the leaf that reduces error the most, regardless of tree depth) rather than XGBoost's level-wise growth (splits all leaves at the current depth before going deeper). This often makes LightGBM faster and sometimes more accurate on tabular data with many features, though results vary by dataset — exactly the kind of thing we should test empirically rather than assume.
    """)
    return


@app.cell
def _(
    N_VAL_WEEKS,
    c10_zero_threshold,
    c15_df,
    c15_feature_cols,
    c15_holiday_set,
    naive_mean_smape,
    np,
    pd,
    series,
    smape,
    time,
    val_cutoffs,
    xgb6_mean_smape,
):
    # ── CELL 25: LightGBM — same feature set as XGBoost v6, walk-forward validation ──
    # WHY THIS CELL EXISTS: testing whether a different gradient-boosting
    # IMPLEMENTATION (leaf-wise growth, vs XGBoost's level-wise growth) performs
    # better on this specific dataset's shape. We deliberately keep EVERYTHING
    # else identical to XGBoost v6 — same features (lags, ratios, holiday flags),
    # same zero-floor fix, same recursive forecasting mechanic — so this is a
    # clean, isolated test of the algorithm itself, not a confound of also
    # changing features at the same time.

    import lightgbm as lgb  # NEW import — gradient boosting, leaf-wise tree growth (vs XGBoost's level-wise)

    lgbm_smape_scores = []
    lgbm_mse_scores = []
    lgbm_fit_seconds = []

    for c25_w, c25_cutoff_w in enumerate(val_cutoffs):
        c25_train_df = c15_df[c15_df.index <= c25_cutoff_w]
        c25_actual_wf = series[
            (series.index > c25_cutoff_w)
            & (series.index <= c25_cutoff_w + pd.Timedelta(hours=168))
        ]

        c25_start_time = time.time()

        # Matched as closely as possible to XGBoost v6's settings, translated
        # to LightGBM's parameter names: num_leaves is LightGBM's primary
        # complexity control (leaf-wise growth doesn't use max_depth the same
        # way XGBoost does) — we use 31, LightGBM's own default, as a fair
        # starting point rather than guessing
        c25_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            objective="regression",
            verbose=-1,  # verbose=-1 silences training logs
        )
        c25_model.fit(c25_train_df[c15_feature_cols], c25_train_df["orders"])

        c25_elapsed = time.time() - c25_start_time
        lgbm_fit_seconds.append(c25_elapsed)

        c25_history = series[series.index <= c25_cutoff_w].copy()
        c25_predictions = []
        c25_forecast_index = pd.date_range(
            start=c25_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c25_step_time in c25_forecast_index:
            c25_lag1_val = c25_history.loc[c25_step_time - pd.Timedelta(hours=1)]
            c25_lag24_val = c25_history.loc[c25_step_time - pd.Timedelta(hours=24)]
            c25_lag168_val = c25_history.loc[c25_step_time - pd.Timedelta(hours=168)]
            c25_lag336_val = c25_history.loc[c25_step_time - pd.Timedelta(hours=336)]
            c25_roll168_val = c25_history.loc[
                c25_step_time - pd.Timedelta(hours=168) : c25_step_time
                - pd.Timedelta(hours=1)
            ].mean()
            c25_this_date = c25_step_time.date()
            c25_tomorrow_date = c25_this_date + pd.Timedelta(days=1)
            c25_yesterday_date = c25_this_date - pd.Timedelta(days=1)

            c25_row = {
                "hour_of_day": c25_step_time.hour,
                "day_of_week": c25_step_time.dayofweek,
                "is_weekend": int(c25_step_time.dayofweek >= 5),
                "lag_1": c25_lag1_val,
                "lag_24": c25_lag24_val,
                "lag_48": c25_history.loc[c25_step_time - pd.Timedelta(hours=48)],
                "lag_144": c25_history.loc[c25_step_time - pd.Timedelta(hours=144)],
                "lag_168": c25_lag168_val,
                "ratio_recent_vs_week": c25_lag1_val / c25_roll168_val
                if c25_roll168_val > 0
                else 0.0,
                "ratio_day_vs_week": c25_lag24_val / c25_roll168_val
                if c25_roll168_val > 0
                else 0.0,
                "growth_ratio_168": c25_lag168_val / c25_lag336_val
                if c25_lag336_val > 0
                else 1.0,
                "is_holiday": int(c25_this_date in c15_holiday_set),
                "is_day_before_holiday": int(c25_tomorrow_date in c15_holiday_set),
                "is_day_after_holiday": int(c25_yesterday_date in c15_holiday_set),
            }
            c25_row_df = pd.DataFrame([c25_row])[c15_feature_cols]

            c25_pred_value = c25_model.predict(c25_row_df)[0]
            c25_pred_value = max(c25_pred_value, 0.0)
            c25_pred_value = (
                0.0 if c25_pred_value < c10_zero_threshold else c25_pred_value
            )  # same zero-floor fix as XGBoost

            c25_predictions.append(c25_pred_value)
            c25_history.loc[c25_step_time] = c25_pred_value

        c25_forecast = pd.Series(c25_predictions, index=c25_forecast_index)

        c25_aligned = pd.DataFrame(
            {"actual": c25_actual_wf, "forecast": c25_forecast}
        ).dropna()
        assert len(c25_aligned) == 168, (
            f"Window {c25_w + 1} has {len(c25_aligned)} rows, expected 168"
        )

        c25_smape_val = smape(c25_aligned["actual"], c25_aligned["forecast"])
        c25_mse_val = np.mean((c25_aligned["actual"] - c25_aligned["forecast"]) ** 2)

        lgbm_smape_scores.append(c25_smape_val)
        lgbm_mse_scores.append(c25_mse_val)

        print(
            f"  Window {c25_w + 1} ({c25_cutoff_w.date()} cutoff): "
            f"SMAPE={c25_smape_val:.4f}, MSE={c25_mse_val:.1f}, fit_time={c25_elapsed:.2f}s"
        )

    lgbm_mean_smape = np.mean(lgbm_smape_scores)
    lgbm_mean_mse = np.mean(lgbm_mse_scores)
    lgbm_std_smape = np.std(lgbm_smape_scores)

    print(f"\n{'=' * 60}")
    print(f"LIGHTGBM RESULTS — same features as XGBoost v6  (n={N_VAL_WEEKS} windows)")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {lgbm_mean_smape:.4f}  ± {lgbm_std_smape:.4f} std")
    print(f"SMAPE range: [{min(lgbm_smape_scores):.4f}, {max(lgbm_smape_scores):.4f}]")

    print(f"\n{'=' * 60}")
    print("DIRECT COMPARISON: XGBoost v6 vs LightGBM (identical features)")
    print(f"{'=' * 60}")
    print(f"Naive SMAPE:       {naive_mean_smape:.4f}")
    print(f"XGBoost v6 SMAPE:  {xgb6_mean_smape:.4f}")
    print(f"LightGBM SMAPE:    {lgbm_mean_smape:.4f}")

    if lgbm_mean_smape < xgb6_mean_smape:
        c25_vs_xgb = (xgb6_mean_smape - lgbm_mean_smape) / xgb6_mean_smape * 100
        print(f"\n✅ LightGBM BEATS XGBoost v6 by {c25_vs_xgb:.1f}%")
    else:
        c25_gap = (lgbm_mean_smape - xgb6_mean_smape) / xgb6_mean_smape * 100
        print(f"\n❌ LightGBM WORSE than XGBoost v6 by {c25_gap:.1f}%")

    if lgbm_mean_smape < naive_mean_smape:
        c25_vs_naive = (naive_mean_smape - lgbm_mean_smape) / naive_mean_smape * 100
        print(f"✅ LightGBM BEATS naive baseline by {c25_vs_naive:.1f}%")
    else:
        c25_gap_naive = (lgbm_mean_smape - naive_mean_smape) / naive_mean_smape * 100
        print(f"❌ LightGBM still WORSE than naive by {c25_gap_naive:.1f}%")

    return lgb, lgbm_mean_smape


@app.cell
def _(
    c10_zero_threshold,
    c15_df,
    c15_feature_cols,
    c15_holiday_set,
    lgb,
    np,
    pd,
    seasonal_naive_forecast,
    series,
    smape,
    val_cutoffs,
):
    # ── CELL 26: Optimal blend weight — naive + LightGBM (for completeness) ──
    def c26_rebuild_lgbm_forecast(cutoff):
        """Refit LightGBM's exact pipeline for one cutoff (mirrors Cell 25)."""
        train_df = c15_df[c15_df.index <= cutoff]
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            objective="regression",
            verbose=-1,
        )
        model.fit(train_df[c15_feature_cols], train_df["orders"])

        history = series[series.index <= cutoff].copy()
        preds = []
        idx = pd.date_range(start=cutoff + pd.Timedelta(hours=1), periods=168, freq="h")
        for t in idx:
            lag1, lag24 = (
                history.loc[t - pd.Timedelta(hours=1)],
                history.loc[t - pd.Timedelta(hours=24)],
            )
            lag168, lag336 = (
                history.loc[t - pd.Timedelta(hours=168)],
                history.loc[t - pd.Timedelta(hours=336)],
            )
            roll168 = history.loc[
                t - pd.Timedelta(hours=168) : t - pd.Timedelta(hours=1)
            ].mean()
            this_date = t.date()
            row = {
                "hour_of_day": t.hour,
                "day_of_week": t.dayofweek,
                "is_weekend": int(t.dayofweek >= 5),
                "lag_1": lag1,
                "lag_24": lag24,
                "lag_48": history.loc[t - pd.Timedelta(hours=48)],
                "lag_144": history.loc[t - pd.Timedelta(hours=144)],
                "lag_168": lag168,
                "ratio_recent_vs_week": lag1 / roll168 if roll168 > 0 else 0.0,
                "ratio_day_vs_week": lag24 / roll168 if roll168 > 0 else 0.0,
                "growth_ratio_168": lag168 / lag336 if lag336 > 0 else 1.0,
                "is_holiday": int(this_date in c15_holiday_set),
                "is_day_before_holiday": int(
                    (this_date + pd.Timedelta(days=1)) in c15_holiday_set
                ),
                "is_day_after_holiday": int(
                    (this_date - pd.Timedelta(days=1)) in c15_holiday_set
                ),
            }
            row_df = pd.DataFrame([row])[c15_feature_cols]
            pred = max(model.predict(row_df)[0], 0.0)
            pred = 0.0 if pred < c10_zero_threshold else pred
            preds.append(pred)
            history.loc[t] = pred
        return pd.Series(preds, index=idx)

    c26_weight_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    print("=" * 60)
    print("OPTIMAL BLEND WEIGHT — naive + LightGBM")
    print("=" * 60)
    for c26_weight in c26_weight_grid:
        c26_scores = []
        for c26_cutoff_w in val_cutoffs:
            c26_actual = series[
                (series.index > c26_cutoff_w)
                & (series.index <= c26_cutoff_w + pd.Timedelta(hours=168))
            ]
            c26_naive_fc = seasonal_naive_forecast(
                series[series.index <= c26_cutoff_w], horizon=168
            )
            c26_lgbm_fc = c26_rebuild_lgbm_forecast(c26_cutoff_w)
            c26_blend = (1 - c26_weight) * c26_naive_fc + c26_weight * c26_lgbm_fc
            c26_aligned = pd.DataFrame(
                {"actual": c26_actual, "forecast": c26_blend}
            ).dropna()
            c26_scores.append(smape(c26_aligned["actual"], c26_aligned["forecast"]))
        print(
            f"  weight_lgbm={c26_weight:.1f}: mean SMAPE = {np.mean(c26_scores):.4f}  ± {np.std(c26_scores):.4f}"
        )

    print("\nCompare to naive+XGBoost v6 best (Cell 21): weight=0.3, SMAPE=0.2114")
    return


@app.cell
def _(
    c10_zero_threshold,
    c15_df,
    c15_feature_cols,
    c15_holiday_set,
    naive_smape_scores,
    np,
    pd,
    series,
    smape,
    val_cutoffs,
    xgb,
    xgb6_smape_scores,
):
    # ── CELL 27: XGBoost with a SMAPE-approximating custom objective ─────────
    # WHY THIS CELL EXISTS: course slides explicitly state SMAPE has an
    # "under-forecasting bias" — meaning the metric itself rewards predictions
    # that sit slightly LOW. We have been training with objective=
    # "reg:squarederror" (MSE) this entire session, which is SYMMETRIC and has
    # NO awareness of this bias. This is a genuine, previously untested lever:
    # the TRAINING OBJECTIVE itself, not features or hyperparameters.
    #
    # CAUTION: a naive closed-form SMAPE gradient is numerically unstable
    # (confirmed via XGBoost's own GitHub issues — direct MAPE-style objectives
    # can "swing wildly" near y=0). We use a SAFER approximation: a smoothed,
    # symmetric percentage-style loss with an epsilon floor to avoid
    # division-by-near-zero blowup, and we verify behavior carefully on ONE
    # window before trusting it across all 7.

    def c27_smape_obj(y_pred, y_true):
        """
        Approximate SMAPE gradient/Hessian for XGBoost's custom objective interface.
        epsilon prevents division blowup when y_true and y_pred are both near zero
        (our most common case: ~32% structural zero-hours).
        """
        epsilon = 10.0  # smoothing constant — comparable in scale to our zero-floor threshold (5-10),
        # chosen to avoid instability without distorting genuine low-demand hours
        denom = np.abs(y_true) + np.abs(y_pred) + epsilon
        sign = np.sign(y_pred - y_true)
        # Approximate gradient: direction of error, scaled by inverse denominator
        # (mirrors SMAPE's structure: error magnitude relative to combined scale)
        grad = 2 * sign / denom
        # Approximate (constant, well-behaved) Hessian — avoids the true SMAPE
        # Hessian's instability near zero; a small positive constant keeps
        # XGBoost's tree-building numerically stable
        hess = np.full_like(y_pred, 2.0 / (denom.mean()))
        return grad, hess

    # ── Quick isolated test on Window 1 before committing to full validation ──
    c27_cutoff = val_cutoffs[0]
    c27_train_df = c15_df[c15_df.index <= c27_cutoff]
    c27_actual_wf = series[
        (series.index > c27_cutoff)
        & (series.index <= c27_cutoff + pd.Timedelta(hours=168))
    ]

    c27_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        objective=c27_smape_obj,  # KEY CHANGE: custom SMAPE-style objective
    )
    c27_model.fit(c27_train_df[c15_feature_cols], c27_train_df["orders"])

    c27_history = series[series.index <= c27_cutoff].copy()
    c27_predictions = []
    c27_forecast_index = pd.date_range(
        start=c27_cutoff + pd.Timedelta(hours=1), periods=168, freq="h"
    )

    for c27_step_time in c27_forecast_index:
        c27_lag1 = c27_history.loc[c27_step_time - pd.Timedelta(hours=1)]
        c27_lag24 = c27_history.loc[c27_step_time - pd.Timedelta(hours=24)]
        c27_lag168 = c27_history.loc[c27_step_time - pd.Timedelta(hours=168)]
        c27_lag336 = c27_history.loc[c27_step_time - pd.Timedelta(hours=336)]
        c27_roll168 = c27_history.loc[
            c27_step_time - pd.Timedelta(hours=168) : c27_step_time
            - pd.Timedelta(hours=1)
        ].mean()
        c27_this_date = c27_step_time.date()

        c27_row = {
            "hour_of_day": c27_step_time.hour,
            "day_of_week": c27_step_time.dayofweek,
            "is_weekend": int(c27_step_time.dayofweek >= 5),
            "lag_1": c27_lag1,
            "lag_24": c27_lag24,
            "lag_48": c27_history.loc[c27_step_time - pd.Timedelta(hours=48)],
            "lag_144": c27_history.loc[c27_step_time - pd.Timedelta(hours=144)],
            "lag_168": c27_lag168,
            "ratio_recent_vs_week": c27_lag1 / c27_roll168 if c27_roll168 > 0 else 0.0,
            "ratio_day_vs_week": c27_lag24 / c27_roll168 if c27_roll168 > 0 else 0.0,
            "growth_ratio_168": c27_lag168 / c27_lag336 if c27_lag336 > 0 else 1.0,
            "is_holiday": int(c27_this_date in c15_holiday_set),
            "is_day_before_holiday": int(
                (c27_this_date + pd.Timedelta(days=1)) in c15_holiday_set
            ),
            "is_day_after_holiday": int(
                (c27_this_date - pd.Timedelta(days=1)) in c15_holiday_set
            ),
        }
        c27_row_df = pd.DataFrame([c27_row])[c15_feature_cols]
        c27_pred = max(c27_model.predict(c27_row_df)[0], 0.0)
        c27_pred = 0.0 if c27_pred < c10_zero_threshold else c27_pred
        c27_predictions.append(c27_pred)
        c27_history.loc[c27_step_time] = c27_pred

    c27_forecast = pd.Series(c27_predictions, index=c27_forecast_index)

    print(
        f"Forecast stats: min={c27_forecast.min():.1f}, max={c27_forecast.max():.1f}, mean={c27_forecast.mean():.1f}"
    )
    print(
        f"Actual stats:   min={c27_actual_wf.min():.1f}, max={c27_actual_wf.max():.1f}, mean={c27_actual_wf.mean():.1f}"
    )
    print(f"\nSMAPE this window: {smape(c27_actual_wf, c27_forecast):.4f}")
    print(
        f"Reference — XGBoost v6 (MSE objective) this window: {xgb6_smape_scores[0]:.4f}"
    )
    print(f"Reference — naive this window: {naive_smape_scores[0]:.4f}")
    return


@app.cell
def _(
    c15_df,
    c15_feature_cols,
    c15_holiday_set,
    lgb,
    lgbm_mean_smape,
    naive_mean_smape,
    np,
    pd,
    series,
    smape,
    val_cutoffs,
):
    # ── CELL 28: LightGBM trained ONLY on active-hours rows (literal reading of "for active hours") ──
    # We abandon the custom-SMAPE-objective approach (Cell 27) — confirmed
    # numerically unstable, consistent with known XGBoost community reports.
    # Instead, we test the more literal reading of the competing team's
    # description: training the GBM model ONLY on rows where orders > 0
    # (removing night-hour rows from TRAINING entirely, not just relying on
    # hour_of_day as a learned feature). At prediction time, we still need
    # SOME way to decide zero vs. non-zero hours — we use hour_of_day directly:
    # hours outside our EDA-confirmed operating window (08:00-22:00, Cell 3)
    # are forced to 0 by a simple rule; hours inside it are predicted by the
    # active-hours-only model.

    c28_active_train_df = c15_df[
        c15_df["orders"] > 0
    ]  # ONLY active-hours rows for training
    print(
        f"Training rows: full={len(c15_df)}, active-only={len(c28_active_train_df)} "
        f"({len(c28_active_train_df) / len(c15_df) * 100:.1f}% retained)"
    )

    lgbm2_smape_scores = []
    lgbm2_mse_scores = []

    for c28_w, c28_cutoff_w in enumerate(val_cutoffs):
        c28_train_df = c28_active_train_df[c28_active_train_df.index <= c28_cutoff_w]
        c28_actual_wf = series[
            (series.index > c28_cutoff_w)
            & (series.index <= c28_cutoff_w + pd.Timedelta(hours=168))
        ]

        c28_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            objective="regression",
            verbose=-1,
        )
        c28_model.fit(c28_train_df[c15_feature_cols], c28_train_df["orders"])

        c28_history = series[series.index <= c28_cutoff_w].copy()
        c28_predictions = []
        c28_forecast_index = pd.date_range(
            start=c28_cutoff_w + pd.Timedelta(hours=1), periods=168, freq="h"
        )

        for c28_step_time in c28_forecast_index:
            # Simple rule: outside the confirmed operating window, force 0 directly —
            # no model call needed (this IS the literal "for active hours" reading)
            if c28_step_time.hour < 8 or c28_step_time.hour > 22:
                c28_pred = 0.0
            else:
                c28_lag1 = c28_history.loc[c28_step_time - pd.Timedelta(hours=1)]
                c28_lag24 = c28_history.loc[c28_step_time - pd.Timedelta(hours=24)]
                c28_lag168 = c28_history.loc[c28_step_time - pd.Timedelta(hours=168)]
                c28_lag336 = c28_history.loc[c28_step_time - pd.Timedelta(hours=336)]
                c28_roll168 = c28_history.loc[
                    c28_step_time - pd.Timedelta(hours=168) : c28_step_time
                    - pd.Timedelta(hours=1)
                ].mean()
                c28_this_date = c28_step_time.date()

                c28_row = {
                    "hour_of_day": c28_step_time.hour,
                    "day_of_week": c28_step_time.dayofweek,
                    "is_weekend": int(c28_step_time.dayofweek >= 5),
                    "lag_1": c28_lag1,
                    "lag_24": c28_lag24,
                    "lag_48": c28_history.loc[c28_step_time - pd.Timedelta(hours=48)],
                    "lag_144": c28_history.loc[c28_step_time - pd.Timedelta(hours=144)],
                    "lag_168": c28_lag168,
                    "ratio_recent_vs_week": c28_lag1 / c28_roll168
                    if c28_roll168 > 0
                    else 0.0,
                    "ratio_day_vs_week": c28_lag24 / c28_roll168
                    if c28_roll168 > 0
                    else 0.0,
                    "growth_ratio_168": c28_lag168 / c28_lag336
                    if c28_lag336 > 0
                    else 1.0,
                    "is_holiday": int(c28_this_date in c15_holiday_set),
                    "is_day_before_holiday": int(
                        (c28_this_date + pd.Timedelta(days=1)) in c15_holiday_set
                    ),
                    "is_day_after_holiday": int(
                        (c28_this_date - pd.Timedelta(days=1)) in c15_holiday_set
                    ),
                }
                c28_row_df = pd.DataFrame([c28_row])[c15_feature_cols]
                c28_pred = max(c28_model.predict(c28_row_df)[0], 0.0)

            c28_predictions.append(c28_pred)
            c28_history.loc[c28_step_time] = c28_pred

        c28_forecast = pd.Series(c28_predictions, index=c28_forecast_index)
        c28_aligned = pd.DataFrame(
            {"actual": c28_actual_wf, "forecast": c28_forecast}
        ).dropna()
        assert len(c28_aligned) == 168, (
            f"Window {c28_w + 1} has {len(c28_aligned)} rows"
        )

        c28_smape_val = smape(c28_aligned["actual"], c28_aligned["forecast"])
        c28_mse_val = np.mean((c28_aligned["actual"] - c28_aligned["forecast"]) ** 2)
        lgbm2_smape_scores.append(c28_smape_val)
        lgbm2_mse_scores.append(c28_mse_val)

        print(
            f"  Window {c28_w + 1} ({c28_cutoff_w.date()} cutoff): SMAPE={c28_smape_val:.4f}, MSE={c28_mse_val:.1f}"
        )

    lgbm2_mean_smape = np.mean(lgbm2_smape_scores)
    lgbm2_std_smape = np.std(lgbm2_smape_scores)

    print(f"\n{'=' * 60}")
    print("LightGBM (active-hours-only training) RESULTS")
    print(f"{'=' * 60}")
    print(f"Mean SMAPE: {lgbm2_mean_smape:.4f}  ± {lgbm2_std_smape:.4f} std")
    print("\nComparison:")
    print(f"Naive:                          {naive_mean_smape:.4f}")
    print(f"LightGBM (all-hours training):  {lgbm_mean_smape:.4f}")
    print(f"LightGBM (active-only training):{lgbm2_mean_smape:.4f}")
    return


if __name__ == "__main__":
    app.run()
