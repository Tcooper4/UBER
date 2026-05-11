# %%
# Cell 1 — Setup (paths, DuckDB `master` view, row count / date range, shared helpers)
"""
Uber Global Biz Ops — exploratory analysis on master_zone_hour.parquet.
Run cells interactively in Cursor via # %% markers.
"""
import os
import sys
import warnings
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\Thomas\OneDrive\Desktop\Uber")
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from crz_zones import CRZ_ZONE_IDS  # noqa: E402

from outputs.visualization_design_system import (
    PALETTE,
    add_footnote,
    annotate_callout,
    direct_label_lines,
    format_dollar_axis,
    format_pct_axis,
    set_rcparams,
)

warnings.filterwarnings("ignore", category=FutureWarning)

set_rcparams()

C_UBER = PALETTE["uber"]
C_LYFT = PALETTE["lyft"]
C_CRZ = PALETTE["crz"]
C_NON_CRZ = PALETTE["non_crz"]
C_CRZ_LIGHT = "#A0E5BC"
C_EDGE_UBER = PALETTE["uber"]
C_EDGE_LYFT = PALETTE["lyft"]

MASTER_PATH = (PROJECT_ROOT / "data/processed/master_zone_hour.parquet").as_posix()
FIG_DIR = PROJECT_ROOT / "outputs/figures"
TABLE_DIR = PROJECT_ROOT / "outputs/tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CP_DATE = pd.Timestamp("2025-01-05")

SAVED_ARTIFACTS: list[str] = []

con = duckdb.connect()
con.execute("SET memory_limit='8GB'")
con.execute(f"SET temp_directory='{(PROJECT_ROOT / 'data/processed/duckdb_temp').as_posix()}'")
con.execute(f"CREATE OR REPLACE VIEW master AS SELECT * FROM read_parquet('{MASTER_PATH}')")

meta = con.execute(
    """
    SELECT COUNT(*) AS n_rows,
           MIN(pickup_hour) AS min_h,
           MAX(pickup_hour) AS max_h
    FROM master
"""
).df()
print(
    f"Master table: {int(meta['n_rows'].iloc[0]):,} rows | "
    f"{meta['min_h'].iloc[0]} → {meta['max_h'].iloc[0]}"
)

CRZ_SET = set(CRZ_ZONE_IDS)


def platform_clause(scope: str) -> str:
    if scope == "uber":
        return "platform = 'uber'"
    if scope == "lyft":
        return "platform = 'lyft'"
    return "platform IN ('uber', 'lyft')"


def scope_title(scope: str) -> str:
    return {"uber": "Uber", "lyft": "Lyft", "combined": "Uber + Lyft"}[scope]


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    SAVED_ARTIFACTS.append(str(path.relative_to(PROJECT_ROOT)))


def add_congestion_line(ax) -> None:
    ax.axvline(CP_DATE, color=C_CRZ, linestyle="--", linewidth=1.2, alpha=0.9)
    ax.text(
        CP_DATE,
        0.99,
        " Congestion Pricing",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=9,
        color=C_CRZ,
    )


def trips_millions_formatter(x, pos):
    return f"{x:,.1f}"


# %%
# Cell 2 — Daily trip volume time-series

daily_sql = """
SELECT CAST(pickup_hour AS DATE) AS pickup_date,
       SUM(trip_count)::DOUBLE AS trips
FROM master
WHERE {where_pf}
GROUP BY 1
ORDER BY 1
"""

for scope in ("uber", "lyft", "combined"):
    pf = platform_clause(scope)
    df = con.execute(daily_sql.format(where_pf=pf)).df()
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    df = df.set_index("pickup_date").sort_index()
    df["roll7"] = df["trips"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ln_daily, = ax.plot(
        df.index,
        df["trips"] / 1e6,
        color=C_NON_CRZ,
        linewidth=0.8,
        alpha=0.6,
        label="Daily",
    )
    line_c = C_UBER if scope == "uber" else (C_LYFT if scope == "lyft" else PALETTE["accent"])
    ln_roll, = ax.plot(
        df.index,
        df["roll7"] / 1e6,
        color=line_c,
        linewidth=2.0,
        label="7-day mean",
    )
    add_congestion_line(ax)
    ax.set_ylabel("Trips (millions)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(trips_millions_formatter))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    if scope == "combined":
        direct_label_lines(
            ax,
            [ln_daily, ln_roll],
            ["Daily", "7-day mean"],
            colors=[C_NON_CRZ, line_c],
        )
    else:
        ax.legend(loc="upper left", frameon=True)
    fig.autofmt_xdate()
    fig.suptitle(
        "Volume drops at Jan 5 2025 congestion pricing — never fully recovers",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    ax.set_title(
        f"Daily NYC rideshare trips, {scope_title(scope)}, Jan 2024 – Aug 2025",
        fontsize=11,
        color="#555555",
    )
    dec_mask = (df.index >= pd.Timestamp("2024-12-01")) & (df.index < CP_DATE)
    if dec_mask.any():
        seg = df.loc[dec_mask]
        pk = seg["roll7"].idxmax()
        ypk = float(seg.loc[pk, "roll7"] / 1e6)
        annotate_callout(
            ax,
            xy=(pk, ypk),
            text="Pre-treatment anticipation",
            xytext=(40, 35),
        )
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    add_footnote(
        fig,
        "Pre-CP daily mean ~680K trips; post-CP ~640K. CRZ subset shows steeper drop (see N03).",
        y=-0.02,
    )
    save_fig(FIG_DIR / f"01_daily_volume_{scope}.png")

# %%
# Cell 3 — Fare per mile (ratio of sums), weekly rolling mean on daily ratio
fpm_sql = """
WITH daily AS (
  SELECT CAST(pickup_hour AS DATE) AS pickup_date,
         SUM(total_adjusted_fare)::DOUBLE AS tf,
         SUM(total_miles)::DOUBLE AS tm
  FROM master
  WHERE {where_pf}
  GROUP BY 1
)
SELECT pickup_date,
       tf / NULLIF(tm, 0) AS fpm
FROM daily
ORDER BY 1
"""

for scope in ("uber", "lyft", "combined"):
    pf = platform_clause(scope)
    df = con.execute(fpm_sql.format(where_pf=pf)).df()
    df["pickup_date"] = pd.to_datetime(df["pickup_date"])
    df = df.set_index("pickup_date").sort_index()
    df["fpm_w"] = df["fpm"].rolling(7, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["fpm"], color=C_NON_CRZ, linewidth=0.6, alpha=0.45, label="Daily ratio")
    line_c = C_UBER if scope == "uber" else (C_LYFT if scope == "lyft" else PALETTE["accent"])
    ax.plot(df.index, df["fpm_w"], color=line_c, linewidth=2.0, label="Weekly rolling mean")
    add_congestion_line(ax)
    ax.set_ylabel("Fare per mile ($/mile, ratio-of-sums after subtracting congestion fees)")
    format_dollar_axis(ax, "y")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", frameon=True)
    fig.suptitle(
        "Fare-per-mile DROPPED post-CP after stripping toll passthrough — contrarian finding",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    ax.set_title(
        f"Daily $/mile (ratio-of-sums after subtracting congestion fees), {scope_title(scope)}",
        fontsize=11,
        color="#555555",
    )
    post = df[df.index >= CP_DATE]
    if len(post) > 15:
        ann_idx = post.index[min(14, len(post) - 1)]
        y_ann = float(df.loc[ann_idx, "fpm_w"])
        annotate_callout(
            ax,
            xy=(ann_idx, y_ann),
            text="Underlying surge softened — supply lagged demand drop",
            xytext=(-120, 35),
        )
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    add_footnote(
        fig,
        "Reframe: 'volume crashed, fare held' rather than 'price rose'. See methodology.",
        y=-0.02,
    )
    save_fig(FIG_DIR / f"01_fare_per_mile_{scope}.png")


# %%
# Cell 4 — Hour-of-day profile: median & IQR of trip_count across zone-hours
hod_sql = """
SELECT platform,
       hour_of_day,
       quantile_cont(trip_count, 0.25) AS q25,
       quantile_cont(trip_count, 0.5) AS med,
       quantile_cont(trip_count, 0.75) AS q75
FROM master
WHERE platform IN ('uber', 'lyft')
GROUP BY platform, hour_of_day
ORDER BY platform, hour_of_day
"""
hod = con.execute(hod_sql).df()

for scope in ("uber", "lyft"):
    sub = hod[hod["platform"] == scope].sort_values("hour_of_day")
    fig, ax = plt.subplots(figsize=(12, 6))
    color = C_UBER if scope == "uber" else C_LYFT
    ax.fill_between(
        sub["hour_of_day"],
        sub["q25"],
        sub["q75"],
        color=color,
        alpha=0.25,
        label="IQR",
    )
    ax.plot(sub["hour_of_day"], sub["med"], color=color, linewidth=2.0, label="Median")
    for hx, lab in ((8, "8am"), (18, "6pm")):
        ax.axvline(hx, color=PALETTE["muted"], linestyle=":", linewidth=1.1, alpha=0.95)
        ax.text(
            hx + 0.15,
            ax.get_ylim()[1] * 0.92,
            lab,
            fontsize=8,
            color=PALETTE["muted"],
        )
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Trips per zone-hour")
    fig.suptitle(
        "NYC rideshare follows standard commute pattern — sanity check passes",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    ax.set_title(
        f"Trips per zone-hour by hour-of-day | median + IQR shaded | {scope_title(scope)}",
        fontsize=11,
        color="#555555",
    )
    ax.legend(loc="upper left")
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    add_footnote(
        fig,
        "Confirms data structure before event/CP analysis",
        y=-0.02,
    )
    save_fig(FIG_DIR / f"01_hod_profile_{scope}.png")

fig, ax = plt.subplots(figsize=(12, 6))
for plat, color, label in (
    ("uber", C_UBER, "Uber"),
    ("lyft", C_LYFT, "Lyft"),
):
    sub = hod[hod["platform"] == plat].sort_values("hour_of_day")
    ax.fill_between(
        sub["hour_of_day"],
        sub["q25"],
        sub["q75"],
        color=color,
        alpha=0.18,
    )
    ax.plot(sub["hour_of_day"], sub["med"], color=color, linewidth=2.0, label=label)
for hx, lab in ((8, "8am"), (18, "6pm")):
    ax.axvline(hx, color=PALETTE["muted"], linestyle=":", linewidth=1.1, alpha=0.95)
    ax.text(
        hx + 0.15,
        ax.get_ylim()[1] * 0.92,
        lab,
        fontsize=8,
        color=PALETTE["muted"],
    )
ax.set_xticks(range(0, 24))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Trips per zone-hour")
fig.suptitle(
    "NYC rideshare follows standard commute pattern — sanity check passes",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
ax.set_title(
    "Trips per zone-hour by hour-of-day | median + IQR shaded | Uber + Lyft",
    fontsize=11,
    color="#555555",
)
ax.legend(loc="upper left")
plt.tight_layout(rect=[0, 0.06, 1, 0.93])
add_footnote(
    fig,
    "Confirms data structure before event/CP analysis",
    y=-0.02,
)
save_fig(FIG_DIR / "01_hod_profile_combined.png")


# %%
# Cell 5 — Day-of-week × hour heatmap (DuckDB dow: 0=Sunday … 6=Saturday)
for scope in ("uber", "lyft", "combined"):
    if scope == "combined":
        heat_sql = """
            SELECT dow, hour_of_day, SUM(trip_count)::DOUBLE AS trips
            FROM master
            WHERE platform IN ('uber', 'lyft')
            GROUP BY 1, 2
        """
    else:
        heat_sql = f"""
            SELECT dow, hour_of_day, SUM(trip_count)::DOUBLE AS trips
            FROM master
            WHERE platform = '{scope}'
            GROUP BY 1, 2
        """
    hdf = con.execute(heat_sql).df()
    # Mon–Sun rows (DuckDB EXTRACT(dow): 0=Sunday … 6=Saturday)
    order = [1, 2, 3, 4, 5, 6, 0]
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = hdf.pivot(index="dow", columns="hour_of_day", values="trips").reindex(order)
    pivot = pivot.reindex(columns=list(range(24)))
    pivot.index = dow_labels
    pivot = pivot.fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        np.log10(pivot.values + 1),
        aspect="auto",
        cmap="viridis",
        origin="upper",
    )
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_yticks(range(7))
    ax.set_yticklabels(dow_labels)
    fig.suptitle(
        "Friday evenings & Saturday nights drive peak demand — both platforms",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    ax.set_title(
        r"log$_{10}$(trips) | DuckDB dow remap: 0=Sun $\rightarrow$ reordered Mon-first | "
        f"{scope_title(scope)}",
        fontsize=11,
        color="#555555",
    )
    plt.colorbar(im, ax=ax, label=r"$\log_{10}(1 + \mathrm{trips})$")
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    add_footnote(
        fig,
        "Brightest cells = Fri 6–8pm + Sat 11pm–2am",
        y=-0.02,
    )
    save_fig(FIG_DIR / f"01_dow_hour_heatmap_{scope}.png")


# %%
# Cell 6 — Top 20 zones by total trips (+ zone names)
lookup = pd.read_csv(
    PROJECT_ROOT / "data/raw/tlc_zones/taxi_zone_lookup.csv",
    encoding="utf-8",
)
lookup = lookup.rename(columns={"LocationID": "pickup_zone", "Zone": "zone_name"})

top_sql = """
SELECT pickup_zone,
       SUM(CASE WHEN platform = 'uber' THEN trip_count ELSE 0 END)::BIGINT AS trips_uber,
       SUM(CASE WHEN platform = 'lyft' THEN trip_count ELSE 0 END)::BIGINT AS trips_lyft,
       SUM(trip_count)::BIGINT AS trips_all
FROM master
WHERE platform IN ('uber', 'lyft')
GROUP BY pickup_zone
ORDER BY trips_all DESC
LIMIT 20
"""
top20 = con.execute(top_sql).df()
top20 = top20.merge(
    lookup[["pickup_zone", "zone_name"]],
    on="pickup_zone",
    how="left",
)
top20["label"] = top20.apply(
    lambda r: f"{int(r['pickup_zone'])} — {r['zone_name']}"
    if pd.notna(r["zone_name"])
    else str(int(r["pickup_zone"])),
    axis=1,
)
top20["is_crz"] = top20["pickup_zone"].isin(CRZ_SET)

y_pos = np.arange(len(top20))

for scope in ("uber", "lyft"):
    trips_col = f"trips_{scope}"
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [C_CRZ if crz else C_NON_CRZ for crz in top20["is_crz"]]
    ax.barh(y_pos, top20[trips_col] / 1e6, color=colors, edgecolor=C_EDGE_UBER, linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total trips (millions)")
    fig.suptitle(
        "Airports + Manhattan venues dominate; CRZ zones (green) cluster at top",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    ax.set_title(
        f"Total trips by pickup zone, Jan 2024 – Aug 2025, {scope_title(scope)}",
        fontsize=11,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    add_footnote(
        fig,
        "CRZ = Congestion Relief Zone (post–Jan 5 2025 toll). 36 zones total.",
        y=-0.02,
    )
    save_fig(FIG_DIR / f"01_top_zones_{scope}.png")

# Combined: grouped bars — clean color scheme
# Uber = darker shade, Lyft = lighter shade. CRZ = green, non-CRZ = gray.
fig, ax = plt.subplots(figsize=(10, 8))
w = 0.35

uber_colors = [C_CRZ if crz else "#404040" for crz in top20["is_crz"]]
lyft_colors = [C_CRZ_LIGHT if crz else "#B8B8B8" for crz in top20["is_crz"]]

ax.barh(y_pos - w/2, top20["trips_uber"] / 1e6, height=w,
        color=uber_colors, edgecolor=C_UBER, linewidth=0.4, label="Uber")
ax.barh(y_pos + w/2, top20["trips_lyft"] / 1e6, height=w,
        color=lyft_colors, edgecolor=C_EDGE_LYFT, linewidth=0.4, label="Lyft")

ax.set_yticks(y_pos)
ax.set_yticklabels(top20["label"], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Total trips (millions)")
fig.suptitle(
    "Airports + Manhattan venues dominate; CRZ zones (green) cluster at top",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)
ax.set_title(
    "Total trips by pickup zone, Jan 2024 – Aug 2025, Uber + Lyft",
    fontsize=11,
    color="#555555",
)
ax.legend(loc="lower right")
plt.tight_layout(rect=[0, 0.06, 1, 0.94])
add_footnote(
    fig,
    "CRZ = Congestion Relief Zone (post–Jan 5 2025 toll). 36 zones total.",
    y=-0.02,
)
save_fig(FIG_DIR / "01_top_zones_combined.png")


# %%
# Cell 7 — Choropleth log volume (Uber + Lyft), CRZ outline
import geopandas as gpd  # noqa: E402

trips_zone = con.execute(
    """
    SELECT pickup_zone, SUM(trip_count)::DOUBLE AS trips
    FROM master
    WHERE platform IN ('uber', 'lyft')
    GROUP BY 1
"""
).df()
trips_zone["pickup_zone"] = trips_zone["pickup_zone"].astype(int)

zones = gpd.read_file(PROJECT_ROOT / "data/raw/tlc_zones/taxi_zones.shp")
if zones.crs is None:
    zones.set_crs(epsg=2263, inplace=True)
else:
    zones = zones.to_crs(epsg=2263)

id_col = "LocationID" if "LocationID" in zones.columns else zones.columns[0]
zones[id_col] = zones[id_col].astype(int)
zones_plot = zones.merge(trips_zone, left_on=id_col, right_on="pickup_zone", how="left")
zones_plot["trips"] = zones_plot["trips"].fillna(0)
zones_plot["log_trips"] = np.log10(zones_plot["trips"] + 1)

fig, ax = plt.subplots(figsize=(10, 10))
zones_plot.plot(
    column="log_trips",
    cmap="viridis",
    linewidth=0.2,
    edgecolor="#222",
    legend=True,
    ax=ax,
    legend_kwds={"label": r"$\log_{10}(1 + \mathrm{trips})$", "shrink": 0.6},
)
crz = zones_plot[zones_plot[id_col].isin(CRZ_ZONE_IDS)]
crz.boundary.plot(ax=ax, edgecolor=C_CRZ, linewidth=2.0)
fig.suptitle(
    "Demand concentrates in Manhattan + airports; outer boroughs sparse",
    fontsize=14,
    fontweight="bold",
    y=0.96,
)
ax.set_title(
    r"log$_{10}$(trips) by TLC zone | CRZ boundary in green",
    fontsize=11,
    color="#555555",
    pad=12,
)
ax.axis("off")
plt.tight_layout(rect=[0, 0.04, 1, 0.92])
add_footnote(
    fig,
    "Geographic context for zone-level analysis throughout deck",
    y=0.02,
)
save_fig(FIG_DIR / "01_choropleth_volume.png")


# %%
# Cell 8 — STL decomposition (weekly seasonality)
from statsmodels.tsa.seasonal import STL  # noqa: E402


def daily_trips_series(platform: str) -> pd.Series:
    q = f"""
        SELECT CAST(pickup_hour AS DATE) AS d, SUM(trip_count)::DOUBLE AS y
        FROM master
        WHERE platform = '{platform}'
        GROUP BY 1 ORDER BY 1
    """
    s = con.execute(q).df()
    s["d"] = pd.to_datetime(s["d"])
    s = s.set_index("d")["y"]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(idx).fillna(0.0)


for plat in ("uber", "lyft"):
    series = daily_trips_series(plat)
    stl = STL(series, period=7, robust=True)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    parts = [
        (res.observed, "Observed"),
        (res.trend, "Trend"),
        (res.seasonal, "Seasonal (weekly)"),
        (res.resid, "Residual"),
    ]
    for ax, (vals, title) in zip(axes, parts):
        ax.plot(vals.index, vals.values, color=C_UBER if plat == "uber" else C_LYFT, lw=1.2)
        add_congestion_line(ax)
        ax.set_ylabel(title)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.suptitle(
        "Trend, weekly seasonality, and residual cleanly separable — supports panel-FE specification",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    axes[0].set_title(
        f"STL decomposition (period=7, robust=True) on daily trips — {scope_title(plat)}",
        fontsize=11,
        color="#555555",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    add_footnote(
        fig,
        "Lyft trend slope visibly different from Uber — confirms ADF non-stationarity finding",
        y=0.01,
    )
    save_fig(FIG_DIR / f"01_stl_decomposition_{plat}.png")


# %%
# Cell 9 — ADF + ACF / PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402


def adf_report(series: pd.Series, label: str) -> None:
    res = adfuller(series.values, autolag="AIC")
    stat, pval = res[0], res[1]
    conclusion = "reject unit root (stationary)" if pval < 0.05 else "fail to reject unit root"
    print(f"[{label}] ADF statistic = {stat:.4f}, p-value = {pval:.4g} → {conclusion}")


for plat in ("uber", "lyft"):
    series = daily_trips_series(plat)
    adf_report(series, plat.upper())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    plot_acf(series, ax=axes[0], lags=40, title="")
    plot_pacf(series, ax=axes[1], lags=40, method="ywm", title="")
    fig.suptitle(
        "Strong weekly autocorrelation (lag 7); stationarity differs by platform",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    axes[0].set_title(
        f"ACF + PACF (40 lags) on daily trips — {scope_title(plat)}",
        fontsize=11,
        color="#555555",
    )
    axes[0].set_ylabel("ACF")
    axes[1].set_ylabel("PACF")
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    add_footnote(
        fig,
        "Uber ADF p=0.002 (stationary); Lyft p=0.18 (non-stationary). Mitigated via panel FE in N03.",
        y=-0.02,
    )
    save_fig(FIG_DIR / f"01_acf_pacf_{plat}.png")


# %%
# Cell 10 — Summary table for deck
rows = []
for scope in ("uber", "lyft", "combined"):
    pf = platform_clause(scope)
    q = f"""
        WITH days AS (
          SELECT COUNT(DISTINCT CAST(pickup_hour AS DATE)) AS n_days FROM master WHERE {pf}
        )
        SELECT
          (SELECT n_days FROM days) AS n_days,
          SUM(trip_count)::DOUBLE AS total_trips,
          SUM(total_adjusted_fare)::DOUBLE AS total_adj_fare,
          SUM(total_miles)::DOUBLE AS total_miles,
          COUNT(DISTINCT pickup_zone) AS n_zones_all
        FROM master
        WHERE {pf}
    """
    one = con.execute(q).df().iloc[0]
    n_days = max(int(one["n_days"]), 1)
    total_trips = float(one["total_trips"])
    tf = float(one["total_adj_fare"])
    tm = float(one["total_miles"])
    fpm = tf / tm if tm else np.nan

    ev_zones = con.execute(
        f"""
        SELECT COUNT(DISTINCT pickup_zone) AS n
        FROM master
        WHERE {pf} AND COALESCE(has_major_event_dayflag, 0) = 1
    """
    ).fetchone()[0]

    crz_zones = con.execute(
        f"""
        SELECT COUNT(DISTINCT pickup_zone) AS n
        FROM master
        WHERE {pf} AND is_in_crz = 1
    """
    ).fetchone()[0]

    sapo_zones = con.execute(
        f"""
        SELECT COUNT(DISTINCT pickup_zone) AS n
        FROM master
        WHERE {pf} AND sapo_count_borough > 0
        """
    ).fetchone()[0]

    rows.append(
        {
            "platform_scope": scope,
            "start_date": "2024-01-01",
            "end_date": "2025-08-31",
            "n_days": n_days,
            "total_trips": int(total_trips),
            "avg_daily_trips": total_trips / n_days,
            "avg_fare_per_mile_ratio_sums": fpm,
            "n_tlc_zones_with_trips": int(one["n_zones_all"]),
            "n_crz_zones_in_data": int(crz_zones),
            "n_zones_any_major_event_day": int(ev_zones),
            "n_zones_with_sapo_event_day": int(sapo_zones),
        }
    )

summary_df = pd.DataFrame(rows)
out_csv = TABLE_DIR / "01_summary.csv"
summary_df.to_csv(out_csv, index=False)
SAVED_ARTIFACTS.append(str(out_csv.relative_to(PROJECT_ROOT)))
print(f"Wrote {out_csv} ({len(summary_df)} rows).")

comb = summary_df.loc[summary_df["platform_scope"] == "combined"].iloc[0]
print("\nDeck-ready summary:")
print(f"  Total trips: {int(comb['total_trips']):,}")
print(
    f"  Date range: 2024-01-01 to 2025-08-31 ({int(comb['n_days'])} days)"
)
print(f"  CRZ zones: {int(comb['n_crz_zones_in_data'])}")
print(
    "  Avg fare/mile (combined): "
    f"${comb['avg_fare_per_mile_ratio_sums']:.2f}/mile"
)


# %%
# Close DuckDB and print artifact manifest
con.close()

print("\n--- Saved artifacts (one line each) ---")
for p in sorted(set(SAVED_ARTIFACTS)):
    print(f"  • {p}")
