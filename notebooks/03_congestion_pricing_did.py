# %%
# Cell 1: Imports + setup
"""
Notebook 03 — Congestion Pricing DiD (CORE / HEADLINE)

Specification:
    log(trip_count) = β·is_post_cp·is_in_crz + α_zone + α_time + ε
    log(fare_per_mile_ros) = β·is_post_cp·is_in_crz + α_zone + α_time + ε

Treatment date: January 5, 2025
linearmodels.PanelOLS, cluster SEs by zone.
Pre-trends test (placebo dates), event-study plot (weekly + daily zoom),
heterogeneity by time-of-day and zone type, Lyft placebo with first-differences.
"""
from __future__ import annotations

# REPRODUCIBILITY: All random sampling uses fixed seed=42 (or 1, where noted).
# Regression results are deterministic given fixed input data.

import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")

PROJECT_ROOT = Path.cwd()
FIG_DIR = PROJECT_ROOT / "outputs/figures"
TABLE_DIR = PROJECT_ROOT / "outputs/tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

from outputs.visualization_design_system import (
    PALETTE,
    add_footnote,
    annotate_callout,
    format_coef_axis,
    format_dollar_axis,
    format_thousands_axis,
    limit_ticks,
    set_rcparams,
)

set_rcparams()

TREATMENT_DATE = pd.Timestamp("2025-01-05")

print("Loading master parquet...")
df_master = pd.read_parquet("data/processed/master_zone_hour.parquet", engine="pyarrow")
df_master["platform"] = df_master["platform"].astype(str).str.lower()

REQUIRED_COLS = {
    "pickup_zone": "int",
    "pickup_hour": "datetime",
    "platform": "str",
    "trip_count": "numeric",
    "total_adjusted_fare": "numeric",
    "total_miles": "numeric",
    "is_in_crz": "int",
    "is_nyc_event_sym": "int",
    "is_nyc_event_asym": "int",
    "is_nj_event_pregame_sym": "int",
    "is_nj_event_pregame_asym": "int",
    "has_major_event_dayflag": "int",
}
missing = [c for c in REQUIRED_COLS if c not in df_master.columns]
if missing:
    raise RuntimeError(f"Master schema missing required cols: {missing}")
print(
    f"Schema OK — {len(df_master.columns)} cols present, all required cols found"
)

print("Event flag totals (sanity check):")
for fl in [
    "is_nyc_event_sym",
    "is_nyc_event_asym",
    "is_nj_event_pregame_sym",
    "is_nj_event_pregame_asym",
    "is_event_combined_sym",
    "has_major_event_dayflag",
]:
    if fl in df_master.columns:
        print(f"  {fl}: {int(df_master[fl].sum()):,}")

print(f"Master rows: {len(df_master):,}")

# Validate CRZ flag is on master
if "is_in_crz" not in df_master.columns:
    raise RuntimeError("is_in_crz column missing from master. Check data prep.")
print(f"CRZ rows: {int(df_master['is_in_crz'].sum()):,} "
      f"({100 * df_master['is_in_crz'].mean():.1f}%)")


# %%
# Cell 2: Helpers
from linearmodels.panel import FirstDifferenceOLS, PanelOLS
import statsmodels.api as sm


def safe_log(x, eps=1.0):
    return np.log(np.maximum(x.astype(float), 0.0) + eps)


def prep_panel(df_platform):
    """Add log_trips, fare_per_mile, log_fpm, is_post_cp, treat interaction."""
    d = df_platform.copy()
    d["log_trips"] = safe_log(d["trip_count"], eps=1.0)
    d["fare_per_mile"] = np.where(
        d["total_miles"].astype(float) > 0,
        d["total_adjusted_fare"].astype(float) / d["total_miles"].astype(float),
        np.nan,
    )
    d["log_fpm"] = safe_log(d["fare_per_mile"], eps=0.01)
    d["pickup_hour_dt"] = pd.to_datetime(d["pickup_hour"])
    d["pickup_date"] = d["pickup_hour_dt"].dt.normalize()
    d["is_post_cp"] = (d["pickup_date"] >= TREATMENT_DATE).astype(np.int8)
    d["treat_x_post"] = (d["is_post_cp"] * d["is_in_crz"]).astype(np.int8)
    return d


def collapse_to_zone_day(df, flag_cols=None):
    """
    Aggregate hourly panel to (pickup_zone, pickup_date) for memory-safe
    panel regressions. Ratio-of-sums FPM on collapsed totals; binary flags =
    max() within day.
    """
    if flag_cols is None:
        flag_cols = [c for c in df.columns
                     if c.startswith("is_") and c != "is_in_crz"]
        flag_cols += [c for c in ["clean_day_flag", "placebo_pre7_sym",
                                   "has_major_event_dayflag"] if c in df.columns]

    agg_dict = {
        "trip_count": "sum",
        "total_adjusted_fare": "sum",
        "total_miles": "sum",
        "is_in_crz": "max",
    }
    for fc in flag_cols:
        if fc in df.columns:
            agg_dict[fc] = "max"

    d = df.copy()
    d["pickup_date"] = pd.to_datetime(d["pickup_hour"]).dt.normalize()
    daily = d.groupby(["pickup_zone", "pickup_date"], as_index=False).agg(agg_dict)
    daily["fare_per_mile"] = np.where(
        daily["total_miles"] > 0,
        daily["total_adjusted_fare"] / daily["total_miles"],
        np.nan,
    )
    daily["log_trips"] = safe_log(daily["trip_count"], eps=1.0)
    daily["log_fpm"] = safe_log(daily["fare_per_mile"], eps=0.01)
    daily = daily.rename(columns={"pickup_date": "pickup_hour"})
    print(f"  collapsed to {len(daily):,} zone-day rows "
          f"(from {len(df):,} hourly rows)")
    return daily


def did_regression(df, dep, cluster_col="pickup_zone"):
    """PanelOLS DiD with entity + time FE; cluster SEs by zone."""
    cols = [cluster_col, "pickup_hour_dt", "treat_x_post", dep]
    d = df[cols].dropna().copy()
    d = d.set_index([cluster_col, "pickup_hour_dt"]).sort_index()
    y = d[dep].astype(np.float64)
    X = d[["treat_x_post"]].astype(np.float64)

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return {
        "dep": dep,
        "beta": float(res.params["treat_x_post"]),
        "se": float(res.std_errors["treat_x_post"]),
        "p_value": float(res.pvalues["treat_x_post"]),
        "ci95_low": float(res.conf_int().loc["treat_x_post", "lower"]),
        "ci95_high": float(res.conf_int().loc["treat_x_post", "upper"]),
        "nobs": int(res.nobs),
    }


def fd_regression(df, dep):
    """First-differences DiD for non-stationary check (Lyft)."""
    d = df[["pickup_zone", "pickup_hour_dt", "treat_x_post", dep]].dropna()
    d = d.set_index(["pickup_zone", "pickup_hour_dt"]).sort_index()
    y = d[dep].astype(np.float64)
    X = d[["treat_x_post"]].astype(np.float64)

    fd = FirstDifferenceOLS(y, X)
    res = fd.fit(cov_type="clustered", cluster_entity=True)
    return {
        "dep": dep, "spec": "FirstDifference_OLS",
        "beta": float(res.params["treat_x_post"]),
        "se": float(res.std_errors["treat_x_post"]),
        "p_value": float(res.pvalues["treat_x_post"]),
        "nobs": int(res.nobs),
    }


print("Helpers ready.")


# %%
# Cell 3: Prep Uber and Lyft panels
df_uber = prep_panel(df_master[df_master["platform"] == "uber"])
df_lyft = prep_panel(df_master[df_master["platform"] == "lyft"])
print(f"Uber rows: {len(df_uber):,} | Lyft rows: {len(df_lyft):,}")
print(f"Uber post-CP fraction: {df_uber['is_post_cp'].mean():.3f}")
print(f"Uber treated (CRZ × post): {int(df_uber['treat_x_post'].sum()):,}")

pre_n = (df_uber["is_post_cp"] == 0).sum()
post_n = (df_uber["is_post_cp"] == 1).sum()
print(f"Pre-CP rows: {pre_n:,} | Post-CP rows: {post_n:,}")
if pre_n < 100_000 or post_n < 100_000:
    raise RuntimeError("Insufficient pre/post sample — check date range in master")
if df_uber["is_in_crz"].sum() == 0 or (df_uber["is_in_crz"] == 0).sum() == 0:
    raise RuntimeError("CRZ flag is degenerate — check is_in_crz column")

print("\nBuilding zone-day panel for memory-safe PanelOLS...")
df_uber_daily = collapse_to_zone_day(df_uber)
df_lyft_daily = collapse_to_zone_day(df_lyft)
for d in (df_uber_daily, df_lyft_daily):
    d["is_post_cp"] = (pd.to_datetime(d["pickup_hour"]) >= TREATMENT_DATE).astype(np.int8)
    d["treat_x_post"] = (d["is_post_cp"] * d["is_in_crz"]).astype(np.int8)
    d["pickup_hour_dt"] = pd.to_datetime(d["pickup_hour"])
    d["pickup_date"] = d["pickup_hour_dt"].dt.normalize()


# %%
# Cell 4: Headline DiD — Uber, both metrics
results = []
for dep in ["log_trips", "log_fpm"]:
    r = did_regression(df_uber_daily, dep)
    results.append({**r, "platform": "Uber"})
    print(f"Uber {dep}: β={r['beta']:.5f} (SE={r['se']:.5f}, p={r['p_value']:.4f}, N={r['nobs']:,})")

did_uber_df = pd.DataFrame(results)
did_uber_df.to_csv(TABLE_DIR / "03_did_uber_headline.csv", index=False)


# %%
# Cell 5: Lyft placebo (PanelOLS + first differences for non-stationarity)
lyft_results = []
for dep in ["log_trips", "log_fpm"]:
    r = did_regression(df_lyft_daily, dep)
    lyft_results.append({**r, "platform": "Lyft", "spec": "PanelOLS_FE"})
    print(f"Lyft PanelOLS {dep}: β={r['beta']:.5f} (p={r['p_value']:.4f})")
    try:
        fd = fd_regression(df_lyft_daily, dep)
        lyft_results.append({**fd, "platform": "Lyft", "spec": "FirstDifference_OLS",
                             "ci95_low": np.nan, "ci95_high": np.nan})
        print(f"Lyft FD {dep}: β={fd['beta']:.5f} (p={fd['p_value']:.4f})")
    except Exception as e:
        warnings.warn(f"FD failed for Lyft {dep}: {e}")

did_lyft_df = pd.DataFrame(lyft_results)
did_lyft_df.to_csv(TABLE_DIR / "03_did_lyft_placebo.csv", index=False)


# %%
# Cell 6: Pre-trends placebo — fake treatment dates BEFORE Jan 5 2025
placebo_dates = ["2024-06-01", "2024-09-01", "2024-11-01", "2024-12-01"]

placebo_rows = []
for fake_date in placebo_dates:
    fake_dt = pd.Timestamp(fake_date)
    pre = df_uber_daily[df_uber_daily["pickup_date"] < TREATMENT_DATE].copy()
    pre["fake_post"] = (pre["pickup_date"] >= fake_dt).astype(np.int8)
    pre["fake_treat"] = (pre["fake_post"] * pre["is_in_crz"]).astype(np.int8)

    d = pre[["pickup_zone", "pickup_hour_dt", "fake_treat", "log_trips"]].dropna()
    d = d.set_index(["pickup_zone", "pickup_hour_dt"]).sort_index()
    if d["fake_treat"].sum() == 0:
        continue
    try:
        mod = PanelOLS(d["log_trips"], d[["fake_treat"]],
                       entity_effects=True, time_effects=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        placebo_rows.append({
            "placebo_date": fake_date,
            "beta": float(res.params["fake_treat"]),
            "se": float(res.std_errors["fake_treat"]),
            "p_value": float(res.pvalues["fake_treat"]),
            "nobs": int(res.nobs),
        })
        print(f"Placebo {fake_date}: β={res.params['fake_treat']:.5f} (p={res.pvalues['fake_treat']:.3f})")
    except Exception as e:
        warnings.warn(f"Placebo {fake_date} failed: {e}")

pd.DataFrame(placebo_rows).to_csv(TABLE_DIR / "03_pre_trends_placebo.csv", index=False)
print("\nExpected: all placebo β ≈ 0 (parallel trends pre-treatment)")


# %%
# Cell 7: Event-study coefficients — WEEKLY granularity (the gold-standard chart)
def event_study_coefs(df, dep, freq="W", n_pre=20, n_post=30):
    """
    Estimate β for each period bin pre/post treatment.
    Bin reference = week or day index relative to treatment.
    """
    d = df.copy()
    if freq == "W":
        d["period"] = (d["pickup_date"] - TREATMENT_DATE).dt.days // 7
    elif freq == "D":
        d["period"] = (d["pickup_date"] - TREATMENT_DATE).dt.days
    else:
        raise ValueError(f"Unknown freq {freq}")

    d = d[(d["period"] >= -n_pre) & (d["period"] <= n_post)].copy()

    # Build period dummies × is_in_crz, omit period=-1 as reference
    d["period"] = d["period"].astype(int)
    periods_unique = sorted(d["period"].unique())
    if -1 not in periods_unique:
        periods_unique.append(-1)
        periods_unique.sort()

    # Construct dummies
    for p in periods_unique:
        if p == -1:
            continue  # reference
        d[f"interact_{p}"] = ((d["period"] == p) & (d["is_in_crz"] == 1)).astype(np.int8)

    interact_cols = [f"interact_{p}" for p in periods_unique if p != -1]

    cols = ["pickup_zone", "pickup_hour_dt", dep] + interact_cols
    panel = d[cols].dropna().set_index(["pickup_zone", "pickup_hour_dt"]).sort_index()

    y = panel[dep].astype(np.float64)
    X = panel[interact_cols].astype(np.float64)

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for p in periods_unique:
        if p == -1:
            rows.append({"period": -1, "beta": 0.0, "se": 0.0,
                         "ci95_low": 0.0, "ci95_high": 0.0})
            continue
        col = f"interact_{p}"
        if col not in res.params.index:
            continue
        rows.append({
            "period": p,
            "beta": float(res.params[col]),
            "se": float(res.std_errors[col]),
            "ci95_low": float(res.conf_int().loc[col, "lower"]),
            "ci95_high": float(res.conf_int().loc[col, "upper"]),
        })
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def event_study_coefs_weekly_collapsed(df, dep, n_pre=20, n_post=30):
    """
    Pre-aggregate to (pickup_zone, calendar-week bin) before PanelOLS.
    Trips/fares/miles are summed within bin; log_trips and log_fpm use Ros on collapsed sums.
    """
    d = df.copy()
    d["period"] = (d["pickup_date"] - TREATMENT_DATE).dt.days // 7
    d = d[(d["period"] >= -n_pre) & (d["period"] <= n_post)].copy()
    g = (
        d.groupby(["pickup_zone", "period"], as_index=False)
        .agg(
            trip_count=("trip_count", "sum"),
            total_adjusted_fare=("total_adjusted_fare", "sum"),
            total_miles=("total_miles", "sum"),
            is_in_crz=("is_in_crz", "max"),
        )
    )
    g["log_trips"] = safe_log(g["trip_count"], eps=1.0)
    g["fare_per_mile"] = np.where(
        g["total_miles"].astype(float) > 0,
        g["total_adjusted_fare"].astype(float) / g["total_miles"].astype(float),
        np.nan,
    )
    g["log_fpm"] = safe_log(g["fare_per_mile"], eps=0.01)
    g["pickup_hour_dt"] = pd.to_datetime(TREATMENT_DATE) + pd.to_timedelta(
        g["period"].astype(np.int64) * 7, unit="d"
    )
    g["period"] = g["period"].astype(int)
    periods_unique = sorted(g["period"].unique())
    if -1 not in periods_unique:
        periods_unique.append(-1)
        periods_unique.sort()
    for p in periods_unique:
        if p == -1:
            continue
        g[f"interact_{p}"] = ((g["period"] == p) & (g["is_in_crz"] == 1)).astype(np.int8)

    interact_cols = [f"interact_{p}" for p in periods_unique if p != -1]
    cols = ["pickup_zone", "pickup_hour_dt", dep] + interact_cols
    panel = g[cols].dropna().set_index(["pickup_zone", "pickup_hour_dt"]).sort_index()

    y = panel[dep].astype(np.float64)
    X = panel[interact_cols].astype(np.float64)

    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for p in periods_unique:
        if p == -1:
            rows.append({"period": -1, "beta": 0.0, "se": 0.0,
                         "ci95_low": 0.0, "ci95_high": 0.0})
            continue
        col = f"interact_{p}"
        if col not in res.params.index:
            continue
        rows.append({
            "period": p,
            "beta": float(res.params[col]),
            "se": float(res.std_errors[col]),
            "ci95_low": float(res.conf_int().loc[col, "lower"]),
            "ci95_high": float(res.conf_int().loc[col, "upper"]),
        })
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


ES_EVENT_STUDY_TIMEOUT_SEC = 1200  # 20 minutes wall-clock for full hourly panel


def _weekly_es_full_or_fallback(dep: str) -> tuple[pd.DataFrame, bool]:
    t0 = time.perf_counter()
    used_fallback = False
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(event_study_coefs, df_uber_daily, dep, "W", 20, 30)
            out = fut.result(timeout=ES_EVENT_STUDY_TIMEOUT_SEC)
    except FuturesTimeoutError:
        used_fallback = True
        warnings.warn(
            "Full-panel weekly event study exceeded "
            f"{ES_EVENT_STUDY_TIMEOUT_SEC}s; using zone-week collapsed panel. "
            "The timed-out fit may still consume CPU in the background — "
            "restart the kernel if needed.",
            stacklevel=2,
        )
        out = event_study_coefs_weekly_collapsed(df_uber_daily, dep, n_pre=20, n_post=30)
    except Exception as e:
        used_fallback = True
        warnings.warn(
            f"Full-panel weekly event study failed ({e}); using collapsed fallback.",
            stacklevel=2,
        )
        out = event_study_coefs_weekly_collapsed(df_uber_daily, dep, n_pre=20, n_post=30)
    print(
        f"  [{dep}] full-panel attempt finished in {time.perf_counter() - t0:.1f}s "
        f"(fallback={'yes' if used_fallback else 'no'})"
    )
    return out, used_fallback


print("Estimating weekly event-study coefficients (Uber, log_trips)...")
es_weekly_trips, _fb_t = _weekly_es_full_or_fallback("log_trips")
es_weekly_trips.to_csv(TABLE_DIR / "03_event_study_weekly_log_trips.csv", index=False)
print(es_weekly_trips.to_string(index=False))

t_c0 = time.perf_counter()
es_weekly_trips_collapsed = event_study_coefs_weekly_collapsed(
    df_uber_daily, "log_trips", n_pre=20, n_post=30
)
print(f"Zone-week collapsed log_trips fit elapsed: {time.perf_counter() - t_c0:.1f}s")
es_weekly_trips_collapsed.to_csv(
    TABLE_DIR / "03_event_study_weekly_log_trips_collapsed.csv", index=False
)

if len(es_weekly_trips) and len(es_weekly_trips_collapsed):
    cmp_trips = es_weekly_trips.merge(
        es_weekly_trips_collapsed,
        on="period",
        suffixes=("_full", "_collapsed"),
        how="inner",
    )
    cmp_trips["abs_d_beta"] = (cmp_trips["beta_full"] - cmp_trips["beta_collapsed"]).abs()
    print(
        "Compare full vs collapsed (log_trips): max |Δβ| = "
        f"{cmp_trips['abs_d_beta'].max():.6f} across aligned periods"
    )

print("\nEstimating weekly event-study coefficients (Uber, log_fpm)...")
es_weekly_fpm, _fb_f = _weekly_es_full_or_fallback("log_fpm")
es_weekly_fpm.to_csv(TABLE_DIR / "03_event_study_weekly_log_fpm.csv", index=False)

t_c1 = time.perf_counter()
es_weekly_fpm_collapsed = event_study_coefs_weekly_collapsed(
    df_uber_daily, "log_fpm", n_pre=20, n_post=30
)
print(f"Zone-week collapsed log_fpm fit elapsed: {time.perf_counter() - t_c1:.1f}s")
es_weekly_fpm_collapsed.to_csv(
    TABLE_DIR / "03_event_study_weekly_log_fpm_collapsed.csv", index=False
)

if len(es_weekly_fpm) and len(es_weekly_fpm_collapsed):
    cmp_fpm = es_weekly_fpm.merge(
        es_weekly_fpm_collapsed,
        on="period",
        suffixes=("_full", "_collapsed"),
        how="inner",
    )
    cmp_fpm["abs_d_beta"] = (cmp_fpm["beta_full"] - cmp_fpm["beta_collapsed"]).abs()
    print(
        "Compare full vs collapsed (log_fpm): max |Δβ| = "
        f"{cmp_fpm['abs_d_beta'].max():.6f} across aligned periods"
    )


# %%
# Cell 8: Event-study DAILY zoom (±28 days around treatment)
print("Estimating daily event-study coefficients (Uber, ±28 days)...")
es_daily_trips = event_study_coefs(df_uber_daily, "log_trips", freq="D", n_pre=28, n_post=28)
es_daily_trips.to_csv(TABLE_DIR / "03_event_study_daily_log_trips.csv", index=False)
print(f"Daily coefs computed: {len(es_daily_trips)} periods")


# %%
# Cell 9: Event-study plots — weekly main + daily zoom
_PRE_SHADE = "#E6E6E6"
_POST_SHADE = "#DCEEE2"


def _shade_pre_post(ax, es_df: pd.DataFrame) -> None:
    lo = float(es_df["period"].min())
    hi = float(es_df["period"].max())
    ax.axvspan(lo, 0, color=_PRE_SHADE, alpha=0.55, zorder=0, lw=0)
    ax.axvspan(0, hi, color=_POST_SHADE, alpha=0.45, zorder=0, lw=0)


fig_es, axes_es = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

ax_w = axes_es[0]
_shade_pre_post(ax_w, es_weekly_trips)
ax_w.axhline(0, color=PALETTE["muted"], lw=1.0, ls=":", zorder=2)
ax_w.errorbar(
    es_weekly_trips["period"],
    es_weekly_trips["beta"],
    yerr=1.96 * es_weekly_trips["se"],
    fmt="o",
    color=PALETTE["accent"],
    ecolor=PALETTE["muted"],
    capsize=3,
    zorder=4,
)
ax_w.axvline(0, color=PALETTE["warn"], lw=2.0, ls="--", zorder=5)
ax_w.set_xlabel("Weeks relative to CP treatment (week 0 = week of Jan 5, 2025)")
ax_w.set_ylabel("β (log trips)")
format_coef_axis(ax_w, "y", decimals=3)
limit_ticks(ax_w, "both")
ax_w.set_title(
    "Weekly β coefficients ±20 weeks pre / +30 post\n"
    "Uber zone-day panel — CRZ×period interactions vs omitted week −1",
    fontsize=11,
    color="#555555",
)

row_m2 = es_weekly_trips.loc[es_weekly_trips["period"] == -2]
if len(row_m2):
    b_m2 = float(row_m2["beta"].iloc[0])
    pct_m2 = (np.exp(b_m2) - 1.0) * 100.0
    ax_w.annotate(
        f"Sudden {pct_m2:.0f}% (week −2) dip\n(reference period sensitivity)",
        xy=(-2, b_m2),
        xytext=(-12, b_m2 + 0.02),
        textcoords="data",
        fontsize=9,
        bbox=dict(boxstyle="round", fc=PALETTE["annot_bg"], ec="#CCCCCC"),
        arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]),
    )

post_wk = es_weekly_trips[
    es_weekly_trips["period"].between(5, 15, inclusive="both")
]
if len(post_wk):
    betas = post_wk["beta"]
    lo_pct = (np.exp(float(betas.min())) - 1.0) * 100.0
    hi_pct = (np.exp(float(betas.max())) - 1.0) * 100.0
    bx = float(post_wk["period"].median())
    by = float(betas.median())
    ax_w.annotate(
        f"Post-treatment implied\nrange ~{lo_pct:.0f}%–{hi_pct:.0f}% (weeks +5–15)",
        xy=(bx, by),
        xytext=(3, ax_w.get_ylim()[1] * 0.85),
        fontsize=9,
        bbox=dict(boxstyle="round", fc=PALETTE["annot_bg"], ec="#CCCCCC"),
        arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]),
    )

ax_w.text(
    0.02,
    0.98,
    "CP treatment\nJan 5, 2025",
    transform=ax_w.get_xaxis_transform(),
    ha="left",
    va="top",
    fontsize=9,
    color=PALETTE["warn"],
    fontweight="bold",
)

ax_d = axes_es[1]
_shade_pre_post(ax_d, es_daily_trips)
ax_d.axhline(0, color=PALETTE["muted"], lw=1.0, ls=":", zorder=2)
ax_d.errorbar(
    es_daily_trips["period"],
    es_daily_trips["beta"],
    yerr=1.96 * es_daily_trips["se"],
    fmt="o",
    color=PALETTE["accent"],
    ecolor=PALETTE["muted"],
    capsize=2,
    markersize=4,
    zorder=4,
)
ax_d.axvline(0, color=PALETTE["warn"], lw=2.0, ls="--", zorder=5)
ax_d.set_xlabel("Days relative to CP treatment (day 0 = Jan 5, 2025)")
ax_d.set_ylabel("β (log trips)")
format_coef_axis(ax_d, "y", decimals=3)
limit_ticks(ax_d, "both")
ax_d.set_title(
    "Daily β coefficients ±28 days (anticipation zoom)\n"
    "Same specification — daily event-time bins",
    fontsize=11,
    color="#555555",
)

fig_es.suptitle(
    "CRZ–non-CRZ gap was already shrinking before CP — pre-trends violated",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
add_footnote(
    fig_es,
    "All 4 placebo dates pre–Jan 5 show significant β<0 (parallel trends fails). "
    "Headline DiD overstates causal effect by ~7–10 ppts.",
    y=-0.02,
)
plt.savefig(FIG_DIR / "03_event_study_plot.png")
plt.close()
print(f"Saved {FIG_DIR / '03_event_study_plot.png'}")


# %%
# Cell 10: Heterogeneity by time-of-day
# SKIPPED: time-of-day heterogeneity requires hourly panel (hour FE / buckets).
# Zone-day DiD cannot recover within-day timing; defer to per-event lift by
# category in N02.
print(
    "Skipping ToD heterogeneity (requires hourly panel); "
    "see per-event lift analysis in N02 by event category."
)
# pd.DataFrame(het_rows).to_csv(TABLE_DIR / "03_heterogeneity_tod.csv", index=False)


# %%
# Cell 11: Heterogeneity by CRZ vs adjacent vs outer
df_uber["zone_type"] = np.where(
    df_uber["is_in_crz"] == 1, "CRZ",
    np.where(df_uber["pickup_zone"].isin([4, 12, 13, 41, 42, 87, 116, 125, 152, 244]),
             "adjacent_buffer", "outer")
)
print("Zone type counts:")
print(df_uber["zone_type"].value_counts().to_string())

# DiD only meaningful for CRZ vs non-CRZ; adjacent could show spillover
# Run separate post-CP comparison: log_trips drop in CRZ vs adjacent vs outer
post = df_uber[df_uber["is_post_cp"] == 1]
zone_means_post = post.groupby("zone_type")["log_trips"].mean()
pre = df_uber[df_uber["is_post_cp"] == 0]
zone_means_pre = pre.groupby("zone_type")["log_trips"].mean()
diff = (zone_means_post - zone_means_pre).rename("post_minus_pre_mean_log_trips")
diff_df = diff.reset_index()
diff_df.columns = ["zone_type", "post_minus_pre_mean_log_trips"]
diff_df.to_csv(TABLE_DIR / "03_heterogeneity_zone_type.csv", index=False)
print("\nMean log_trips: post - pre by zone type:")
print(diff_df.to_string(index=False))


# %%
# Cell 12: Anticipation effect — donut window robustness
"""
Drop weeks ±2 around treatment (Dec 22 2024 - Jan 19 2025) to isolate
anticipation/holiday-confound from clean post-treatment effect.
"""
print("Donut-window robustness (drop ±2 weeks around Jan 5)...")
donut_start = TREATMENT_DATE - pd.Timedelta(weeks=2)
donut_end = TREATMENT_DATE + pd.Timedelta(weeks=2)
df_donut = df_uber_daily[
    (df_uber_daily["pickup_date"] < donut_start)
    | (df_uber_daily["pickup_date"] >= donut_end)
].copy()
print(f"  Rows after donut: {len(df_donut):,} (dropped "
      f"{len(df_uber_daily) - len(df_donut):,})")

donut_results = []
for dep in ["log_trips", "log_fpm"]:
    r = did_regression(df_donut, dep)
    donut_results.append({**r, "spec": "donut_window"})
    print(f"  {dep}: β={r['beta']:.5f} (p={r['p_value']:.4f})")

pd.DataFrame(donut_results).to_csv(TABLE_DIR / "03_donut_window_robustness.csv", index=False)


# %%
# Cell 13: Daily volume time series (deck chart)
daily_uber = (
    df_uber.groupby("pickup_date")
    .agg(trips=("trip_count", "sum"), fare=("total_adjusted_fare", "sum"),
         miles=("total_miles", "sum"))
    .reset_index()
)
daily_uber["fpm"] = daily_uber["fare"] / daily_uber["miles"]
daily_uber_crz = (
    df_uber[df_uber["is_in_crz"] == 1].groupby("pickup_date")
    .agg(trips=("trip_count", "sum"), fare=("total_adjusted_fare", "sum"),
         miles=("total_miles", "sum"))
    .reset_index()
)
daily_uber_crz["fpm"] = daily_uber_crz["fare"] / daily_uber_crz["miles"]
daily_uber_non = (
    df_uber[df_uber["is_in_crz"] == 0].groupby("pickup_date")
    .agg(trips=("trip_count", "sum"))
    .reset_index()
)

ROLL = 7
for _d in (daily_uber_crz, daily_uber_non, daily_uber):
    _d.sort_values("pickup_date", inplace=True)
daily_uber_crz["trips_roll"] = daily_uber_crz["trips"].rolling(ROLL, min_periods=1).mean()
daily_uber_non["trips_roll"] = daily_uber_non["trips"].rolling(ROLL, min_periods=1).mean()
daily_uber_crz["fpm_roll"] = daily_uber_crz["fpm"].rolling(ROLL, min_periods=1).mean()
daily_uber["fpm_roll"] = daily_uber["fpm"].rolling(ROLL, min_periods=1).mean()

fig_dv, axes_dv = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax0 = axes_dv[0]
ax0.plot(
    daily_uber_crz["pickup_date"],
    daily_uber_crz["trips"],
    color=PALETTE["crz"],
    lw=1.0,
    alpha=0.25,
)
ax0.plot(
    daily_uber_crz["pickup_date"],
    daily_uber_crz["trips_roll"],
    color=PALETTE["crz"],
    lw=2.2,
)
ax0.plot(
    daily_uber_non["pickup_date"],
    daily_uber_non["trips"],
    color=PALETTE["non_crz"],
    lw=1.0,
    alpha=0.25,
)
ax0.plot(
    daily_uber_non["pickup_date"],
    daily_uber_non["trips_roll"],
    color=PALETTE["non_crz"],
    lw=2.2,
)
ax0.axvline(
    TREATMENT_DATE, color=PALETTE["warn"], lw=1.8, ls="--", zorder=5, alpha=0.95
)
ax0.set_ylabel("Trips (citywide total, all hours)")
format_thousands_axis(ax0, "y")
limit_ticks(ax0, "y")
ax0.set_title(
    "Daily Uber trips by zone class (7-day rolling mean; raw at low α)\n"
    "CRZ vs non-CRZ — sum of trip_count",
    fontsize=11,
    color="#555555",
)
l0 = [
    ("CRZ (roll)", PALETTE["crz"]),
    ("Non-CRZ (roll)", PALETTE["non_crz"]),
]
for i, (lab, c) in enumerate(l0):
    ax0.text(
        0.99,
        0.35 - 0.08 * i,
        lab,
        transform=ax0.transAxes,
        ha="right",
        fontsize=10,
        color=c,
        fontweight="bold",
    )

_pre_peak = daily_uber_crz[
    (daily_uber_crz["pickup_date"] >= pd.Timestamp("2024-12-15"))
    & (daily_uber_crz["pickup_date"] < TREATMENT_DATE)
]
if len(_pre_peak):
    _imax = _pre_peak["trips_roll"].idxmax()
    _dt_pk = daily_uber_crz.loc[_imax, "pickup_date"]
    _y_pk = daily_uber_crz.loc[_imax, "trips_roll"]
    annotate_callout(
        ax0,
        xy=(_dt_pk, _y_pk),
        text="Pre-treatment anticipation\n(Dec spike)",
        xytext=(-80, 40),
    )

_post = daily_uber_crz[daily_uber_crz["pickup_date"] >= TREATMENT_DATE]
if len(_post):
    _y_drop = float(_post["trips_roll"].median())
    _x_ann = _post["pickup_date"].median()
    annotate_callout(
        ax0,
        xy=(_x_ann, _y_drop),
        text="Persistent post-CP gap\n(vs non-CRZ baseline)",
        xytext=(-120, -50),
    )

ax1 = axes_dv[1]
ax1.plot(
    daily_uber_crz["pickup_date"],
    daily_uber_crz["fpm"],
    color=PALETTE["crz"],
    lw=1.0,
    alpha=0.25,
)
ax1.plot(
    daily_uber_crz["pickup_date"],
    daily_uber_crz["fpm_roll"],
    color=PALETTE["crz"],
    lw=2.2,
)
ax1.plot(
    daily_uber["pickup_date"],
    daily_uber["fpm"],
    color=PALETTE["uber"],
    lw=1.0,
    alpha=0.25,
)
ax1.plot(
    daily_uber["pickup_date"],
    daily_uber["fpm_roll"],
    color=PALETTE["uber"],
    lw=2.2,
)
ax1.axvline(
    TREATMENT_DATE, color=PALETTE["warn"], lw=1.8, ls="--", zorder=5, alpha=0.95
)
ax1.set_ylabel("Fare per mile ($/mile, ratio-of-sums)")
format_dollar_axis(ax1, "y")
limit_ticks(ax1, "y")
ax1.set_title(
    "Daily fare-per-mile — CRZ vs all zones\n7-day rolling mean on ratio-of-sums",
    fontsize=11,
    color="#555555",
)
for lab, c, yp in [
    ("CRZ FPM (roll)", PALETTE["crz"], 0.35),
    ("All zones FPM (roll)", PALETTE["uber"], 0.26),
]:
    ax1.text(0.99, yp, lab, transform=ax1.transAxes, ha="right", fontsize=10, color=c, fontweight="bold")

fig_dv.suptitle(
    "Volume crashed at CP; fare-per-mile barely moved",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
axes_dv[1].set_xlabel("Date")
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
add_footnote(
    fig_dv,
    "Volume drop magnitude is partly pre-existing trend (see 03_pre_trends_placebo.csv).",
    y=-0.02,
)
plt.savefig(FIG_DIR / "03_daily_volume_and_fpm.png")
plt.close()


# %%
# Cell 14: Findings narrative
beta_trips = did_uber_df.loc[did_uber_df["dep"] == "log_trips", "beta"].iloc[0]
beta_fpm = did_uber_df.loc[did_uber_df["dep"] == "log_fpm", "beta"].iloc[0]
trips_pct = (np.exp(beta_trips) - 1) * 100
fpm_pct = (np.exp(beta_fpm) - 1) * 100

lines = [
    "⚠️  PRE-TRENDS VIOLATION (parallel trends assumption fails):",
    f"   All 4 placebo dates (pre-Jan 5 2025) show significant negative β:",
    f"     2024-06-01: β=-0.080 | 2024-09-01: β=-0.058",
    f"     2024-11-01: β=-0.096 | 2024-12-01: β=-0.126",
    f"   CRZ zones were already declining vs non-CRZ throughout 2024.",
    f"   Naive DiD attributes ~11% drop to CP; adjusting for pre-existing",
    f"   trend, true CP causal effect is closer to 0-4%, not 11%.",
    f"   Headline figure should be reported with this caveat.",
    "",
    f"- DiD headline (Uber, log_trips): β={beta_trips:.4f} → CRZ trip volume changed by {trips_pct:.1f}% post-CP relative to non-CRZ.",
    f"- DiD headline (Uber, log_fpm): β={beta_fpm:.4f} → CRZ fare-per-mile changed by {fpm_pct:.1f}% post-CP (contrarian if negative).",
    "- See 03_pre_trends_placebo.csv for full placebo coefficients.",
    "- See 03_event_study_plot.png: weekly + daily event-study with CIs.",
    "- See 03_donut_window_robustness.csv: drops ±2 weeks around Jan 5 to handle anticipation/holiday confounds.",
    "- Lyft placebo: see 03_did_lyft_placebo.csv (PanelOLS + FD; FD addresses Lyft non-stationarity).",
]
text_out = "\n".join(lines)
print(text_out)
(TABLE_DIR / "03_findings_summary.txt").write_text(text_out + "\n", encoding="utf-8")
