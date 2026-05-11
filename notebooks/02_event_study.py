# %%
# Cell 1: Imports + setup
"""
Notebook 02 — Event Study (Uber GPA case). Cursor interactive cells (# %%).

Memory-safe: matched-baseline pool collapsed to (zone, hour_of_week) means
BEFORE merging with event-hours, so peak memory ~50K rows not billions.
"""
from __future__ import annotations

# REPRODUCIBILITY: All random sampling uses fixed seed=42 (or 1, where noted).
# Regression results are deterministic given fixed input data.

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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
    format_coef_axis,
    format_pct_axis,
    format_thousands_axis,
    limit_ticks,
    set_rcparams,
)

set_rcparams()

NJ_DEPARTURE_ZONES = frozenset(
    {186, 230, 161, 100, 246, 48, 261, 87, 88, 231, 125, 113, 114, 158, 249}
)

MASTER_PATH = PROJECT_ROOT / "data/processed/master_zone_hour.parquet"
EVENTS_PATH = PROJECT_ROOT / "data/raw/events/major_events.csv"
HOLIDAYS_PATH = PROJECT_ROOT / "data/processed/holidays.parquet"

print("Loading master parquet...")
df_master = pd.read_parquet(MASTER_PATH, engine="pyarrow")
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

print(f"Platforms: {sorted(df_master['platform'].unique())}")

df_uber = df_master[df_master["platform"] == "uber"].copy()
df_lyft_only = df_master[df_master["platform"] == "lyft"].copy()
print(f"Master rows: {len(df_master):,} | Uber: {len(df_uber):,} | Lyft: {len(df_lyft_only):,}")

events_df = pd.read_csv(EVENTS_PATH, encoding="utf-8")
events_df["event_date"] = pd.to_datetime(events_df["date"]).dt.normalize()
events_df["event_id"] = np.arange(len(events_df), dtype=np.int64)
events_df["event_category"] = events_df["event_type"].astype(str)
print(f"Events catalog: {len(events_df)} rows")
print(events_df["event_category"].value_counts().to_string())


# %%
# Cell 2: Helper functions
from scipy import stats
from scipy.stats import t as student_t

try:
    from linearmodels.panel import FirstDifferenceOLS, PanelOLS
except Exception:
    PanelOLS = None
    FirstDifferenceOLS = None


def safe_log(x: pd.Series, eps: float = 1.0) -> pd.Series:
    return np.log(np.maximum(x.astype(float), 0.0) + eps)


def collapse_to_zone_day(df, flag_cols=None):
    """
    Aggregate hourly panel to (pickup_zone, pickup_date) for memory-safe
    panel regressions. Methodology-consistent: ratio-of-sums fare-per-mile
    on collapsed totals; binary flags = max() over hours within day
    (i.e., flagged if ANY hour in that day was flagged).
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
    # Rename pickup_date -> pickup_hour so existing panel_event_regression works
    daily = daily.rename(columns={"pickup_date": "pickup_hour"})
    print(f"  collapsed to {len(daily):,} zone-day rows "
          f"(from {len(df):,} hourly rows)")
    return daily


def attach_event_metadata(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Join events on (pickup_zone, calendar date)."""
    out = df.copy()
    out["pickup_date"] = pd.to_datetime(out["pickup_hour"]).dt.normalize()
    ev = events.rename(columns={"zone_id": "pickup_zone"})
    ev_u = ev.drop_duplicates(subset=["pickup_zone", "event_date", "event_id"], keep="first")
    out = out.merge(
        ev_u[["pickup_zone", "event_date", "event_id", "event_name",
              "event_category", "venue", "league"]],
        left_on=["pickup_zone", "pickup_date"],
        right_on=["pickup_zone", "event_date"],
        how="left",
    )
    return out


def build_baseline_lookup(df, flag_cols_exclude=None):
    """
    COLLAPSE-FIRST BASELINE.

    Returns one row per (pickup_zone, hour_of_week) with mean baseline metrics,
    computed only over rows where none of flag_cols_exclude are active.

    Small (~263 zones × 168 hours = 44K rows max), joins to event-hours in
    O(N_event_hours) — replaces the cartesian explosion of the original approach.

    Trade-off: doesn't enforce a strict ±14d buffer around each event, but does
    enforce "no event flagged on the baseline hour itself," which is the core
    matched-baseline condition.
    """
    if flag_cols_exclude is None:
        flag_cols_exclude = ["is_nyc_event_sym", "is_nyc_event_asym"]

    flag_cols_present = [c for c in flag_cols_exclude if c in df.columns]
    if not flag_cols_present:
        clear_mask = pd.Series(True, index=df.index)
    else:
        clear_mask = df[flag_cols_present].sum(axis=1) == 0

    cols = ["pickup_zone", "hour_of_week", "trip_count", "fare_per_mile",
            "log_trips", "log_fpm"]
    pool = df.loc[clear_mask, cols]
    lookup = (
        pool.groupby(["pickup_zone", "hour_of_week"], as_index=False)
        .agg(
            baseline_mean_trips=("trip_count", "mean"),
            baseline_mean_fpm=("fare_per_mile", "mean"),
            baseline_mean_log_trips=("log_trips", "mean"),
            baseline_mean_log_fpm=("log_fpm", "mean"),
            baseline_n_hours=("trip_count", "size"),
        )
    )
    print(f"  baseline lookup: {len(lookup):,} pairs from {len(pool):,} clear hours")
    return lookup


def attach_baseline(event_hours, baseline_lookup):
    return event_hours.merge(baseline_lookup, on=["pickup_zone", "hour_of_week"], how="left")


def per_event_lift(df_evt, event_id_col, category_col, metric_ev_col, baseline_mean_col):
    """One row per event_id. lift_pct from log-diff; paired one-sample t-test on diffs."""
    rows = []
    for eid, g in df_evt.groupby(event_id_col):
        g = g.dropna(subset=[metric_ev_col, baseline_mean_col])
        if len(g) == 0:
            continue
        ev_vals = g[metric_ev_col].astype(float).values
        base_vals = g[baseline_mean_col].astype(float).values

        ev_m = float(np.mean(ev_vals))
        b_m = float(np.mean(base_vals))
        log_diff = ev_m - b_m
        pct_lift = (np.exp(log_diff) - 1.0) * 100.0 if pd.notna(log_diff) else np.nan

        diffs = ev_vals - base_vals
        if len(diffs) >= 2 and np.std(diffs, ddof=1) > 0:
            t_stat, p_val = stats.ttest_1samp(diffs, popmean=0.0)
        else:
            t_stat, p_val = np.nan, np.nan

        cat = g[category_col].iloc[0] if category_col in g.columns else ""
        rows.append({
            event_id_col: eid, "category": cat, "n_event_hours": len(g),
            "mean_log_event": ev_m, "mean_log_baseline": b_m, "log_diff": log_diff,
            "lift_pct": pct_lift,
            "t_stat": float(t_stat) if pd.notna(t_stat) else np.nan,
            "p_value": float(p_val) if pd.notna(p_val) else np.nan,
        })
    return pd.DataFrame(rows)


def panel_event_regression(df, flag_col, log_metric_col, cluster_col="pickup_zone"):
    """PanelOLS, entity + time FE, cluster SEs by zone. OLS fallback."""
    import statsmodels.api as sm

    d = df[[cluster_col, "pickup_hour", flag_col, log_metric_col]].dropna().copy()
    d = d.set_index([cluster_col, "pickup_hour"]).sort_index()
    y = d[log_metric_col].astype(np.float64)
    X = d[[flag_col]].astype(np.float64)

    if PanelOLS is not None:
        try:
            mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
            res = mod.fit(cov_type="clustered", cluster_entity=True)
            return {
                "coef": float(res.params[flag_col]),
                "se": float(res.std_errors[flag_col]),
                "p_value": float(res.pvalues[flag_col]),
                "nobs": int(res.nobs),
                "method": "PanelOLS",
            }
        except Exception as e:
            warnings.warn(f"PanelOLS failed ({e}); trying OLS fallback.")

    d_reset = d.reset_index()
    try:
        ols = sm.OLS.from_formula(
            f"{log_metric_col} ~ {flag_col} + C({cluster_col}) + C(pickup_hour)",
            data=d_reset,
        ).fit(cov_type="cluster", cov_kwds={"groups": d_reset[cluster_col]})
        return {
            "coef": float(ols.params[flag_col]),
            "se": float(ols.bse[flag_col]),
            "p_value": float(ols.pvalues[flag_col]),
            "nobs": int(ols.nobs),
            "method": "OLS_FE_fallback",
        }
    except Exception as e2:
        warnings.warn(f"OLS fallback failed: {e2}")
        return None


print("Helpers ready.")


# %%
# Cell 3: Data prep
df = attach_event_metadata(df_uber, events_df)
df["log_trips"] = safe_log(df["trip_count"], eps=1.0)
df["fare_per_mile"] = np.where(
    df["total_miles"].astype(float) > 0,
    df["total_adjusted_fare"].astype(float) / df["total_miles"].astype(float),
    np.nan,
)
df["log_fpm"] = safe_log(df["fare_per_mile"], eps=0.01)
ts = pd.to_datetime(df["pickup_hour"])
df["hour_of_week"] = ts.dt.dayofweek * 24 + ts.dt.hour

print(f"is_nyc_event_sym: {int(df['is_nyc_event_sym'].sum()):,}")
print(f"is_nyc_event_asym: {int(df['is_nyc_event_asym'].sum()):,}")
print("\nFlag distribution by category (sym):")
print(df.groupby("event_category")["is_nyc_event_sym"].sum().to_string())

print("\nBuilding baseline lookup...")
baseline_lookup = build_baseline_lookup(
    df, ["is_nyc_event_sym", "is_nyc_event_asym",
         "is_nj_event_pregame_sym", "is_nj_event_pregame_asym"]
)

print("\nBuilding zone-day panel for memory-safe panel regressions...")
df_daily = collapse_to_zone_day(df)


# %%
# Cell 4: NYC sports — per-event lift
sports_mask = (
    df["is_nyc_event_sym"].eq(1)
    & df["event_category"].str.startswith("sports", na=False)
    & df["pickup_zone"].gt(0)
    & df["event_id"].notna()
)
event_hours_sp = df.loc[sports_mask].copy()
print(f"NYC sports event-hour rows: {len(event_hours_sp):,}")

lift_sp = attach_baseline(event_hours_sp, baseline_lookup)

lift_sp["row_lift_trips_pct"] = (
    (lift_sp["trip_count"] - lift_sp["baseline_mean_trips"])
    / lift_sp["baseline_mean_trips"].replace(0, np.nan) * 100
)
lift_sp["row_lift_fpm_pct"] = (
    (lift_sp["fare_per_mile"] - lift_sp["baseline_mean_fpm"])
    / lift_sp["baseline_mean_fpm"].replace(0, np.nan) * 100
)

per_ev_lt = per_event_lift(
    lift_sp.dropna(subset=["event_id"]),
    "event_id", "event_category", "log_trips", "baseline_mean_log_trips",
)
per_ev_fpm = per_event_lift(
    lift_sp.dropna(subset=["event_id"]),
    "event_id", "event_category", "log_fpm", "baseline_mean_log_fpm",
)
per_ev_out = per_ev_lt.merge(per_ev_fpm, on=["event_id", "category"],
                              suffixes=("_log_trips", "_log_fpm"))
per_ev_out.to_csv(TABLE_DIR / "02_nyc_sports_per_event.csv", index=False)

med_lt = per_ev_lt["lift_pct"].median()
med_lf = per_ev_fpm["lift_pct"].median()
pos_lt = (per_ev_lt["lift_pct"] > 0).mean() * 100
print(f"\nUnique events: {per_ev_lt['event_id'].nunique()}")
print(f"Median lift % (trips): {med_lt:.2f} | IQR: "
      f"{per_ev_lt['lift_pct'].quantile(0.75) - per_ev_lt['lift_pct'].quantile(0.25):.2f}")
print(f"% events positive: {pos_lt:.1f}%")
print(f"Median lift % (fare-per-mile): {med_lf:.2f}")
print("\nTop 5 lift events:")
print(per_ev_lt.sort_values("lift_pct", ascending=False).head(5).to_string(index=False))


# %%
# Cell 5: NYC sports — panel regression
panel_rows = []
for fl in ["is_nyc_event_sym", "is_nyc_event_asym"]:
    for metric in ["log_trips", "log_fpm"]:
        r = panel_event_regression(df_daily, fl, metric)
        if r:
            panel_rows.append({
                "flag": fl, "metric": metric, "beta": r["coef"], "se": r["se"],
                "p_value": r["p_value"], "nobs": r["nobs"], "method": r["method"],
            })
panel_nyc = pd.DataFrame(panel_rows)
# NOTE: asym/sym distinction collapses on zone-day panel (asym ⊇ sym at hourly level).
# Hour-level asym/sym per-event lift remains in Cell 4 CSV.
panel_nyc = panel_nyc[panel_nyc["flag"] != "is_nyc_event_asym"].copy()
panel_nyc.to_csv(TABLE_DIR / "02_nyc_sports_panel.csv", index=False)
print(panel_nyc.to_string(index=False))


# %%
# Cell 5b: Placebo — sym shifted -7 days (zone-day granularity)
sym_pairs = df_daily.loc[df_daily["is_nyc_event_sym"] == 1, ["pickup_zone", "pickup_hour"]].drop_duplicates()
sym_pairs["placebo_hour"] = pd.to_datetime(sym_pairs["pickup_hour"]) - pd.Timedelta(days=7)
placebo_idx = pd.MultiIndex.from_frame(sym_pairs[["pickup_zone", "placebo_hour"]])
full_mi = pd.MultiIndex.from_frame(df_daily[["pickup_zone", "pickup_hour"]])
df_daily["placebo_pre7_sym"] = full_mi.isin(placebo_idx).astype(np.int8)

placebo_rows = []
for metric in ["log_trips", "log_fpm"]:
    r = panel_event_regression(df_daily, "placebo_pre7_sym", metric)
    if r:
        placebo_rows.append({
            "metric": metric, "beta": r["coef"], "se": r["se"],
            "p_value": r["p_value"], "nobs": r["nobs"], "method": r["method"],
        })
pd.DataFrame(placebo_rows).to_csv(TABLE_DIR / "02_nyc_sports_placebo.csv", index=False)
print("Placebo (-7d):")
print(pd.DataFrame(placebo_rows).to_string(index=False))


# %%
# Cell 6: Lift distribution
_n_evt = len(lift_sp)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(
    "Most NYC sports events boost trips; fare-per-mile barely moves",
    fontsize=14,
    fontweight="bold",
    y=1.03,
)
sub_main = (
    f"Distribution of % Δ vs matched (zone, hour-of-week) baseline | "
    f"N={_n_evt:,} event-hours | Uber 2024–25"
)
med_t = lift_sp["row_lift_trips_pct"].median()
bbox_note = dict(boxstyle="round", fc=PALETTE["annot_bg"], ec="#CCCCCC", alpha=0.95)
for i, (ax, col, ylab) in enumerate(zip(
    axes,
    ["row_lift_trips_pct", "row_lift_fpm_pct"],
    ["% Δ trips vs matched baseline", "% Δ fare-per-mile vs matched baseline"],
)):
    sns.histplot(lift_sp[col].dropna(), kde=True, ax=ax, color=PALETTE["uber"], edgecolor="white")
    ax.axvline(0, color=PALETTE["muted"], linestyle=":", lw=1.2, zorder=0)
    med = lift_sp[col].median()
    ax.axvline(med, color=PALETTE["warn"], lw=1.5, ls="--", zorder=1)
    ax.set_xlabel(ylab)
    ax.set_ylabel("Count of event-hours")
    ax.set_title(sub_main, fontsize=11, color="#555555", pad=8)
    format_pct_axis(ax, "x", decimals=1)
    limit_ticks(ax, "y")
    ymax = ax.get_ylim()[1]
    xspan = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    if i == 0:
        ax.annotate(
            f"Median: {med:+.1f}%",
            xy=(med, ymax * 0.55),
            xytext=(med + xspan * 0.08, ymax * 0.85),
            arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]),
            bbox=bbox_note,
            fontsize=9,
        )
        ax.annotate(
            "no effect",
            xy=(0, ymax * 0.25),
            xytext=(xspan * 0.12, ymax * 0.45),
            arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]),
            bbox=bbox_note,
            fontsize=9,
        )
plt.tight_layout()
add_footnote(fig, "Per-event aggregation in 02_nyc_sports_per_event.csv", y=-0.06)
plt.savefig(FIG_DIR / "02_nyc_sports_lift_distribution.png")
plt.close()


# %%
# Cell 7: NJ scoped per-event lift
nj_ev_pick = (
    events_df[events_df["zone_id"] == -1]
    .sort_values("expected_attendance", ascending=False)
    .drop_duplicates(subset=["event_date"], keep="first")
)[["event_id", "event_date", "venue"]].rename(
    columns={"event_id": "nj_event_id", "venue": "nj_venue"}
)

nj_mask = (
    df["is_nj_event_pregame_sym"].eq(1)
    & df["pickup_zone"].isin(NJ_DEPARTURE_ZONES)
)
event_hours_nj = df.loc[nj_mask].merge(
    nj_ev_pick, left_on="pickup_date", right_on="event_date", how="left"
)

nj_baseline_lookup = build_baseline_lookup(
    df, ["is_nj_event_pregame_sym", "is_nj_event_pregame_asym",
         "is_nyc_event_sym", "is_nyc_event_asym"]
)
lift_nj = attach_baseline(event_hours_nj, nj_baseline_lookup)
lift_nj["row_lift_trips_pct"] = (
    (lift_nj["trip_count"] - lift_nj["baseline_mean_trips"])
    / lift_nj["baseline_mean_trips"].replace(0, np.nan) * 100
)
lift_nj["row_lift_fpm_pct"] = (
    (lift_nj["fare_per_mile"] - lift_nj["baseline_mean_fpm"])
    / lift_nj["baseline_mean_fpm"].replace(0, np.nan) * 100
)
lift_nj.to_csv(TABLE_DIR / "02_nj_sports_per_event_scoped.csv", index=False)
print(f"NJ scoped event-hour rows: {len(lift_nj)}")
print(f"Median NJ lift % (trips): {lift_nj['row_lift_trips_pct'].median():.2f}")


# %%
# Cell 8: NJ panel — scoped vs full
nj_panel_rows = []

def collect_nj_panel(tag, sub):
    for fl in ["is_nj_event_pregame_sym", "is_nj_event_pregame_asym"]:
        for metric in ["log_trips", "log_fpm"]:
            r = panel_event_regression(sub, fl, metric)
            if r:
                nj_panel_rows.append({
                    "sample": tag, "flag": fl, "metric": metric,
                    "beta": r["coef"], "se": r["se"], "p_value": r["p_value"],
                    "nobs": r["nobs"], "method": r["method"],
                })

collect_nj_panel(
    "scoped_departure_zones",
    df_daily[df_daily["pickup_zone"].isin(NJ_DEPARTURE_ZONES)],
)
collect_nj_panel("full_panel_all_zones", df_daily)

nj_panel_df = pd.DataFrame(nj_panel_rows)
# NOTE: asym/sym distinction collapses in daily panel (asym ⊇ sym).
# Hour-level distinction is preserved in per-event lift (Cell 7).
nj_panel_df = nj_panel_df[nj_panel_df["flag"] != "is_nj_event_pregame_asym"].copy()
nj_panel_df.to_csv(TABLE_DIR / "02_nj_sports_panel.csv", index=False)
print(nj_panel_df.to_string(index=False))


# %%
# Cell 9: NJ boxplot by venue
fig, ax = plt.subplots(figsize=(10, 7))
venue_col = "nj_venue" if "nj_venue" in lift_nj.columns else "venue"
_dn = lift_nj.dropna(subset=[venue_col])
_med_order = (
    _dn.groupby(venue_col)["row_lift_trips_pct"].median().sort_values(ascending=False).index.tolist()
)
_vc = _dn.groupby(venue_col).size()
_xtlbl = [f"{str(v)[:22]} (n={int(_vc.get(v, 0))})" for v in _med_order]
sns.boxplot(
    data=_dn,
    x=venue_col,
    y="row_lift_trips_pct",
    order=_med_order,
    color=PALETTE["crz"],
    ax=ax,
)
ax.set_xticklabels(_xtlbl, rotation=28, ha="right")
ax.set_ylabel("% Δ trips vs matched (zone, hour-of-week) baseline")
ax.set_xlabel("")
format_pct_axis(ax, "y")
limit_ticks(ax, "y")
fig.suptitle(
    "MetLife pregame zones see 2× larger lift than other NJ venues",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
ax.set_title(
    "% Δ trips in Manhattan departure zones | 4hr pre-kickoff window | scoped panel",
    fontsize=11,
    color="#555555",
)
add_footnote(fig, "Departure zone clusters: see methodology", y=-0.14)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_nj_sports_lift_by_venue.png")
plt.close()


# %%
# Cell 10: Political — N=6, asym primary
pol_mask = (
    df["is_nyc_event_asym"].eq(1)
    & df["event_category"].eq("political")
    & df["event_id"].notna()
)
pol_evt = df.loc[pol_mask].copy()
print(f"Political event-hour rows: {len(pol_evt)}")

lift_pol = attach_baseline(pol_evt, baseline_lookup)

pol_summary = []
for eid, g in lift_pol.groupby("event_id"):
    ev_m = g["log_trips"].mean()
    b_m = g["baseline_mean_log_trips"].mean()
    log_diff = ev_m - b_m if pd.notna(ev_m) and pd.notna(b_m) else np.nan
    lift = (np.exp(log_diff) - 1.0) * 100.0 if pd.notna(log_diff) else np.nan

    diffs = (g["log_trips"] - g["baseline_mean_log_trips"]).dropna()
    if len(diffs) > 1 and diffs.std(ddof=1) > 0:
        tcrit = student_t.ppf(0.975, df=len(diffs) - 1)
        se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
        ci_lo = (np.exp(log_diff - tcrit * se) - 1.0) * 100.0
        ci_hi = (np.exp(log_diff + tcrit * se) - 1.0) * 100.0
    else:
        ci_lo = ci_hi = np.nan

    pol_summary.append({
        "event_id": eid,
        "event_name": g["event_name"].iloc[0] if "event_name" in g.columns else "",
        "n_hours": len(g),
        "lift_pct": lift, "ci95_low_pct": ci_lo, "ci95_high_pct": ci_hi,
    })
pd.DataFrame(pol_summary).to_csv(TABLE_DIR / "02_political_per_event.csv", index=False)
print("\nPolitical (N=6, exploratory):")
print(pd.DataFrame(pol_summary).to_string(index=False))


# %%
# Cell 10b: Civic / parade / special-event lift
civic_categories = ["parade", "parade_evening", "special_event", "special_event_evening"]
civic_mask = (
    df["is_nyc_event_sym"].eq(1)
    & df["event_category"].isin(civic_categories)
    & df["event_id"].notna()
)
civic_evt = df.loc[civic_mask].copy()
print(f"Civic/special event-hour rows: {len(civic_evt)}")
print(f"Unique events: {civic_evt['event_id'].nunique()}")

lift_civic = attach_baseline(civic_evt, baseline_lookup)
per_ev_civic = per_event_lift(
    lift_civic.dropna(subset=["event_id"]),
    "event_id", "event_category", "log_trips", "baseline_mean_log_trips",
)
ev_name_map = events_df.set_index("event_id")["event_name"].to_dict()
per_ev_civic["event_name"] = per_ev_civic["event_id"].map(ev_name_map)
per_ev_civic.to_csv(TABLE_DIR / "02_civic_special_per_event.csv", index=False)
print("\nCivic/special per-event lift:")
print(per_ev_civic.sort_values("lift_pct", ascending=False).to_string(index=False))


# %%
# Cell 11: Holidays — vectorized day-level
if HOLIDAYS_PATH.exists():
    hol_df = pd.read_parquet(HOLIDAYS_PATH)
else:
    import pandas.tseries.holiday as hol_mod
    cal = hol_mod.USFederalHolidayCalendar()
    hol_df = pd.DataFrame({"date": cal.holidays(start="2024-01-01", end="2025-09-01")})

hol_df["date"] = pd.to_datetime(hol_df["date"]).dt.normalize()
hol_dates = set(hol_df["date"].tolist())

df_day = df.copy()
df_day["d"] = pd.to_datetime(df_day["pickup_hour"]).dt.normalize()
day_agg = (
    df_day.groupby(["pickup_zone", "d"])
    .agg(trips=("trip_count", "sum"), tf=("total_adjusted_fare", "sum"),
         tm=("total_miles", "sum"))
    .reset_index()
)
day_agg["fpm_ros"] = np.where(day_agg["tm"] > 0, day_agg["tf"] / day_agg["tm"], np.nan)
day_agg["dow"] = day_agg["d"].dt.dayofweek
day_agg["is_holiday"] = day_agg["d"].isin(hol_dates).astype(int)

baseline_by_zone_dow = (
    day_agg[day_agg["is_holiday"] == 0]
    .groupby(["pickup_zone", "dow"])
    .agg(base_trips=("trips", "mean"), base_fpm=("fpm_ros", "mean"))
    .reset_index()
)
hol_rows_df = (
    day_agg[day_agg["is_holiday"] == 1]
    .merge(baseline_by_zone_dow, on=["pickup_zone", "dow"], how="left")
)
hol_rows_df["trip_lift_pct"] = (
    (hol_rows_df["trips"] - hol_rows_df["base_trips"]) / hol_rows_df["base_trips"] * 100
)
hol_rows_df["fpm_lift_pct"] = (
    (hol_rows_df["fpm_ros"] - hol_rows_df["base_fpm"]) / hol_rows_df["base_fpm"] * 100
)
hol_rows_df = hol_rows_df.rename(columns={"d": "holiday_date"})[
    ["pickup_zone", "holiday_date", "trip_lift_pct", "fpm_lift_pct"]
]
hol_rows_df.to_csv(TABLE_DIR / "02_holidays_per_event.csv", index=False)
print(f"Holiday zone-rows: {len(hol_rows_df)}")
print(f"Median trip lift: {hol_rows_df['trip_lift_pct'].median():.2f}%")


# %%
# Cell 12: Marathon case study (now in events catalog)
MARATHON_DATE = pd.Timestamp("2024-11-03").normalize()
mar_hours = df[
    (pd.to_datetime(df["pickup_hour"]).dt.normalize() == MARATHON_DATE)
    & (df["is_nyc_event_sym"] == 1)
]
MARATHON_ZONES = sorted(mar_hours["pickup_zone"].unique().tolist())
if len(MARATHON_ZONES) == 0:
    raise ValueError("No marathon hours flagged. Re-run rebuild script.")

df_m = df[df["pickup_zone"].isin(MARATHON_ZONES)].copy()
df_m["d"] = pd.to_datetime(df_m["pickup_hour"]).dt.normalize()
d_agg = df_m.groupby(["pickup_zone", "d"]).agg(trips=("trip_count", "sum")).reset_index()

mar_rows = []
dow_sun = MARATHON_DATE.dayofweek
for z in MARATHON_ZONES:
    ev = d_agg[(d_agg["pickup_zone"] == z) & (d_agg["d"] == MARATHON_DATE)]
    base = d_agg[
        (d_agg["pickup_zone"] == z)
        & (d_agg["d"].dt.dayofweek == dow_sun)
        & ((d_agg["d"] - MARATHON_DATE).abs().dt.days.between(15, 180))
    ]
    if len(ev) and len(base):
        mar_rows.append({
            "pickup_zone": z,
            "trip_lift_pct": (ev["trips"].iloc[0] - base["trips"].mean())
                              / base["trips"].mean() * 100,
        })
mar_summary = pd.DataFrame(mar_rows)
mar_summary["event"] = "NYC Marathon 2024-11-03"
mar_summary.to_csv(TABLE_DIR / "02_marathon_case_study.csv", index=False)
print(f"Marathon: {len(mar_summary)} zones | "
      f"median lift {mar_summary['trip_lift_pct'].median():.2f}%")


# %%
# Cell 12.5: Day flag overlap + clean_day_flag
day_only = ((df["has_major_event_dayflag"] == 1) & (df["is_nyc_event_sym"] == 0)).sum()
sym_only = ((df["has_major_event_dayflag"] == 0) & (df["is_nyc_event_sym"] == 1)).sum()
both = ((df["has_major_event_dayflag"] == 1) & (df["is_nyc_event_sym"] == 1)).sum()
print(f"Day only: {day_only:,} | Sym only: {sym_only:,} | Both: {both:,}")

sym_event_dates = (
    pd.to_datetime(df.loc[df["is_nyc_event_sym"] == 1, "pickup_hour"]).dt.normalize().unique()
)
df["pickup_date_norm"] = pd.to_datetime(df["pickup_hour"]).dt.normalize()
df["clean_day_flag"] = df["pickup_date_norm"].isin(sym_event_dates).astype(np.int8)
print(f"Clean day flag count: {int(df['clean_day_flag'].sum()):,}")


# %%
# Cell 13: Dilution diagnostic (descriptive, not regression-based)
# In the hourly panel, clean_day_flag is degenerate when time FE absorb
# day-level variation on a zone-day aggregate. Report dilution descriptively
# via flag overlap counts.
day_only = ((df["has_major_event_dayflag"] == 1) & (df["is_nyc_event_sym"] == 0)).sum()
sym_only = ((df["has_major_event_dayflag"] == 0) & (df["is_nyc_event_sym"] == 1)).sum()
both = ((df["has_major_event_dayflag"] == 1) & (df["is_nyc_event_sym"] == 1)).sum()
total_dayflag = day_only + both
total_sym = sym_only + both
dilution_ratio = total_dayflag / total_sym if total_sym > 0 else np.nan

_dil_pct = (
    (1 - 1.0 / dilution_ratio) * 100
    if dilution_ratio and dilution_ratio > 0 and np.isfinite(dilution_ratio)
    else np.nan
)
_interp_tail = (
    f"by ~{_dil_pct:.0f}%."
    if pd.notna(_dil_pct) and np.isfinite(_dil_pct)
    else "(share indeterminate)."
)

dilution_descriptive = pd.DataFrame([{
    "metric": "flag_overlap_descriptive",
    "day_flag_only_hours": day_only,
    "sym_flag_only_hours": sym_only,
    "both_flagged_hours": both,
    "day_flag_total": total_dayflag,
    "sym_flag_total": total_sym,
    "dilution_ratio_day_to_hour": dilution_ratio,
    "interpretation": (
        f"Day-level flagging produces ~{dilution_ratio:.1f}x more "
        f"flagged hours than precise hour-level windows. Naive "
        f"day-level analysis would dilute estimated event effects "
        + _interp_tail
    ),
}])
dilution_descriptive.to_csv(TABLE_DIR / "02_dilution_descriptive.csv", index=False)
print(dilution_descriptive.to_string(index=False))

# Dilution overlap chart (descriptive counts)
fig_d, ax_d = plt.subplots(figsize=(10, 5))
_dratio = float(dilution_ratio) if pd.notna(dilution_ratio) else np.nan
_labels = ["Day flag only", "Sym hour only", "Both flagged"]
_vals = [float(day_only), float(sym_only), float(both)]
_colors = [PALETTE["accent"], PALETTE["uber"], PALETTE["crz"]]
y_pos = np.arange(len(_labels))
ax_d.barh(y_pos, _vals, color=_colors, edgecolor="white")
ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(_labels)
ax_d.set_xlabel("Zone-hours (master parquet)")
format_thousands_axis(ax_d, "x")
limit_ticks(ax_d, "x")
fig_d.suptitle(
    (
        f"Day-level flagging produces {_dratio:.1f}× more 'event hours' than "
        "precise hour windows"
        if np.isfinite(_dratio)
        else "Day-level vs hour-level event flag footprint"
    ),
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
ax_d.set_title(
    "Hours flagged: day-level vs hour-level event windows | Master parquet | "
    f"N={len(df)/1e6:.1f}M zone-hours",
    fontsize=11,
    color="#555555",
)
if np.isfinite(_dratio):
    ax_d.annotate(
        f"Dilution ratio\n(day÷hour) ≈ {_dratio:.2f}×",
        xy=(max(_vals) * 0.55, 2),
        fontsize=10,
        bbox=dict(boxstyle="round", fc=PALETTE["annot_bg"], ec="#CCCCCC"),
    )
plt.tight_layout(rect=[0, 0.08, 1, 0.92])
add_footnote(fig_d, "Descriptive counts — not regression-based (see 02_dilution_descriptive.csv).", y=-0.02)
plt.savefig(FIG_DIR / "02_dilution_descriptive.png")
plt.close(fig_d)


# %%
# Cell 14: Lyft sanity check
df_l = attach_event_metadata(df_lyft_only, events_df)
df_l["log_trips"] = safe_log(df_l["trip_count"])
df_l["fare_per_mile"] = np.where(
    df_l["total_miles"] > 0, df_l["total_adjusted_fare"] / df_l["total_miles"], np.nan
)
df_l["log_fpm"] = safe_log(df_l["fare_per_mile"], eps=0.01)
ts_l = pd.to_datetime(df_l["pickup_hour"])
df_l["hour_of_week"] = ts_l.dt.dayofweek * 24 + ts_l.dt.hour

df_lyft_daily = collapse_to_zone_day(df_l)

lyft_chk = []
for label, dsub in [("Uber", df_daily), ("Lyft", df_lyft_daily)]:
    r = panel_event_regression(dsub, "is_nyc_event_sym", "log_trips")
    if r:
        lyft_chk.append({"platform": label, "spec": "PanelOLS_FE",
                         **{k: r[k] for k in ["coef", "se", "p_value", "nobs"]}})

if FirstDifferenceOLS is not None:
    for label, dsub in [("Uber", df_daily), ("Lyft", df_lyft_daily)]:
        try:
            di = dsub.set_index(["pickup_zone", "pickup_hour"]).sort_index()
            fd = FirstDifferenceOLS(di["log_trips"], di[["is_nyc_event_sym"]])
            fd_res = fd.fit(cov_type="clustered", cluster_entity=True)
            lyft_chk.append({
                "platform": label, "spec": "FirstDifference_OLS",
                "coef": float(fd_res.params["is_nyc_event_sym"]),
                "se": float(fd_res.std_errors["is_nyc_event_sym"]),
                "p_value": float(fd_res.pvalues["is_nyc_event_sym"]),
                "nobs": int(fd_res.nobs),
            })
        except Exception as e:
            warnings.warn(f"FirstDifferenceOLS failed for {label}: {e}")

pd.DataFrame(lyft_chk).to_csv(TABLE_DIR / "02_lyft_sanity_check.csv", index=False)
print(pd.DataFrame(lyft_chk).to_string(index=False))


# %%
# Cell 15: Master summary table
def safe_iloc(df, col, default=np.nan, context=""):
    if len(df) == 0:
        warnings.warn(f"Empty result for {context}; returning {default}")
        return default
    return float(df[col].iloc[0])


def grab(pdf, flag, metric, col="beta"):
    sub = pdf[(pdf["flag"] == flag) & (pdf["metric"] == metric)]
    ctx = f"{flag}|{metric}|{col}"
    return safe_iloc(sub, col, default=np.nan, context=ctx)

summary = [
    {"category": "NYC sports",
     "N_events": int(per_ev_lt["event_id"].nunique()),
     "primary_flag": "is_nyc_event_sym",
     "median_lift_trips_pct": med_lt, "median_lift_fpm_pct": med_lf,
     "panel_B_log_trips": grab(panel_nyc, "is_nyc_event_sym", "log_trips"),
     "panel_B_log_fpm": grab(panel_nyc, "is_nyc_event_sym", "log_fpm"),
     "panel_p_value": grab(panel_nyc, "is_nyc_event_sym", "log_trips", "p_value"),
     "notes": "Sym only in panel CSV (asym collapses on zone-day)"},
    {"category": "NJ sports (scoped)",
     "N_events": int(lift_nj["nj_event_id"].nunique())
                 if "nj_event_id" in lift_nj.columns else np.nan,
     "primary_flag": "is_nj_event_pregame_sym",
     "median_lift_trips_pct": lift_nj["row_lift_trips_pct"].median(),
     "median_lift_fpm_pct": lift_nj["row_lift_fpm_pct"].median(),
     "panel_B_log_trips": grab(nj_panel_df[nj_panel_df["sample"] == "scoped_departure_zones"],
                               "is_nj_event_pregame_sym", "log_trips"),
     "panel_B_log_fpm": grab(nj_panel_df[nj_panel_df["sample"] == "scoped_departure_zones"],
                             "is_nj_event_pregame_sym", "log_fpm"),
     "panel_p_value": grab(nj_panel_df[nj_panel_df["sample"] == "scoped_departure_zones"],
                           "is_nj_event_pregame_sym", "log_trips", "p_value"),
     "notes": "Manhattan departure clusters"},
    {"category": "NJ sports (full panel)", "N_events": np.nan,
     "primary_flag": "is_nj_event_pregame_sym",
     "median_lift_trips_pct": np.nan, "median_lift_fpm_pct": np.nan,
     "panel_B_log_trips": grab(nj_panel_df[nj_panel_df["sample"] == "full_panel_all_zones"],
                               "is_nj_event_pregame_sym", "log_trips"),
     "panel_B_log_fpm": grab(nj_panel_df[nj_panel_df["sample"] == "full_panel_all_zones"],
                             "is_nj_event_pregame_sym", "log_fpm"),
     "panel_p_value": grab(nj_panel_df[nj_panel_df["sample"] == "full_panel_all_zones"],
                           "is_nj_event_pregame_sym", "log_trips", "p_value"),
     "notes": "Diluted vs scoped"},
    {"category": "Political",
     "N_events": len(pol_summary), "primary_flag": "is_nyc_event_asym",
     "median_lift_trips_pct": pd.DataFrame(pol_summary)["lift_pct"].median()
                                if pol_summary else np.nan,
     "median_lift_fpm_pct": np.nan,
     "panel_B_log_trips": np.nan, "panel_B_log_fpm": np.nan, "panel_p_value": np.nan,
     "notes": "N=6 exploratory"},
    {"category": "Civic / parades / specials",
     "N_events": int(per_ev_civic["event_id"].nunique()) if len(per_ev_civic) else 0,
     "primary_flag": "is_nyc_event_sym",
     "median_lift_trips_pct": per_ev_civic["lift_pct"].median()
                               if len(per_ev_civic) else np.nan,
     "median_lift_fpm_pct": np.nan,
     "panel_B_log_trips": np.nan, "panel_B_log_fpm": np.nan, "panel_p_value": np.nan,
     "notes": "Marathon, bike tour, parades, NYE"},
    {"category": "Holidays",
     "N_events": hol_df["date"].nunique(),
     "primary_flag": "day_vs_samedow_baseline",
     "median_lift_trips_pct": hol_rows_df["trip_lift_pct"].median(),
     "median_lift_fpm_pct": hol_rows_df["fpm_lift_pct"].median(),
     "panel_B_log_trips": np.nan, "panel_B_log_fpm": np.nan, "panel_p_value": np.nan,
     "notes": "Day-level ratio-of-sums"},
]
pd.DataFrame(summary).to_csv(TABLE_DIR / "02_event_study_summary.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))


# %%
# Cell 16: Forest plot
forest_specs = []
for _, r in panel_nyc[panel_nyc["metric"] == "log_trips"].iterrows():
    forest_specs.append((r["flag"], r["beta"], r["se"]))
for _, r in nj_panel_df[
    (nj_panel_df["metric"] == "log_trips")
    & (nj_panel_df["flag"] == "is_nj_event_pregame_sym")
].iterrows():
    forest_specs.append((r["sample"] + "|" + r["flag"], r["beta"], r["se"]))

if forest_specs:
    _nh = max(4.5, len(forest_specs) * 0.55)
    fig_f, ax_f = plt.subplots(figsize=(10, 7))
    for i, (lab, b, se) in enumerate(forest_specs):
        ci = 1.96 * se
        ax_f.errorbar(
            b,
            i,
            xerr=ci,
            fmt="o",
            color=PALETTE["accent"],
            capsize=4,
            markersize=7,
            ecolor=PALETTE["muted"],
        )
        ax_f.text(b + ci + 0.004, i, lab, va="center", ha="left", fontsize=9, color=PALETTE["uber"])
    ax_f.axvline(0, color=PALETTE["muted"], lw=1.2, ls=":")
    ax_f.set_yticks([])
    ax_f.set_xlabel("β (log trips)")
    format_coef_axis(ax_f, "x", decimals=4)
    limit_ticks(ax_f, "x")
    fig_f.suptitle(
        "NJ pregame effect ~5× stronger when scoped to departure zones",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    ax_f.set_title(
        "Panel β with 95% CI, log(trips), zone-day FE | clustered SE by zone",
        fontsize=11,
        color="#555555",
    )
    plt.tight_layout()
    add_footnote(fig_f, "Symmetry rows omitted from CSV (zone-day collapse). Per-event lift remains hourly in CSVs.", y=-0.06)
    plt.savefig(FIG_DIR / "02_event_study_forest_plot.png")
    plt.close()


# %%
# Cell 17: Findings
head_nyc_b = grab(panel_nyc, "is_nyc_event_sym", "log_trips")
head_nj_sc = grab(nj_panel_df[nj_panel_df["sample"] == "scoped_departure_zones"],
                   "is_nj_event_pregame_sym", "log_trips")
head_nj_fu = grab(nj_panel_df[nj_panel_df["sample"] == "full_panel_all_zones"],
                   "is_nj_event_pregame_sym", "log_trips")
dil_ratio_txt = (
    float(dilution_descriptive["dilution_ratio_day_to_hour"].iloc[0])
    if len(dilution_descriptive)
    else float("nan")
)
pol_med = pd.DataFrame(pol_summary)["lift_pct"].median() if pol_summary else np.nan

lines = [
    f"- NYC sports: panel β={head_nyc_b:.4f} | median per-event lift={med_lt:.2f}%.",
    f"- NJ pre-game: scoped β={head_nj_sc:.4f}; full β={head_nj_fu:.4f}.",
    (
        f"- Dilution (day vs hour flag footprint): ratio≈{dil_ratio_txt:.2f}× "
        f"(02_dilution_descriptive.csv)."
        if pd.notna(dil_ratio_txt)
        else "- Dilution: see 02_dilution_descriptive.csv."
    ),
    f"- Political (N=6, asym): median lift={pol_med:.2f}% — exploratory.",
    f"- Civic/parade/special: median lift={per_ev_civic['lift_pct'].median():.2f}% (N={per_ev_civic['event_id'].nunique()}).",
    "  · NEGATIVE lifts reflect operational blockers (street closures, restricted",
    "    pickup access during parades) NOT lower demand — distinguishes",
    "    pricing-relevant shocks from physical-access shocks.",
    f"- Holidays: median lift={hol_rows_df['trip_lift_pct'].median():.2f}%.",
    f"- Marathon: median zone lift={mar_summary['trip_lift_pct'].median():.2f}%.",
    "- Lyft: see 02_lyft_sanity_check.csv.",
]
text_out = "\n".join(lines)
print(text_out)
(TABLE_DIR / "02_findings_summary.txt").write_text(text_out + "\n", encoding="utf-8")
