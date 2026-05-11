# %%
# Cell 1: Setup
"""
Notebook 07 — Strategy Comparison: Pricing vs Supply Optimization

Runs parallel counterfactuals on per-shock data:
  COUNTERFACTUAL A — PRICING optimization
    Raise fare-per-mile to demand-implied level during shocks
    Capture: ΔP × Q (with elasticity-adjusted Q)
    Public framing: extractive; competitive risk if Lyft doesn't follow

  COUNTERFACTUAL B — SUPPLY optimization
    Pre-position drivers in shock zones BEFORE event start
    Capture: Δtrips at observed base fare-per-mile
    Public framing: operational; serves more riders

Caveat: surge multiplier and driver positioning are INFERRED, not observed.
We use total_driver_pay and fare_per_mile as proxies for supply tightness.
"""
from __future__ import annotations

# REPRODUCIBILITY: All random sampling uses fixed seed=42 (or 1, where noted).
# Regression results are deterministic given fixed input data.

import os
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")

PROJECT_ROOT = Path.cwd()
FIG_DIR = PROJECT_ROOT / "outputs/figures"
TABLE_DIR = PROJECT_ROOT / "outputs/tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

from outputs.visualization_design_system import (
    PALETTE,
    add_footnote,
    format_dollar_axis,
    limit_ticks,
    set_rcparams,
)

set_rcparams()

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

df = df_master[df_master["platform"] == "uber"].copy()
print(f"Uber rows: {len(df):,}")


# %%
# Cell 2: Feature prep + supply-tightness proxies
avg_miles_per_trip = duckdb.query("""
    SELECT SUM(total_miles) / SUM(trip_count) AS avg_miles
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE platform = 'uber' AND trip_count > 0
""").df()["avg_miles"].iloc[0]
print(f"Average Uber trip distance (data-derived): {avg_miles_per_trip:.2f} miles")
AVG_TRIP_MILES = avg_miles_per_trip

df["fare_per_mile"] = np.where(
    df["total_miles"].astype(float) > 0,
    df["total_adjusted_fare"].astype(float) / df["total_miles"].astype(float),
    np.nan,
)
df["pickup_hour_dt"] = pd.to_datetime(df["pickup_hour"])
df["pickup_date"] = df["pickup_hour_dt"].dt.normalize()
df["hour_of_week"] = df["pickup_hour_dt"].dt.dayofweek * 24 + df["pickup_hour_dt"].dt.hour

# Supply tightness proxies (we don't have driver count directly):
#  - driver_pay_per_trip = total_driver_pay / trip_count (rises when drivers are scarce/incentivized)
#  - fare_per_driver_pay = total_adjusted_fare / total_driver_pay (rises when surge active)
if "total_driver_pay" in df.columns:
    df["driver_pay_per_trip"] = np.where(
        df["trip_count"] > 0,
        df["total_driver_pay"] / df["trip_count"],
        np.nan,
    )
    df["fare_to_pay_ratio"] = np.where(
        df["total_driver_pay"] > 0,
        df["total_adjusted_fare"] / df["total_driver_pay"],
        np.nan,
    )
    print(f"Driver pay per trip — overall median: ${df['driver_pay_per_trip'].median():.2f}")
    print(f"Fare/pay ratio — overall median: {df['fare_to_pay_ratio'].median():.2f}")
else:
    print("⚠️ total_driver_pay not in master; proxies unavailable")
    df["driver_pay_per_trip"] = np.nan
    df["fare_to_pay_ratio"] = np.nan

# Heavy rain flag
if "precip_in" in df.columns:
    df["heavy_rain"] = (df["precip_in"] > 0.30).astype(int)
else:
    df["heavy_rain"] = 0


# %%
# Cell 3: Per-shock observed metrics with supply proxy
shock_specs = [
    {"name": "NYC sports", "flag": "is_nyc_event_sym"},
    {"name": "NJ pregame (departure)", "flag": "is_nj_event_pregame_sym"},
    {"name": "Heavy rain", "flag": "heavy_rain"},
]
if "is_storm_active" in df.columns:
    shock_specs.append({"name": "Storm active", "flag": "is_storm_active"})

# Compute baseline once (no shock at all)
all_shock_flags = [s["flag"] for s in shock_specs if s["flag"] in df.columns]
no_shock_mask = (df[all_shock_flags].sum(axis=1) == 0)
baseline_lookup = (
    df[no_shock_mask]
    .groupby(["pickup_zone", "hour_of_week"], as_index=False)
    .agg(base_trips=("trip_count", "mean"),
         base_fpm=("fare_per_mile", "mean"),
         base_pay_per_trip=("driver_pay_per_trip", "mean"),
         base_fare_pay_ratio=("fare_to_pay_ratio", "mean"))
)

shock_metrics = []
for spec in shock_specs:
    flag = spec["flag"]
    if flag not in df.columns:
        continue
    shock_hours = df[df[flag] == 1].copy()
    if len(shock_hours) < 100:
        continue

    sw = shock_hours.merge(baseline_lookup,
                           on=["pickup_zone", "hour_of_week"], how="left")
    obs_trips = sw["trip_count"].sum()
    base_trips = sw["base_trips"].sum()
    obs_fpm = (sw["total_adjusted_fare"].sum() /
               sw["total_miles"].sum() if sw["total_miles"].sum() > 0 else np.nan)
    base_fpm = sw["base_fpm"].mean()
    obs_pay_per_trip = sw["driver_pay_per_trip"].mean()
    base_pay_per_trip = sw["base_pay_per_trip"].mean()
    obs_fare_pay = sw["fare_to_pay_ratio"].mean()
    base_fare_pay = sw["base_fare_pay_ratio"].mean()
    # FIX E: shock-specific trip distance from observed shock-hour trips (Uber).
    shock_avg_miles = (
        float(sw["total_miles"].astype(float).sum() / sw["trip_count"].astype(float).sum())
        if sw["trip_count"].sum() > 0
        else np.nan
    )

    shock_metrics.append({
        "shock": spec["name"], "flag": flag, "n_shock_hours": len(shock_hours),
        "obs_trips": int(obs_trips), "base_trips": float(base_trips),
        "trip_lift_pct": (obs_trips/base_trips - 1) * 100 if base_trips > 0 else np.nan,
        "obs_fpm": obs_fpm, "base_fpm": base_fpm,
        "fpm_lift_pct": (obs_fpm/base_fpm - 1) * 100 if base_fpm > 0 else np.nan,
        "obs_pay_per_trip": obs_pay_per_trip,
        "pay_lift_pct": ((obs_pay_per_trip/base_pay_per_trip - 1) * 100
                          if (pd.notna(base_pay_per_trip) and base_pay_per_trip > 0) else np.nan),
        "obs_fare_to_pay": obs_fare_pay, "base_fare_to_pay": base_fare_pay,
        "shock_avg_miles": shock_avg_miles,
    })

shock_obs = pd.DataFrame(shock_metrics)
shock_obs.to_csv(TABLE_DIR / "07_shock_observed_with_supply.csv", index=False)
print(shock_obs.to_string(index=False))


# %%
# Cell 4: Counterfactual A — Pricing optimization
"""
At elasticity ε, optimal price increase = excess_demand% / |ε|.
Revenue capture per shock-hour = (P_optimal - P_observed) × Q_observed × avg_miles.
"""
ELASTICITY_CENTRAL = -0.7

ctf_a_rows = []
for _, row in shock_obs.iterrows():
    # FIX A: Cohen et al. 2016 — pricing counterfactual only for excess-demand shocks;
    # storms are demand collapses, not surge-clearing events.
    _is_storm = str(row["shock"]) == "Storm active"
    _trip_bad = pd.isna(row["trip_lift_pct"]) or row["trip_lift_pct"] <= 0
    if _is_storm or _trip_bad:
        ctf_a_rows.append({
            "shock": row["shock"], "spec": "pricing",
            "implied_price_lift_pct": np.nan,
            "rev_capture_per_hr": 0,
            "annualized_capture": 0,
            "note": (
                "demand FALLING — pricing inapplicable per Cohen 2016 scoping"
                if _is_storm
                else "demand dropped — pricing not the lever"
            ),
        })
        continue
    _miles = (
        float(row["shock_avg_miles"])
        if "shock_avg_miles" in row.index and pd.notna(row["shock_avg_miles"])
        else float(AVG_TRIP_MILES)
    )
    impl_pct = row["trip_lift_pct"] / abs(ELASTICITY_CENTRAL)
    new_fpm = row["obs_fpm"] * (1 + impl_pct / 100)
    trips_per_hr = row["obs_trips"] / row["n_shock_hours"]
    cap_per_hr = (new_fpm - row["obs_fpm"]) * _miles * trips_per_hr
    annual = cap_per_hr * row["n_shock_hours"]
    ctf_a_rows.append({
        "shock": row["shock"], "spec": "pricing",
        "implied_price_lift_pct": impl_pct,
        "rev_capture_per_hr": cap_per_hr,
        "annualized_capture": annual,
        "note": "extractive framing",
    })


# %%
# Cell 5: Counterfactual B — Supply optimization
"""
If supply were pre-positioned to match excess demand, additional trips at
BASE fare-per-mile = capture (without raising prices).

Supply gap proxy: trip_lift_pct (the "missing" trips that would happen if
supply scaled with demand).
"""
# Literature-anchored supply capture range:
# Hall, Kendrick & Nosko 2015 (Uber): driver supply increases ~10-20% under
#   2x surge -> suggests 30-50% capture of demand spikes is feasible with
#   moderate pre-positioning incentives
# Chen & Sheldon 2016: Uber driver labor supply elasticity ~1.0
# Cohen et al. 2016: similar findings
# For forecastable shocks (better than reactive surge), upper end is more
# plausible because drivers can plan ahead. Range tested: [0.30, 0.50, 0.70].
SUPPLY_CAPTURE_FRACS = [0.30, 0.50, 0.70]
SUPPLY_CAPTURE_CENTRAL = 0.50  # central estimate
ctf_b_rows = []
for _, row in shock_obs.iterrows():
    if pd.isna(row["trip_lift_pct"]) or row["trip_lift_pct"] <= 0:
        ctf_b_rows.append({
            "shock": row["shock"], "spec": "supply",
            "implied_extra_trips_pct": np.nan,
            "rev_capture_per_hr": 0,
            "annualized_capture": 0,
            "note": "no demand spike to capture",
        })
        continue
    _miles_b = (
        float(row["shock_avg_miles"])
        if "shock_avg_miles" in row.index and pd.notna(row["shock_avg_miles"])
        else float(AVG_TRIP_MILES)
    )
    extra_trips_pct = row["trip_lift_pct"] * SUPPLY_CAPTURE_CENTRAL
    extra_trips_per_hr = (row["obs_trips"] / row["n_shock_hours"]) * (extra_trips_pct / 100)
    cap_per_hr = extra_trips_per_hr * _miles_b * row["obs_fpm"]
    annual = cap_per_hr * row["n_shock_hours"]
    ctf_b_rows.append({
        "shock": row["shock"], "spec": "supply",
        "implied_extra_trips_pct": extra_trips_pct,
        "rev_capture_per_hr": cap_per_hr,
        "annualized_capture": annual,
        "note": "operational framing",
    })

sensitivity_rows = []
for frac in SUPPLY_CAPTURE_FRACS:
    for _, row in shock_obs.iterrows():
        if pd.isna(row["trip_lift_pct"]) or row["trip_lift_pct"] <= 0:
            continue
        _miles_s = (
            float(row["shock_avg_miles"])
            if "shock_avg_miles" in row.index and pd.notna(row["shock_avg_miles"])
            else float(AVG_TRIP_MILES)
        )
        extra_trips_pct = row["trip_lift_pct"] * frac
        extra_trips_per_hr = (row["obs_trips"] / row["n_shock_hours"]) * (extra_trips_pct / 100)
        cap_per_hr = extra_trips_per_hr * _miles_s * row["obs_fpm"]
        annual = cap_per_hr * row["n_shock_hours"]
        sensitivity_rows.append({
            "shock": row["shock"],
            "supply_capture_frac": frac,
            "annualized_capture": annual,
        })
pd.DataFrame(sensitivity_rows).to_csv(
    TABLE_DIR / "07_supply_sensitivity.csv", index=False
)

ctf_a = pd.DataFrame(ctf_a_rows)
ctf_b = pd.DataFrame(ctf_b_rows)
counterfactual_compare = pd.concat([ctf_a, ctf_b], ignore_index=True)
counterfactual_compare.to_csv(TABLE_DIR / "07_strategy_comparison.csv", index=False)
print(counterfactual_compare.to_string(index=False))


# %%
# Cell 6: Side-by-side bar chart
shocks_with_capture = (
    counterfactual_compare[counterfactual_compare["annualized_capture"] > 0]
    ["shock"].unique().tolist()
)

if len(shocks_with_capture):
    fig7, ax7 = plt.subplots(figsize=(11, 7))
    width = 0.36
    x = np.arange(len(shocks_with_capture))
    ap_vals, sup_vals = [], []
    for shock in shocks_with_capture:
        ap_vals.append(float(ctf_a.loc[ctf_a["shock"] == shock, "annualized_capture"].iloc[0]))
        sup_vals.append(float(ctf_b.loc[ctf_b["shock"] == shock, "annualized_capture"].iloc[0]))
    mx_bar = max(ap_vals + sup_vals + [1.0])
    for i, shock in enumerate(shocks_with_capture):
        a, b = ap_vals[i], sup_vals[i]
        ax7.bar(i - width / 2, a, width, color=PALETTE["uber"], edgecolor="white")
        ax7.bar(i + width / 2, b, width, color=PALETTE["crz"], edgecolor="white")
        ax7.text(
            i - width / 2,
            a + mx_bar * 0.02,
            f"${a/1e6:.2f}M",
            ha="center",
            fontsize=9,
            color=PALETTE["uber"],
        )
        ax7.text(
            i + width / 2,
            b + mx_bar * 0.02,
            f"${b/1e6:.2f}M",
            ha="center",
            fontsize=9,
            color=PALETTE["crz"],
        )
    ax7.set_xticks(x)
    ax7.set_xticklabels(shocks_with_capture, rotation=22, ha="right")
    ax7.set_ylabel("Annualized revenue capture ($)")
    format_dollar_axis(ax7, "y")
    limit_ticks(ax7, "y")
    tot_p = sum(ap_vals)
    tot_s = sum(sup_vals)
    ratio = tot_p / tot_s if tot_s > 0 else np.nan
    fig7.suptitle(
        (
            f"Pricing captures ~{ratio:.1f}× more headline revenue — "
            "supply is operationally & reputationally safer"
        )
        if pd.notna(ratio)
        else "Pricing vs supply optimization — counterfactual comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    ax7.set_title(
        "Counterfactual annualization by shock | same baseline lifts",
        fontsize=11,
        color="#555555",
    )
    if pd.notna(ratio) and tot_s > 0:
        ax7.annotate(
            f"Aggregate ratio\n(pricing ÷ supply) ≈ {ratio:.1f}×",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", fc=PALETTE["annot_bg"], ec="#CCCCCC"),
        )
    ax7.annotate(
        "Trade-off: pricing is extractive & fragile vs PR;\n"
        "supply pre-positioning preserves fares but caps upside.",
        xy=(0.98, 0.72),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        style="italic",
        color="#555555",
        bbox=dict(boxstyle="round", fc="#F8F8F8", ec="#DDDDDD"),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    add_footnote(
        fig7,
        "Pricing: extractive; supply: operational. Recommendation favors supply despite lower headline (see findings).",
        y=-0.02,
    )
    plt.savefig(FIG_DIR / "07_strategy_comparison.png")
    plt.close()


# %%
# Cell 7: Total annualized — both strategies
totals = counterfactual_compare.groupby("spec")["annualized_capture"].sum()
print("Annualized revenue capture by strategy:")
print(totals.to_string())
totals.to_csv(TABLE_DIR / "07_annualized_totals.csv")

# Own-price demand response (Cohen et al.-anchored): central retain frac 0.65
# of headline counterfactual capture (60-70% band in findings narrative).
PRICING_DEMAND_RESPONSE_RETAIN = 0.65
pricing_headline_usd = float(totals.get("pricing", 0.0))
pricing_after_demand_response_usd = pricing_headline_usd * PRICING_DEMAND_RESPONSE_RETAIN
pricing_summary = pd.DataFrame(
    [
        {"metric": "pricing_headline", "value_usd": pricing_headline_usd},
        {"metric": "pricing_after_demand_response", "value_usd": pricing_after_demand_response_usd},
    ]
)
pricing_summary.to_csv(TABLE_DIR / "07_pricing_scenarios.csv", index=False)
print("Wrote outputs/tables/07_pricing_scenarios.csv (headline vs demand-response-adjusted pricing).")


# %%
# Cell 8: Findings narrative
total_pricing = pricing_headline_usd
total_supply = float(totals.get("supply", 0))
ratio = total_supply / total_pricing if total_pricing > 0 else np.nan

_supply_sens = pd.read_csv(TABLE_DIR / "07_supply_sensitivity.csv")
_supply_tot = (
    _supply_sens.groupby("supply_capture_frac", as_index=False)["annualized_capture"].sum()
)
_low = float(_supply_tot.loc[_supply_tot["supply_capture_frac"] == 0.30, "annualized_capture"].iloc[0])
_high = float(_supply_tot.loc[_supply_tot["supply_capture_frac"] == 0.70, "annualized_capture"].iloc[0])

lines = [
    f"- COUNTERFACTUAL A (pricing): annualized capture = ${total_pricing:,.0f}.",
    f"- COUNTERFACTUAL B (supply): annualized capture = ${total_supply:,.0f}.",
    (f"- Ratio (supply/pricing): {ratio:.2f}x." if pd.notna(ratio) else ""),
    "- Supply capture fraction (0.50 central) anchored to literature:",
    "  Hall et al. 2015 + Chen & Sheldon 2016 (driver supply elasticity ~1.0).",
    f"- Sensitivity range: 30% capture = ${_low:,.0f}; 70% capture = ${_high:,.0f}.",
    "- Central estimate plausible for FORECASTABLE shocks (drivers can plan ahead);",
    "  reactive surge would be at lower bound.",
    "",
    "TRADE-OFF FRAMING (not a quantitative dominance claim):",
    f"- Pricing captures more headline revenue but is extractive: raises prices",
    f"  on captive riders during shocks. After elasticity haircut (60-70% of",
    f"  headline due to demand response), realistic capture is ~${pricing_after_demand_response_usd:,.0f}.",
    f"- Supply captures less but is operational: pre-positions drivers to",
    f"  serve more riders at base fare. Captures ~${total_supply:,.0f} (central",
    f"  case with literature-anchored sensitivity in 07_supply_sensitivity.csv).",
    "",
    "STRATEGIC RECOMMENDATION: Supply optimization, despite smaller headline.",
    "  · Aligns with Uber's post-NYC-strike regulatory posture.",
    "  · Avoids competitive vulnerability (Lyft holds prices = Uber loses share).",
    "  · Avoids PR risk (no surcharges on event/storm riders).",
    "  · Builds long-term driver supply infrastructure usable across all shocks.",
    "  · Tradeoff: lower per-shock revenue, higher franchise value.",
    "",
    "CAVEATS:",
    "- Surge multipliers and driver count are INFERRED, not observed.",
    "- fare_per_mile and total_driver_pay used as supply-tightness proxies.",
    "- Supply-capture fraction is scenario-based and should be updated with",
    "  observed driver response from pilot incentive programs.",
]
text_out = "\n".join(lines)
print(text_out)
(TABLE_DIR / "07_findings_summary.txt").write_text(text_out + "\n", encoding="utf-8")

NB_NAME = "N07 strategy"
_tp = float(totals.get("pricing", 0.0))
_ts = float(totals.get("supply", 0.0))
_storm_p = float(
    ctf_a.loc[ctf_a["shock"] == "Storm active", "annualized_capture"].sum()
    if len(ctf_a) and (ctf_a["shock"] == "Storm active").any()
    else 0.0
)
print(f"\n=== {NB_NAME} POST-FIX SUMMARY ===")
print("Total pricing capture (post-fix):", f"${_tp:,.0f}")
print("Total supply capture:", f"${_ts:,.0f}")
print("Storm active pricing capture (expect $0 post-FIX A):", f"${_storm_p:,.0f}")
