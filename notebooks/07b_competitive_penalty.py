# %%
# Cell 1 — Setup + FIX D per-shock cross-price elasticity
"""
Notebook 07b — Competitive response penalty (pricing vs supply framing).

FIX D: shock-specific cross-price elasticity η from simultaneous Uber vs Lyft
trip lifts and observed Uber fare-per-mile lift (hour-level; Lam & Liu 2017 MIT
IDE motivates cross-platform substitution timing; we replicate coarsely).

η_obs ≈ max(0, (lyft_lift_pct − uber_lift_pct) / uber_fpm_lift_pct).

Bracket ±50%; fall back to literature midpoint 0.4 if η ∉ [0, 1] or invalid denom.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\Thomas\OneDrive\Desktop\Uber")
os.chdir(PROJECT_ROOT)

TABLE_DIR = PROJECT_ROOT / "outputs/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MASTER_PARQUET = (PROJECT_ROOT / "data/processed/master_zone_hour.parquet").as_posix()
LIT_ETA_MID = 0.4

con = duckdb.connect()
con.execute(
    f"CREATE OR REPLACE VIEW master AS SELECT * FROM read_parquet('{MASTER_PARQUET}')"
)

shares = con.execute(
    """
    WITH shock_periods AS (
        SELECT pickup_hour, pickup_zone, platform, trip_count, total_adjusted_fare
        FROM master
        WHERE COALESCE(has_major_event_dayflag, 0) = 1
           OR COALESCE(precip_in, 0) > 0.30
           OR COALESCE(is_storm_active, 0) = 1
    ),
    by_platform AS (
        SELECT platform,
               SUM(trip_count)::DOUBLE AS total_trips,
               SUM(total_adjusted_fare)::DOUBLE AS total_fare
        FROM shock_periods
        GROUP BY platform
    )
    SELECT platform, total_trips, total_fare,
           total_fare / NULLIF((SELECT SUM(total_fare) FROM by_platform), 0) AS share
    FROM by_platform
    ORDER BY platform
"""
).df()
print("Market shares during shock periods (trip-weighted fare share):")
print(shares.to_string(index=False))
con.close()

# --- FIX D: per-shock η from Uber/Lyft panel (pandas; both platforms) ---
dfm = pd.read_parquet(PROJECT_ROOT / "data/processed/master_zone_hour.parquet", engine="pyarrow")
dfm["platform"] = dfm["platform"].astype(str).str.lower()
dfm["pickup_hour_dt"] = pd.to_datetime(dfm["pickup_hour"])
dfm["hour_of_week"] = dfm["pickup_hour_dt"].dt.dayofweek * 24 + dfm["pickup_hour_dt"].dt.hour
if "precip_in" in dfm.columns:
    dfm["heavy_rain"] = (dfm["precip_in"].astype(float) > 0.30).astype(int)
else:
    dfm["heavy_rain"] = 0

shock_specs_d = [
    ("NYC sports", "is_nyc_event_sym"),
    ("NJ pregame (departure)", "is_nj_event_pregame_sym"),
    ("Heavy rain", "heavy_rain"),
]
if "is_storm_active" in dfm.columns:
    shock_specs_d.append(("Storm active", "is_storm_active"))

all_flags_d = [f for _, f in shock_specs_d if f in dfm.columns]
if not all_flags_d:
    no_all_mask = pd.Series(True, index=dfm.index)
else:
    no_all_mask = dfm[all_flags_d].sum(axis=1) == 0

_shock_obs_path = TABLE_DIR / "07_shock_observed_with_supply.csv"
uber_fpm_lift_by_shock: dict[str, float] = {}
if _shock_obs_path.exists():
    _so = pd.read_csv(_shock_obs_path)
    if "shock" in _so.columns and "fpm_lift_pct" in _so.columns:
        uber_fpm_lift_by_shock = _so.set_index("shock")["fpm_lift_pct"].astype(float).to_dict()

eta_rows = []
for shock_name, flag in shock_specs_d:
    if flag not in dfm.columns:
        continue
    lifts: dict[str, float] = {}
    for plat in ("uber", "lyft"):
        base = (
            dfm.loc[(dfm["platform"] == plat) & no_all_mask]
            .groupby(["pickup_zone", "hour_of_week"], as_index=False)
            .agg(base_trips=("trip_count", "mean"))
        )
        sh = dfm.loc[(dfm["platform"] == plat) & (dfm[flag] == 1)].merge(
            base, on=["pickup_zone", "hour_of_week"], how="left"
        )
        ot = sh["trip_count"].astype(float).sum()
        bt = sh["base_trips"].astype(float).sum()
        lifts[plat] = float((ot / bt - 1.0) * 100.0) if bt > 0 else np.nan

    uber_lift_pct = lifts.get("uber", np.nan)
    lyft_lift_pct = lifts.get("lyft", np.nan)
    uber_fpm_lift_pct = float(uber_fpm_lift_by_shock.get(shock_name, np.nan))
    fallback_note = ""
    observed_eta = np.nan
    if pd.notna(uber_fpm_lift_pct) and abs(uber_fpm_lift_pct) > 1e-6:
        observed_eta = max(0.0, (lyft_lift_pct - uber_lift_pct) / uber_fpm_lift_pct)
    raw = float(observed_eta) if pd.notna(observed_eta) else np.nan
    if pd.notna(raw) and 0.0 <= raw <= 1.0:
        eta_low = max(0.0, raw * 0.5)
        eta_high = min(1.0, raw * 1.5)
        use_eta = raw
    else:
        use_eta = LIT_ETA_MID
        eta_low = max(0.0, use_eta * 0.5)
        eta_high = min(1.0, use_eta * 1.5)
        fallback_note = "fallback literature midpoint η=0.4 (invalid or out-of-range observed η)"
    eta_rows.append(
        {
            "shock": shock_name,
            "uber_lift_pct": uber_lift_pct,
            "lyft_lift_pct": lyft_lift_pct,
            "uber_fpm_lift_pct": uber_fpm_lift_pct,
            "observed_eta": float(observed_eta) if pd.notna(observed_eta) else np.nan,
            "eta_low": eta_low,
            "eta_high": eta_high,
            "eta_used": float(use_eta),
            "fallback_note": fallback_note,
        }
    )

per_shock_eta = pd.DataFrame(eta_rows)
per_shock_eta.to_csv(TABLE_DIR / "07b_per_shock_eta.csv", index=False)
print("\nFIX D — per-shock cross-price elasticity (hour-level):")
print(per_shock_eta.to_string(index=False))


# %%
# Cell 2 — Competitive penalty (per-shock η × implied price lift × DR allocation)
"""
Cohen et al. (2016) NBER w22627: own-price ε during surge; demand-response
retention written by N07 to 07_pricing_scenarios.csv.

Competitive penalty (per shock s, post own-price DR allocation w_s):
  penalty_s ≈ w_s × PRICING_AFTER_DR × η_s × (ΔP_s / 100)

Summed across shocks for aggregate low / central / high using η_s brackets.
"""
_pricing_headline = 19_948_660.0
_pricing_after_dr = 12_966_629.0
_supply_central = 6_982_031.0
_supply_low = 4_189_219.0
_supply_high = 9_774_844.0

_ann_path = TABLE_DIR / "07_annualized_totals.csv"
if _ann_path.exists():
    ann = pd.read_csv(_ann_path)
    if "spec" in ann.columns and "annualized_capture" in ann.columns:
        _p = ann.loc[ann["spec"] == "pricing", "annualized_capture"]
        _s = ann.loc[ann["spec"] == "supply", "annualized_capture"]
        if len(_p):
            _pricing_headline = float(_p.iloc[0])
        if len(_s):
            _supply_central = float(_s.iloc[0])

_sens_path = TABLE_DIR / "07_supply_sensitivity.csv"
if _sens_path.exists():
    st = pd.read_csv(_sens_path).groupby("supply_capture_frac", as_index=False)[
        "annualized_capture"
    ].sum()
    if 0.30 in set(st["supply_capture_frac"]):
        _supply_low = float(st.loc[st["supply_capture_frac"] == 0.30, "annualized_capture"].iloc[0])
    if 0.70 in set(st["supply_capture_frac"]):
        _supply_high = float(st.loc[st["supply_capture_frac"] == 0.70, "annualized_capture"].iloc[0])
    if 0.50 in set(st["supply_capture_frac"]):
        _supply_central = float(st.loc[st["supply_capture_frac"] == 0.50, "annualized_capture"].iloc[0])

_ps_path = TABLE_DIR / "07_pricing_scenarios.csv"
if _ps_path.exists():
    pricing_scenarios = pd.read_csv(TABLE_DIR / "07_pricing_scenarios.csv")
    PRICING_AFTER_DEMAND_RESPONSE = float(
        pricing_scenarios.loc[
            pricing_scenarios["metric"] == "pricing_after_demand_response",
            "value_usd",
        ].iloc[0]
    )
    _ph = pricing_scenarios.loc[pricing_scenarios["metric"] == "pricing_headline", "value_usd"]
    if len(_ph):
        _pricing_headline = float(_ph.iloc[0])
else:
    PRICING_AFTER_DEMAND_RESPONSE = _pricing_after_dr

PRICING_HEADLINE = _pricing_headline
SUPPLY_CENTRAL = _supply_central
SUPPLY_LOW = _supply_low
SUPPLY_HIGH = _supply_high

strategy_csv = pd.read_csv(TABLE_DIR / "07_strategy_comparison.csv")
pricing_df = strategy_csv[strategy_csv["spec"] == "pricing"].copy()
_eta_path = TABLE_DIR / "07b_per_shock_eta.csv"
eta_df = pd.read_csv(_eta_path) if _eta_path.exists() else pd.DataFrame()
eta_by_shock = eta_df.set_index("shock") if len(eta_df) and "shock" in eta_df.columns else pd.DataFrame().set_index(
    pd.Index([], name="shock")
)

sum_p_headline = max(float(pricing_df["annualized_capture"].sum()), 1e-9)
per_shock_pricing_net: dict[str, float] = {}
penalty_central_total = 0.0
penalty_low_total = 0.0
penalty_high_total = 0.0

for _, prow in pricing_df.iterrows():
    shock = str(prow["shock"])
    cap_h = float(prow["annualized_capture"])
    dp = float(prow["implied_price_lift_pct"]) if pd.notna(prow["implied_price_lift_pct"]) else 0.0
    w = (cap_h / sum_p_headline) if sum_p_headline > 0 else 0.0
    alloc_dr = PRICING_AFTER_DEMAND_RESPONSE * w
    if shock in eta_by_shock.index and len(eta_by_shock):
        er = eta_by_shock.loc[shock]
        eta_c = float(er["eta_used"])
        eta_lo = float(er["eta_low"])
        eta_hi = float(er["eta_high"])
    else:
        eta_c = eta_lo = eta_hi = LIT_ETA_MID
    pen_c = alloc_dr * eta_c * (dp / 100.0)
    pen_lo = alloc_dr * eta_lo * (dp / 100.0)
    pen_hi = alloc_dr * eta_hi * (dp / 100.0)
    penalty_central_total += pen_c
    penalty_low_total += pen_lo
    penalty_high_total += pen_hi
    net_s = alloc_dr - pen_c
    per_shock_pricing_net[shock] = net_s

avg_price_lift_pct = float(pricing_df["implied_price_lift_pct"].mean())

pricing_net_central = PRICING_AFTER_DEMAND_RESPONSE - penalty_central_total
pricing_net_low = PRICING_AFTER_DEMAND_RESPONSE - penalty_high_total
pricing_net_high = PRICING_AFTER_DEMAND_RESPONSE - penalty_low_total

print(f"Average implied price lift (mean across pricing rows): {avg_price_lift_pct:.1f}%")
print("\nCompetitive penalty (per-shock η, summed; applied to post-DR pricing capture):")
print(
    f"  Low bracket:     ${penalty_low_total/1e6:.2f}M → net pricing ${pricing_net_high/1e6:.2f}M"
)
print(
    f"  Central (η_used): ${penalty_central_total/1e6:.2f}M → net pricing ${pricing_net_central/1e6:.2f}M"
)
print(
    f"  High bracket:    ${penalty_high_total/1e6:.2f}M → net pricing ${pricing_net_low/1e6:.2f}M"
)

print("\nFinal comparison (annualized $):")
print(
    f"  Supply: ${SUPPLY_LOW/1e6:.1f}M - ${SUPPLY_HIGH/1e6:.1f}M "
    f"(central ${SUPPLY_CENTRAL/1e6:.1f}M)"
)
print(
    f"  Pricing (net of competitive penalty): ${pricing_net_low/1e6:.1f}M - "
    f"${pricing_net_high/1e6:.1f}M (central ${pricing_net_central/1e6:.1f}M)"
)


# %%
# Cell 3 — Save outputs
results = pd.DataFrame(
    [
        {"scenario": "Pricing headline", "value_M": PRICING_HEADLINE / 1e6, "note": "before demand response"},
        {
            "scenario": "Pricing after demand response",
            "value_M": PRICING_AFTER_DEMAND_RESPONSE / 1e6,
            "note": "from 07_pricing_scenarios.csv",
        },
        {
            "scenario": "Pricing net of competitive penalty (low)",
            "value_M": pricing_net_low / 1e6,
            "note": "sum_s alloc×η_high×ΔP_s (FIX D brackets)",
        },
        {
            "scenario": "Pricing net of competitive penalty (central)",
            "value_M": pricing_net_central / 1e6,
            "note": "sum_s alloc×η_used×ΔP_s",
        },
        {
            "scenario": "Pricing net of competitive penalty (high)",
            "value_M": pricing_net_high / 1e6,
            "note": "sum_s alloc×η_low×ΔP_s",
        },
        {"scenario": "Supply central", "value_M": SUPPLY_CENTRAL / 1e6, "note": "50% capture frac (N07 central)"},
        {"scenario": "Supply low sensitivity", "value_M": SUPPLY_LOW / 1e6, "note": "30% capture frac"},
        {"scenario": "Supply high sensitivity", "value_M": SUPPLY_HIGH / 1e6, "note": "70% capture frac"},
    ]
)
results.to_csv(TABLE_DIR / "07b_competitive_penalty.csv", index=False)
print(results.to_string(index=False))


# %%
# Cell 4 — Findings narrative + post-fix summary
_net_rng = f"${pricing_net_low/1e6:.1f}M - ${pricing_net_high/1e6:.1f}M"
_sup_rng = f"${SUPPLY_LOW/1e6:.1f}M - ${SUPPLY_HIGH/1e6:.1f}M"

findings = [
    "COMPETITIVE RESPONSE PENALTY — Path 3-lite analysis (FIX D: per-shock η)",
    "",
    "ACADEMIC ANCHORS:",
    "  - Cohen et al. 2016 NBER w22627: own-price elasticity during surge context",
    "  - Lam & Liu 2017 MIT IDE: cross-platform substitution (5-min windows);",
    "    we use hour-level Uber vs Lyft trip lifts / Uber fpm lift as coarse η.",
    "",
    "CALCULATION:",
    f"  Per-shock η from 07b_per_shock_eta.csv; implied ΔP_s from N07 pricing rows.",
    f"  Aggregate competitive penalty (central): ${penalty_central_total/1e6:.2f}M.",
    "",
    "RESULT:",
    f"  Pricing strategy NET capture: {_net_rng}",
    f"  Supply strategy capture: {_sup_rng}",
    "",
    "INTERPRETATION:",
    "  After shock-specific competitive response, compare net pricing band to supply.",
    "",
    "CAVEATS:",
    "  - Hour-level η is noisy; ±50% brackets and literature fallback for invalid η.",
    "  - Regulatory / franchise factors not quantified here.",
]
text_out = "\n".join(findings)
print("\n" + text_out)
(TABLE_DIR / "07b_findings_summary.txt").write_text(text_out + "\n", encoding="utf-8")

NB_NAME = "N07b competitive penalty"
print(f"\n=== {NB_NAME} POST-FIX SUMMARY ===")
print("Per-shock η_used:", per_shock_eta[["shock", "eta_used", "fallback_note"]].to_string(index=False))
print("Per-shock pricing nets (post-DR alloc − competitive penalty, central η):", per_shock_pricing_net)
