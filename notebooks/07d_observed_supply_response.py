# %%
"""
Notebook 07d — Observed driver supply response (sanity check, not a parameter change).

Compares realized trip expansion during each shock (Uber, baseline = same zone ×
hour-of-week when THAT shock flag is off) to the implied expansion under N07's
50% capture of the demand spike (trip_lift_pct × 0.5).

Appendix framing: is the 50% literature-anchored capture optimistic vs history?
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

NB_NAME = "N07d observed supply response"

PROJECT_ROOT = Path(r"C:\Users\Thomas\OneDrive\Desktop\Uber")
os.chdir(PROJECT_ROOT)
TABLE_DIR = PROJECT_ROOT / "outputs/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

MASTER = PROJECT_ROOT / "data/processed/master_zone_hour.parquet"
df = pd.read_parquet(MASTER, engine="pyarrow")
df = df[df["platform"].astype(str).str.lower() == "uber"].copy()
df["pickup_hour_dt"] = pd.to_datetime(df["pickup_hour"])
df["hour_of_week"] = df["pickup_hour_dt"].dt.dayofweek * 24 + df["pickup_hour_dt"].dt.hour
if "precip_in" in df.columns:
    df["heavy_rain"] = (df["precip_in"].astype(float) > 0.30).astype(int)
else:
    df["heavy_rain"] = 0

shock_specs = [
    ("NYC sports", "is_nyc_event_sym"),
    ("NJ pregame (departure)", "is_nj_event_pregame_sym"),
    ("Heavy rain", "heavy_rain"),
]
if "is_storm_active" in df.columns:
    shock_specs.append(("Storm active", "is_storm_active"))

_lift_path = TABLE_DIR / "07_shock_observed_with_supply.csv"
lift_by = {}
if _lift_path.exists():
    lo = pd.read_csv(_lift_path)
    if "shock" in lo.columns and "trip_lift_pct" in lo.columns:
        lift_by = lo.set_index("shock")["trip_lift_pct"].astype(float).to_dict()

rows = []
for shock_name, flag in shock_specs:
    if flag not in df.columns:
        continue
    m = df[["pickup_zone", "hour_of_week", "pickup_hour_dt", "trip_count", flag]].copy()
    base = (
        m.loc[m[flag] == 0]
        .groupby(["pickup_zone", "hour_of_week"], as_index=False)
        .agg(baseline_trips=("trip_count", "mean"))
    )
    sh = m.loc[m[flag] == 1].merge(base, on=["pickup_zone", "hour_of_week"], how="left")
    if len(sh) < 10:
        continue
    g = (
        sh.groupby(["pickup_hour_dt", "pickup_zone"], as_index=False)
        .agg(shock_trips=("trip_count", "sum"), baseline_trips=("baseline_trips", "mean"))
    )
    if g["baseline_trips"].sum() <= 0:
        continue
    observed_expansion_ratio = float(g["shock_trips"].mean() / g["baseline_trips"].mean())
    n_events = int(len(g))
    trip_lift = float(lift_by.get(shock_name, np.nan))
    if pd.notna(trip_lift):
        assumed_capture_at_50pct = 0.50 * (trip_lift / 100.0)
    else:
        assumed_capture_at_50pct = np.nan
    obs_lift = observed_expansion_ratio - 1.0
    if pd.notna(assumed_capture_at_50pct) and abs(assumed_capture_at_50pct) > 1e-9:
        ratio_observed_to_assumed = obs_lift / assumed_capture_at_50pct
    else:
        ratio_observed_to_assumed = np.nan

    if pd.notna(ratio_observed_to_assumed):
        if ratio_observed_to_assumed >= 1.0:
            interp = "Observed expansion meets or exceeds the 50%-capture-implied lift; 50% assumption is conservative on this metric."
        else:
            interp = (
                "Observed expansion is below the 50%-capture-implied lift; "
                "N07 supply capture may be optimistic unless pre-positioning closes the gap."
            )
    elif pd.notna(obs_lift):
        interp = "Compare observed trip lift to pilot / ops data; assumed capture unavailable."
    else:
        interp = "Insufficient data for ratio vs assumed capture."

    rows.append(
        {
            "shock": shock_name,
            "observed_expansion_ratio": observed_expansion_ratio,
            "assumed_capture_at_50pct": assumed_capture_at_50pct,
            "ratio_observed_to_assumed": ratio_observed_to_assumed,
            "n_events": n_events,
            "interpretation": interp,
        }
    )

out = pd.DataFrame(rows)
out.to_csv(TABLE_DIR / "07d_observed_supply.csv", index=False)
print(out.to_string(index=False))

print(f"\n=== {NB_NAME} POST-FIX SUMMARY ===")
print("Observed vs assumed supply expansion:")
print(out[["shock", "observed_expansion_ratio", "assumed_capture_at_50pct", "ratio_observed_to_assumed"]].to_string(index=False))
