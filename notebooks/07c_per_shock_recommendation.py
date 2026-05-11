# %%
"""
Notebook 07c — Per-shock recommendation matrix (deck artifact).

Reads N07 strategy comparison and N07b per-shock η (FIX D). For each shock,
applies user framework:
  pricing_net = pricing_annual × (1 − η) × (1 − demand_response_pct)
where demand_response_pct = 1 − retain from 07_pricing_scenarios.csv (N07).

Recommendation bands vs supply (central capture from same shock row).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

NB_NAME = "N07c per-shock recommendation"

PROJECT_ROOT = Path(r"C:\Users\Thomas\OneDrive\Desktop\Uber")
os.chdir(PROJECT_ROOT)
TABLE_DIR = PROJECT_ROOT / "outputs/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

LIT_ETA = 0.4
SUPPLY_CAPTURE_USED = 0.50
RETAIN_DEFAULT = 0.65

strat = pd.read_csv(TABLE_DIR / "07_strategy_comparison.csv")
pricing = strat[strat["spec"] == "pricing"].copy()
supply = strat[strat["spec"] == "supply"].copy()
supply_by = supply.set_index("shock")["annualized_capture"].to_dict()

_eta_path = TABLE_DIR / "07b_per_shock_eta.csv"
eta_df = pd.read_csv(_eta_path) if _eta_path.exists() else pd.DataFrame()
eta_by = (
    eta_df.set_index("shock")["eta_used"].astype(float).to_dict()
    if len(eta_df) and "shock" in eta_df.columns and "eta_used" in eta_df.columns
    else {}
)

_ps = TABLE_DIR / "07_pricing_scenarios.csv"
if _ps.exists():
    ps = pd.read_csv(_ps)
    ph = float(ps.loc[ps["metric"] == "pricing_headline", "value_usd"].iloc[0])
    pa = float(ps.loc[ps["metric"] == "pricing_after_demand_response", "value_usd"].iloc[0])
    retain = pa / ph if ph > 0 else RETAIN_DEFAULT
else:
    retain = RETAIN_DEFAULT
demand_response_pct = 1.0 - retain

rows_out = []
for _, pr in pricing.iterrows():
    shock = str(pr["shock"])
    p_ann = float(pr["annualized_capture"])
    eta_u = float(eta_by.get(shock, LIT_ETA))
    eta_u = float(np.clip(eta_u, 0.0, 0.999))
    pricing_net = p_ann * (1.0 - eta_u) * (1.0 - demand_response_pct)
    s_cen = float(supply_by.get(shock, 0.0))

    if abs(pricing_net) < 1e-6:
        rec = "Operational lever (not pricing or supply)"
        rationale = "Zero net pricing uplift after η and demand-response framing."
    elif pricing_net > s_cen * 1.5:
        rec = "Pricing dominant"
        rationale = "pricing_net exceeds 1.5× supply central capture."
    elif pricing_net > s_cen:
        rec = "Pricing wins, supply viable"
        rationale = "pricing_net exceeds supply central but not by 1.5×."
    else:
        rec = "Supply wins"
        rationale = "supply central capture exceeds pricing_net."

    rows_out.append(
        {
            "shock": shock,
            "pricing_final_M": pricing_net / 1e6,
            "supply_central_M": s_cen / 1e6,
            "recommendation": rec,
            "rationale": rationale,
            "eta_used": eta_u,
            "supply_capture_used": SUPPLY_CAPTURE_USED,
        }
    )

out = pd.DataFrame(rows_out)
out.to_csv(TABLE_DIR / "07c_per_shock_recommendation.csv", index=False)
print(out.to_string(index=False))

print(f"\n=== {NB_NAME} POST-FIX SUMMARY ===")
print("Per-shock recommendations:")
print(out[["shock", "recommendation", "pricing_final_M", "supply_central_M"]].to_string(index=False))
