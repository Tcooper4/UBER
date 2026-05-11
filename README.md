# Pricing the Predictable — Uber GPA Case Study

A framework for capturing forecastable demand shocks in NYC rideshare,
sized against public TLC data, and forward-applied to MetLife World Cup 2026.

**Submitted by:** Thomas Cooper  
**Role:** Senior Quantitative Modeling Analyst, UBS → Uber GPA Lead L3/4 candidate  
**Date:** May 2026

## Headline Findings

| Shock | Recommended Lever | Annual NYC Capture |
|---|---|---|
| NJ pregame (MetLife) | Pricing | $1.86M |
| Heavy rain | Pricing | $2.95M |
| NYC sports | Pricing (tie) | $0.77M |
| Storm warning | Withdraw drivers | N/A |
| Civic events | Operational (reroute) | N/A |
| Political (N=6) | Exploratory | N/A |

**Total pricing capture: ~$5.6M annually.**  
**World Cup 2026 (8 matches): $0.8M verified narrow → $8.1M-$10.8M broader scenario.**

## Methodology Anchors

- Cohen et al. 2016 (NBER w22627) — own-price elasticity
- Hall, Kendrick, Nosko 2015 — driver supply elasticity
- Chen & Sheldon 2016 — driver labor supply
- Lam & Liu 2017 (MIT IDE) — cross-platform substitution
- Wooldridge 2010 — panel collapse defense

## Repository Structure

```
notebooks/         # 9 analysis notebooks (N01-N08 + N07b/c/d)
src/               # Shared utilities (crz_zones.py, etc.)
outputs/
  tables/          # CSV outputs (read into deck + dashboard)
  figures/         # PNG charts (excluded from repo via .gitignore;
                   #  regenerate by running notebooks)
deck/              # Final deck + interactive HTML dashboard
PROJECT_GUIDE.md   # Hypothesis, methodology, defense playbook
```

## Reproducing the Analysis

1. Clone repo
2. Download TLC HVFHS data (Jan 2024 – Aug 2025) from  
   https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
3. Place parquet files in `data/raw/`
4. Run notebooks in order: 00 → 01 → 02 → ... → 08
5. Charts appear in `outputs/figures/`; dollar tables in `outputs/tables/`

## Interactive Dashboard

Open `deck/Cooper_Uber_Strategy_Explorer.html` in any modern browser.  
Two tabs: NYC strategy matrix + World Cup 2026 application.

## Note on Data

Raw data (`data/raw/`, `data/processed/`) excluded from repo due to size
(>10GB). Download from public sources listed in deck citations appendix.
