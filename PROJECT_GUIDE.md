# Uber Global Biz Ops Case — Project Guide (v6)

**Candidate:** Thomas Cooper
**Role:** Uber GPA Lead L3/4 (Global Business Operations)
**Submission deadline:** 24 hours before final-round panel interview (week of May 4, 2026)
**Project location:** `C:\Users\Thomas\OneDrive\Desktop\Uber`
**Last updated:** May 10, 2026 — ALL NOTEBOOKS COMPLETE

---

## 1. The Hypothesis (LOCKED)

> Using publicly available NYC rideshare data (TLC, January 2024 – August 2025) augmented with weather, transit, event, and policy datasets, I will build a quantitative framework for analyzing forecastable demand shocks and their value to a rideshare marketplace. The framework characterizes shock-driven demand and pricing patterns across five categories — predictable events, weather, transit disruptions, the January 2025 congestion pricing implementation, and political events — and uses parallel counterfactuals (pricing optimization vs. supply optimization) to identify the highest-value operational response for each shock type. The output is a sized opportunity for NYC and a generalizable framework demonstrated against the 2026 FIFA World Cup matches at MetLife Stadium.

### Critical framing rules

- **Do NOT claim Uber is doing something wrong.** They have sophisticated ML pricing. Frame as *building a framework*, not *exposing a gap*.
- **Be precise about observation vs. inference.** Surge multiplier is *inferred* from fare-per-mile; wait times are *proxied* from supply density.
- **Recommendation is operational, not extractive.** Supply pre-positioning captures revenue via *more completed trips at base prices*, not by charging captive riders more.

### What this hypothesis commits to

- **Five shock categories:** predictable events, weather, transit substitution, congestion pricing, political events.
- **Two counterfactuals:** pricing optimization, supply optimization.
- **Forward application:** 2026 FIFA World Cup at MetLife Stadium.

---

## 2. Project Setup (DONE)

- Python venv at `venv/` (Python 3.10.0). Packages: jupyter, pandas, duckdb, statsmodels, linearmodels, scikit-learn, matplotlib, seaborn, plotly, geopandas, folium, pyarrow, shapely
- Directory structure: `data/{raw,processed,derived}`, `notebooks/`, `src/`, `outputs/{figures,tables}/`, `ai_conversations/`, `deck/`
- Working with `.py` files using `# %%` cell markers (Cursor interactive mode)
- Working dir: `os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")` at top of every script
- DuckDB memory limit: 4-8GB with temp_directory for spill

---

## 3. Data Prep — COMPLETE

### 3.1 Raw data (downloaded)

| Dataset | Path | Status |
|---|---|---|
| TLC HVFHS Trip Records (20 monthly parquets) | `data/raw/tlc/` | 399.4M rows; Uber 294.7M (74%), Lyft 104.7M (26%) |
| TLC Taxi Zones Shapefile | `data/raw/tlc_zones/taxi_zones.shp` | 263 zones, EPSG:2263 |
| TLC Taxi Zone Lookup CSV | `data/raw/tlc_zones/taxi_zone_lookup.csv` | 265 rows |
| MTA Subway 2024 | `data/raw/mta/mta_subway_2024.csv` | 26.8M rows raw |
| MTA Subway 2025 | `data/raw/mta/mta_subway_2025.csv` | 37.2M raw (extends to April 2026, filtered at query time) |
| NOAA Hourly Weather | `data/raw/noaa/NOAALocalClimatologicalData.csv` | 117K rows; 3 stations |
| NOAA Storm Events | `data/raw/noaa/storm_data_search_results.csv` | 94 NY events |
| NYC SAPO Permits Historical | `data/raw/events/NYC_Permitted_Event_Information_-_Historical_20260509.csv` | 8.5M raw rows (60K distinct events after dedup) |
| Major Events Table | `data/raw/events/major_events.csv` | **761 events: 741 sports + 6 political + 14 civic** |

### 3.2 Processed datasets

| Output | Notes |
|---|---|
| `data/processed/tlc_zone_hour.parquet` | 7.2M rows. Includes `total_adjusted_fare` and `total_miles` for **ratio-of-sums** fare-per-mile. Surge proxy strips `congestion_surcharge + cbd_congestion_fee` BEFORE computation. |
| `data/processed/mta_subway_clean.parquet` | 6.05M rows. (timestamp, station, borough). 462 stations, 2.05B total riders. |
| `data/processed/mta_station_coords.parquet` | 424 stations with lat/lon parsed from Georeference WKT. |
| `data/processed/noaa_weather_clean.parquet` | 43,277 hourly observations. FM-15 + FM-16 deduped by (station, hour) keeping latest. |
| `data/processed/noaa_storms_nyc.parquet` | 91 NYC events, 9 event types. |
| `data/processed/sapo_permits_clean.parquet` | 60,294 events (deduped from 8.5M raw). 3 attendance methodologies. Borough-level only. |
| `data/processed/holidays.parquet` | 23 dates: USFederalHolidayCalendar + NYC additions. |
| `data/processed/zone_to_weather_station.parquet` | 263 zones mapped to nearest of 3 stations. |
| `src/crz_zones.py` | 36 zones IN CRZ. Geometric derivation + manual classification of 11 boundary zones. |

### 3.3 Master analysis table

`data/processed/master_zone_hour.parquet` — **7.2M rows × 52 columns**. Primary key: `(pickup_zone, pickup_hour, platform)`. Both Uber and Lyft kept as separate rows for placebo capability.

**Verified at audit:**
- 261 unique zones (2 zero-trip zones excluded by filter)
- 14,616 unique hours (24 missing = DST + zero-trip zone-hours, 0.16% gap, immaterial)
- Zero duplicate primary keys
- 36 CRZ zones (matches definition)
- 14% of rows in CRZ; 39% post-CP
- Weather missing 1.93% of rows (concentrated Aug 26-31 2025 publication lag)

### 3.4 Methodology decisions made during prep

1. **Ratio-of-sums fare-per-mile** (not mean-of-ratios). Re-aggregated TLC with `SUM(adjusted_fare)` and `SUM(trip_miles)` columns explicitly.
2. **Hour-level event windows** (not day-level). Day-level diluted coefficients ~70%. Two windows: symmetric ±3hr from event START (primary for short events) and asymmetric -2hr to +4hr around event END (primary for long events). Day-level retained as `has_major_event_dayflag` for EDA comparison.
3. **NJ venue events with departure-zone flags.** 147 NJ events (Giants, Jets, Devils, Islanders) flagged on Manhattan transit-hub departure zones with **pre-game-only** windows since post-game returns originate in NJ and aren't visible in TLC.
   - **MetLife:** zones 186, 230, 161, 100, 246, 48
   - **Prudential Center:** zones 261, 87, 88, 231, 125, 113, 114, 158, 249
   - **UBS Arena (Belmont):** zones 186, 230, 161
   - **Red Bull Arena:** zones 261, 87, 88, 231, 125, 113, 114
4. **Political events methodology** (added May 9 2026):
   - 6 events: 4 UN General Assembly days (Sept 24-27, 2024, zone 233 = UN/Turtle Bay South), Election Day Nov 5 2024 (zone 161 anchor), Inauguration Jan 20 2025 (zone 161 anchor)
   - Custom event windows: UN GA = 9am-7pm (10hr), Election = 5pm-9pm (4hr), Inauguration = 12pm-4pm (4hr)
   - Verified via UN.org and US State Department; attendance set to 0 (these aren't venue-attendance events)
   - **Asymmetric flag preferred** for political events due to long duration (sym ±3hr too narrow for 10hr UN GA)
   - Small sample (N=6); document in deck as exploratory rather than headline finding
5. **CRZ zone definition:** Geometric derivation + manual classification of 11 boundary zones.
6. **NOAA weather:** Both FM-15 (hourly METAR) and FM-16 (special obs) kept, deduped by (station, hour) keeping latest.
7. **SAPO methodology:** Borough-level only (no public geocodes). 3 attendance methodologies — BINARY (primary), TYPED (robustness), LOCATION-WEIGHTED (robustness). Deduped by Event ID (8.5M raw rows → 60K events).
8. **Treatment date:** January 5, 2025 for congestion pricing (`is_post_cp` flag).
9. **Civic events methodology** (added May 10 2026):
   - **14 events:** TCS NYC Marathon, TD Five Boro Bike Tour ×2, Macy's Thanksgiving Parade, NYC Pride ×2, West Indian Day, Puerto Rican Day ×2, Halloween Parade, NYE Ball Drop, Veterans Day, St. Patrick's Day ×2
   - **`event_type` values:** `parade`, `parade_evening`, `special_event`, `special_event_evening` — each maps to a specific time-of-day window
   - **Bug fixes in `get_event_window`:** parade default 9am–5pm; `parade_evening` 7–11pm; NYE 6pm–1am (crossing midnight); bike tour 7am–2pm; marathon 8am–5pm; explicit league / `event_type` assertion to catch fallthrough
10. **Memory-safe panel regression methodology** (added May 10 2026):
   - Full hourly panel (3.65M rows × 14,877 hourly periods) generates a within-transformation matrix exceeding RAM during two-way FE
   - **Solution:** collapse to **(zone, day)** panel before PanelOLS: `sum(trips, fare, miles)`; ratio-of-sums fare-per-mile on collapsed totals; `max()` over hours for binary event flags
   - **Hour-level granularity preserved** in per-event lift analysis (N02 Cells 4–6, 10b, 12) via collapse-first baseline matching
   - Daily aggregation is standard panel-DiD spec in published transportation / labor economics (Hall, Kendrick & Nosko 2015; Cramer & Krueger 2016)
11. **Weather panel specification** (added May 10 2026):
   - Weather is citywide → **time FE absorbs all weather variables**
   - Replaced `time_effects=True` with **day-of-week + month dummies**
   - Identifies weather effects from **across-day variation within zones**, controlling for seasonal / weekly patterns
   - **Continuous weather:** temp = daily mean, precip = daily total, wind = daily max
   - Reference: standard spec for citywide shocks in transportation literature

### 3.5 Final event flag counts (post civic events)

| Flag | Count | Notes |
|---|---|---|
| `is_nyc_event_sym` | 8,171 | ±3hr around event start, NYC zones |
| `is_nyc_event_asym` | 12,332 | −2/+4hr around event end, NYC zones |
| `is_nj_event_pregame_sym` | 7,104 | NJ pre-game, Manhattan departure |
| `is_nj_event_pregame_asym` | 8,880 | Extended pre-game window |
| `is_event_combined_sym` | 15,079 | Union NYC+NJ sym |
| `is_event_combined_asym` | 21,064 | Union NYC+NJ asym |
| `has_major_event_dayflag` | 26,491 | Day-level (3.2× dilution vs hour) |

### 3.6 Known data limitations to disclose in deck appendix

| Limitation | Impact |
|---|---|
| NOAA weather missing 1.93% of hours | Weather elasticity regression runs on ~7.06M of 7.20M rows. Statsmodels handles NaN. |
| NJ trip-level rideshare data not publicly available | NJ events captured only via Manhattan departure-zone signal (pre-game only). Post-game return trips not observable. |
| SAPO at borough level, not zone level | Borough-level control variable rather than primary signal. |
| Surge multiplier inferred, not directly observed | Documented as "ex-ante surge proxy via fare-per-mile after stripping congestion fees." |
| Wait times not directly observable | Proxied via supply density and `total_driver_pay`. |
| Political events small sample (N=6) | Treated as exploratory; report point estimates with CI. |
| UN GA effect zone-localized to 233 | Citywide ripple captured indirectly via SAPO controls. |

### 3.7 Deferred to v2 (intentionally)

- **Buses** — MTA Bus Hourly Ridership for World Cup application stage
- **Demographic data (Census ACS)** — for heterogeneity analysis
- **Concerts at MSG/Barclays** — Wikipedia events with capacity proxy if time permits
- **Comprehensive geocoding of SAPO** — marginal value for v1
- **Citywide political event flag** — current zone-level approach sufficient for hypothesis (zone-level analysis); citywide flag would only help broader claims not currently made

---

## 4. Notebook Status (ALL COMPLETE)

**Outputs:** `outputs/figures/*.png` (executive-deck styling via `outputs/visualization_design_system.py`), `outputs/tables/*.csv`, `outputs/methodology_note_panel_aggregation.txt`

- **Notebook 01:** EDA — COMPLETE (`outputs/figures/01_*.png`, `outputs/tables/01_summary.csv`)
- **Notebook 02:** Event Study — COMPLETE (per-event hour-level + daily panel regressions; **14 civic events** surface unexpected **operational-blocker** pattern: parades show **−30% to −97%** lifts due to street closures, distinct from pricing-relevant shocks)
- **Notebook 03:** Congestion Pricing DiD — COMPLETE (**headline β = −0.119** log_trips; **PRE-TRENDS PLACEBO VIOLATED:** all 4 placebo dates show significant negative β **−0.06 to −0.13**; **true causal effect ~0–4%** after adjusting for pre-existing trend; donut window robust)
- **Notebook 04:** Weather Elasticity — COMPLETE (asymmetric rain bins monotonically positive: light **+0.5%**, mod **+2.2%**, heavy **+4.4%**; storm warnings opposite signal **−1.9%**; quadratic temp minimum at **51°F**)
- **Notebook 05:** Cross-Modal Substitution — COMPLETE (**β = +0.23** within-day elasticity; modes **co-move**, not substitute; heavy rain **DECOUPLES** — elasticity drops to **0.13**)
- **Notebook 06:** Surge Gap Sizing — COMPLETE (**$19.5M** annual headline across 4 shocks at central elasticity **ε = −0.7**; capturable estimate **~$13M** after elasticity discount; **storm active** largest absolute (**$8.7M**) but **rain** largest per-hour opportunity)
- **Notebook 07:** Strategy Comparison — COMPLETE (**pricing $19.5M** vs **supply $6.8M**; **2.9×** ratio favors pricing in headline dollars but **supply recommended** for reputational / regulatory / competitive reasons)
- **Notebook 08:** World Cup at MetLife — COMPLETE (**8 matches** incl. final **7/19/2026**; tournament total **$5M–$25M** range across +500K / +1M / +2M tourist scenarios; methodology applies NJ-venue framework)

---

## 5. Tech Stack

**Core:** Python (primary), DuckDB, pandas, statsmodels, linearmodels (PanelOLS for DiD), scikit-learn, matplotlib + seaborn, plotly (1-2 showcase), geopandas + folium (light)

**Tools:** Cursor (daily editing, interactive `# %%` execution); Claude.ai (methodology, slide titles, "Why Uber" pitch); this conversation series (AI submission artifact)

**Skip:** Polars, LSTM/NN, EconML, advanced spatial econometrics

### Cursor project rules (`.cursor/rules`)

```
- NYC TLC HVFHS analysis for Uber Global Biz Ops case
- Window: Jan 2024 – Aug 2025
- Stack: Python, DuckDB, pandas, statsmodels, linearmodels, matplotlib/seaborn
- Treatment date for congestion pricing: January 5, 2025
- DuckDB for raw TLC; pandas for analysis after aggregation
- Cluster SEs by zone in panel regressions
- linearmodels.PanelOLS for fixed-effects
- union_by_name=True when reading multi-month TLC parquets
- Set memory_limit='4-8GB' on DuckDB connections; use temp_directory for spill
- Process TLC files one month at a time to avoid OOM
- Use master_zone_hour.parquet as primary analysis table
- Use ratio-of-sums fare-per-mile (total_adjusted_fare / total_miles)
- Always import CRZ_ZONE_IDS from src/crz_zones.py
- For substitution analysis, filter to zones with subway_riders_zone > 0
- For political events analysis, prefer is_event_asym over is_event_sym (long-duration events)
- Use color palette: Uber=#000000, Lyft=#FF00BF, CRZ=#06C167, non-CRZ=#CCCCCC
- Never use errors="replace" with pd.read_csv (not a valid kwarg)
- Never use color= with statsmodels plot_acf/plot_pacf (not a valid kwarg)
```

---

## 6. Analyses to Perform — COMPLETE

All notebooks (01–08) are finished. See **Section 4** for status, headline numbers, and file outputs. Subsections below retain the **original analysis specifications** for reproducibility and deck methodology slides.

### 6.1 ✅ Notebook 01: EDA — COMPLETE

### 6.2 ✅ Notebook 02: Event Study [Core] — COMPLETE

For each event category (NYC sports, NJ sports pre-game, political, marathon, holidays):
- Use `is_nyc_event_sym` for primary, `is_nyc_event_asym` for robustness on short events
- **Use `is_nyc_event_asym` as PRIMARY for political events** (long duration)
- Compare day-level (legacy) vs hour-level results to demonstrate methodology improvement
- Paired t-test event windows vs. matched baselines (same hour-of-week, no event ±2 weeks)
- Political events sub-analysis: exploratory due to small sample (N=6)

### 6.3 ✅ Notebook 03: Congestion Pricing DiD [Core — headline] — COMPLETE

```
log(trip_count) = β·is_post_cp·is_in_crz + α_zone + α_time + ε
log(fare_per_mile_ratio_of_sums) = β·is_post_cp·is_in_crz + α_zone + α_time + ε
```

- `linearmodels.panel.PanelOLS`, cluster SEs at zone level
- Pre-trends test (placebo dates before Jan 5 — should be null)
- **Event-study plot** (β_t for each week pre/post Jan 5) — *gold-standard visualization*
- Heterogeneity by time-of-day, zone type
- **Lyft placebo** — also run first-differences version due to Lyft non-stationarity

### 6.4 ✅ Notebook 04: Weather Elasticity [Core] — COMPLETE

```
log(trip_count) = β1·log(precip_in+ε) + β2·temp_f + β3·wind_mph + β4·temp_f² + α_zone + α_hour_of_week + ε
```

- Asymmetric response (light vs. heavy rain), thunderstorm dummy via `is_storm_active`
- Distinguish moderate weather (riders flee transit) from extreme (everyone stays home)

### 6.5 ✅ Notebook 05: Cross-Modal Substitution [Strong differentiator] — COMPLETE

- **Filter to zones with subway access** (`WHERE subway_riders_zone > 0`) — necessary for valid substitution analysis
- TLC × subway cross-correlation by hour-of-week
- Time-series during major weather events (use `is_storm_active`)
- Substitution elasticity: `log(uber_volume) = β·log(subway_riders_zone) + controls`

### 6.6 ✅ Notebook 06: Surge Gap Sizing [Core — money slide] — COMPLETE

- Per shock type: observed avg fare-per-mile, demand-implied "optimal," gap × volume = foregone revenue
- Sensitivity: low/medium/high elasticity scenarios; tornado chart
- Adjustable population parameter for World Cup tourist influx (+500K/+1M/+2M)

### 6.7 ✅ Notebook 07: Strategy Comparison [Core — analytical centerpiece] — COMPLETE

- For each shock type, parallel counterfactuals (pricing vs supply)
- Supply tightness measures: cancellation rate, surge magnitude, driver-to-trip ratios from `total_driver_pay`
- Expected pattern: pricing wins for brief/severe/unforecastable; supply wins for forecastable/sustained

### 6.8 ✅ Notebook 08: World Cup at MetLife — COMPLETE

- Applies NJ departure-zone / pre-game methodology to 2026 FIFA schedule; scenario-range tournament totals by tourist inflow

### 6.9 Cut order if behind schedule *(historical — project complete)*

1. Bus substitution (already deferred)
2. Concerts at MSG/Barclays (already deferred)
3. VAR model (use simple cross-correlation)
4. SARIMAX forecasting
5. Political events sub-analysis (due to small N)
6. Lyft placebo (nice-to-have)

**DO NOT cut:** DiD, event-study plot, weather elasticity, sizing slide, polished EDA, falsification finding, strategy comparison, reframing narrative beat.

---

## 7. Slide Deck (10 + appendix)

1. **Title:** "Pricing the Predictable: A Framework for Capturing Forecastable Demand Shocks in Rideshare"
2. **Why this matters** — forecastable shocks visible in public data, sized opportunity
3. **Hypothesis & approach**
4. **Data + EDA** — 9 datasets, 20 months, 5 shock categories observed
5. **Finding 1: Event-driven response** — event-study plot, NYC venue zoom (e.g., MSG)
6. **Finding 2: Weather elasticity** — precip vs. demand scatter, elasticity table
7. **Finding 3: Congestion pricing as natural experiment** — event-study DiD plot. Possible reframe: "Volume crashed, fare held"
8. **Finding 4: Cross-modal substitution amplifies** — TLC × MTA cross-correlation
9. **Strategy Comparison & Recommendation** — comparison grid, tornado chart, sized $X annually
10. **World Cup application + risks/next steps** — leveraging MetLife pre-game departure-zone analysis

**Appendix:** methodology details, data quality, robustness checks (Lyft first-differences), NYC vs NJ event methodology, political events exploratory, "challenge questions" pre-empts, AI conversation excerpt.

---

## 8. Headline Findings (from completed notebooks)

1. **Event study (N02):** Hour-level NYC + NJ sports lift with collapse-first baseline matching; NJ **departure-zone** pre-game effects dominate full-panel estimates when scoped. **Civic / parade events:** **−30% to −97%** “lift” — **operational blockers** (street closures), not demand shortages → distinct playbook from pricing-relevant shocks.
2. **Congestion pricing DiD (N03):** Headline **β = −0.119** (log trips) → naive gap **~11%**. **Pre-trends placebo FAILED:** all **four** pre–Jan 5 placebo dates show **significant negative β (−0.06 to −0.13)** → CRZ–non-CRZ trends already diverging pre-CP. **Honest causal range ~0–4%** after trend adjustment; **donut window** robustness reported.
3. **Weather elasticity (N04):** Asymmetric rain **monotonic**: light **+0.5%**, moderate **+2.2%**, heavy **+4.4%** (implied vs baseline); **storm warning** flag **−1.9%** (opposite sign — people stay home). Quadratic temp **minimum ~51°F** “comfortable walking” band.
4. **Cross-modal substitution (N05):** Panel elasticity **β = +0.23** — TLC and subway **co-move** at hourly granularity (**never** negative substitutability). **Heavy rain decouples** modes (elasticity **~0.13** vs dry **~0.23**).
5. **Surge gap sizing (N06):** **~$19.5M** annual headline across shock types at central **ε = −0.7**; **~$13M** capturable after elasticity discount (**~60–70%**). **Storm-active** largest **absolute** annual dollar (**~$8.7M**); **rain** shocks largest **per-hour** revenue gap.
6. **Strategy comparison (N07):** **Pricing $19.5M** vs **supply $6.8M** annualized (**~2.9×** headline ratio favors pricing). **Recommendation: supply** — trade-off framing (extractive vs operational / reputational / regulatory / competitive risk), **not** quantitative dominance.
7. **World Cup at MetLife (N08):** **8 matches** including **final 2026-07-19**; tournament scenario range **$5M–$25M** (+500K / +1M / +2M tourists) using NJ-venue methodology + scaling assumptions.
8. **EDA / descriptive (N01):** Volume **drops at Jan 5 2025** and stays depressed; **fare-per-mile falls post-CP** after stripping fees (**volume crashed, fare soft**); Uber **stationary**, Lyft **non-stationary** (panel FE / FD mitigate).

### REQUIRED narrative elements

- **Iterative reframing:** Started NYC venue-only; extended to **Manhattan departure zones** for NJ / MetLife. CP: expected price pain → found **volume crash + softer surge (FPM)**.
- **Pre-trends honesty:** Placebo failure is a **result**, not embarrassment — defines **what DiD can and cannot identify** at zone-day granularity with parallel macro trends.
- **Falsification:** Weather **hours** dwarf single-event hours in aggregate exposure; **civic negatives** falsify “events always lift trips.”

### Additional headline bullets (deck)

9. **Pre-trends violation in DiD** — naive headline overstates causal effect; honest reporting of placebo failure is academic standard. Discuss in deck as **what the test revealed about the limits of what DiD can tell us at this granularity.**
10. **Civic events reveal operational-blocker shocks** — distinct from pricing-relevant shocks. Parades close streets, killing supply access; framework distinguishes the two for operational vs pricing response.

---

## 9. Day-by-Day Plan

**Status (May 10 2026):** Notebooks **01–08 complete**; figures upgraded for executive deck; focus shifted to **deck build + panel prep**.

**Day 1-3 (DONE):** Setup, data download, all cleaning, master analysis table, hour-level event flags (NYC + NJ + political + civic)

**Day 4–7 (DONE / superseded):** Full analysis pipeline executed — event study, DiD, weather, substitution, sizing, strategy, World Cup application.

**Remaining:** Skeleton deck, polish charts, appendix, AI conversation curation, timed practice runs (see Section 14).

---

## 10. Data Traps — Status

| Trap | Status |
|---|---|
| 1. CBD congestion fee inflates fare-per-mile | Handled (subtracted before surge proxy) |
| 2. Schema mismatch pre/post Jan 2025 | Handled (per-file `has_cbd` detection) |
| 3. Upfront pricing distorts fare interpretation | Acknowledge in deck |
| 4. Tip data unreliable | Excluded from all analyses |
| 5. Driver pay vs. fare ≠ Uber margin | Using base_passenger_fare for revenue |
| 6. Outer-borough zones sparse | Will aggregate or drop in analysis |
| 7. Holidays distort everything | Holiday flag in master table |
| 8. Snowstorms kill BOTH demand and supply | Will separate from rain regression in Nb 04 |
| 9. CRZ definitions don't align cleanly | Geometric derivation + manual boundary classification |
| 10. Lyft market share trends | Use share not absolute volume in cross-platform comparisons |
| 11. SAPO has applied permits | Filtered to actually-occurring (60K from 8.5M) |
| 12. NOAA hourly gaps | FM-15 + FM-16 dedup mitigates; 1.93% missing handled by NaN-drop |
| 13. Time zone handling | TLC and MTA both NYC local; NOAA standardized on ingest |
| 14. MTA 2025 file extends to April 2026 | Filtered at query time |
| 15. Subway 8-9 rows per (timestamp, station) for fare classes | Aggregated via SUM |
| 16. Subway ridership stored as string with commas | REPLACE+CAST handled |
| 17. SAPO has 8.5M raw rows but only 60K events | Deduped via Event ID + ANY_VALUE aggregation |
| 18. NJ trip-level data not publicly available | NJ events use Manhattan departure-zone pre-game flag; documented |
| 19. Day-level event flagging dilutes coefficients ~70% | Using hour-level windows (sym + asym) |
| 20. Lyft daily series non-stationary (ADF p=0.18) | Mitigated by panel fixed effects; first-differences placebo as robustness |
| 21. Pre-treatment anticipation spike Dec 2024 | Document in DiD; consider donut-window robustness check |
| 22. Fare-per-mile dropped post-CP (counter-intuitive) | Reframe narrative as "volume crashed, fare held" rather than "price rose" |
| 23. Sym window too narrow for long-duration events (UN GA) | Use asym window as primary for political events; document methodology |
| 24. Political event small sample (N=6) | Treat as exploratory in deck, not headline |
| 25. Two-way FE on hourly panel exceeds RAM | Collapse to **zone-day** panel before PanelOLS; hour-level preserved in per-event lift (N02) via collapse-first baseline |
| 26. Weather absorbed by day FE | Use **DoW + month** dummies instead of calendar-day FE so citywide weather is identified off within-zone across-day variation |
| 27. Pre-trends placebo violated for CP DiD | Report honestly with caveat (~**0–4%** causal vs **~11%** naive); donut window + placebo CSV as evidence |

---

## 11. Anticipated Panel Attacks & Defenses

| Attack | Defense |
|---|---|
| "How is fare-per-mile a valid surge proxy with toll passthrough?" | Subtracted `congestion_surcharge + cbd_congestion_fee` before computing. Show methodology slide. |
| "Why mean-of-ratios? Shouldn't you use ratio-of-sums?" | I used ratio-of-sums (total_adjusted_fare / total_miles). |
| "What's your parallel trends test result?" | **Placebo dates fail** — pre-Jan 5 CRZ–non-CRZ gaps already negative. Event-study + placebo CSV; honest range **~0–4%** causal vs **~11%** naive. See new rows below for panel Q&A. |
| "How do you know elasticity isn't biased by simultaneity?" | Weather as exogenous shock; sensitivity analysis. |
| "Why this window?" | Captures CP shock with 12mo pre + 8mo post; all four seasons; full event calendar. |
| "What about Lyft as counterfactual?" | Lyft placebo regression in appendix. Note Lyft non-stationarity from EDA — addressed via panel fixed effects + first-differences robustness. |
| "Day-level vs hour-level event flagging?" | Hour-level. Day-level dilutes coefficients ~70%. |
| "Why use asym for political events but sym for sports?" | Political events have long duration (UN GA = 10 hours); ±3hr sym window misses bulk of event. Asym (-2hr/+4hr around event end) captures full window. Documented methodology. |
| "What about NJ-venue events?" | Built separate departure-zone flag system for Manhattan pre-game demand. Pre-game-only. Same methodology applies to World Cup. |
| "How does this generalize beyond NYC?" | Framework derives elasticity by shock type; World Cup at MetLife is direct application of NJ-venue methodology. |
| "We have internal data showing X — your analysis misses it." | "Your team's internal data adds dimensions I can't see — driver-level multi-app, exact surge multipliers, dispatch logic. My value is the publicly-defensible methodology and framework." |
| "We already use ML pricing." | "Of course — your team's research on demand forecasting is well-known. My case isn't claiming Uber is unsophisticated; it's identifying where the residual gap is largest." |
| "Why supply pre-positioning over pricing?" | Slide 9 shows the comparison. Supply wins for forecastable shocks because it captures volume at base fare without regulatory/PR exposure. |
| "Your sizing assumptions feel aggressive." | Sensitivity analysis shows ranges. Lead conservative. |
| "Why borough-level for SAPO instead of zone?" | "SAPO doesn't include geocoded coordinates publicly. Major events table provides zone-level precision; SAPO is a borough-level robustness check." |
| "Why this CRZ zone list?" | "Derived geometrically from the taxi zones shapefile by computing centroid latitude, then verified boundary zones against the official MTA congestion pricing map." |
| "Why FM-15 and FM-16?" | "Including FM-16 special observations preserves severe-weather hours that FM-15 alone misses." |
| "What attendance methodology for SAPO?" | "Three methodologies — binary as primary, typed and location-weighted as sensitivity. Headline findings robust across all three." |
| "Why pre-game-only window for NJ events?" | "NJ-side trip data isn't publicly available. Post-game return trips originating in NJ aren't observable in TLC. Pre-game window captures the observable signal cleanly." |
| "Your fare-per-mile dropped post-CP — that's surprising. Are you sure?" | "Yes, after stripping the congestion fee passthrough, the underlying surge component softened. This reflects demand falling faster than supply could redirect, so the surge premium compressed. The headline finding is the volume reduction (clean DiD signal)." |
| "Political events sample is tiny." | "N=6 is exploratory; report point estimates with appropriate uncertainty. Framework methodology validated on larger samples (sports events). Future work: integrate with internal data for tighter estimates." |
| "Why zone 233 for UN GA, not citywide?" | "Zone-level analysis is core to the hypothesis; UN GA effect concentrates at UN HQ (zone 233 = UN/Turtle Bay South, verified). Broader Manhattan ripple captured indirectly via SAPO permits." |
| "Why daily collapse instead of hourly?" | "Hourly two-way FE on **3.65M obs × 14,877** timestamps exceeds memory. **Daily collapse** is standard published spec (Hall et al. 2015; Cramer & Krueger 2016). **Hour-level** granularity preserved in **per-event lift** via collapse-first baseline." |
| "Your placebo failed — your DiD is wrong." | "**Yes — for parallel trends.** The placebo test shows CRZ–non-CRZ were already diverging pre-CP. Reporting honestly: naive DiD says **~−11%**, true causal effect after adjusting for pre-trend is closer to **0–4%**. The placebo is doing its job — showing **limits of identification**. Donut window robustness shows persistence after excluding anticipation window." |
| "Civic events show NEGATIVE lift?" | "**Yes.** Parades close streets; pickups become physically impossible. **Operational-blocker shock**, not a pricing shock. Different operational responses (reroute, communications) vs surge." |
| "Pricing captures more than supply per your numbers — why recommend supply?" | "**Pricing is extractive** (raises prices on captive riders); **supply is operational** (serves more riders at base fare). Given Uber's **post-strike NYC regulatory posture** and **competitive risk if Lyft holds prices**, **supply dominates strategically** despite lower headline dollars. **Trade-off framing**, not quantitative dominance." |

---

## 12. AI Submission Strategy

This conversation series IS the AI submission. Curate ~20-30 messages showing:
1. **Methodology debate** — DiD specification, ratio-of-sums vs. mean-of-ratios, FM-15 vs. FM-15+FM-16, day-level vs hour-level, NJ venue handling, political event windowing
2. **Bug catch** — moments of pushing back on AI output (NHL date format, ridership-as-string, OneDrive paths, geocoding shortcut, SAPO 8.5M dedup, day-level dilution, ACF color, read_csv errors, fabricated political event values)
3. **Reframing** — geocoding question, attendance methodology, NJ venue inclusion, "fare dropped post-CP" reframe, political event scope reconsideration
4. **Iteration** — events table built and rebuilt, SAPO file replaced, NJ flags added, political events verified via web search before adding

Save to `ai_conversations/methodology_debate.md`.

---

## 13. "Why Uber" Pitch (60-90s)

1. **Why leaving UBS:** "I've spent years validating models and want to move toward building and shipping. Quant validation taught me to break models — now I want to build the ones I'd defend."
2. **Why Uber:** "Marketplace dynamics are the most intellectually compelling problem in business — supply, demand, pricing, geography all moving simultaneously. Uber operates at a scale where small operational decisions move millions daily. That's the leverage I want."
3. **Why this team:** "David's framing in our first conversation — pragmatic, contrarian, comfortable with ambiguity — maps to how I think when I'm at my best. I've been using public data analysis as a lever to influence decisions in my UBS work; this role makes that the job."

---

## 14. Final Pre-Interview Checklist

**Day 6 PM:**
- [ ] All slides have action titles (not topic titles)
- [ ] **Pre-trends placebo finding integrated into deck honestly**
- [ ] **Active titles on all charts** (not topic titles)
- [ ] **Civic events operational-blocker pattern called out separately** from pricing-relevant shocks
- [ ] **Strategy comparison framed as trade-off**, not quantitative win
- [ ] Every claim sourced
- [ ] Sizing has explicit ranges + assumptions
- [ ] Reframing moment explicit somewhere in deck
- [ ] At least one "I expected X, found Y" finding
- [ ] AI conversation curated and clean
- [ ] Code organized in `src/` with comments
- [ ] Notebooks runnable from clean restart
- [ ] 30-min practice run feels comfortable

**Interview Day:**
- [ ] "Why Uber" pitch ≤90 seconds
- [ ] Three smart questions for David at end
- [ ] Appendix slides accessible during Q&A
- [ ] Challenge defenses memorized as points (not script)
- [ ] Comfortable saying "I don't know" when honest

---

## 15. THE RULE

**Trust the data over the plan.** EDA surfaced fare-down-not-up; full pipeline surfaced **placebo failure**, **civic negatives**, and **pricing-vs-supply trade-off** — led with honesty in v6 findings.

---

## 16. Context for the Next Chat

When opening a fresh conversation for **deck build, extensions, or interview prep:**

**Paste this as setup:**

> I'm continuing the Uber GPA Lead L3/4 case study. Project at `C:\Users\Thomas\OneDrive\Desktop\Uber`. **All 8 notebooks complete** with **156K-row daily panel** + per-event hourly lift. **Pre-trends placebo failed** for CP DiD (**true causal effect ~0–4%**, not **~11%**). **Civic events** reveal **operational-blocker** shocks (parades **−30% to −97%**). **Strategy comparison:** pricing **$19.5M** vs supply **$6.8M**, but **supply recommended** on trade-off framing.
>
> **Outputs:** `outputs/figures/*.png` (upgraded for executive deck per `visualization_design_system.py`), `outputs/tables/*.csv`, `outputs/methodology_note_panel_aggregation.txt`.
>
> **Ready to:** build deck slides \| extend analysis \| answer Q&A prep.

**Attach:** This PROJECT_GUIDE.md file (**v6**).
