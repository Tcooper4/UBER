# %%
# Cell 1 — Append 14 verified civic + special events to major_events.csv
"""
Adds:
  - 1 marathon (TCS NYC Marathon 2024-11-03)
  - 2 bike tours (TD Five Boro 2024-05-05, 2025-05-04)
  - 11 parades (Thanksgiving, Pride x2, West Indian, Puerto Rican x2, Halloween,
                NYE Ball Drop, Veterans Day, St. Patrick's x2)

Verified dates via web search; lead_time_days=365 (annual recurring); team="".
event_type values:
  - "special_event" → marathon, bike tour, NYE
  - "parade"        → all parades
"""
import os

os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")

import pandas as pd

new_events = pd.DataFrame(
    [
        # Marathon
        {
            "date": "2024-11-03",
            "venue": "Central Park (finish)",
            "zone_id": 43,
            "event_name": "TCS NYC Marathon 2024",
            "expected_attendance": 50000,
            "event_type": "special_event",
            "lead_time_days": 365,
            "team": "",
            "league": "MARATHON",
        },
        # Bike tours
        {
            "date": "2024-05-05",
            "venue": "Financial District (start)",
            "zone_id": 87,
            "event_name": "TD Five Boro Bike Tour 2024",
            "expected_attendance": 32000,
            "event_type": "special_event",
            "lead_time_days": 365,
            "team": "",
            "league": "BIKE_TOUR",
        },
        {
            "date": "2025-05-04",
            "venue": "Financial District (start)",
            "zone_id": 87,
            "event_name": "TD Five Boro Bike Tour 2025",
            "expected_attendance": 32000,
            "event_type": "special_event",
            "lead_time_days": 365,
            "team": "",
            "league": "BIKE_TOUR",
        },
        # Macy's Thanksgiving Day Parade 2024
        {
            "date": "2024-11-28",
            "venue": "Midtown Manhattan",
            "zone_id": 161,
            "event_name": "Macy's Thanksgiving Day Parade 2024",
            "expected_attendance": 3500000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        # NYC Pride Marches
        {
            "date": "2024-06-30",
            "venue": "West Village",
            "zone_id": 113,
            "event_name": "NYC Pride March 2024",
            "expected_attendance": 2000000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        {
            "date": "2025-06-29",
            "venue": "West Village",
            "zone_id": 113,
            "event_name": "NYC Pride March 2025",
            "expected_attendance": 2000000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        # West Indian Day Parade (Labor Day)
        {
            "date": "2024-09-02",
            "venue": "Crown Heights, Brooklyn",
            "zone_id": 17,
            "event_name": "West Indian American Day Parade 2024",
            "expected_attendance": 2000000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        # Puerto Rican Day Parades
        {
            "date": "2024-06-09",
            "venue": "5th Ave (Midtown)",
            "zone_id": 161,
            "event_name": "National Puerto Rican Day Parade 2024",
            "expected_attendance": 1000000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        {
            "date": "2025-06-08",
            "venue": "5th Ave (Midtown)",
            "zone_id": 161,
            "event_name": "National Puerto Rican Day Parade 2025",
            "expected_attendance": 1000000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        # Village Halloween Parade
        {
            "date": "2024-10-31",
            "venue": "West Village",
            "zone_id": 113,
            "event_name": "Village Halloween Parade 2024",
            "expected_attendance": 60000,
            "event_type": "parade_evening",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        # NYE Ball Drop
        {
            "date": "2024-12-31",
            "venue": "Times Square",
            "zone_id": 230,
            "event_name": "Times Square NYE Ball Drop 2024-25",
            "expected_attendance": 1000000,
            "event_type": "special_event_evening",
            "lead_time_days": 365,
            "team": "",
            "league": "SPECIAL",
        },
        # Veterans Day Parade
        {
            "date": "2024-11-11",
            "venue": "5th Ave (Midtown)",
            "zone_id": 161,
            "event_name": "Veterans Day Parade 2024",
            "expected_attendance": 25000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        # St. Patrick's Day Parades
        # 2024: Sat March 16 (moved off Sunday March 17 because parade isn't held on Sundays)
        {
            "date": "2024-03-16",
            "venue": "5th Ave (Midtown)",
            "zone_id": 161,
            "event_name": "St. Patrick's Day Parade 2024",
            "expected_attendance": 150000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
        {
            "date": "2025-03-17",
            "venue": "5th Ave (Midtown)",
            "zone_id": 161,
            "event_name": "St. Patrick's Day Parade 2025",
            "expected_attendance": 150000,
            "event_type": "parade",
            "lead_time_days": 365,
            "team": "",
            "league": "PARADE",
        },
    ]
)

existing = pd.read_csv("data/raw/events/major_events.csv", encoding="utf-8")
print(f"Before: {len(existing)} events")

# Reorder to match existing schema
new_events = new_events[existing.columns.tolist()]

dedupe_key = ["date", "zone_id", "event_name"]
combined = pd.concat([existing, new_events], ignore_index=True)
n_before = len(combined)
combined = combined.drop_duplicates(subset=dedupe_key, keep="first")
duplicates_removed = n_before - len(combined)
combined.to_csv("data/raw/events/major_events.csv", index=False)
print(f"After: {len(combined)} events (removed {duplicates_removed} duplicate key rows if re-run)")
print("\nLeague breakdown:")
print(combined["league"].value_counts())
print("\nEvent type breakdown:")
print(combined["event_type"].value_counts())


# %%
# Cell 2 — Rebuild NYC hour-level event flags with civic event windowing
# Bug fixes from previous version:
#   - parade default now 9am-5pm (8hr) instead of 9am-3pm to cover full disruption
#   - parade_evening (Halloween) handled at 7pm-11pm
#   - special_event default now 8am-5pm (covers marathon, daytime specials)
#   - special_event_evening (NYE) at 6pm-1am (rolls past midnight cleanly)
#   - bike tour gets 7:30am-2pm (specific case via event_name match)
import shutil
import time

import duckdb

print("=" * 70)
print("REBUILDING NYC HOUR-LEVEL EVENT FLAGS")
print("=" * 70)
start = time.perf_counter()

print("\n[1/4] Computing event start/end times...")

events = pd.read_csv("data/raw/events/major_events.csv", encoding="utf-8")
events["date"] = pd.to_datetime(events["date"])
events["dow"] = events["date"].dt.dayofweek


def get_event_window(row):
    """Returns start_hour, end_hour (end = start + duration), and branch_id for validation."""
    league = str(row.get("league", "")).upper()
    event_type = str(row.get("event_type", "")).lower()
    event_name = str(row.get("event_name", "")).lower()
    is_weekend = row["dow"] in [5, 6]

    branch_id = "fallback_default"
    start = 19
    duration = 3

    # Sports
    if "MLB" in league:
        branch_id = "MLB"
        start = 13 if is_weekend else 19
        duration = 4
    elif "NBA" in league:
        branch_id = "NBA"
        start = 19
        duration = 3
    elif "NHL" in league:
        branch_id = "NHL"
        start = 19
        duration = 3
    elif "NFL" in league:
        branch_id = "NFL"
        start = 13
        duration = 4
    elif "MLS" in league:
        branch_id = "MLS"
        start = 19
        duration = 3

    # Civic / special events — checked BEFORE generic event_type fallthrough
    elif "bike tour" in event_name:
        branch_id = "civic_bike_tour_name"
        start = 7
        duration = 7
    elif "marathon" in event_name:
        branch_id = "civic_marathon_name"
        start = 8
        duration = 9
    elif "ball drop" in event_name or "nye" in event_name:
        branch_id = "civic_nye_name"
        start = 18
        duration = 7
    elif event_type == "parade_evening":
        branch_id = "parade_evening"
        start = 19
        duration = 4
    elif event_type == "special_event_evening":
        branch_id = "special_event_evening"
        start = 18
        duration = 6
    elif event_type == "parade":
        branch_id = "parade_daytime"
        start = 9
        duration = 8
    elif event_type == "special_event":
        branch_id = "special_event_daytime"
        start = 8
        duration = 9

    # Political events
    elif "political" in event_type:
        if "un ga" in event_name or "general debate" in event_name:
            branch_id = "POLITICAL_UN_GA"
            start = 9
            duration = 10
        elif "election" in event_name:
            branch_id = "POLITICAL_ELECTION"
            start = 17
            duration = 4
        elif "inauguration" in event_name:
            branch_id = "POLITICAL_INAUGURATION"
            start = 12
            duration = 4
        else:
            branch_id = "POLITICAL_OTHER"
            start = 9
            duration = 10

    else:
        branch_id = "fallback_default"
        start = 19
        duration = 3

    end = start + duration
    return pd.Series({"start_hour": start, "end_hour": end, "branch_id": branch_id})


events[["start_hour", "end_hour", "branch_id"]] = events.apply(get_event_window, axis=1)

# --- Branch validation (no silent fallback for civic/political leagues) ---
_CRITICAL_LEAGUES = {"MARATHON", "BIKE_TOUR", "PARADE", "SPECIAL", "POLITICAL"}
_events_league_upper = events["league"].astype(str).str.upper()
_bad = events[
    _events_league_upper.isin(_CRITICAL_LEAGUES) & (events["branch_id"] == "fallback_default")
]
if len(_bad) > 0:
    names = _bad["event_name"].tolist()
    raise AssertionError(
        "get_event_window fallback_default for critical league — fix event_type/name in CSV: "
        + repr(names[:25])
        + (f" ... (+{len(names) - 25} more)" if len(names) > 25 else "")
    )

print("\n  Branch coverage (league × event_type × branch_id):")
print(
    events.groupby(["league", "event_type", "branch_id"], dropna=False)
    .size()
    .reset_index(name="n_events")
    .to_string(index=False)
)
events["event_start_dt"] = events.apply(
    lambda r: r["date"] + pd.Timedelta(hours=int(r["start_hour"])), axis=1
)
events["event_end_dt"] = events.apply(
    lambda r: r["date"] + pd.Timedelta(hours=int(r["end_hour"])), axis=1
)

events.to_parquet("data/processed/_events_with_times.parquet")
print(f"  ✅ {len(events)} events with start/end times")
print("  Sample windows by league:")
print(
    events.groupby("league")[["start_hour", "end_hour"]]
    .agg(["min", "max"])
    .to_string()
)

# Sanity checks on new events
print("\n  Spot-checks for new events:")
for name in [
    "TCS NYC Marathon 2024",
    "TD Five Boro Bike Tour 2024",
    "Times Square NYE Ball Drop 2024-25",
    "Village Halloween Parade 2024",
    "Macy's Thanksgiving Day Parade 2024",
]:
    row = events[events["event_name"] == name]
    if len(row):
        r = row.iloc[0]
        print(
            f"    {name}: {r['event_start_dt']} → {r['event_end_dt']} "
            f"(zone {r['zone_id']})"
        )

print("\n[2/4] Expanding NYC events into (zone, hour) windows...")

events_nyc = events[events["zone_id"] > 0].copy()
print(
    f"  Events in NYC zones: {len(events_nyc)} (dropped {len(events) - len(events_nyc)} NJ events)"
)

sym_rows = []
asym_rows = []

for _, ev in events_nyc.iterrows():
    zone = int(ev["zone_id"])

    # Symmetric: ±3hr around event START
    sym_start = ev["event_start_dt"] - pd.Timedelta(hours=3)
    sym_end = ev["event_start_dt"] + pd.Timedelta(hours=3)
    for h in pd.date_range(sym_start, sym_end, freq="h"):
        sym_rows.append({"pickup_zone": zone, "pickup_hour": h, "is_event_sym": 1})

    # Asymmetric: -2hr from start to +4hr after end
    # NB: this naturally widens for long-duration events (UN GA = 10hr → 16hr asym window)
    asym_start = ev["event_start_dt"] - pd.Timedelta(hours=2)
    asym_end = ev["event_end_dt"] + pd.Timedelta(hours=4)
    for h in pd.date_range(asym_start, asym_end, freq="h"):
        asym_rows.append({"pickup_zone": zone, "pickup_hour": h, "is_event_asym": 1})

sym_raw = pd.DataFrame(sym_rows)
asym_raw = pd.DataFrame(asym_rows)

sym_count = (
    sym_raw.groupby(["pickup_zone", "pickup_hour"])
    .size()
    .reset_index(name="event_count_sym")
)
asym_count = (
    asym_raw.groupby(["pickup_zone", "pickup_hour"])
    .size()
    .reset_index(name="event_count_asym")
)

sym_df = (
    sym_raw[["pickup_zone", "pickup_hour", "is_event_sym"]]
    .drop_duplicates()
    .merge(sym_count, on=["pickup_zone", "pickup_hour"])
)
asym_df = (
    asym_raw[["pickup_zone", "pickup_hour", "is_event_asym"]]
    .drop_duplicates()
    .merge(asym_count, on=["pickup_zone", "pickup_hour"])
)

sym_df.to_parquet("data/processed/_event_flags_sym.parquet")
asym_df.to_parquet("data/processed/_event_flags_asym.parquet")
print(f"  Symmetric flag rows: {len(sym_df):,}")
print(f"  Asymmetric flag rows: {len(asym_df):,}")

print("\n[3/4] Updating master_zone_hour.parquet...")
t0 = time.perf_counter()

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET temp_directory='data/processed/duckdb_temp'")

mp = os.path.abspath("data/processed/master_zone_hour.parquet").replace("\\", "/")
col_df = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{mp}')").df()
col_set = set(col_df["column_name"].tolist())

exclude_candidates = [
    "is_event_sym",
    "is_event_asym",
    "event_count_sym",
    "event_count_asym",
    "has_major_event_dayflag",
    "has_major_event",
    "total_event_attendance",
    "is_nyc_event_sym",
    "is_nyc_event_asym",
    "is_event_combined_sym",
    "is_event_combined_asym",
]
exclude_sql = ", ".join(c for c in exclude_candidates if c in col_set)
if not exclude_sql.strip():
    raise RuntimeError(
        "master_zone_hour.parquet has no expected event columns to replace; check schema."
    )

if "has_major_event_dayflag" in col_set:
    dayflag_expr = "m.has_major_event_dayflag AS has_major_event_dayflag"
elif "has_major_event" in col_set:
    dayflag_expr = "m.has_major_event AS has_major_event_dayflag"
else:
    dayflag_expr = "CAST(0 AS INTEGER) AS has_major_event_dayflag"

if "total_event_attendance" in col_set:
    attend_expr = "m.total_event_attendance AS total_event_attendance"
else:
    attend_expr = "CAST(NULL AS DOUBLE) AS total_event_attendance"

nj_sym_expr = (
    "COALESCE(m.is_nj_event_pregame_sym, 0)"
    if "is_nj_event_pregame_sym" in col_set
    else "CAST(0 AS INTEGER)"
)
nj_asym_expr = (
    "COALESCE(m.is_nj_event_pregame_asym, 0)"
    if "is_nj_event_pregame_asym" in col_set
    else "CAST(0 AS INTEGER)"
)

copy_sql = f"""
    COPY (
        SELECT
            m.* EXCLUDE ({exclude_sql}),
            COALESCE(s.is_event_sym, 0) AS is_event_sym,
            COALESCE(s.event_count_sym, 0) AS event_count_sym,
            COALESCE(a.is_event_asym, 0) AS is_event_asym,
            COALESCE(a.event_count_asym, 0) AS event_count_asym,
            {dayflag_expr},
            {attend_expr},
            CASE
                WHEN COALESCE(s.is_event_sym, 0) = 1 OR {nj_sym_expr} = 1 THEN 1 ELSE 0
            END AS is_event_combined_sym,
            CASE
                WHEN COALESCE(a.is_event_asym, 0) = 1 OR {nj_asym_expr} = 1 THEN 1 ELSE 0
            END AS is_event_combined_asym
        FROM read_parquet('{mp}') m
        LEFT JOIN 'data/processed/_event_flags_sym.parquet' s
            ON m.pickup_zone = s.pickup_zone AND m.pickup_hour = s.pickup_hour
        LEFT JOIN 'data/processed/_event_flags_asym.parquet' a
            ON m.pickup_zone = a.pickup_zone AND m.pickup_hour = a.pickup_hour
    ) TO 'data/processed/master_zone_hour_v2.parquet' (FORMAT PARQUET)
"""
con.execute(copy_sql)

os.remove("data/processed/master_zone_hour.parquet")
shutil.move(
    "data/processed/master_zone_hour_v2.parquet",
    "data/processed/master_zone_hour.parquet",
)
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

print("\n[4/4] Verification (NYC flags only at this stage)...")
result = con.execute("""
    SELECT
        COUNT(*) AS rows,
        SUM(is_event_sym) AS event_hours_sym,
        SUM(is_event_asym) AS event_hours_asym,
        SUM(has_major_event_dayflag) AS event_hours_dayflag,
        ROUND(100.0 * SUM(is_event_sym) / COUNT(*), 2) AS pct_sym,
        ROUND(100.0 * SUM(is_event_asym) / COUNT(*), 2) AS pct_asym
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))

con.close()

for tmp in [
    "_events_with_times.parquet",
    "_event_flags_sym.parquet",
    "_event_flags_asym.parquet",
]:
    p = f"data/processed/{tmp}"
    if os.path.exists(p):
        os.remove(p)

print(f"\n  ⏱ NYC flags rebuild: {(time.perf_counter() - start) / 60:.1f} min")


# %%
# Cell 3 — Rebuild NJ-venue outbound event flags
import shutil
import time

import duckdb

print("=" * 70)
print("REBUILDING NJ-VENUE OUTBOUND EVENT FLAGS")
print("=" * 70)
start = time.perf_counter()

VENUE_DEPARTURE_ZONES = {
    "MetLife Stadium (NJ)": [186, 230, 161, 100, 246, 48],
    "Prudential Center (NJ)": [261, 87, 88, 231, 125, 113, 114, 158, 249],
    "UBS Arena (Belmont)": [186, 230, 161],
    "Red Bull Arena": [261, 87, 88, 231, 125, 113, 114],
}

events = pd.read_csv("data/raw/events/major_events.csv", encoding="utf-8")
events["date"] = pd.to_datetime(events["date"])
events["dow"] = events["date"].dt.dayofweek

nj_events = events[events["zone_id"] == -1].copy()
print(f"NJ events: {len(nj_events)}")
print(nj_events["venue"].value_counts().to_string())


def get_event_start_hour(row):
    league = str(row.get("league", "")).upper()
    is_weekend = row["dow"] in [5, 6]
    if "MLB" in league:
        return 13 if is_weekend else 19
    elif "NBA" in league or "NHL" in league or "MLS" in league:
        return 19
    elif "NFL" in league:
        return 13
    return 19


nj_events["start_hour"] = nj_events.apply(get_event_start_hour, axis=1)
nj_events["event_start_dt"] = nj_events.apply(
    lambda r: r["date"] + pd.Timedelta(hours=int(r["start_hour"])), axis=1
)

nj_sym_rows = []
nj_asym_rows = []

for _, ev in nj_events.iterrows():
    venue = ev["venue"]
    if venue not in VENUE_DEPARTURE_ZONES:
        print(f"  ⚠️ Unknown venue: {venue}, skipping")
        continue

    departure_zones = VENUE_DEPARTURE_ZONES[venue]

    sym_start = ev["event_start_dt"] - pd.Timedelta(hours=4)
    sym_end = ev["event_start_dt"] - pd.Timedelta(hours=1)
    for zone in departure_zones:
        for h in pd.date_range(sym_start, sym_end, freq="h"):
            nj_sym_rows.append(
                {
                    "pickup_zone": zone,
                    "pickup_hour": h,
                    "is_nj_event_pregame_sym": 1,
                }
            )

    asym_start = ev["event_start_dt"] - pd.Timedelta(hours=5)
    asym_end = ev["event_start_dt"] - pd.Timedelta(hours=1)
    for zone in departure_zones:
        for h in pd.date_range(asym_start, asym_end, freq="h"):
            nj_asym_rows.append(
                {
                    "pickup_zone": zone,
                    "pickup_hour": h,
                    "is_nj_event_pregame_asym": 1,
                }
            )

nj_sym_raw = pd.DataFrame(nj_sym_rows)
nj_asym_raw = pd.DataFrame(nj_asym_rows)

nj_sym_count = (
    nj_sym_raw.groupby(["pickup_zone", "pickup_hour"])
    .size()
    .reset_index(name="nj_event_count_sym")
)
nj_asym_count = (
    nj_asym_raw.groupby(["pickup_zone", "pickup_hour"])
    .size()
    .reset_index(name="nj_event_count_asym")
)

nj_sym_final = (
    nj_sym_raw[["pickup_zone", "pickup_hour", "is_nj_event_pregame_sym"]]
    .drop_duplicates()
    .merge(nj_sym_count, on=["pickup_zone", "pickup_hour"])
)
nj_asym_final = (
    nj_asym_raw[["pickup_zone", "pickup_hour", "is_nj_event_pregame_asym"]]
    .drop_duplicates()
    .merge(nj_asym_count, on=["pickup_zone", "pickup_hour"])
)

nj_sym_final.to_parquet("data/processed/_nj_event_flags_sym.parquet")
nj_asym_final.to_parquet("data/processed/_nj_event_flags_asym.parquet")
print(f"  NJ symmetric flag rows: {len(nj_sym_final):,}")
print(f"  NJ asymmetric flag rows: {len(nj_asym_final):,}")

print("\nJoining NJ flags to master and rebuilding NYC aliases + combined flags...")
t0 = time.perf_counter()

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET temp_directory='data/processed/duckdb_temp'")

mp = os.path.abspath("data/processed/master_zone_hour.parquet").replace("\\", "/")
col_set = set(
    con.execute(f"DESCRIBE SELECT * FROM read_parquet('{mp}')").df()["column_name"].tolist()
)
exclude_nj = [
    "is_nj_event_pregame_sym",
    "nj_event_count_sym",
    "is_nj_event_pregame_asym",
    "nj_event_count_asym",
    "is_event_combined_sym",
    "is_event_combined_asym",
    "is_nyc_event_sym",
    "is_nyc_event_asym",
]
exc = ", ".join(c for c in exclude_nj if c in col_set)
if not exc.strip():
    raise RuntimeError("Master missing expected NJ/alias columns. Re-run Cell 2 first.")

nj_copy_sql = f"""
    COPY (
        SELECT
            m.* EXCLUDE ({exc}),
            COALESCE(s.is_nj_event_pregame_sym, 0) AS is_nj_event_pregame_sym,
            COALESCE(s.nj_event_count_sym, 0) AS nj_event_count_sym,
            COALESCE(a.is_nj_event_pregame_asym, 0) AS is_nj_event_pregame_asym,
            COALESCE(a.nj_event_count_asym, 0) AS nj_event_count_asym,
            CASE
                WHEN m.is_event_sym = 1 OR COALESCE(s.is_nj_event_pregame_sym, 0) = 1
                THEN 1 ELSE 0
            END AS is_event_combined_sym,
            CASE
                WHEN m.is_event_asym = 1 OR COALESCE(a.is_nj_event_pregame_asym, 0) = 1
                THEN 1 ELSE 0
            END AS is_event_combined_asym,
            m.is_event_sym AS is_nyc_event_sym,
            m.is_event_asym AS is_nyc_event_asym
        FROM read_parquet('{mp}') m
        LEFT JOIN 'data/processed/_nj_event_flags_sym.parquet' s
            ON m.pickup_zone = s.pickup_zone AND m.pickup_hour = s.pickup_hour
        LEFT JOIN 'data/processed/_nj_event_flags_asym.parquet' a
            ON m.pickup_zone = a.pickup_zone AND m.pickup_hour = a.pickup_hour
    ) TO 'data/processed/master_zone_hour_v3.parquet' (FORMAT PARQUET)
"""
con.execute(nj_copy_sql)

os.remove("data/processed/master_zone_hour.parquet")
shutil.move(
    "data/processed/master_zone_hour_v3.parquet",
    "data/processed/master_zone_hour.parquet",
)
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")
con.close()

for tmp in ["_nj_event_flags_sym.parquet", "_nj_event_flags_asym.parquet"]:
    p = f"data/processed/{tmp}"
    if os.path.exists(p):
        os.remove(p)

print(f"\n  ⏱ NJ rebuild: {(time.perf_counter() - start) / 60:.1f} min")


# %%
# Cell 4 — Final verification (all flags + spot-checks)
import duckdb
import pandas as pd

print("=" * 70)
print("FINAL VERIFICATION")
print("=" * 70)

print("\n[A] Total event flag distribution:")
result = duckdb.query("""
    SELECT
        COUNT(*) AS total_rows,
        SUM(is_nyc_event_sym) AS nyc_sym,
        SUM(is_nyc_event_asym) AS nyc_asym,
        SUM(is_nj_event_pregame_sym) AS nj_sym,
        SUM(is_nj_event_pregame_asym) AS nj_asym,
        SUM(is_event_combined_sym) AS combined_sym,
        SUM(is_event_combined_asym) AS combined_asym
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))

print("\n[B] Spot-check: Marathon (zone 43, 2024-11-03 8am-5pm should be flagged):")
result = duckdb.query("""
    SELECT pickup_hour, MAX(is_nyc_event_sym) AS sym, MAX(is_nyc_event_asym) AS asym
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE pickup_zone = 43
        AND pickup_hour BETWEEN '2024-11-03 04:00' AND '2024-11-03 23:00'
    GROUP BY 1 ORDER BY 1
""").df()
print(result.to_string(index=False))

print("\n[C] Spot-check: NYE Ball Drop (zone 230, midnight crossing 2024-12-31→2025-01-01):")
result = duckdb.query("""
    SELECT pickup_hour, MAX(is_nyc_event_sym) AS sym, MAX(is_nyc_event_asym) AS asym
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE pickup_zone = 230
        AND pickup_hour BETWEEN '2024-12-31 14:00' AND '2025-01-01 06:00'
    GROUP BY 1 ORDER BY 1
""").df()
print(result.to_string(index=False))

print("\n[D] Spot-check: Halloween Parade (zone 113, 2024-10-31 7pm-11pm):")
result = duckdb.query("""
    SELECT pickup_hour, MAX(is_nyc_event_sym) AS sym, MAX(is_nyc_event_asym) AS asym
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE pickup_zone = 113
        AND pickup_hour BETWEEN '2024-10-31 14:00' AND '2024-11-01 04:00'
    GROUP BY 1 ORDER BY 1
""").df()
print(result.to_string(index=False))

print("\n[E] Spot-check: Bike Tour 2024 (zone 87, 2024-05-05 7am-2pm):")
result = duckdb.query("""
    SELECT pickup_hour, MAX(is_nyc_event_sym) AS sym, MAX(is_nyc_event_asym) AS asym
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE pickup_zone = 87
        AND pickup_hour BETWEEN '2024-05-05 04:00' AND '2024-05-05 20:00'
    GROUP BY 1 ORDER BY 1
""").df()
print(result.to_string(index=False))

print("\n✅ MASTER REBUILD COMPLETE. Ready to run notebooks 02-08.")
