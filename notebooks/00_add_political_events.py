# %%
# Cell 1 — Append 6 verified political events to major_events.csv
import os

os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")

import pandas as pd

# Political events verified via UN.org and US State Department
# UN GA dates: General Debate Sept 24-30, 2024 (using Sept 24-27, the high-density Manhattan period)
# Zone 233 = UN/Turtle Bay South (UN HQ location, verified against TLC zone shapefile)
# Election Day = Nov 5, 2024 (federal law)
# Inauguration = Jan 20, 2025 (federal law, held in DC, NYC ripple effects only)

political_events = pd.DataFrame(
    [
        {
            "date": "2024-09-24",
            "venue": "UN HQ",
            "zone_id": 233,
            "event_name": "UN GA General Debate Day 1",
            "expected_attendance": 0,
            "event_type": "political",
            "lead_time_days": 365,
            "team": "",
            "league": "POLITICAL",
        },
        {
            "date": "2024-09-25",
            "venue": "UN HQ",
            "zone_id": 233,
            "event_name": "UN GA General Debate",
            "expected_attendance": 0,
            "event_type": "political",
            "lead_time_days": 365,
            "team": "",
            "league": "POLITICAL",
        },
        {
            "date": "2024-09-26",
            "venue": "UN HQ",
            "zone_id": 233,
            "event_name": "UN GA General Debate",
            "expected_attendance": 0,
            "event_type": "political",
            "lead_time_days": 365,
            "team": "",
            "league": "POLITICAL",
        },
        {
            "date": "2024-09-27",
            "venue": "UN HQ",
            "zone_id": 233,
            "event_name": "UN GA General Debate",
            "expected_attendance": 0,
            "event_type": "political",
            "lead_time_days": 365,
            "team": "",
            "league": "POLITICAL",
        },
        {
            "date": "2024-11-05",
            "venue": "Citywide",
            "zone_id": 161,
            "event_name": "Election Day 2024",
            "expected_attendance": 0,
            "event_type": "political",
            "lead_time_days": 1460,
            "team": "",
            "league": "POLITICAL",
        },
        {
            "date": "2025-01-20",
            "venue": "DC (NYC ripple)",
            "zone_id": 161,
            "event_name": "Presidential Inauguration",
            "expected_attendance": 0,
            "event_type": "political",
            "lead_time_days": 90,
            "team": "",
            "league": "POLITICAL",
        },
    ]
)

existing = pd.read_csv(
    "data/raw/events/major_events.csv",
    encoding="utf-8",
)
print(f"Before: {len(existing)} events")

political_events = political_events[existing.columns.tolist()]
dedupe_key = ["date", "zone_id", "event_name"]
combined = pd.concat([existing, political_events], ignore_index=True)
n_before = len(combined)
combined = combined.drop_duplicates(subset=dedupe_key, keep="first")
duplicates_removed = n_before - len(combined)
combined.to_csv("data/raw/events/major_events.csv", index=False)
print(f"After: {len(combined)} events (removed {duplicates_removed} duplicate key rows if re-run)")
print("\nLeague breakdown:")
print(combined["league"].value_counts())


# %%
# Cell 2 — Rebuild NYC hour-level event flags with updated political event windowing
import shutil
import time

import duckdb

print("=" * 70)
print("ADDING HOUR-LEVEL EVENT FLAGS (with political timing)")
print("=" * 70)
start = time.perf_counter()

print("\n[1/4] Event start/end times from major_events.csv...")

events = pd.read_csv(
    "data/raw/events/major_events.csv",
    encoding="utf-8",
)
events["date"] = pd.to_datetime(events["date"])
events["dow"] = events["date"].dt.dayofweek


def get_event_window(row):
    """Returns (start_hour, end_hour) — both inclusive — for the event."""
    league = str(row.get("league", "")).upper()
    event_type = str(row.get("event_type", "")).lower()
    is_weekend = row["dow"] in [5, 6]

    if "MLB" in league:
        if is_weekend:
            start = 13
        else:
            start = 19
        duration = 4
    elif "NBA" in league:
        start = 19
        duration = 3
    elif "NHL" in league:
        start = 19
        duration = 3
    elif "NFL" in league:
        start = 13
        duration = 4
    elif "MLS" in league:
        start = 19
        duration = 3
    elif "parade" in event_type or "marathon" in event_type:
        start = 9
        duration = 6
    elif "political" in event_type:
        event_name = str(row.get("event_name", "")).lower()
        if "un ga" in event_name or "general debate" in event_name:
            start = 9
            duration = 10
        elif "election" in event_name:
            start = 17
            duration = 4
        elif "inauguration" in event_name:
            start = 12
            duration = 4
        else:
            start = 9
            duration = 10
    else:
        start = 19
        duration = 3

    end = start + duration
    return pd.Series({"start_hour": start, "end_hour": end})


events[["start_hour", "end_hour"]] = events.apply(get_event_window, axis=1)
events["event_start_dt"] = events.apply(
    lambda r: r["date"] + pd.Timedelta(hours=r["start_hour"]), axis=1
)
events["event_end_dt"] = events.apply(
    lambda r: r["date"] + pd.Timedelta(hours=r["end_hour"]), axis=1
)

events.to_parquet("data/processed/_events_with_times.parquet")
print(f"  ✅ {len(events)} events with start/end times")
print("  Sample by league:")
print(events.groupby("league")[["start_hour", "end_hour"]].first().to_string())

print("\n[2/4] Expanding events into (zone, hour) windows...")

events_nyc = events[events["zone_id"] > 0].copy()
print(f"  Events in NYC zones: {len(events_nyc)} (dropped {len(events) - len(events_nyc)} NJ events)")

sym_rows = []
asym_rows = []

for _, ev in events_nyc.iterrows():
    zone = int(ev["zone_id"])

    sym_start = ev["event_start_dt"] - pd.Timedelta(hours=3)
    sym_end = ev["event_start_dt"] + pd.Timedelta(hours=3)
    for h in pd.date_range(sym_start, sym_end, freq="h"):
        sym_rows.append({"pickup_zone": zone, "pickup_hour": h, "is_event_sym": 1})

    asym_start = ev["event_start_dt"] - pd.Timedelta(hours=2)
    asym_end = ev["event_end_dt"] + pd.Timedelta(hours=4)
    for h in pd.date_range(asym_start, asym_end, freq="h"):
        asym_rows.append({"pickup_zone": zone, "pickup_hour": h, "is_event_asym": 1})

sym_df = pd.DataFrame(sym_rows).drop_duplicates(subset=["pickup_zone", "pickup_hour"])
asym_df = pd.DataFrame(asym_rows).drop_duplicates(subset=["pickup_zone", "pickup_hour"])

sym_count = (
    pd.DataFrame(sym_rows).groupby(["pickup_zone", "pickup_hour"]).size().reset_index(name="event_count_sym")
)
asym_count = (
    pd.DataFrame(asym_rows).groupby(["pickup_zone", "pickup_hour"]).size().reset_index(name="event_count_asym")
)

sym_df = sym_df.merge(sym_count, on=["pickup_zone", "pickup_hour"], how="left")
asym_df = asym_df.merge(asym_count, on=["pickup_zone", "pickup_hour"], how="left")

sym_df.to_parquet("data/processed/_event_flags_sym.parquet")
asym_df.to_parquet("data/processed/_event_flags_asym.parquet")
print(f"  Symmetric flag rows: {len(sym_df):,}")
print(f"  Asymmetric flag rows: {len(asym_df):,}")

print("\n[3/4] Updating master_zone_hour.parquet (in-place via COPY swap)...")
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

# Exclude NYC/combined aliases so Cell 3 can rebuild is_nyc_* ; keep NJ columns in m.*
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
shutil.move("data/processed/master_zone_hour_v2.parquet", "data/processed/master_zone_hour.parquet")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

print("\n[4/4] Verification (hour-level vs day-level)...")
result = con.execute("""
    SELECT
        COUNT(*) AS rows,
        SUM(is_event_sym) AS event_hours_sym,
        SUM(is_event_asym) AS event_hours_asym,
        SUM(has_major_event_dayflag) AS event_hours_dayflag,
        ROUND(100.0 * SUM(is_event_sym) / COUNT(*), 2) AS pct_sym,
        ROUND(100.0 * SUM(is_event_asym) / COUNT(*), 2) AS pct_asym,
        ROUND(100.0 * SUM(has_major_event_dayflag) / COUNT(*), 2) AS pct_dayflag,
        SUM(is_event_combined_sym) AS combined_sym
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))

print("\nFlag overlap (sample zones):")
overlap = con.execute("""
    SELECT
        is_event_sym,
        has_major_event_dayflag,
        COUNT(*) AS rows
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE pickup_zone IN (161, 247, 90, 233)
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()
print(overlap.to_string(index=False))

con.close()

for tmp in ["_events_with_times.parquet", "_event_flags_sym.parquet", "_event_flags_asym.parquet"]:
    path = f"data/processed/{tmp}"
    if os.path.exists(path):
        os.remove(path)

print(f"\n{'=' * 70}")
print(f"✅ NYC EVENT FLAGS COMPLETE — {(time.perf_counter() - start) / 60:.1f} min")
print(f"{'=' * 70}")
print("Note: Run Cell 3 next to refresh is_nyc_event_* aliases and NJ joins.")


# %%
# Cell 3 — Rebuild NJ-venue outbound event flags (same logic as build_master)
import shutil
import time

import duckdb
import pandas as pd

print("=" * 70)
print("ADDING NJ-VENUE OUTBOUND EVENT FLAGS")
print("=" * 70)
start = time.perf_counter()

print("\n[1/4] Defining per-venue departure zone clusters...")

VENUE_DEPARTURE_ZONES = {
    "MetLife Stadium (NJ)": [
        186,
        230,
        161,
        100,
        246,
        48,
    ],
    "Prudential Center (NJ)": [
        261,
        87,
        88,
        231,
        125,
        113,
        114,
        158,
        249,
    ],
    "UBS Arena (Belmont)": [
        186,
        230,
        161,
    ],
    "Red Bull Arena": [
        261,
        87,
        88,
        231,
        125,
        113,
        114,
    ],
}

for venue, zones in VENUE_DEPARTURE_ZONES.items():
    print(f"  {venue}: {len(zones)} departure zones")

print("\n[2/4] Building NJ event windows with departure zones...")

events = pd.read_csv(
    "data/raw/events/major_events.csv",
    encoding="utf-8",
)
events["date"] = pd.to_datetime(events["date"])
events["dow"] = events["date"].dt.dayofweek

nj_events = events[events["zone_id"] == -1].copy()
print(f"  NJ events: {len(nj_events)}")
print("  By venue:")
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
    else:
        return 19


nj_events["start_hour"] = nj_events.apply(get_event_start_hour, axis=1)
nj_events["event_start_dt"] = nj_events.apply(
    lambda r: r["date"] + pd.Timedelta(hours=r["start_hour"]), axis=1
)

print("\n[3/4] Expanding NJ events into hour-level flags on departure zones...")

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
                    "venue": venue,
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
                    "venue": venue,
                }
            )

nj_sym = pd.DataFrame(nj_sym_rows)
nj_asym = pd.DataFrame(nj_asym_rows)

nj_sym_counts = nj_sym.groupby(["pickup_zone", "pickup_hour"]).size().reset_index(name="nj_event_count_sym")
nj_asym_counts = nj_asym.groupby(["pickup_zone", "pickup_hour"]).size().reset_index(name="nj_event_count_asym")

nj_sym_flags = nj_sym[["pickup_zone", "pickup_hour", "is_nj_event_pregame_sym"]].drop_duplicates()
nj_asym_flags = nj_asym[["pickup_zone", "pickup_hour", "is_nj_event_pregame_asym"]].drop_duplicates()

nj_sym_final = nj_sym_flags.merge(nj_sym_counts, on=["pickup_zone", "pickup_hour"], how="left")
nj_asym_final = nj_asym_flags.merge(nj_asym_counts, on=["pickup_zone", "pickup_hour"], how="left")

nj_sym_final.to_parquet("data/processed/_nj_event_flags_sym.parquet")
nj_asym_final.to_parquet("data/processed/_nj_event_flags_asym.parquet")

print(f"  NJ symmetric flag rows (zone-hour): {len(nj_sym_final):,}")
print(f"  NJ asymmetric flag rows (zone-hour): {len(nj_asym_final):,}")

print("\n[4/4] Joining NJ flags to master table...")
t0 = time.perf_counter()

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET temp_directory='data/processed/duckdb_temp'")

mp3 = os.path.abspath("data/processed/master_zone_hour.parquet").replace("\\", "/")
c3 = set(
    con.execute(f"DESCRIBE SELECT * FROM read_parquet('{mp3}')").df()["column_name"].tolist()
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
exc3 = ", ".join(c for c in exclude_nj if c in c3)
if not exc3.strip():
    raise RuntimeError(
        "master_zone_hour.parquet missing NJ / combined / alias columns to refresh; "
        "run Cell 2 first on a full pipeline master."
    )

nj_copy_sql = f"""
    COPY (
        SELECT
            m.* EXCLUDE ({exc3}),
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
        FROM read_parquet('{mp3}') m
        LEFT JOIN 'data/processed/_nj_event_flags_sym.parquet' s
            ON m.pickup_zone = s.pickup_zone AND m.pickup_hour = s.pickup_hour
        LEFT JOIN 'data/processed/_nj_event_flags_asym.parquet' a
            ON m.pickup_zone = a.pickup_zone AND m.pickup_hour = a.pickup_hour
    ) TO 'data/processed/master_zone_hour_v3.parquet' (FORMAT PARQUET)
"""
con.execute(nj_copy_sql)

os.remove("data/processed/master_zone_hour.parquet")
shutil.move("data/processed/master_zone_hour_v3.parquet", "data/processed/master_zone_hour.parquet")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

print("\nVerification: event flag distribution")
result = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        SUM(is_nyc_event_sym) AS nyc_sym_hours,
        SUM(is_nyc_event_asym) AS nyc_asym_hours,
        SUM(is_nj_event_pregame_sym) AS nj_sym_hours,
        SUM(is_nj_event_pregame_asym) AS nj_asym_hours,
        SUM(is_event_combined_sym) AS combined_sym_hours,
        SUM(is_event_combined_asym) AS combined_asym_hours
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))

print("\nFinal master table columns (preview):")
schema = con.execute("DESCRIBE SELECT * FROM 'data/processed/master_zone_hour.parquet'").df()
print(schema[["column_name", "column_type"]].head(20).to_string(index=False))
print(f"... ({len(schema)} columns total)")

con.close()

for tmp in ["_nj_event_flags_sym.parquet", "_nj_event_flags_asym.parquet"]:
    path = f"data/processed/{tmp}"
    if os.path.exists(path):
        os.remove(path)

print(f"\n{'=' * 70}")
print(f"✅ NJ EVENT FLAGS COMPLETE — {(time.perf_counter() - start) / 60:.1f} min")
print(f"{'=' * 70}")


# %%
# Cell 4 — Final verification
import duckdb

print("Final master table event flag distribution:")
result = duckdb.query("""
    SELECT
        COUNT(*) AS total_rows,
        SUM(is_nyc_event_sym) AS nyc_sym_hours,
        SUM(is_nyc_event_asym) AS nyc_asym_hours,
        SUM(is_nj_event_pregame_sym) AS nj_sym_hours,
        SUM(is_nj_event_pregame_asym) AS nj_asym_hours,
        SUM(is_event_combined_sym) AS combined_sym_hours,
        SUM(is_event_combined_asym) AS combined_asym_hours
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))

print("\nUN GA dates check (zone 233, hours 9-18 should have is_nyc_event_sym = 1):")
result = duckdb.query("""
    SELECT
        pickup_hour,
        SUM(is_nyc_event_sym) AS flagged
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE pickup_zone = 233
        AND pickup_hour BETWEEN '2024-09-24 06:00' AND '2024-09-24 23:00'
    GROUP BY 1
    ORDER BY 1
""").df()
print(result.to_string(index=False))

print("\n✅ All political events now in master table.")
print("Ready to start Notebook 02 (Event Study) in next chat.")

# %%
import duckdb
result = duckdb.query("""
    SELECT 
        pickup_hour, 
        platform,
        is_event_sym,
        event_count_sym
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE pickup_zone = 233
        AND pickup_hour BETWEEN '2024-09-24 08:00' AND '2024-09-24 13:00'
    ORDER BY pickup_hour, platform
""").df()
print(result.to_string(index=False))
# %%
import pandas as pd
events = pd.read_csv("data/raw/events/major_events.csv")
print(events[events["date"] == "2024-11-03"])
# %%
import duckdb
duckdb.query("SELECT DISTINCT platform FROM 'data/processed/master_zone_hour.parquet'").df()
# %%
