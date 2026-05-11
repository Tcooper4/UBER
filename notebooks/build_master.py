# %%
import os
os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")
print(f"Working dir: {os.getcwd()}")
print(f"\nDoes src/ exist? {os.path.exists('src')}")
print(f"\nContents of src/:")
if os.path.exists('src'):
    for f in os.listdir('src'):
        print(f"  {f}")
else:
    print("  src/ doesn't exist!")

# %%
import os
import sys

# Force working dir to project root
os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")
print(f"Working dir: {os.getcwd()}")

# Add src/ to path with absolute path (more reliable than relative)
src_path = os.path.join(os.getcwd(), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
print(f"sys.path[0]: {sys.path[0]}")
print(f"crz_zones.py exists: {os.path.exists(os.path.join(src_path, 'crz_zones.py'))}")

# Now import
from crz_zones import CRZ_ZONE_IDS
print(f"Loaded {len(CRZ_ZONE_IDS)} CRZ zones")

print("=" * 70)
print("BUILDING MASTER ANALYSIS TABLE")
print("=" * 70)

start = time.perf_counter()

# Set up DuckDB with memory limit
con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET temp_directory='data/processed/duckdb_temp'")

# Step 1: Subway aggregated to zone-hour
# Each TLC zone gets the SUM of ridership across all subway stations
# whose centroid falls within that zone's polygon. We approximate by
# using nearest station via haversine — simple and avoids spatial join.
# Better: load station coords + zone polygons, do real spatial join.

print("\n[1/7] Building zone-hour subway aggregation...")
t0 = time.perf_counter()

# First, build station-to-zone mapping using zone polygons + station coords
# This is more accurate than nearest-station approximation
import geopandas as gpd
zones_gdf = gpd.read_file("data/raw/tlc_zones/taxi_zones.shp").to_crs(epsg=4326)
stations_df = pd.read_parquet("data/processed/mta_station_coords.parquet")

# Convert stations to GeoDataFrame
from shapely.geometry import Point
stations_df["geometry"] = stations_df.apply(
    lambda r: Point(r["longitude"], r["latitude"]), axis=1
)
stations_gdf = gpd.GeoDataFrame(stations_df, geometry="geometry", crs="EPSG:4326")

# Spatial join: each station gets its containing TLC zone
station_zone = gpd.sjoin(stations_gdf, zones_gdf[["LocationID", "geometry"]], 
                         how="left", predicate="within")
station_zone = station_zone[["station_complex", "LocationID"]].rename(
    columns={"LocationID": "pickup_zone"}
)
station_zone = station_zone.dropna(subset=["pickup_zone"])
station_zone["pickup_zone"] = station_zone["pickup_zone"].astype(int)
station_zone.to_parquet("data/processed/_station_to_zone.parquet")
print(f"  Mapped {len(station_zone)} stations to zones")

# Aggregate subway ridership to (zone, hour)
con.execute("""
    COPY (
        SELECT 
            sz.pickup_zone,
            mta.timestamp as pickup_hour,
            SUM(mta.ridership) as subway_riders_zone
        FROM 'data/processed/mta_subway_clean.parquet' mta
        INNER JOIN 'data/processed/_station_to_zone.parquet' sz
            ON mta.station_complex = sz.station_complex
        GROUP BY sz.pickup_zone, mta.timestamp
    ) TO 'data/processed/_subway_zone_hour.parquet' (FORMAT PARQUET)
""")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

# Step 2: Major events to (zone, date) flag
print("\n[2/7] Aggregating major events to zone-day...")
t0 = time.perf_counter()

events = pd.read_csv("data/raw/events/major_events.csv")
events["date"] = pd.to_datetime(events["date"]).dt.date
# For NJ venues (zone_id = -1), we'll skip since they're outside NYC zones.
# Their effect manifests as outbound trips from NYC zones, captured by trip data.
events_nyc = events[events["zone_id"] > 0].copy()

events_agg = events_nyc.groupby(["date", "zone_id"]).agg(
    has_major_event=("event_name", "count"),
    total_event_attendance=("expected_attendance", "sum"),
).reset_index()
events_agg["has_major_event"] = (events_agg["has_major_event"] > 0).astype(int)
events_agg = events_agg.rename(columns={"zone_id": "pickup_zone"})
events_agg.to_parquet("data/processed/_events_zone_day.parquet")
print(f"  {len(events_agg)} zone-day event records")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

# Step 3: SAPO events to (borough, date) — borough-level since no zone info
print("\n[3/7] Aggregating SAPO permits to borough-day...")
t0 = time.perf_counter()

sapo = pd.read_parquet("data/processed/sapo_permits_clean.parquet")
sapo_agg = sapo.groupby(["event_date", "borough"]).agg(
    sapo_event_count=("attend_binary", "sum"),
    sapo_typed_attendance=("attend_typed", "sum"),
    sapo_location_attendance=("attend_location", "sum"),
).reset_index()
sapo_agg.columns = ["event_date", "borough", "sapo_count", "sapo_typed", "sapo_location"]
sapo_agg["event_date"] = pd.to_datetime(sapo_agg["event_date"]).dt.date
sapo_agg.to_parquet("data/processed/_sapo_borough_day.parquet")
print(f"  {len(sapo_agg)} borough-day SAPO records")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

# Step 4: Storm events to date flag (citywide)
print("\n[4/7] Building storm event flags by date...")
t0 = time.perf_counter()

storms = pd.read_parquet("data/processed/noaa_storms_nyc.parquet")
storm_dates = []
for _, row in storms.iterrows():
    if pd.notna(row["begin_date"]):
        # Expand multi-day storms across each date
        end = row["end_date"] if pd.notna(row["end_date"]) else row["begin_date"]
        for d in pd.date_range(row["begin_date"], end, freq="D"):
            storm_dates.append({"event_date": d.date(), "event_type": row["event_type"]})

storm_dates_df = pd.DataFrame(storm_dates).drop_duplicates(subset=["event_date"])
storm_dates_df["is_storm_active"] = 1
storm_dates_df.to_parquet("data/processed/_storm_dates.parquet")
print(f"  {len(storm_dates_df)} storm-active dates")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

# Step 5: Build zone -> borough map for SAPO join
print("\n[5/7] Building zone-to-borough lookup...")
zone_lookup = pd.read_csv("data/raw/tlc_zones/taxi_zone_lookup.csv")
zone_lookup = zone_lookup[["LocationID", "Borough"]].rename(
    columns={"LocationID": "pickup_zone", "Borough": "borough"}
)
zone_lookup.to_parquet("data/processed/_zone_borough.parquet")

# Step 6: The big join
print("\n[6/7] Joining everything to TLC zone-hour...")
t0 = time.perf_counter()

# Build CRZ list as SQL string
crz_str = ",".join(str(z) for z in CRZ_ZONE_IDS)

con.execute(f"""
    COPY (
        WITH base AS (
            SELECT 
                tlc.*,
                CAST(tlc.pickup_hour AS DATE) as pickup_date,
                EXTRACT(dow FROM tlc.pickup_hour) as dow,
                EXTRACT(hour FROM tlc.pickup_hour) as hour_of_day,
                EXTRACT(dow FROM tlc.pickup_hour) * 24 + EXTRACT(hour FROM tlc.pickup_hour) as hour_of_week,
                CASE WHEN EXTRACT(dow FROM tlc.pickup_hour) IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
                CASE WHEN tlc.pickup_zone IN ({crz_str}) THEN 1 ELSE 0 END as is_in_crz,
                CASE WHEN tlc.pickup_hour >= '2025-01-05' THEN 1 ELSE 0 END as is_post_cp
            FROM 'data/processed/tlc_zone_hour.parquet' tlc
        ),
        with_weather AS (
            SELECT 
                base.*,
                w.temp_f,
                w.precip_in,
                w.wind_mph,
                w.visibility_mi,
                CAST(w.is_rain AS INT) as is_rain,
                CAST(w.is_heavy_rain AS INT) as is_heavy_rain,
                CAST(w.is_freezing AS INT) as is_freezing,
                CAST(w.is_extreme_cold AS INT) as is_extreme_cold,
                CAST(w.is_extreme_heat AS INT) as is_extreme_heat
            FROM base
            LEFT JOIN 'data/processed/zone_to_weather_station.parquet' zs
                ON base.pickup_zone = zs.LocationID
            LEFT JOIN 'data/processed/noaa_weather_clean.parquet' w
                ON zs.nearest_station = w.station_name
                AND base.pickup_hour = w.timestamp
        ),
        with_subway AS (
            SELECT 
                ww.*,
                COALESCE(s.subway_riders_zone, 0) as subway_riders_zone
            FROM with_weather ww
            LEFT JOIN 'data/processed/_subway_zone_hour.parquet' s
                ON ww.pickup_zone = s.pickup_zone
                AND ww.pickup_hour = s.pickup_hour
        ),
        with_events AS (
            SELECT 
                ws.*,
                COALESCE(e.has_major_event, 0) as has_major_event,
                COALESCE(e.total_event_attendance, 0) as total_event_attendance
            FROM with_subway ws
            LEFT JOIN 'data/processed/_events_zone_day.parquet' e
                ON ws.pickup_zone = e.pickup_zone
                AND ws.pickup_date = e.date
        ),
        with_borough AS (
            SELECT 
                we.*,
                zb.borough
            FROM with_events we
            LEFT JOIN 'data/processed/_zone_borough.parquet' zb
                ON we.pickup_zone = zb.pickup_zone
        ),
        with_sapo AS (
            SELECT 
                wb.*,
                COALESCE(sp.sapo_count, 0) as sapo_count_borough,
                COALESCE(sp.sapo_typed, 0) as sapo_typed_borough,
                COALESCE(sp.sapo_location, 0) as sapo_location_borough
            FROM with_borough wb
            LEFT JOIN 'data/processed/_sapo_borough_day.parquet' sp
                ON wb.borough = sp.borough
                AND wb.pickup_date = sp.event_date
        ),
        with_storms AS (
            SELECT 
                ws.*,
                COALESCE(st.is_storm_active, 0) as is_storm_active
            FROM with_sapo ws
            LEFT JOIN 'data/processed/_storm_dates.parquet' st
                ON ws.pickup_date = st.event_date
        ),
        with_holidays AS (
            SELECT 
                wst.*,
                COALESCE(CAST(h.is_holiday AS INT), 0) as is_holiday
            FROM with_storms wst
            LEFT JOIN 'data/processed/holidays.parquet' h
                ON wst.pickup_date = h.date
        )
        SELECT * FROM with_holidays
    ) TO 'data/processed/master_zone_hour.parquet' (FORMAT PARQUET)
""")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

# Step 7: Verify the master table
print("\n[7/7] Verifying master table...")
result = con.execute("""
    SELECT 
        COUNT(*) as rows,
        MIN(pickup_hour) as earliest,
        MAX(pickup_hour) as latest,
        COUNT(DISTINCT pickup_zone) as zones,
        COUNT(DISTINCT platform) as platforms,
        SUM(CASE WHEN is_in_crz = 1 THEN 1 ELSE 0 END) as crz_rows,
        SUM(CASE WHEN is_post_cp = 1 THEN 1 ELSE 0 END) as post_cp_rows,
        SUM(CASE WHEN is_holiday = 1 THEN 1 ELSE 0 END) as holiday_rows,
        SUM(CASE WHEN has_major_event = 1 THEN 1 ELSE 0 END) as major_event_rows,
        SUM(CASE WHEN is_rain = 1 THEN 1 ELSE 0 END) as rain_rows,
        ROUND(AVG(subway_riders_zone), 1) as avg_subway_riders,
        ROUND(AVG(temp_f), 1) as avg_temp,
        SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) as missing_weather_rows
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))

print("\n--- Schema ---")
schema = con.execute("DESCRIBE SELECT * FROM 'data/processed/master_zone_hour.parquet'").df()
print(schema.to_string(index=False))

con.close()

# Clean up intermediate files
for tmp in ["_subway_zone_hour.parquet", "_events_zone_day.parquet", 
            "_sapo_borough_day.parquet", "_storm_dates.parquet",
            "_station_to_zone.parquet", "_zone_borough.parquet"]:
    path = f"data/processed/{tmp}"
    if os.path.exists(path):
        os.remove(path)

total = time.perf_counter() - start
print(f"\n{'=' * 70}")
print(f"✅ MASTER TABLE COMPLETE — {total/60:.1f} minutes total")
print(f"{'=' * 70}")
print("\nOutput: data/processed/master_zone_hour.parquet")
print("Every analysis runs on this table.")
# %%

# %%
import duckdb

# Where are the missing weather rows?
result = duckdb.query("""
    SELECT 
        DATE_TRUNC('month', pickup_hour) as month,
        SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) as missing,
        COUNT(*) as total,
        ROUND(100.0 * SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_missing
    FROM 'data/processed/master_zone_hour.parquet'
    GROUP BY 1
    ORDER BY 1
""").df()
print("Weather missing by month:")
print(result.to_string(index=False))
# %%

# %%
import duckdb

# Check which station is missing data, by month
result = duckdb.query("""
    SELECT 
        DATE_TRUNC('month', timestamp) as month,
        station_name,
        COUNT(*) as obs,
        SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) as missing_temp
    FROM 'data/processed/noaa_weather_clean.parquet'
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()
print(result.to_string(index=False))
# %%

# %%
# ============================================================
# ADD HOUR-LEVEL EVENT FLAGS TO MASTER TABLE
# Builds both symmetric (±3hr) and asymmetric (-2hr to +4hr) windows
# ============================================================
import os
os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")

import pandas as pd
import duckdb
import time
from datetime import datetime, timedelta

print("=" * 70)
print("ADDING HOUR-LEVEL EVENT FLAGS")
print("=" * 70)
start = time.perf_counter()

# ============================================================
# Step 1: Load events and add start times based on league + day-of-week
# ============================================================
print("\n[1/4] Adding event start times to major_events.csv...")

events = pd.read_csv("data/raw/events/major_events.csv")
events["date"] = pd.to_datetime(events["date"])
events["dow"] = events["date"].dt.dayofweek  # 0=Mon, 6=Sun

# Default start hours by league/category
def get_event_window(row):
    """Returns (start_hour, end_hour) — both inclusive — for the event."""
    league = str(row.get("league", "")).upper()
    event_type = str(row.get("event_type", "")).lower()
    is_weekend = row["dow"] in [5, 6]  # Sat or Sun
    
    # Sports leagues
    if "MLB" in league:
        # Weekend day games at 1:30pm, weekday night games at 7pm
        if is_weekend:
            start = 13  # 1:30pm — round to 13
        else:
            start = 19  # 7pm
        duration = 4  # ~3 hour game + 1 hour for travel/clearing
    elif "NBA" in league:
        start = 19  # 7:30pm — round to 19
        duration = 3
    elif "NHL" in league:
        start = 19  # 7pm
        duration = 3
    elif "NFL" in league:
        start = 13  # 1pm Sunday default
        duration = 4
    elif "MLS" in league:
        start = 19  # 7:30pm — round to 19
        duration = 3
    # Non-sports
    elif "parade" in event_type or "marathon" in event_type:
        start = 9  # 9am parade/marathon start
        duration = 6  # runs through afternoon
    elif "political" in event_type:
        # Citywide all-day effect
        start = 0
        duration = 24
    else:
        # Default: evening event
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

# Save the augmented events file (preserves original)
events.to_parquet("data/processed/_events_with_times.parquet")
print(f"  ✅ {len(events)} events with start/end times")
print(f"  Sample by league:")
print(events.groupby("league")[["start_hour", "end_hour"]].first().to_string())

# ============================================================
# Step 2: Expand events into (zone, hour) windows for both methods
# ============================================================
print("\n[2/4] Expanding events into hour-level windows...")

# Filter to events in NYC zones (drop NJ events with zone_id = -1)
events_nyc = events[events["zone_id"] > 0].copy()
print(f"  Events in NYC zones: {len(events_nyc)} (dropped {len(events) - len(events_nyc)} NJ events)")

sym_rows = []      # symmetric: ±3hr from event_start_dt
asym_rows = []     # asymmetric: -2hr to +4hr from event_end_dt

for _, ev in events_nyc.iterrows():
    zone = int(ev["zone_id"])
    
    # Symmetric: 3 hours before to 3 hours after event_start_dt
    sym_start = ev["event_start_dt"] - pd.Timedelta(hours=3)
    sym_end = ev["event_start_dt"] + pd.Timedelta(hours=3)
    for h in pd.date_range(sym_start, sym_end, freq="h"):
        sym_rows.append({"pickup_zone": zone, "pickup_hour": h, "is_event_sym": 1})
    
    # Asymmetric: 2 hours before event_start to 4 hours after event_end
    asym_start = ev["event_start_dt"] - pd.Timedelta(hours=2)
    asym_end = ev["event_end_dt"] + pd.Timedelta(hours=4)
    for h in pd.date_range(asym_start, asym_end, freq="h"):
        asym_rows.append({"pickup_zone": zone, "pickup_hour": h, "is_event_asym": 1})

sym_df = pd.DataFrame(sym_rows).drop_duplicates(subset=["pickup_zone", "pickup_hour"])
asym_df = pd.DataFrame(asym_rows).drop_duplicates(subset=["pickup_zone", "pickup_hour"])

# Also build a richer flag: count of events in the (zone, hour) window
sym_count = pd.DataFrame(sym_rows).groupby(["pickup_zone", "pickup_hour"]).size().reset_index(name="event_count_sym")
asym_count = pd.DataFrame(asym_rows).groupby(["pickup_zone", "pickup_hour"]).size().reset_index(name="event_count_asym")

# Merge counts back
sym_df = sym_df.merge(sym_count, on=["pickup_zone", "pickup_hour"], how="left")
asym_df = asym_df.merge(asym_count, on=["pickup_zone", "pickup_hour"], how="left")

sym_df.to_parquet("data/processed/_event_flags_sym.parquet")
asym_df.to_parquet("data/processed/_event_flags_asym.parquet")
print(f"  Symmetric flag rows: {len(sym_df):,}")
print(f"  Asymmetric flag rows: {len(asym_df):,}")

# ============================================================
# Step 3: Update master table with new event flags
# ============================================================
print("\n[3/4] Updating master table with hour-level event flags...")
t0 = time.perf_counter()

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET temp_directory='data/processed/duckdb_temp'")

# Build new master table with hour-level flags joined
con.execute("""
    COPY (
        SELECT 
            m.* EXCLUDE (has_major_event, total_event_attendance),
            COALESCE(s.is_event_sym, 0) as is_event_sym,
            COALESCE(s.event_count_sym, 0) as event_count_sym,
            COALESCE(a.is_event_asym, 0) as is_event_asym,
            COALESCE(a.event_count_asym, 0) as event_count_asym,
            -- Keep the day-level flag for comparison/EDA
            m.has_major_event as has_major_event_dayflag,
            m.total_event_attendance
        FROM 'data/processed/master_zone_hour.parquet' m
        LEFT JOIN 'data/processed/_event_flags_sym.parquet' s
            ON m.pickup_zone = s.pickup_zone
            AND m.pickup_hour = s.pickup_hour
        LEFT JOIN 'data/processed/_event_flags_asym.parquet' a
            ON m.pickup_zone = a.pickup_zone
            AND m.pickup_hour = a.pickup_hour
    ) TO 'data/processed/master_zone_hour_v2.parquet' (FORMAT PARQUET)
""")

# Replace the original master table
import shutil
os.remove("data/processed/master_zone_hour.parquet")
shutil.move("data/processed/master_zone_hour_v2.parquet", "data/processed/master_zone_hour.parquet")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

# ============================================================
# Step 4: Verify
# ============================================================
print("\n[4/4] Verification...")
result = con.execute("""
    SELECT 
        COUNT(*) as rows,
        SUM(is_event_sym) as event_hours_sym,
        SUM(is_event_asym) as event_hours_asym,
        SUM(has_major_event_dayflag) as event_hours_dayflag,
        ROUND(100.0 * SUM(is_event_sym) / COUNT(*), 2) as pct_sym,
        ROUND(100.0 * SUM(is_event_asym) / COUNT(*), 2) as pct_asym,
        ROUND(100.0 * SUM(has_major_event_dayflag) / COUNT(*), 2) as pct_dayflag
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print("Event flag comparison:")
print(result.to_string(index=False))

print("\nFlag overlap (do hour-level flags fall within day-level flag?):")
overlap = con.execute("""
    SELECT 
        is_event_sym,
        has_major_event_dayflag,
        COUNT(*) as rows
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE pickup_zone IN (SELECT DISTINCT zone_id FROM (SELECT 161 as zone_id UNION SELECT 247 UNION SELECT 90))
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()
print(overlap.to_string(index=False))

con.close()

# Cleanup temp files
for tmp in ["_events_with_times.parquet", "_event_flags_sym.parquet", "_event_flags_asym.parquet"]:
    path = f"data/processed/{tmp}"
    if os.path.exists(path):
        os.remove(path)

total = time.perf_counter() - start
print(f"\n{'=' * 70}")
print(f"✅ EVENT FLAGS COMPLETE — {total/60:.1f} minutes total")
print(f"{'=' * 70}")
# %%

# %%
# ============================================================
# ADD NJ-VENUE OUTBOUND EVENT FLAGS
# Captures pre-game demand from Manhattan departure zones
# for events at MetLife, Prudential Center, UBS Arena, Red Bull Arena
# ============================================================
import os
os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")

import pandas as pd
import duckdb
import time
import shutil

print("=" * 70)
print("ADDING NJ-VENUE OUTBOUND EVENT FLAGS")
print("=" * 70)
start = time.perf_counter()

# ============================================================
# Step 1: Define per-venue departure zone clusters
# ============================================================
print("\n[1/4] Defining per-venue departure zone clusters...")

VENUE_DEPARTURE_ZONES = {
    "MetLife Stadium (NJ)": [
        186,  # Penn Station/Madison Sq West (NJ Transit terminus)
        230,  # Times Sq/Theatre District (Port Authority bus terminal)
        161,  # Midtown Center
        100,  # Garment District
        246,  # West Chelsea/Hudson Yards
        48,   # Clinton East (Hell's Kitchen south)
    ],
    "Prudential Center (NJ)": [
        261,  # World Trade Center (PATH terminus)
        87,   # Financial District North
        88,   # Financial District South
        231,  # TriBeCa/Civic Center
        125,  # Hudson Sq
        113,  # Greenwich Village North (PATH)
        114,  # Greenwich Village South (PATH)
        158,  # Meatpacking/West Village West (PATH)
        249,  # West Village (Christopher St PATH)
    ],
    "UBS Arena (Belmont)": [
        186,  # Penn Station (LIRR Hempstead branch)
        230,  # Times Sq area
        161,  # Midtown Center
    ],
    "Red Bull Arena": [
        261,  # WTC PATH
        87, 88,  # Financial District
        231,  # TriBeCa
        125,  # Hudson Sq
        113, 114,  # Greenwich Village
    ],
}

for venue, zones in VENUE_DEPARTURE_ZONES.items():
    print(f"  {venue}: {len(zones)} departure zones")

# ============================================================
# Step 2: Load NJ events and add start times
# ============================================================
print("\n[2/4] Building NJ event windows with departure zones...")

events = pd.read_csv("data/raw/events/major_events.csv")
events["date"] = pd.to_datetime(events["date"])
events["dow"] = events["date"].dt.dayofweek

# Filter to NJ events only (zone_id = -1)
nj_events = events[events["zone_id"] == -1].copy()
print(f"  NJ events: {len(nj_events)}")
print(f"  By venue:")
print(nj_events["venue"].value_counts().to_string())

# Determine start hour using same logic as before
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

# ============================================================
# Step 3: Expand each NJ event into (departure_zone, hour) flags
# Pre-game-only window: -4hr to -1hr before kickoff (sym)
#                      -5hr to -1hr before kickoff (asym - extended)
# ============================================================
print("\n[3/4] Expanding NJ events into hour-level flags on departure zones...")

nj_sym_rows = []   # 4hr pre to 1hr pre
nj_asym_rows = []  # 5hr pre to 1hr pre (extended)

for _, ev in nj_events.iterrows():
    venue = ev["venue"]
    if venue not in VENUE_DEPARTURE_ZONES:
        # Should never happen if our venue list is complete
        print(f"  ⚠️ Unknown venue: {venue}, skipping")
        continue
    
    departure_zones = VENUE_DEPARTURE_ZONES[venue]
    
    # Symmetric (pre-game only): 4hr before to 1hr before
    sym_start = ev["event_start_dt"] - pd.Timedelta(hours=4)
    sym_end = ev["event_start_dt"] - pd.Timedelta(hours=1)
    for zone in departure_zones:
        for h in pd.date_range(sym_start, sym_end, freq="h"):
            nj_sym_rows.append({
                "pickup_zone": zone,
                "pickup_hour": h,
                "is_nj_event_pregame_sym": 1,
                "venue": venue
            })
    
    # Asymmetric (extended pre-game): 5hr before to 1hr before
    asym_start = ev["event_start_dt"] - pd.Timedelta(hours=5)
    asym_end = ev["event_start_dt"] - pd.Timedelta(hours=1)
    for zone in departure_zones:
        for h in pd.date_range(asym_start, asym_end, freq="h"):
            nj_asym_rows.append({
                "pickup_zone": zone,
                "pickup_hour": h,
                "is_nj_event_pregame_asym": 1,
                "venue": venue
            })

# Build flag tables
nj_sym = pd.DataFrame(nj_sym_rows)
nj_asym = pd.DataFrame(nj_asym_rows)

# Count events per (zone, hour) — for handling overlapping NJ events
nj_sym_counts = nj_sym.groupby(["pickup_zone", "pickup_hour"]).size().reset_index(name="nj_event_count_sym")
nj_asym_counts = nj_asym.groupby(["pickup_zone", "pickup_hour"]).size().reset_index(name="nj_event_count_asym")

# Dedupe binary flags
nj_sym_flags = nj_sym[["pickup_zone", "pickup_hour", "is_nj_event_pregame_sym"]].drop_duplicates()
nj_asym_flags = nj_asym[["pickup_zone", "pickup_hour", "is_nj_event_pregame_asym"]].drop_duplicates()

# Merge counts back
nj_sym_final = nj_sym_flags.merge(nj_sym_counts, on=["pickup_zone", "pickup_hour"], how="left")
nj_asym_final = nj_asym_flags.merge(nj_asym_counts, on=["pickup_zone", "pickup_hour"], how="left")

nj_sym_final.to_parquet("data/processed/_nj_event_flags_sym.parquet")
nj_asym_final.to_parquet("data/processed/_nj_event_flags_asym.parquet")

print(f"  NJ symmetric flag rows (zone-hour): {len(nj_sym_final):,}")
print(f"  NJ asymmetric flag rows (zone-hour): {len(nj_asym_final):,}")
print(f"  Sample of zones flagged: {sorted(nj_sym_final['pickup_zone'].unique())[:10]}")

# ============================================================
# Step 4: Update master table with NJ flags + create combined flags
# ============================================================
print("\n[4/4] Joining NJ flags to master table...")
t0 = time.perf_counter()

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")
con.execute("SET temp_directory='data/processed/duckdb_temp'")

con.execute("""
    COPY (
        SELECT 
            m.*,
            COALESCE(s.is_nj_event_pregame_sym, 0) as is_nj_event_pregame_sym,
            COALESCE(s.nj_event_count_sym, 0) as nj_event_count_sym,
            COALESCE(a.is_nj_event_pregame_asym, 0) as is_nj_event_pregame_asym,
            COALESCE(a.nj_event_count_asym, 0) as nj_event_count_asym,
            -- Combined flags: NYC event OR NJ pre-game outbound
            CASE 
                WHEN m.is_event_sym = 1 OR COALESCE(s.is_nj_event_pregame_sym, 0) = 1 
                THEN 1 ELSE 0 
            END as is_event_combined_sym,
            CASE 
                WHEN m.is_event_asym = 1 OR COALESCE(a.is_nj_event_pregame_asym, 0) = 1 
                THEN 1 ELSE 0 
            END as is_event_combined_asym,
            -- Renamed for clarity
            m.is_event_sym as is_nyc_event_sym,
            m.is_event_asym as is_nyc_event_asym
        FROM 'data/processed/master_zone_hour.parquet' m
        LEFT JOIN 'data/processed/_nj_event_flags_sym.parquet' s
            ON m.pickup_zone = s.pickup_zone
            AND m.pickup_hour = s.pickup_hour
        LEFT JOIN 'data/processed/_nj_event_flags_asym.parquet' a
            ON m.pickup_zone = a.pickup_zone
            AND m.pickup_hour = a.pickup_hour
    ) TO 'data/processed/master_zone_hour_v3.parquet' (FORMAT PARQUET)
""")

os.remove("data/processed/master_zone_hour.parquet")
shutil.move("data/processed/master_zone_hour_v3.parquet", "data/processed/master_zone_hour.parquet")
print(f"  ⏱ {time.perf_counter() - t0:.1f}s")

# ============================================================
# Verification
# ============================================================
print("\nVerification: event flag distribution")
result = con.execute("""
    SELECT 
        COUNT(*) as total_rows,
        SUM(is_nyc_event_sym) as nyc_sym_hours,
        SUM(is_nyc_event_asym) as nyc_asym_hours,
        SUM(is_nj_event_pregame_sym) as nj_sym_hours,
        SUM(is_nj_event_pregame_asym) as nj_asym_hours,
        SUM(is_event_combined_sym) as combined_sym_hours,
        SUM(is_event_combined_asym) as combined_asym_hours
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))

# Confirm new column structure
print("\nFinal master table columns:")
schema = con.execute("DESCRIBE SELECT * FROM 'data/processed/master_zone_hour.parquet'").df()
print(schema[["column_name", "column_type"]].to_string(index=False))

con.close()

# Cleanup temp files
for tmp in ["_nj_event_flags_sym.parquet", "_nj_event_flags_asym.parquet"]:
    path = f"data/processed/{tmp}"
    if os.path.exists(path):
        os.remove(path)

total = time.perf_counter() - start
print(f"\n{'=' * 70}")
print(f"✅ NJ EVENT FLAGS COMPLETE — {total/60:.1f} minutes total")
print(f"{'=' * 70}")
print("""
Master table now has THREE event flag systems:
- NYC events (sym/asym): demand at venue zone, ±3hr or -2/+4hr
- NJ events pregame (sym/asym): demand at Manhattan departure zones, -4/-1hr
- Combined: union of both, for pooled analyses

Use whichever fits the analysis question.
""")
# %%

# %%
# ============================================================
# FINAL DATA QUALITY CHECK — last sanity pass before analysis
# ============================================================
import os
os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Uber")

import duckdb
import pandas as pd

print("=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

issues = []

# ============================================================
# CHECK 1: Master table integrity
# ============================================================
print("\n[1/10] Master table integrity")
result = duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        COUNT(DISTINCT pickup_zone) as zones,
        COUNT(DISTINCT platform) as platforms,
        MIN(pickup_hour) as min_hour,
        MAX(pickup_hour) as max_hour,
        COUNT(DISTINCT pickup_hour) as unique_hours
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))
expected_max_hours = 14640  # 610 days * 24 hours
actual_hours = int(result.iloc[0]["unique_hours"])
if actual_hours != expected_max_hours:
    issues.append(f"Expected {expected_max_hours} unique hours, got {actual_hours}")

# ============================================================
# CHECK 2: Primary key uniqueness
# ============================================================
print("\n[2/10] Primary key uniqueness (pickup_zone, pickup_hour, platform)")
result = duckdb.query("""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT (pickup_zone, pickup_hour, platform)) as unique_keys,
        COUNT(*) - COUNT(DISTINCT (pickup_zone, pickup_hour, platform)) as duplicates
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))
if result.iloc[0]["duplicates"] > 0:
    issues.append(f"Duplicate primary keys found: {result.iloc[0]['duplicates']}")

# ============================================================
# CHECK 3: Treatment indicators sanity
# ============================================================
print("\n[3/10] Treatment indicators (CRZ, post-CP)")
result = duckdb.query("""
    SELECT 
        is_in_crz,
        is_post_cp,
        COUNT(*) as rows,
        COUNT(DISTINCT pickup_zone) as zones
    FROM 'data/processed/master_zone_hour.parquet'
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()
print(result.to_string(index=False))

# Verify CRZ has exactly 36 zones
result = duckdb.query("""
    SELECT COUNT(DISTINCT pickup_zone) as crz_zones
    FROM 'data/processed/master_zone_hour.parquet'
    WHERE is_in_crz = 1
""").df()
crz_zones = int(result.iloc[0]["crz_zones"])
print(f"\nCRZ zones in data: {crz_zones} (expected 36)")
if crz_zones != 36:
    issues.append(f"CRZ zone count mismatch: {crz_zones} vs expected 36")

# ============================================================
# CHECK 4: Fare-per-mile sanity (ratio-of-sums)
# ============================================================
print("\n[4/10] Fare-per-mile distributions (ratio-of-sums)")
result = duckdb.query("""
    SELECT 
        platform,
        is_post_cp,
        is_in_crz,
        ROUND(SUM(total_adjusted_fare) / SUM(total_miles), 3) as ratio_of_sums_fpm,
        SUM(trip_count) as trips
    FROM 'data/processed/master_zone_hour.parquet'
    GROUP BY platform, is_post_cp, is_in_crz
    ORDER BY platform, is_post_cp, is_in_crz
""").df()
print(result.to_string(index=False))
# Expected: post-CP fare-per-mile in CRZ should be HIGHER than pre-CP
# (CRZ traffic reduction pushes prices up, supply constrained)

# ============================================================
# CHECK 5: Event flag overlap and counts
# ============================================================
print("\n[5/10] Event flag distribution")
result = duckdb.query("""
    SELECT 
        SUM(is_nyc_event_sym) as nyc_sym,
        SUM(is_nyc_event_asym) as nyc_asym,
        SUM(is_nj_event_pregame_sym) as nj_sym,
        SUM(is_nj_event_pregame_asym) as nj_asym,
        SUM(is_event_combined_sym) as combined_sym,
        SUM(is_event_combined_asym) as combined_asym,
        SUM(has_major_event_dayflag) as dayflag_legacy
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))

# Sanity: combined should equal NYC + NJ - overlap
expected_combined_sym = (result.iloc[0]["nyc_sym"] + result.iloc[0]["nj_sym"])
actual_combined_sym = result.iloc[0]["combined_sym"]
overlap_sym = expected_combined_sym - actual_combined_sym
print(f"\nNYC+NJ sym overlap: {overlap_sym} rows (expected small, since flag systems target different zones)")

# ============================================================
# CHECK 6: Weather coverage by zone
# ============================================================
print("\n[6/10] Weather coverage by zone")
result = duckdb.query("""
    SELECT 
        SUM(CASE WHEN temp_f IS NOT NULL THEN 1 ELSE 0 END) as has_weather,
        SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) as missing_weather,
        ROUND(100.0 * SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_missing
    FROM 'data/processed/master_zone_hour.parquet'
""").df()
print(result.to_string(index=False))
pct_missing_weather = float(result.iloc[0]["pct_missing"])
if pct_missing_weather > 5.0:
    issues.append(f"Weather missing in {pct_missing_weather}% of rows (>5% threshold)")

# ============================================================
# CHECK 7: Subway coverage by borough
# ============================================================
print("\n[7/10] Subway data coverage by borough")
result = duckdb.query("""
    SELECT 
        borough,
        COUNT(*) as rows,
        SUM(CASE WHEN subway_riders_zone > 0 THEN 1 ELSE 0 END) as rows_with_subway,
        ROUND(100.0 * SUM(CASE WHEN subway_riders_zone > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_with_subway,
        ROUND(AVG(CASE WHEN subway_riders_zone > 0 THEN subway_riders_zone END), 1) as avg_riders_when_present
    FROM 'data/processed/master_zone_hour.parquet'
    GROUP BY borough
    ORDER BY borough
""").df()
print(result.to_string(index=False))
# Expected: Manhattan and Brooklyn have high subway coverage; Staten Island near 0

# ============================================================
# CHECK 8: Trip volume sanity
# ============================================================
print("\n[8/10] Trip volume sanity")
result = duckdb.query("""
    SELECT 
        platform,
        COUNT(*) as zone_hours,
        SUM(trip_count) as total_trips,
        ROUND(AVG(trip_count), 1) as avg_trips_per_zone_hour,
        ROUND(MEDIAN(trip_count), 1) as median_trips_per_zone_hour,
        MAX(trip_count) as max_trips_per_zone_hour
    FROM 'data/processed/master_zone_hour.parquet'
    GROUP BY platform
""").df()
print(result.to_string(index=False))

# ============================================================
# CHECK 9: Time fixed effects coverage
# ============================================================
print("\n[9/10] Time variable distributions")
result = duckdb.query("""
    SELECT 
        dow,
        COUNT(DISTINCT pickup_date) as unique_dates,
        SUM(trip_count) as total_trips
    FROM 'data/processed/master_zone_hour.parquet'
    GROUP BY dow
    ORDER BY dow
""").df()
print("Day-of-week distribution:")
print(result.to_string(index=False))

result = duckdb.query("""
    SELECT 
        hour_of_day,
        ROUND(AVG(trip_count), 1) as avg_trips
    FROM 'data/processed/master_zone_hour.parquet'
    GROUP BY hour_of_day
    ORDER BY hour_of_day
""").df()
print("\nHour-of-day demand profile (avg trips per zone-hour):")
print(result.to_string(index=False))
# Expected: peak around 5-9pm, trough around 4-5am

# ============================================================
# CHECK 10: Zero trip counts and outliers
# ============================================================
print("\n[10/10] Trip count distribution (sanity check on outliers)")
result = duckdb.query("""
    SELECT 
        CASE 
            WHEN trip_count = 0 THEN '0'
            WHEN trip_count BETWEEN 1 AND 10 THEN '1-10'
            WHEN trip_count BETWEEN 11 AND 50 THEN '11-50'
            WHEN trip_count BETWEEN 51 AND 200 THEN '51-200'
            WHEN trip_count BETWEEN 201 AND 1000 THEN '201-1000'
            ELSE '1000+'
        END as bucket,
        COUNT(*) as rows
    FROM 'data/processed/master_zone_hour.parquet'
    GROUP BY 1
    ORDER BY MIN(trip_count)
""").df()
print(result.to_string(index=False))

# Top 5 highest trip-count zone-hours (sanity check the extreme)
result = duckdb.query("""
    SELECT 
        pickup_hour,
        pickup_zone,
        platform,
        trip_count,
        ROUND(total_adjusted_fare / NULLIF(total_miles, 0), 2) as fpm
    FROM 'data/processed/master_zone_hour.parquet'
    ORDER BY trip_count DESC
    LIMIT 5
""").df()
print("\nTop 5 highest-volume zone-hours:")
print(result.to_string(index=False))

# ============================================================
# FINAL VERDICT
# ============================================================
print("\n" + "=" * 70)
if issues:
    print(f"⚠️  {len(issues)} ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("✅ ALL CHECKS PASSED — data is ready for analysis")
print("=" * 70)
# %%
