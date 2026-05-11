# %%
import importlib
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(r"C:\Users\Thomas\OneDrive\Desktop\Uber")
os.chdir(PROJECT_ROOT)

print("=== ✅ Cell 1: Environment + Imports ===")
print(f"Working directory: {Path.cwd()}")

required_imports = ["duckdb", "pandas", "geopandas"]
loaded_modules = {}

for module_name in required_imports:
    try:
        loaded_modules[module_name] = importlib.import_module(module_name)
    except Exception as e:
        print(f"❌ Required import failed: {module_name} -> {e}")
        raise SystemExit(
            "Stopping data check: required dependency import failed. "
            "Install missing package(s) in your active venv and rerun Cell 1."
        ) from e

duckdb = loaded_modules["duckdb"]
pd = loaded_modules["pandas"]
gpd = loaded_modules["geopandas"]

print(f"duckdb version: {duckdb.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"geopandas version: {gpd.__version__}")

summary_rows = []


# %%
print("\n=== ✅ Cell 2: Non-TLC CSV checks ===")

csv_files = [
    PROJECT_ROOT / "data/raw/tlc_zones/taxi_zone_lookup.csv",
    PROJECT_ROOT / "data/raw/mta/mta_subway_2024.csv",
    PROJECT_ROOT / "data/raw/mta/mta_subway_2025.csv",
    PROJECT_ROOT / "data/raw/noaa/NOAALocalClimatologicalData.csv",
    PROJECT_ROOT / "data/raw/noaa/storm_data_search_results.csv",
    PROJECT_ROOT / "data/raw/events/major_events.csv",
    PROJECT_ROOT / "data/raw/events/NYC_Permitted_Event_Information_20260507.csv",
]

for file_path in csv_files:
    file_label = str(file_path.relative_to(PROJECT_ROOT))
    section_start = time.perf_counter()
    status = "✅"
    row_count = None
    print(f"\n--- 📄 {file_label} ---")

    try:
        size_start = time.perf_counter()
        size_mb = file_path.stat().st_size / (1024 * 1024)
        size_elapsed = time.perf_counter() - size_start
        print(f"✅ Size: {size_mb:.2f} MB")
        print(f"⏱ size check: {size_elapsed:.3f}s")

        sample_start = time.perf_counter()
        sample_df = pd.read_csv(file_path, nrows=5)
        sample_elapsed = time.perf_counter() - sample_start
        print(f"✅ Columns ({len(sample_df.columns)}): {list(sample_df.columns)}")
        print("✅ Dtypes:")
        print(sample_df.dtypes)
        print(f"⏱ pandas.read_csv(nrows=5): {sample_elapsed:.3f}s")

        count_start = time.perf_counter()
        normalized_file_path = str(file_path).replace("\\", "/")
        with duckdb.connect() as con:
            row_count = con.execute(
                f"SELECT COUNT(*) FROM '{normalized_file_path}'"
            ).fetchone()[0]
        count_elapsed = time.perf_counter() - count_start
        print(f"✅ Row count (DuckDB): {row_count:,}")
        print(f"⏱ duckdb row count: {count_elapsed:.3f}s")

        preview_start = time.perf_counter()
        print("✅ First 3 rows (transposed):")
        print(sample_df.head(3).T)
        preview_elapsed = time.perf_counter() - preview_start
        print(f"⏱ preview print: {preview_elapsed:.3f}s")

    except Exception as e:
        status = "❌"
        print(f"❌ Error checking {file_label}: {e}")

    total_elapsed = time.perf_counter() - section_start
    print(f"⏱ total file check time: {total_elapsed:.3f}s")
    summary_rows.append(
        {
            "file_name": file_label,
            "status": status,
            "row_count": row_count,
            "time_to_load": round(total_elapsed, 3),
        }
    )


# %%
print("\n=== ✅ Cell 3: TLC Parquet checks (DuckDB only) ===")
tlc_glob = str((PROJECT_ROOT / "data/raw/tlc/*.parquet").as_posix())

try:
    with duckdb.connect() as con:
        q1_start = time.perf_counter()
        tlc_stats = con.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                MIN(pickup_datetime) AS earliest,
                MAX(pickup_datetime) AS latest
            FROM read_parquet('{tlc_glob}', union_by_name=True)
            """
        ).df()
        q1_elapsed = time.perf_counter() - q1_start
        print("✅ TLC full range stats:")
        print(tlc_stats)
        print(f"⏱ stats query: {q1_elapsed:.3f}s")

        q2_start = time.perf_counter()
        platform_breakdown = con.execute(
            f"""
            SELECT hvfhs_license_num, COUNT(*) AS rows
            FROM read_parquet('{tlc_glob}', union_by_name=True)
            GROUP BY 1
            ORDER BY rows DESC
            """
        ).df()
        q2_elapsed = time.perf_counter() - q2_start
        print("✅ Platform breakdown:")
        print(platform_breakdown)
        print(f"⏱ platform breakdown query: {q2_elapsed:.3f}s")

        jan_2024_path = str((PROJECT_ROOT / "data/raw/tlc/fhvhv_tripdata_2024-01.parquet").as_posix())
        jan_2025_path = str((PROJECT_ROOT / "data/raw/tlc/fhvhv_tripdata_2025-01.parquet").as_posix())

        q3_start = time.perf_counter()
        jan_2024_cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{jan_2024_path}')"
        ).df()["column_name"].tolist()
        q3_elapsed = time.perf_counter() - q3_start
        print(f"✅ Jan 2024 column count: {len(jan_2024_cols)}")
        print(jan_2024_cols)
        print(f"⏱ Jan 2024 schema query: {q3_elapsed:.3f}s")

        q4_start = time.perf_counter()
        jan_2025_cols = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{jan_2025_path}')"
        ).df()["column_name"].tolist()
        q4_elapsed = time.perf_counter() - q4_start
        print(f"✅ Jan 2025 column count: {len(jan_2025_cols)}")
        print(jan_2025_cols)
        print(f"⏱ Jan 2025 schema query: {q4_elapsed:.3f}s")

    summary_rows.append(
        {
            "file_name": "data/raw/tlc/*.parquet",
            "status": "✅",
            "row_count": int(tlc_stats.loc[0, "rows"]),
            "time_to_load": round(q1_elapsed + q2_elapsed + q3_elapsed + q4_elapsed, 3),
        }
    )
except Exception as e:
    print(f"❌ TLC parquet check failed: {e}")
    summary_rows.append(
        {
            "file_name": "data/raw/tlc/*.parquet",
            "status": "❌",
            "row_count": None,
            "time_to_load": None,
        }
    )


# %%
print("\n=== ✅ Cell 4: Taxi zones shapefile check ===")
shp_path = PROJECT_ROOT / "data/raw/tlc_zones/taxi_zones.shp"
shp_start = time.perf_counter()
shp_status = "✅"
shp_rows = None

try:
    zones_gdf = gpd.read_file(shp_path)
    shp_rows = len(zones_gdf)
    print(f"✅ Number of zones: {shp_rows}")
    print(f"✅ CRS: {zones_gdf.crs}")
    print(f"✅ Columns: {list(zones_gdf.columns)}")
    print("✅ First 3 rows:")
    print(zones_gdf.head(3))
except Exception as e:
    shp_status = "❌"
    print(f"❌ Shapefile check failed: {e}")

shp_elapsed = time.perf_counter() - shp_start
print(f"⏱ shapefile check time: {shp_elapsed:.3f}s")
summary_rows.append(
    {
        "file_name": str(shp_path.relative_to(PROJECT_ROOT)),
        "status": shp_status,
        "row_count": shp_rows,
        "time_to_load": round(shp_elapsed, 3),
    }
)


# %%
print("\n=== ✅ Cell 5: Summary ===")
summary_df = pd.DataFrame(summary_rows, columns=["file_name", "status", "row_count", "time_to_load"])
print(summary_df.to_string(index=False))

# %%
# Verify MTA subway data structure
import duckdb

print("=== MTA 2024 inspection ===")
result = duckdb.query("""
    SELECT 
        MIN(transit_timestamp) as earliest,
        MAX(transit_timestamp) as latest,
        COUNT(*) as total_rows,
        COUNT(DISTINCT transit_timestamp) as unique_timestamps,
        COUNT(DISTINCT station_complex) as unique_stations
    FROM 'data/raw/mta/mta_subway_2024.csv'
""").df()
print(result.to_string())

# Check row multiplier per (timestamp, station)
print("\n=== Rows per (timestamp, station) for 2024 ===")
result = duckdb.query("""
    SELECT rows_per_pair, COUNT(*) as count
    FROM (
        SELECT COUNT(*) as rows_per_pair
        FROM 'data/raw/mta/mta_subway_2024.csv'
        GROUP BY transit_timestamp, station_complex
    ) t
    GROUP BY rows_per_pair
    ORDER BY count DESC
    LIMIT 10
""").df()
print(result.to_string())

print("\n=== MTA 2025 inspection ===")
result = duckdb.query("""
    SELECT 
        MIN(transit_timestamp) as earliest,
        MAX(transit_timestamp) as latest,
        COUNT(*) as total_rows,
        COUNT(DISTINCT transit_timestamp) as unique_timestamps,
        COUNT(DISTINCT station_complex) as unique_stations
    FROM 'data/raw/mta/mta_subway_2025.csv'
""").df()
print(result.to_string())

# Check if 2025 is actually filtered to your window or has extra data
print("\n=== Date distribution for 2025 ===")
result = duckdb.query("""
    SELECT 
        DATE_TRUNC('month', CAST(transit_timestamp AS TIMESTAMP)) as month,
        COUNT(*) as rows
    FROM 'data/raw/mta/mta_subway_2025.csv'
    GROUP BY 1
    ORDER BY 1
""").df()
print(result.to_string())
# %%

# %%
import duckdb
import os

os.makedirs("data/processed", exist_ok=True)

print("Aggregating MTA subway data...")

duckdb.query("""
    COPY (
        SELECT 
            CAST(transit_timestamp AS TIMESTAMP) as timestamp,
            station_complex,
            borough,
            SUM(ridership) as ridership
        FROM (
            SELECT transit_timestamp, station_complex, borough, ridership
            FROM 'data/raw/mta/mta_subway_2024.csv'
            UNION ALL
            SELECT transit_timestamp, station_complex, borough, ridership
            FROM 'data/raw/mta/mta_subway_2025.csv'
        )
        WHERE CAST(transit_timestamp AS TIMESTAMP) >= '2024-01-01'
          AND CAST(transit_timestamp AS TIMESTAMP) < '2025-09-01'
        GROUP BY transit_timestamp, station_complex, borough
    ) TO 'data/processed/mta_subway_clean.parquet' (FORMAT PARQUET)
""")

result = duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT station_complex) as stations
    FROM 'data/processed/mta_subway_clean.parquet'
""").df()
print(result.to_string())

# %%
# Build station coordinate lookup by parsing Georeference WKT field
print("\nBuilding station coordinate lookup...")
duckdb.query("""
    COPY (
        SELECT 
            station_complex,
            -- Parse "POINT (lon lat)" format
            CAST(SPLIT_PART(REGEXP_EXTRACT(Georeference, 'POINT \\(([^)]+)\\)', 1), ' ', 1) AS DOUBLE) as longitude,
            CAST(SPLIT_PART(REGEXP_EXTRACT(Georeference, 'POINT \\(([^)]+)\\)', 1), ' ', 2) AS DOUBLE) as latitude
        FROM (
            SELECT DISTINCT station_complex, Georeference
            FROM 'data/raw/mta/mta_subway_2025.csv'
            WHERE Georeference IS NOT NULL
        )
        GROUP BY station_complex, Georeference
    ) TO 'data/processed/mta_station_coords.parquet' (FORMAT PARQUET)
""")

result = duckdb.query("""
    SELECT 
        COUNT(*) as stations,
        MIN(latitude) as min_lat, MAX(latitude) as max_lat,
        MIN(longitude) as min_lon, MAX(longitude) as max_lon
    FROM 'data/processed/mta_station_coords.parquet'
""").df()
print(result.to_string())

# %%
# Check schemas of both files
import duckdb

print("=== 2024 columns ===")
print(duckdb.query("SELECT * FROM 'data/raw/mta/mta_subway_2024.csv' LIMIT 0").df().columns.tolist())

print("\n=== 2025 columns ===")
print(duckdb.query("SELECT * FROM 'data/raw/mta/mta_subway_2025.csv' LIMIT 0").df().columns.tolist())
# %%

# %%
import duckdb
import os

os.makedirs("data/processed", exist_ok=True)

print("Aggregating MTA subway data...")

duckdb.query("""
    COPY (
        SELECT 
            CAST(transit_timestamp AS TIMESTAMP) as timestamp,
            station_complex,
            borough,
            SUM(ridership) as ridership
        FROM (
            SELECT transit_timestamp, station_complex, borough, ridership
            FROM 'data/raw/mta/mta_subway_2024.csv'
            UNION ALL
            SELECT transit_timestamp, station_complex, borough, ridership
            FROM 'data/raw/mta/mta_subway_2025.csv'
        )
        WHERE CAST(transit_timestamp AS TIMESTAMP) >= '2024-01-01'
          AND CAST(transit_timestamp AS TIMESTAMP) < '2025-09-01'
        GROUP BY transit_timestamp, station_complex, borough
    ) TO 'data/processed/mta_subway_clean.parquet' (FORMAT PARQUET)
""")

result = duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT station_complex) as stations
    FROM 'data/processed/mta_subway_clean.parquet'
""").df()
print(result.to_string())

# %%
# Build station coordinate lookup by parsing Georeference WKT field
print("\nBuilding station coordinate lookup...")
duckdb.query("""
    COPY (
        SELECT 
            station_complex,
            -- Parse "POINT (lon lat)" format
            CAST(SPLIT_PART(REGEXP_EXTRACT(Georeference, 'POINT \\(([^)]+)\\)', 1), ' ', 1) AS DOUBLE) as longitude,
            CAST(SPLIT_PART(REGEXP_EXTRACT(Georeference, 'POINT \\(([^)]+)\\)', 1), ' ', 2) AS DOUBLE) as latitude
        FROM (
            SELECT DISTINCT station_complex, Georeference
            FROM 'data/raw/mta/mta_subway_2025.csv'
            WHERE Georeference IS NOT NULL
        )
        GROUP BY station_complex, Georeference
    ) TO 'data/processed/mta_station_coords.parquet' (FORMAT PARQUET)
""")

result = duckdb.query("""
    SELECT 
        COUNT(*) as stations,
        MIN(latitude) as min_lat, MAX(latitude) as max_lat,
        MIN(longitude) as min_lon, MAX(longitude) as max_lon
    FROM 'data/processed/mta_station_coords.parquet'
""").df()
print(result.to_string())
# %%

# %%
import os
import duckdb

# Check file was created
path = "data/processed/mta_subway_clean.parquet"
if os.path.exists(path):
    size_mb = os.path.getsize(path) / 1e6
    print(f"✅ File exists: {size_mb:.1f} MB")
else:
    print(f"❌ File not found: {path}")

# Check contents
result = duckdb.query(f"""
    SELECT 
        COUNT(*) as rows,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT station_complex) as stations
    FROM '{path}'
""").df()
print(result.to_string())
# %%

# %%
import duckdb
import os

os.makedirs("data/processed", exist_ok=True)

print("Aggregating MTA subway data with explicit numeric cast...")

duckdb.query("""
    COPY (
        SELECT 
            CAST(transit_timestamp AS TIMESTAMP) as timestamp,
            station_complex,
            borough,
            SUM(CAST(ridership AS DOUBLE)) as ridership
        FROM (
            SELECT transit_timestamp, station_complex, borough, ridership
            FROM 'data/raw/mta/mta_subway_2024.csv'
            UNION ALL
            SELECT transit_timestamp, station_complex, borough, ridership
            FROM 'data/raw/mta/mta_subway_2025.csv'
        )
        WHERE CAST(transit_timestamp AS TIMESTAMP) >= '2024-01-01'
          AND CAST(transit_timestamp AS TIMESTAMP) < '2025-09-01'
        GROUP BY transit_timestamp, station_complex, borough
    ) TO 'data/processed/mta_subway_clean.parquet' (FORMAT PARQUET)
""")

result = duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT station_complex) as stations,
        ROUND(SUM(ridership)/1e6, 1) as total_riders_millions
    FROM 'data/processed/mta_subway_clean.parquet'
""").df()
print(result.to_string())
# %%

# %%
import duckdb
import os

os.makedirs("data/processed", exist_ok=True)

print("Aggregating MTA subway data...")

duckdb.query("""
    COPY (
        SELECT 
            CAST(transit_timestamp AS TIMESTAMP) as timestamp,
            station_complex,
            borough,
            SUM(CAST(REPLACE(ridership, ',', '') AS DOUBLE)) as ridership
        FROM (
            SELECT transit_timestamp, station_complex, borough, ridership
            FROM 'data/raw/mta/mta_subway_2024.csv'
            UNION ALL
            SELECT transit_timestamp, station_complex, borough, ridership
            FROM 'data/raw/mta/mta_subway_2025.csv'
        )
        WHERE CAST(transit_timestamp AS TIMESTAMP) >= '2024-01-01'
          AND CAST(transit_timestamp AS TIMESTAMP) < '2025-09-01'
        GROUP BY transit_timestamp, station_complex, borough
    ) TO 'data/processed/mta_subway_clean.parquet' (FORMAT PARQUET)
""")

result = duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT station_complex) as stations,
        ROUND(SUM(ridership)/1e6, 1) as total_riders_millions
    FROM 'data/processed/mta_subway_clean.parquet'
""").df()
print(result.to_string())
# %%

# %%
import duckdb
import os
import time

os.makedirs("data/processed", exist_ok=True)

print("Aggregating TLC trips to zone-hour grain...")
start = time.perf_counter()

duckdb.query("""
    COPY (
        WITH cleaned AS (
            SELECT
                CASE 
                    WHEN hvfhs_license_num = 'HV0003' THEN 'uber'
                    WHEN hvfhs_license_num = 'HV0005' THEN 'lyft'
                    ELSE 'other'
                END as platform,
                DATE_TRUNC('hour', pickup_datetime) as pickup_hour,
                PULocationID as pickup_zone,
                DOLocationID as dropoff_zone,
                trip_miles,
                trip_time,
                base_passenger_fare,
                congestion_surcharge,
                COALESCE(cbd_congestion_fee, 0) as cbd_congestion_fee,
                tolls,
                airport_fee,
                driver_pay,
                shared_match_flag,
                -- Surge proxy: subtract congestion fees BEFORE computing fare per mile
                (base_passenger_fare - congestion_surcharge - COALESCE(cbd_congestion_fee, 0)) / NULLIF(trip_miles, 0) as fare_per_mile_clean
            FROM read_parquet('data/raw/tlc/*.parquet', union_by_name=True)
            WHERE 
                trip_miles > 0 AND trip_miles < 100
                AND trip_time > 60 AND trip_time < 10800
                AND base_passenger_fare > 0
                AND PULocationID IS NOT NULL AND PULocationID <= 263
                AND DOLocationID IS NOT NULL AND DOLocationID <= 263
                AND hvfhs_license_num IN ('HV0003', 'HV0005')
        )
        SELECT
            pickup_hour,
            pickup_zone,
            platform,
            COUNT(*) as trip_count,
            AVG(fare_per_mile_clean) as mean_fare_per_mile,
            MEDIAN(fare_per_mile_clean) as median_fare_per_mile,
            QUANTILE_CONT(fare_per_mile_clean, 0.75) as p75_fare_per_mile,
            AVG(trip_miles) as mean_trip_miles,
            AVG(trip_time) as mean_trip_time,
            SUM(base_passenger_fare) as total_base_fare,
            SUM(driver_pay) as total_driver_pay,
            AVG(CASE WHEN shared_match_flag = 'Y' THEN 1.0 ELSE 0.0 END) as pct_shared,
            AVG(CASE WHEN airport_fee > 0 THEN 1.0 ELSE 0.0 END) as pct_airport
        FROM cleaned
        GROUP BY pickup_hour, pickup_zone, platform
    ) TO 'data/processed/tlc_zone_hour.parquet' (FORMAT PARQUET)
""")

elapsed = time.perf_counter() - start
print(f"⏱ Aggregation took {elapsed/60:.1f} minutes")

# Verify
result = duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(pickup_hour) as earliest,
        MAX(pickup_hour) as latest,
        COUNT(DISTINCT pickup_zone) as zones,
        COUNT(DISTINCT platform) as platforms,
        SUM(trip_count) as total_trips
    FROM 'data/processed/tlc_zone_hour.parquet'
""").df()
print(result.to_string())

# Platform breakdown
result = duckdb.query("""
    SELECT platform, COUNT(*) as zone_hours, SUM(trip_count) as trips
    FROM 'data/processed/tlc_zone_hour.parquet'
    GROUP BY platform
""").df()
print("\nPlatform breakdown:")
print(result.to_string())
# %%

# %%
import duckdb
import os
import time
import glob

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/processed/duckdb_temp", exist_ok=True)

print("Aggregating TLC trips to zone-hour grain (one file at a time)...")
overall_start = time.perf_counter()

# Get all monthly files
files = sorted(glob.glob("data/raw/tlc/*.parquet"))
print(f"Processing {len(files)} files")

# Delete any old output to start fresh
output_path = "data/processed/tlc_zone_hour.parquet"
if os.path.exists(output_path):
    os.remove(output_path)

# Process each file separately, writing to its own temp parquet
temp_files = []
for i, file_path in enumerate(files, 1):
    file_start = time.perf_counter()
    file_name = os.path.basename(file_path)
    temp_out = f"data/processed/_tlc_temp_{i:02d}.parquet"
    
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET temp_directory='data/processed/duckdb_temp'")
    
    con.execute(f"""
        COPY (
            WITH cleaned AS (
                SELECT
                    CASE 
                        WHEN hvfhs_license_num = 'HV0003' THEN 'uber'
                        WHEN hvfhs_license_num = 'HV0005' THEN 'lyft'
                        ELSE 'other'
                    END as platform,
                    DATE_TRUNC('hour', pickup_datetime) as pickup_hour,
                    PULocationID as pickup_zone,
                    trip_miles,
                    trip_time,
                    base_passenger_fare,
                    congestion_surcharge,
                    COALESCE(TRY_CAST(cbd_congestion_fee AS DOUBLE), 0) as cbd_fee,
                    airport_fee,
                    driver_pay,
                    shared_match_flag,
                    (base_passenger_fare - congestion_surcharge - COALESCE(TRY_CAST(cbd_congestion_fee AS DOUBLE), 0)) / NULLIF(trip_miles, 0) as fare_per_mile_clean
                FROM read_parquet('{file_path}', union_by_name=True)
                WHERE 
                    trip_miles > 0 AND trip_miles < 100
                    AND trip_time > 60 AND trip_time < 10800
                    AND base_passenger_fare > 0
                    AND PULocationID IS NOT NULL AND PULocationID <= 263
                    AND hvfhs_license_num IN ('HV0003', 'HV0005')
            )
            SELECT
                pickup_hour,
                pickup_zone,
                platform,
                COUNT(*) as trip_count,
                AVG(fare_per_mile_clean) as mean_fare_per_mile,
                MEDIAN(fare_per_mile_clean) as median_fare_per_mile,
                QUANTILE_CONT(fare_per_mile_clean, 0.75) as p75_fare_per_mile,
                AVG(trip_miles) as mean_trip_miles,
                AVG(trip_time) as mean_trip_time,
                SUM(base_passenger_fare) as total_base_fare,
                SUM(driver_pay) as total_driver_pay,
                AVG(CASE WHEN shared_match_flag = 'Y' THEN 1.0 ELSE 0.0 END) as pct_shared,
                AVG(CASE WHEN airport_fee > 0 THEN 1.0 ELSE 0.0 END) as pct_airport
            FROM cleaned
            GROUP BY pickup_hour, pickup_zone, platform
        ) TO '{temp_out}' (FORMAT PARQUET)
    """)
    con.close()
    
    temp_files.append(temp_out)
    elapsed = time.perf_counter() - file_start
    print(f"  [{i:02d}/{len(files)}] {file_name}: {elapsed:.1f}s")

# Combine all temp files into final output
print("\nCombining monthly aggregations...")
con = duckdb.connect()
con.execute(f"""
    COPY (
        SELECT * FROM read_parquet('data/processed/_tlc_temp_*.parquet')
    ) TO '{output_path}' (FORMAT PARQUET)
""")
con.close()

# Clean up temp files
for f in temp_files:
    os.remove(f)

total_elapsed = time.perf_counter() - overall_start
print(f"\n⏱ Total time: {total_elapsed/60:.1f} minutes")

# Verify
result = duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(pickup_hour) as earliest,
        MAX(pickup_hour) as latest,
        COUNT(DISTINCT pickup_zone) as zones,
        COUNT(DISTINCT platform) as platforms,
        SUM(trip_count) as total_trips
    FROM 'data/processed/tlc_zone_hour.parquet'
""").df()
print("\nFinal output:")
print(result.to_string())

result = duckdb.query("""
    SELECT platform, COUNT(*) as zone_hours, SUM(trip_count) as trips
    FROM 'data/processed/tlc_zone_hour.parquet'
    GROUP BY platform
""").df()
print("\nPlatform breakdown:")
print(result.to_string())
# %%

# %%
import duckdb
import os
import time
import glob

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/processed/duckdb_temp", exist_ok=True)

print("Aggregating TLC trips to zone-hour grain (one file at a time)...")
overall_start = time.perf_counter()

files = sorted(glob.glob("data/raw/tlc/*.parquet"))
print(f"Processing {len(files)} files")

output_path = "data/processed/tlc_zone_hour.parquet"
if os.path.exists(output_path):
    os.remove(output_path)

temp_files = []
for i, file_path in enumerate(files, 1):
    file_start = time.perf_counter()
    file_name = os.path.basename(file_path)
    temp_out = f"data/processed/_tlc_temp_{i:02d}.parquet"
    
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET temp_directory='data/processed/duckdb_temp'")
    
    # Detect whether this file has cbd_congestion_fee
    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}')").df()["column_name"].tolist()
    has_cbd = 'cbd_congestion_fee' in cols
    cbd_expr = "COALESCE(cbd_congestion_fee, 0)" if has_cbd else "0"
    
    con.execute(f"""
        COPY (
            WITH cleaned AS (
                SELECT
                    CASE 
                        WHEN hvfhs_license_num = 'HV0003' THEN 'uber'
                        WHEN hvfhs_license_num = 'HV0005' THEN 'lyft'
                        ELSE 'other'
                    END as platform,
                    DATE_TRUNC('hour', pickup_datetime) as pickup_hour,
                    PULocationID as pickup_zone,
                    trip_miles,
                    trip_time,
                    base_passenger_fare,
                    congestion_surcharge,
                    {cbd_expr} as cbd_fee,
                    airport_fee,
                    driver_pay,
                    shared_match_flag,
                    (base_passenger_fare - congestion_surcharge - {cbd_expr}) / NULLIF(trip_miles, 0) as fare_per_mile_clean
                FROM read_parquet('{file_path}')
                WHERE 
                    trip_miles > 0 AND trip_miles < 100
                    AND trip_time > 60 AND trip_time < 10800
                    AND base_passenger_fare > 0
                    AND PULocationID IS NOT NULL AND PULocationID <= 263
                    AND hvfhs_license_num IN ('HV0003', 'HV0005')
            )
            SELECT
                pickup_hour,
                pickup_zone,
                platform,
                COUNT(*) as trip_count,
                AVG(fare_per_mile_clean) as mean_fare_per_mile,
                MEDIAN(fare_per_mile_clean) as median_fare_per_mile,
                QUANTILE_CONT(fare_per_mile_clean, 0.75) as p75_fare_per_mile,
                AVG(trip_miles) as mean_trip_miles,
                AVG(trip_time) as mean_trip_time,
                SUM(base_passenger_fare) as total_base_fare,
                SUM(driver_pay) as total_driver_pay,
                AVG(CASE WHEN shared_match_flag = 'Y' THEN 1.0 ELSE 0.0 END) as pct_shared,
                AVG(CASE WHEN airport_fee > 0 THEN 1.0 ELSE 0.0 END) as pct_airport
            FROM cleaned
            GROUP BY pickup_hour, pickup_zone, platform
        ) TO '{temp_out}' (FORMAT PARQUET)
    """)
    con.close()
    
    temp_files.append(temp_out)
    elapsed = time.perf_counter() - file_start
    cbd_status = "CBD" if has_cbd else "no-CBD"
    print(f"  [{i:02d}/{len(files)}] {file_name} [{cbd_status}]: {elapsed:.1f}s")

# Combine all temp files into final output
print("\nCombining monthly aggregations...")
con = duckdb.connect()
con.execute(f"""
    COPY (
        SELECT * FROM read_parquet('data/processed/_tlc_temp_*.parquet')
    ) TO '{output_path}' (FORMAT PARQUET)
""")
con.close()

for f in temp_files:
    os.remove(f)

total_elapsed = time.perf_counter() - overall_start
print(f"\n⏱ Total time: {total_elapsed/60:.1f} minutes")

result = duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(pickup_hour) as earliest,
        MAX(pickup_hour) as latest,
        COUNT(DISTINCT pickup_zone) as zones,
        COUNT(DISTINCT platform) as platforms,
        SUM(trip_count) as total_trips
    FROM 'data/processed/tlc_zone_hour.parquet'
""").df()
print("\nFinal output:")
print(result.to_string())

result = duckdb.query("""
    SELECT platform, COUNT(*) as zone_hours, SUM(trip_count) as trips
    FROM 'data/processed/tlc_zone_hour.parquet'
    GROUP BY platform
""").df()
print("\nPlatform breakdown:")
print(result.to_string())
# %%

# %%
import pandas as pd
import geopandas as gpd

# Check the lookup CSV
print("=== taxi_zone_lookup.csv ===")
lookup = pd.read_csv("data/raw/tlc_zones/taxi_zone_lookup.csv")
print(f"Columns: {lookup.columns.tolist()}")
print(f"Rows: {len(lookup)}")
print("\nFirst 5 Manhattan rows:")
print(lookup[lookup["Borough"] == "Manhattan"].head().to_string())

# Check the shapefile (much richer data)
print("\n\n=== taxi_zones.shp ===")
zones = gpd.read_file("data/raw/tlc_zones/taxi_zones.shp")
print(f"Columns: {zones.columns.tolist()}")
print(f"CRS: {zones.crs}")
print(f"Rows: {len(zones)}")
print("\nFirst 3 Manhattan rows:")
print(zones[zones["borough"] == "Manhattan"].head(3).to_string())
# %%

# %%
import geopandas as gpd
import pandas as pd

# Load shapefile and reproject to lat/lon (EPSG:4326)
zones = gpd.read_file("data/raw/tlc_zones/taxi_zones.shp").to_crs(epsg=4326)

# Filter to Manhattan
manhattan = zones[zones["borough"] == "Manhattan"].copy()

# Compute centroid latitude AND southern boundary (min lat) for each zone
manhattan["centroid_lat"] = manhattan.geometry.centroid.y
manhattan["centroid_lon"] = manhattan.geometry.centroid.x
manhattan["south_lat"] = manhattan.geometry.bounds["miny"]
manhattan["north_lat"] = manhattan.geometry.bounds["maxy"]

# 60th Street is approximately 40.7625 latitude
CRZ_CUTOFF = 40.7625

# Three classification flags for review
manhattan["centroid_below_60th"] = manhattan["centroid_lat"] <= CRZ_CUTOFF
manhattan["entirely_below_60th"] = manhattan["north_lat"] <= CRZ_CUTOFF
manhattan["entirely_above_60th"] = manhattan["south_lat"] > CRZ_CUTOFF

# Sort by centroid lat descending — north to south
manhattan_sorted = manhattan[[
    "LocationID", "zone", "centroid_lat", "south_lat", "north_lat",
    "centroid_below_60th", "entirely_below_60th", "entirely_above_60th"
]].sort_values("centroid_lat", ascending=False)

# Print full list
print("=== All Manhattan zones, sorted north to south ===")
print(manhattan_sorted.to_string(index=False))

# Print boundary cases (zones that straddle 60th Street)
print("\n=== BOUNDARY ZONES (straddle 60th Street) — verify these manually ===")
boundary = manhattan_sorted[
    (manhattan_sorted["south_lat"] <= CRZ_CUTOFF) & 
    (manhattan_sorted["north_lat"] > CRZ_CUTOFF)
]
print(boundary.to_string(index=False))

# Save full output for review
manhattan_sorted.to_csv("data/processed/manhattan_zones_classified.csv", index=False)
print(f"\nSaved to data/processed/manhattan_zones_classified.csv")
print(f"\nProvisional CRZ count (centroid below 60th): {manhattan['centroid_below_60th'].sum()}")
print(f"Boundary zones needing review: {len(boundary)}")
# %%

# %%
import geopandas as gpd
import pandas as pd

# Load shapefile in original projected CRS
zones = gpd.read_file("data/raw/tlc_zones/taxi_zones.shp")

# Compute centroid in projected CRS first (correct)
manhattan = zones[zones["borough"] == "Manhattan"].copy()
manhattan["centroid_proj"] = manhattan.geometry.centroid

# Now reproject ONLY the centroids (and bounds) to lat/lon
manhattan["centroid_latlon"] = manhattan.set_geometry("centroid_proj").to_crs(epsg=4326).geometry
manhattan["centroid_lat"] = manhattan["centroid_latlon"].y
manhattan["centroid_lon"] = manhattan["centroid_latlon"].x

# Reproject full zones to lat/lon for bounds
manhattan_latlon = manhattan.to_crs(epsg=4326)
manhattan["south_lat"] = manhattan_latlon.geometry.bounds["miny"]
manhattan["north_lat"] = manhattan_latlon.geometry.bounds["maxy"]

CRZ_CUTOFF = 40.7625

manhattan["centroid_below_60th"] = manhattan["centroid_lat"] <= CRZ_CUTOFF
manhattan["entirely_below_60th"] = manhattan["north_lat"] <= CRZ_CUTOFF
manhattan["entirely_above_60th"] = manhattan["south_lat"] > CRZ_CUTOFF

cols = ["LocationID", "zone", "centroid_lat", "south_lat", "north_lat",
        "centroid_below_60th", "entirely_below_60th", "entirely_above_60th"]
result = manhattan[cols].sort_values("centroid_lat", ascending=False)

# Save full output
result.to_csv("data/processed/manhattan_zones_classified.csv", index=False)

# Print summary counts
print(f"Total Manhattan zones: {len(result)}")
print(f"Entirely above 60th (NOT CRZ): {result['entirely_above_60th'].sum()}")
print(f"Entirely below 60th (CRZ): {result['entirely_below_60th'].sum()}")
print(f"Boundary (straddle 60th): {(~result['entirely_above_60th'] & ~result['entirely_below_60th']).sum()}")

# Print zones IN CRZ (centroid below 60th)
print("\n=== ZONES IN CRZ (centroid <= 40.7625) ===")
in_crz = result[result["centroid_below_60th"]]
print(in_crz[["LocationID", "zone", "centroid_lat"]].to_string(index=False))

# Print boundary zones
print("\n=== BOUNDARY ZONES (straddle 60th) ===")
boundary = result[~result["entirely_above_60th"] & ~result["entirely_below_60th"]]
print(boundary[["LocationID", "zone", "centroid_lat", "south_lat", "north_lat"]].to_string(index=False))
# %%

# %%
import pandas as pd

# NOAA weather
print("=== NOAA Local Climatological Data ===")
df = pd.read_csv("data/raw/noaa/NOAALocalClimatologicalData.csv", nrows=5)
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 3 rows of key fields:")
key_cols = [c for c in df.columns if c in ["STATION", "DATE", "REPORT_TYPE", 
            "HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyWindSpeed",
            "HourlyVisibility", "HourlyPresentWeatherType", "HourlySkyConditions"]]
print(df[key_cols].head(3).to_string())

# Storm events
print("\n=== NOAA Storm Events ===")
df = pd.read_csv("data/raw/noaa/storm_data_search_results.csv", nrows=5)
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 3 rows:")
print(df.head(3).to_string())
# %%

# %%
print("=== SAPO permits ===")
df = pd.read_csv("data/raw/events/NYC_Permitted_Event_Information_20260507.csv", nrows=5)
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 3 rows transposed:")
print(df.head(3).T.to_string())
# %%

# %%
import pandas as pd
df = pd.read_csv("data/raw/noaa/NOAALocalClimatologicalData.csv", 
                 usecols=["STATION", "LATITUDE", "LONGITUDE"], nrows=10000)
print("Unique stations and their coords:")
print(df.groupby("STATION")[["LATITUDE", "LONGITUDE"]].first())
# %%

# %%
import duckdb
print(duckdb.query("DESCRIBE SELECT * FROM 'data/processed/tlc_zone_hour.parquet'").df())
# %%

# %%
# ============================================================
# DATA CLEANING — Items 1-6 (TLC re-agg + 5 supporting datasets)
# ============================================================
import pandas as pd
import duckdb
import os
import glob
import time
from pandas.tseries.holiday import USFederalHolidayCalendar

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/processed/duckdb_temp", exist_ok=True)

# ============================================================
# 0. RE-AGGREGATE TLC for ratio-of-sums fare-per-mile
# ============================================================
print("=" * 60)
print("0. Re-aggregating TLC with total_adjusted_fare and total_miles")
print("=" * 60)

start = time.perf_counter()
files = sorted(glob.glob("data/raw/tlc/*.parquet"))

output_path = "data/processed/tlc_zone_hour.parquet"
if os.path.exists(output_path):
    os.remove(output_path)

temp_files = []
for i, file_path in enumerate(files, 1):
    file_start = time.perf_counter()
    file_name = os.path.basename(file_path)
    temp_out = f"data/processed/_tlc_temp_{i:02d}.parquet"

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET temp_directory='data/processed/duckdb_temp'")

    cols = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{file_path}')").df()["column_name"].tolist()
    has_cbd = 'cbd_congestion_fee' in cols
    cbd_expr = "COALESCE(cbd_congestion_fee, 0)" if has_cbd else "0"

    con.execute(f"""
        COPY (
            WITH cleaned AS (
                SELECT
                    CASE
                        WHEN hvfhs_license_num = 'HV0003' THEN 'uber'
                        WHEN hvfhs_license_num = 'HV0005' THEN 'lyft'
                        ELSE 'other'
                    END as platform,
                    DATE_TRUNC('hour', pickup_datetime) as pickup_hour,
                    PULocationID as pickup_zone,
                    trip_miles,
                    trip_time,
                    base_passenger_fare,
                    congestion_surcharge,
                    {cbd_expr} as cbd_fee,
                    airport_fee,
                    driver_pay,
                    shared_match_flag,
                    -- Adjusted fare strips congestion fees BEFORE surge proxy
                    (base_passenger_fare - congestion_surcharge - {cbd_expr}) as adjusted_fare,
                    (base_passenger_fare - congestion_surcharge - {cbd_expr}) / NULLIF(trip_miles, 0) as fare_per_mile_clean
                FROM read_parquet('{file_path}')
                WHERE
                    trip_miles > 0 AND trip_miles < 100
                    AND trip_time > 60 AND trip_time < 10800
                    AND base_passenger_fare > 0
                    AND PULocationID IS NOT NULL AND PULocationID <= 263
                    AND hvfhs_license_num IN ('HV0003', 'HV0005')
            )
            SELECT
                pickup_hour,
                pickup_zone,
                platform,
                COUNT(*) as trip_count,
                -- For ratio-of-sums: total_adjusted_fare / total_miles
                SUM(adjusted_fare) as total_adjusted_fare,
                SUM(trip_miles) as total_miles,
                SUM(base_passenger_fare) as total_base_fare,
                SUM(driver_pay) as total_driver_pay,
                -- Mean-of-ratios kept for comparison
                AVG(fare_per_mile_clean) as mean_fare_per_mile,
                MEDIAN(fare_per_mile_clean) as median_fare_per_mile,
                QUANTILE_CONT(fare_per_mile_clean, 0.75) as p75_fare_per_mile,
                AVG(trip_miles) as mean_trip_miles,
                AVG(trip_time) as mean_trip_time,
                AVG(CASE WHEN shared_match_flag = 'Y' THEN 1.0 ELSE 0.0 END) as pct_shared,
                AVG(CASE WHEN airport_fee > 0 THEN 1.0 ELSE 0.0 END) as pct_airport
            FROM cleaned
            GROUP BY pickup_hour, pickup_zone, platform
        ) TO '{temp_out}' (FORMAT PARQUET)
    """)
    con.close()

    temp_files.append(temp_out)
    elapsed = time.perf_counter() - file_start
    print(f"  [{i:02d}/{len(files)}] {file_name}: {elapsed:.1f}s")

# Combine all temp files
print("\nCombining monthly aggregations...")
con = duckdb.connect()
con.execute(f"""
    COPY (SELECT * FROM read_parquet('data/processed/_tlc_temp_*.parquet'))
    TO '{output_path}' (FORMAT PARQUET)
""")
con.close()

for f in temp_files:
    os.remove(f)

print(f"⏱ TLC re-aggregation total: {(time.perf_counter() - start)/60:.1f} min")

# Verify ratio-of-sums computation
result = duckdb.query("""
    SELECT
        COUNT(*) as rows,
        SUM(trip_count) as total_trips,
        SUM(total_adjusted_fare)/SUM(total_miles) as global_ratio_of_sums_fpm
    FROM 'data/processed/tlc_zone_hour.parquet'
""").df()
print(result.to_string())

# ============================================================
# 1. NOAA WEATHER CLEANUP
# ============================================================
print("\n" + "=" * 60)
print("1. Cleaning NOAA weather (FM-15 + FM-16, de-duplicated)")
print("=" * 60)

noaa_cols = ["STATION", "DATE", "REPORT_TYPE",
             "HourlyDryBulbTemperature", "HourlyPrecipitation",
             "HourlyWindSpeed", "HourlyVisibility",
             "HourlyPresentWeatherType", "HourlySkyConditions"]

weather = pd.read_csv("data/raw/noaa/NOAALocalClimatologicalData.csv",
                      usecols=noaa_cols, low_memory=False)
weather["DATE"] = pd.to_datetime(weather["DATE"], errors="coerce")
weather = weather[(weather["DATE"] >= "2024-01-01") & (weather["DATE"] < "2025-09-01")]

# Keep both FM-15 (hourly METAR) and FM-16 (special obs during severe weather)
weather = weather[weather["REPORT_TYPE"].str.strip().isin(["FM-15", "FM-16"])]

# Map station ID to readable name
def map_station(s):
    s = str(s)
    if s.endswith("94728"): return "Central Park"
    if s.endswith("14732"): return "LGA"
    if s.endswith("94789"): return "JFK"
    return f"Other_{s}"

weather["station_name"] = weather["STATION"].apply(map_station)

# Convert numeric fields (some have trailing s/T characters)
for col in ["HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyWindSpeed", "HourlyVisibility"]:
    weather[col] = pd.to_numeric(
        weather[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
        errors="coerce"
    )

# Floor to hour for joining
weather["timestamp"] = weather["DATE"].dt.floor("H")

# De-duplicate by (station, hour) keeping the LATEST observation
weather = weather.sort_values(["station_name", "timestamp", "DATE"])
weather_clean = weather.groupby(["station_name", "timestamp"]).agg(
    temp_f=("HourlyDryBulbTemperature", "last"),
    precip_in=("HourlyPrecipitation", "last"),
    wind_mph=("HourlyWindSpeed", "last"),
    visibility_mi=("HourlyVisibility", "last"),
).reset_index()

# Weather flags
weather_clean["is_rain"] = weather_clean["precip_in"] > 0
weather_clean["is_heavy_rain"] = weather_clean["precip_in"] > 0.25
weather_clean["is_freezing"] = weather_clean["temp_f"] < 32
weather_clean["is_extreme_cold"] = weather_clean["temp_f"] < 20
weather_clean["is_extreme_heat"] = weather_clean["temp_f"] > 90

weather_clean.to_parquet("data/processed/noaa_weather_clean.parquet")
print(f"✅ Saved {len(weather_clean):,} weather observations")
print(f"   Date range: {weather_clean['timestamp'].min()} to {weather_clean['timestamp'].max()}")
print(f"   Stations: {weather_clean['station_name'].unique().tolist()}")

# ============================================================
# 2. NOAA STORM EVENTS — filter to NYC
# ============================================================
print("\n" + "=" * 60)
print("2. Filtering NOAA storm events to NYC counties")
print("=" * 60)

storms = pd.read_csv("data/raw/noaa/storm_data_search_results.csv")

# NYC county keywords
nyc_keywords = ["QUEENS", "KINGS", "BROOKLYN", "BRONX", "RICHMOND",
                "NEW YORK (MANHATTAN)", "NEW YORK COUNTY", "MANHATTAN",
                "NEW YORK (ZONE)"]
storms["is_nyc"] = storms["CZ_NAME_STR"].str.upper().str.contains(
    "|".join(nyc_keywords), na=False
)
storms_nyc = storms[storms["is_nyc"]].copy()

storms_nyc["begin_date"] = pd.to_datetime(storms_nyc["BEGIN_DATE"], format="%m/%d/%Y", errors="coerce")
storms_nyc["end_date"] = pd.to_datetime(storms_nyc["END_DATE"], format="%m/%d/%Y", errors="coerce")
storms_nyc = storms_nyc[(storms_nyc["begin_date"] >= "2024-01-01") &
                         (storms_nyc["begin_date"] < "2025-09-01")]

keep_cols = ["EVENT_ID", "CZ_NAME_STR", "EVENT_TYPE", "MAGNITUDE",
             "begin_date", "end_date", "BEGIN_TIME", "END_TIME",
             "DAMAGE_PROPERTY_NUM", "EVENT_NARRATIVE"]
storms_nyc = storms_nyc[keep_cols]
storms_nyc.columns = ["event_id", "county", "event_type", "magnitude",
                       "begin_date", "end_date", "begin_time", "end_time",
                       "damage_property", "narrative"]

storms_nyc.to_parquet("data/processed/noaa_storms_nyc.parquet")
print(f"✅ Saved {len(storms_nyc)} NYC storm events")
print(f"\nEvent type distribution:")
print(storms_nyc["event_type"].value_counts())

# ============================================================
# 3. SAPO PERMITS — borough-level, 3 attendance methodologies
# ============================================================
print("\n" + "=" * 60)
print("3. Cleaning SAPO permits (borough-level, 3 methodologies)")
print("=" * 60)

sapo = pd.read_csv("data/raw/events/NYC_Permitted_Event_Information_20260507.csv")
sapo["start_date"] = pd.to_datetime(sapo["Start Date/Time"], errors="coerce")
sapo["end_date"] = pd.to_datetime(sapo["End Date/Time"], errors="coerce")

sapo_clean = sapo[(sapo["start_date"] >= "2024-01-01") &
                   (sapo["start_date"] < "2025-09-01")].copy()

# Methodology 1 (PRIMARY): Binary event flag
# Most defensible — no hidden assumptions about attendance magnitude
sapo_clean["attend_binary"] = 1

# Methodology 2 (ROBUSTNESS): Hard-coded by event type
# Defensible only as sensitivity check
event_type_attendance = {
    "Parade": 50000,
    "Festival": 5000,
    "Block Party": 500,
    "Special Event": 1000,
    "Street Fair": 5000,
    "Demonstration": 2000,
    "Athletic": 2000,
    "Sport - Youth": 200,
    "Farmer's Market": 500,
    "Religious": 500,
    "Filming": 100,
}

def estimate_attendance(event_type):
    if pd.isna(event_type):
        return 100
    for key, val in event_type_attendance.items():
        if key.lower() in str(event_type).lower():
            return val
    return 100

sapo_clean["attend_typed"] = sapo_clean["Event Type"].apply(estimate_attendance)

# Methodology 3 (ROBUSTNESS): Location-weighted (major street vs. side street)
def is_major_location(loc):
    if pd.isna(loc):
        return False
    major_streets = ["5th Ave", "5 Ave", "6th Ave", "6 Ave", "7th Ave", "7 Ave",
                     "Broadway", "Times Square", "Park Ave", "Madison Ave",
                     "Central Park", "Brooklyn Bridge"]
    loc_str = str(loc)
    return any(s.lower() in loc_str.lower() for s in major_streets)

sapo_clean["is_major_location"] = sapo_clean["Event Location"].apply(is_major_location)
sapo_clean["attend_location"] = sapo_clean.apply(
    lambda r: r["attend_typed"] * (3 if r["is_major_location"] else 1), axis=1
)

sapo_clean["event_date"] = sapo_clean["start_date"].dt.date

keep = ["Event ID", "Event Name", "start_date", "end_date", "event_date",
        "Event Agency", "Event Type", "Event Borough", "Event Location",
        "attend_binary", "attend_typed", "attend_location", "is_major_location"]
sapo_final = sapo_clean[keep].copy()
sapo_final.columns = ["event_id", "event_name", "start_dt", "end_dt", "event_date",
                       "agency", "event_type", "borough", "location",
                       "attend_binary", "attend_typed", "attend_location",
                       "is_major_location"]

sapo_final.to_parquet("data/processed/sapo_permits_clean.parquet")
print(f"✅ Saved {len(sapo_final):,} SAPO events (Jan 2024 – Aug 2025)")
print(f"\nBorough breakdown:")
print(sapo_final["borough"].value_counts())
print(f"\nTop event types:")
print(sapo_final["event_type"].value_counts().head(10))

# ============================================================
# 4. HOLIDAY FLAGS
# ============================================================
print("\n" + "=" * 60)
print("4. Building holiday calendar")
print("=" * 60)

cal = USFederalHolidayCalendar()
fed_holidays = cal.holidays(start="2024-01-01", end="2025-09-01")

# NYC-specific additions (recurring annual events that distort baseline)
nyc_extras = pd.to_datetime([
    "2024-10-31",  # Halloween
    "2024-12-31",  # NYE
    "2024-11-03",  # Marathon
    "2025-03-17",  # St Patrick's Day Parade
    "2025-06-29",  # Pride
])
all_holidays = pd.DatetimeIndex(list(fed_holidays) + list(nyc_extras)).unique()

holidays_df = pd.DataFrame({"date": all_holidays.date, "is_holiday": True})
holidays_df.to_parquet("data/processed/holidays.parquet")
print(f"✅ Saved {len(holidays_df)} holiday dates")
print(holidays_df.to_string(index=False))

# ============================================================
# 5. ZONE-TO-WEATHER-STATION MAPPING (haversine to nearest)
# ============================================================
print("\n" + "=" * 60)
print("5. Mapping TLC zones to nearest NOAA station")
print("=" * 60)

import geopandas as gpd
import numpy as np

# Hardcoded NOAA station coordinates (well-known)
stations = pd.DataFrame([
    {"station_name": "Central Park", "lat": 40.7794, "lon": -73.9692},
    {"station_name": "LGA",          "lat": 40.7791, "lon": -73.8803},
    {"station_name": "JFK",          "lat": 40.6413, "lon": -73.7781},
])

# Load TLC zones — compute centroid in projected CRS for accuracy
zones = gpd.read_file("data/raw/tlc_zones/taxi_zones.shp")
zones["centroid_proj"] = zones.geometry.centroid
centroids_latlon = zones.set_geometry("centroid_proj").to_crs(epsg=4326).geometry
zones["zone_lat"] = centroids_latlon.y
zones["zone_lon"] = centroids_latlon.x

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

zone_station_rows = []
for _, zone in zones.iterrows():
    distances = stations.apply(
        lambda s: haversine(zone["zone_lat"], zone["zone_lon"], s["lat"], s["lon"]),
        axis=1
    )
    nearest_idx = distances.idxmin()
    zone_station_rows.append({
        "LocationID": zone["LocationID"],
        "zone": zone["zone"],
        "borough": zone["borough"],
        "nearest_station": stations.iloc[nearest_idx]["station_name"],
        "distance_mi": round(distances.iloc[nearest_idx], 2),
    })

zone_station_df = pd.DataFrame(zone_station_rows)
zone_station_df.to_parquet("data/processed/zone_to_weather_station.parquet")
print(f"✅ Saved zone-to-station mapping for {len(zone_station_df)} zones")
print(f"\nStation assignment distribution:")
print(zone_station_df["nearest_station"].value_counts())
print(f"\nMax distance: {zone_station_df['distance_mi'].max():.1f} miles")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ ALL CLEANING COMPLETE")
print("=" * 60)
print("\nOutputs in data/processed/:")
print("  - tlc_zone_hour.parquet (re-aggregated with total_adjusted_fare, total_miles)")
print("  - noaa_weather_clean.parquet")
print("  - noaa_storms_nyc.parquet")
print("  - sapo_permits_clean.parquet")
print("  - holidays.parquet")
print("  - zone_to_weather_station.parquet")
# %%

# %%
import pandas as pd
sapo = pd.read_csv("data/raw/events/NYC_Permitted_Event_Information_20260507.csv")
sapo["start_date"] = pd.to_datetime(sapo["Start Date/Time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
print(f"Total rows: {len(sapo)}")
print(f"Date range: {sapo['start_date'].min()} to {sapo['start_date'].max()}")
print(f"\nRows per year:")
print(sapo["start_date"].dt.year.value_counts().sort_index())
# %%

# %%
# Re-do SAPO with historical dataset
import pandas as pd

print("=" * 60)
print("3b. Cleaning SAPO permits (historical dataset)")
print("=" * 60)

sapo = pd.read_csv("data/raw/events/NYC_Permitted_Event_Information_Historical.csv")
print(f"Raw rows: {len(sapo):,}")
print(f"Columns: {sapo.columns.tolist()}")

sapo["start_date"] = pd.to_datetime(sapo["Start Date/Time"],
                                     format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
sapo["end_date"] = pd.to_datetime(sapo["End Date/Time"],
                                   format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

print(f"\nDate range: {sapo['start_date'].min()} to {sapo['start_date'].max()}")
print(f"Rows per year:\n{sapo['start_date'].dt.year.value_counts().sort_index().to_string()}")

sapo_clean = sapo[(sapo["start_date"] >= "2024-01-01") &
                   (sapo["start_date"] < "2025-09-01")].copy()
print(f"\nRows in window (Jan 2024 – Aug 2025): {len(sapo_clean):,}")

# Methodology 1 (PRIMARY): Binary
sapo_clean["attend_binary"] = 1

# Methodology 2 (ROBUSTNESS): Hard-coded by event type
event_type_attendance = {
    "Parade": 50000, "Festival": 5000, "Block Party": 500,
    "Special Event": 1000, "Street Fair": 5000, "Demonstration": 2000,
    "Athletic": 2000, "Sport - Youth": 200, "Farmer's Market": 500,
    "Religious": 500, "Filming": 100,
}

def estimate_attendance(event_type):
    if pd.isna(event_type):
        return 100
    for key, val in event_type_attendance.items():
        if key.lower() in str(event_type).lower():
            return val
    return 100

sapo_clean["attend_typed"] = sapo_clean["Event Type"].apply(estimate_attendance)

# Methodology 3 (ROBUSTNESS): Location-weighted
def is_major_location(loc):
    if pd.isna(loc):
        return False
    major_streets = ["5th Ave", "5 Ave", "6th Ave", "6 Ave", "7th Ave", "7 Ave",
                     "Broadway", "Times Square", "Park Ave", "Madison Ave",
                     "Central Park", "Brooklyn Bridge"]
    loc_str = str(loc)
    return any(s.lower() in loc_str.lower() for s in major_streets)

sapo_clean["is_major_location"] = sapo_clean["Event Location"].apply(is_major_location)
sapo_clean["attend_location"] = sapo_clean.apply(
    lambda r: r["attend_typed"] * (3 if r["is_major_location"] else 1), axis=1
)

sapo_clean["event_date"] = sapo_clean["start_date"].dt.date

keep = ["Event ID", "Event Name", "start_date", "end_date", "event_date",
        "Event Agency", "Event Type", "Event Borough", "Event Location",
        "attend_binary", "attend_typed", "attend_location", "is_major_location"]
sapo_final = sapo_clean[keep].copy()
sapo_final.columns = ["event_id", "event_name", "start_dt", "end_dt", "event_date",
                       "agency", "event_type", "borough", "location",
                       "attend_binary", "attend_typed", "attend_location",
                       "is_major_location"]

sapo_final.to_parquet("data/processed/sapo_permits_clean.parquet")
print(f"\n✅ Saved {len(sapo_final):,} SAPO events")
print(f"\nBorough breakdown:")
print(sapo_final["borough"].value_counts())
print(f"\nTop 10 event types:")
print(sapo_final["event_type"].value_counts().head(10))
# %%

# %%
import os
print("Files in data/raw/events/:")
for f in os.listdir("data/raw/events"):
    size_kb = os.path.getsize(f"data/raw/events/{f}") / 1024
    print(f"  {f}: {size_kb:.1f} KB")
# %%

# %%
import pandas as pd

print("=" * 60)
print("3b. Cleaning SAPO permits (historical dataset)")
print("=" * 60)

# Only read columns we need to speed up the load on this 1.8GB file
needed_cols = ["Event ID", "Event Name", "Start Date/Time", "End Date/Time",
               "Event Agency", "Event Type", "Event Borough", "Event Location"]

sapo = pd.read_csv(
    "data/raw/events/NYC_Permitted_Event_Information_-_Historical_20260509.csv",
    usecols=needed_cols,
    low_memory=False
)
print(f"Raw rows: {len(sapo):,}")

sapo["start_date"] = pd.to_datetime(sapo["Start Date/Time"],
                                     format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
sapo["end_date"] = pd.to_datetime(sapo["End Date/Time"],
                                   format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

print(f"\nDate range: {sapo['start_date'].min()} to {sapo['start_date'].max()}")
print(f"Rows per year:\n{sapo['start_date'].dt.year.value_counts().sort_index().to_string()}")

sapo_clean = sapo[(sapo["start_date"] >= "2024-01-01") &
                   (sapo["start_date"] < "2025-09-01")].copy()
print(f"\nRows in window (Jan 2024 – Aug 2025): {len(sapo_clean):,}")

# Methodology 1 (PRIMARY): Binary
sapo_clean["attend_binary"] = 1

# Methodology 2 (ROBUSTNESS): Hard-coded by event type
event_type_attendance = {
    "Parade": 50000, "Festival": 5000, "Block Party": 500,
    "Special Event": 1000, "Street Fair": 5000, "Demonstration": 2000,
    "Athletic": 2000, "Sport - Youth": 200, "Farmer's Market": 500,
    "Religious": 500, "Filming": 100,
}

def estimate_attendance(event_type):
    if pd.isna(event_type):
        return 100
    for key, val in event_type_attendance.items():
        if key.lower() in str(event_type).lower():
            return val
    return 100

sapo_clean["attend_typed"] = sapo_clean["Event Type"].apply(estimate_attendance)

# Methodology 3 (ROBUSTNESS): Location-weighted
def is_major_location(loc):
    if pd.isna(loc):
        return False
    major_streets = ["5th Ave", "5 Ave", "6th Ave", "6 Ave", "7th Ave", "7 Ave",
                     "Broadway", "Times Square", "Park Ave", "Madison Ave",
                     "Central Park", "Brooklyn Bridge"]
    loc_str = str(loc)
    return any(s.lower() in loc_str.lower() for s in major_streets)

sapo_clean["is_major_location"] = sapo_clean["Event Location"].apply(is_major_location)
sapo_clean["attend_location"] = sapo_clean.apply(
    lambda r: r["attend_typed"] * (3 if r["is_major_location"] else 1), axis=1
)

sapo_clean["event_date"] = sapo_clean["start_date"].dt.date

keep = ["Event ID", "Event Name", "start_date", "end_date", "event_date",
        "Event Agency", "Event Type", "Event Borough", "Event Location",
        "attend_binary", "attend_typed", "attend_location", "is_major_location"]
sapo_final = sapo_clean[keep].copy()
sapo_final.columns = ["event_id", "event_name", "start_dt", "end_dt", "event_date",
                       "agency", "event_type", "borough", "location",
                       "attend_binary", "attend_typed", "attend_location",
                       "is_major_location"]

sapo_final.to_parquet("data/processed/sapo_permits_clean.parquet")
print(f"\n✅ Saved {len(sapo_final):,} SAPO events")
print(f"\nBorough breakdown:")
print(sapo_final["borough"].value_counts())
print(f"\nTop 10 event types:")
print(sapo_final["event_type"].value_counts().head(10))
# %%

# %%
import duckdb
import pandas as pd

print("=" * 60)
print("3b. Cleaning SAPO permits via DuckDB")
print("=" * 60)

# Use DuckDB to filter + aggregate the 1.8GB file efficiently
sapo = duckdb.query("""
    SELECT 
        "Event ID" as event_id,
        "Event Name" as event_name,
        "Start Date/Time" as start_dt_raw,
        "End Date/Time" as end_dt_raw,
        "Event Agency" as agency,
        "Event Type" as event_type,
        "Event Borough" as borough,
        "Event Location" as location
    FROM read_csv_auto('data/raw/events/NYC_Permitted_Event_Information_-_Historical_20260509.csv',
                       header=True,
                       all_varchar=True)
    WHERE "Start Date/Time" IS NOT NULL
""").df()
print(f"Raw rows loaded: {len(sapo):,}")

# Parse dates
sapo["start_date"] = pd.to_datetime(sapo["start_dt_raw"],
                                     format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
sapo["end_date"] = pd.to_datetime(sapo["end_dt_raw"],
                                   format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

print(f"Date range: {sapo['start_date'].min()} to {sapo['start_date'].max()}")

# Filter to window
sapo_clean = sapo[(sapo["start_date"] >= "2024-01-01") &
                   (sapo["start_date"] < "2025-09-01")].copy()
print(f"\nRows in window (Jan 2024 – Aug 2025): {len(sapo_clean):,}")

# Methodology 1 (PRIMARY): Binary
sapo_clean["attend_binary"] = 1

# Methodology 2 (ROBUSTNESS): Hard-coded by event type
event_type_attendance = {
    "Parade": 50000, "Festival": 5000, "Block Party": 500,
    "Special Event": 1000, "Street Fair": 5000, "Demonstration": 2000,
    "Athletic": 2000, "Sport - Youth": 200, "Farmer's Market": 500,
    "Religious": 500, "Filming": 100,
}

def estimate_attendance(event_type):
    if pd.isna(event_type):
        return 100
    for key, val in event_type_attendance.items():
        if key.lower() in str(event_type).lower():
            return val
    return 100

sapo_clean["attend_typed"] = sapo_clean["event_type"].apply(estimate_attendance)

# Methodology 3 (ROBUSTNESS): Location-weighted
def is_major_location(loc):
    if pd.isna(loc):
        return False
    major_streets = ["5th Ave", "5 Ave", "6th Ave", "6 Ave", "7th Ave", "7 Ave",
                     "Broadway", "Times Square", "Park Ave", "Madison Ave",
                     "Central Park", "Brooklyn Bridge"]
    loc_str = str(loc)
    return any(s.lower() in loc_str.lower() for s in major_streets)

sapo_clean["is_major_location"] = sapo_clean["location"].apply(is_major_location)
sapo_clean["attend_location"] = sapo_clean.apply(
    lambda r: r["attend_typed"] * (3 if r["is_major_location"] else 1), axis=1
)

sapo_clean["event_date"] = sapo_clean["start_date"].dt.date

# Drop the raw date strings, keep clean version
keep = ["event_id", "event_name", "start_date", "end_date", "event_date",
        "agency", "event_type", "borough", "location",
        "attend_binary", "attend_typed", "attend_location", "is_major_location"]
sapo_final = sapo_clean[keep].copy()
sapo_final.columns = ["event_id", "event_name", "start_dt", "end_dt", "event_date",
                       "agency", "event_type", "borough", "location",
                       "attend_binary", "attend_typed", "attend_location",
                       "is_major_location"]

sapo_final.to_parquet("data/processed/sapo_permits_clean.parquet")
print(f"\n✅ Saved {len(sapo_final):,} SAPO events")
print(f"\nBorough breakdown:")
print(sapo_final["borough"].value_counts())
print(f"\nTop 10 event types:")
print(sapo_final["event_type"].value_counts().head(10))
# %%

# %%
import duckdb

print("Counting distinct events...")
result = duckdb.query("""
    SELECT 
        COUNT(*) as total_rows,
        COUNT(DISTINCT "Event ID") as distinct_events
    FROM read_csv_auto('data/raw/events/NYC_Permitted_Event_Information_-_Historical_20260509.csv',
                       header=True, all_varchar=True)
""").df()
print(result.to_string())
# %%

# %%
import duckdb
import pandas as pd

print("=" * 60)
print("3c. Cleaning SAPO permits — DEDUPED by Event ID")
print("=" * 60)

# Use DuckDB to dedupe by Event ID first, then load to pandas
sapo = duckdb.query("""
    WITH deduped AS (
        SELECT 
            "Event ID" as event_id,
            ANY_VALUE("Event Name") as event_name,
            ANY_VALUE("Start Date/Time") as start_dt_raw,
            ANY_VALUE("End Date/Time") as end_dt_raw,
            ANY_VALUE("Event Agency") as agency,
            ANY_VALUE("Event Type") as event_type,
            ANY_VALUE("Event Borough") as borough,
            ANY_VALUE("Event Location") as location
        FROM read_csv_auto('data/raw/events/NYC_Permitted_Event_Information_-_Historical_20260509.csv',
                           header=True, all_varchar=True)
        GROUP BY "Event ID"
    )
    SELECT * FROM deduped
""").df()
print(f"Distinct events loaded: {len(sapo):,}")

# Parse dates
sapo["start_date"] = pd.to_datetime(sapo["start_dt_raw"],
                                     format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
sapo["end_date"] = pd.to_datetime(sapo["end_dt_raw"],
                                   format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

print(f"Date range: {sapo['start_date'].min()} to {sapo['start_date'].max()}")

# Filter to window
sapo_clean = sapo[(sapo["start_date"] >= "2024-01-01") &
                   (sapo["start_date"] < "2025-09-01")].copy()
print(f"\nRows in window (Jan 2024 – Aug 2025): {len(sapo_clean):,}")

# Methodology 1 (PRIMARY): Binary
sapo_clean["attend_binary"] = 1

# Methodology 2 (ROBUSTNESS): Hard-coded by event type
event_type_attendance = {
    "Parade": 50000, "Festival": 5000, "Block Party": 500,
    "Special Event": 1000, "Street Fair": 5000, "Demonstration": 2000,
    "Athletic": 2000, "Sport - Youth": 200, "Farmer's Market": 500,
    "Religious": 500, "Filming": 100,
}

def estimate_attendance(event_type):
    if pd.isna(event_type):
        return 100
    for key, val in event_type_attendance.items():
        if key.lower() in str(event_type).lower():
            return val
    return 100

sapo_clean["attend_typed"] = sapo_clean["event_type"].apply(estimate_attendance)

# Methodology 3 (ROBUSTNESS): Location-weighted
def is_major_location(loc):
    if pd.isna(loc):
        return False
    major_streets = ["5th Ave", "5 Ave", "6th Ave", "6 Ave", "7th Ave", "7 Ave",
                     "Broadway", "Times Square", "Park Ave", "Madison Ave",
                     "Central Park", "Brooklyn Bridge"]
    loc_str = str(loc)
    return any(s.lower() in loc_str.lower() for s in major_streets)

sapo_clean["is_major_location"] = sapo_clean["location"].apply(is_major_location)
sapo_clean["attend_location"] = sapo_clean.apply(
    lambda r: r["attend_typed"] * (3 if r["is_major_location"] else 1), axis=1
)

sapo_clean["event_date"] = sapo_clean["start_date"].dt.date

keep = ["event_id", "event_name", "start_date", "end_date", "event_date",
        "agency", "event_type", "borough", "location",
        "attend_binary", "attend_typed", "attend_location", "is_major_location"]
sapo_final = sapo_clean[keep].copy()
sapo_final.columns = ["event_id", "event_name", "start_dt", "end_dt", "event_date",
                       "agency", "event_type", "borough", "location",
                       "attend_binary", "attend_typed", "attend_location",
                       "is_major_location"]

sapo_final.to_parquet("data/processed/sapo_permits_clean.parquet")
print(f"\n✅ Saved {len(sapo_final):,} SAPO events")
print(f"\nBorough breakdown:")
print(sapo_final["borough"].value_counts())
print(f"\nTop 10 event types:")
print(sapo_final["event_type"].value_counts().head(10))
# %%

# %%
# ============================================================
# FINAL DATA AUDIT — verify everything is ready for analysis
# ============================================================
import os
import duckdb
import pandas as pd

print("=" * 70)
print("FINAL DATA AUDIT — Are we ready for analysis?")
print("=" * 70)

# Expected files
files_to_check = [
    # Processed datasets (the ones analysis will use)
    ("data/processed/tlc_zone_hour.parquet", "TLC zone-hour aggregation"),
    ("data/processed/mta_subway_clean.parquet", "MTA subway aggregation"),
    ("data/processed/mta_station_coords.parquet", "MTA station coordinates"),
    ("data/processed/noaa_weather_clean.parquet", "NOAA weather hourly"),
    ("data/processed/noaa_storms_nyc.parquet", "NOAA storms NYC"),
    ("data/processed/sapo_permits_clean.parquet", "SAPO permits"),
    ("data/processed/holidays.parquet", "Holiday flags"),
    ("data/processed/zone_to_weather_station.parquet", "Zone-to-weather mapping"),
    # Source-of-truth files
    ("data/raw/events/major_events.csv", "Major events table"),
    ("data/raw/tlc_zones/taxi_zones.shp", "Taxi zones shapefile"),
    ("data/raw/tlc_zones/taxi_zone_lookup.csv", "Taxi zone lookup"),
    ("src/crz_zones.py", "CRZ zone definition"),
]

print("\n--- File existence check ---")
all_present = True
for path, desc in files_to_check:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"  ✅ {desc:40s} ({size_mb:>7.2f} MB)")
    else:
        print(f"  ❌ MISSING: {desc} -> {path}")
        all_present = False

if not all_present:
    print("\n⚠️ Some files missing. Fix before proceeding.")

# Row counts and date ranges for each processed dataset
print("\n--- Processed data summaries ---")

print("\n📊 TLC zone-hour:")
print(duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(pickup_hour) as earliest,
        MAX(pickup_hour) as latest,
        COUNT(DISTINCT pickup_zone) as zones,
        COUNT(DISTINCT platform) as platforms,
        SUM(trip_count) as total_trips,
        ROUND(SUM(total_adjusted_fare)/SUM(total_miles), 2) as ratio_of_sums_fpm
    FROM 'data/processed/tlc_zone_hour.parquet'
""").df().to_string(index=False))

print("\n📊 MTA subway:")
print(duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT station_complex) as stations,
        ROUND(SUM(ridership)/1e6, 1) as total_riders_M
    FROM 'data/processed/mta_subway_clean.parquet'
""").df().to_string(index=False))

print("\n📊 NOAA weather:")
print(duckdb.query("""
    SELECT 
        COUNT(*) as rows,
        MIN(timestamp) as earliest,
        MAX(timestamp) as latest,
        COUNT(DISTINCT station_name) as stations,
        SUM(CAST(is_rain AS INT)) as rainy_hours,
        SUM(CAST(is_heavy_rain AS INT)) as heavy_rain_hours
    FROM 'data/processed/noaa_weather_clean.parquet'
""").df().to_string(index=False))

print("\n📊 NOAA storms:")
print(duckdb.query("""
    SELECT 
        COUNT(*) as events,
        MIN(begin_date) as earliest,
        MAX(begin_date) as latest,
        COUNT(DISTINCT event_type) as event_types
    FROM 'data/processed/noaa_storms_nyc.parquet'
""").df().to_string(index=False))

print("\n📊 SAPO permits:")
print(duckdb.query("""
    SELECT 
        COUNT(*) as events,
        MIN(event_date) as earliest,
        MAX(event_date) as latest,
        COUNT(DISTINCT borough) as boroughs,
        COUNT(DISTINCT event_type) as types
    FROM 'data/processed/sapo_permits_clean.parquet'
""").df().to_string(index=False))

print("\n📊 Major events table:")
df = pd.read_csv("data/raw/events/major_events.csv")
print(f"  Rows: {len(df)}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
print(f"  Leagues: {df['league'].value_counts().to_dict()}")

print("\n📊 Holidays:")
df = pd.read_parquet("data/processed/holidays.parquet")
print(f"  Holiday dates: {len(df)}")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

print("\n📊 Zone-station mapping:")
df = pd.read_parquet("data/processed/zone_to_weather_station.parquet")
print(f"  Zones mapped: {len(df)}")
print(f"  Station distribution: {df['nearest_station'].value_counts().to_dict()}")

print("\n📊 CRZ zones:")
import sys
sys.path.insert(0, "src")
from crz_zones import CRZ_ZONE_IDS
print(f"  CRZ zone count: {len(CRZ_ZONE_IDS)}")
print(f"  Sample: {CRZ_ZONE_IDS[:5]}...")

# Cross-check: do the date ranges align?
print("\n--- Cross-check: date alignment ---")
print("All processed datasets should cover Jan 1 2024 – Aug 31 2025.")
print("If anything is missing data at the edges, flag it.")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
# %%
