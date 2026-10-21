
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_FOLDER = Path(r"C:\Users\dlihi\Documents\PRT661 Data Science\Assessment 2\PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2\Source code\Datasets").resolve()
OUTPUT_FILE = Path(
    r"C:\Users\dlihi\Documents\PRT661 Data Science\Assessment 2\PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2\Dashboard\Fire2Air_dashboard_fire_points_2018_2024.csv"
)
PROJECT_YEARS = list(range(2018, 2025))
MAX_FIRE_DISTANCE_KM = 500.0
PREFILTER_DISTANCE_KM = 520.0
CHUNK_SIZE = 500_000

STATION_COORDINATES = {
    "Palmerston": (-12.507753, 130.948253),
    "Stokes Hill": (-12.459991, 130.847847),
    "Winnellie": (-12.424017, 130.893346),
}

def haversine_distance(lat, lon, target_lat, target_lon):
    R = 6371.0088
    lat1 = np.radians(np.asarray(lat, dtype=float))
    lon1 = np.radians(np.asarray(lon, dtype=float))
    lat2 = np.radians(float(target_lat))
    lon2 = np.radians(float(target_lon))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

firms_files = sorted(PROJECT_FOLDER.glob("fire_archive_SV-C2_*.csv"))

if not firms_files:
    raise FileNotFoundError("No FIRMS CSV files found matching fire_archive_SV-C2_*.csv")

print(f"FIRMS CSV files found: {len(firms_files)}")
for file in firms_files:
    print(" -", file.name)

parts = []
audit_rows = []

for i, file in enumerate(firms_files, start=1):
    print(f"\nProcessing {i}/{len(firms_files)}: {file.name}")
    raw_rows = 0
    kept_rows = 0

    for chunk_no, chunk in enumerate(pd.read_csv(file, chunksize=CHUNK_SIZE), start=1):
        raw_rows += len(chunk)

        required = {"latitude", "longitude", "acq_date", "acq_time", "frp"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"{file.name} missing required columns: {sorted(missing)}")

        for col in ["latitude", "longitude", "frp"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        parsed_date = pd.to_datetime(chunk["acq_date"], errors="coerce")

        valid = (
            chunk["latitude"].between(-90, 90)
            & chunk["longitude"].between(-180, 180)
            & parsed_date.notna()
        )
        chunk = chunk.loc[valid].copy()
        if chunk.empty:
            continue

        # Same broad Top End filter used by Assessment 3.
        chunk = chunk[
            chunk["latitude"].between(-17.5, -7.5)
            & chunk["longitude"].between(125.5, 136.5)
        ].copy()
        if chunk.empty:
            continue

        distances = {}
        for station, (slat, slon) in STATION_COORDINATES.items():
            key = station.lower().replace(" ", "_")
            dist = haversine_distance(
                chunk["latitude"].to_numpy(),
                chunk["longitude"].to_numpy(),
                slat,
                slon,
            )
            chunk[f"distance_{key}_km"] = dist
            distances[station] = dist

        distance_df = pd.DataFrame(distances, index=chunk.index)
        chunk["nearest_station"] = distance_df.idxmin(axis=1)
        chunk["nearest_station_distance_km"] = distance_df.min(axis=1)

        # Same prefilter used in the Assessment 3 pipeline.
        chunk = chunk[
            chunk["nearest_station_distance_km"] <= PREFILTER_DISTANCE_KM
        ].copy()
        if chunk.empty:
            continue

        chunk.loc[chunk["frp"] < 0, "frp"] = np.nan

        # Vegetation fires only.
        if "type" in chunk.columns:
            chunk["type"] = pd.to_numeric(chunk["type"], errors="coerce")
            chunk = chunk[chunk["type"].eq(0)].copy()
        if chunk.empty:
            continue

        # UTC -> Darwin local time.
        chunk["acq_time_str"] = (
            chunk["acq_time"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(4)
        )
        chunk["datetime_utc"] = pd.to_datetime(
            chunk["acq_date"].astype(str) + " " + chunk["acq_time_str"],
            format="%Y-%m-%d %H%M",
            utc=True,
            errors="coerce",
        )
        chunk["datetime_darwin"] = (
            chunk["datetime_utc"]
            .dt.tz_convert("Australia/Darwin")
            .dt.tz_localize(None)
        )
        chunk["date_darwin"] = chunk["datetime_darwin"].dt.normalize()

        # Final dashboard range = <=500 km from at least one station.
        chunk = chunk[
            chunk["nearest_station_distance_km"] <= MAX_FIRE_DISTANCE_KM
        ].copy()
        if chunk.empty:
            continue

        keep = [
            "date_darwin",
            "datetime_darwin",
            "latitude",
            "longitude",
            "frp",
            "nearest_station",
            "nearest_station_distance_km",
            "distance_palmerston_km",
            "distance_stokes_hill_km",
            "distance_winnellie_km",
        ]
        for col in ["confidence", "daynight", "satellite", "instrument", "type"]:
            if col in chunk.columns:
                keep.append(col)

        compact = chunk[keep].copy()
        compact["source_file"] = file.name

        parts.append(compact)
        kept_rows += len(compact)

        print(f"   chunk {chunk_no}: retained {len(compact):,}")

    audit_rows.append({
        "file": file.name,
        "raw_rows": raw_rows,
        "retained_rows": kept_rows,
    })

    print(f"Completed {file.name} | raw={raw_rows:,} | retained={kept_rows:,}")

if not parts:
    raise ValueError("No FIRMS rows remained after filtering.")

dashboard_fire = pd.concat(parts, ignore_index=True)

dedup_cols = [
    c for c in [
        "latitude", "longitude", "datetime_darwin", "frp",
        "confidence", "daynight", "satellite", "instrument", "type"
    ]
    if c in dashboard_fire.columns
]

duplicates_before = int(
    dashboard_fire.duplicated(subset=dedup_cols).sum()
)

dashboard_fire = dashboard_fire.drop_duplicates(
    subset=dedup_cols,
    keep="first"
).copy()

dashboard_fire["year"] = pd.to_datetime(
    dashboard_fire["date_darwin"],
    errors="coerce"
).dt.year

actual_years = sorted(
    dashboard_fire["year"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)

missing_years = [y for y in PROJECT_YEARS if y not in actual_years]
if missing_years:
    raise ValueError(f"Missing FIRMS years in dashboard output: {missing_years}")

# Add simple intensity category; dashboard still keeps raw FRP for icon sizing.
valid_frp = dashboard_fire["frp"].dropna()

if len(valid_frp):
    q33 = float(valid_frp.quantile(0.33))
    q67 = float(valid_frp.quantile(0.67))

    dashboard_fire["fire_intensity"] = np.select(
        [
            dashboard_fire["frp"] <= q33,
            (dashboard_fire["frp"] > q33) & (dashboard_fire["frp"] <= q67),
            dashboard_fire["frp"] > q67,
        ],
        ["Low", "Medium", "High"],
        default="Unknown",
    )
else:
    q33 = q67 = np.nan
    dashboard_fire["fire_intensity"] = "Unknown"

# Round to reduce CSV size.
dashboard_fire["latitude"] = dashboard_fire["latitude"].round(5)
dashboard_fire["longitude"] = dashboard_fire["longitude"].round(5)

for col in [
    "frp",
    "nearest_station_distance_km",
    "distance_palmerston_km",
    "distance_stokes_hill_km",
    "distance_winnellie_km",
]:
    if col in dashboard_fire.columns:
        dashboard_fire[col] = dashboard_fire[col].round(2)

dashboard_fire = dashboard_fire.sort_values(
    ["date_darwin", "datetime_darwin", "frp"],
    ascending=[True, True, False]
).reset_index(drop=True)

dashboard_fire["date_darwin"] = pd.to_datetime(
    dashboard_fire["date_darwin"]
).dt.strftime("%Y-%m-%d")

dashboard_fire["datetime_darwin"] = pd.to_datetime(
    dashboard_fire["datetime_darwin"]
).dt.strftime("%Y-%m-%d %H:%M:%S")

dashboard_fire.to_csv(OUTPUT_FILE, index=False)

audit_file = PROJECT_FOLDER / "Fire2Air_dashboard_fire_points_audit_2018_2024.csv"
pd.DataFrame(audit_rows).to_csv(audit_file, index=False)

print("\n" + "=" * 70)
print("DASHBOARD-READY FIRE CSV CREATED")
print("=" * 70)
print("Output:", OUTPUT_FILE)
print("Rows:", f"{len(dashboard_fire):,}")
print("Years:", actual_years)
print("Duplicates removed:", f"{duplicates_before:,}")
print("Date range:", dashboard_fire["date_darwin"].min(), "to", dashboard_fire["date_darwin"].max())

if pd.notna(q33):
    print(f"FRP categories: Low <= {q33:.2f}, Medium <= {q67:.2f}, High > {q67:.2f} MW")

print("Audit:", audit_file)
print("=" * 70)
