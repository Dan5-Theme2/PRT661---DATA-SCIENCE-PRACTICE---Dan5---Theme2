# %% [markdown]
# # PART 1 — THI PHAM: DATA PROCESSING AND FEATURE ENGINEERING
#
# Extracted directly from Fire2Air_Darwin_Complete_Hybrid_All_Team_FIXED_v2.py
# Team responsibility version — code preserved from the corrected combined file.

# ==================================================================================================
# PART 1 — THI PHAM: DATA PROCESSING AND FEATURE ENGINEERING
# ==================================================================================================

# %% [markdown]
# # Fire2Air Darwin — 01 Data Processing and Feature Engineering
# **Member:** Thi Pham  
# **Responsibility:** Clean the data and prepare features for modelling.
# 
# This is the hybrid version requested for the final team workflow:
# - uses the newer `01_data_preparation` workflow as the executable base;
# - adds the stronger older data-quality audit and cleaning safeguards;
# - extends station-specific FIRMS features to **500 km**;
# - deduplicates overlapping FIRMS detections;
# - creates leakage-safe next-day targets only when the next row is exactly the next calendar day;
# - creates `target_date`, so downstream train/validation/test splits are based on the **date being predicted**, not only the feature date.
# 
# Run this notebook first. It creates `Fire2Air_model_ready_checkpoint.csv` and `Fire2Air_feature_manifest.json` for the other team notebooks.

# %%
from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 661
PROJECT_YEARS = list(range(2018, 2025))
PM25_THRESHOLD = 25.0
MIN_VALID_PM_HOURS = 18
MIN_VALID_COUNT_HOURS = 24
MAX_FIRE_DISTANCE_KM = 500.0
PREFILTER_DISTANCE_KM = 520.0

# Find the project folder. Put the notebook in the same folder as the data,
# in a Data Science subfolder, or on the Desktop/Data Science folder.
possible_folders = [
    Path(__file__).resolve().parent,
    Path.cwd(),
    Path.cwd() / "Data Science",
    Path.home() / "Desktop" / "Data Science",
]

project_folder = None
for folder in possible_folders:
    if (folder / "ntepa_greater_darwin_hourly_2018_2024.csv").exists():
        project_folder = folder.resolve()
        break

if project_folder is None:
    raise FileNotFoundError(
        "The NT EPA dataset could not be found. Place "
        "'ntepa_greater_darwin_hourly_2018_2024.csv' and the FIRMS yearly files "
        "in the project folder."
    )

output_root = project_folder / "outputs_prt661"
table_dir = output_root / "tables"
processed_dir = output_root / "processed"
for folder in [output_root, table_dir, processed_dir]:
    folder.mkdir(parents=True, exist_ok=True)

print("Project folder:", project_folder)

# %% [markdown]
# ## A. NT EPA schema, coverage and quality audit
# The raw file is inspected before cleaning. Statistical extremes are reported, but only physically invalid or clearly inconsistent values are removed automatically.

# %%
ntepa_file = project_folder / "ntepa_greater_darwin_hourly_2018_2024.csv"
df = pd.read_csv(ntepa_file)
df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors="coerce")
df["date"] = df["datetime_local"].dt.normalize()
df["year"] = df["datetime_local"].dt.year

required_columns = {
    "datetime_local", "station", "latitude", "longitude",
    "pm25_ug_m3", "pm10_ug_m3", "relative_humidity_pct",
    "air_temperature_c", "wind_speed_m_s", "wind_direction_deg",
    "air_pressure_hpa", "rainfall_mm",
}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"NT EPA file is missing required columns: {sorted(missing_columns)}")

quality_columns = [
    "pm25_ug_m3", "pm10_ug_m3", "relative_humidity_pct",
    "air_temperature_c", "wind_speed_m_s", "wind_direction_deg",
    "air_pressure_hpa", "rainfall_mm",
]

overview = pd.DataFrame({
    "metric": [
        "rows", "columns", "date_min", "date_max", "stations",
        "full_row_duplicates", "station_timestamp_duplicates"
    ],
    "value": [
        len(df), df.shape[1], df["datetime_local"].min(), df["datetime_local"].max(),
        ", ".join(sorted(df["station"].dropna().astype(str).unique())),
        int(df.duplicated().sum()),
        int(df.duplicated(["station", "datetime_local"]).sum()),
    ]
})
print("\nRaw NT EPA overview:")
print(overview.to_string(index=False))
overview.to_csv(table_dir / "epa_raw_overview.csv", index=False)

missing_by_year_station = (
    df.groupby(["year", "station"])[quality_columns]
      .agg(lambda s: s.isna().mean() * 100)
      .round(2)
      .reset_index()
)
missing_by_year_station.to_csv(
    table_dir / "epa_missing_pct_year_station_variable.csv", index=False
)

raw_summary = df[quality_columns].describe(
    percentiles=[0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
).T
raw_summary.to_csv(table_dir / "epa_raw_numeric_summary.csv")
print("\nRaw numeric summary:")
print(raw_summary.round(3).to_string())

# %% [markdown]
# ## B. Cleaning and unit harmonisation
# Key safeguards combined from the two earlier versions:
# - all negative PM₂.₅ and PM₁₀ values are treated as invalid;
# - humidity must be 0–100%;
# - broad sensor plausibility rules are applied to temperature, wind and pressure;
# - pressure is audited by station and values on an approximately 1-atmosphere scale are converted to hPa;
# - rainfall is not trusted as a simple hourly value until station behaviour is checked. A cumulative counter is differenced across consecutive hours, while resets/gaps are kept missing;
# - a cleaning log records the number of affected values.

# %%
df_clean = df.copy()
cleaning_log = []

def log_change(variable, rule, before_nonmissing, after_nonmissing):
    cleaning_log.append({
        "variable": variable,
        "rule": rule,
        "values_set_or_changed": int(before_nonmissing - after_nonmissing),
    })

# Remove physically impossible negative PM concentrations.
for col in ["pm25_ug_m3", "pm10_ug_m3"]:
    before = int(df_clean[col].notna().sum())
    df_clean.loc[pd.to_numeric(df_clean[col], errors="coerce") < 0, col] = np.nan
    after = int(df_clean[col].notna().sum())
    log_change(col, "negative concentration -> NaN", before, after)

# Pressure audit. The newer pipeline identified Stokes Hill on an ~1 atmosphere scale.
pressure_profile = (
    df_clean.groupby("station")["air_pressure_hpa"]
    .agg(["count", "median", "min", "max"])
    .reset_index()
)
pressure_profile["conversion_factor"] = np.where(
    pressure_profile["median"].between(0.8, 1.2, inclusive="both"),
    1013.25,
    1.0,
)
pressure_profile.to_csv(table_dir / "epa_pressure_unit_audit.csv", index=False)
print("\nPressure unit audit:")
print(pressure_profile.round(4).to_string(index=False))

for row in pressure_profile.itertuples(index=False):
    if row.conversion_factor != 1.0:
        mask = df_clean["station"].eq(row.station) & df_clean["air_pressure_hpa"].notna()
        df_clean.loc[mask, "air_pressure_hpa"] = (
            df_clean.loc[mask, "air_pressure_hpa"] * row.conversion_factor
        )
        cleaning_log.append({
            "variable": "air_pressure_hpa",
            "rule": f"{row.station}: ~1-atmosphere scale converted to hPa using x1013.25",
            "values_set_or_changed": int(mask.sum()),
        })

plausible_ranges = {
    "relative_humidity_pct": (0.0, 100.0),
    "air_temperature_c": (-20.0, 60.0),
    "wind_speed_m_s": (0.0, 80.0),
    "air_pressure_hpa": (850.0, 1100.0),
}

for col, (low, high) in plausible_ranges.items():
    before = int(df_clean[col].notna().sum())
    valid = pd.to_numeric(df_clean[col], errors="coerce").between(low, high, inclusive="both")
    df_clean.loc[df_clean[col].notna() & ~valid, col] = np.nan
    after = int(df_clean[col].notna().sum())
    log_change(col, f"outside [{low}, {high}] -> NaN", before, after)

# Wind direction is circular and valid in [0, 360).
before = int(df_clean["wind_direction_deg"].notna().sum())
valid_wind_dir = pd.to_numeric(df_clean["wind_direction_deg"], errors="coerce").between(
    0.0, 360.0, inclusive="left"
)
df_clean.loc[df_clean["wind_direction_deg"].notna() & ~valid_wind_dir, "wind_direction_deg"] = np.nan
after = int(df_clean["wind_direction_deg"].notna().sum())
log_change("wind_direction_deg", "outside [0, 360) -> NaN", before, after)

# Rainfall behaviour audit and harmonised hourly rainfall.
rain_profile_rows = []
for station, group in df_clean.sort_values("datetime_local").groupby("station"):
    s = pd.to_numeric(group["rainfall_mm"], errors="coerce")
    nonmissing = s.dropna()
    rain_profile_rows.append({
        "station": station,
        "nonmissing_pct": 100 * s.notna().mean(),
        "median_raw": nonmissing.median() if not nonmissing.empty else np.nan,
        "p95_raw": nonmissing.quantile(0.95) if not nonmissing.empty else np.nan,
        "zero_fraction_raw": nonmissing.eq(0).mean() if not nonmissing.empty else np.nan,
        "n_unique_raw": nonmissing.nunique(),
    })

rain_profile = pd.DataFrame(rain_profile_rows)
rain_profile["cumulative_counter_flag"] = (
    (rain_profile["median_raw"] > 100)
    & (rain_profile["zero_fraction_raw"].fillna(1.0) < 0.20)
)
rain_profile.to_csv(table_dir / "epa_rainfall_station_behaviour_audit.csv", index=False)
print("\nRainfall behaviour audit:")
print(rain_profile.round(3).to_string(index=False))

df_clean = df_clean.sort_values(["station", "datetime_local"]).copy()
df_clean["rainfall_hourly_mm"] = np.nan
cumulative_stations = set(rain_profile.loc[rain_profile["cumulative_counter_flag"], "station"])

for station, idx in df_clean.groupby("station").groups.items():
    g = df_clean.loc[idx].sort_values("datetime_local")
    raw_rain = pd.to_numeric(g["rainfall_mm"], errors="coerce")

    if station in cumulative_stations:
        diffs = raw_rain.diff()
        hour_gap = g["datetime_local"].diff().dt.total_seconds().div(3600)
        derived = diffs.where(hour_gap.eq(1))
        derived = derived.where(derived >= 0)
        derived = derived.where(derived <= 300)
        df_clean.loc[g.index, "rainfall_hourly_mm"] = derived
    else:
        direct = raw_rain.where(raw_rain.between(0, 300))
        df_clean.loc[g.index, "rainfall_hourly_mm"] = direct

cleaning_log_df = pd.DataFrame(cleaning_log)
cleaning_log_df.to_csv(table_dir / "epa_cleaning_log.csv", index=False)
print("\nCleaning log:")
print(cleaning_log_df.to_string(index=False))

cleaned_ntepa_file = project_folder / "NT_EPA_Air_Quality_Cleaned_2018_2024.csv"
df_clean.to_csv(cleaned_ntepa_file, index=False)
print("\nCleaned NT EPA shape:", df_clean.shape)
print("Saved:", cleaned_ntepa_file.name)

# %% [markdown]
# ## C. NASA FIRMS S-NPP VIIRS processing and 500 km station features
# The newer workflow's UTC→Darwin conversion and local-day coverage logic are retained. The older workflow's wider **500 km** scope and duplicate-removal safeguard are added.

# %%
firms_files = sorted(project_folder.glob("fire_archive_SV-C2_*.csv"))
if not firms_files:
    raise FileNotFoundError(
        "No FIRMS yearly files found. Expected files matching 'fire_archive_SV-C2_*.csv'."
    )

print("\nFIRMS files found:", len(firms_files))
for file in firms_files:
    print(" -", file.name)

station_coordinates = {
    "Palmerston": (-12.507753, 130.948253),
    "Stokes Hill": (-12.459991, 130.847847),
    "Winnellie": (-12.424017, 130.893346),
}

def haversine_distance(lat, lon, station_lat, station_lon):
    earth_radius = 6371.0088
    lat1 = np.radians(np.asarray(lat, dtype=float))
    lon1 = np.radians(np.asarray(lon, dtype=float))
    lat2 = np.radians(station_lat)
    lon2 = np.radians(station_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return 2 * earth_radius * np.arcsin(np.sqrt(a))

processed_chunks = []
firms_audit_rows = []
raw_utc_dates = set()

for file in firms_files:
    print("Processing:", file.name)
    file_rows = 0
    kept_rows = 0
    min_date = None
    max_date = None

    for chunk in pd.read_csv(file, chunksize=500_000):
        file_rows += len(chunk)

        if "acq_date" not in chunk.columns:
            raise ValueError(f"{file.name} is missing acq_date")

        parsed_dates = pd.to_datetime(chunk["acq_date"], errors="coerce")
        raw_utc_dates.update(parsed_dates.dropna().dt.normalize().tolist())
        cmin, cmax = parsed_dates.min(), parsed_dates.max()
        if pd.notna(cmin):
            min_date = cmin if min_date is None or cmin < min_date else min_date
        if pd.notna(cmax):
            max_date = cmax if max_date is None or cmax > max_date else max_date

        for col in ["latitude", "longitude", "frp"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        valid_coord = (
            chunk["latitude"].between(-90, 90)
            & chunk["longitude"].between(-180, 180)
            & parsed_dates.notna()
        )
        chunk = chunk.loc[valid_coord].copy()
        if chunk.empty:
            continue

        # Broad geographic box first, then exact station distance.
        chunk = chunk[
            chunk["latitude"].between(-17.5, -7.5)
            & chunk["longitude"].between(125.5, 136.5)
        ].copy()
        if chunk.empty:
            continue

        station_distances = []
        for station_lat, station_lon in station_coordinates.values():
            station_distances.append(
                haversine_distance(
                    chunk["latitude"].to_numpy(),
                    chunk["longitude"].to_numpy(),
                    station_lat,
                    station_lon,
                )
            )
        chunk["nearest_distance_km"] = np.min(np.vstack(station_distances), axis=0)
        chunk = chunk[chunk["nearest_distance_km"] <= PREFILTER_DISTANCE_KM].copy()
        if chunk.empty:
            continue

        if "frp" in chunk.columns:
            chunk.loc[chunk["frp"] < 0, "frp"] = np.nan

        chunk["acq_time_str"] = (
            chunk["acq_time"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
        )
        chunk["datetime_utc"] = pd.to_datetime(
            chunk["acq_date"].astype(str) + " " + chunk["acq_time_str"],
            format="%Y-%m-%d %H%M",
            utc=True,
            errors="coerce",
        )
        chunk["datetime_darwin"] = chunk["datetime_utc"].dt.tz_convert("Australia/Darwin")
        chunk["date_darwin"] = chunk["datetime_darwin"].dt.tz_localize(None).dt.normalize()
        kept_rows += len(chunk)
        processed_chunks.append(chunk)

    firms_audit_rows.append({
        "file": file.name,
        "raw_rows": file_rows,
        "prefiltered_rows": kept_rows,
        "min_acq_date": min_date,
        "max_acq_date": max_date,
    })

if not processed_chunks:
    raise ValueError("No FIRMS detections remained after the 520 km Darwin prefilter.")

firms_darwin = pd.concat(processed_chunks, ignore_index=True)
del processed_chunks

firms_file_audit = pd.DataFrame(firms_audit_rows)
firms_file_audit.to_csv(table_dir / "firms_file_audit.csv", index=False)
print("\nFIRMS file audit:")
print(firms_file_audit.to_string(index=False))

# Deduplicate detections across overlapping archive files using available detection fields.
dedup_candidates = [
    "latitude", "longitude", "brightness", "acq_date", "acq_time",
    "satellite", "instrument", "confidence", "frp", "daynight", "type"
]
dedup_key = [c for c in dedup_candidates if c in firms_darwin.columns]
duplicates_before = int(firms_darwin.duplicated(subset=dedup_key).sum())
firms_darwin = firms_darwin.drop_duplicates(subset=dedup_key, keep="first").copy()

if "type" in firms_darwin.columns:
    firms_fire = firms_darwin[firms_darwin["type"] == 0].copy()
else:
    print("WARNING: FIRMS 'type' column was not present; all retained detections are used.")
    firms_fire = firms_darwin.copy()

print("\nDarwin-region FIRMS detections after deduplication:", len(firms_darwin))
print("Duplicate detections removed:", duplicates_before)
print("Vegetation-fire detections used:", len(firms_fire))

# Exact station-specific distances.
for station_name, (station_lat, station_lon) in station_coordinates.items():
    station_key = station_name.lower().replace(" ", "_")
    firms_fire[f"distance_{station_key}_km"] = haversine_distance(
        firms_fire["latitude"].to_numpy(),
        firms_fire["longitude"].to_numpy(),
        station_lat,
        station_lon,
    )

# %%
# Distance bands preserve the newer short-range resolution and add the older 500 km scope.
distance_bins = [0, 25, 50, 100, 200, 500]
distance_labels = ["0_25km", "25_50km", "50_100km", "100_200km", "200_500km"]

def create_station_daily_features(data, station_name):
    station_key = station_name.lower().replace(" ", "_")
    distance_col = f"distance_{station_key}_km"
    station_data = data[data[distance_col] <= MAX_FIRE_DISTANCE_KM].copy()

    if station_data.empty:
        return pd.DataFrame(columns=["date_darwin", "station"])

    station_data["distance_band"] = pd.cut(
        station_data[distance_col],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
        right=True,
    )

    band_features = (
        station_data.groupby(["date_darwin", "distance_band"], observed=True)
        .agg(
            fire_count=("latitude", "size"),
            frp_sum=("frp", "sum"),
            frp_mean=("frp", "mean"),
            frp_max=("frp", "max"),
        )
        .unstack()
    )

    expected_columns = pd.MultiIndex.from_product(
        [["fire_count", "frp_sum", "frp_mean", "frp_max"], distance_labels]
    )
    band_features = band_features.reindex(columns=expected_columns)
    band_features.columns = [f"{metric}_{band}" for metric, band in band_features.columns]
    band_features = band_features.reset_index()

    overall = (
        station_data.groupby("date_darwin")
        .agg(
            fire_count_0_500km=("latitude", "size"),
            frp_sum_0_500km=("frp", "sum"),
            frp_mean_0_500km=("frp", "mean"),
            frp_max_0_500km=("frp", "max"),
            nearest_fire_km=(distance_col, "min"),
        )
        .reset_index()
    )

    station_daily = band_features.merge(overall, on="date_darwin", how="outer", validate="one_to_one")
    station_daily["station"] = station_name
    return station_daily

station_daily_fire = pd.concat(
    [create_station_daily_features(firms_fire, station) for station in station_coordinates],
    ignore_index=True,
)

all_dates = pd.date_range("2018-01-01", "2024-12-31", freq="D")
station_calendar = (
    pd.MultiIndex.from_product(
        [all_dates, list(station_coordinates.keys())],
        names=["date_darwin", "station"],
    )
    .to_frame(index=False)
)

station_fire_complete = station_calendar.merge(
    station_daily_fire,
    on=["date_darwin", "station"],
    how="left",
    validate="one_to_one",
)

# A Darwin local day spans parts of two UTC acquisition dates.
coverage = pd.DataFrame({"date_darwin": all_dates})
coverage["firms_local_day_complete"] = coverage["date_darwin"].apply(
    lambda date: int(
        date.normalize() in raw_utc_dates
        and (date - pd.Timedelta(days=1)).normalize() in raw_utc_dates
    )
)
station_fire_complete = station_fire_complete.merge(
    coverage, on="date_darwin", how="left", validate="many_to_one"
)

count_sum_cols = [
    c for c in station_fire_complete.columns
    if c.startswith("fire_count_") or c.startswith("frp_sum_")
]
fire_feature_cols = [
    c for c in station_fire_complete.columns
    if c.startswith("fire_count_")
    or c.startswith("frp_sum_")
    or c.startswith("frp_mean_")
    or c.startswith("frp_max_")
    or c == "nearest_fire_km"
]

complete_firms_days = station_fire_complete["firms_local_day_complete"].eq(1)
incomplete_firms_days = ~complete_firms_days
station_fire_complete.loc[complete_firms_days, count_sum_cols] = (
    station_fire_complete.loc[complete_firms_days, count_sum_cols].fillna(0)
)
station_fire_complete.loc[incomplete_firms_days, fire_feature_cols] = np.nan

station_fire_complete["fire_day"] = pd.NA
station_fire_complete.loc[complete_firms_days, "fire_day"] = (
    station_fire_complete.loc[complete_firms_days, "fire_count_0_500km"] > 0
).astype(int)
station_fire_complete["fire_day"] = station_fire_complete["fire_day"].astype("Int64")

print("\nFIRMS station-day rows:", len(station_fire_complete))
print("Complete FIRMS station-days:", int(complete_firms_days.sum()))
print("Incomplete FIRMS station-days:", int(incomplete_firms_days.sum()))
station_fire_complete.to_csv(table_dir / "firms_daily_station_features_500km.csv", index=False)

# %% [markdown]
# ## D. Daily NT EPA aggregation
# Daily PM₂.₅/PM₁₀ means require at least 18 valid hourly observations. The count target requires all 24 PM₂.₅ hours. Wind direction is aggregated with sine/cosine components.

# %%
# Ensure clean date fields after sorting.
df_clean["date"] = df_clean["datetime_local"].dt.normalize()

# PM2.5 daily values.
daily_pm25 = (
    df_clean.groupby(["station", "date"])
    .agg(
        pm25_mean=("pm25_ug_m3", "mean"),
        pm25_max=("pm25_ug_m3", "max"),
        pm25_valid_hours=("pm25_ug_m3", "count"),
        hours_pm25_ge_25=("pm25_ug_m3", lambda x: (x >= PM25_THRESHOLD).sum()),
    )
    .reset_index()
)
daily_pm25.loc[
    daily_pm25["pm25_valid_hours"] < MIN_VALID_PM_HOURS,
    ["pm25_mean", "pm25_max"],
] = np.nan
daily_pm25.loc[
    daily_pm25["pm25_valid_hours"] < MIN_VALID_COUNT_HOURS,
    "hours_pm25_ge_25",
] = np.nan

# PM10 daily values.
daily_pm10 = (
    df_clean.groupby(["station", "date"])
    .agg(
        pm10_mean=("pm10_ug_m3", "mean"),
        pm10_max=("pm10_ug_m3", "max"),
        pm10_valid_hours=("pm10_ug_m3", "count"),
    )
    .reset_index()
)
daily_pm10.loc[
    daily_pm10["pm10_valid_hours"] < MIN_VALID_PM_HOURS,
    ["pm10_mean", "pm10_max"],
] = np.nan
daily_pm10 = daily_pm10.drop(columns=["pm10_valid_hours"])

# Circular wind components.
wind_rad = np.deg2rad(df_clean["wind_direction_deg"])
df_clean["wind_dir_sin"] = np.sin(wind_rad)
df_clean["wind_dir_cos"] = np.cos(wind_rad)

daily_weather = (
    df_clean.groupby(["station", "date"])
    .agg(
        humidity_mean=("relative_humidity_pct", "mean"),
        temperature_mean=("air_temperature_c", "mean"),
        wind_speed_mean=("wind_speed_m_s", "mean"),
        pressure_mean=("air_pressure_hpa", "mean"),
        rainfall_total=("rainfall_hourly_mm", lambda x: x.sum(min_count=1)),
        wind_dir_sin_mean=("wind_dir_sin", "mean"),
        wind_dir_cos_mean=("wind_dir_cos", "mean"),
    )
    .reset_index()
)

daily_weather_coverage = (
    df_clean.groupby(["station", "date"])
    .agg(
        humidity_hours=("relative_humidity_pct", "count"),
        temperature_hours=("air_temperature_c", "count"),
        wind_speed_hours=("wind_speed_m_s", "count"),
        pressure_hours=("air_pressure_hpa", "count"),
        wind_direction_hours=("wind_direction_deg", "count"),
    )
    .reset_index()
)

ntepa_daily = (
    daily_pm25.merge(daily_pm10, on=["station", "date"], how="left", validate="one_to_one")
    .merge(daily_weather, on=["station", "date"], how="left", validate="one_to_one")
)

ntepa_daily["year"] = ntepa_daily["date"].dt.year
ntepa_daily["month"] = ntepa_daily["date"].dt.month
print("\nNT EPA daily rows:", len(ntepa_daily))

# %% [markdown]
# ## E. Integrate sources, create continuity-safe next-day targets and shared features
# The `target_date` column is the date being predicted. Downstream model notebooks split by `target_date`, which keeps all 2024 outcomes in the untouched test period.

# %%
fire_daily = station_fire_complete.rename(columns={"date_darwin": "date"})

integrated_daily = ntepa_daily.merge(
    fire_daily,
    on=["station", "date"],
    how="left",
    validate="one_to_one",
)

integrated_daily = integrated_daily.merge(
    daily_weather_coverage,
    on=["station", "date"],
    how="left",
    validate="one_to_one",
)

# Apply 18-hour weather coverage rule.
integrated_daily.loc[integrated_daily["humidity_hours"] < 18, "humidity_mean"] = np.nan
integrated_daily.loc[integrated_daily["temperature_hours"] < 18, "temperature_mean"] = np.nan
integrated_daily.loc[integrated_daily["wind_speed_hours"] < 18, "wind_speed_mean"] = np.nan
integrated_daily.loc[integrated_daily["pressure_hours"] < 18, "pressure_mean"] = np.nan
integrated_daily.loc[
    integrated_daily["wind_direction_hours"] < 18,
    ["wind_dir_sin_mean", "wind_dir_cos_mean"],
] = np.nan

integrated_daily["month_sin"] = np.sin(2 * np.pi * integrated_daily["month"] / 12)
integrated_daily["month_cos"] = np.cos(2 * np.pi * integrated_daily["month"] / 12)

integrated_daily = integrated_daily.sort_values(["station", "date"]).reset_index(drop=True)

# Current/past-day features only.
integrated_daily["pm25_lag1"] = integrated_daily.groupby("station")["pm25_mean"].shift(1)
integrated_daily["pm25_3day_mean"] = (
    integrated_daily.groupby("station")["pm25_mean"]
    .transform(lambda x: x.rolling(window=3, min_periods=3).mean())
)

# Continuity-safe next-day targets.
next_date = integrated_daily.groupby("station")["date"].shift(-1)
next_pm25 = integrated_daily.groupby("station")["pm25_mean"].shift(-1)
next_hours = integrated_daily.groupby("station")["hours_pm25_ge_25"].shift(-1)
is_next_calendar_day = next_date.eq(integrated_daily["date"] + pd.Timedelta(days=1))

integrated_daily["target_date"] = next_date.where(is_next_calendar_day)
integrated_daily["target_pm25_next_day"] = next_pm25.where(is_next_calendar_day)
integrated_daily["target_hours_ge_25_next_day"] = next_hours.where(is_next_calendar_day)

integrated_daily["target_pm25_over_25_next_day"] = pd.NA
valid_target = integrated_daily["target_pm25_next_day"].notna()
integrated_daily.loc[valid_target, "target_pm25_over_25_next_day"] = (
    integrated_daily.loc[valid_target, "target_pm25_next_day"] > PM25_THRESHOLD
).astype(int)
integrated_daily["target_pm25_over_25_next_day"] = integrated_daily[
    "target_pm25_over_25_next_day"
].astype("Int64")

# Model rows require actual FIRMS local-day coverage. Missing archive days are not treated as zero-fire days.
model_ready = integrated_daily[integrated_daily["firms_local_day_complete"].eq(1)].copy()

model_features = [
    "station",
    "pm25_mean", "pm25_max", "pm10_mean", "pm10_max",
    "humidity_mean", "temperature_mean", "wind_speed_mean", "pressure_mean",
    "wind_dir_sin_mean", "wind_dir_cos_mean",
    "fire_count_0_25km", "fire_count_25_50km", "fire_count_50_100km",
    "fire_count_100_200km", "fire_count_200_500km",
    "frp_sum_0_500km", "nearest_fire_km",
    "month_sin", "month_cos",
    "pm25_lag1", "pm25_3day_mean",
]

missing_features = [c for c in model_features if c not in model_ready.columns]
if missing_features:
    raise ValueError(f"Expected model features were not created: {missing_features}")

# Leakage and physical sanity checks.
assert not model_ready.duplicated(["station", "date"]).any()
assert model_ready["target_pm25_next_day"].dropna().ge(0).all()
assert model_ready["target_hours_ge_25_next_day"].dropna().between(0, 24).all()
assert model_ready.loc[
    model_ready["target_date"].notna(), "target_date"
].eq(
    model_ready.loc[model_ready["target_date"].notna(), "date"] + pd.Timedelta(days=1)
).all()

print("\nIntegrated rows before FIRMS coverage restriction:", len(integrated_daily))
print("Model-ready rows with complete FIRMS local-day coverage:", len(model_ready))
print("Final number of shared predictors:", len(model_features))

# %% [markdown]
# ## F. Save checkpoints and split evidence
# The shared model checkpoint is used by all three modelling notebooks. The feature manifest prevents each modeller from accidentally using a different predictor set.

# %%
# Split evidence is based on target_date, not feature date.
model_ready["target_year"] = model_ready["target_date"].dt.year
split_summary = []
for target in [
    "target_pm25_over_25_next_day",
    "target_pm25_next_day",
    "target_hours_ge_25_next_day",
]:
    available = model_ready.dropna(subset=[target, "target_date"]).copy()
    split_summary.append({
        "target": target,
        "train_rows_target_2018_2022": int((available["target_year"] <= 2022).sum()),
        "validation_rows_target_2023": int((available["target_year"] == 2023).sum()),
        "test_rows_target_2024": int((available["target_year"] == 2024).sum()),
        "min_target_date": available["target_date"].min(),
        "max_target_date": available["target_date"].max(),
    })
split_summary_df = pd.DataFrame(split_summary)
print("\nChronological split evidence:")
print(split_summary_df.to_string(index=False))

# Save checkpoints used by other members.
ntepa_daily.to_csv(project_folder / "Fire2Air_ntepa_daily_checkpoint.csv", index=False)
station_fire_complete.to_csv(project_folder / "Fire2Air_firms_daily_checkpoint.csv", index=False)
integrated_daily.to_csv(project_folder / "Fire2Air_integrated_daily_full.csv", index=False)
model_ready.to_csv(project_folder / "Fire2Air_model_ready_checkpoint.csv", index=False)
split_summary_df.to_csv(table_dir / "chronological_split_summary.csv", index=False)

manifest = {
    "project": "Fire2Air Darwin",
    "threshold_pm25_ug_m3": PM25_THRESHOLD,
    "max_fire_distance_km": MAX_FIRE_DISTANCE_KM,
    "split_basis": "target_date",
    "train_target_years": "2018-2022",
    "validation_target_year": 2023,
    "test_target_year": 2024,
    "features": model_features,
    "targets": {
        "classification": "target_pm25_over_25_next_day",
        "regression": "target_pm25_next_day",
        "count": "target_hours_ge_25_next_day",
    },
}
with open(project_folder / "Fire2Air_feature_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nSaved shared checkpoints:")
for name in [
    "Fire2Air_ntepa_daily_checkpoint.csv",
    "Fire2Air_firms_daily_checkpoint.csv",
    "Fire2Air_integrated_daily_full.csv",
    "Fire2Air_model_ready_checkpoint.csv",
    "Fire2Air_feature_manifest.json",
]:
    print(" -", name)

# %%
