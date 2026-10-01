from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Fire2Air Darwin
# Dashboard Data Integration
#
# Combines row-level outputs from the three A2 models:
#   Model 1: PM2.5 exceedance classification
#   Model 2: next-day PM2.5 regression
#   Model 3: elevated-hours count prediction
# ---------------------------------------------------------


# Project root
ROOT = Path(__file__).resolve().parents[2]

MODEL1_PATH = (
    ROOT
    / "Source code"
    / "outputs_prt661"
    / "tables"
    / "model1_2024_predictions.csv"
)

MODEL2_PATH = (
    ROOT
    / "Outputs"
    / "Fire2Air_regression_predictions_2024.csv"
)

MODEL3_PATH = (
    ROOT
    / "Source code"
    / "outputs_prt661"
    / "tables"
    / "model3_2024_predictions.csv"
)

CONTEXT_PATH = (
    ROOT
    / "Source code"
    / "Fire2Air_integrated_daily_full.csv"
)

OUTPUT_PATH = (
    ROOT
    / "Visualisations"
    / "dashboard"
    / "data"
    / "dashboard_predictions.csv"
)


# ---------------------------------------------------------
# Load model outputs
# ---------------------------------------------------------

model1 = pd.read_csv(
    MODEL1_PATH,
    parse_dates=["date", "target_date"]
)

model2 = pd.read_csv(
    MODEL2_PATH,
    parse_dates=["date"]
)

model3 = pd.read_csv(
    MODEL3_PATH,
    parse_dates=["date", "target_date"]
)

context = pd.read_csv(
    CONTEXT_PATH,
    parse_dates=["date", "target_date"]
)

print("Raw model outputs")
print("-----------------")
print("Model 1 rows:", len(model1))
print("Model 2 rows:", len(model2))
print("Model 3 rows:", len(model3))
print("Context rows:", len(context))

# ---------------------------------------------------------
# Standardise forecast date
# ---------------------------------------------------------

# Model 1 predicts target_date using information from date.
model1 = model1.rename(
    columns={
        "date": "predictor_date",
        "target_date": "forecast_date",
        "target_pm25_over_25_next_day": "observed_exceedance"
    }
)

# Model 2 already stores the forecast/target day as date.
model2 = model2.rename(
    columns={
        "date": "forecast_date",
        "observed_pm25_next_day": "observed_pm25",
        "predicted_pm25_next_day": "predicted_pm25"
    }
)

# Model 3 predicts target_date using information from date.
model3 = model3.rename(
    columns={
        "date": "duration_predictor_date",
        "target_date": "forecast_date"
    }
)


# ---------------------------------------------------------
# Keep dashboard-relevant fields
# ---------------------------------------------------------

model1_dashboard = model1[
    [
        "predictor_date",
        "forecast_date",
        "station",
        "observed_exceedance",
        "predicted_probability",
        "predicted_class"
    ]
].copy()

model2_dashboard = model2[
    [
        "forecast_date",
        "station",
        "observed_pm25",
        "predicted_pm25"
    ]
].copy()

model3_dashboard = model3[
    [
        "forecast_date",
        "station",
        "observed_elevated_hours",
        "predicted_elevated_hours"
    ]
].copy()

# ---------------------------------------------------------
# Prepare predictor-day environmental and fire context
# ---------------------------------------------------------

context = context.rename(
    columns={
        "date": "predictor_date",
        "target_date": "forecast_date"
    }
)

context_dashboard = context[
    [
        "predictor_date",
        "forecast_date",
        "station",
        "pm25_mean",
        "humidity_mean",
        "temperature_mean",
        "wind_speed_mean",
        "wind_dir_sin_mean",
        "wind_dir_cos_mean",
        "fire_count_0_25km",
        "fire_count_25_50km",
        "fire_count_50_100km",
        "fire_count_100_200km",
        "fire_count_200_500km",
        "fire_count_0_500km",
        "frp_sum_0_500km",
        "nearest_fire_km"
    ]
].copy()

# ---------------------------------------------------------
# Validate keys before integration
# ---------------------------------------------------------

for name, df in {
    "Model 1": model1_dashboard,
    "Model 2": model2_dashboard,
    "Model 3": model3_dashboard,
}.items():

    duplicates = df.duplicated(
        subset=["forecast_date", "station"]
    ).sum()

    print(f"{name} duplicate forecast keys: {duplicates}")

# Validate environmental/fire context keys
context_duplicates = context_dashboard.duplicated(
    subset=["predictor_date", "forecast_date", "station"]
).sum()

print(
    f"Context duplicate predictor/forecast keys: {context_duplicates}"
)

assert context_duplicates == 0

# ---------------------------------------------------------
# Integrate predictions
#
# Model 1 is the base forecast population.
# LEFT JOIN is intentional because Model 3 does not contain
# predictions for every Model 1 station-date.
# ---------------------------------------------------------

dashboard = model1_dashboard.merge(
    model2_dashboard,
    on=["forecast_date", "station"],
    how="left",
    validate="one_to_one"
)

dashboard = dashboard.merge(
    model3_dashboard,
    on=["forecast_date", "station"],
    how="left",
    validate="one_to_one"
)

dashboard = dashboard.merge(
    context_dashboard,
    on=["predictor_date", "forecast_date", "station"],
    how="left",
    validate="one_to_one"
)

# ---------------------------------------------------------
# Final checks
# ---------------------------------------------------------

dashboard = dashboard.sort_values(
    ["forecast_date", "station"]
).reset_index(drop=True)

print()
print("Dashboard integration")
print("---------------------")

print("Dashboard rows:", len(dashboard))

print(
    "Missing Model 2 predictions:",
    dashboard["predicted_pm25"].isna().sum()
)

print(
    "Missing Model 3 predictions:",
    dashboard["predicted_elevated_hours"].isna().sum()
)

print(
    "Forecast period:",
    dashboard["forecast_date"].min(),
    "to",
    dashboard["forecast_date"].max()
)

print(
    "Stations:",
    dashboard["station"].unique()
)


# Dashboard should preserve all Model 1 forecast rows.
assert len(dashboard) == len(model1_dashboard)


# ---------------------------------------------------------
# Export
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

dashboard.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("Dashboard dataset saved to:")
print(OUTPUT_PATH.resolve())

print()
print(dashboard.head())