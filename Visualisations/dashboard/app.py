from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import math

# ---------------------------------------------------------
# Fire2Air Darwin Dashboard
# ---------------------------------------------------------

st.set_page_config(
    page_title="Fire2Air Darwin",
    page_icon="🔥",
    layout="wide"
)


DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "dashboard_predictions.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["predictor_date", "forecast_date"]
    )

    return df.sort_values(
        ["forecast_date", "station"]
    )


df = load_data()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🔥 Fire2Air Darwin")

st.subheader(
    "Next-Day PM₂.₅ Prediction and Smoke Outlook"
)

st.caption(
    "Historical test predictions from the 2024 held-out evaluation period "
    "for Darwin monitoring stations."
)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

st.sidebar.header("Forecast Controls")

stations = sorted(df["station"].dropna().unique())

selected_station = st.sidebar.selectbox(
    "Monitoring station",
    stations
)

station_df = df[
    df["station"] == selected_station
].copy()

available_dates = sorted(
    station_df["forecast_date"].dt.date.unique()
)

selected_date = st.sidebar.selectbox(
    "Forecast date",
    available_dates,
    index=len(available_dates) - 1
)


selected = station_df[
    station_df["forecast_date"].dt.date == selected_date
]

if selected.empty:
    st.error("No forecast is available for this selection.")
    st.stop()

row = selected.iloc[0]


# ---------------------------------------------------------
# Next-Day Smoke Outlook
# ---------------------------------------------------------

st.header("Next-Day Smoke Outlook")


st.info(
    "Historical Test Forecast — this view shows model predictions "
    "for the selected date in the 2024 held-out test period."
)

st.write(
    f"**Station:** {selected_station}  |  "
    f"**Forecast date:** {selected_date}"
)


probability = row["predicted_probability"]

if pd.notna(probability):
    probability_text = f"{probability * 100:.1f}%"
else:
    probability_text = "N/A"


predicted_pm25 = row["predicted_pm25"]

if pd.notna(predicted_pm25):
    pm25_text = f"{predicted_pm25:.1f} µg/m³"
else:
    pm25_text = "N/A"


predicted_hours = row["predicted_elevated_hours"]

if pd.notna(predicted_hours):
    hours_text = f"{predicted_hours:.1f} h"
else:
    hours_text = "N/A"


col1, col2, col3 = st.columns(3)

col1.metric(
    "Elevated PM₂.₅ Probability",
    probability_text
)

col2.metric(
    "Expected PM₂.₅",
    pm25_text
)

col3.metric(
    "Expected Elevated Hours",
    hours_text
)

missing_models = []

if pd.isna(row["predicted_pm25"]):
    missing_models.append("PM₂.₅ concentration prediction (Model 2)")

if pd.isna(row["predicted_elevated_hours"]):
    missing_models.append("elevated-hours prediction (Model 3)")

if missing_models:
    st.warning(
        "Some model outputs are unavailable for this station-date: "
        + ", ".join(missing_models)
        + ". Available predictions are still shown."
    )

# ---------------------------------------------------------
# Model 1 classification interpretation
# ---------------------------------------------------------

CLASSIFICATION_THRESHOLD = 0.78

if pd.notna(probability):

    if probability >= CLASSIFICATION_THRESHOLD:
        classification_text = "Elevated PM₂.₅ predicted"
    else:
        classification_text = "Elevated PM₂.₅ not predicted"

    st.info(
        f"**Model 1 classification:** {classification_text}  \n"
        f"Predicted probability: {probability * 100:.1f}% | "
        f"Decision threshold: {CLASSIFICATION_THRESHOLD * 100:.0f}%"
    )

# ---------------------------------------------------------
# Predictor-day environmental context
# ---------------------------------------------------------

st.header("Current-Day Conditions")

st.caption(
    f"Conditions observed on {row['predictor_date'].strftime('%Y-%m-%d')}, "
    "used as input information for the next-day forecast."
)

# Convert circular wind components back to mean wind direction
if (
    pd.notna(row["wind_dir_sin_mean"])
    and pd.notna(row["wind_dir_cos_mean"])
):
    wind_degrees = (
        math.degrees(
            math.atan2(
                row["wind_dir_sin_mean"],
                row["wind_dir_cos_mean"]
            )
        )
        + 360
    ) % 360

    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    wind_direction = directions[
        round(wind_degrees / 45) % 8
    ]

    wind_direction_text = f"{wind_direction} ({wind_degrees:.0f}°)"

else:
    wind_direction_text = "N/A"

c1, c2, c3, c4, c5 = st.columns(5)

pm25_today = (
    f"{row['pm25_mean']:.1f} µg/m³"
    if pd.notna(row["pm25_mean"])
    else "N/A"
)

temperature = (
    f"{row['temperature_mean']:.1f} °C"
    if pd.notna(row["temperature_mean"])
    else "N/A"
)

humidity = (
    f"{row['humidity_mean']:.1f}%"
    if pd.notna(row["humidity_mean"])
    else "N/A"
)

wind_speed = (
    f"{row['wind_speed_mean']:.1f} m/s"
    if pd.notna(row["wind_speed_mean"])
    else "N/A"
)

c1.metric("PM₂.₅ Mean", pm25_today)
c2.metric("Temperature", temperature)
c3.metric("Humidity", humidity)
c4.metric("Wind Speed", wind_speed)
c5.metric("Wind Direction", wind_direction_text)

# ---------------------------------------------------------
# Predictor-day fire activity
# ---------------------------------------------------------

st.header("Nearby Fire Activity")

st.caption(
    f"Satellite fire detections associated with the predictor day "
    f"{row['predictor_date'].strftime('%Y-%m-%d')}."
)

fire_0_25 = row["fire_count_0_25km"]
fire_25_50 = row["fire_count_25_50km"]
fire_50_100 = row["fire_count_50_100km"]
fire_100_200 = row["fire_count_100_200km"]
fire_200_500 = row["fire_count_200_500km"]

f1, f2, f3, f4, f5 = st.columns(5)

f1.metric("0–25 km", f"{fire_0_25:.0f} fires")
f2.metric("25–50 km", f"{fire_25_50:.0f} fires")
f3.metric("50–100 km", f"{fire_50_100:.0f} fires")
f4.metric("100–200 km", f"{fire_100_200:.0f} fires")
f5.metric("200–500 km", f"{fire_200_500:.0f} fires")


total_fires = (
    f"{row['fire_count_0_500km']:.0f}"
    if pd.notna(row["fire_count_0_500km"])
    else "N/A"
)

total_frp = (
    f"{row['frp_sum_0_500km']:.1f} MW"
    if pd.notna(row["frp_sum_0_500km"])
    else "N/A"
)

nearest_fire = (
    f"{row['nearest_fire_km']:.1f} km"
    if pd.notna(row["nearest_fire_km"])
    else "N/A"
)

s1, s2, s3 = st.columns(3)

s1.metric("Total Fires within 500 km", total_fires)
s2.metric("Total FRP within 500 km", total_frp)
s3.metric("Nearest Fire", nearest_fire)

# ---------------------------------------------------------
# Historical model predictions
# ---------------------------------------------------------

st.header("PM₂.₅ Prediction Performance — 2024")

st.caption(
    "Comparison between predicted and observed next-day PM₂.₅ "
    f"for {selected_station} during the held-out 2024 test period."
)

history = station_df[
    [
        "forecast_date",
        "predicted_pm25",
        "observed_pm25"
    ]
].copy()

history_long = history.melt(
    id_vars="forecast_date",
    value_vars=[
        "predicted_pm25",
        "observed_pm25"
    ],
    var_name="Series",
    value_name="PM2.5"
)

history_long["Series"] = history_long["Series"].replace(
    {
        "predicted_pm25": "Predicted PM₂.₅",
        "observed_pm25": "Observed PM₂.₅"
    }
)

fig = px.line(
    history_long,
    x="forecast_date",
    y="PM2.5",
    color="Series",
    labels={
        "forecast_date": "Forecast Date",
        "PM2.5": "PM₂.₅ (µg/m³)"
    }
)

fig.add_hline(
    y=25,
    line_dash="dash",
    annotation_text="25 µg/m³ reference"
)

fig.add_vline(
    x=pd.Timestamp(selected_date).timestamp() * 1000,
    line_dash="dot",
    annotation_text="Selected date"
)

fig.update_layout(
    xaxis_title="Forecast date",
    yaxis_title="PM₂.₅ (µg/m³)",
    legend_title_text="",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Model Output Availability")

total_days = len(station_df)

model2_available = station_df["predicted_pm25"].notna().sum()
model3_available = station_df["predicted_elevated_hours"].notna().sum()

a1, a2, a3 = st.columns(3)

a1.metric(
    "Classification",
    f"{total_days}/{total_days} dates"
)

a2.metric(
    "PM₂.₅ Regression",
    f"{model2_available}/{total_days} dates"
)

a3.metric(
    "Duration Model",
    f"{model3_available}/{total_days} dates"
)

# ---------------------------------------------------------
# Model details
# ---------------------------------------------------------

with st.expander("View model output details"):

    details = pd.DataFrame(
        {
            "Measure": [
                "Observed exceedance",
                "Predicted probability",
                "Predicted class",
                "Observed PM₂.₅",
                "Predicted PM₂.₅",
                "Observed elevated hours",
                "Predicted elevated hours"
            ],
            "Value": [
                row["observed_exceedance"],
                row["predicted_probability"],
                row["predicted_class"],
                row["observed_pm25"],
                row["predicted_pm25"],
                row["observed_elevated_hours"],
                row["predicted_elevated_hours"]
            ]
        }
    )

    st.dataframe(
        details,
        hide_index=True,
        use_container_width=True
    )

# ---------------------------------------------------------
# Limitations
# ---------------------------------------------------------

st.divider()

st.subheader("Prototype limitations")

st.caption(
    "Fire2Air is a research decision-support prototype, "
    "not an official NT EPA warning system or medical "
    "advice. Predictions contain uncertainty and some "
    "station-date combinations do not have outputs from "
    "all three models."
)