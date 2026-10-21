from pathlib import Path
from datetime import datetime
import json
import base64
import os
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

st.set_page_config(
    page_title="Fire2Air Darwin | Admin Analytics Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PATHS / PROJECT FILES
# ============================================================
APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
default_export = Path(os.environ.get("FIRE2AIR_DATA_DIR", APP_DIR)).expanduser()
default_tables = REPO_ROOT / "Source code" / "outputs_prt661" / "tables"
if not default_tables.is_dir():
    default_tables = default_export / "tables" if (default_export / "tables").is_dir() else default_export
default_regression = Path.home() / "Downloads" / "Fire2Air_Model2" / "outputs_prt661_a3" / "tables"
if not default_regression.is_dir() or (default_tables / "a3_model2_final_2024_metrics.csv").is_file():
    default_regression = default_tables
DATA_ROOT = default_export.resolve()
TABLE_ROOT = Path(os.environ.get("FIRE2AIR_TABLE_DIR", default_tables)).expanduser().resolve()
REGRESSION_ROOT = Path(os.environ.get("FIRE2AIR_REGRESSION_TABLE_DIR", default_regression)).expanduser().resolve()
SEARCH_DIRS = [DATA_ROOT, DATA_ROOT / "tables", DATA_ROOT / "processed", DATA_ROOT / "metadata"]
FIRMS_DATA_DIR = Path(os.environ.get("FIRE2AIR_FIRMS_DIR", str(REPO_ROOT / "Source code" / "Datasets"))).expanduser()
ACTIVE_FIRMS_DATA_DIR = FIRMS_DATA_DIR
LOGO_PATH = next((p for p in [APP_DIR / "Fire2Air_logo.png", REPO_ROOT / "Dashboard" / "Fire2Air_logo.png"] if p.is_file()), None)
LOADED_FILES = {}
LOAD_ISSUES = []

PM25_THRESHOLD = 25.0

STATION_COORDINATES = {
    "Palmerston": (-12.507753, 130.948253),
    "Stokes Hill": (-12.459991, 130.847847),
    "Winnellie": (-12.424017, 130.893346),
}

DARWIN_LOCATIONS = {
    "Alawa": (-12.3799, 130.8739),
    "Casuarina": (-12.3740, 130.8820),
    "Coconut Grove": (-12.3978, 130.8528),
    "Darwin CBD": (-12.4634, 130.8456),
    "Howard Springs": (-12.4954, 131.0500),
    "Humpty Doo": (-12.5757, 131.1037),
    "Nightcliff": (-12.3827, 130.8527),
    "Palmerston": (-12.507753, 130.948253),
    "Rapid Creek": (-12.3817, 130.8597),
    "Stokes Hill": (-12.459991, 130.847847),
    "Winnellie": (-12.424017, 130.893346),
}


def find_file(filename):
    if filename.startswith(("a3_model2_", "a4_model2_")):
        folders = [REGRESSION_ROOT]
    elif filename.startswith(("a3_model1_", "a4_model1_", "a3_model3_", "a4_model3_")):
        folders = [TABLE_ROOT]
    else:
        folders = SEARCH_DIRS
    matches = list(dict.fromkeys((folder / filename).resolve()
                   for folder in folders if (folder / filename).is_file()))
    if len(matches) > 1:
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in matches}
        if len(hashes) > 1:
            LOAD_ISSUES.append(f"Conflicting copies of {filename}; select one export using FIRE2AIR_DATA_DIR.")
            return None
    return matches[0] if matches else None


def read_csv_safe(filename, parse_dates=None):
    path = find_file(filename)
    if path is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        for col in parse_dates or []:
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        LOADED_FILES[filename] = path
        return df
    except Exception as exc:
        LOAD_ISSUES.append(f"Could not read {filename}: {exc}")
        return pd.DataFrame()


def read_json_safe(filename):
    path = find_file(filename)
    if path is None:
        return {}
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        LOADED_FILES[filename] = path
        return result
    except Exception as exc:
        LOAD_ISSUES.append(f"Could not read {filename}: {exc}")
        return {}


def saved_threshold(metrics):
    if metrics.empty:
        return np.nan
    for col in ["Threshold", "threshold", "Selected_Threshold", "selected_threshold",
                "Decision_Threshold", "decision_threshold", "Best_Threshold"]:
        if col in metrics:
            values = pd.to_numeric(metrics[col], errors="coerce").dropna().unique()
            if len(values) == 1 and 0 <= values[0] <= 1:
                return float(values[0])
    return np.nan


def duration_diagnostics(predictions, manifest, metrics):
    """Calculate supplementary metrics only when the saved pooled errors agree."""
    target = manifest.get("targets", {}).get("count")
    actual = next((c for c in [target, "target_hours_next_day", "target_hours_pm25_ge_25_next_day",
                                "actual", "observed", "y_true"] if c and c in predictions), None)
    predicted = "bounded_prediction_0_24"
    if not actual or predicted not in predictions or "target_date" not in predictions:
        return {}, "Duration prediction file or recognised target columns unavailable."
    data = predictions.copy()
    dates = pd.to_datetime(data["target_date"], errors="coerce")
    data = data.loc[dates.dt.year.eq(2024)].copy()
    if data.empty:
        return {}, "No 2024 duration prediction records available."
    keys = [c for c in ["station", "date", "target_date"] if c in data]
    if len(keys) != 3 or data.duplicated(keys).any():
        return {}, "Duration predictions require unique station/observation/target-date records."
    y = pd.to_numeric(data[actual], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(data[predicted], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(y).all() and np.isfinite(pred).all()
            and ((y >= 0) & (y <= 24)).all() and ((pred >= 0) & (pred <= 24)).all()):
        return {}, "Duration predictions contain missing, invalid or out-of-range values."
    errors = pred-y
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    if metrics.empty:
        return {}, "Saved duration metrics are required to check the prediction cohort."
    row = metrics.iloc[0]
    for name, calculated in [("MAE",mae),("RMSE",rmse)]:
        saved = pd.to_numeric(row.get(name), errors="coerce")
        if pd.isna(saved) or not np.isclose(saved,calculated,rtol=0,atol=0.0001):
            return {}, "Duration predictions do not agree with the saved pooled metrics; supplementary metrics withheld."
    return {"Bias_PredMinusObs": float(errors.mean()),
            "Within_3h": float(np.mean(np.abs(errors)<=3)), "Rows":len(data)}, "Calculated from saved 2024 bounded duration predictions; pooled MAE/RMSE agree with the metrics file."


def darwin_dates(frame):
    if "date_darwin" in frame:
        return pd.to_datetime(frame["date_darwin"], errors="coerce").dt.normalize()
    if not {"acq_date", "acq_time"}.issubset(frame.columns):
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    times = pd.to_numeric(frame["acq_time"], errors="coerce")
    valid = times.notna() & times.eq(times.round()) & times.between(0,2359) & times.mod(100).lt(60)
    text = times.where(valid).astype("Int64").astype("string").str.zfill(4)
    stamp = pd.to_datetime(frame["acq_date"].astype("string") + " " + text,
                           format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    return stamp.dt.tz_convert("Australia/Darwin").dt.tz_localize(None).dt.normalize()


def image_data_uri(path):
    """Return a local image as a data URI so it can sit inside the header card."""
    if path is None or not Path(path).exists():
        return ""
    suffix = Path(path).suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def load_data():
    return {
        "manifest": read_json_safe("Fire2Air_feature_manifest_A3.json"),
        "outlook": read_csv_safe(
            "Fire2Air_dashboard_ready_outlook_A3.csv", ["date", "target_date"]
        ),
        "daily_air": read_csv_safe(
            "Fire2Air_ntepa_daily_checkpoint_A3.csv", ["date"]
        ),
        "model_ready": read_csv_safe(
            "Fire2Air_model_ready_checkpoint_A3.csv", ["date", "target_date"]
        ),
        "class_metrics": read_csv_safe("a3_model1_final_2024_metrics.csv"),
        "reg_metrics": read_csv_safe("a3_model2_final_2024_metrics.csv"),
        "count_metrics": read_csv_safe("a3_model3_final_2024_metrics.csv"),
        "reg_predictions": read_csv_safe(
            "a3_model2_2024_predictions.csv", ["date", "target_date"]
        ),
        "class_predictions": read_csv_safe("a3_model1_2024_predictions.csv", ["date", "target_date"]),
        "count_predictions": read_csv_safe("a3_model3_2024_predictions.csv", ["date", "target_date"]),
        "duration_strata": read_csv_safe("a3_model3_2024_duration_strata.csv"),
        "prolonged_cases": read_csv_safe("a4_model3_prolonged_event_cases.csv"),
        "duration_baseline": read_csv_safe("a4_model3_duration_baseline_comparison.csv"),
        "permutation": read_csv_safe("a3_model3_count_permutation_importance.csv"),
        "shap": read_csv_safe("a3_model3_shap_importance.csv"),
    }


DATA = load_data()
for issue in LOAD_ISSUES:
    st.error(issue)
if LOAD_ISSUES:
    st.stop()
CLASS_THRESHOLD = saved_threshold(DATA["class_metrics"])
status_paths = list(dict.fromkeys(p for p in [
    DATA_ROOT / "latest_preprocessing_attempt.json",
    DATA_ROOT.parent / "latest_preprocessing_attempt.json"] if p.is_file()))
for status_path in status_paths:
    try:
        state = json.loads(status_path.read_text(encoding="utf-8"))
        state_name = str(state.get("status", "unknown")).lower()
        if state_name not in {"success", "succeeded", "successful", "completed"}:
            st.warning(f"Latest preprocessing attempt status: {state_name}. Displayed files may belong to an earlier run; they are not a successful refresh.")
    except Exception as exc:
        st.warning(f"Could not verify preprocessing attempt status: {exc}")


if DATA["manifest"]:
    PM25_THRESHOLD = float(DATA["manifest"].get("threshold_pm25_ug_m3", 25.0))


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_station(lat, lon):
    vals = []
    for station, (slat, slon) in STATION_COORDINATES.items():
        vals.append((station, float(haversine_km(lat, lon, slat, slon))))
    return sorted(vals, key=lambda x: x[1])[0]


def safe_num(v, fmt=".1f"):
    try:
        return format(float(v), fmt) if pd.notna(v) else "—"
    except Exception:
        return "—"


def first_available(row, names, default=np.nan):
    for name in names:
        try:
            value = row.get(name, np.nan)
        except Exception:
            value = np.nan
        if pd.notna(value):
            return value
    return default


def risk_label(prob):
    if pd.isna(prob):
        return "Unknown"
    if prob < .30:
        return "Low"
    if prob < .50:
        return "Moderate"
    if prob < .70:
        return "High"
    return "Very High"


def gauge(value, title, suffix="", max_value=100, threshold=None):
    v = 0 if pd.isna(value) else float(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={"suffix": suffix, "font": {"size": 32}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, max_value]},
            "bar": {"color": "#12345f", "thickness": .24},
            "steps": [
                {"range": [0, max_value*.35], "color": "#31c27c"},
                {"range": [max_value*.35, max_value*.65], "color": "#ffd03e"},
                {"range": [max_value*.65, max_value*.82], "color": "#ff8a32"},
                {"range": [max_value*.82, max_value], "color": "#e8413a"},
            ],
            "threshold": ({"line": {"color": "#8b0000", "width": 4}, "thickness": .8, "value": threshold}
                          if threshold is not None else None),
        },
    ))
    fig.update_layout(
        height=225,
        margin=dict(l=10, r=10, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#12345f"},
    )
    return fig


def load_fires_for_date(date_text):
    selected_date = pd.Timestamp(date_text).normalize()
    year_files = [
        p for p in ACTIVE_FIRMS_DATA_DIR.glob("fire_archive_SV-C2_*.csv")
        if any(str(y) in p.name for y in [selected_date.year, (selected_date-pd.Timedelta(days=1)).year])
    ]
    parts = []
    for file_path in year_files:
        try:
            for chunk in pd.read_csv(file_path, chunksize=250_000):
                if not {"latitude", "longitude", "acq_date"}.issubset(chunk.columns):
                    continue
                dates = darwin_dates(chunk)
                mask = dates.eq(selected_date)
                if not mask.any():
                    continue
                keep = [c for c in [
                    "latitude", "longitude", "acq_date", "acq_time", "frp",
                    "type", "confidence", "brightness", "bright_ti4"
                ] if c in chunk.columns]
                parts.append(chunk.loc[mask, keep].copy())
        except Exception as exc:
            print(f"Could not read FIRMS file {file_path.name}: {exc}")

    if not parts:
        compact = find_file("Fire2Air_dashboard_fire_points_2018_2024.csv")
        if compact is not None:
            try:
                df = pd.read_csv(compact)
                date_col = next((c for c in ["date_darwin", "acq_date", "date"] if c in df.columns), None)
                if date_col:
                    d = darwin_dates(df) if date_col != "date" else pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
                    df = df.loc[d.eq(selected_date)].copy()
                    if not df.empty:
                        parts.append(df)
            except Exception as exc:
                print(f"Could not read compact fire points: {exc}")

    if not parts:
        return pd.DataFrame()

    fire = pd.concat(parts, ignore_index=True)
    fire["latitude"] = pd.to_numeric(fire["latitude"], errors="coerce")
    fire["longitude"] = pd.to_numeric(fire["longitude"], errors="coerce")
    fire["frp"] = pd.to_numeric(fire.get("frp", pd.Series(np.nan, index=fire.index)), errors="coerce")
    if "type" in fire.columns:
        fire_type = pd.to_numeric(fire["type"], errors="coerce")
        fire = fire.loc[fire_type.eq(0)].copy()
    fire = fire.dropna(subset=["latitude", "longitude"])
    fire = fire[
        fire["latitude"].between(-17.5, -7.5)
        & fire["longitude"].between(125.5, 136.5)
    ].copy()
    return fire.drop_duplicates()


# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
:root {--ink:#173451;--muted:#62758a;--line:#dce5ee;--section-gap:.7rem;}
.stApp {background:#f4f7fb!important;color:#173451;}
.block-container {max-width:1560px;padding:2.8rem 1.6rem 1.2rem!important;}
header[data-testid="stHeader"] {background:rgba(244,247,251,.95);}
footer {visibility:hidden;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #dce5ee;}
[data-testid="stSidebar"] .block-container {padding:1rem!important;}
[data-testid="stVerticalBlock"] {gap:.55rem;}
[data-testid="stHorizontalBlock"] {gap:1rem;align-items:stretch;}
.header-card {display:flex;align-items:center;gap:14px;min-height:100px;padding:12px 16px;background:#fff;border:1px solid #dce5ee;border-radius:16px;box-sizing:border-box;}
.header-logo,.header-logo-fallback {width:64px;height:64px;object-fit:contain;flex:0 0 64px;}
.header-logo-fallback {display:flex;align-items:center;justify-content:center;background:#e9f3fc;border-radius:16px;font-size:2rem;}
.admin-title {font-size:1.55rem;font-weight:800;color:#153c63;line-height:1.2;}
.admin-dashboard-title {font-size:1rem;font-weight:650;margin-top:3px;}
.admin-sub {font-size:.76rem;color:#62758a;margin-top:4px;}
.metric-card {height:112px;box-sizing:border-box;background:#fff!important;border:1px solid #dce5ee;border-top:3px solid #387bb6;border-radius:12px;padding:12px 14px;}
.metric-label {font-size:.78rem;font-weight:650;color:#526b82;}
.metric-value {font-size:1.5rem;font-weight:800;margin-top:6px;color:#173451;}
.metric-sub {font-size:.71rem;line-height:1.35;color:#62758a;margin-top:5px;}
.kpi-purple {border-top-color:#8876c7;}.kpi-orange {border-top-color:#dc9d42;}.kpi-red {border-top-color:#bd7481;}
.section-band {display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:44px;margin:.65rem 0 .3rem;padding:10px 14px;border:1px solid #dce5ee;border-radius:10px;background:#eaf1f8;color:#173f65;font-weight:750;}
.section-title {font-size:1rem;font-weight:750;color:#173f65;margin:.35rem 0 .7rem;}
[data-testid="stMetric"] {height:100%;min-height:104px;box-sizing:border-box;padding:14px;background:#fff;border:1px solid #dce5ee;border-radius:12px;}
[data-testid="stMetricLabel"] {font-size:.8rem;color:#526b82;}
[data-testid="stMetricValue"] {font-size:1.8rem;color:#173451;font-weight:700;}
[data-testid="stMetricLabel"] p {white-space:normal!important;}
[data-testid="stAlert"] {min-height:0!important;border-radius:10px;font-size:.88rem;}
[data-testid="stCaptionContainer"] {font-size:.78rem;line-height:1.5;color:#62758a;margin:0;}
[data-baseweb="select"]>div {border-radius:10px;border-color:#dce5ee;background:white;min-height:44px;}
.stPlotlyChart {background:white;border:1px solid #dce5ee;border-radius:14px;padding:5px;}
[data-testid="stDataFrame"] {border:1px solid #dce5ee;border-radius:12px;overflow:hidden;}
[data-baseweb="tab-list"] {gap:8px;margin-top:14px;border-bottom:1px solid #dce5ee;}
[data-baseweb="tab"] {height:44px;padding:10px 16px;background:#e8eef5;border-radius:9px 9px 0 0;white-space:nowrap;color:#45617c;}
[data-baseweb="tab"][aria-selected="true"] {background:#173f65;color:#fff;}
[data-baseweb="tab-highlight"] {background:#173f65;}
.footer-note {margin-top:1rem;border-top:1px solid #dce5ee;padding:14px 6px;color:#62758a;font-size:.75rem;text-align:center;}
@media(max-width:800px) {.block-container {padding:2.8rem .8rem 1rem!important;}.section-band {flex-wrap:wrap;}.metric-sub {font-size:.67rem;}}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SELECTORS
# ============================================================
outlook = DATA["outlook"]
daily_air = DATA["daily_air"]
model_ready = DATA["model_ready"]

if outlook.empty:
    st.error("Missing Fire2Air_dashboard_ready_outlook_A3.csv. Run the integration workflow first.")
    st.stop()

required = {"station", "date", "target_date", "elevated_pm25_probability",
            "expected_daily_mean_pm25", "expected_elevated_hours"}
missing = required.difference(outlook.columns)
if missing:
    st.error(f"Outlook is missing required columns: {sorted(missing)}")
    st.stop()
if outlook.duplicated(["station", "date", "target_date"]).any():
    st.error("Duplicate station/date records in the outlook. Regenerate integration.")
    st.stop()


for df, cols in [(outlook,["date","target_date"]),(daily_air,["date"]),(model_ready,["date","target_date"])]:
    for col in cols:
        if not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

if outlook[["date", "target_date"]].isna().any().any():
    st.error("Invalid or missing outlook dates. Regenerate the integrated output.")
    st.stop()
if not (outlook["target_date"].dt.normalize() == outlook["date"].dt.normalize()+pd.Timedelta(days=1)).all():
    st.error("Outlook contains records whose target date is not the next day.")
    st.stop()
for col, low, high in [("elevated_pm25_probability",0,1), ("expected_elevated_hours",0,24)]:
    values = pd.to_numeric(outlook[col], errors="coerce")
    if (values.notna() & ~values.between(low,high)).any():
        st.error(f"Out-of-range values in {col}.")
        st.stop()
    outlook[col] = values
outlook["expected_daily_mean_pm25"] = pd.to_numeric(outlook["expected_daily_mean_pm25"], errors="coerce")
logo_uri = image_data_uri(LOGO_PATH)

nav_header, nav_location, nav_date = st.columns([2.3, 1.6, 1.25], gap="medium")

with nav_header:
    if logo_uri:
        logo_html = f'<img class="header-logo" src="{logo_uri}" alt="Fire2Air logo">'
    else:
        logo_html = '<div class="header-logo-fallback">🔥</div>'

    st.markdown(
        f"""
        <div class="header-card">
          {logo_html}
          <div>
            <div class="admin-title">Fire2Air Darwin</div>
            <div class="admin-dashboard-title">Admin Analytics Dashboard</div>
            <div class="admin-sub">Historical evaluation · model diagnostics · data quality</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_location:
    location_name = st.selectbox(
        "📍 Search / select location",
        sorted(DARWIN_LOCATIONS),
        index=sorted(DARWIN_LOCATIONS).index("Palmerston"),
    )

lat, lon = DARWIN_LOCATIONS[location_name]
station_name, station_distance = nearest_station(lat, lon)
station_outlook = outlook[outlook["station"].astype(str).eq(station_name)].sort_values("target_date")
if station_outlook.empty:
    st.warning(f"No outlook data is available for {station_name}.")
    st.stop()
available_dates = station_outlook["target_date"].dropna().dt.date.drop_duplicates().tolist()
with nav_date:
    selected_date = st.selectbox("📅 Forecast target date", available_dates, index=len(available_dates)-1)
selected_ts = pd.Timestamp(selected_date)
row = station_outlook[station_outlook["target_date"].dt.date.eq(selected_date)].tail(1)
if row.empty:
    st.stop()
r = row.iloc[0]
predictor_date = pd.Timestamp(r.get("date", selected_ts-pd.Timedelta(days=1))).normalize()

prob = first_available(r,["elevated_pm25_probability","predicted_probability"])
pred_pm = first_available(r,["expected_daily_mean_pm25","predicted_pm25","prediction"])
hours = first_available(r,["expected_elevated_hours","predicted_hours_ge_25"])
fsi = first_available(r,["fsi_total","fire_smoke_influence_index"])
nearest_fire = first_available(r,["nearest_upwind_fire_km","nearest_fire_km"])
upwind_frp = first_available(r,["upwind_frp_sum","wind_aligned_frp","aligned_frp_sum"])
reliability = str(r.get("input_data_completeness", r.get("input_reliability","Unknown")))
influence = str(r.get("fire_smoke_influence","Unknown")) if pd.notna(fsi) else "Unknown"
st.info("Historical forecasts — not live. Research prototype; low predicted probability does not rule out elevated pollution.")
st.caption(f"Observation date: {predictor_date.date()} · Forecast target date: {selected_date}")
issue_value = next((r.get(k) for k in ["forecast_issue_time", "forecast_issued_at", "issue_timestamp"] if pd.notna(r.get(k))), None)
st.caption(f"Recorded forecast issue timestamp: {issue_value}" if issue_value is not None
           else "Forecast issue timestamp: not recorded; this is a historical reconstruction.")
if pd.isna(prob) or pd.isna(pred_pm) or pd.isna(hours):
    st.warning("One or more model outputs are missing for this record. A complete combined forecast is unavailable.")

# ============================================================
# SAVED EVALUATION FILES
# ============================================================
# Evaluation is displayed from saved files; no silent reconstruction on a smaller cohort.

# ============================================================
# SYSTEM OVERVIEW
# ============================================================
st.markdown(f'<div class="section-band"><span>🧭 System Overview</span><span style="font-size:.75rem;color:#6d8094">Nearest station: {station_name} · {station_distance:.1f} km</span></div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5, gap="small")
start_year = int(model_ready["date"].dt.year.min()) if not model_ready.empty and "date" in model_ready and model_ready["date"].notna().any() else 2018
end_year = int(model_ready["date"].dt.year.max()) if not model_ready.empty and "date" in model_ready and model_ready["date"].notna().any() else 2024
predictor_count = len(DATA["manifest"].get("features",[])) if DATA["manifest"] else max(0,model_ready.shape[1]-5) if not model_ready.empty else 0
cards = [
    (c1,"📍 Monitoring Stations",str(len(STATION_COORDINATES)),"Greater Darwin monitoring network","kpi-blue"),
    (c2,"📅 Study Period",f"{start_year}–{end_year}","Integrated historical analysis","kpi-red"),
    (c3,"🗃️ Predictors",str(predictor_count),"Fire, weather, AQ and location features","kpi-orange"),
    (c4,"🧠 Models","3","Classification · Regression · Count","kpi-purple"),
    (c5,"🔥 Fire Study Range","0–500 km","Distance bands around stations","kpi-red"),
]
for col,label,value,sub,css in cards:
    with col:
        st.markdown(f'<div class="metric-card {css}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

# ============================================================
# TOMORROW SMOKE OUTLOOK
# ============================================================
st.markdown(f'<div class="section-band"><span>☁️ Historical Next-Day Smoke Outlook</span><span style="font-size:.75rem;color:#6d8094">{selected_ts.strftime("%a %d %b %Y")}</span></div>', unsafe_allow_html=True)
g1,g2,g3,g4 = st.columns([1,1,1,1.05], gap="small")
with g1:
    st.plotly_chart(gauge(prob*100 if pd.notna(prob) else np.nan,"Model 1 – Classification","%",100,100*CLASS_THRESHOLD if pd.notna(CLASS_THRESHOLD) else None), width="stretch", config={"displayModeBar":False})
    st.caption("Estimated probability of an elevated next-day daily mean")
    if pd.notna(CLASS_THRESHOLD):
        st.caption(f"Saved decision threshold: {CLASS_THRESHOLD:.2f}")
        if pd.notna(prob):
            st.caption("Model decision: " + ("Elevated" if prob >= CLASS_THRESHOLD else "Not elevated"))
    else:
        st.caption("Decision threshold unavailable in saved metrics; classification decision not inferred.")
with g2:
    pm_max = max(60,float(pred_pm)*1.35) if pd.notna(pred_pm) else 60
    st.plotly_chart(gauge(pred_pm,"Model 2 – Regression"," µg/m³",pm_max,PM25_THRESHOLD), width="stretch", config={"displayModeBar":False})
    st.caption(f"Expected next-day mean PM2.5 · threshold {PM25_THRESHOLD:g} µg/m³")
with g3:
    st.plotly_chart(gauge(hours,"Model 3 – Count Model"," h",24,12), width="stretch", config={"displayModeBar":False})
    st.caption("Expected next-day elevated PM2.5 hours")
with g4:
    a,b = st.columns(2)
    with a:
        st.metric("🔥 Fire Smoke Influence", safe_num(fsi,".2f"), influence)
        st.metric("📍 Nearest Upwind Fire", f"{safe_num(nearest_fire)} km")
    with b:
        st.metric("🌬️ Upwind / Aligned FRP", f"{safe_num(upwind_frp)} MW")
        st.metric("🛡️ Input Data Completeness", reliability)

# ============================================================
# AIR QUALITY MONITORING
# ============================================================
air_tab, fire_tab, performance_tab, events_tab, diagnostics_tab = st.tabs([
    "Air quality", "Fire context", "Model performance", "Elevated events", "Diagnostics & inputs"])

with air_tab:
    st.markdown(
        f'<div class="section-band"><span>📊 Air Quality Monitoring ({station_name})</span>'
        '<span style="font-size:.75rem;color:#6d8094">PM2.5 trends · exceedances · station comparison</span></div>',
        unsafe_allow_html=True,
    )

    # All three widgets deliberately use the same Plotly height and internal margins.
    AQ_CHART_HEIGHT = 310
    aq1, aq2, aq3 = st.columns([1.75, 1.00, 1.05], gap="small")

    station_air = (
        daily_air[daily_air["station"].astype(str).eq(station_name)].copy()
        if not daily_air.empty and "station" in daily_air.columns
        else pd.DataFrame()
    )
    y_col = next(
        (c for c in ["pm25_mean", "pm2_5_mean", "pm25"] if c in station_air.columns),
        None,
    )

    with aq1:
        if not station_air.empty and y_col:
            start = max(station_air["date"].min(), selected_ts - pd.Timedelta(days=365))
            period = station_air[station_air["date"].between(start, selected_ts)]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=period["date"],
                    y=period[y_col],
                    mode="lines",
                    name="Daily mean PM2.5",
                    line=dict(color="#168ee2", width=2.2),
                )
            )
            fig.add_hline(
                y=PM25_THRESHOLD,
                line_dash="dash",
                line_color="#e6403a",
                annotation_text=f"Threshold {PM25_THRESHOLD:g}",
            )
            fig.update_layout(
                height=AQ_CHART_HEIGHT,
                title="PM2.5 Time Series",
                margin=dict(l=28, r=14, t=46, b=30),
                yaxis_title="PM2.5 (µg/m³)",
                xaxis_title=None,
                legend=dict(orientation="h", y=1.10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Daily PM2.5 data not available.")

    with aq2:
        if not station_air.empty and y_col:
            year_df = station_air[
                station_air["date"].dt.year.eq(selected_ts.year)
            ].copy()

            exceed = int((year_df[y_col] >= PM25_THRESHOLD).sum())
            monthly = (
                year_df.assign(month=year_df["date"].dt.month)
                .groupby("month")[y_col]
                .apply(lambda s: int((s >= PM25_THRESHOLD).sum()))
                .reindex(range(1, 13), fill_value=0)
            )

            fig = go.Figure(
                go.Bar(
                    x=list(range(1, 13)),
                    y=monthly.values,
                    marker_color="#ef4b46",
                )
            )
            fig.add_annotation(
                x=.03,
                y=.96,
                xref="paper",
                yref="paper",
                text=f"<b>{exceed} days</b><br><span style='font-size:11px'>above {PM25_THRESHOLD:g} µg/m³</span>",
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,.86)",
                bordercolor="#e3e3e3",
                borderpad=6,
            )
            fig.update_layout(
                height=AQ_CHART_HEIGHT,
                title=f"PM2.5 Exceedance Days ({selected_ts.year})",
                margin=dict(l=28, r=14, t=46, b=30),
                xaxis_title="Month",
                yaxis_title="Days",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Exceedance data not available.")

    with aq3:
        all_y = next(
            (c for c in ["pm25_mean", "pm2_5_mean", "pm25"] if c in daily_air.columns),
            None,
        ) if not daily_air.empty else None

        if not daily_air.empty and "station" in daily_air.columns and all_y:
            comp = (
                daily_air[daily_air["date"].dt.year.eq(selected_ts.year)]
                .groupby("station", as_index=False)[all_y]
                .mean()
                .sort_values(all_y)
            )

            fig = px.bar(
                comp,
                x=all_y,
                y="station",
                orientation="h",
                title=f"Station Comparison ({selected_ts.year})",
                labels={all_y: "Mean PM2.5", "station": ""},
            )
            fig.update_traces(marker_color="#2f8ee5")
            fig.update_layout(
                height=AQ_CHART_HEIGHT,
                margin=dict(l=28, r=14, t=46, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="white",
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Station comparison data not available.")

    # ============================================================
    # FIRE MAP + ANALYTICS
    # ============================================================

with fire_tab:
    st.markdown(f'<div class="section-band"><span>🔥 Fire Intelligence & FSI ({location_name})</span><span style="font-size:.75rem;color:#6d8094">Historical detections · intensity · distance · fire-smoke relationships</span></div>', unsafe_allow_html=True)
    map_col,analytics_col=st.columns([1.3,1.25],gap="small")
    fires=load_fires_for_date(str(predictor_date.date()))
    with map_col:
        st.markdown('<div class="section-title">🗺️ Historical Fire Detection Map</div>',unsafe_allow_html=True)
        st.caption(f"Predictor date: {predictor_date.date()} · FIRMS detections loaded: {len(fires):,}")
        st.caption("No loaded detections does not establish zero fires or complete satellite coverage. Map uses the observation day, not the full 48-hour feature window.")
        if HAS_FOLIUM:
            fmap=folium.Map(location=[lat,lon],zoom_start=8,tiles="OpenStreetMap",control_scale=True)
            folium.Marker([lat,lon],tooltip=f"Admin location: {location_name}",popup=f"<b>{location_name}</b><br>Admin reference point",icon=folium.Icon(color="blue",icon="home")).add_to(fmap)
            for name,(slat,slon) in STATION_COORDINATES.items():
                folium.CircleMarker([slat,slon],radius=7,color="#0d74c8",fill=True,fill_opacity=.95,tooltip=f"Air quality station: {name}").add_to(fmap)
            if not fires.empty:
                fires=fires.copy()
                fires["distance_to_admin_km"]=haversine_km(fires["latitude"].to_numpy(),fires["longitude"].to_numpy(),lat,lon)
                fires=fires[fires["distance_to_admin_km"]<=500].copy()
                def fire_intensity(frp):
                    if pd.isna(frp): return "Unknown","#6b7280",22
                    frp=float(frp)
                    if frp<10: return "Low","#f6b73c",24
                    if frp<50: return "Moderate","#f57c00",30
                    return "High","#d62828",38
                for _,fr in fires.sort_values("frp",ascending=False,na_position="last").head(800).iterrows():
                    level,color,size=fire_intensity(fr.get("frp",np.nan))
                    frp_v=fr.get("frp",np.nan); dist=fr.get("distance_to_admin_km",np.nan)
                    html=f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{color};border:3px solid white;box-shadow:0 2px 7px rgba(0,0,0,.4);display:flex;align-items:center;justify-content:center;font-size:{max(15,int(size*.6))}px">🔥</div>'
                    folium.Marker([float(fr["latitude"]),float(fr["longitude"])],icon=folium.DivIcon(html=html,icon_size=(size,size),icon_anchor=(size//2,size//2)),tooltip=f"🔥 {level} · {safe_num(frp_v)} MW · {safe_num(dist)} km from {location_name}",popup=f"<b>Active fire detection</b><br>Intensity: {level}<br>FRP: {safe_num(frp_v)} MW<br>Distance to admin: {safe_num(dist)} km").add_to(fmap)
                legend='''<div style="position:fixed;bottom:28px;left:28px;z-index:9999;background:white;border:1px solid #d7e1ea;border-radius:10px;padding:9px 11px;font-size:12px;color:#153d70;box-shadow:0 2px 8px rgba(0,0,0,.15)"><b>🔥 Fire intensity (FRP)</b><br><span style="color:#f6b73c">●</span> Low &lt; 10 MW<br><span style="color:#f57c00">●</span> Moderate 10–49.9 MW<br><span style="color:#d62828">●</span> High ≥ 50 MW</div>'''
                fmap.get_root().html.add_child(folium.Element(legend))
            st_folium(fmap,height=520,width="stretch",returned_objects=[],key=f"admin_map_{location_name}_{selected_date}")
        else:
            st.info("Install folium and streamlit-folium for the interactive map.")
    with analytics_col:
        a,b,c=st.columns(3)
        fire_count=len(fires) if not fires.empty else 0
        total_frp = float(fires["frp"].sum()) if not fires.empty and "frp" in fires and fires["frp"].notna().all() else np.nan
        with a: st.metric("🔥 Fire Count",f"{fire_count:,}")
        with b: st.metric("🔥 Total FRP",f"{safe_num(total_frp,'.0f')} MW")
        with c: st.metric("🌫️ FSI",safe_num(fsi,".2f"),influence)
        if not model_ready.empty:
            fsi_col=next((c for c in ["fsi_total","fire_smoke_influence_index"] if c in model_ready.columns),None)
            pm_col=next((c for c in ["target_pm25_next_day","pm25_mean","pm25"] if c in model_ready.columns),None)
            if fsi_col and pm_col:
                s=model_ready[[fsi_col,pm_col]].dropna()
                if len(s)>2500: s=s.sample(2500,random_state=42)
                fig=px.scatter(s,x=fsi_col,y=pm_col,title="FSI vs PM2.5",labels={fsi_col:"Fire Smoke Influence Index",pm_col:"PM2.5"})
                fig.update_traces(marker=dict(size=6,opacity=.55,color="#ff7f2a")); fig.update_layout(height=205,margin=dict(l=20,r=10,t=40,b=24),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
                st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
            aligned_col=next((c for c in ["upwind_frp_sum","wind_aligned_frp","aligned_frp_sum","frp_upwind"] if c in model_ready.columns),None)
            if aligned_col and pm_col:
                s=model_ready[[aligned_col,pm_col]].dropna()
                if len(s)>2500: s=s.sample(2500,random_state=42)
                fig=px.scatter(s,x=aligned_col,y=pm_col,title="Aligned / Upwind FRP vs PM2.5",labels={aligned_col:"Aligned / Upwind FRP",pm_col:"PM2.5"})
                fig.update_traces(marker=dict(size=6,opacity=.55,color="#168ee2")); fig.update_layout(height=205,margin=dict(l=20,r=10,t=40,b=24),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
                st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})

    # ============================================================
    # MODEL PERFORMANCE
    # ============================================================

with performance_tab:
    st.markdown('<div class="section-band"><span>🎯 Model Performance</span><span style="font-size:.75rem;color:#6d8094">Final test metrics where available</span></div>',unsafe_allow_html=True)
    mp1,mp2,mp3=st.columns(3,gap="small")
    with mp1:
        st.markdown('<div class="section-title">Model 1 – Classification</div>',unsafe_allow_html=True)
        df=DATA["class_metrics"]
        if not df.empty:
            rr=df.iloc[0]; a,b,c=st.columns(3); a.metric("Accuracy",safe_num(rr.get("Accuracy"),".3f")); b.metric("Recall",safe_num(rr.get("Recall"),".3f")); c.metric("F1",safe_num(rr.get("F1"),".3f")); a,b,c=st.columns(3); a.metric("Precision",safe_num(rr.get("Precision"),".3f")); b.metric("PR-AUC",safe_num(rr.get("PR_AUC"),".3f")); c.metric("ROC-AUC",safe_num(rr.get("ROC_AUC"),".3f"))
        else: st.info("Model 1 metrics file not found.")
    with mp2:
        st.markdown('<div class="section-title">Model 2 – Regression</div>', unsafe_allow_html=True)
        df = DATA["reg_metrics"]

        if not df.empty:
            if "Scope" in df.columns and df["Scope"].astype(str).eq("All 2024 model rows").any():
                rr = df[df["Scope"].astype(str).eq("All 2024 model rows")].iloc[0]
            else:
                rr = df.iloc[0]

            a, b, c = st.columns(3)
            a.metric("RMSE", safe_num(first_available(rr, ["RMSE"]), ".2f"))
            b.metric("MAE", safe_num(first_available(rr, ["MAE"]), ".2f"))
            c.metric("R²", safe_num(first_available(rr, ["R2", "R_squared"]), ".3f"))

            a, b = st.columns(2)
            a.metric(
                "Bias",
                safe_num(
                    first_available(
                        rr,
                        ["Bias_PredMinusObs", "Bias", "bias"],
                    ),
                    ".2f",
                ),
            )
            b.metric(
                "Underprediction",
                (
                    f"{100 * float(first_available(rr, ['Underprediction_Rate', 'Underprediction'])):.1f}%"
                    if pd.notna(first_available(rr, ["Underprediction_Rate", "Underprediction"]))
                    else "—"
                ),
            )
        else:
            st.info(
                f"Regression metrics are missing from {REGRESSION_ROOT}. "
                "Place a3_model2_final_2024_metrics.csv in the configured regression tables folder."
            )
    with mp3:
        st.markdown('<div class="section-title">Model 3 – Count</div>',unsafe_allow_html=True)
        df=DATA["count_metrics"]
        if not df.empty:
            rr=df.iloc[0]; a,b,c=st.columns(3); a.metric("RMSE",safe_num(first_available(rr,["RMSE"]),".2f")); b.metric("MAE",safe_num(first_available(rr,["MAE"]),".2f")); c.metric("R²",safe_num(first_available(rr,["R2","R_squared"]),".3f")); a,b=st.columns(2); extra, diagnostic_note = duration_diagnostics(DATA["count_predictions"], DATA["manifest"], df)
            within = first_available(rr,["Within_3h","Within_3_hours"])
            bias = first_available(rr,["Bias_PredMinusObs","Bias_hours","Bias","bias"])
            if pd.isna(within): within = extra.get("Within_3h", np.nan)
            if pd.isna(bias): bias = extra.get("Bias_PredMinusObs", np.nan)
            a.metric("Within ±3 h", f"{100*float(within):.1f}%" if pd.notna(within) and 0 <= float(within) <= 1 else "—")
            b.metric("Bias (hours)", safe_num(bias,".2f") if pd.notna(bias) else "—", help="Mean predicted minus observed duration. Positive values indicate overprediction.")
            st.caption(diagnostic_note)
        else: st.info("Model 3 metrics file not found.")

    # ============================================================
    # ELEVATED-EVENT DIAGNOSTICS
    # ============================================================

with events_tab:
    with st.container():
        cp = DATA["class_predictions"]
        target_col = DATA["manifest"].get("targets", {}).get("classification", "target_elevated_next_day")
        actual_col = next((c for c in [target_col, "actual", "y_true", "observed"] if c in cp), None)
        decision_col = next((c for c in ["predicted_class", "elevated_pm25_prediction"] if c in cp), None)
        if actual_col and decision_col:
            valid = cp.copy()
            if "target_date" in valid:
                valid = valid[valid["target_date"].dt.year.eq(2024)]
            obs = pd.to_numeric(valid[actual_col], errors="coerce")
            dec = pd.to_numeric(valid[decision_col], errors="coerce")
            mask = obs.isin([0,1]) & dec.isin([0,1])
            obs, dec = obs[mask], dec[mask]
            positives = int(obs.eq(1).sum())
            missed = int((obs.eq(1) & dec.eq(0)).sum())
            detected = int((obs.eq(1) & dec.eq(1)).sum())
            st.write(f"Saved classification predictions: {detected} detected and {missed} missed elevated station-days out of {positives} positives ({len(obs)} eligible records).")
            if positives:
                st.write(f"Recall recalculated from these records: {detected/positives:.1%}")
        else:
            st.info("Add the classification prediction table with observed and predicted classes to inspect missed events.")
        st.caption("Concentration errors and missed events can be much larger than pooled averages suggest. Assess station-level and elevated-event results together.")
        for title, key in [("Duration errors by observed-hour stratum", "duration_strata"),
                           ("Prolonged-event cases", "prolonged_cases"),
                           ("Duration baseline comparison", "duration_baseline")]:
            st.markdown(f"**{title}**")
            if DATA[key].empty:
                st.info("Supporting result table is unavailable in the selected export.")
            else:
                st.dataframe(DATA[key], hide_index=True)
        st.caption("Prolonged station-day observations are not necessarily independent pollution episodes. Metric uncertainty intervals are not individual forecast intervals.")

    # ============================================================
    # DIAGNOSTICS + EXPLAINABILITY + DATA QUALITY
    # ============================================================

with diagnostics_tab:
    st.markdown('<div class="section-band"><span>🧪 Diagnostics, Explainability & Data Quality</span><span style="font-size:.75rem;color:#6d8094">Observed vs predicted · feature importance · completeness</span></div>',unsafe_allow_html=True)
    d1,d2,d3=st.columns([1.1,1.1,.9],gap="small")
    with d1:
        rp=DATA["reg_predictions"]
        if not rp.empty:
            actual=next((c for c in ["target_pm25_next_day","actual","observed","y_true"] if c in rp.columns),None)
            pred=next((c for c in ["predicted_pm25","prediction","predicted","y_pred"] if c in rp.columns),None)
            if actual and pred:
                fig=px.scatter(rp,x=pred,y=actual,title="Model 2: Observed vs Predicted",labels={pred:"Predicted PM₂.₅ (µg/m³)",actual:"Observed PM₂.₅ (µg/m³)"})
                fig.update_traces(marker=dict(size=6,opacity=.55,color="#1c85e8")); vals=pd.concat([rp[actual],rp[pred]]).dropna()
                if len(vals):
                    mn,mx=float(vals.min()),float(vals.max()); fig.add_shape(type="line",x0=mn,y0=mn,x1=mx,y1=mx,line=dict(color="#5f6d7a",dash="dash"))
                fig.update_layout(height=310,margin=dict(l=24,r=12,t=44,b=28),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
                st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
            else: st.info("Regression prediction columns were not recognised.")
        else: st.info("Regression predictions file not found.")
    with d2:
        imp=DATA["permutation"] if not DATA["permutation"].empty else DATA["shap"]
        if not imp.empty:
            fc=next((c for c in ["Feature","feature","variable"] if c in imp.columns),None)
            vc=next((c for c in ["importance_mean","Importance","importance","Permutation_Importance","mean_importance","SHAP_Importance","mean_abs_shap"] if c in imp.columns),None)
            if fc and vc:
                top=imp[[fc,vc]].dropna().sort_values(vc,ascending=False).head(10).sort_values(vc)
                fig=px.bar(top,x=vc,y=fc,orientation="h",title="Model 3 – " + ("Permutation Importance" if not DATA["permutation"].empty else "Mean Absolute SHAP"))
                fig.update_traces(marker_color="#326fc4"); fig.update_layout(height=310,margin=dict(l=24,r=12,t=44,b=28),yaxis_title=None,xaxis_title="Importance",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
                st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
            else: st.info("Explainability columns were not recognised.")
        else: st.info("Permutation / SHAP importance file not found.")
    with d3:
        st.markdown('<div class="section-title">Input Data Completeness</div>',unsafe_allow_html=True)
        rows=[]
        for name,df in {"NT EPA daily":daily_air,"Model-ready":model_ready,"Dashboard outlook":outlook,"FIRMS detection records":fires}.items():
            if df is None or df.empty: completeness=0.0; status="Missing"
            else:
                completeness=1-float(df.isna().mean().mean())
                status="Excellent" if completeness>=.97 else "Good" if completeness>=.90 else "Moderate" if completeness>=.75 else "Limited"
            rows.append({"Source":name,"Non-missing cells":f"{completeness:.1%}","Status":status})
        st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True)
        st.metric("Selected Input Data Completeness",reliability)
        st.caption("Input availability only; not calibrated prediction confidence. Table percentages describe non-missing cells in loaded records, not satellite observation coverage. A full detection table does not establish that every fire was detected.")
        st.metric("Selected Location",location_name)
        st.metric("Nearest Monitoring Station",station_name)

st.markdown('''<div class="footer-note">ⓘ <b>Fire2Air Darwin is a research prototype.</b> This admin dashboard combines NT EPA air-quality/weather data, NASA FIRMS fire detections, FSI features and machine-learning outputs for research decision-support only. Please use official NT EPA sources for operational air-quality advice.</div>''',unsafe_allow_html=True)

