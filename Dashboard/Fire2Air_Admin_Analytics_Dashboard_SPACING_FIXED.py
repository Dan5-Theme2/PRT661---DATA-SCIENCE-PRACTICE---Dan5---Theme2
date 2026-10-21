from pathlib import Path
from datetime import datetime
import json
import base64

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
LOCAL_REPO_ROOT = Path(
    r"C:\Users\dlihi\Documents\PRT661 Data Science\Assessment 2\PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2"
)

FIRMS_DATA_DIR = REPO_ROOT / "Datasets"
LOCAL_FIRMS_DATA_DIR = LOCAL_REPO_ROOT / "Datasets"
ACTIVE_FIRMS_DATA_DIR = (
    FIRMS_DATA_DIR if FIRMS_DATA_DIR.exists()
    else LOCAL_FIRMS_DATA_DIR if LOCAL_FIRMS_DATA_DIR.exists()
    else FIRMS_DATA_DIR
)

LOGO_CANDIDATES = [
    APP_DIR / "Fire2Air_logo.png",
    REPO_ROOT / "Dashboard" / "Fire2Air_logo.png",
    LOCAL_REPO_ROOT / "Dashboard" / "Fire2Air_logo.png",
]
LOGO_PATH = next((p for p in LOGO_CANDIDATES if p.exists()), None)

SEARCH_DIRS = [
    APP_DIR,
    APP_DIR / "outputs_prt661_a3",
    APP_DIR / "outputs_prt661_a3" / "tables",
    APP_DIR / "outputs_prt661_a3" / "processed",
    APP_DIR / "outputs_prt661",
    APP_DIR / "outputs_prt661" / "tables",
    APP_DIR / "outputs_prt661" / "processed",

    REPO_ROOT / "Dashboard",
    REPO_ROOT / "outputs_prt661_a3",
    REPO_ROOT / "outputs_prt661_a3" / "tables",
    REPO_ROOT / "outputs_prt661_a3" / "processed",
    REPO_ROOT / "Source code",
    REPO_ROOT / "Source code" / "outputs_prt661_a3",
    REPO_ROOT / "Source code" / "outputs_prt661_a3" / "tables",
    REPO_ROOT / "Source code" / "outputs_prt661_a3" / "processed",
    REPO_ROOT / "Source code" / "outputs_prt661" / "processed",
    REPO_ROOT / "Source code" / "outputs_prt661" / "tables",
    REPO_ROOT / "Source code" / "outputs_prt661" / "metadata",

    LOCAL_REPO_ROOT / "Dashboard",
    LOCAL_REPO_ROOT / "outputs_prt661_a3",
    LOCAL_REPO_ROOT / "outputs_prt661_a3" / "tables",
    LOCAL_REPO_ROOT / "outputs_prt661_a3" / "processed",
    LOCAL_REPO_ROOT / "Source code",
    LOCAL_REPO_ROOT / "Source code" / "outputs_prt661_a3",
    LOCAL_REPO_ROOT / "Source code" / "outputs_prt661_a3" / "tables",
    LOCAL_REPO_ROOT / "Source code" / "outputs_prt661_a3" / "processed",
    LOCAL_REPO_ROOT / "Source code" / "outputs_prt661" / "processed",
    LOCAL_REPO_ROOT / "Source code" / "outputs_prt661" / "tables",
    LOCAL_REPO_ROOT / "Source code" / "outputs_prt661" / "metadata",
]

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
    for folder in SEARCH_DIRS:
        p = folder / filename
        if p.exists():
            return p
    for root in [REPO_ROOT, LOCAL_REPO_ROOT]:
        if root.exists():
            matches = list(root.rglob(filename))
            if matches:
                return matches[0]
    return None


def read_csv_safe(filename, parse_dates=None):
    path = find_file(filename)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        df = pd.read_csv(path)
        for c in parse_dates or []:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df


def read_json_safe(filename):
    path = find_file(filename)
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def image_data_uri(path):
    """Return a local image as a data URI so it can sit inside the header card."""
    if path is None or not Path(path).exists():
        return ""
    suffix = Path(path).suffix.lower().replace(".", "")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


@st.cache_data(show_spinner=False)
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
        "permutation": read_csv_safe("a3_model3_count_permutation_importance.csv"),
        "shap": read_csv_safe("a3_model3_shap_importance.csv"),
    }


DATA = load_data()
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


@st.cache_data(show_spinner=False)
def load_fires_for_date(date_text):
    selected_date = pd.Timestamp(date_text).normalize()
    year_files = [
        p for p in ACTIVE_FIRMS_DATA_DIR.glob("fire_archive_SV-C2_*.csv")
        if str(selected_date.year) in p.name
    ]
    parts = []
    for file_path in year_files:
        try:
            for chunk in pd.read_csv(file_path, chunksize=250_000):
                if not {"latitude", "longitude", "acq_date"}.issubset(chunk.columns):
                    continue
                dates = pd.to_datetime(chunk["acq_date"], errors="coerce").dt.normalize()
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
                    d = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
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
    fire["frp"] = pd.to_numeric(fire.get("frp"), errors="coerce")
    if "type" in fire.columns:
        fire_type = pd.to_numeric(fire["type"], errors="coerce")
        fire = fire.loc[fire_type.isna() | fire_type.eq(0)].copy()
    fire = fire.dropna(subset=["latitude", "longitude"])
    fire = fire[
        fire["latitude"].between(-17.5, -7.5)
        & fire["longitude"].between(125.5, 136.5)
    ].copy()
    return fire


# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
:root{
  --ink:#12345f;
  --blue:#0f68c7;
  --line:#c4c4c4;
  --muted:#738499;
  --green:#25bd78;
  --yellow:#ffd044;
  --orange:#ff8430;
  --red:#e8413a;

  --page-x:1.00rem;
  --page-y:.55rem;
  --row-gap:.72rem;
  --section-gap:.72rem;
  --column-gap:8px;
  --card-pad-x:13px;
  --card-pad-y:10px;
}

/* Requested admin background */
.stApp{
  background:#D9D9D9!important;
}

/* Equal outside padding */
.block-container{
  max-width:1800px!important;
  padding:var(--page-y) var(--page-x) .85rem var(--page-x)!important;
}

#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent!important}
[data-testid="stSidebar"]{display:none!important}

/* Consistent horizontal and vertical spacing */
[data-testid="stHorizontalBlock"]{
  gap:var(--column-gap)!important;
  align-items:stretch!important;
  margin:0 0 var(--row-gap) 0!important;
}
[data-testid="column"]{
  padding:0!important;
}
[data-testid="stVerticalBlock"]{
  gap:var(--row-gap)!important;
}

/* Header */
.header-card{
  width:100%;
  height:88px;
  box-sizing:border-box;
  background:#fff;
  border:1px solid var(--line);
  border-radius:15px;
  padding:9px 14px;
  display:flex;
  align-items:center;
  gap:14px;
  box-shadow:0 3px 12px rgba(25,76,125,.07);
  overflow:hidden;
}
.header-logo{
  width:74px;
  height:74px;
  object-fit:contain;
  flex:0 0 74px;
}
.header-logo-fallback{
  width:74px;
  height:74px;
  border-radius:50%;
  display:flex;
  align-items:center;
  justify-content:center;
  background:#eef7ff;
  font-size:2rem;
  flex:0 0 74px;
}
.admin-title{
  font-size:1.42rem;
  font-weight:950;
  color:#103b75;
  line-height:1.05;
}
.admin-dashboard-title{
  font-weight:900;
  color:#174a83;
  font-size:.98rem;
  margin-top:3px;
}
.admin-sub{
  font-size:.74rem;
  color:#607890;
  font-weight:700;
  margin-top:4px;
}

/* Cards */
.metric-card{
  box-sizing:border-box;
  height:104px;
  background:#fff;
  border:1px solid var(--line);
  border-radius:14px;
  padding:var(--card-pad-y) var(--card-pad-x);
  box-shadow:0 3px 12px rgba(25,76,125,.06);
}
.metric-label{font-size:.78rem;color:#2a5380;font-weight:850}
.metric-value{font-size:1.48rem;color:#0e3470;font-weight:950;margin-top:5px}
.metric-sub{font-size:.69rem;color:#77889a;margin-top:4px}

.kpi-blue{background:linear-gradient(135deg,#eef7ff,#e3f1ff)}
.kpi-red{background:linear-gradient(135deg,#fff4f2,#ffe9e5)}
.kpi-orange{background:linear-gradient(135deg,#fff8ee,#fff0dd)}
.kpi-purple{background:linear-gradient(135deg,#f8f2ff,#eee4ff)}

/* Compact section headers */
.section-band{
  display:flex;
  align-items:center;
  justify-content:space-between;
  box-sizing:border-box;
  min-height:50px;
  background:linear-gradient(90deg,#eaf6ff,#f9fcff);
  border:1px solid #c5d8e8;
  border-radius:12px;
  padding:8px 12px;
  margin:var(--section-gap) 0;
  color:#103f7d;
  font-weight:950;
}
.section-title{
  font-size:.98rem;
  font-weight:950;
  color:#103f7d;
  margin:.12rem 0 .28rem;
}

/* Select widgets */
div[data-baseweb="select"]>div{
  min-height:45px!important;
  background:#fff!important;
  border:1px solid var(--line)!important;
  color:#17395f!important;
  border-radius:10px!important;
}
label[data-testid="stWidgetLabel"] p{
  color:#526b83!important;
  font-size:.72rem!important;
  font-weight:850!important;
}

/* Equal Streamlit metric cards */
[data-testid="stMetric"]{
  box-sizing:border-box;
  background:#fff;
  border:1px solid var(--line);
  border-radius:12px;
  padding:10px 12px;
  min-height:88px;
}
[data-testid="stMetricLabel"]{font-weight:850;color:#3d5e7e}
[data-testid="stMetricValue"]{color:#12345f}

/* Chart cards */
.stPlotlyChart{
  box-sizing:border-box;
  background:#fff;
  border:1px solid var(--line);
  border-radius:13px;
  padding:5px;
  margin:0!important;
}

/* Prevent info/warning blocks becoming short odd-sized widgets */
[data-testid="stAlert"]{
  min-height:88px;
  box-sizing:border-box;
  border-radius:12px;
  display:flex;
  align-items:center;
}

/* Tables */
[data-testid="stDataFrame"]{
  background:#fff;
  border-radius:12px;
  overflow:hidden;
}

/* Tight captions */
[data-testid="stCaptionContainer"]{
  margin-top:-2px!important;
  margin-bottom:1px!important;
}

.footer-note{
  margin-top:var(--section-gap);
  background:#eef4f8;
  border:1px solid #c5c5c5;
  border-radius:12px;
  padding:9px 13px;
  color:#506980;
  text-align:center;
  font-size:.75rem;
}

/* Extra breathing room between stacked dashboard rows */
div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"]{
  margin-bottom:.20rem!important;
}

/* Keep section bands separated from rows above and below */
.section-band{
  margin:.72rem 0 .72rem 0!important;
}

/* Add balanced spacing under chart / metric rows */
.stPlotlyChart,
[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stAlert"]{
  margin-bottom:.28rem!important;
}

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

for df, cols in [(outlook,["date","target_date"]),(daily_air,["date"]),(model_ready,["date","target_date"])]:
    for col in cols:
        if not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

logo_uri = image_data_uri(LOGO_PATH)

nav_header, nav_location, nav_date = st.columns([2.65, 2.10, 1.45], gap="small")

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
            <div class="admin-sub">Advanced analytics for research, modelling and decision support</div>
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
    selected_date = st.selectbox("📅 Analysis / outlook date", available_dates, index=len(available_dates)-1)
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
reliability = str(r.get("input_reliability","Unknown"))
influence = str(r.get("fire_smoke_influence","Unknown"))

# ============================================================
# MODEL 2 FALLBACK / VALIDATION DATA
# ============================================================
# Favour's Model 2 writes:
#   outputs_prt661_a3/tables/a3_model2_final_2024_metrics.csv
#   outputs_prt661_a3/tables/a3_model2_2024_predictions.csv
# If these CSVs have not been copied into the dashboard repository yet,
# reconstruct the same 2024 prediction table from the integrated dashboard
# outlook + model-ready target so the admin view still has evaluation data.

if DATA["reg_predictions"].empty:
    if (
        not outlook.empty
        and not model_ready.empty
        and {"date", "target_date", "station", "expected_daily_mean_pm25"}.issubset(outlook.columns)
        and {"date", "target_date", "station", "target_pm25_next_day"}.issubset(model_ready.columns)
    ):
        reg_fallback = (
            outlook[
                ["date", "target_date", "station", "expected_daily_mean_pm25"]
            ]
            .merge(
                model_ready[
                    ["date", "target_date", "station", "target_pm25_next_day"]
                ],
                on=["date", "target_date", "station"],
                how="inner",
                validate="one_to_one",
            )
            .rename(columns={"expected_daily_mean_pm25": "prediction"})
        )
        reg_fallback["residual_observed_minus_predicted"] = (
            reg_fallback["target_pm25_next_day"] - reg_fallback["prediction"]
        )
        reg_fallback["underprediction"] = (
            reg_fallback["prediction"] < reg_fallback["target_pm25_next_day"]
        ).astype(int)
        reg_fallback["elevated_actual"] = (
            reg_fallback["target_pm25_next_day"] >= PM25_THRESHOLD
        ).astype(int)
        DATA["reg_predictions"] = reg_fallback

if DATA["reg_metrics"].empty and not DATA["reg_predictions"].empty:
    rp_for_metrics = DATA["reg_predictions"].dropna(
        subset=["target_pm25_next_day", "prediction"]
    ).copy()

    if "target_date" in rp_for_metrics.columns:
        rp_for_metrics["target_date"] = pd.to_datetime(
            rp_for_metrics["target_date"], errors="coerce"
        )
        rp_for_metrics = rp_for_metrics[
            rp_for_metrics["target_date"].dt.year.eq(2024)
        ]

    if not rp_for_metrics.empty:
        y = rp_for_metrics["target_pm25_next_day"].to_numpy(dtype=float)
        p = rp_for_metrics["prediction"].to_numpy(dtype=float)

        mae = float(np.mean(np.abs(y - p)))
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        denom = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1 - np.sum((y - p) ** 2) / denom) if denom > 0 else np.nan

        elevated_mask = y >= PM25_THRESHOLD

        DATA["reg_metrics"] = pd.DataFrame([{
            "Model": "Model 2 selected regressor",
            "Scope": "All 2024 model rows",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Bias_PredMinusObs": float(np.mean(p - y)),
            "Underprediction_Rate": float(np.mean(p < y)),
            "Elevated_Rows": int(elevated_mask.sum()),
            "Elevated_MAE": (
                float(np.mean(np.abs(y[elevated_mask] - p[elevated_mask])))
                if elevated_mask.any() else np.nan
            ),
            "Elevated_RMSE": (
                float(np.sqrt(np.mean((y[elevated_mask] - p[elevated_mask]) ** 2)))
                if elevated_mask.any() else np.nan
            ),
            "Elevated_Bias_PredMinusObs": (
                float(np.mean(p[elevated_mask] - y[elevated_mask]))
                if elevated_mask.any() else np.nan
            ),
            "Elevated_Underprediction_Rate": (
                float(np.mean(p[elevated_mask] < y[elevated_mask]))
                if elevated_mask.any() else np.nan
            ),
        }])

# ============================================================
# SYSTEM OVERVIEW
# ============================================================
st.markdown(f'<div class="section-band"><span>🧭 System Overview</span><span style="font-size:.75rem;color:#6d8094">Nearest station: {station_name} · {station_distance:.1f} km</span></div>', unsafe_allow_html=True)
c1,c2,c3,c4,c5 = st.columns(5, gap="small")
start_year = int(model_ready["date"].dt.year.min()) if not model_ready.empty else 2018
end_year = int(model_ready["date"].dt.year.max()) if not model_ready.empty else 2024
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
st.markdown(f'<div class="section-band"><span>☁️ Tomorrow Smoke Outlook</span><span style="font-size:.75rem;color:#6d8094">{selected_ts.strftime("%a %d %b %Y")}</span></div>', unsafe_allow_html=True)
g1,g2,g3,g4 = st.columns([1,1,1,1.05], gap="small")
with g1:
    st.plotly_chart(gauge(prob*100 if pd.notna(prob) else np.nan,"Model 1 – Classification","%",100,50), width="stretch", config={"displayModeBar":False})
    st.caption(f"Elevated PM2.5 probability · **{risk_label(prob)} risk**")
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
        st.metric("🛡️ Input Reliability", reliability)

# ============================================================
# AIR QUALITY MONITORING
# ============================================================
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
st.markdown(f'<div class="section-band"><span>🔥 Fire Intelligence & FSI ({location_name})</span><span style="font-size:.75rem;color:#6d8094">Active fires · intensity · distance · fire-smoke relationships</span></div>', unsafe_allow_html=True)
map_col,analytics_col=st.columns([1.3,1.25],gap="small")
fires=load_fires_for_date(str(predictor_date.date()))
with map_col:
    st.markdown('<div class="section-title">🗺️ Active Fire Map</div>',unsafe_allow_html=True)
    st.caption(f"Predictor date: {predictor_date.date()} · FIRMS detections loaded: {len(fires):,}")
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
    total_frp=float(fires["frp"].sum()) if not fires.empty and "frp" in fires.columns else np.nan
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
            "Model 2 metrics are unavailable. Add "
            "outputs_prt661_a3/tables/a3_model2_final_2024_metrics.csv."
        )
with mp3:
    st.markdown('<div class="section-title">Model 3 – Count</div>',unsafe_allow_html=True)
    df=DATA["count_metrics"]
    if not df.empty:
        rr=df.iloc[0]; a,b,c=st.columns(3); a.metric("RMSE",safe_num(first_available(rr,["RMSE"]),".2f")); b.metric("MAE",safe_num(first_available(rr,["MAE"]),".2f")); c.metric("R²",safe_num(first_available(rr,["R2","R_squared"]),".3f")); a,b=st.columns(2); a.metric("Within ±3 h",safe_num(first_available(rr,["Within_3h","Within_3_hours"]),".2f")); b.metric("Bias",safe_num(first_available(rr,["Bias"]),".2f"))
    else: st.info("Model 3 metrics file not found.")

# ============================================================
# DIAGNOSTICS + EXPLAINABILITY + DATA QUALITY
# ============================================================
st.markdown('<div class="section-band"><span>🧪 Diagnostics, Explainability & Data Quality</span><span style="font-size:.75rem;color:#6d8094">Observed vs predicted · feature importance · completeness</span></div>',unsafe_allow_html=True)
d1,d2,d3=st.columns([1.1,1.1,.9],gap="small")
with d1:
    rp=DATA["reg_predictions"]
    if not rp.empty:
        actual=next((c for c in ["target_pm25_next_day","actual","observed","y_true"] if c in rp.columns),None)
        pred=next((c for c in ["predicted_pm25","prediction","predicted","y_pred"] if c in rp.columns),None)
        if actual and pred:
            fig=px.scatter(rp,x=pred,y=actual,title="Model 2: Observed vs Predicted",labels={pred:"Predicted PM2.5",actual:"Observed PM2.5"})
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
            fig=px.bar(top,x=vc,y=fc,orientation="h",title="Explainability – Top Feature Importance")
            fig.update_traces(marker_color="#326fc4"); fig.update_layout(height=310,margin=dict(l=24,r=12,t=44,b=28),yaxis_title=None,xaxis_title="Importance",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
            st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        else: st.info("Explainability columns were not recognised.")
    else: st.info("Permutation / SHAP importance file not found.")
with d3:
    st.markdown('<div class="section-title">Data Quality & Reliability</div>',unsafe_allow_html=True)
    rows=[]
    for name,df in {"NT EPA daily":daily_air,"Model-ready":model_ready,"Dashboard outlook":outlook,"FIRMS selected date":fires}.items():
        if df is None or df.empty: completeness=0.0; status="Missing"
        else:
            completeness=1-float(df.isna().mean().mean())
            status="Excellent" if completeness>=.97 else "Good" if completeness>=.90 else "Moderate" if completeness>=.75 else "Limited"
        rows.append({"Source":name,"Completeness":f"{completeness:.0%}","Status":status})
    st.dataframe(pd.DataFrame(rows),width="stretch",hide_index=True)
    st.metric("Selected Outlook Reliability",reliability)
    st.metric("Selected Location",location_name)
    st.metric("Nearest Monitoring Station",station_name)

st.markdown('''<div class="footer-note">ⓘ <b>Fire2Air Darwin is a research prototype.</b> This admin dashboard combines NT EPA air-quality/weather data, NASA FIRMS fire detections, FSI features and machine-learning outputs for research decision-support only. Please use official NT EPA sources for operational air-quality advice.</div>''',unsafe_allow_html=True)
