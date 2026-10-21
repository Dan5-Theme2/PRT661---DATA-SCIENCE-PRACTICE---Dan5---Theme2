
from pathlib import Path
import os
from html import escape
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False

st.set_page_config(page_title="Fire2Air Darwin", page_icon="🔥", layout="wide", initial_sidebar_state="collapsed")

# PROJECT PATH + FILES

PROJECT_FOLDER = Path(os.environ.get("FIRE2AIR_DATA_DIR", Path(__file__).resolve().parent)).expanduser().resolve()
OUTPUT_ROOT = PROJECT_FOLDER / "outputs_prt661_a3"
TABLE_DIR = OUTPUT_ROOT / "tables"

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

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

# Raw NASA FIRMS yearly archives in the GitHub repository.
FIRMS_DATA_DIR = REPO_ROOT / "Datasets"

# Local Windows fallback.
LOCAL_FIRMS_DATA_DIR = Path(
    r"C:\Users\dlihi\Documents\PRT661 Data Science\Assessment 2\PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2\Datasets"
)

if FIRMS_DATA_DIR.exists():
    ACTIVE_FIRMS_DATA_DIR = FIRMS_DATA_DIR
elif LOCAL_FIRMS_DATA_DIR.exists():
    ACTIVE_FIRMS_DATA_DIR = LOCAL_FIRMS_DATA_DIR
else:
    ACTIVE_FIRMS_DATA_DIR = FIRMS_DATA_DIR

# Logo: Dashboard/Fire2Air_logo.png
LOCAL_LOGO_FALLBACK = Path(
    r"C:\Users\dlihi\Documents\PRT661 Data Science\Assessment 2\PRT661---DATA-SCIENCE-PRACTICE---Dan5---Theme2\Dashboard\Fire2Air_logo.png"
)

LOGO_CANDIDATES = [
    APP_DIR / "Fire2Air_logo.png",
    LOCAL_LOGO_FALLBACK,
]

LOGO_PATH = next(
    (p for p in LOGO_CANDIDATES if p.is_file()),
    None,
)

LOAD_ERRORS = []

def read_csv_safe(path, parse_dates=None):
    try:
        df = pd.read_csv(path)
        for col in parse_dates or []:
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception as exc:
        LOAD_ERRORS.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return pd.DataFrame()

def read_json_safe(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOAD_ERRORS.append(f"{path.name}: {type(exc).__name__}: {exc}")
        return {}

def load_data():
    # Reload files on every rerun; a failed read never substitutes cached predictions.
    manifest_path = PROJECT_FOLDER / "Fire2Air_feature_manifest_A3.json"
    if not manifest_path.exists():
        manifest_path = PROJECT_FOLDER.parent / "metadata" / manifest_path.name
    return {
        "manifest": read_json_safe(manifest_path),
        "outlook": read_csv_safe(PROJECT_FOLDER / "Fire2Air_dashboard_ready_outlook_A3.csv", ["date", "target_date", "forecast_issue_date"]),
        "daily_air": read_csv_safe(PROJECT_FOLDER / "Fire2Air_ntepa_daily_checkpoint_A3.csv", ["date"]),
        "model_ready": read_csv_safe(PROJECT_FOLDER / "Fire2Air_model_ready_checkpoint_A3.csv", ["date", "target_date"]),
    }

DATA = load_data()
if DATA["manifest"]:
    PM25_THRESHOLD = float(DATA["manifest"].get("threshold_pm25_ug_m3", 25.0))

# HELPERS
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def nearest_station(lat, lon):
    vals = []
    for s,(slat,slon) in STATION_COORDINATES.items():
        vals.append((s,float(haversine_km(lat,lon,slat,slon))))
    return sorted(vals,key=lambda x:x[1])[0]

def safe_num(v, fmt=".1f"):
    try:
        return format(float(v), fmt) if pd.notna(v) else "—"
    except Exception:
        return "—"

def pm_status(pm):
    if pd.isna(pm): return "Unknown","status-neutral"
    if pm < 12: return "Low","status-good"
    if pm < PM25_THRESHOLD: return "Moderate","status-moderate"
    if pm < 50: return "High","status-high"
    return "Very High","status-danger"

def probability_status(p):
    if pd.isna(p): return "Unknown","status-neutral"
    if p < .30: return "Low","status-good"
    if p < .50: return "Moderate","status-moderate"
    if p < .70: return "High","status-high"
    return "Very High","status-danger"

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

                dates = pd.to_datetime(
                    chunk["acq_date"],
                    errors="coerce"
                ).dt.normalize()

                mask = dates.eq(selected_date)
                if not mask.any():
                    continue

                keep = [
                    c for c in [
                        "latitude", "longitude", "acq_date", "acq_time",
                        "frp", "type", "confidence", "brightness", "bright_ti4"
                    ]
                    if c in chunk.columns
                ]
                parts.append(chunk.loc[mask, keep].copy())

        except Exception as exc:
            print(f"Could not read FIRMS file {file_path.name}: {exc}")

    # Fallback to compact dashboard fire-point file.
    if not parts:
        compact_file = APP_DIR / "Fire2Air_dashboard_fire_points_2018_2024.csv"
        if compact_file.exists():
            try:
                compact = pd.read_csv(compact_file)
                date_col = next(
                    (c for c in ["date_darwin", "acq_date", "date"] if c in compact.columns),
                    None,
                )
                if date_col:
                    compact_dates = pd.to_datetime(
                        compact[date_col],
                        errors="coerce"
                    ).dt.normalize()
                    compact = compact.loc[compact_dates.eq(selected_date)].copy()
                    if not compact.empty:
                        parts.append(compact)
            except Exception as exc:
                print(f"Could not read compact fire-point file: {exc}")

    if not parts:
        return pd.DataFrame()

    fire = pd.concat(parts, ignore_index=True)

    fire["latitude"] = pd.to_numeric(fire["latitude"], errors="coerce")
    fire["longitude"] = pd.to_numeric(fire["longitude"], errors="coerce")
    fire["frp"] = pd.to_numeric(fire.get("frp"), errors="coerce")

    # Important: type may be loaded as string "0"; convert before filtering.
    if "type" in fire.columns:
        fire_type = pd.to_numeric(fire["type"], errors="coerce")
        fire = fire.loc[fire_type.isna() | fire_type.eq(0)].copy()

    fire = fire.dropna(subset=["latitude", "longitude"])

    # Broad Top End region.
    fire = fire[
        fire["latitude"].between(-17.5, -7.5)
        & fire["longitude"].between(125.5, 136.5)
    ].copy()

    return fire

# CSS — MATCHES THE APPROVED MOCKUP

st.markdown("""
<style>
:root{--ink:#12345f;--stroke:#d9dde2;--muted:#748292;--green:#35c975;--yellow:#ffc53d;--orange:#ff7a32;--red:#eb4238;--widget-pad:16px;}
.stApp{background:#FFFFFF}
.block-container{padding-top:.55rem!important;padding-bottom:1.2rem!important;max-width:1650px!important;padding-left:1.35rem!important;padding-right:1.35rem!important}
#MainMenu,footer{visibility:hidden} header[data-testid="stHeader"]{background:transparent} [data-testid="stSidebar"]{display:none!important}
.clock-card,.panel,.outlook,.summary,.warning,.weather,.fsi,.about-card{border:1px solid var(--stroke)!important;box-sizing:border-box}
.nav-title-card{height:82px;border:none!important;background:transparent!important;box-shadow:none!important;border-radius:0!important;padding:var(--widget-pad) 0!important;display:flex;flex-direction:column;justify-content:center;box-sizing:border-box}.nav-title{font-size:1.45rem;font-weight:900;color:var(--ink)}.nav-sub{font-size:.78rem;color:#60758a;font-weight:700;margin-top:5px}
.clock-card{height:82px;background:#fff;border-radius:16px;padding:11px 13px;display:flex;flex-direction:column;justify-content:center}.clock-label{font-size:.7rem;color:#758495;font-weight:800;text-transform:uppercase}.clock-value{font-size:.92rem;color:var(--ink);font-weight:900;line-height:1.35;margin-top:4px}
div[data-baseweb="select"]>div{background:#fff!important;color:#17395f!important;border:1px solid var(--stroke)!important;min-height:46px!important} div[data-baseweb="select"] input{color:#17395f!important;-webkit-text-fill-color:#17395f!important;caret-color:#17395f!important} div[data-baseweb="select"] span{color:#17395f!important} div[data-baseweb="select"] svg{fill:#49647e!important} div[role="listbox"]{background:#fff!important} div[role="option"]{color:#17395f!important;background:#fff!important} label[data-testid="stWidgetLabel"] p{color:#566d84!important;font-size:.74rem!important;font-weight:800!important}
.panel{background:#fff;border-radius:15px;padding:var(--widget-pad);min-height:52px}.card-title{font-weight:900;color:#173e75;font-size:1rem}.muted{color:var(--muted);font-size:.79rem}.big{font-size:2rem;font-weight:900;color:#102f68;line-height:1}.unit{font-size:.88rem;font-weight:750;color:#173e75}.section{font-size:1.06rem;font-weight:900;color:#143f78;margin:8px 0 8px}
.outlook{min-height:174px!important;border-radius:16px;padding:var(--widget-pad);display:flex;flex-direction:column;justify-content:space-between;overflow:hidden}.greenbg{background:linear-gradient(135deg,#effcf6,#e2f6ec)}.yellowbg{background:linear-gradient(135deg,#fffaf2,#fff0d1)}.redbg{background:linear-gradient(135deg,#fff6f3,#ffe5e1)}
.pill{display:inline-flex;align-items:center;justify-content:center;height:42px;min-width:126px;padding:0 16px;border-radius:999px;font-weight:900}.status-good{background:#3ad07c;color:#083f2e}.status-moderate{background:#ffc533;color:#4c3a00}.status-high{background:#ff6f36;color:#fff}.status-danger{background:#e63e38;color:#fff}.status-neutral{background:#dae2e8;color:#334d66}
.summary{min-height:142px!important;border-radius:16px;padding:var(--widget-pad);display:flex;flex-direction:column;justify-content:space-between;overflow:hidden}.sred{background:#fff1ef}.sorange{background:#fff7ec}.syellow{background:#fff9e9}.sgreen{background:#eefbf5}.slabel{font-size:.86rem;font-weight:850;color:#193f75}.svalue{font-size:1.52rem;font-weight:900;color:#102f68;line-height:1.05}
.warning{height:142px!important;min-height:142px!important;max-height:142px!important;background:linear-gradient(95deg,#fff2ef,#fff8f6);border-radius:16px;padding:var(--widget-pad);overflow:hidden}.warning h3{color:#d8271e;margin:0;font-size:1.17rem}.warning p{color:#173f75;margin:.45rem 0 0;font-size:.86rem;line-height:1.35}
.weather{height:100px!important;min-height:100px!important;max-height:100px!important;background:#fff;border-radius:14px;padding:var(--widget-pad);overflow:hidden}.weather .label{font-size:.78rem;font-weight:850;color:#204a82}.weather .value{font-size:1.28rem;font-weight:900;color:#112f68;margin-top:4px}.fsi{height:170px!important;min-height:170px!important;max-height:170px!important;background:linear-gradient(90deg,#fff9e9,#fff3ed);border-radius:16px;padding:var(--widget-pad);overflow:hidden}.fsi .level{font-size:1.3rem;font-weight:900;color:#f06d22}
.about-card{background:#fff;border-radius:16px;padding:var(--widget-pad);margin-top:14px}.about-card h3{color:var(--ink);margin:0 0 9px}.about-card p,.about-card li{color:#405a72;font-size:.88rem;line-height:1.48}.footer-note{background:#edf5fa;border:1px solid var(--stroke);border-radius:14px;padding:9px 12px;text-align:center;color:#49667f;font-size:.78rem;margin-top:9px}
/* Consistent vertical breathing room inside dashboard widgets */
.outlook,.summary,.warning,.weather,.fsi,.panel,.context,.about-card{
  box-sizing:border-box;
}
[data-testid="stHorizontalBlock"]{align-items:stretch;margin-bottom:4px}
div[data-testid="stVerticalBlock"] > div:has(.outlook),
div[data-testid="stVerticalBlock"] > div:has(.summary),
div[data-testid="stVerticalBlock"] > div:has(.panel),
div[data-testid="stVerticalBlock"] > div:has(.weather),
div[data-testid="stVerticalBlock"] > div:has(.warning),
div[data-testid="stVerticalBlock"] > div:has(.fsi){
  margin-bottom:4px;
}
</style>
""", unsafe_allow_html=True)

# TOP NAVIGATION + DATA SELECTION

outlook = DATA["outlook"]
daily_air = DATA["daily_air"]
model_ready = DATA["model_ready"]
st.warning("🗂️ HISTORICAL FORECASTS · Retrospective research outputs, not a live forecast or current conditions.")
if LOAD_ERRORS:
    st.warning("Some exported files could not be loaded. Missing observations remain unavailable; forecasts are never substituted for observations.")
    with st.expander("File loading diagnostics"):
        st.code("\n".join(LOAD_ERRORS))
required = {"station", "date", "target_date", "expected_daily_mean_pm25", "elevated_pm25_probability", "expected_elevated_hours"}
if not outlook.empty and not required.issubset(outlook.columns):
    st.error("Invalid outlook export: missing columns " + ", ".join(sorted(required - set(outlook.columns))))
    st.stop()
if not outlook.empty:
    invalid_dates = outlook["date"].isna() | outlook["target_date"].isna() | outlook["target_date"].dt.normalize().ne(outlook["date"].dt.normalize() + pd.Timedelta(days=1))
    duplicates = outlook.duplicated(["station", "target_date"], keep=False)
    if invalid_dates.any() or duplicates.any():
        st.error("Outlook export rejected: invalid next-day dates or duplicate station/target-date records. Rebuild the export.")
        st.stop()
for key, cols in [("daily_air", {"station", "date", "pm25_mean"}), ("model_ready", {"station", "date"})]:
    if not DATA[key].empty and not cols.issubset(DATA[key].columns):
        st.warning(f"{key} has an invalid schema; its context is unavailable.")
        DATA[key] = pd.DataFrame()
daily_air, model_ready = DATA["daily_air"], DATA["model_ready"]
if outlook.empty:
    st.error("Missing Fire2Air_dashboard_ready_outlook_A3.csv. Run scripts 01–05 first.")
    st.stop()

nav_logo, nav_title, nav_location, nav_date = st.columns([0.62,1.45,2.35,1.35], gap="small")
with nav_logo:
    if LOGO_PATH is not None: st.image(str(LOGO_PATH), width="stretch")
    else: st.warning("Fire2Air_logo.png not found")
with nav_title:
    st.markdown("<div class=\"nav-title-card\"><div class=\"nav-title\">Fire2Air Darwin</div><div class=\"nav-sub\">Historical smoke forecasts for Greater Darwin</div></div>", unsafe_allow_html=True)
with nav_location:
    location_name = st.selectbox("📍 Search / select location", sorted(DARWIN_LOCATIONS), index=sorted(DARWIN_LOCATIONS).index("Palmerston"), help="Click the field and type a suburb name to search.")
lat,lon = DARWIN_LOCATIONS[location_name]
station_name,station_distance = nearest_station(lat,lon)
o = outlook[outlook["station"].astype(str).eq(station_name)].sort_values("target_date")
if o.empty:
    st.error("No historical forecasts are available for this station.")
    st.stop()
dates = o["target_date"].dropna().dt.date.unique().tolist()
with nav_date:
    selected_date = st.selectbox("📅 Forecast target date", dates, index=len(dates)-1)
selected_ts = pd.Timestamp(selected_date)
row = o[o["target_date"].dt.date.eq(selected_date)].tail(1)
if row.empty: st.stop()
r = row.iloc[0]
nav = "🏠 Smoke Outlook"

# LOCATION CONTEXT

a,b = st.columns([2.2,1], gap="small")
with a:
    st.markdown(f'<div class="panel">📍 <b style="font-size:1.08rem;color:#143f78">{location_name}</b><span style="float:right;color:#748292">nearest station: {station_name} · {station_distance:.1f} km</span></div>', unsafe_allow_html=True)
with b:
    st.markdown('<div class="panel"><b style="color:#143f78">🌏 Greater Darwin</b><span class="muted"> &nbsp; Darwin · Palmerston · Rural areas</span></div>', unsafe_allow_html=True)

# TOP OUTLOOK CARDS — date is the observation day; target_date is the next day.
observation_date = pd.Timestamp(r["date"]).normalize()
issue_value = r.get("forecast_issue_date", pd.NaT)
issue_label = pd.Timestamp(issue_value).isoformat() if pd.notna(issue_value) else "Not recorded in export"
st.info(f"Observation date: {observation_date:%d %b %Y} · Forecast issue date: {issue_label} · Target date: {selected_ts:%d %b %Y}. Daily inputs require the observation day to finish; an actual issue time cannot be inferred from these exports.")
station_air = daily_air[daily_air["station"].astype(str).eq(station_name)].copy() if not daily_air.empty else pd.DataFrame()
prev_date = observation_date
prev_pm = today_pm = np.nan
if not station_air.empty:
    p = station_air[station_air["date"].dt.normalize().eq(observation_date)]
    t = station_air[station_air["date"].dt.normalize().eq(selected_ts)]
    if not p.empty: prev_pm = pd.to_numeric(p.iloc[-1].get("pm25_mean"), errors="coerce")
    if not t.empty: today_pm = pd.to_numeric(t.iloc[-1].get("pm25_mean"), errors="coerce")

pred_pm, prob, hours = [pd.to_numeric(r.get(c), errors="coerce") for c in
    ["expected_daily_mean_pm25", "elevated_pm25_probability", "expected_elevated_hours"]]
if not np.isfinite(pred_pm) or pred_pm < 0: pred_pm = np.nan
if not np.isfinite(prob) or not 0 <= prob <= 1: prob = np.nan
if not np.isfinite(hours) or not 0 <= hours <= 24: hours = np.nan
if any(pd.isna(x) for x in [pred_pm, prob, hours]):
    st.warning("One or more model outputs are missing or invalid and are displayed as unavailable. Re-run the relevant model and export step.")
influence = escape(str(r.get("fire_smoke_influence", "Unknown")))
coverage_cols = ["pm25_coverage_ratio", "humidity_coverage_ratio", "wind_speed_coverage_ratio",
                 "wind_direction_coverage_ratio", "firms_local_day_complete", "firms_48h_complete"]
coverage = pd.to_numeric(pd.Series({c: r.get(c, np.nan) for c in coverage_cols}), errors="coerce")
coverage = coverage.where(coverage.between(0, 1))
score = coverage.mean() if coverage.notna().all() else np.nan
reliability = "Unknown" if pd.isna(score) else ("Limited" if score < .75 else "Moderate" if score < .90 else "High")
completeness_text = f"{score:.0%} · six coverage indicators" if pd.notna(score) else "coverage indicators missing"

pl,pc=pm_status(prev_pm); tl,tc=pm_status(today_pm); nl,nc=probability_status(prob)

c1,c2,c3=st.columns(3)
with c1:
    st.markdown(f"""<div class="outlook greenbg"><div class="card-title">📅 Observed PM2.5 · Input Day</div><div class="muted">{prev_date.strftime("%a %d %b %Y")}</div>
    <div style="display:flex;justify-content:space-between;align-items:end;margin-top:12px"><div><span class="big">{safe_num(prev_pm)}</span><br><span class="unit">µg/m³</span></div><span class="pill {pc}">{pl}</span></div></div>""",unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="outlook yellowbg"><div class="card-title">📅 Observed PM2.5 · Target Day</div><div class="muted">{selected_ts.strftime("%a %d %b %Y")}</div>
    <div style="display:flex;justify-content:space-between;align-items:end;margin-top:12px"><div><span class="big">{safe_num(today_pm)}</span><br><span class="unit">µg/m³</span></div><span class="pill {tc}">{tl}</span></div></div>""",unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="outlook redbg"><div class="card-title">📅 Historical Next-day Forecast</div><div class="muted">{selected_ts.strftime("%a %d %b %Y")}</div>
    <div style="display:flex;justify-content:space-between;align-items:end;margin-top:12px"><div><span class="big">{safe_num(pred_pm)}</span><br><span class="unit">µg/m³</span></div><span class="pill {nc}">{nl}</span></div></div>""",unsafe_allow_html=True)

st.caption(
    f"Project-defined PM₂.₅ concentration categories (µg/m³): Low <12; "
    f"Moderate 12–<{PM25_THRESHOLD:g}; High {PM25_THRESHOLD:g}–<50; Very High ≥50. "
    "These are research display categories, not an official air-quality index. "
    "The historical forecast badge instead describes the model probability of elevated daily PM₂.₅: "
    "Low <30%; Moderate 30–<50%; High 50–<70%; Very High ≥70%."
)

s1,s2,s3,s4=st.columns(4)
with s1: st.markdown(f'<div class="summary sred"><div class="slabel">🫧 Expected PM2.5</div><div class="svalue">{safe_num(pred_pm)} µg/m³</div><div class="muted">forecast target date</div></div>',unsafe_allow_html=True)
with s2: st.markdown(f'<div class="summary sorange"><div class="slabel">🕒 Expected Elevated PM₂.₅ Hours</div><div class="svalue">{safe_num(hours)} hours</div><div class="muted">Hourly PM₂.₅ ≥{PM25_THRESHOLD:g} µg/m³</div></div>',unsafe_allow_html=True)
with s3: st.markdown(f'<div class="summary syellow"><div class="slabel">🌬️ Smoke Influence</div><div class="svalue" style="color:#e77917">{influence}</div><div class="muted">fire-smoke influence</div></div>',unsafe_allow_html=True)
with s4: st.markdown(f'<div class="summary sgreen"><div class="slabel">🛡️ Input Data Completeness</div><div class="svalue" style="color:#168d59">{reliability}</div><div class="muted">{completeness_text}</div></div>',unsafe_allow_html=True)

st.caption("Target-day observations are shown for retrospective comparison only and are not forecast inputs. Missing observations show —. Completeness describes six input coverage indicators, not predictive reliability; FIRMS coverage flags are archive-date proxies.")
st.info(f"Historical model probability of elevated daily PM2.5: {safe_num(prob * 100)}%. This probability is not a confidence interval. Individual forecast prediction intervals are not supplied by the current exports.")
with st.expander("Input completeness and uncertainty details"):
    st.dataframe(coverage.rename("Coverage ratio (0–1)"))
    st.write("High ≥90%; Moderate ≥75%; Limited <75%. Every indicator must be present and valid, otherwise completeness is Unknown. Model preprocessing may impute missing features; completeness does not establish accuracy or calibration.")
    st.write("Updated regression and count notebooks resample blocks of calendar dates with stations kept together. Their metric confidence intervals describe evaluation uncertainty, not ranges for this selected forecast. Check the classification resampling before claiming a common method across all models.")

# SMOKE OUTLOOK PAGE

if nav=="🏠 Smoke Outlook":
    left,right=st.columns([1.65,1])
    with left:
        st.markdown('<div class="section">🗺️ Historical Fire Detections · Input Day</div>',unsafe_allow_html=True)
        predictor_date=observation_date
        fires=load_fires_for_date(str(predictor_date.date()))
        st.caption(f"Fire detections: {observation_date:%d %b %Y}. Detections are not a smoke-plume forecast. Empty results may mean unavailable files or no matching detections.")
        if HAS_FOLIUM:
            fmap=folium.Map(location=[lat,lon],zoom_start=9,tiles="OpenStreetMap",control_scale=True)
            folium.Marker([lat,lon],tooltip=f"Selected: {location_name}",icon=folium.Icon(color="red",icon="home")).add_to(fmap)
            for name,(slat,slon) in STATION_COORDINATES.items():
                folium.CircleMarker([slat,slon],radius=7,color="#0d74c8",fill=True,fill_opacity=.95,tooltip=f"Air quality station: {name}").add_to(fmap)
            st.caption(
                f"🔥 FIRMS detections for {predictor_date.date()}: {len(fires):,}"
            )

            if not fires.empty:
                fires["distance_to_user_km"] = haversine_km(
                    fires["latitude"].to_numpy(),
                    fires["longitude"].to_numpy(),
                    lat,
                    lon
                )

                fires = fires[
                    fires["distance_to_user_km"] <= 500
                ].copy()

                def fire_intensity(frp):
                    if pd.isna(frp):
                        return "Unknown", "#6b7280", 24
                    frp = float(frp)
                    if frp < 10:
                        return "Low", "#f6b73c", 26
                    if frp < 50:
                        return "Moderate", "#f57c00", 32
                    return "High", "#d62828", 40

                map_fires = fires.sort_values(
                    "frp",
                    ascending=False,
                    na_position="last"
                ).head(800)

                for _, fire_row in map_fires.iterrows():
                    fire_lat = float(fire_row["latitude"])
                    fire_lon = float(fire_row["longitude"])
                    frp = fire_row.get("frp", np.nan)
                    dist = fire_row.get("distance_to_user_km", np.nan)

                    level, marker_color, size = fire_intensity(frp)

                    icon_html = f"""
                    <div style="
                        width:{size}px;
                        height:{size}px;
                        border-radius:50%;
                        background:{marker_color};
                        border:3px solid white;
                        box-shadow:0 2px 7px rgba(0,0,0,.45);
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:{max(16, int(size*0.62))}px;
                        line-height:1;
                    ">🔥</div>
                    """

                    fire_icon = folium.DivIcon(
                        html=icon_html,
                        icon_size=(size, size),
                        icon_anchor=(size // 2, size // 2),
                    )

                    folium.Marker(
                        [fire_lat, fire_lon],
                        icon=fire_icon,
                        tooltip=(
                            f"🔥 {level} intensity · "
                            f"{safe_num(frp)} MW · "
                            f"{safe_num(dist)} km away"
                        ),
                        popup=(
                            f"<b>Active fire detection</b><br>"
                            f"Intensity: {level}<br>"
                            f"FRP: {safe_num(frp)} MW<br>"
                            f"Distance: {safe_num(dist)} km"
                        ),
                    ).add_to(fmap)

                legend_html = """
                <div style="
                    position: fixed;
                    bottom: 30px;
                    left: 30px;
                    z-index: 9999;
                    background: rgba(255,255,255,.96);
                    border: 1px solid #d8dee6;
                    border-radius: 10px;
                    padding: 10px 12px;
                    font-size: 12px;
                    color: #17395f;
                    box-shadow: 0 2px 8px rgba(0,0,0,.15);
                ">
                    <b>🔥 Fire intensity (FRP)</b><br>
                    <span style="color:#f6b73c">●</span> Low: &lt; 10 MW<br>
                    <span style="color:#f57c00">●</span> Moderate: 10–49.9 MW<br>
                    <span style="color:#d62828">●</span> High: ≥ 50 MW
                </div>
                """
                fmap.get_root().html.add_child(
                    folium.Element(legend_html)
                )

            st_folium(
                fmap,
                height=520,
                width="stretch",
                returned_objects=[],
                key=f"map_{location_name}_{selected_date}",
            )
        else:
            st.info("Install folium and streamlit-folium for the interactive map.")

        st.markdown(f'<div class="section">📊 Recent PM2.5 at {station_name}</div>',unsafe_allow_html=True)
        recent=station_air[station_air["date"].between(observation_date-pd.Timedelta(days=30),observation_date)] if not station_air.empty else pd.DataFrame()
        if not recent.empty:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=recent["date"],y=recent["pm25_mean"],mode="lines+markers",line=dict(color="#168ee2",width=3),
                                     marker=dict(size=5),fill="tozeroy",fillcolor="rgba(22,142,226,.08)"))
            fig.add_hline(y=PM25_THRESHOLD,line_dash="dash",line_color="#ef4136",annotation_text=f"{PM25_THRESHOLD:g} µg/m³")
            fig.update_layout(height=285,margin=dict(l=25,r=20,t=15,b=30),showlegend=False,yaxis_title="PM2.5 (µg/m³)",
                              paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="white")
            st.plotly_chart(fig,use_container_width=True)

    with right:
        headline = "Historical forecast interpretation"
        decision = "Retrospective model output for the selected target date. This panel does not describe current outdoor conditions."
        st.markdown('<div class="section">🚶 Historical Outlook Context</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="warning"><h3>⚠️ {headline}</h3><p>{decision}</p></div>',unsafe_allow_html=True)

        context=pd.DataFrame()
        if not model_ready.empty:
            context=model_ready[(model_ready["station"].astype(str).eq(station_name)) & (model_ready["date"].dt.normalize().eq(pd.Timestamp(r.get("date")).normalize()))].tail(1)
        ctx=context.iloc[0] if not context.empty else {}
        temp=ctx.get("temperature_mean",np.nan) if hasattr(ctx,"get") else np.nan
        hum=ctx.get("humidity_mean",np.nan) if hasattr(ctx,"get") else np.nan
        wind=ctx.get("wind_speed_mean",np.nan) if hasattr(ctx,"get") else np.nan
        rain=ctx.get("rainfall_total",np.nan) if hasattr(ctx,"get") else np.nan

        st.markdown(f'<div class="section" style="margin-top:10px">☁️ Observed Weather · {observation_date:%d %b %Y} ({station_name})</div>',unsafe_allow_html=True)
        w1,w2=st.columns(2)
        with w1: st.markdown(f'<div class="weather"><div class="label">🌡️ Temperature</div><div class="value">{safe_num(temp)}°C</div><div class="muted">daily mean</div></div>',unsafe_allow_html=True)
        with w2: st.markdown(f'<div class="weather"><div class="label">💧 Humidity</div><div class="value">{safe_num(hum,".0f")}%</div><div class="muted">daily mean</div></div>',unsafe_allow_html=True)
        w3,w4=st.columns(2)
        with w3: st.markdown(f'<div class="weather"><div class="label">🌬️ Wind Speed</div><div class="value">{safe_num(wind)} m/s</div><div class="muted">daily mean</div></div>',unsafe_allow_html=True)
        with w4: st.markdown(f'<div class="weather"><div class="label">🌧️ Rainfall</div><div class="value">{safe_num(rain)} mm</div><div class="muted">daily total</div></div>',unsafe_allow_html=True)

        fsi=r.get("fsi_total",np.nan); nearest=r.get("nearest_upwind_fire_km",np.nan); upfrp=r.get("upwind_frp_sum",np.nan)
        st.markdown('<div class="section" style="margin-top:10px">🔥 Fire Influence & FSI</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="fsi"><div style="display:flex;justify-content:space-between"><div><div style="font-size:2rem">🔥</div><div class="level">{influence}</div></div><div style="text-align:right"><div class="muted">FSI score</div><div class="big" style="font-size:1.5rem">{safe_num(fsi,".2f")}</div></div></div><hr style="border:none;border-top:1px solid #f3d7ad"><div class="muted">Engineered fire-influence indicator; not a validated smoke-dispersion estimate.</div></div>',unsafe_allow_html=True)
        m1,m2=st.columns(2); m1.metric("📍 Nearest upwind fire",f"{safe_num(nearest)} km"); m2.metric("🌬️ Upwind FRP",f"{safe_num(upfrp)} MW")


# ABOUT FIRE2AIR DARWIN — BOTTOM OF PAGE
st.markdown("""
<div class="about-card">
<h3>ℹ️ About Fire2Air Darwin</h3>
<p><b>Fire2Air Darwin</b> is a research decision-support prototype combining NT EPA air-quality/weather data with NASA FIRMS fire detections.</p>
<p>The system produces three next-day outputs:</p>
<ul>
<li><b>Classification:</b> whether PM2.5 may be elevated.</li>
<li><b>Regression:</b> predicted next-day mean PM2.5.</li>
<li><b>Count model:</b> predicted hours with elevated PM2.5.</li>
</ul>
<p>Fire distance, FRP, fire age, wind alignment and the Fire Smoke Influence Index (FSI) are used with air-quality and weather predictors.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="footer-note">ⓘ <b>Fire2Air Darwin is a research prototype.</b> Displayed outputs are historical forecasts, not live updates. Input completeness is not prediction confidence. Please check NT EPA for official air-quality information.</div>',unsafe_allow_html=True)
