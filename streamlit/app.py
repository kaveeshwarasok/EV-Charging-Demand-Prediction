from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from typing import Any
import html
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# 1. PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EV Intelligence | Charging Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# 2. PROJECT PATHS (preserves the existing project structure)
# ============================================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
REPORT_DIR = PROJECT_DIR / "reports"
EXPLAIN_DIR = OUTPUT_DIR / "explainability"

DATA_CANDIDATES = [
    PROJECT_DIR / "data" / "processed" / "ChargingRecords_cleaned.csv",
    PROJECT_DIR / "data_processed" / "ChargingRecords_cleaned.csv",
    PROJECT_DIR / "ChargingRecords.csv",
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)

REGRESSION_MODEL_PATH = MODEL_DIR / "best_demand_regression_model.joblib"
CLASSIFICATION_MODEL_PATH = MODEL_DIR / "best_demand_classification_model.joblib"
FORECAST_CANDIDATES = [
    OUTPUT_DIR / "future_demand_forecast.csv",
    PROJECT_DIR / "notebook" / "outputs" / "future_demand_forecast.csv",
]


# ============================================================================
# 3. PREMIUM DARK SAAS THEME
# ============================================================================
CUSTOM_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --bg: #080B0A;
  --sidebar: #0D1310;
  --card: #141B17;
  --card-2: #18211C;
  --border: rgba(164, 190, 174, .12);
  --green: #00D99A;
  --green-2: #34E6A5;
  --text: #F4F8F5;
  --muted: #93A39A;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: rgba(8,11,10,.92); }
[data-testid="stToolbar"] { right: 1rem; }
[data-testid="stMain"] { background: var(--bg); }
.block-container { padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1600px; }
[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
[data-testid="stSidebar"] * { color: #DCE7E0; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .42rem .55rem; border-radius: 9px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: rgba(0,217,154,.08); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: rgba(0,217,154,.13); border-left: 3px solid var(--green); }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p { color: var(--green) !important; font-weight: 700; }
[data-testid="stSidebar"] hr { border-color: var(--border); }
[data-testid="stMetric"] { background: transparent; }
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input, [data-testid="stTimeInput"] input,
[data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div {
  background: #111915 !important; color: var(--text) !important; border-color: var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stTextInput"] label, [data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label, [data-testid="stNumberInput"] label,
[data-testid="stDateInput"] label, [data-testid="stTimeInput"] label { color: var(--muted) !important; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
  background: linear-gradient(135deg, #00D99A, #27B982); color: #06110C; border: 0;
  border-radius: 10px; font-weight: 800; padding: .58rem 1rem; transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
  transform: translateY(-1px); box-shadow: 0 8px 22px rgba(0,217,154,.18); color: #04100B;
}
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: var(--border) !important; }
.eyebrow { color: var(--green); text-transform: uppercase; letter-spacing: .16em; font-size: .70rem; font-weight: 800; }
.page-title { color: var(--text); font-family: Manrope, sans-serif; font-size: 1.75rem; font-weight: 800; line-height: 1.15; margin: .12rem 0 .25rem; }
.page-subtitle { color: var(--muted); font-size: .92rem; margin-bottom: 1.25rem; }
.brand { display:flex; align-items:center; gap: .65rem; padding: .4rem .2rem 1.2rem; }
.brand-mark { width: 39px; height: 39px; border-radius: 12px; display:flex; align-items:center; justify-content:center;
  color:#06110C; font-size:1.25rem; font-weight:900; background:linear-gradient(135deg,#34E6A5,#00B982); box-shadow:0 6px 20px rgba(0,217,154,.18); }
.brand-name { font-family:Manrope,sans-serif; font-size:1rem; font-weight:900; letter-spacing:.06em; color:#F4F8F5; }
.brand-caption { color:#819188; font-size:.65rem; letter-spacing:.13em; margin-top:-2px; }
.topbar { display:flex; align-items:center; justify-content:flex-end; gap:.8rem; color:#AAB8B0; padding:.2rem 0 .6rem; }
.top-icon { width:34px; height:34px; display:inline-flex; align-items:center; justify-content:center; border:1px solid var(--border); border-radius:50%; background:#121915; }
.profile-chip { width:34px; height:34px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; background:#183B2C; color:var(--green); font-weight:900; border:1px solid rgba(0,217,154,.3); }
.metric-card { min-height:126px; padding:17px 18px 15px; background:linear-gradient(145deg,#17201B,#121815); border:1px solid var(--border); border-radius:15px; box-shadow:0 8px 26px rgba(0,0,0,.16); transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease; }
.metric-card:hover { transform:translateY(-3px); border-color:rgba(0,217,154,.42); box-shadow:0 12px 28px rgba(0,0,0,.25),0 0 18px rgba(0,217,154,.05); }
.metric-top { display:flex; justify-content:space-between; align-items:center; color:var(--muted); font-size:.78rem; font-weight:600; }
.metric-icon { color:var(--green); background:rgba(0,217,154,.10); border:1px solid rgba(0,217,154,.15); width:31px; height:31px; display:inline-flex; align-items:center; justify-content:center; border-radius:9px; font-size:1rem; }
.metric-value { color:#F7FBF8; font-family:Manrope,sans-serif; font-size:1.62rem; font-weight:800; margin-top:12px; letter-spacing:-.03em; }
.metric-foot { color:#7F9187; font-size:.7rem; margin-top:2px; }
.panel { background:linear-gradient(145deg,#151D18,#111713); border:1px solid var(--border); border-radius:15px; padding:15px 17px; }
.panel-title { color:#EFF7F1; font-weight:800; font-family:Manrope,sans-serif; font-size:.98rem; margin-bottom:.15rem; }
.panel-caption { color:var(--muted); font-size:.75rem; margin-bottom:.55rem; }
.section-kicker { color:var(--green); font-size:.68rem; text-transform:uppercase; letter-spacing:.14em; font-weight:800; margin:1.1rem 0 .3rem; }
.result-card { background:linear-gradient(135deg,rgba(0,217,154,.16),rgba(0,217,154,.035)); border:1px solid rgba(0,217,154,.28); border-radius:16px; padding:20px; }
.result-label { color:#A8C7B7; font-size:.8rem; font-weight:700; }
.result-value { color:#43E8AE; font-family:Manrope,sans-serif; font-size:2.15rem; font-weight:900; margin-top:4px; }
.status-pill { display:inline-block; padding:.32rem .68rem; border-radius:999px; font-size:.76rem; font-weight:800; background:rgba(0,217,154,.13); color:#42E7AC; border:1px solid rgba(0,217,154,.25); }
.status-pill.normal { background:rgba(142,163,151,.12); color:#C4D0C8; border-color:rgba(142,163,151,.2); }
.footer { text-align:center; color:#6F8177; font-size:.74rem; padding:1.4rem 0 .4rem; }
div[data-testid="stTabs"] button { color:#9DAEA4; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--green); border-bottom-color:var(--green); }
[data-testid="stExpander"] { background:#121915; border:1px solid var(--border); border-radius:12px; }
@media (max-width: 900px) { .page-title{font-size:1.45rem;} .metric-value{font-size:1.35rem;} .block-container{padding-left:1rem;padding-right:1rem;} }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# 4. HELPERS
# ============================================================================
GREEN = "#00D99A"
TEAL = "#31C7B0"
MINT = "#8BE7C1"
PALETTE = [GREEN, TEAL, MINT, "#7A9E91", "#B1C9BE", "#D2A85A", "#9D83D8"]
DAY_NAMES = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}


def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def page_heading(title: str, subtitle: str, eyebrow: str = "EV INTELLIGENCE") -> None:
    st.markdown(
        f'<div class="eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="page-title">{html.escape(title)}</div>'
        f'<div class="page-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, icon: str, foot: str = "From selected dataset") -> None:
    st.markdown(
        f'''<div class="metric-card">
          <div class="metric-top"><span>{html.escape(label)}</span><span class="metric-icon">{icon}</span></div>
          <div class="metric-value">{html.escape(value)}</div>
          <div class="metric-foot">{html.escape(foot)}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def panel_heading(title: str, caption: str = "") -> None:
    st.markdown(
        f'<div class="panel-title">{html.escape(title)}</div>'
        f'<div class="panel-caption">{html.escape(caption)}</div>',
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#DCE8E0", size=11),
        title=dict(font=dict(size=14, color="#F0F7F2"), x=0.02, xanchor="left"),
        margin=dict(l=12, r=12, t=52, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#A8B8AE")),
        hoverlabel=dict(bgcolor="#18231D", bordercolor="#315242", font_color="#F3FBF5"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(160,190,170,.12)", tickfont=dict(color="#9AA99F"))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(160,190,170,.10)", zeroline=False, tickfont=dict(color="#9AA99F"))
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    st.plotly_chart(style_figure(fig), use_container_width=True, config={"displayModeBar": False}, key=key)


def first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def read_csv_optional(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


@st.cache_data(show_spinner="Loading EV charging records...")
def load_data(path_string: str) -> pd.DataFrame:
    data = pd.read_csv(path_string)
    for col in ["StartDatetime", "EndDatetime"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    for col in ["Demand", "Duration", "ChargerCompany", "ChargerType"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    if "StartDatetime" in data.columns:
        data["StartHour"] = data["StartDatetime"].dt.hour
        data["DayOfWeek"] = data["StartDatetime"].dt.dayofweek
        data["Month"] = data["StartDatetime"].dt.month
        data["IsWeekend"] = (data["DayOfWeek"] >= 5).astype("Int64")
    return data


@st.cache_resource(show_spinner="Loading saved machine-learning models...")
def load_model(path_string: str):
    # Reuses the saved sklearn/XGBoost pipeline; no model training occurs here.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        return joblib.load(path_string)


def model_feature_names(model) -> list[str]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    raise ValueError("The saved pipeline does not expose its expected input feature names.")


def unique_options(series: pd.Series) -> list[Any]:
    values = series.dropna().unique().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def render_metric_row(items: list[tuple[str, str, str, str]], columns: int = 3) -> None:
    cols = st.columns(columns)
    for index, (label, value, icon, foot) in enumerate(items):
        with cols[index % columns]:
            metric_card(label, value, icon, foot)


def demand_time_data(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    if "StartDatetime" in work.columns:
        work = work.dropna(subset=["StartDatetime"])
        work["Date"] = work["StartDatetime"].dt.floor("D")
        work["Hour"] = work["StartDatetime"].dt.hour
        work["DayName"] = work["StartDatetime"].dt.day_name()
        work["DayType"] = np.where(work["StartDatetime"].dt.dayofweek >= 5, "Weekend", "Weekday")
    return work


# ============================================================================
# 5. LOAD ACTUAL PROJECT DATA
# ============================================================================
if DATA_PATH is None:
    st.error("Charging dataset not found. Expected data/processed/ChargingRecords_cleaned.csv or ChargingRecords.csv in the project root.")
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Dataset loading failed: {exc}")
    st.stop()

required_columns = {"Demand", "Location", "ChargerType"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.error(f"Dataset is missing required columns: {sorted(missing_columns)}")
    st.stop()


# ============================================================================
# 6. SIDEBAR BRAND, NAVIGATION, AND FILTERS
# ============================================================================
NAV_ITEMS = [
    "Dashboard Overview",
    "Data Analytics",
    "Location Analysis",
    "Time Series Analysis",
    "Regression Prediction",
    "Classification Prediction",
    "Future Demand Forecast",
    "Model Performance",
    "Explainable AI",
    "Dataset Explorer",  # Retained from the existing dashboard.
]
NAV_ICONS = {
    "Dashboard Overview": "▦",
    "Data Analytics": "◫",
    "Location Analysis": "⌖",
    "Time Series Analysis": "◷",
    "Regression Prediction": "⌁",
    "Classification Prediction": "◈",
    "Future Demand Forecast": "↗",
    "Model Performance": "▤",
    "Explainable AI": "✳",
    "Dataset Explorer": "▧",
}

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">ϟ</div><div><div class="brand-name">EV INTELLIGENCE</div><div class="brand-caption">CHARGING ANALYTICS PLATFORM</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
    selected_page = st.radio(
        "Navigation",
        NAV_ITEMS,
        format_func=lambda item: f"{NAV_ICONS[item]}   {item}",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div class="eyebrow">DATA FILTERS</div>', unsafe_allow_html=True)

    location_options = unique_options(df["Location"].astype(str))
    charger_options = unique_options(df["ChargerType"])
    company_options = unique_options(df["ChargerCompany"]) if "ChargerCompany" in df.columns else []

    selected_locations = st.multiselect("Charging locations", location_options, default=location_options)
    selected_chargers = st.multiselect("Charger types", charger_options, default=charger_options, format_func=lambda x: str(x))
    selected_companies = st.multiselect("Charger companies", company_options, default=company_options, format_func=lambda x: str(x)) if company_options else []

    if "StartDatetime" in df.columns and df["StartDatetime"].notna().any():
        min_date = df["StartDatetime"].min().date()
        max_date = df["StartDatetime"].max().date()
        selected_dates = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        selected_dates = None

    st.markdown("---")
    st.caption(f"Source: {DATA_PATH.name}")
    st.caption(f"Dataset records: {len(df):,}")

filtered_df = df.copy()
filtered_df = filtered_df[filtered_df["Location"].astype(str).isin([str(x) for x in selected_locations])]
filtered_df = filtered_df[filtered_df["ChargerType"].isin(selected_chargers)]
if "ChargerCompany" in filtered_df.columns and selected_companies:
    filtered_df = filtered_df[filtered_df["ChargerCompany"].isin(selected_companies)]
if selected_dates and isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2 and "StartDatetime" in filtered_df.columns:
    start_date, end_date = selected_dates
    filtered_df = filtered_df[
        filtered_df["StartDatetime"].notna()
        & (filtered_df["StartDatetime"].dt.date >= start_date)
        & (filtered_df["StartDatetime"].dt.date <= end_date)
    ]

if filtered_df.empty:
    st.warning("No records match the selected filters. Adjust the filters in the sidebar.")
    st.stop()


# ============================================================================
# 7. GLOBAL TOP BAR AND SEARCH
# ============================================================================
top_left, top_right = st.columns([4.8, 1.5])
with top_left:
    st.markdown('<div class="eyebrow">LIVE ANALYTICS WORKSPACE</div>', unsafe_allow_html=True)
with top_right:
    st.markdown('<div class="topbar"><span class="top-icon">♧</span><span class="top-icon">◐</span><span class="profile-chip">EV</span></div>', unsafe_allow_html=True)

search_text = st.text_input("Search charging records", placeholder="Search location, charger, company or record value...", label_visibility="collapsed")
if search_text.strip():
    mask = filtered_df.astype(str).apply(lambda col: col.str.contains(search_text.strip(), case=False, na=False)).any(axis=1)
    filtered_df = filtered_df[mask]
    if filtered_df.empty:
        st.warning("No matching records found for your search.")
        st.stop()

work_df = demand_time_data(filtered_df)


# ============================================================================
# 8. SHARED ANALYTICS CHARTS
# ============================================================================
def daily_chart(data: pd.DataFrame, title: str = "Daily charging demand") -> go.Figure | None:
    if "StartDatetime" not in data.columns:
        return None
    temp = data.dropna(subset=["StartDatetime"]).copy()
    if temp.empty:
        return None
    daily = temp.assign(Date=temp["StartDatetime"].dt.floor("D")).groupby("Date", as_index=False)["Demand"].sum()
    fig = px.area(daily, x="Date", y="Demand", title=title)
    fig.update_traces(line_color=GREEN, fillcolor="rgba(0,217,154,.12)", line_width=2.5)
    return fig


def hourly_chart(data: pd.DataFrame, title: str = "Hourly charging pattern") -> go.Figure | None:
    if "StartDatetime" not in data.columns:
        return None
    temp = data.dropna(subset=["StartDatetime"]).copy()
    if temp.empty:
        return None
    temp["Hour"] = temp["StartDatetime"].dt.hour
    hourly = temp.groupby("Hour", as_index=False)["Demand"].sum().set_index("Hour").reindex(range(24), fill_value=0).reset_index()
    fig = px.bar(hourly, x="Hour", y="Demand", title=title)
    fig.update_traces(marker_color=GREEN, marker_line_width=0)
    fig.update_xaxes(dtick=1, title="Hour of day")
    return fig


def location_chart(data: pd.DataFrame, title: str = "Demand by charging location") -> go.Figure:
    grouped = data.groupby("Location", as_index=False)["Demand"].sum().sort_values("Demand", ascending=True).tail(15)
    fig = px.bar(grouped, x="Demand", y="Location", orientation="h", title=title, color_discrete_sequence=[GREEN])
    return fig


def charger_type_chart(data: pd.DataFrame, title: str = "Charger type distribution") -> go.Figure:
    grouped = data.groupby("ChargerType", as_index=False).agg(TotalDemand=("Demand", "sum"), Sessions=("Demand", "size"))
    fig = px.pie(grouped, names="ChargerType", values="Sessions", hole=.68, title=title, color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return fig


def company_chart(data: pd.DataFrame, title: str = "Demand by charger company") -> go.Figure | None:
    if "ChargerCompany" not in data.columns:
        return None
    grouped = data.groupby("ChargerCompany", as_index=False)["Demand"].sum().sort_values("Demand", ascending=False)
    fig = px.bar(grouped, x="ChargerCompany", y="Demand", title=title, color_discrete_sequence=[TEAL])
    return fig


def weekday_weekend_chart(data: pd.DataFrame, title: str = "Weekday vs weekend demand") -> go.Figure | None:
    if "StartDatetime" not in data.columns:
        return None
    temp = data.dropna(subset=["StartDatetime"]).copy()
    if temp.empty:
        return None
    temp["DayType"] = np.where(temp["StartDatetime"].dt.dayofweek >= 5, "Weekend", "Weekday")
    grouped = temp.groupby("DayType", as_index=False).agg(TotalDemand=("Demand", "sum"), Sessions=("Demand", "size"))
    fig = px.bar(grouped, x="DayType", y="TotalDemand", color="DayType", title=title, color_discrete_map={"Weekday": "#6E8B7D", "Weekend": GREEN})
    fig.update_layout(showlegend=False)
    return fig


# ============================================================================
# 9. PAGE: DASHBOARD OVERVIEW
# ============================================================================
if selected_page == "Dashboard Overview":
    page_heading("EV Intelligence", "Welcome back. Monitor charging activity, demand patterns and infrastructure utilization.")

    total_sessions = len(filtered_df)
    total_demand = safe_number(filtered_df["Demand"].sum())
    average_demand = safe_number(filtered_df["Demand"].mean())
    total_locations = filtered_df["Location"].nunique()
    peak_demand = safe_number(filtered_df["Demand"].max())
    average_duration = safe_number(filtered_df["Duration"].mean()) if "Duration" in filtered_df.columns else np.nan
    duration_label = f"{average_duration:,.0f} min" if np.isfinite(average_duration) else "N/A"

    render_metric_row([
        ("Total Charging Sessions", f"{total_sessions:,}", "ϟ", "Filtered charging records"),
        ("Total Energy Demand", f"{total_demand:,.1f}", "◉", "Sum of Demand field"),
        ("Average Demand", f"{average_demand:,.2f}", "⌁", "Mean per charging session"),
        ("Charging Locations", f"{total_locations:,}", "⌖", "Unique selected locations"),
        ("Peak Demand", f"{peak_demand:,.2f}", "↗", "Highest single-session demand"),
        ("Avg. Charging Duration", duration_label, "◷", "Mean session duration"),
    ], columns=3)

    st.markdown('<div class="section-kicker">DEMAND MONITORING</div>', unsafe_allow_html=True)
    left, right = st.columns([1.65, 1])
    with left:
        fig = daily_chart(filtered_df, "Daily demand trend")
        if fig is not None:
            show_chart(fig, "overview_daily")
        else:
            st.info("StartDatetime is not available for daily trend analysis.")
    with right:
        fig = hourly_chart(filtered_df, "Hourly charging pattern")
        if fig is not None:
            show_chart(fig, "overview_hourly")
        else:
            st.info("StartDatetime is not available for hourly analysis.")

    left, right = st.columns(2)
    with left:
        show_chart(location_chart(filtered_df, "Top locations by demand"), "overview_location")
    with right:
        show_chart(charger_type_chart(filtered_df, "Charging sessions by charger type"), "overview_charger")


# ============================================================================
# 10. PAGE: DATA ANALYTICS
# ============================================================================
elif selected_page == "Data Analytics":
    page_heading("Data Analytics", "Explore demand behavior across time, charger technologies and charging operators.")
    c1, c2 = st.columns(2)
    with c1:
        fig = daily_chart(filtered_df, "Daily demand trends")
        if fig is not None: show_chart(fig, "analytics_daily")
    with c2:
        fig = hourly_chart(filtered_df, "Hourly charging patterns")
        if fig is not None: show_chart(fig, "analytics_hourly")
    c1, c2 = st.columns(2)
    with c1:
        show_chart(charger_type_chart(filtered_df, "Charger type distribution"), "analytics_charger_type")
    with c2:
        fig = company_chart(filtered_df, "Charger company analysis")
        if fig is not None: show_chart(fig, "analytics_company")
    c1, c2 = st.columns(2)
    with c1:
        fig = weekday_weekend_chart(filtered_df)
        if fig is not None: show_chart(fig, "analytics_weekend")
    with c2:
        if "ChargerCompany" in filtered_df.columns:
            company_summary = filtered_df.groupby("ChargerCompany", as_index=False).agg(Sessions=("Demand", "size"), TotalDemand=("Demand", "sum"), AverageDemand=("Demand", "mean"))
            st.dataframe(company_summary, use_container_width=True, hide_index=True)


# ============================================================================
# 11. PAGE: LOCATION ANALYSIS
# ============================================================================
elif selected_page == "Location Analysis":
    page_heading("Location Analysis", "Compare charging demand, session volume and average utilization by location.")
    location_summary = filtered_df.groupby("Location", as_index=False).agg(
        Sessions=("Demand", "size"), TotalDemand=("Demand", "sum"), AverageDemand=("Demand", "mean"), PeakSessionDemand=("Demand", "max")
    ).sort_values("TotalDemand", ascending=False)
    top_location = location_summary.iloc[0] if not location_summary.empty else None
    if top_location is not None:
        render_metric_row([
            ("Locations in View", f"{len(location_summary):,}", "⌖", "Unique filtered locations"),
            ("Highest-Demand Location", str(top_location["Location"]), "↗", "By total recorded demand"),
            ("Top Location Demand", f"{top_location['TotalDemand']:,.1f}", "◉", "Sum of session demand"),
        ], columns=3)
    show_chart(location_chart(filtered_df, "Total demand by location"), "location_demand")
    st.markdown('<div class="section-kicker">LOCATION PERFORMANCE TABLE</div>', unsafe_allow_html=True)
    st.dataframe(location_summary, use_container_width=True, hide_index=True)


# ============================================================================
# 12. PAGE: TIME SERIES ANALYSIS
# ============================================================================
elif selected_page == "Time Series Analysis":
    page_heading("Time Series Analysis", "Review daily, weekly, monthly and hourly demand patterns using the selected records.")
    if "StartDatetime" not in filtered_df.columns or filtered_df["StartDatetime"].notna().sum() == 0:
        st.warning("StartDatetime is unavailable. Time-series charts cannot be generated.")
    else:
        temp = filtered_df.dropna(subset=["StartDatetime"]).copy().set_index("StartDatetime").sort_index()
        daily = temp["Demand"].resample("D").sum().rename("Demand").reset_index()
        weekly = temp["Demand"].resample("W").sum().rename("Demand").reset_index()
        monthly = temp["Demand"].resample("MS").sum().rename("Demand").reset_index()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(daily, x="StartDatetime", y="Demand", markers=True, title="Daily demand")
            fig.update_traces(line_color=GREEN)
            show_chart(fig, "ts_daily")
        with c2:
            fig = px.bar(weekly, x="StartDatetime", y="Demand", title="Weekly demand", color_discrete_sequence=[TEAL])
            show_chart(fig, "ts_weekly")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(monthly, x="StartDatetime", y="Demand", markers=True, title="Monthly demand")
            fig.update_traces(line_color=MINT)
            show_chart(fig, "ts_monthly")
        with c2:
            fig = hourly_chart(filtered_df, "Demand by hour of day")
            if fig is not None: show_chart(fig, "ts_hourly")
        st.markdown('<div class="section-kicker">TIME-SERIES SUMMARY</div>', unsafe_allow_html=True)
        render_metric_row([
            ("Days Analyzed", f"{len(daily):,}", "◷", "Daily resampled observations"),
            ("Weeks Analyzed", f"{len(weekly):,}", "▦", "Weekly resampled observations"),
            ("Months Analyzed", f"{len(monthly):,}", "▤", "Monthly resampled observations"),
        ], columns=3)


# ============================================================================
# 13. PAGE: REGRESSION PREDICTION
# ============================================================================
elif selected_page == "Regression Prediction":
    page_heading("Regression Prediction", "Estimate charging-session demand with your saved regression pipeline.")
    if not REGRESSION_MODEL_PATH.exists():
        st.error(f"Regression model not found: {REGRESSION_MODEL_PATH}")
    else:
        try:
            regression_model = load_model(str(REGRESSION_MODEL_PATH))
            features = model_feature_names(regression_model)
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Charging session inputs**")
            with st.form("regression_prediction_form"):
                input_values: dict[str, Any] = {}
                cols = st.columns(2)
                for i, feature in enumerate(features):
                    with cols[i % 2]:
                        if feature in {"StartHour"}:
                            input_values[feature] = st.slider("Start hour", 0, 23, 12)
                        elif feature == "DayOfWeek":
                            input_values[feature] = st.selectbox("Day of week", list(range(7)), format_func=lambda x: f"{DAY_NAMES[x]} ({x})")
                        elif feature == "Month":
                            input_values[feature] = st.selectbox("Month", list(range(1, 13)), index=0)
                        elif feature == "IsWeekend":
                            input_values[feature] = st.selectbox("Weekend status", [0, 1], format_func=lambda x: "Weekend" if x == 1 else "Weekday")
                        elif feature in df.columns:
                            opts = unique_options(df[feature])
                            if not opts:
                                st.warning(f"No available values for {feature}.")
                            else:
                                input_values[feature] = st.selectbox(feature.replace("_", " "), opts, format_func=lambda x: str(x))
                        else:
                            st.error(f"Model feature '{feature}' is not available in the dataset or input form.")
                submitted = st.form_submit_button("Run demand prediction", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if submitted:
                missing = [f for f in features if f not in input_values]
                if missing:
                    st.error(f"Missing required model inputs: {missing}")
                else:
                    model_input = pd.DataFrame([{f: input_values[f] for f in features}], columns=features)
                    prediction = float(regression_model.predict(model_input)[0])
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">PREDICTED CHARGING DEMAND</div><div class="result-value">{prediction:,.2f}</div><div class="panel-caption">Output from the existing saved regression pipeline</div></div>',
                        unsafe_allow_html=True,
                    )
        except Exception as exc:
            st.error(f"Regression prediction could not be completed: {exc}")


# ============================================================================
# 14. PAGE: CLASSIFICATION PREDICTION
# ============================================================================
elif selected_page == "Classification Prediction":
    page_heading("Classification Prediction", "Classify a charging session as High Demand or Normal Demand using the saved classifier.")
    if not CLASSIFICATION_MODEL_PATH.exists():
        st.error(f"Classification model not found: {CLASSIFICATION_MODEL_PATH}")
    else:
        try:
            classification_model = load_model(str(CLASSIFICATION_MODEL_PATH))
            features = model_feature_names(classification_model)
            with st.form("classification_prediction_form"):
                input_values: dict[str, Any] = {}
                cols = st.columns(2)
                for i, feature in enumerate(features):
                    with cols[i % 2]:
                        if feature == "StartDay":
                            input_values[feature] = st.date_input("Start date", value=date(2022, 9, 15)).isoformat()
                        elif feature == "StartTime":
                            input_values[feature] = st.time_input("Start time", value=time(12, 0)).strftime("%H:%M:%S")
                        elif feature in df.columns:
                            opts = unique_options(df[feature])
                            if opts:
                                input_values[feature] = st.selectbox(feature.replace("_", " "), opts, format_func=lambda x: str(x))
                            else:
                                st.warning(f"No values available for {feature}.")
                        else:
                            st.error(f"Model feature '{feature}' is not available in the dataset.")
                submitted = st.form_submit_button("Classify demand", use_container_width=True)
            if submitted:
                missing = [f for f in features if f not in input_values]
                if missing:
                    st.error(f"Missing required model inputs: {missing}")
                else:
                    model_input = pd.DataFrame([{f: input_values[f] for f in features}], columns=features)
                    predicted_class = int(classification_model.predict(model_input)[0])
                    label = "High Demand" if predicted_class == 1 else "Normal Demand"
                    pill_class = "" if predicted_class == 1 else "normal"
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">PREDICTED DEMAND CATEGORY</div><div class="result-value">{html.escape(label)}</div><span class="status-pill {pill_class}">{html.escape(label)}</span></div>',
                        unsafe_allow_html=True,
                    )
                    if hasattr(classification_model, "predict_proba"):
                        probabilities = classification_model.predict_proba(model_input)[0]
                        classes = list(getattr(classification_model, "classes_", range(len(probabilities))))
                        probability_df = pd.DataFrame({"Class": ["High Demand" if int(c) == 1 else "Normal Demand" for c in classes], "Probability": probabilities})
                        probability_fig = px.bar(probability_df, x="Class", y="Probability", title="Class probability", color="Class", color_discrete_map={"High Demand": GREEN, "Normal Demand": "#789187"})
                        probability_fig.update_layout(showlegend=False, yaxis_range=[0, 1])
                        show_chart(probability_fig, "classification_probability")
        except Exception as exc:
            st.error(f"Classification prediction could not be completed: {exc}")


# ============================================================================
# 15. PAGE: FUTURE DEMAND FORECAST
# ============================================================================
elif selected_page == "Future Demand Forecast":
    page_heading("Future Demand Forecast", "Seven-day demand outlook generated by your existing time-series forecasting workflow.")
    forecast_path = first_existing(FORECAST_CANDIDATES)
    forecast_df = read_csv_optional(forecast_path)
    if forecast_df is None:
        st.warning("The saved future_demand_forecast.csv was not found in outputs/ or notebook/outputs/.")
    else:
        date_col = next((c for c in forecast_df.columns if "date" in c.lower()), None)
        demand_col = next((c for c in forecast_df.columns if "predicted" in c.lower() or "forecast" in c.lower() or "demand" in c.lower()), None)
        if date_col is None or demand_col is None:
            st.error(f"Forecast file columns could not be identified. Found: {forecast_df.columns.tolist()}")
        else:
            forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors="coerce")
            forecast_df[demand_col] = pd.to_numeric(forecast_df[demand_col], errors="coerce")
            forecast_df = forecast_df.dropna(subset=[date_col, demand_col]).sort_values(date_col)
            if forecast_df.empty:
                st.warning("The forecast file contains no valid date/demand rows.")
            else:
                peak_row = forecast_df.loc[forecast_df[demand_col].idxmax()]
                render_metric_row([
                    ("Forecast Days", f"{len(forecast_df):,}", "◷", "Rows in saved forecast CSV"),
                    ("Average Forecast Demand", f"{forecast_df[demand_col].mean():,.2f}", "⌁", "Mean of predicted daily demand"),
                    ("Peak Forecast Demand", f"{peak_row[demand_col]:,.2f}", "↗", f"Peak date: {peak_row[date_col].date()}"),
                ], columns=3)
                fig = px.line(forecast_df, x=date_col, y=demand_col, markers=True, title="7-day EV charging demand forecast")
                fig.update_traces(line=dict(color=GREEN, width=3), marker=dict(size=9, color=MINT))
                fig.add_hline(y=forecast_df[demand_col].mean(), line_dash="dot", line_color="#7B9587", annotation_text="Forecast average")
                show_chart(fig, "future_forecast_chart")
                st.markdown('<div class="section-kicker">DAILY PREDICTED DEMAND</div>', unsafe_allow_html=True)
                st.dataframe(forecast_df.rename(columns={date_col: "Date", demand_col: "PredictedDemand"}), use_container_width=True, hide_index=True)


# ============================================================================
# 16. PAGE: MODEL PERFORMANCE
# ============================================================================
elif selected_page == "Model Performance":
    page_heading("Model Performance", "Compare saved evaluation results. Metrics shown here are read from your project CSV reports.")
    regression_results_path = first_existing([
        REPORT_DIR / "regression_model_comparison.csv",
        REPORT_DIR / "regression_model_results.csv",
    ])
    best_regression_path = first_existing([REPORT_DIR / "best_regression_model_metrics.csv"])
    classification_results_path = first_existing([
        OUTPUT_DIR / "classification_model_results.csv",
        OUTPUT_DIR / "dashboard" / "classification_model_comparison.csv",
        REPORT_DIR / "classification_results.csv",
    ])
    reg_df = read_csv_optional(regression_results_path)
    best_reg_df = read_csv_optional(best_regression_path)
    cls_df = read_csv_optional(classification_results_path)

    st.markdown('<div class="section-kicker">REGRESSION MODEL METRICS</div>', unsafe_allow_html=True)
    if best_reg_df is not None and not best_reg_df.empty:
        row = best_reg_df.iloc[0]
        r2_key = "R2" if "R2" in best_reg_df.columns else ("R2 Score" if "R2 Score" in best_reg_df.columns else None)
        render_metric_row([
            ("MAE", f"{safe_number(row.get('MAE')):,.4f}", "⌁", "Saved best-model metric"),
            ("MSE", f"{safe_number(row.get('MSE')):,.4f}", "▤", "Saved best-model metric"),
            ("RMSE", f"{safe_number(row.get('RMSE')):,.4f}", "↗", "Saved best-model metric"),
        ], columns=3)
        if r2_key:
            st.metric("R² Score", f"{safe_number(row.get(r2_key)):,.4f}")
    elif reg_df is not None and not reg_df.empty:
        st.info("Best regression metrics file not found; displaying available comparison results below.")
    else:
        st.info("Regression model metrics CSV was not found in reports/.")
    if reg_df is not None:
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
        if "Model" in reg_df.columns:
            metric = "RMSE" if "RMSE" in reg_df.columns else ("MAE" if "MAE" in reg_df.columns else None)
            if metric:
                fig = px.bar(reg_df.sort_values(metric), x="Model", y=metric, title=f"Regression model comparison — {metric}", color_discrete_sequence=[GREEN])
                show_chart(fig, "regression_model_comparison")

    st.markdown('<div class="section-kicker">CLASSIFICATION MODEL METRICS</div>', unsafe_allow_html=True)
    if cls_df is not None and not cls_df.empty:
        metric_aliases = {"F1-Score": ["F1-Score", "F1 Score"], "ROC-AUC": ["ROC-AUC", "ROC AUC"]}
        available = []
        for wanted in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
            actual = next((c for c in metric_aliases.get(wanted, [wanted]) if c in cls_df.columns), None)
            if actual: available.append(actual)
        st.dataframe(cls_df, use_container_width=True, hide_index=True)
        if available:
            id_vars = ["Model"] if "Model" in cls_df.columns else []
            if id_vars:
                long_df = cls_df.melt(id_vars=id_vars, value_vars=available, var_name="Metric", value_name="Score")
                fig = px.bar(long_df, x="Model", y="Score", color="Metric", barmode="group", title="Classification model comparison", color_discrete_sequence=PALETTE)
                fig.update_yaxes(range=[0, 1])
                show_chart(fig, "classification_model_comparison")
            else:
                st.metric("Accuracy", f"{safe_number(cls_df.iloc[0].get('Accuracy')):.4f}")
    else:
        st.info("Classification model results CSV was not found in outputs/ or reports/.")


# ============================================================================
# 17. PAGE: EXPLAINABLE AI
# ============================================================================
elif selected_page == "Explainable AI":
    page_heading("Explainable AI", "Understand which charging-session features influence the saved model outputs.")
    shap_importance = read_csv_optional(EXPLAIN_DIR / "regression_shap_feature_importance.csv")
    shap_values = read_csv_optional(EXPLAIN_DIR / "regression_shap_values.csv")
    regression_lime = read_csv_optional(EXPLAIN_DIR / "regression_lime_explanation.csv")
    classification_lime = read_csv_optional(EXPLAIN_DIR / "classification_lime_explanation.csv")
    regression_importance = read_csv_optional(REPORT_DIR / "regression_feature_importance.csv")
    classification_importance = read_csv_optional(REPORT_DIR / "feature_importance.csv")

    tab_shap, tab_reg_lime, tab_cls_lime, tab_importance = st.tabs(["SHAP Analysis", "Regression LIME", "Classification LIME", "Model Feature Importance"])
    with tab_shap:
        if shap_importance is not None and not shap_importance.empty:
            feature_col = next((c for c in shap_importance.columns if "feature" in c.lower()), shap_importance.columns[0])
            value_col = next((c for c in shap_importance.columns if c != feature_col), shap_importance.columns[-1])
            sorted_shap = shap_importance.sort_values(value_col, ascending=True).tail(15)
            fig = px.bar(sorted_shap, x=value_col, y=feature_col, orientation="h", title="SHAP feature importance (mean absolute value)", color_discrete_sequence=[GREEN])
            show_chart(fig, "shap_importance")
            st.dataframe(shap_importance, use_container_width=True, hide_index=True)
        else:
            st.info("Regression SHAP feature importance CSV was not found.")
        if shap_values is not None and not shap_values.empty:
            st.markdown('<div class="section-kicker">SHAP SUMMARY DISTRIBUTION</div>', unsafe_allow_html=True)
            numeric_cols = shap_values.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                long_shap = shap_values[numeric_cols].melt(var_name="Feature", value_name="SHAP value")
                fig = px.strip(long_shap, x="SHAP value", y="Feature", color="Feature", title="SHAP value distribution by feature", color_discrete_sequence=PALETTE)
                fig.update_traces(jitter=.45, marker=dict(size=4, opacity=.62))
                show_chart(fig, "shap_summary_distribution")
            else:
                st.info("No numeric SHAP values are available for the summary distribution.")
        else:
            st.info("Regression SHAP values CSV was not found.")
    with tab_reg_lime:
        if regression_lime is not None:
            st.subheader("Regression LIME explanation")
            if len(regression_lime.columns) >= 2:
                fig = px.bar(regression_lime, x=regression_lime.columns[1], y=regression_lime.columns[0], orientation="h", title="Local feature contributions", color_discrete_sequence=[TEAL])
                show_chart(fig, "regression_lime_chart")
            st.dataframe(regression_lime, use_container_width=True, hide_index=True)
        else:
            st.info("Regression LIME explanation CSV was not found.")
    with tab_cls_lime:
        if classification_lime is not None:
            st.subheader("Classification LIME explanation")
            if len(classification_lime.columns) >= 2:
                fig = px.bar(classification_lime, x=classification_lime.columns[1], y=classification_lime.columns[0], orientation="h", title="Local classification feature contributions", color_discrete_sequence=[GREEN])
                show_chart(fig, "classification_lime_chart")
            st.dataframe(classification_lime, use_container_width=True, hide_index=True)
        else:
            st.info("Classification LIME explanation CSV was not found.")
    with tab_importance:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Regression feature importance")
            if regression_importance is not None:
                fcol = regression_importance.columns[0]
                vcol = regression_importance.columns[1]
                fig = px.bar(regression_importance.sort_values(vcol).tail(12), x=vcol, y=fcol, orientation="h", title="Saved regression importance", color_discrete_sequence=[GREEN])
                show_chart(fig, "reg_feature_importance")
                st.dataframe(regression_importance, use_container_width=True, hide_index=True)
            else:
                st.info("Regression feature importance report was not found.")
        with c2:
            st.subheader("Classification feature importance")
            if classification_importance is not None:
                fcol = classification_importance.columns[0]
                vcol = classification_importance.columns[1]
                fig = px.bar(classification_importance.sort_values(vcol).tail(12), x=vcol, y=fcol, orientation="h", title="Saved classification importance", color_discrete_sequence=[TEAL])
                show_chart(fig, "cls_feature_importance")
                st.dataframe(classification_importance, use_container_width=True, hide_index=True)
            else:
                st.info("Classification feature importance report was not found.")


# ============================================================================
# 18. PAGE: DATASET EXPLORER (EXISTING FEATURE RETAINED)
# ============================================================================
elif selected_page == "Dataset Explorer":
    page_heading("Dataset Explorer", "Search, inspect and export the actual filtered charging records.")
    search_col = st.selectbox("Search column", filtered_df.columns.tolist())
    search_value = st.text_input("Find a value", placeholder="Enter text or a number...")
    if search_value.strip():
        view_df = filtered_df[filtered_df[search_col].astype(str).str.contains(search_value.strip(), case=False, na=False)]
    else:
        view_df = filtered_df
    st.caption(f"{len(view_df):,} matching records")
    st.dataframe(view_df.head(500), use_container_width=True, height=460, hide_index=True)
    st.download_button(
        "Download filtered dataset",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_ev_charging_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================================
# 19. FOOTER
# ============================================================================
st.markdown(
    '<div class="footer">⚡ EV INTELLIGENCE &nbsp;·&nbsp; EV Charging Demand Prediction & Usage Pattern Analysis<br>Machine Learning · Time-Series Forecasting · Explainable AI</div>',
    unsafe_allow_html=True,
)
