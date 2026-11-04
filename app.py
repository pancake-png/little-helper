
from pathlib import Path
from datetime import date
from urllib.parse import urlencode
from urllib.request import urlopen
import json

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="LITTLE HELPER",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "staffing_data.csv"

ROLE_LABELS = {
    "Hrs_RN": "Registered nurses (RN)",
    "Hrs_LPN": "Licensed practical nurses (LPN)",
    "Hrs_CNA": "Nursing assistants (CNA)",
    "Hrs_NAtrn": "Nurse aide trainees",
    "Hrs_MedAide": "Medication aides",
}

WEATHER_OPTIONS = ["Clear", "Rain", "Snow / ice"]

# Fixed demo locations avoid exposing any user's location.
DEMO_CITIES = {
    "San Francisco, CA": (37.7749, -122.4194),
    "New York, NY": (40.7128, -74.0060),
    "Chicago, IL": (41.8781, -87.6298),
    "Miami, FL": (25.7617, -80.1918),
    "Toronto, ON": (43.6532, -79.3832),
}

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_weather(city):
    """Get model-based current conditions from Open-Meteo (cached 30 min)."""
    latitude, longitude = DEMO_CITIES[city]
    params = urlencode({
        "latitude": latitude, "longitude": longitude,
        "current": "temperature_2m,weather_code,precipitation,wind_speed_10m",
        "timezone": "auto", "temperature_unit": "celsius",
    })
    with urlopen(f"https://api.open-meteo.com/v1/forecast?{params}", timeout=8) as response:
        payload = json.load(response)
    current = payload["current"]
    code = int(current["weather_code"])
    if code in (71, 73, 75, 77, 85, 86, 56, 57, 66, 67):
        category, label = "Snow / ice", "Snow / ice"
    elif code in (51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99):
        category, label = "Rain", "Rain / storm"
    elif code in (45, 48):
        category, label = "Clear", "Fog"
    elif code in (1, 2, 3):
        category, label = "Clear", "Cloudy / partly cloudy"
    else:
        category, label = "Clear", "Clear"
    return {
        "category": category, "label": label,
        "temperature": float(current["temperature_2m"]),
        "precipitation": float(current["precipitation"]),
        "wind": float(current["wind_speed_10m"]),
        "observed_at": current.get("time", ""),
    }

# Illustrative assumptions for the visitor simulation.
# These are not fitted to the CMS dataset.
VISITOR_ACTIVITY_FACTORS = {
    "Quiet": 0.55,
    "Typical": 1.00,
    "Busy": 1.45,
}

WEATHER_FACTORS = {
    "Clear": 1.00,
    "Rain": 0.85,
    "Snow / ice": 0.60,
}

APPOINTMENT_FACTORS = {
    "Low": 0.80,
    "Typical": 1.00,
    "High": 1.25,
}


# ==================================================
# 2. VISUAL STYLE
# ==================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.7rem !important;
        font-weight: 700 !important;
        letter-spacing: -1.5px;
    }

    h2, h3 {
        letter-spacing: -0.4px;
    }

    .subtitle {
        color: #667085;
        font-size: 1rem;
        margin-top: -0.8rem;
        margin-bottom: 1.5rem;
    }

    .disclaimer {
        background: #EDF5F8;
        border: 1px solid #D6E7EE;
        border-radius: 12px;
        padding: 14px 17px;
        color: #344D5A;
        font-size: 0.86rem;
        margin: 1rem 0 1.5rem 0;
    }

    .section-intro {
        color: #667085;
        font-size: 0.88rem;
        margin-top: -0.35rem;
        margin-bottom: 1rem;
    }

    .small-note {
        color: #667085;
        font-size: 0.78rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #DFE5E9;
        border-radius: 15px;
    }

    div[data-testid="stMetric"] {
        background: #F4F7F8;
        border-radius: 11px;
        padding: 12px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    div[data-testid="stChatInput"] {
        padding-top: 0.5rem;
    }

    hr {
        border-color: #E6EAED;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# 3. LOAD HISTORICAL DATA
# ==================================================

@st.cache_data
def load_data(path_string, modified_time, file_size):
    data = pd.read_csv(path_string, encoding="cp1252")

    required_columns = ["STATE", "WorkDate"]
    missing = [
        col for col in required_columns if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing required dataset columns: "
            + ", ".join(missing)
        )

    data["STATE"] = data["STATE"].astype(str).str.strip()
    # CMS WorkDate values are YYYYMMDD integers; parse as calendar dates,
    # not nanoseconds since the Unix epoch.
    data["WorkDate"] = pd.to_datetime(
        data["WorkDate"].astype(str), format="%Y%m%d", errors="coerce"
    )

    census_candidates = [
        "MDScensus",
        "Census",
        "ResidentCensus",
        "Resident_Count",
    ]

    census_column = next(
        (
            col for col in census_candidates
            if col in data.columns
        ),
        None,
    )

    if census_column is None:
        raise ValueError(
            "Could not identify the resident census column. "
            "Check the column names in staffing_data.csv."
        )

    data["resident_census"] = pd.to_numeric(
        data[census_column], errors="coerce"
    )

    data["weekday"] = data["WorkDate"].dt.dayofweek
    data["month"] = data["WorkDate"].dt.month

    available_roles = [
        role for role in ROLE_LABELS if role in data.columns
    ]

    if not available_roles:
        raise ValueError(
            "No supported staffing-hours columns were found."
        )

    for role in available_roles:
        data[role] = pd.to_numeric(
            data[role], errors="coerce"
        )

    data = data.dropna(
        subset=[
            "STATE",
            "WorkDate",
            "resident_census",
            "weekday",
            "month",
        ]
    ).copy()

    data = data[
        (data["STATE"] != "")
        & (data["resident_census"] >= 0)
    ].copy()

    return data, available_roles, census_column


@st.cache_resource
def train_models(_data, available_roles, data_signature):
    models = {}
    feature_columns = [
        "STATE",
        "resident_census",
        "weekday",
        "month",
    ]

    training_data = _data.copy()

    # Keep local training manageable.
    if len(training_data) > 30000:
        training_data = training_data.sample(
            n=30000,
            random_state=42,
        )

    for role in available_roles:
        role_data = training_data.dropna(
            subset=[role]
        ).copy()

        role_data = role_data[role_data[role] >= 0]

        if len(role_data) < 20:
            continue

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "state",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["STATE"],
                ),
                (
                    "numeric",
                    "passthrough",
                    [
                        "resident_census",
                        "weekday",
                        "month",
                    ],
                ),
            ]
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=40,
                        max_depth=12,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(
            role_data[feature_columns],
            role_data[role],
        )

        models[role] = model

    return models


if not DATA_PATH.exists():
    st.error(
        "Could not find staffing_data.csv. "
        "Place it in the same folder as app.py."
    )
    st.stop()

try:
    file_info = DATA_PATH.stat()

    data_signature = (
        str(DATA_PATH),
        file_info.st_mtime,
        file_info.st_size,
    )

    data, available_roles, census_column = load_data(
        *data_signature
    )

    with st.spinner("Preparing historical benchmarks..."):
        models = train_models(
            data,
            available_roles,
            data_signature,
        )

except Exception as error:
    st.error(f"Could not prepare the dataset: {error}")
    st.stop()

if not models:
    st.error(
        "Not enough valid records were available to train "
        "the staffing benchmark models."
    )
    st.stop()


# ==================================================
# 4. SHARED SCENARIO STATE
# ==================================================

states = sorted(data["STATE"].unique().tolist())

if not states:
    st.error("No usable state values were found in the dataset.")
    st.stop()

default_state = "CA" if "CA" in states else states[0]

if "scenario_state" not in st.session_state:
    st.session_state["scenario_state"] = default_state

if st.session_state["scenario_state"] not in states:
    st.session_state["scenario_state"] = default_state

if "resident_count" not in st.session_state:
    st.session_state["resident_count"] = 84

if "facility_capacity" not in st.session_state:
    st.session_state["facility_capacity"] = 100

if "scenario_date" not in st.session_state:
    st.session_state["scenario_date"] = date.today()

if "weather_city" not in st.session_state:
    st.session_state["weather_city"] = "San Francisco, CA"
if "use_live_weather" not in st.session_state:
    st.session_state["use_live_weather"] = True

if "scenario_weather" not in st.session_state:
    st.session_state["scenario_weather"] = "Rain"

if "scenario_visitor_activity" not in st.session_state:
    st.session_state["scenario_visitor_activity"] = "Typical"

if "scenario_appointments" not in st.session_state:
    st.session_state["scenario_appointments"] = "Typical"


# ==================================================
# 5. PREDICT HISTORICAL STAFFING BENCHMARKS
# ==================================================

def get_predictions():
    scenario_date = st.session_state["scenario_date"]

    features = pd.DataFrame(
        [
            {
                "STATE": st.session_state["scenario_state"],
                "resident_census": st.session_state["resident_count"],
                "weekday": scenario_date.weekday(),
                "month": scenario_date.month,
            }
        ]
    )

    results = {}

    for role, model in models.items():
        estimate = float(model.predict(features)[0])
        results[role] = max(0.0, estimate)

    return results


predictions = get_predictions()
roles_to_show = [
    role for role in available_roles if role in predictions
]


# ==================================================
# 6. ILLUSTRATIVE OPERATING CONDITIONS
# ==================================================

resident_count = st.session_state["resident_count"]
facility_capacity = st.session_state["facility_capacity"]
weather = st.session_state["scenario_weather"]
visitor_activity = st.session_state["scenario_visitor_activity"]
appointment_activity = st.session_state["scenario_appointments"]

occupancy_rate = resident_count / facility_capacity
occupancy_percent = occupancy_rate * 100
remaining_capacity = facility_capacity - resident_count

# This is an invented baseline for the demonstration, not a
# measured visitor rate from the CMS staffing data.
baseline_visitors = resident_count * 0.30

estimated_visitors = round(
    baseline_visitors
    * VISITOR_ACTIVITY_FACTORS[visitor_activity]
    * WEATHER_FACTORS[weather]
)

coordination_index = round(
    100 * APPOINTMENT_FACTORS[appointment_activity]
)


# ==================================================
# 7. PLANNED HOURS AND COMPARISON DATA
# ==================================================

# Seed a clearly illustrative demo schedule on first run so the dashboard
# can demonstrate how role-specific differences are surfaced. These offsets
# are NOT staffing recommendations. Later manager edits are retained.
DEMO_PLAN_OFFSETS = {
    "Hrs_RN": -4,
    "Hrs_LPN": 2,
    "Hrs_CNA": -8,
    "Hrs_NAtrn": 1,
    "Hrs_MedAide": 0,
}

for role in roles_to_show:
    planned_key = f"planned_{role}"

    if planned_key not in st.session_state:
        demo_offset = DEMO_PLAN_OFFSETS.get(role, 0)
        st.session_state[planned_key] = max(
            0, int(round(predictions[role] + demo_offset))
        )

planned_hours = {
    role: float(st.session_state[f"planned_{role}"])
    for role in roles_to_show
}


def build_comparison():
    rows = []

    for role in roles_to_show:
        estimate = predictions[role]
        planned = planned_hours[role]
        difference = planned - estimate

        if difference < -0.5:
            status = "Below benchmark"
        elif difference > 0.5:
            status = "Above benchmark"
        else:
            status = "Near benchmark"

        rows.append(
            {
                "Role": ROLE_LABELS[role],
                "Historical estimate (hours)": round(estimate, 1),
                "Planned hours": round(planned, 1),
                "Planned − estimate": round(difference, 1),
                "Comparison": status,
            }
        )

    return pd.DataFrame(rows)


comparison = build_comparison()

total_estimated = comparison[
    "Historical estimate (hours)"
].sum()

total_planned = comparison["Planned hours"].sum()

total_difference = total_planned - total_estimated

below = comparison[
    comparison["Planned − estimate"] < -0.5
]

above = comparison[
    comparison["Planned − estimate"] > 0.5
]


# ==================================================
# 8. IRENE: RULE-BASED EXPLANATION
# ==================================================

def answer_irene(question):
    q = question.lower()

    if any(
        word in q
        for word in [
            "visitor", "visitors", "weather", "rain", "snow"
        ]
    ):
        return (
            f"This scenario estimates about {estimated_visitors} visitors. "
            f"The selected weather is {weather.lower()}, and visitor "
            f"activity is {visitor_activity.lower()}. The estimate is based "
            "on illustrative assumptions, not a live visitor feed or a "
            "validated forecasting model. Weather and visitors do not "
            "change the staffing benchmark."
        )

    if any(
        word in q
        for word in [
            "appointment", "appointments", "transport"
        ]
    ):
        return (
            f"Appointment activity is set to "
            f"{appointment_activity.lower()}. The relative coordination "
            f"index is {coordination_index}, with Typical set to 100. "
            "This is a scenario indicator, not an actual appointment count."
        )

    if any(
        word in q
        for word in [
            "occupancy", "population", "residents",
            "capacity", "census",
        ]
    ):
        return (
            f"The scenario contains {resident_count} residents against "
            f"an assumed capacity of {facility_capacity}, which is "
            f"{occupancy_percent:.1f}% occupancy. There are "
            f"{remaining_capacity} spaces remaining under this assumption. "
            "The capacity is manually entered; resident census is an input "
            "to the historical staffing models."
        )

    if any(
        word in q
        for word in [
            "gap", "difference", "short", "below",
            "surplus", "coverage",
        ]
    ):
        if below.empty:
            below_text = (
                "No displayed role is below its historical estimate."
            )
        else:
            below_text = "; ".join(
                f"{row['Role']} is "
                f"{abs(row['Planned − estimate']):.1f} hours below"
                for _, row in below.iterrows()
            )

        if above.empty:
            above_text = (
                "No displayed role is above its historical estimate."
            )
        else:
            above_text = "; ".join(
                f"{row['Role']} is "
                f"{row['Planned − estimate']:.1f} hours above"
                for _, row in above.iterrows()
            )

        return (
            f"Planned coverage is {total_difference:+.1f} hours compared "
            f"with the combined historical estimates. {below_text} "
            f"{above_text} These differences are prompts to review the "
            "schedule, not proof of understaffing or overstaffing."
        )

    if any(
        word in q
        for word in [
            "model", "estimate", "prediction", "ai", "data"
        ]
    ):
        return (
            "The machine-learning component uses Random Forest regression "
            "trained on historical daily staffing records. The inputs are "
            "state, resident census, weekday and month. A separate model "
            "is trained for each available staffing role. Weather, visitor "
            "activity and appointments are simulated separately and are "
            "not inputs to the staffing models."
        )

    if any(
        word in q
        for word in [
            "recommend", "should we", "safe",
            "safety", "clinical", "understaffed",
        ]
    ):
        return (
            "I can explain differences from historical patterns, but "
            "I cannot determine whether a schedule is clinically safe "
            "or appropriate. Consider resident acuity, required coverage, "
            "applicable regulations, staff qualifications and professional "
            "judgement before making staffing decisions."
        )

    return (
        "I can explain occupancy, the weather and visitor scenario, "
        "historical staffing estimates, planned hours and differences "
        "between them. Try asking: 'Which roles are below the benchmark?' "
        "or 'How does the weather affect the visitor estimate?'"
    )


# ==================================================
# 9. DESIGN-REFERENCE INTERFACE
# ==================================================

st.markdown("""<style>
[data-testid="stAppViewContainer"] {background:linear-gradient(145deg,#fbf7e9 0%,#c5e5df 46%,#6cc9e6 100%);color:#144a83;}
[data-testid="stSidebar"] {background:linear-gradient(175deg,#fdfcf6,#c6eaf1);}
[data-testid="stSidebar"] * {color:#225c90;}
.block-container {max-width:1250px;padding-top:1.2rem;}
h1,h2,h3,p,label {color:#164d86;}
div[data-testid="stMetric"] {background:#f9fcff;border-radius:18px;padding:18px 20px;min-height:104px;}
div[data-testid="stMetricLabel"] {font-size:.85rem;}
div[data-testid="stMetricValue"] {color:#1b568e;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#f8fcff;border:0!important;border-radius:19px!important;padding:7px;}
.stButton>button {border-radius:24px;color:#164d86;border:1px solid #c5d8e5;background:#fff;}
[data-testid="stChatMessage"] {background:#f8fcff;border-radius:18px;padding:18px;}
[data-testid="stTabs"] button {font-weight:600;color:#245c8e;}
[data-testid="stTabs"] [aria-selected="true"] {color:#10467d;border-bottom-color:#10467d;}
.lh-logo {font-size:2.1rem;letter-spacing:-1.5px;font-weight:300;color:#18528a;line-height:1.1;}
.lh-tag {font-size:.8rem;color:#396f9a;margin-bottom:1rem;}
.irene-title {font-size:clamp(5rem,13vw,9rem);text-align:center;letter-spacing:.06em;font-weight:600;color:#fff;text-shadow:0 8px 22px #5c9ab455;margin:1rem 0 0;}
.irene-sub {text-align:center;font-size:1.1rem;margin-bottom:1.6rem;color:#245b8d;}
.lh-note {background:#104b87;color:white;padding:20px 24px;border-radius:18px;margin-top:12px;}
.lh-note * {color:white!important;}
</style>""",unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="lh-logo">little helper</div><div class="lh-tag">Proof of concept</div>',unsafe_allow_html=True)
    st.selectbox("State",options=states,key="scenario_state")
    st.number_input("Residents present",min_value=0,max_value=5000,step=1,key="resident_count")
    st.number_input("Facility capacity",min_value=1,max_value=5000,step=1,key="facility_capacity")
    st.date_input("Scenario date",key="scenario_date")
    st.selectbox("Weather location", options=list(DEMO_CITIES), index=4, key="weather_city")
    st.toggle("Use live weather", key="use_live_weather")
    if not st.session_state["use_live_weather"]:
        st.selectbox("Simulated weather",options=WEATHER_OPTIONS,key="scenario_weather")
    st.select_slider("Visitor activity",options=["Quiet","Typical","Busy"],key="scenario_visitor_activity")
    st.select_slider("Appointment activity",options=["Low","Typical","High"],key="scenario_appointments")
    st.caption("Weather is live when enabled; population, visitors and appointments are simulated.")

# Recompute after widgets to reflect the currently selected scenario.
predictions=get_predictions()
resident_count=st.session_state["resident_count"]
facility_capacity=st.session_state["facility_capacity"]
weather=st.session_state["scenario_weather"]
weather_info=None
weather_error=False
if st.session_state["use_live_weather"]:
    try:
        weather_info=fetch_live_weather(st.session_state["weather_city"])
        weather=weather_info["category"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        weather_error=True
        st.sidebar.warning("Weather service unavailable; using simulated conditions.")
visitor_activity=st.session_state["scenario_visitor_activity"]
appointment_activity=st.session_state["scenario_appointments"]
occupancy_percent=100*resident_count/facility_capacity
remaining_capacity=facility_capacity-resident_count
estimated_visitors=round(resident_count*.30*VISITOR_ACTIVITY_FACTORS[visitor_activity]*WEATHER_FACTORS[weather])

for role in roles_to_show:
    if f"planned_{role}" not in st.session_state:
        st.session_state[f"planned_{role}"]=max(0,int(round(predictions[role]+DEMO_PLAN_OFFSETS.get(role,0))))

tab_dashboard,tab_irene=st.tabs(["Staff dashboard","Ask IRENE"])

with tab_dashboard:
    st.markdown("## Today's operating picture")
    a,b,c,d=st.columns(4)
    a.metric("Occupancy",f"{occupancy_percent:.0f}%")
    b.metric("Remaining spaces",f"{remaining_capacity:,}")
    c.metric("Illustrative visitors",f"~{estimated_visitors}")
    if weather_info:
        d.metric("Weather (Open-Meteo)", f"{weather_info['temperature']:.0f}°C", weather_info["label"], delta_color="off")
        st.caption(f"Weather: {st.session_state['weather_city']} · Open-Meteo model-based current conditions as of {weather_info['observed_at']} local time · wind {weather_info['wind']:.0f} km/h · precipitation {weather_info['precipitation']:.1f} mm. [Source: Open-Meteo](https://open-meteo.com/)")
    else:
        weather_temperature={"Clear":"21°C","Rain":"14°C","Snow / ice":"−2°C"}[weather]
        d.metric("Weather (simulated)",f"{weather_temperature} {weather}")

    with st.container(border=True):
        st.markdown("### Staffing: planned vs forecast")
        st.caption("Forecast means a modelled historical staffing-hours benchmark, not a clinical staffing requirement.")
        rows=[]
        for role in roles_to_show:
            planned=float(st.session_state[f"planned_{role}"])
            forecast=float(predictions[role])
            gap=planned-forecast
            status="Below benchmark" if gap < -.5 else "On track" if gap <= .5 else "Above benchmark"
            rows.append({"Role":ROLE_LABELS[role],"Planned hours":round(planned,1),"Forecast hours":round(forecast,1),"Status":status,"Gap":gap})
        st.dataframe(pd.DataFrame(rows).drop(columns="Gap"),hide_index=True,use_container_width=True)

    below_rows=[r for r in rows if r["Gap"]<-.5]
    if below_rows:
        biggest=min(below_rows,key=lambda r:r["Gap"])
        suggestion=f"Review {biggest['Role']} coverage: planned hours are {abs(biggest['Gap']):.1f} below the historical estimate."
    else:
        suggestion="No roles are currently below the historical benchmark; review coverage against actual resident needs."
    st.markdown(f'<div class="lh-note"><strong>ⓘ &nbsp;Suggestion:</strong> {suggestion}<div style="font-size:.82rem;margin-top:6px;">IRENE supports staffing decisions. It does not replace professional judgment.</div></div>',unsafe_allow_html=True)

    with st.expander("Edit planned coverage"):
        st.caption("These initial hours are illustrative, not a real care-home schedule.")
        for role in roles_to_show:
            st.number_input(ROLE_LABELS[role],min_value=0,max_value=100000,step=1,key=f"planned_{role}")
    with st.expander("Population and daily conditions"):
        occupied=min(resident_count,facility_capacity)
        donut=go.Figure(go.Pie(labels=["Occupied","Available"],values=[occupied,max(0,facility_capacity-occupied)],hole=.72,marker_colors=["#13538e","#d8eaf1"],sort=False))
        donut.update_layout(height=270,margin=dict(l=0,r=0,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",showlegend=True)
        st.plotly_chart(donut,use_container_width=True,config={"displayModeBar":False})
        st.caption("Visitor counts are illustrative. When live weather is enabled, its condition category is used in a hypothetical visitor adjustment; no weather-to-visitors relationship has been validated.")
    with st.expander("Data and methodology"):
        st.write(f"Demo sample: {len(data):,} public U.S. nursing-home daily records, {data['STATE'].nunique()} state/territory codes.")
        st.write(f"Historical dates: {data['WorkDate'].min():%Y-%m-%d} to {data['WorkDate'].max():%Y-%m-%d}.")
        st.write("Random forest models estimate daily hours by staffing role using state, census, weekday and month. The source data is historical and does not provide live staffing or resident acuity. Live weather does not change staffing model predictions; it only changes the illustrative visitor scenario.")

with tab_irene:
    st.markdown('<div class="irene-title">IRENE</div><div class="irene-sub">Your friendly helper for the day.</div>',unsafe_allow_html=True)
    if "irene_messages" not in st.session_state:
        st.session_state["irene_messages"]=[{"role":"assistant","content":"Hi! What would you like to know about today?"}]
    with st.container(border=True):
        for message in st.session_state["irene_messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        st.caption("Try asking")
        suggestions=["Which roles are below the benchmark?","How does the weather affect visitors?","How is the staffing estimate calculated?"]
        cols=st.columns(2)
        for i,q in enumerate(suggestions):
            if cols[i%2].button(q,key=f"suggest_{i}"):
                st.session_state["irene_messages"].extend([{"role":"user","content":q},{"role":"assistant","content":answer_irene(q)}])
                st.rerun()
    q=st.chat_input("Type your question...")
    if q:
        st.session_state["irene_messages"].extend([{"role":"user","content":q},{"role":"assistant","content":answer_irene(q)}])
        st.rerun()
    st.caption("IRENE is a rule-based demonstration assistant, not a live clinical or facility-information service. Activities, visitors' identities and meals are not available in this dataset.")
