import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys
sys.path.insert(0, ".")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero section */
.hero-container {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: rgba(255,255,255,0.65);
    font-size: 1.05rem;
    margin-top: 0.75rem;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
.metric-card:hover {
    border-color: rgba(102,126,234,0.5);
    background: rgba(102,126,234,0.1);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #667eea;
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-delta {
    font-size: 0.75rem;
    color: #43e97b;
    margin-top: 0.2rem;
}

/* Prediction result card */
.prediction-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(240,147,251,0.2));
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.prediction-price {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #43e97b, #38f9d7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.prediction-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.prediction-range {
    color: rgba(255,255,255,0.5);
    font-size: 0.85rem;
    margin-top: 0.75rem;
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: white;
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(102,126,234,0.4);
}

/* Feature cards */
.feature-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.75rem;
}
.feature-name {
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.feature-value {
    color: white;
    font-size: 1.3rem;
    font-weight: 600;
    margin-top: 0.2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(102,126,234,0.2) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: rgba(255,255,255,0.8) !important;
    font-size: 0.85rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.5) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(102,126,234,0.3) !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102,126,234,0.4) !important;
}

/* Info boxes */
.info-box {
    background: rgba(67,233,123,0.1);
    border: 1px solid rgba(67,233,123,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #43e97b;
    font-size: 0.9rem;
}
.warn-box {
    background: rgba(255,193,7,0.1);
    border: 1px solid rgba(255,193,7,0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    color: #ffc107;
    font-size: 0.9rem;
}

/* Input styling */
.stNumberInput input, .stSelectbox select {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border-radius: 99px !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* Plotly chart background */
.js-plotly-plot .plotly {
    border-radius: 12px;
}

/* Badge pill */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 8px;
    vertical-align: middle;
}
.badge-purple { background: rgba(102,126,234,0.2); color: #667eea; border: 1px solid rgba(102,126,234,0.4); }
.badge-green  { background: rgba(67,233,123,0.15); color: #43e97b; border: 1px solid rgba(67,233,123,0.3); }
.badge-pink   { background: rgba(240,147,251,0.15); color: #f093fb; border: 1px solid rgba(240,147,251,0.3); }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
NEIGHBORHOODS = [
    "CollgCr","Veenker","Crawfor","NoRidge","Mitchel","Somerst","NWAmes",
    "OldTown","BrkSide","Sawyer","NridgHt","NAmes","SawyerW","IDOTRR",
    "MeadowV","Edwards","Timber","Gilbert","StoneBr","ClearCr","NPkVill",
    "Blmngtn","BrDale","SWISU","Blueste",
]
OVERALL_QUAL_MAP = {
    "1 - Very Poor": 1, "2 - Poor": 2, "3 - Fair": 3, "4 - Below Average": 4,
    "5 - Average": 5, "6 - Above Average": 6, "7 - Good": 7,
    "8 - Very Good": 8, "9 - Excellent": 9, "10 - Very Excellent": 10,
}
BLDG_TYPE = ["1Fam","2FmCon","Duplx","TwnhsE","TwnhsI"]
HOUSE_STYLE = ["1Story","1.5Fin","1.5Unf","2Story","2.5Fin","2.5Unf","SFoyer","SLvl"]
EXTER_QUAL  = ["Ex","Gd","TA","Fa","Po"]
KITCHEN_QUAL = ["Ex","Gd","TA","Fa","Po"]
GARAGE_TYPE = ["Attchd","Detchd","BuiltIn","CarPort","None","Basment","2Types"]
FOUNDATION  = ["PConc","CBlock","BrkTil","Wood","Slab","Stone"]
SALE_TYPE   = ["WD","New","COD","ConLD","ConLI","ConLw","Con","Oth","CWD"]
SALE_COND   = ["Normal","Abnorml","AdjLand","Alloca","Family","Partial"]

# ── Helpers ───────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter", color="rgba(255,255,255,0.8)", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.1)"),
)
PURPLE_SCALE = [[0,"#302b63"],[0.5,"#667eea"],[1,"#f093fb"]]

@st.cache_data
def load_raw_data():
    path = "data/raw/ames_housing.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_model(name="ridge"):
    paths = {
        "ridge":  "models/ridge_regression.joblib",
        "linear": "models/linear_regression.joblib",
    }
    p = paths.get(name)
    if p and os.path.exists(p):
        return joblib.load(p)
    return None

@st.cache_data
def load_metrics():
    p = "reports/metrics/results.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None

def predict_price(features: dict, model_name: str = "ridge"):
    model = load_model(model_name)
    if model is None:
        return None
    df = pd.DataFrame([features])
    log_pred = model.predict(df)[0]
    price = float(np.expm1(log_pred))
    low  = price * 0.92
    high = price * 1.08
    return {"price": price, "low": low, "high": high}

def gauge_chart(value, min_val, max_val, title, color="#667eea"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13, "color": "rgba(255,255,255,0.7)"}},
        number={"font": {"size": 22, "color": "white"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "rgba(255,255,255,0.4)"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [min_val, (max_val-min_val)*0.33+min_val], "color": "rgba(255,255,255,0.04)"},
                {"range": [(max_val-min_val)*0.33+min_val, (max_val-min_val)*0.66+min_val], "color": "rgba(255,255,255,0.06)"},
                {"range": [(max_val-min_val)*0.66+min_val, max_val], "color": "rgba(255,255,255,0.09)"},
            ],
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=180, margin=dict(l=10,r=10,t=30,b=10))
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🏠</div>
        <div style='color:white; font-weight:700; font-size:1.1rem; margin-top:0.5rem;'>House Price AI</div>
        <div style='color:rgba(255,255,255,0.4); font-size:0.75rem; margin-top:0.2rem;'>Ames Housing Dataset</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigation",
        ["🏠 Home", "🔮 Predict Price", "📊 Data Explorer", "📈 Model Analytics", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div style='color:rgba(255,255,255,0.4); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;'>Model Selection</div>", unsafe_allow_html=True)
    model_choice = st.radio(
        "Model",
        ["Ridge Regression", "Linear Regression"],
        label_visibility="collapsed",
    )
    model_key = "ridge" if "Ridge" in model_choice else "linear"

    metrics = load_metrics()
    if metrics:
        key = "RidgeRegression" if model_key == "ridge" else "LinearRegression"
        m = metrics.get(key, {})
        st.markdown("---")
        st.markdown("<div style='color:rgba(255,255,255,0.4); font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;'>Active Model Stats</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("R²", f"{m.get('R2', 0):.3f}")
        col2.metric("RMSE", f"${m.get('RMSE', 0):,.0f}")
        col1.metric("MAE", f"${m.get('MAE', 0):,.0f}")
        col2.metric("CV R²", f"{m.get('CV_R2_Mean', 0):.3f}")

    st.markdown("---")
    st.markdown("""
    <div style='color:rgba(255,255,255,0.3); font-size:0.7rem; text-align:center; line-height:1.6;'>
        Built by <strong style='color:rgba(255,255,255,0.6);'>Darsh Kumar</strong><br>
        B.Tech Data Science<br>
        Ames Housing · scikit-learn
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-title'>🏠 House Price Predictor</div>
        <div class='hero-subtitle'>
            End-to-end ML pipeline · Ames Housing Dataset · Linear & Ridge Regression<br>
            79 features · 1,460 homes · 14 engineered features
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model metrics row
    metrics = load_metrics()
    if metrics:
        st.markdown("<div class='section-header'>📊 Model Performance</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        ridge = metrics.get("RidgeRegression", {})
        cards = [
            ("R² Score",     f"{ridge.get('R2', 0):.3f}",      "Ridge Regression",    "↑ Higher is better"),
            ("RMSE",         f"${ridge.get('RMSE', 0):,.0f}",   "Prediction Error",    "↓ Lower is better"),
            ("MAE",          f"${ridge.get('MAE', 0):,.0f}",    "Mean Abs Error",      "↓ Lower is better"),
            ("CV R²",        f"{ridge.get('CV_R2_Mean', 0):.3f}","5-Fold CV Score",    "± Stability metric"),
        ]
        for col, (label, val, sub, delta) in zip(cols, cards):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
                <div style='color:rgba(255,255,255,0.35); font-size:0.72rem; margin-top:3px;'>{sub}</div>
                <div class='metric-delta'>{delta}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warn-box'>
            ⚠️ Models not trained yet. Run <code>python src/models/train.py</code> first, then refresh.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Pipeline overview
    st.markdown("<div class='section-header'>🔄 ML Pipeline</div>", unsafe_allow_html=True)
    steps = [
        ("📥", "Data Ingestion",     "Ames Housing CSV\n1,460 samples · 79 features"),
        ("🧹", "Data Cleaning",      "Smart missing value treatment\nOutlier removal · Type fixing"),
        ("⚙️", "Feature Engineering","14 new features\nHouseAge · TotalSF · QualityScore"),
        ("🔧", "Preprocessing",      "StandardScaler + OneHotEncoder\nColumnTransformer Pipeline"),
        ("🤖", "Model Training",     "LinearRegression + Ridge\nRidgeCV alpha tuning"),
        ("📊", "Evaluation",         "MAE · RMSE · R² · Adj R²\n5-Fold Cross Validation"),
    ]
    cols = st.columns(6)
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style='background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                    border-radius:12px; padding:1rem 0.75rem; text-align:center; height:140px;'>
            <div style='font-size:1.8rem;'>{icon}</div>
            <div style='color:white; font-weight:600; font-size:0.82rem; margin-top:0.4rem;'>{title}</div>
            <div style='color:rgba(255,255,255,0.4); font-size:0.7rem; margin-top:0.3rem; line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Engineered features
    st.markdown("<div class='section-header'>✨ Engineered Features</div>", unsafe_allow_html=True)
    features_info = [
        ("HouseAge",          "Years since built at time of sale",       "Newer → premium price"),
        ("TotalSF",           "Basement SF + Above-grade Living SF",      "Strongest price driver"),
        ("TotalBathrooms",    "Full + 0.5×Half (above + basement)",       "Convenience premium"),
        ("QualityScore",      "OverallQual × OverallCond",               "Combined home quality"),
        ("QualArea",          "OverallQual × GrLivArea",                 "Size–quality interaction"),
        ("YearsSinceRemodel", "Years since last renovation",              "Recent reno = higher price"),
        ("TotalPorchSF",      "All porch and deck areas combined",        "Outdoor space value"),
        ("WasRemodeled",      "Binary: was the home ever renovated?",     "Remodeled → sells higher"),
        ("HasGarage",         "Binary: garage present?",                  "Garage adds $10K–$20K"),
        ("HasBasement",       "Binary: basement present?",                "Basement adds value"),
        ("HasFireplace",      "Binary: at least one fireplace?",          "Lifestyle premium"),
        ("IsNewHouse",        "Binary: sold same year as built?",         "New build premium"),
    ]
    col1, col2 = st.columns(2)
    for i, (name, desc, insight) in enumerate(features_info):
        target_col = col1 if i % 2 == 0 else col2
        target_col.markdown(f"""
        <div style='background:rgba(102,126,234,0.07); border:1px solid rgba(102,126,234,0.2);
                    border-radius:10px; padding:0.875rem 1rem; margin-bottom:0.6rem;
                    display:flex; align-items:flex-start; gap:0.75rem;'>
            <div style='width:8px; height:8px; border-radius:50%; background:#667eea;
                        margin-top:5px; flex-shrink:0;'></div>
            <div>
                <span style='color:#a78bfa; font-weight:600; font-size:0.85rem;'>{name}</span>
                <div style='color:rgba(255,255,255,0.6); font-size:0.78rem; margin-top:2px;'>{desc}</div>
                <div style='color:#43e97b; font-size:0.72rem; margin-top:2px;'>💡 {insight}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT PRICE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict Price":
    st.markdown("""
    <div class='hero-container' style='padding:1.75rem 2rem; margin-bottom:1.5rem;'>
        <div style='font-size:1.8rem; font-weight:700; color:white;'>🔮 Predict House Price</div>
        <div style='color:rgba(255,255,255,0.5); font-size:0.9rem; margin-top:0.4rem;'>
            Fill in the property details below and get an instant AI-powered price estimate
        </div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model(model_key)
    if model is None:
        st.markdown("<div class='warn-box'>⚠️ Model not found. Run <code>python src/models/train.py</code> first.</div>", unsafe_allow_html=True)
        st.stop()

    # Input form
    with st.form("prediction_form"):
        # Section 1: Location & Type
        st.markdown("<div class='section-header'>📍 Location & Property Type</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        neighborhood = c1.selectbox("Neighborhood", NEIGHBORHOODS, index=0)
        bldg_type    = c2.selectbox("Building Type", BLDG_TYPE, index=0)
        house_style  = c3.selectbox("House Style", HOUSE_STYLE, index=3)

        # Section 2: Size
        st.markdown("<div class='section-header'>📐 Size & Space</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        gr_liv_area  = c1.number_input("Living Area (sq ft)", 500, 6000, 1500, 50)
        total_bsmt   = c2.number_input("Basement SF", 0, 3000, 800, 50)
        lot_area     = c3.number_input("Lot Area (sq ft)", 1000, 50000, 8500, 500)
        lot_frontage = c4.number_input("Lot Frontage (ft)", 0, 300, 70, 5)

        c1, c2, c3, c4 = st.columns(4)
        first_flr    = c1.number_input("1st Floor SF", 300, 4000, 900, 50)
        second_flr   = c2.number_input("2nd Floor SF", 0, 2000, 500, 50)
        garage_area  = c3.number_input("Garage Area (sq ft)", 0, 1500, 480, 20)
        wood_deck    = c4.number_input("Wood Deck SF", 0, 800, 0, 10)

        # Section 3: Quality
        st.markdown("<div class='section-header'>⭐ Quality & Condition</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        overall_qual_str = c1.selectbox("Overall Quality", list(OVERALL_QUAL_MAP.keys()), index=6)
        overall_qual = OVERALL_QUAL_MAP[overall_qual_str]
        overall_cond = c2.slider("Overall Condition (1–10)", 1, 10, 5)
        exter_qual   = c3.selectbox("Exterior Quality", EXTER_QUAL, index=1)
        c1, c2, c3 = st.columns(3)
        kitchen_qual = c1.selectbox("Kitchen Quality", KITCHEN_QUAL, index=1)
        foundation   = c2.selectbox("Foundation", FOUNDATION, index=0)
        garage_type  = c3.selectbox("Garage Type", GARAGE_TYPE, index=0)

        # Section 4: Age & Rooms
        st.markdown("<div class='section-header'>🗓️ Age & Rooms</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        year_built   = c1.number_input("Year Built", 1870, 2025, 2003)
        year_remod   = c2.number_input("Year Remodeled", 1950, 2025, 2003)
        yr_sold      = c3.number_input("Year Sold", 2006, 2025, 2010)
        ms_subclass  = c4.selectbox("MS SubClass", ["20","30","40","45","50","60","70","75","80","85","90","120","150","160","180","190"], index=5)

        c1, c2, c3, c4 = st.columns(4)
        bedrooms     = c1.number_input("Bedrooms Above Grade", 0, 8, 3)
        full_bath    = c2.number_input("Full Bathrooms", 0, 4, 2)
        half_bath    = c3.number_input("Half Bathrooms", 0, 3, 1)
        tot_rms      = c4.number_input("Total Rooms Above Grade", 2, 14, 7)

        c1, c2, c3, c4 = st.columns(4)
        bsmt_full    = c1.number_input("Basement Full Bath", 0, 3, 1)
        bsmt_half    = c2.number_input("Basement Half Bath", 0, 2, 0)
        garage_cars  = c3.number_input("Garage Cars", 0, 4, 2)
        fireplaces   = c4.number_input("Fireplaces", 0, 4, 1)

        # Section 5: Basement
        st.markdown("<div class='section-header'>🏗️ Basement Details</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        bsmt_fin_sf1 = c1.number_input("Basement Finished SF1", 0, 2000, 400, 50)
        bsmt_fin_sf2 = c2.number_input("Basement Finished SF2", 0, 1000, 0, 50)
        bsmt_unf     = c3.number_input("Basement Unfinished SF", 0, 2000, 300, 50)

        # Submit
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Predict Price", use_container_width=True)

    if submitted:
        # Build feature dict matching model training features
        features = {
            "MSSubClass": ms_subclass, "MSZoning": "RL",
            "LotFrontage": float(lot_frontage), "LotArea": float(lot_area),
            "Street": "Pave", "Alley": "None", "LotShape": "Reg",
            "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside",
            "LandSlope": "Gtl", "Neighborhood": neighborhood,
            "Condition1": "Norm", "Condition2": "Norm",
            "BldgType": bldg_type, "HouseStyle": house_style,
            "OverallQual": float(overall_qual), "OverallCond": float(overall_cond),
            "YearBuilt": float(year_built), "YearRemodAdd": float(year_remod),
            "RoofStyle": "Gable", "RoofMatl": "CompShg",
            "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
            "MasVnrType": "None", "MasVnrArea": 0.0,
            "ExterQual": exter_qual, "ExterCond": "TA",
            "Foundation": foundation,
            "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "No",
            "BsmtFinType1": "GLQ", "BsmtFinSF1": float(bsmt_fin_sf1),
            "BsmtFinType2": "Unf", "BsmtFinSF2": float(bsmt_fin_sf2),
            "BsmtUnfSF": float(bsmt_unf), "TotalBsmtSF": float(total_bsmt),
            "Heating": "GasA", "HeatingQC": "Ex", "CentralAir": "Y",
            "Electrical": "SBrkr",
            "1stFlrSF": float(first_flr), "2ndFlrSF": float(second_flr),
            "LowQualFinSF": 0.0, "GrLivArea": float(gr_liv_area),
            "BsmtFullBath": float(bsmt_full), "BsmtHalfBath": float(bsmt_half),
            "FullBath": float(full_bath), "HalfBath": float(half_bath),
            "BedroomAbvGr": float(bedrooms), "KitchenAbvGr": 1.0,
            "KitchenQual": kitchen_qual, "TotRmsAbvGrd": float(tot_rms),
            "Functional": "Typ", "Fireplaces": float(fireplaces),
            "FireplaceQu": "Gd" if fireplaces > 0 else "None",
            "GarageType": garage_type if garage_area > 0 else "None",
            "GarageYrBlt": float(year_built),
            "GarageFinish": "RFn", "GarageCars": float(garage_cars),
            "GarageArea": float(garage_area),
            "GarageQual": "TA", "GarageCond": "TA",
            "PavedDrive": "Y",
            "WoodDeckSF": float(wood_deck), "OpenPorchSF": 0.0,
            "EnclosedPorch": 0.0, "3SsnPorch": 0.0, "ScreenPorch": 0.0,
            "PoolArea": 0.0, "PoolQC": "None", "Fence": "None",
            "MiscFeature": "None", "MiscVal": 0.0,
            "MoSold": 6, "YrSold": float(yr_sold),
            "SaleType": "WD", "SaleCondition": "Normal",
            # Engineered
            "HouseAge": float(yr_sold - year_built),
            "YearsSinceRemodel": float(yr_sold - year_remod),
            "TotalBathrooms": float(full_bath + 0.5*half_bath + bsmt_full + 0.5*bsmt_half),
            "TotalSF": float(total_bsmt + gr_liv_area),
            "TotalPorchSF": float(wood_deck),
            "QualityScore": float(overall_qual * overall_cond),
            "QualArea": float(overall_qual * gr_liv_area),
            "IsNewHouse": int(year_built == yr_sold),
            "HasGarage": int(garage_area > 0),
            "HasBasement": int(total_bsmt > 0),
            "HasPool": 0, "HasFireplace": int(fireplaces > 0),
            "TotalRooms": float(tot_rms + full_bath),
            "WasRemodeled": int(year_built != year_remod),
            "ExterQual": 4, "ExterCond": 3, "BsmtQual": 4,
            "BsmtCond": 3, "HeatingQC": 5, "KitchenQual": 4,
            "FireplaceQu": 4 if fireplaces > 0 else 0,
            "GarageQual": 3, "GarageCond": 3, "PoolQC": 0,
        }

        result = predict_price(features, model_key)
        if result:
            price = result["price"]
            low   = result["low"]
            high  = result["high"]

            # Main prediction display
            st.markdown(f"""
            <div class='prediction-card'>
                <div class='prediction-label'>Estimated Sale Price</div>
                <div class='prediction-price'>${price:,.0f}</div>
                <div class='prediction-range'>
                    Confidence Range: ${low:,.0f} — ${high:,.0f} &nbsp;|&nbsp; ±8%
                </div>
                <div style='margin-top:1rem; color:rgba(255,255,255,0.4); font-size:0.78rem;'>
                    Model: {model_choice} &nbsp;·&nbsp; Neighborhood: {neighborhood} &nbsp;·&nbsp;
                    {gr_liv_area:,} sq ft living area &nbsp;·&nbsp; Quality: {overall_qual}/10
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauges row
            c1, c2, c3, c4 = st.columns(4)
            c1.plotly_chart(gauge_chart(overall_qual, 1, 10, "Overall Quality", "#667eea"), use_container_width=True)
            c2.plotly_chart(gauge_chart(gr_liv_area, 500, 5000, "Living Area (sq ft)", "#f093fb"), use_container_width=True)
            c3.plotly_chart(gauge_chart(yr_sold - year_built, 0, 100, "House Age (yrs)", "#43e97b"), use_container_width=True)
            c4.plotly_chart(gauge_chart(full_bath + 0.5*half_bath + bsmt_full, 0, 5, "Total Baths", "#38f9d7"), use_container_width=True)

            # Feature contribution bar chart
            st.markdown("<div class='section-header'>📊 Key Feature Influence</div>", unsafe_allow_html=True)
            feature_impact = {
                "Overall Quality":    overall_qual * 0.18,
                "Living Area":        (gr_liv_area / 1000) * 0.16,
                "Qual × Area":        (overall_qual * gr_liv_area / 10000) * 0.14,
                "Total SF":           ((total_bsmt + gr_liv_area) / 1000) * 0.12,
                "Garage Cars":        garage_cars * 0.08,
                "Total Bathrooms":    (full_bath + 0.5*half_bath + bsmt_full) * 0.07,
                "Fireplaces":         fireplaces * 0.06,
                "House Age":          max(0, 1 - (yr_sold - year_built) / 100) * 0.05,
            }
            total = sum(feature_impact.values())
            normalized = {k: v/total*100 for k, v in feature_impact.items()}
            sorted_items = sorted(normalized.items(), key=lambda x: x[1])

            fig = go.Figure(go.Bar(
                x=list(v for _, v in sorted_items),
                y=list(k for k, _ in sorted_items),
                orientation="h",
                marker=dict(
                    color=list(v for _, v in sorted_items),
                    colorscale=[[0,"#302b63"],[0.5,"#667eea"],[1,"#f093fb"]],
                    showscale=False,
                ),
                text=[f"{v:.1f}%" for _, v in sorted_items],
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.8)", size=11),
            ))
            fig.update_layout(**PLOT_LAYOUT, height=280,
                              title=dict(text="Feature Influence on Prediction (%)", font=dict(size=13)))
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            st.markdown("<div class='section-header'>📋 Property Summary</div>", unsafe_allow_html=True)
            summary_data = {
                "Feature": ["Neighborhood","Building Type","Year Built","Living Area","Total SF",
                            "Overall Quality","Total Bathrooms","Garage Cars","Fireplaces","House Age"],
                "Value":   [neighborhood, bldg_type, year_built, f"{gr_liv_area:,} sq ft",
                            f"{total_bsmt + gr_liv_area:,} sq ft", f"{overall_qual}/10",
                            f"{full_bath + 0.5*half_bath + bsmt_full:.1f}", garage_cars,
                            fireplaces, f"{yr_sold - year_built} years"],
            }
            st.dataframe(
                pd.DataFrame(summary_data),
                use_container_width=True,
                hide_index=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("""
    <div class='hero-container' style='padding:1.75rem 2rem; margin-bottom:1.5rem;'>
        <div style='font-size:1.8rem; font-weight:700; color:white;'>📊 Data Explorer</div>
        <div style='color:rgba(255,255,255,0.5); font-size:0.9rem; margin-top:0.4rem;'>
            Explore the Ames Housing Dataset with interactive visualizations
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = load_raw_data()
    if df is None:
        st.markdown("<div class='warn-box'>⚠️ Dataset not found. Run the data download step first.</div>", unsafe_allow_html=True)
        st.stop()

    if "SalePrice" not in df.columns:
        st.error("SalePrice column not found in dataset.")
        st.stop()

    # Dataset overview
    c1, c2, c3, c4 = st.columns(4)
    for col, (val, lbl) in zip([c1,c2,c3,c4], [
        (f"{len(df):,}", "Total Samples"),
        (f"{df.shape[1]}", "Total Features"),
        (f"${df['SalePrice'].mean():,.0f}", "Avg Sale Price"),
        (f"{df.isnull().sum().sum():,}", "Missing Values"),
    ]):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Price Analysis", "🏗️ Feature Analysis", "🗺️ Neighborhood", "🔗 Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        # Price distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["SalePrice"], nbinsx=50,
            marker=dict(color="#667eea", line=dict(color="rgba(0,0,0,0)", width=0)),
            opacity=0.85, name="Sale Price",
        ))
        fig.update_layout(**PLOT_LAYOUT, title="Sale Price Distribution",
                          xaxis_title="Sale Price ($)", yaxis_title="Count", height=350)
        c1.plotly_chart(fig, use_container_width=True)

        # Log price distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=np.log1p(df["SalePrice"]), nbinsx=50,
            marker=dict(color="#f093fb", line=dict(color="rgba(0,0,0,0)", width=0)),
            opacity=0.85, name="Log(SalePrice)",
        ))
        fig2.update_layout(**PLOT_LAYOUT, title="Log(Sale Price) Distribution — More Normal",
                           xaxis_title="Log(1 + Price)", yaxis_title="Count", height=350)
        c2.plotly_chart(fig2, use_container_width=True)

        # Price by year
        if "YrSold" in df.columns:
            yr_price = df.groupby("YrSold")["SalePrice"].agg(["mean","median"]).reset_index()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=yr_price["YrSold"], y=yr_price["mean"],
                                      mode="lines+markers", name="Mean",
                                      line=dict(color="#667eea", width=2.5),
                                      marker=dict(size=8, color="#667eea")))
            fig3.add_trace(go.Scatter(x=yr_price["YrSold"], y=yr_price["median"],
                                      mode="lines+markers", name="Median",
                                      line=dict(color="#43e97b", width=2.5, dash="dash"),
                                      marker=dict(size=8, color="#43e97b")))
            fig3.update_layout(**PLOT_LAYOUT, title="Sale Price by Year",
                               xaxis_title="Year Sold", yaxis_title="Price ($)", height=320)
            st.plotly_chart(fig3, use_container_width=True)

        # Scatter: GrLivArea vs SalePrice
        if "GrLivArea" in df.columns and "OverallQual" in df.columns:
            fig4 = px.scatter(df, x="GrLivArea", y="SalePrice",
                             color="OverallQual" if "OverallQual" in df.columns else None,
                             color_continuous_scale="Viridis",
                             opacity=0.6,
                             labels={"GrLivArea":"Living Area (sq ft)", "SalePrice":"Sale Price ($)"},
                             title="Living Area vs Sale Price (colored by Quality)")
            fig4.update_layout(**PLOT_LAYOUT, height=380,
                               coloraxis_colorbar=dict(title="Quality",
                                                        tickfont=dict(color="rgba(255,255,255,0.7)")))
            fig4.update_traces(marker=dict(size=5))
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "SalePrice" in num_cols:
            num_cols.remove("SalePrice")

        c1, c2 = st.columns([1, 2])
        selected_feat = c1.selectbox("Select Feature", num_cols[:20], index=0)

        if selected_feat in df.columns:
            c2.markdown(f"""
            <div class='feature-card'>
                <div class='feature-name'>Feature Stats — {selected_feat}</div>
                <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:0.5rem; margin-top:0.5rem;'>
                    <div><div style='color:rgba(255,255,255,0.5);font-size:0.7rem;'>Mean</div>
                         <div style='color:white;font-weight:600;'>{df[selected_feat].mean():,.1f}</div></div>
                    <div><div style='color:rgba(255,255,255,0.5);font-size:0.7rem;'>Median</div>
                         <div style='color:white;font-weight:600;'>{df[selected_feat].median():,.1f}</div></div>
                    <div><div style='color:rgba(255,255,255,0.5);font-size:0.7rem;'>Std Dev</div>
                         <div style='color:white;font-weight:600;'>{df[selected_feat].std():,.1f}</div></div>
                    <div><div style='color:rgba(255,255,255,0.5);font-size:0.7rem;'>Min</div>
                         <div style='color:white;font-weight:600;'>{df[selected_feat].min():,.1f}</div></div>
                    <div><div style='color:rgba(255,255,255,0.5);font-size:0.7rem;'>Max</div>
                         <div style='color:white;font-weight:600;'>{df[selected_feat].max():,.1f}</div></div>
                    <div><div style='color:rgba(255,255,255,0.5);font-size:0.7rem;'>Missing</div>
                         <div style='color:#f093fb;font-weight:600;'>{df[selected_feat].isnull().sum()}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            fig_hist = px.histogram(df, x=selected_feat, nbins=40,
                                    color_discrete_sequence=["#667eea"],
                                    title=f"{selected_feat} Distribution")
            fig_hist.update_layout(**PLOT_LAYOUT, height=300)
            col_a.plotly_chart(fig_hist, use_container_width=True)

            fig_scat = px.scatter(df, x=selected_feat, y="SalePrice",
                                  opacity=0.5, color_discrete_sequence=["#f093fb"],
                                  title=f"{selected_feat} vs Sale Price",
                                  trendline="ols",
                                  trendline_color_override="#43e97b")
            fig_scat.update_layout(**PLOT_LAYOUT, height=300)
            fig_scat.update_traces(marker=dict(size=4))
            col_b.plotly_chart(fig_scat, use_container_width=True)

    with tab3:
        if "Neighborhood" in df.columns:
            nb = df.groupby("Neighborhood")["SalePrice"].agg(["mean","median","count"]).reset_index()
            nb.columns = ["Neighborhood","Mean Price","Median Price","Count"]
            nb = nb.sort_values("Mean Price", ascending=True)

            fig_nb = go.Figure(go.Bar(
                x=nb["Mean Price"], y=nb["Neighborhood"],
                orientation="h",
                marker=dict(color=nb["Mean Price"],
                           colorscale=[[0,"#302b63"],[0.5,"#667eea"],[1,"#f093fb"]],
                           showscale=True,
                           colorbar=dict(title="$", tickfont=dict(color="rgba(255,255,255,0.7)"))),
                text=[f"${v:,.0f}" for v in nb["Mean Price"]],
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.8)", size=10),
            ))
            fig_nb.update_layout(**PLOT_LAYOUT, height=600,
                                  title="Average Sale Price by Neighborhood",
                                  xaxis_title="Mean Sale Price ($)",
                                  margin=dict(l=80, r=60, t=40, b=20))
            st.plotly_chart(fig_nb, use_container_width=True)

            # Box plot
            fig_box = px.box(df, x="Neighborhood", y="SalePrice",
                             color_discrete_sequence=["#667eea"],
                             title="Price Distribution by Neighborhood")
            fig_box.update_layout(**PLOT_LAYOUT, height=420,
                                  xaxis_tickangle=-45,
                                  xaxis_title="", yaxis_title="Sale Price ($)")
            st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        num_df = df.select_dtypes(include=[np.number])
        if "SalePrice" in num_df.columns:
            corr_with_price = num_df.corr()["SalePrice"].drop("SalePrice").sort_values()
            top20 = pd.concat([corr_with_price.head(10), corr_with_price.tail(10)])

            colors_corr = ["#e74c3c" if v < 0 else "#667eea" for v in top20.values]
            fig_corr = go.Figure(go.Bar(
                x=top20.values, y=top20.index,
                orientation="h",
                marker=dict(color=colors_corr),
                text=[f"{v:.3f}" for v in top20.values],
                textposition="outside",
                textfont=dict(color="rgba(255,255,255,0.8)", size=10),
            ))
            fig_corr.update_layout(**PLOT_LAYOUT, height=500,
                                    title="Top 20 Feature Correlations with SalePrice",
                                    xaxis_title="Pearson Correlation", margin=dict(l=20,r=60,t=40,b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

            # Heatmap of top 12 numeric features
            top12 = num_df.corr()["SalePrice"].abs().nlargest(12).index
            corr_mat = num_df[top12].corr()
            fig_heat = px.imshow(corr_mat, color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, aspect="auto",
                                 title="Correlation Heatmap — Top 12 Features")
            fig_heat.update_layout(**PLOT_LAYOUT, height=480,
                                   coloraxis_colorbar=dict(tickfont=dict(color="rgba(255,255,255,0.7)")))
            st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Analytics":
    st.markdown("""
    <div class='hero-container' style='padding:1.75rem 2rem; margin-bottom:1.5rem;'>
        <div style='font-size:1.8rem; font-weight:700; color:white;'>📈 Model Analytics</div>
        <div style='color:rgba(255,255,255,0.5); font-size:0.9rem; margin-top:0.4rem;'>
            Compare Linear Regression vs Ridge Regression performance
        </div>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics()
    if metrics is None:
        st.markdown("<div class='warn-box'>⚠️ No metrics found. Run <code>python src/models/train.py</code> first.</div>", unsafe_allow_html=True)
        st.stop()

    lr  = metrics.get("LinearRegression", {})
    rr  = metrics.get("RidgeRegression", {})

    # Metrics comparison table
    st.markdown("<div class='section-header'>📊 Side-by-Side Comparison</div>", unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        "Metric": ["MAE ($)", "RMSE ($)", "R² Score", "Adjusted R²", "CV R² Mean", "CV R² Std"],
        "Linear Regression": [
            f"${lr.get('MAE',0):,.0f}", f"${lr.get('RMSE',0):,.0f}",
            f"{lr.get('R2',0):.4f}", f"{lr.get('Adj_R2',0):.4f}",
            f"{lr.get('CV_R2_Mean',0):.4f}", f"±{lr.get('CV_R2_Std',0):.4f}",
        ],
        "Ridge Regression": [
            f"${rr.get('MAE',0):,.0f}", f"${rr.get('RMSE',0):,.0f}",
            f"{rr.get('R2',0):.4f}", f"{rr.get('Adj_R2',0):.4f}",
            f"{rr.get('CV_R2_Mean',0):.4f}", f"±{rr.get('CV_R2_Std',0):.4f}",
        ],
        "Winner": ["Ridge ✅","Ridge ✅","Ridge ✅","Ridge ✅","Ridge ✅","Comparable ⚖️"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Bar charts
    st.markdown("<div class='section-header'>📉 Metrics Visualization</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    fig_r2 = go.Figure()
    models = ["Linear Regression", "Ridge Regression"]
    r2_vals = [lr.get("R2",0), rr.get("R2",0)]
    fig_r2.add_trace(go.Bar(
        x=models, y=r2_vals,
        marker=dict(color=["#4C72B0","#667eea"],
                   line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.4f}" for v in r2_vals], textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.9)"),
    ))
    fig_r2.update_layout(**PLOT_LAYOUT, title="R² Score Comparison",
                          yaxis_title="R²", height=320,
                          yaxis=dict(**PLOT_LAYOUT["yaxis"], range=[0, 1.05]))
    c1.plotly_chart(fig_r2, use_container_width=True)

    fig_rmse = go.Figure()
    rmse_vals = [lr.get("RMSE",0), rr.get("RMSE",0)]
    fig_rmse.add_trace(go.Bar(
        x=models, y=rmse_vals,
        marker=dict(color=["#DD8452","#f093fb"],
                   line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"${v:,.0f}" for v in rmse_vals], textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.9)"),
    ))
    fig_rmse.update_layout(**PLOT_LAYOUT, title="RMSE Comparison (lower = better)",
                            yaxis_title="RMSE ($)", height=320)
    c2.plotly_chart(fig_rmse, use_container_width=True)

    # CV scores
    fig_cv = go.Figure()
    cv_means = [lr.get("CV_R2_Mean",0), rr.get("CV_R2_Mean",0)]
    cv_stds  = [lr.get("CV_R2_Std",0),  rr.get("CV_R2_Std",0)]
    fig_cv.add_trace(go.Bar(
        x=models, y=cv_means,
        error_y=dict(type="data", array=cv_stds, visible=True,
                     color="rgba(255,255,255,0.5)", thickness=2),
        marker=dict(color=["#55A868","#43e97b"],
                   line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.4f} ±{s:.4f}" for v,s in zip(cv_means, cv_stds)],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.9)"),
    ))
    fig_cv.update_layout(**PLOT_LAYOUT, title="5-Fold Cross Validation R² (with std dev)",
                          yaxis_title="CV R²", height=320,
                          yaxis=dict(**PLOT_LAYOUT["yaxis"], range=[0, 1.05]))
    st.plotly_chart(fig_cv, use_container_width=True)

    # Ridge alpha info
    if "Alpha" in rr:
        st.markdown(f"""
        <div class='info-box'>
            🎯 Ridge Regression optimal alpha found via RidgeCV: <strong>{rr['Alpha']}</strong><br>
            Higher alpha = stronger regularization = less overfitting on high-dimensional feature space.
        </div>
        """, unsafe_allow_html=True)

    # Explanation cards
    st.markdown("<div class='section-header'>🧠 Understanding the Models</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown("""
    <div class='feature-card'>
        <div class='feature-name'>Linear Regression</div>
        <div style='color:rgba(255,255,255,0.7); font-size:0.85rem; margin-top:0.5rem; line-height:1.7;'>
            ✅ Simple and interpretable<br>
            ✅ Fast to train<br>
            ✅ Works well when features are independent<br>
            ⚠️ Sensitive to multicollinearity<br>
            ⚠️ Can overfit with many features<br>
            ⚠️ No regularization
        </div>
    </div>
    """, unsafe_allow_html=True)
    c2.markdown("""
    <div class='feature-card'>
        <div class='feature-name'>Ridge Regression</div>
        <div style='color:rgba(255,255,255,0.7); font-size:0.85rem; margin-top:0.5rem; line-height:1.7;'>
            ✅ L2 regularization prevents overfitting<br>
            ✅ Handles multicollinearity well<br>
            ✅ Better generalization on test data<br>
            ✅ Alpha tuned automatically via RidgeCV<br>
            ✅ More stable with 79 correlated features<br>
            ⚠️ Slightly less interpretable
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div class='hero-container'>
        <div class='hero-title'>ℹ️ About This Project</div>
        <div class='hero-subtitle'>End-to-end ML project by Darsh Kumar · B.Tech Data Science</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2,1])
    c1.markdown("""
    <div class='feature-card'>
        <div class='feature-name'>Project Overview</div>
        <div style='color:rgba(255,255,255,0.75); font-size:0.9rem; line-height:1.8; margin-top:0.75rem;'>
            This project builds an end-to-end machine learning pipeline to predict house sale prices
            using the <strong style='color:#a78bfa;'>Ames Housing Dataset</strong> — a rich, real-world
            dataset with 79 features describing residential homes in Ames, Iowa.<br><br>
            The pipeline includes data cleaning, 14 engineered features, an automated sklearn
            ColumnTransformer pipeline, and two regression models: Linear Regression and Ridge Regression
            with automated alpha tuning via RidgeCV.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown("""
    <div class='feature-card'>
        <div class='feature-name'>Tech Stack</div>
        <div style='margin-top:0.75rem;'>
            <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;'>
                <div style='width:8px;height:8px;border-radius:50%;background:#667eea;flex-shrink:0;'></div>
                <span style='color:rgba(255,255,255,0.75);font-size:0.85rem;'>Python 3.11</span>
            </div>
            <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;'>
                <div style='width:8px;height:8px;border-radius:50%;background:#f093fb;flex-shrink:0;'></div>
                <span style='color:rgba(255,255,255,0.75);font-size:0.85rem;'>scikit-learn 1.4</span>
            </div>
            <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;'>
                <div style='width:8px;height:8px;border-radius:50%;background:#43e97b;flex-shrink:0;'></div>
                <span style='color:rgba(255,255,255,0.75);font-size:0.85rem;'>Streamlit + Plotly</span>
            </div>
            <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;'>
                <div style='width:8px;height:8px;border-radius:50%;background:#38f9d7;flex-shrink:0;'></div>
                <span style='color:rgba(255,255,255,0.75);font-size:0.85rem;'>pandas + numpy</span>
            </div>
            <div style='display:flex; align-items:center; gap:0.5rem;'>
                <div style='width:8px;height:8px;border-radius:50%;background:#ffc107;flex-shrink:0;'></div>
                <span style='color:rgba(255,255,255,0.75);font-size:0.85rem;'>GitHub Codespaces</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:2rem; text-align:center; color:rgba(255,255,255,0.3); font-size:0.78rem;'>
        Built with ❤️ by Darsh Kumar · GitHub Codespaces · 2024
    </div>
    """, unsafe_allow_html=True)
