import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Aerisense | Air Intelligence", page_icon="🍃", layout="wide")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "air_quality_demo.csv")
MODEL_PATH = os.path.join(ROOT, "models", "aqi_random_forest.joblib")
FEATURES = ["temperature_c", "humidity_pct", "pm25", "pm10", "no2", "co", "so2", "o3"]
POLLUTANTS = ["pm25", "pm10", "no2", "co", "so2", "o3"]


st.markdown("""
<style>
/* ===== Aerisense Pastel Theme ===== */

:root {
    --page: #F5F7F2;
    --panel: #FFFFFF;
    --sidebar: #EAF1E9;
    --sage: #E4EFE7;
    --mint: #D3E7D8;
    --green: #174B3D;
    --text: #283B34;
    --muted: #68776F;
    --border: #DCE6DD;
}

/* Main application background */
html, body, [data-testid="stAppViewContainer"] {
    background: #F5F7F2 !important;
    color: #283B34 !important;
}

[data-testid="stHeader"] {
    background: rgba(245, 247, 242, 0.95) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #EAF1E9 !important;
    border-right: 1px solid #DCE6DD;
}

[data-testid="stSidebar"] * {
    color: #283B34;
}

/* Page spacing */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Dark green branding banner */
.hero {
    background: linear-gradient(120deg, #174B3D, #2C7059);
    padding: 26px 30px;
    border-radius: 20px;
    color: #FFFFFF;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(23, 75, 61, 0.10);
}

.hero h1 {
    color: #FFFFFF !important;
    margin: 0;
    font-size: 2.2rem;
}

.hero p {
    color: #E3F1E8 !important;
    margin: 8px 0 0 0;
}

/* Headings and text */
h1, h2, h3, h4, p, label {
    color: #283B34;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE6DD;
    border-radius: 15px;
    padding: 15px 18px;
    box-shadow: 0 4px 14px rgba(40, 59, 52, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #68776F !important;
}

[data-testid="stMetricValue"] {
    color: #174B3D !important;
}

/* Information and notification boxes */
[data-testid="stAlert"] {
    background: #E7F0E9;
    border: 1px solid #D1E2D5;
    color: #283B34;
    border-radius: 12px;
}

/* Dropdowns and input fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #FFFFFF !important;
    border-color: #C9D9CD !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: #174B3D !important;
    color: #FFFFFF !important;
    border: 1px solid #174B3D !important;
    border-radius: 10px !important;
    padding: 0.55rem 1rem;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #246650 !important;
    border-color: #246650 !important;
    color: #FFFFFF !important;
}

/* Data tables and separators */
[data-testid="stDataFrame"] {
    border: 1px solid #DCE6DD;
    border-radius: 12px;
}

hr {
    border-color: #DCE6DD;
}

.small-note {
    color: #68776F;
    font-size: 0.86rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_resource
def train_model():
    data = load_data()
    X = data[FEATURES]
    y = data["aqi"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=180, max_depth=14, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R²": float(r2_score(y_test, pred)),
        "test_rows": int(len(y_test))
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    return model, metrics, X_test, y_test, pred

def aqi_category(value):
    if value <= 50: return "Good", "Air quality is considered satisfactory."
    if value <= 100: return "Satisfactory", "Generally acceptable; unusually sensitive people should take care."
    if value <= 200: return "Moderate", "Sensitive people may experience discomfort."
    if value <= 300: return "Poor", "People with lung or heart conditions should reduce prolonged outdoor exertion."
    if value <= 400: return "Very Poor", "Avoid prolonged outdoor exertion and follow local health guidance."
    return "Severe", "Minimize outdoor exposure and follow official health advisories."

def advice(value):
    category, message = aqi_category(value)
    if value <= 100:
        return "Outdoor activities are generally reasonable for most people. Sensitive individuals should monitor symptoms."
    if value <= 200:
        return "Consider reducing long or strenuous outdoor activities if you are sensitive to air pollution."
    return "Reduce outdoor exertion, keep indoor air as clean as possible, and check official local advisories."

data = load_data()
model, metrics, X_test, y_test, predictions = train_model()

st.sidebar.markdown("## 🍃 Aerisense")
st.sidebar.caption("AIR INTELLIGENCE")
page = st.sidebar.radio("Navigate", ["Dashboard", "AQI Prediction", "Data Science Workflow", "Dataset & Model"])
st.sidebar.divider()
st.sidebar.caption("Academic demonstration • Offline demo dataset")
st.sidebar.caption("Not a live monitoring or medical service")

st.markdown('<div class="hero"><h1>🍃 Aerisense</h1><p>Air Intelligence · Explore air quality, understand the data, and estimate AQI with machine learning.</p></div>', unsafe_allow_html=True)
st.info("Transparency note: this project ships with a simulated teaching dataset. The displayed readings are not live CPCB/OpenAQ measurements, and the model is a learning demonstration—not an official AQI calculator.", icon="ℹ️")

if page == "Dashboard":
    st.subheader("Air Quality Overview")
    city = st.selectbox("Select city", sorted(data["city"].unique()))
    city_df = data[data["city"] == city].sort_values("date")
    latest = city_df.iloc[-1]
    score = float(latest["aqi"])
    category, _ = aqi_category(score)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demo AQI", f"{score:.0f}", category)
    c2.metric("PM2.5", f"{latest['pm25']:.1f} µg/m³")
    c3.metric("PM10", f"{latest['pm10']:.1f} µg/m³")
    c4.metric("Temperature", f"{latest['temperature_c']:.1f} °C")
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown("#### AQI trend in demo records")
        trend = city_df.tail(90)
        fig = px.line(
            trend,
            x="date",
            y="aqi",
            labels={"date": "Date", "aqi": "AQI"},
            title=f"{city}: latest 90 demo records",
            template="plotly_white"
        )
        fig.update_layout(margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Pollutant profile")
        vals = pd.DataFrame({"Pollutant":["PM2.5","PM10","NO₂","CO","SO₂","O₃"],
                             "Reading":[latest["pm25"],latest["pm10"],latest["no2"],latest["co"],latest["so2"],latest["o3"]]})
        
        fig2 = px.bar(
            vals,
            x="Pollutant",
            y="Reading",
            title="Latest demo row (units differ)",
            template="plotly_white"
        )

        fig2.update_layout(margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("#### Health guidance")
    st.write(advice(score))
    st.caption("AQI bands shown here are simplified for demonstration. Regulatory AQI systems use pollutant-specific breakpoints and averaging periods.")

elif page == "AQI Prediction":
    st.subheader("Machine Learning AQI Estimator")
    st.write("Enter environmental and pollutant readings. The Random Forest model estimates an AQI-like target learned from the included demo dataset.")
    with st.form("prediction_form"):
        a,b,c = st.columns(3)
        with a:
            temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=29.0)
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=58.0)
            pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=1000.0, value=32.0)
        with b:
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=1500.0, value=61.0)
            no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, max_value=500.0, value=21.0)
            co = st.number_input("CO (demo units)", min_value=0.0, max_value=100.0, value=0.7)
        with c:
            so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, max_value=500.0, value=8.0)
            o3 = st.number_input("O₃ (µg/m³)", min_value=0.0, max_value=500.0, value=32.0)
        submitted = st.form_submit_button("Estimate AQI", type="primary")
    if submitted:
        row = pd.DataFrame([[temperature,humidity,pm25,pm10,no2,co,so2,o3]], columns=FEATURES)
        estimate = float(model.predict(row)[0])
        category, description = aqi_category(estimate)
        st.success(f"Estimated demo AQI: {estimate:.1f} — {category}")
        st.write(description)
        st.write("**Recommendation:** " + advice(estimate))
        st.caption("This is a model estimate, not a future forecast. Real forecasting requires timestamped historical data and forecast-horizon features.")

elif page == "Data Science Workflow":
    st.subheader("End-to-end Data Science Process")
    stages = [
        ("1. Problem definition", "Estimate an AQI-like value from weather and pollutant measurements."),
        ("2. Data collection", "Load the included CSV dataset. In a real deployment, replace it with verified CPCB/OpenAQ or other authoritative data."),
        ("3. Data understanding", "Inspect columns, data types, distributions, summary statistics, and target range."),
        ("4. Data cleaning", "Check missing values, duplicate records, invalid ranges, and inconsistent units before model training."),
        ("5. Exploratory Data Analysis", "Visualize AQI trends, pollutant distributions, correlations, and city-level patterns."),
        ("6. Feature selection", "Use temperature, humidity, PM2.5, PM10, NO₂, CO, SO₂ and O₃ as model inputs."),
        ("7. Train/test split", "Keep 20% of rows as a held-out test set; random split is for this demo, not a time-series forecast."),
        ("8. Model training", "Train a Random Forest Regressor to learn nonlinear relationships."),
        ("9. Evaluation", "Report MAE, RMSE and R² on the held-out test set; compare with a baseline before real-world use."),
        ("10. Prediction & communication", "Accept new input values, estimate the target, explain limitations, and present results in a dashboard.")
    ]
    for title, desc in stages:
        with st.expander(title, expanded=False):
            st.write(desc)
    st.markdown("#### Current model evaluation")
    m1,m2,m3 = st.columns(3)
    m1.metric("MAE", f"{metrics['MAE']:.2f}")
    m2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    m3.metric("R²", f"{metrics['R²']:.3f}")
    st.caption(f"Evaluation uses {metrics['test_rows']} held-out rows from the simulated dataset. Scores are educational only and should not be interpreted as real-world performance.")

elif page == "Dataset & Model":
    st.subheader("Dataset inspection")
    a,b,c = st.columns(3)
    a.metric("Rows", f"{len(data):,}")
    b.metric("Columns", str(data.shape[1]))
    c.metric("Missing cells", str(int(data.isna().sum().sum())))
    st.markdown("#### Sample records")
    st.dataframe(data.head(20), use_container_width=True)
    with st.expander("Descriptive statistics"):
        st.dataframe(data.select_dtypes(include="number").describe().T, use_container_width=True)
    with st.expander("Correlation heatmap"):
        corr = data.select_dtypes(include="number").corr().round(2)
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Numeric feature correlation",
            color_continuous_scale="Greens"
        )

        st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Model feature importance")
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
    fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Random Forest feature importance")
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download demo dataset CSV", data=data.to_csv(index=False).encode("utf-8"),
                       file_name="air_quality_demo.csv", mime="text/csv")
