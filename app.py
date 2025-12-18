import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from v2g_controller import v2g_controller, soc_optimization_factor, battery_degradation_cost_per_kwh


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(layout="wide", page_title="EV Flex Forecast")

MODELS_DIR = "models"
RESULTS_DIR = "./results"
PROCESSED_PATH = os.path.join(RESULTS_DIR, "processed_ev_data.csv")

# ======================================================
# HEADER
# ======================================================
st.title("🔋 EV Flexible Regulation Forecast — Interactive Dashboard")
st.markdown("""
**Physics-aware + ML-based forecasting** of EV flexible power  
with **probabilistic uncertainty** and **time-series prediction**.
""")

# ======================================================
# LOAD DATA (ROBUST)
# ======================================================
uploaded = st.file_uploader("📂 Upload processed_ev_data.csv (optional)", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("✅ Uploaded dataset loaded.")
elif os.path.exists(PROCESSED_PATH):
    df = pd.read_csv(PROCESSED_PATH)
    st.info("ℹ️ Loaded processed dataset from results/")
else:
    st.error("❌ processed_ev_data.csv not found.")
    st.stop()
# ======================================================
# COLUMN NORMALIZATION (FINAL + SAFE)
# ======================================================
df.columns = df.columns.str.strip()

# Explicit alias handling ONLY (no blanket replacements)
COLUMN_ALIASES = {
    "Energy_Consumed_(kWh)": "Energy_Consumed_(kWh)",
    "EnergyConsumed(kWh)": "Energy_Consumed_(kWh)",
    "Energy Consumed kWh": "Energy_Consumed_(kWh)",
    "Energy Consumed (kWh)": "Energy_Consumed_(kWh)",
    "Charging_Start_Time": "Charging Start Time",
    "Charging_End_Time": "Charging End Time"
}

df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

# ---- HARD ASSERT (DEBUG SAFETY)
REQUIRED_COLS = [
    "Energy_Consumed_(kWh)",
    "Charging Start Time",
    "duration_min",
    "start_hour",
    "day_of_week",
    "is_weekend",
    "soc_est",
    "physics_flexible_kW"
]

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"❌ Missing required columns: {missing}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# ======================================================
# DATETIME SAFETY
# ======================================================
df["Charging Start Time"] = pd.to_datetime(df["Charging Start Time"], errors="coerce")
df["Charging End Time"] = pd.to_datetime(df["Charging End Time"], errors="coerce")
df = df.dropna(subset=["Charging Start Time", "Charging End Time"])
df = df[df["duration_min"] > 0]

# ======================================================
# TARGET SELECTION (NO FALLBACK LOOPS)
# ======================================================
if "physics_flexible_kW" not in df.columns:
    st.error(
        "❌ physics_flexible_kW not found.\n\n"
        "Re-run `train_and_save_lightgbm_quantile.py` to regenerate processed data."
    )
    st.stop()

TARGET_COL = "physics_flexible_kW"
TARGET_LABEL = "Physics-Constrained Flexible kW"

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("⚙️ Controls")
page = st.sidebar.radio(
    "Choose View",
    ["Dataset Forecast", "Manual Forecast", "Time-Series Forecast"]
)

# ======================================================
# MODEL PATHS
# ======================================================
POINT_MODEL = os.path.join(MODELS_DIR, "lightgbm_point_model.pkl")
Q10_MODEL = os.path.join(MODELS_DIR, "quantile_q10.pkl")
Q50_MODEL = os.path.join(MODELS_DIR, "quantile_q50.pkl")
Q90_MODEL = os.path.join(MODELS_DIR, "quantile_q90.pkl")
TS_MODEL = os.path.join(MODELS_DIR, "lightgbm_timeseries_model.pkl")

for p in [POINT_MODEL, Q10_MODEL, Q50_MODEL, Q90_MODEL, TS_MODEL]:
    if not os.path.exists(p):
        st.error(f"❌ Missing model file: {os.path.basename(p)}")
        st.stop()

# ======================================================
# FEATURE CONTRACT (MUST MATCH TRAINING)
# ======================================================
SESSION_FEATURES = [
    "duration_min",
    "Energy_Consumed_(kWh)",
    "start_hour",
    "day_of_week",
    "is_weekend",
    "soc_est"
]

TS_FEATURES = [
    "start_hour",
    "day_of_week",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_3"
]

# ======================================================
# PAGE 1 — DATASET FORECAST
# ======================================================
if page == "Dataset Forecast":

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    model_type = st.sidebar.selectbox(
        "Model Type",
        ["LightGBM (Point)", "Probabilistic (Q10/Q50/Q90)"]
    )
    n_display = st.sidebar.slider("Rows to display", 20, 1000, 200)

    if model_type == "LightGBM (Point)":

        model = joblib.load(POINT_MODEL)
        feats = model.feature_name_

        preds = model.predict(df[feats])
        out = df.copy()
        out["Prediction"] = preds

        st.dataframe(
            out[["Charging Start Time", TARGET_COL, "Prediction"]].tail(n_display)
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(out[TARGET_COL].tail(n_display).values, label="True")
        ax.plot(out["Prediction"].tail(n_display).values, label="Predicted")
        ax.legend()
        ax.set_title("Dataset Forecast — Point Model")
        st.pyplot(fig)

    else:
        q10 = joblib.load(Q10_MODEL)
        q50 = joblib.load(Q50_MODEL)
        q90 = joblib.load(Q90_MODEL)

        feats = q50.feature_names_in_

        out = df.copy()
        # --- Align dataframe columns to quantile model feature contract ---
        Xq = df.copy()

        # Explicit feature name fix
        if "Energy Consumed (kWh)" in feats and "Energy_Consumed_(kWh)" in Xq.columns:
            Xq = Xq.rename(columns={"Energy_Consumed_(kWh)": "Energy Consumed (kWh)"})

        # Now select features safely
        Xq = Xq[list(feats)]

        out["Q10"] = q10.predict(Xq)
        out["Q50"] = q50.predict(Xq)
        out["Q90"] = q90.predict(Xq)

        # ── SoC-aware scaling for probabilistic outputs ──
        out["soc_factor"] = out["soc_est"].apply(soc_optimization_factor)

        out["Q10_soc"] = out["Q10"] * out["soc_factor"]
        out["Q50_soc"] = out["Q50"] * out["soc_factor"]
        out["Q90_soc"] = out["Q90"] * out["soc_factor"]

        st.dataframe(
            out[
                [
                    "Charging Start Time",
                    TARGET_COL,
                    "Q10_soc",
                    "Q50_soc",
                    "Q90_soc"
                ]
            ].tail(n_display)
        )
        st.write("Quantile model expects:", feats)
        st.write("App dataframe columns:", df.columns.tolist())
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(out[TARGET_COL].tail(n_display).values, label="True")
        ax.plot(out["Q50_soc"].tail(n_display).values, label="Median (SoC-aware)")

        ax.fill_between(
            range(n_display),
            out["Q10_soc"].tail(n_display).values,
            out["Q90_soc"].tail(n_display).values,
            alpha=0.25,
            label="SoC-aware uncertainty band"
        )

        ax.legend()
        ax.set_title("Probabilistic Forecast with SoC-Aware V2G Constraints")
        st.pyplot(fig)

# ======================================================
# PAGE 2 — MANUAL FORECAST
# ======================================================
elif page == "Manual Forecast":

    st.subheader("🧭 Manual Flexible kW Prediction")

    with st.form("manual_form"):
        energy = st.number_input("Energy Consumed (kWh)", 0.0, 60.0, 20.0)
        duration_min = st.number_input("Duration (minutes)", 1.0, 600.0, 60.0)
        start_hour = st.slider("Start Hour", 0, 23, 10)
        day_of_week = st.selectbox("Day of Week", list(range(7)))
        is_weekend = st.selectbox("Is Weekend?", [0, 1])
        submit = st.form_submit_button("🔮 Predict")

    if submit:

        soc_est = np.clip(0.3 + 0.7 * (energy / 60.0), 0.2, 0.95)

        X = pd.DataFrame([{
            "duration_min": duration_min,
            "Energy_Consumed_(kWh)": energy,
            "start_hour": start_hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "soc_est": soc_est
        }])

        model = joblib.load(POINT_MODEL)
        ml_pred = model.predict(X)[0]

        duration_hr = duration_min / 60
        physics_est = (
            min(energy / duration_hr, 7.2) * 0.3 * soc_est
            if duration_hr > 0 else 0
        )
        physics_est = min(physics_est, 3.6)

        final_pred = max(ml_pred, physics_est)
        st.success(f"⚡ Predicted Flexible kW: **{final_pred:.3f} kW**")

        deg_cost_per_kwh = battery_degradation_cost_per_kwh()
        st.success(f"🔋 Estimated degradation cost: ₹{deg_cost_per_kwh:.2f} per kWh")

        grid_price = st.slider("Grid Price (₹/kWh)", 2.0, 10.0, 8.0)
        soc_real = st.slider("Battery SoC", 0.0, 1.0, soc_est)

        # V2G Controller
        soc_factor = soc_optimization_factor(soc_real)
        soc_scaled_kw = final_pred * soc_factor

        st.info(f"🔧 SoC-Optimized Available Power: **{soc_scaled_kw:.2f} kW**")
        
        dispatch = v2g_controller(final_pred, grid_price, soc_real, duration_hr=duration_min / 60)
        st.info(f"⚙️ V2G Dispatch Command: **{dispatch:.2f} kW**")
        st.markdown("---")
        st.subheader("📈 SoC-Aware V2G Optimization Curve")

        # Generate SoC range
        soc_range = np.linspace(0, 1, 100)
        soc_factors = [soc_optimization_factor(s) for s in soc_range]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(soc_range, soc_factors, linewidth=2)
        ax.axvline(soc_real, linestyle="--", color="red", label="Current SoC")

        ax.set_xlabel("State of Charge (SoC)")
        ax.set_ylabel("Scaling Factor")
        ax.set_title("SoC vs V2G Export Permission")
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

# ======================================================
# PAGE 3 — TIME-SERIES FORECAST
# ======================================================
elif page == "Time-Series Forecast":

    st.subheader("⏳ Time-Series Flexible kW Forecast")

    horizon = st.slider("Forecast Horizon (hours)", 24, 72, 24, step=24)

    ts_model = joblib.load(TS_MODEL)

    df_ts = df.copy()
    df_ts["timestamp"] = df_ts["Charging Start Time"].dt.floor("h")

    hourly = (
        df_ts.groupby("timestamp")
        .agg({
            TARGET_COL: "sum",
            "start_hour": "first",
            "day_of_week": "first",
            "is_weekend": "first"
        })
        .reset_index()
    )

    for lag in [1, 2, 3]:
        hourly[f"lag_{lag}"] = hourly[TARGET_COL].shift(lag)

    hourly = hourly.dropna().reset_index(drop=True)

    X_hist = hourly[TS_FEATURES].iloc[-horizon:]
    y_hist = hourly[TARGET_COL].iloc[-horizon:]

    preds = ts_model.predict(X_hist)

    sigma = np.std(y_hist - preds)
    upper = preds + 1.28 * sigma
    lower = np.maximum(preds - 1.28 * sigma, 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_hist.values, label="Observed", color="black")
    ax.plot(preds, label="Forecast", color="blue")
    ax.fill_between(range(len(preds)), lower, upper, alpha=0.25)
    ax.legend()
    ax.set_title("Time-Series Flexible kW Forecast")
    st.pyplot(fig)
