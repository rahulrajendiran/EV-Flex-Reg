import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import plotly.express as px
import plotly.graph_objects as go
from v2g_controller import v2g_controller, soc_optimization_factor, battery_degradation_cost_per_kwh

# ======================================================
# CONSTANTS & PATHS
# ======================================================
MODELS_DIR = "models"
RESULTS_DIR = "results"
PROCESSED_PATH = os.path.join(RESULTS_DIR, "processed_ev_data.csv")

POINT_MODEL = os.path.join(MODELS_DIR, "lightgbm_point_model.pkl")
Q10_MODEL = os.path.join(MODELS_DIR, "quantile_q10.pkl")
Q50_MODEL = os.path.join(MODELS_DIR, "quantile_q50.pkl")
Q90_MODEL = os.path.join(MODELS_DIR, "quantile_q90.pkl")
TS_MODEL = os.path.join(MODELS_DIR, "lightgbm_timeseries_model.pkl")

TARGET_COL = "physics_flexible_kW"

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
# CORE UTILITY FUNCTIONS
# ======================================================

# ======================================================
# EXPORT UTILITIES (BI REPORT GENERATION)
# ======================================================

def export_csv(df):
    """Generates a CSV string from a dataframe."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()

def export_excel(df_dict):
    """Generates a multi-sheet Excel file buffer from a dictionary of dataframes."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for sheet, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    buffer.seek(0)
    return buffer

@st.cache_data
def load_ev_data(uploaded, processed_path):
    """Loads and cleans the EV dataset."""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
    else:
        return None

    # Column Normalization
    df.columns = df.columns.str.strip()
    COLUMN_ALIASES = {
        "Energy_Consumed_(kWh)": "Energy_Consumed_(kWh)",
        "EnergyConsumed(kWh)": "Energy_Consumed_(kWh)",
        "Energy Consumed kWh": "Energy_Consumed_(kWh)",
        "Energy Consumed (kWh)": "Energy_Consumed_(kWh)",
        "Charging_Start_Time": "Charging Start Time",
        "Charging_End_Time": "Charging End Time"
    }
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    # Datetime Safety
    df["Charging Start Time"] = pd.to_datetime(df["Charging Start Time"], errors="coerce")
    df["Charging End Time"] = pd.to_datetime(df["Charging End Time"], errors="coerce")
    df = df.dropna(subset=["Charging Start Time", "Charging End Time"])
    
    # Ensure duration and basic filters
    if "duration_min" in df.columns:
        df = df[df["duration_min"] > 0]
    
    return df

def apply_dataset_filters(df, hour_range, soc_range, weekend_only):
    """Applies sidebar filters to the dataframe."""
    if df is None:
        return None
        
    df_filt = df[
        df["start_hour"].between(*hour_range) &
        df["soc_est"].between(*soc_range)
    ].copy()
    
    if weekend_only:
        df_filt = df_filt[df_filt["is_weekend"] == 1]
        
    return df_filt

def check_models_built():
    """Ensures models exist, or triggers auto-training."""
    models = [POINT_MODEL, Q10_MODEL, Q50_MODEL, Q90_MODEL, TS_MODEL]
    if not all(os.path.exists(p) for p in models):
        with st.spinner("📦 Models missing. Auto-training starting..."):
            from train_and_save_lightbgm_quantile import main as train_main
            train_main()
    
    # Final safety check
    for p in models:
        if not os.path.exists(p):
            st.error(f"❌ Missing model file: {os.path.basename(p)}")
            st.stop()

# ======================================================
# SOCIETAL & GRID UTILITIES
# ======================================================

def grid_stress_weight(stress_level):
    """
    Increases dispatch priority during grid stress.
    """
    return {
        "Low": 1.0,
        "Medium": 1.2,
        "High": 1.5
    }.get(stress_level, 1.0)

def carbon_benefit_value(
    dispatch_kw,
    duration_hr,
    carbon_intensity,
    carbon_price=0.002
):
    """
    Estimates monetary value of CO₂ avoided.
    carbon_price: ₹ per gram CO₂ (policy-adjustable)
    """
    energy_exported = dispatch_kw * duration_hr  # kWh
    avoided_co2 = energy_exported * carbon_intensity  # grams
    return avoided_co2 * carbon_price

# ======================================================
# UI COMPONENTS (SIDEBAR & PAGES)
# ======================================================

def inject_custom_css():
    """Injects Power BI-style CSS for executive-grade typography and spacing."""
    st.markdown("""
    <style>
    /* Power BI–like background */
    .main {
        background-color: #0e1117;
    }

    /* Global font smoothing & Hierarchy */
    html, body, [class*="css"] {
        font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        letter-spacing: 0.2px;
    }

    /* Reduce vertical clutter */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    /* Section spacing */
    section.main > div {
        padding-bottom: 1.5rem;
    }

    /* Professional Metric cards */
    [data-testid="metric-container"] {
        background-color: #1c1f26;
        border: 1px solid #2c2f36;
        padding: 12px;
        border-radius: 8px;
        box-shadow: none;
        margin-bottom: 0.5rem;
    }

    /* Headers */
    h1, h2, h3 {
        color: #f0f2f6;
        font-weight: 600;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111418;
    }

    /* Caption styling */
    .stCaption {
        color: #a0a4ad;
        font-size: 0.9rem;
    }

    /* Cleaner expanders */
    details summary {
        font-size: 0.95rem;
        color: #d0d4dc;
    }
    </style>
    """, unsafe_allow_html=True)

def sidebar_controls():
    """Renders the sidebar navigation and slicers (Power BI Slicers Panel)."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose View",
        ["Dataset Forecast", "Manual Forecast", "Time-Series Forecast"]
    )
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📂 Dataset Slicers")
    hour_range = st.sidebar.slider(
        "Hour Range", 
        0, 23, (0, 23),
        help="Filter charging sessions by start hour"
    )
    soc_range = st.sidebar.slider(
        "SoC Range", 
        0.0, 1.0, (0.0, 1.0),
        help="Filter by estimated battery State of Charge"
    )
    weekend_only = st.sidebar.checkbox(
        "Weekend Only",
        help="Show only weekend charging sessions"
    )
    st.sidebar.markdown("---")
    
    # ==============================
    # ADVANCED GRID & POLICY CONTROLS
    # ==============================
    st.sidebar.subheader("⚡ Grid Conditions")

    grid_stress_level = st.sidebar.selectbox(
        "Grid Stress Level",
        ["Low", "Medium", "High"],
        help="Represents grid congestion or emergency conditions"
    )

    carbon_intensity = st.sidebar.slider(
        "Grid Carbon Intensity (gCO₂/kWh)",
        200, 900, 600,
        help="Higher value means dirtier grid generation"
    )

    st.sidebar.subheader("🏛 Policy Simulation")

    v2g_incentive = st.sidebar.slider(
        "V2G Incentive (₹/kWh)",
        0.0, 5.0, 1.5,
        help="Extra incentive paid for V2G participation"
    )

    battery_compensation = st.sidebar.slider(
        "Battery Compensation (₹/kWh)",
        0.0, 4.0, 1.0,
        help="Compensation to offset battery degradation"
    )
    
    return page, hour_range, soc_range, weekend_only, grid_stress_level, carbon_intensity, v2g_incentive, battery_compensation

def render_dataset_kpis(df_filt):
    """Synchronized KPI Row with refined labels."""
    k1, k2, k3, k4, k5 = st.columns(5)
    
    if df_filt.empty:
        for k in [k1, k2, k3, k4, k5]: k.metric("N/A", "0.00")
        return

    k1.metric("Sessions", f"{len(df_filt):,}")
    k2.metric("Avg Flex (kW)", f"{df_filt[TARGET_COL].mean():.2f}")
    
    peak_val = df_filt.groupby('start_hour')[TARGET_COL].mean().max()
    k3.metric("Peak Hour (kW)", f"{peak_val:.2f}")
    
    k4.metric("Median (kW)", f"{df_filt[TARGET_COL].median():.2f}")
    k5.metric("V2G Eligibility", f"{(df_filt[TARGET_COL] > 0).mean()*100:.1f}%")

def render_dataset_charts(df_filt):
    """Synchronized Charts with question-based titles."""
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            df_filt,
            x="Charging Start Time",
            y=TARGET_COL,
            title="How Flexible Power Varies Over Time?",
            labels={TARGET_COL: "Flexible kW"}
        )
        fig.update_layout(template="plotly_dark", margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig, width='stretch')

    with c2:
        hourly = df_filt.groupby("start_hour")[TARGET_COL].mean().reset_index()
        fig2 = px.bar(
            hourly,
            x="start_hour",
            y=TARGET_COL,
            title="When is Flexibility Highest during the Day?",
            labels={"start_hour": "Hour of Day", TARGET_COL: "Avg kW"}
        )
        fig2.update_layout(template="plotly_dark", margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig2, width='stretch')

def render_dataset_table(df_filt):
    """Synchronized Drill-Down Table."""
    with st.expander("🔍 Drill Down: Session-Level Data"):
        st.dataframe(df_filt.tail(300), width='stretch')

def render_dataset_page(df_filt):
    """Main Dataset Page Controller with Cross-Filtering."""
    st.header("EV Flexibility — Dataset Analytics")
    st.caption("Historical charging trends and regulatory flexibility availability")

    if df_filt is None or df_filt.empty:
        st.warning("No data matches the selected filters.")
        return

    # Secondary Cross-Highlighting Selector (Chart-Driven concept)
    st.markdown("---")
    col_sel, _ = st.columns([1, 2])
    with col_sel:
        selected_hour = st.selectbox(
            "🎯 Focus on Hour (Cross-Highlight)",
            [None] + sorted(df_filt["start_hour"].unique().tolist()),
            help="Re-filters all visuals below to a specific hour."
        )
    
    if selected_hour is not None:
        df_filt = df_filt[df_filt["start_hour"] == selected_hour]

    # Execute Synchronized Visuals
    render_dataset_kpis(df_filt)
    render_dataset_charts(df_filt)
    render_dataset_table(df_filt)

    # --- EXPORT BI REPORT ---
    st.markdown("---")
    with st.expander("📤 Export Dataset Analytics"):
        st.caption("Generate a comprehensive Power BI-style report of the current filtered data.")
        
        hourly_summary = (
            df_filt
            .groupby("start_hour")[TARGET_COL]
            .agg(["mean", "max", "count"])
            .reset_index()
            .rename(columns={
                "mean": "Avg_Flex_kW",
                "max": "Peak_Flex_kW",
                "count": "Sessions"
            })
        )

        kpi_summary = pd.DataFrame([{
            "Total_Sessions": len(df_filt),
            "Avg_Flex_kW": df_filt[TARGET_COL].mean(),
            "Peak_Flex_kW": hourly_summary["Peak_Flex_kW"].max() if not hourly_summary.empty else 0,
            "V2G_Eligible_%": (df_filt[TARGET_COL] > 0).mean() * 100
        }])

        excel_report = export_excel({
            "Session_Data": df_filt,
            "Hourly_Summary": hourly_summary,
            "KPI_Summary": kpi_summary
        })

        st.download_button(
            "⬇️ Download Dataset BI Report (Excel)",
            excel_report,
            file_name="EV_Flexibility_Dataset_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width='stretch'
        )


def render_manual_page(grid_stress_level, carbon_intensity, v2g_incentive, battery_compensation):
    """Renders the Manual Decision Simulator page."""
    st.header("Manual V2G Decision Simulator")
    st.caption("Scenario-based evaluation of flexibility, battery safety, and economics")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("🔌 Charging Session")
        energy = st.number_input("Energy (kWh)", 0.0, 60.0, 20.0)
        duration_min = st.number_input("Duration (min)", 1.0, 600.0, 60.0)
        start_hour = st.slider("Start Hour", 0, 23, 10)
        day_of_week = st.selectbox("Day of Week", list(range(7)), index=0)
        is_weekend = st.selectbox("Is Weekend?", [0, 1], index=0)

        st.subheader("🔋 Battery & Grid")
        soc_real = st.slider("Battery SoC", 0.0, 1.0, 0.6)
        grid_price = st.slider("Grid Price (₹/kWh)", 2.0, 10.0, 8.0)

    # --- Computation Logic ---
    duration_hr = duration_min / 60
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

    physics_est = min(energy / duration_hr, 7.2) * 0.3 * soc_est if duration_hr > 0 else 0
    physics_est = min(physics_est, 3.6)
    
    final_pred = max(ml_pred, physics_est)

    # ==============================
    # ECONOMIC & SOCIETAL DECISION LOGIC
    # ==============================

    # Base dispatch from existing controller
    base_dispatch = v2g_controller(
        final_pred,
        grid_price,
        soc_real,
        duration_hr=duration_hr
    )

    # Apply grid stress weighting
    stress_factor = grid_stress_weight(grid_stress_level)
    stress_adjusted_dispatch = min(
        base_dispatch * stress_factor,
        3.6  # export limit
    )

    # Battery degradation cost
    deg_per_kwh = battery_degradation_cost_per_kwh()
    degradation_cost = deg_per_kwh * stress_adjusted_dispatch * duration_hr

    # Grid revenue (with policy incentives)
    effective_price = grid_price + v2g_incentive + battery_compensation
    grid_revenue = stress_adjusted_dispatch * duration_hr * effective_price

    # Carbon benefit
    carbon_benefit = carbon_benefit_value(
        stress_adjusted_dispatch,
        duration_hr,
        carbon_intensity
    )

    # FINAL DECISION
    if (grid_revenue + carbon_benefit) > degradation_cost:
        final_dispatch = stress_adjusted_dispatch
    else:
        final_dispatch = 0.0

    soc_factor = soc_optimization_factor(soc_real)
    soc_scaled_kw = final_pred * soc_factor

    with right:
        # KPI Decision Summary
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ML Flex", f"{ml_pred:.2f}")
        k2.metric("Physics Limit", f"{physics_est:.2f}")
        k3.metric("SoC-Scaled", f"{soc_scaled_kw:.2f}")
        k4.metric("Dispatch", f"{final_dispatch:.2f}")

        st.markdown("---")
        st.markdown("### 🌍 Societal Impact Metrics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Grid Revenue (₹)", f"{grid_revenue:.2f}")
        c2.metric("Degradation Cost (₹)", f"{degradation_cost:.2f}")
        c3.metric("Carbon Benefit (₹)", f"{carbon_benefit:.2f}")

        st.success(f"⚙️ Final V2G Dispatch (Societal-Aware): **{final_dispatch:.2f} kW**")
        
        # Optimization Curve Visual
        soc_range = np.linspace(0, 1, 100)
        soc_factors = [soc_optimization_factor(s) for s in soc_range]

        fig = px.line(
            x=soc_range,
            y=soc_factors,
            labels={"x": "SoC", "y": "Scaling Factor"},
            title="SoC-Aware V2G Optimization Curve"
        )
        fig.add_vline(x=soc_real, line_dash="dash", line_color="red")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, width='stretch')

        st.subheader("💰 Economics: Revenue vs Degradation")
        # Use calculated degradation and revenue including incentives
        revenue_vals = [grid_revenue, carbon_benefit]
        labels = ["Grid Revenue + Incentives", "Carbon Benefit"]
        
        econ_df = pd.DataFrame({
            "Category": labels + ["Degradation Cost"],
            "Amount (₹)": revenue_vals + [degradation_cost]
        })
        fig_econ = px.bar(
            econ_df, x="Amount (₹)", y="Category", orientation='h',
            color="Category", color_discrete_map={"Revenue": "#2CA02C", "Degradation Cost": "#D62728"}
        )
        fig_econ.update_layout(height=180, showlegend=False, template="plotly_white")
        st.plotly_chart(fig_econ, width='stretch')

    # --- EXPORT DECISION REPORT ---
    st.markdown("---")
    with st.expander("📤 Export V2G Decision Report"):
        st.caption("Export a traceable audit record of this manual dispatch decision.")
        
        decision_report = pd.DataFrame([{
            "Timestamp": pd.Timestamp.now(),
            "Energy_kWh": energy,
            "Duration_min": duration_min,
            "Start_Hour": start_hour,
            "SoC": soc_real,
            "Grid_Price_Rs_per_kWh": grid_price,
            "ML_Flex_kW": ml_pred,
            "Physics_Limit_kW": physics_est,
            "SoC_Scaled_kW": soc_scaled_kw,
            "Grid_Stress": grid_stress_level,
            "Carbon_Intensity": carbon_intensity,
            "V2G_Incentive": v2g_incentive,
            "Battery_Comp": battery_compensation,
            "Grid_Revenue": grid_revenue,
            "Degradation_Cost": degradation_cost,
            "Carbon_Benefit": carbon_benefit,
            "Final_Dispatch_kW": final_dispatch
        }])

        st.download_button(
            "⬇️ Download Decision Summary (CSV)",
            export_csv(decision_report),
            file_name="V2G_Manual_Decision_Report.csv",
            mime="text/csv",
            width='stretch'
        )

    # Explanation
    with st.expander("🧠 Why this decision?"):
        reasons = []
        if soc_real > 0.8: reasons.append("- High SoC: Maximum V2G export allowed.")
        elif soc_real < 0.2: reasons.append("- Low SoC: Export strictly limited to preserve battery health.")
        else: reasons.append("- Moderate SoC: Balanced export permitted.")
        
        if (grid_revenue + carbon_benefit) > degradation_cost:
            reasons.append(f"- Net Positive Impact: (Revenue ₹{grid_revenue:.2f} + Carbon ₹{carbon_benefit:.2f}) > Degradation ₹{degradation_cost:.2f}.")
        else:
            reasons.append(f"- Economic/Societal Loss: (Revenue ₹{grid_revenue:.2f} + Carbon ₹{carbon_benefit:.2f}) <= Degradation ₹{degradation_cost:.2f}. Dispatch paused.")
        
        if grid_stress_level != "Low":
            reasons.append(f"- Grid Stress ({grid_stress_level}): Dispatch priority increased by {stress_factor}x.")
            
        reasons.append(f"- Physics limit calculated at {physics_est:.2f} kW based on charger and session constraints.")
        st.markdown("\n".join(reasons))

def render_timeseries_page(df):
    """Renders the Time-Series Forecast page."""
    st.header("EV Flexibility — Planning Forecast")
    st.caption("Risk-aware aggregate flexibility for future grid planning")

    st.sidebar.subheader("📅 Forecast Controls")
    horizon = st.sidebar.slider("Horizon (hours)", 24, 72, 24, step=24)
    risk_mode = st.sidebar.radio("Planning Mode", ["Conservative", "Expected", "Optimistic"])

    # --- Data Prep ---
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

    # Uncertainty based on planning mode
    if risk_mode == "Conservative":
        display_line = lower
    elif risk_mode == "Optimistic":
        display_line = upper
    else:
        display_line = preds

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Forecast", f"{preds.mean():.2f}")
    k2.metric(f"Worst Case ({risk_mode})", f"{lower.mean():.2f}")
    k3.metric(f"Best Case ({risk_mode})", f"{upper.mean():.2f}")
    k4.metric("Peak Forecast", f"{preds.max():.2f}")

    # Forecast Visualization
    fig = go.Figure()
    
    # Highlight the display line based on risk mode
    fig.add_trace(go.Scatter(
        y=display_line, 
        mode="lines", 
        name=f"{risk_mode} Forecast", 
        line=dict(color='blue', width=3)
    ))
    
    if risk_mode == "Expected":
        fig.add_trace(go.Scatter(y=upper, fill=None, mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(y=lower, fill="tonexty", mode="lines", line=dict(width=0), name="Uncertainty Band", fillcolor='rgba(0,0,255,0.1)'))

    fig.update_layout(
        title=f"Scenario-Based Flexibility Forecast: {risk_mode} Analysis",
        xaxis_title="Hours into Future",
        yaxis_title="kW",
        template="plotly_white"
    )
    st.plotly_chart(fig, width='stretch')

    # --- EXPORT PLANNING REPORT ---
    st.markdown("---")
    with st.expander("📤 Export Planning Forecast"):
        st.caption("Generate a forecast report for grid planning and utility operations.")
        
        forecast_df = pd.DataFrame({
            "Hour_Index": range(len(preds)),
            "Expected_Flex_kW": preds,
            "Worst_Case_kW": lower,
            "Best_Case_kW": upper
        })

        st.download_button(
            "⬇️ Download Forecast Report (CSV)",
            export_csv(forecast_df),
            file_name="EV_Flexibility_TimeSeries_Forecast.csv",
            mime="text/csv",
            width='stretch'
        )

# ======================================================
# MAIN EXECUTION
# ======================================================

def main():
    st.set_page_config(layout="wide", page_title="EV Flex Forecast")
    inject_custom_css()
    
    st.header("EV Flexible Regulation Forecast Dashboard")
    st.caption("Physics-aware + ML-based forecasting of EV flexible power with probabilistic uncertainty")

    # 1. Sidebar & Navigation
    (page, hour_range, soc_range, weekend_only, 
     grid_stress_level, carbon_intensity, 
     v2g_incentive, battery_compensation) = sidebar_controls()

    # 2. Model Integrity Check
    check_models_built()

    # 3. Data Loading
    uploaded = st.file_uploader("📂 Upload processed_ev_data.csv (optional)", type="csv")
    df = load_ev_data(uploaded, PROCESSED_PATH)

    if df is None:
        st.error("❌ processed_ev_data.csv not found.")
        st.stop()
        
    # Consistency check
    if TARGET_COL not in df.columns:
        st.error(f"❌ {TARGET_COL} not found. Re-run training script.")
        st.stop()

    # 4. Page Routing
    if page == "Dataset Forecast":
        df_filt = apply_dataset_filters(df, hour_range, soc_range, weekend_only)
        render_dataset_page(df_filt)

    elif page == "Manual Forecast":
        render_manual_page(grid_stress_level, carbon_intensity, v2g_incentive, battery_compensation)

    elif page == "Time-Series Forecast":
        render_timeseries_page(df)

if __name__ == "__main__":
    main()
