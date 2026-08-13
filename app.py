"""
Aircraft Engine Health Monitoring Dashboard
NASA C-MAPS Turbofan Predictive Maintenance
Professional industrial dashboard for RUL prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from data_loader import load_data

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Aircraft Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - DARK PROFESSIONAL THEME
# ============================================================================

st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }

    /* Title styling */
    .main-title {
        color: #ffffff;
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .sub-title {
        color: #00d4ff;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* Health status badges */
    .health-container {
        text-align: center;
        margin: 30px 0;
    }

    .health-badge {
        display: inline-block;
        padding: 20px 50px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 32px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .badge-healthy {
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        color: white;
    }

    .badge-monitor {
        background: linear-gradient(135deg, #FF9800 0%, #FFB74D 100%);
        color: white;
    }

    .badge-critical {
        background: linear-gradient(135deg, #F44336 0%, #EF5350 100%);
        color: white;
    }

    .rul-value {
        font-size: 52px;
        font-weight: 700;
        margin-top: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .rul-label {
        font-size: 18px;
        color: #aaa;
        margin-top: 5px;
    }

    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 24px;
        font-weight: 600;
        margin-top: 40px;
        margin-bottom: 20px;
        border-left: 4px solid #00d4ff;
        padding-left: 15px;
    }

    /* Metrics */
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3347;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1d2e;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #888;
        font-size: 14px;
        border-top: 2px solid #2e3347;
        margin-top: 60px;
        background-color: #0a0c10;
    }

    .footer-title {
        font-weight: 700;
        color: #00d4ff;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING (CACHED)
# ============================================================================

@st.cache_data
def load_model_and_scaler():
    """Load trained XGBoost model and scaler."""
    try:
        models_dir = Path('models')

        # Load model
        with open(models_dir / 'xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        return model, scaler

    except FileNotFoundError as e:
        st.error(f"❌ Error: Model files not found. Please run training first.")
        st.stop()

@st.cache_data
def load_test_dataset():
    """Load and prepare test dataset with predictions."""
    try:
        # Load raw data
        _, test_df, test_rul_df = load_data('data/')

        # Load model and scaler
        model, scaler = load_model_and_scaler()

        # Prepare features manually (simplified version)
        from feature_engineering import prepare_features
        X_train, y_train, X_test, test_rul, _, feature_names = prepare_features('data/')

        # Generate predictions for all test cycles
        all_predictions = model.predict(X_test)
        all_predictions = np.maximum(all_predictions, 0)

        # Add predictions to test dataframe
        test_df['predicted_RUL'] = all_predictions

        # Get last prediction per engine
        engine_predictions = {}
        engine_actuals = {}
        current_idx = 0

        for unit_id in sorted(test_df['unit_id'].unique()):
            unit_mask = test_df['unit_id'] == unit_id
            unit_cycle_count = unit_mask.sum()

            # Get last prediction for this engine
            last_pred = all_predictions[current_idx + unit_cycle_count - 1]
            engine_predictions[unit_id] = last_pred

            # Map to actual RUL (index-based)
            engine_idx = list(sorted(test_df['unit_id'].unique())).index(unit_id)
            engine_actuals[unit_id] = test_rul[engine_idx]

            current_idx += unit_cycle_count

        return test_df, engine_predictions, engine_actuals

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_health_status(rul):
    """Determine health status based on RUL."""
    if rul > 50:
        return "HEALTHY", "badge-healthy", "#00C853"
    elif 20 <= rul <= 50:
        return "MONITOR", "badge-monitor", "#FF9800"
    else:
        return "CRITICAL", "badge-critical", "#F44336"

def plot_sensor_degradation(engine_data):
    """Plot sensor readings over time for selected engine."""
    fig = go.Figure()

    # Plot key sensors
    sensors = ['sensor_2', 'sensor_3', 'sensor_4']
    colors = ['#00d4ff', '#ff6b9d', '#ffd700']

    for sensor, color in zip(sensors, colors):
        if sensor in engine_data.columns:
            # Normalize sensor values for better visualization
            values = engine_data[sensor].values
            normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)

            fig.add_trace(go.Scatter(
                x=engine_data['cycle'],
                y=normalized,
                mode='lines',
                name=sensor.replace('_', ' ').title(),
                line=dict(color=color, width=2.5),
                hovertemplate='Cycle: %{x}<br>Value: %{y:.3f}<extra></extra>'
            ))

    fig.update_layout(
        title={
            'text': "Sensor Degradation Over Time",
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Flight Cycles",
        yaxis_title="Normalized Sensor Value",
        template="plotly_dark",
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        plot_bgcolor='#1a1d2e',
        paper_bgcolor='#0e1117',
    )

    return fig

def plot_fleet_overview(engine_predictions):
    """Create bar chart showing RUL distribution across fleet."""
    # Prepare data
    units = sorted(engine_predictions.keys())
    ruls = [engine_predictions[u] for u in units]

    # Assign colors based on health status
    colors = []
    for rul in ruls:
        if rul > 50:
            colors.append('#00C853')  # Green
        elif 20 <= rul <= 50:
            colors.append('#FF9800')  # Orange
        else:
            colors.append('#F44336')  # Red

    fig = go.Figure(data=[
        go.Bar(
            x=units,
            y=ruls,
            marker_color=colors,
            text=[f'{r:.0f}' for r in ruls],
            textposition='outside',
            hovertemplate='Engine %{x}<br>RUL: %{y:.0f} cycles<extra></extra>'
        )
    ])

    # Add threshold lines
    fig.add_hline(y=50, line_dash="dash", line_color="#00C853",
                  annotation_text="Healthy Threshold", annotation_position="right")
    fig.add_hline(y=20, line_dash="dash", line_color="#F44336",
                  annotation_text="Critical Threshold", annotation_position="right")

    fig.update_layout(
        title={
            'text': "Fleet Health Overview - All 100 Engines",
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis_title="Engine Unit ID",
        yaxis_title="Remaining Useful Life (cycles)",
        template="plotly_dark",
        height=500,
        showlegend=False,
        plot_bgcolor='#1a1d2e',
        paper_bgcolor='#0e1117',
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=5
        )
    )

    return fig

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard application."""

    # ========================================================================
    # HEADER
    # ========================================================================
    st.markdown('<div class="main-title">✈️ Aircraft Engine Health Monitor</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-title">NASA C-MAPS Turbofan Predictive Maintenance</div>',
                unsafe_allow_html=True)

    # ========================================================================
    # LOAD DATA
    # ========================================================================
    with st.spinner("🔄 Loading model and data..."):
        test_df, engine_predictions, engine_actuals = load_test_dataset()

    # ========================================================================
    # SIDEBAR
    # ========================================================================
    st.sidebar.markdown("## 🎛️ Control Panel")
    st.sidebar.markdown("---")

    # Engine selector
    st.sidebar.markdown("### Select Engine")
    available_engines = sorted(engine_predictions.keys())
    selected_engine = st.sidebar.selectbox(
        "Engine Unit ID",
        available_engines,
        format_func=lambda x: f"Engine #{x}",
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Model metrics
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.metric("RMSE", "18.90 cycles", help="Root Mean Squared Error")
    st.sidebar.metric("R² Score", "0.78", help="Coefficient of Determination")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This dashboard predicts remaining useful life (RUL) of turbofan engines "
        "using machine learning on NASA C-MAPS sensor data."
    )

    # ========================================================================
    # MAIN CONTENT
    # ========================================================================

    # Get data for selected engine
    engine_data = test_df[test_df['unit_id'] == selected_engine].copy()
    predicted_rul = engine_predictions[selected_engine]
    actual_rul = engine_actuals[selected_engine]

    # ------------------------------------------------------------------------
    # SECTION 1: HEALTH STATUS
    # ------------------------------------------------------------------------
    st.markdown('<div class="section-header">🩺 Engine Health Status</div>',
                unsafe_allow_html=True)

    status_text, status_class, status_color = get_health_status(predicted_rul)

    # Create centered health badge
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div class="health-container">
                <div class="health-badge {status_class}">
                    {status_text}
                </div>
                <div class="rul-value" style="color: {status_color};">
                    {predicted_rul:.0f} Cycles
                </div>
                <div class="rul-label">
                    Predicted Remaining Useful Life
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Quick metrics
    st.markdown("")
    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric("Predicted RUL", f"{predicted_rul:.0f} cycles")

    with metric2:
        st.metric("Actual RUL", f"{actual_rul:.0f} cycles")

    with metric3:
        error = abs(predicted_rul - actual_rul)
        st.metric("Prediction Error", f"{error:.1f} cycles")

    with metric4:
        current_cycle = engine_data['cycle'].max()
        st.metric("Total Cycles", f"{current_cycle}")

    # ------------------------------------------------------------------------
    # SECTION 2: DEGRADATION CURVE
    # ------------------------------------------------------------------------
    st.markdown('<div class="section-header">📉 Sensor Degradation Analysis</div>',
                unsafe_allow_html=True)

    sensor_fig = plot_sensor_degradation(engine_data)
    st.plotly_chart(sensor_fig, use_container_width=True)

    # ------------------------------------------------------------------------
    # SECTION 3: FLEET OVERVIEW
    # ------------------------------------------------------------------------
    st.markdown('<div class="section-header">🛠️ Fleet Health Overview</div>',
                unsafe_allow_html=True)

    # Fleet statistics
    healthy_count = sum(1 for rul in engine_predictions.values() if rul > 50)
    monitor_count = sum(1 for rul in engine_predictions.values() if 20 <= rul <= 50)
    critical_count = sum(1 for rul in engine_predictions.values() if rul < 20)

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric("Total Engines", len(engine_predictions))

    with stat2:
        st.metric("Healthy", healthy_count, delta=None, delta_color="normal")

    with stat3:
        st.metric("Monitor", monitor_count, delta=None, delta_color="normal")

    with stat4:
        st.metric("Critical", critical_count, delta=None, delta_color="inverse")

    # Fleet bar chart
    fleet_fig = plot_fleet_overview(engine_predictions)
    st.plotly_chart(fleet_fig, use_container_width=True)

    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("""
        <div class="footer">
            <div class="footer-title">Powered by NASA C-MAPS Dataset</div>
            Built by <strong>Rayhane Nouri</strong>, ENSIT Tunisia<br>
            Commercial Modular Aero-Propulsion System Simulation | FD001 Dataset<br>
            XGBoost Regression Model for Predictive Maintenance
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
