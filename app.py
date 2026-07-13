"""
Aircraft Engine Health Monitoring Dashboard
Real-time RUL prediction visualization for turbofan engines
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from data_loader import CMAPSDataLoader
from feature_engineering import FeatureEngineer
from model import RULPredictor


# Page configuration
st.set_page_config(
    page_title="Aircraft Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional dark theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3347;
    }
    .health-badge {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
        text-align: center;
        margin: 20px 0;
    }
    .healthy {
        background-color: #00C853;
        color: white;
    }
    .monitor {
        background-color: #FF9800;
        color: white;
    }
    .critical {
        background-color: #F44336;
        color: white;
    }
    h1 {
        color: #ffffff;
        font-weight: 700;
    }
    h2, h3 {
        color: #e0e0e0;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
        font-size: 14px;
        border-top: 1px solid #2e3347;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    """Load trained model and test data (cached for performance)."""
    try:
        # Load data
        loader = CMAPSDataLoader()
        train_df = loader.load_train_data()
        test_df, true_rul = loader.load_test_data()

        # Feature engineering
        engineer = FeatureEngineer()
        engineer.fit_transform(train_df)
        test_processed = engineer.transform(test_df)

        # Load model
        predictor = RULPredictor()
        predictor.load_model("model.pkl")

        # Generate predictions for all test cycles
        test_processed['predicted_RUL'] = predictor.predict(test_processed)

        # Add true RUL for final cycles
        last_cycles = test_processed.groupby('unit_id').tail(1)
        unit_to_true_rul = dict(zip(range(1, len(true_rul) + 1), true_rul['RUL']))

        return test_processed, unit_to_true_rul, predictor

    except FileNotFoundError as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Please ensure model.pkl and data files are present.")
        st.stop()


def get_health_status(rul: float) -> tuple:
    """
    Determine health status based on RUL.

    Returns:
        tuple: (status_text, status_class)
    """
    if rul > 50:
        return "Healthy", "healthy"
    elif 20 <= rul <= 50:
        return "Monitor", "monitor"
    else:
        return "Critical", "critical"


def plot_degradation_curve(engine_data: pd.DataFrame, true_rul: float):
    """
    Plot RUL degradation over engine lifecycle.

    Args:
        engine_data (pd.DataFrame): All cycles for selected engine
        true_rul (float): Ground truth RUL at final cycle
    """
    fig = go.Figure()

    # Predicted RUL line
    fig.add_trace(go.Scatter(
        x=engine_data['cycle'],
        y=engine_data['predicted_RUL'],
        mode='lines',
        name='Predicted RUL',
        line=dict(color='#00C853', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 200, 83, 0.1)'
    ))

    # True RUL point at final cycle
    final_cycle = engine_data['cycle'].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[final_cycle],
        y=[true_rul],
        mode='markers',
        name='Actual RUL',
        marker=dict(color='#FF5722', size=12, symbol='diamond')
    ))

    # Critical threshold line
    fig.add_hline(
        y=20,
        line_dash="dash",
        line_color="#F44336",
        annotation_text="Critical Threshold",
        annotation_position="right"
    )

    fig.update_layout(
        title="Engine Degradation Curve",
        xaxis_title="Operational Cycle",
        yaxis_title="Remaining Useful Life (cycles)",
        template="plotly_dark",
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


def main():
    """Main dashboard application."""

    # Header
    st.title("✈️ Aircraft Engine Health Monitor")
    st.markdown("**Predictive Maintenance Dashboard** | Real-time RUL Analysis Using NASA C-MAPS Data")

    # Load data
    with st.spinner("Loading model and data..."):
        test_data, true_rul_dict, predictor = load_model_and_data()

    # Sidebar - Engine Selection
    st.sidebar.header("🔧 Engine Selection")

    available_engines = sorted(test_data['unit_id'].unique())
    selected_engine = st.sidebar.selectbox(
        "Select Engine Unit",
        available_engines,
        format_func=lambda x: f"Engine Unit {x}"
    )

    # Filter data for selected engine
    engine_data = test_data[test_data['unit_id'] == selected_engine].copy()

    # Get current (final cycle) RUL
    current_rul = engine_data['predicted_RUL'].iloc[-1]
    true_rul = true_rul_dict.get(selected_engine, 0)
    current_cycle = engine_data['cycle'].iloc[-1]

    # Sidebar - Engine Info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Engine Details")
    st.sidebar.metric("Total Cycles", current_cycle)
    st.sidebar.metric("Predicted RUL", f"{current_rul:.0f} cycles")
    st.sidebar.metric("Actual RUL", f"{true_rul:.0f} cycles")

    # Prediction error
    error = abs(current_rul - true_rul)
    st.sidebar.metric("Prediction Error", f"{error:.1f} cycles")

    st.sidebar.markdown("---")

    # Main content
    col1, col2, col3 = st.columns(3)

    # Health Status Badge
    status_text, status_class = get_health_status(current_rul)

    with col2:
        st.markdown(
            f'<div class="health-badge {status_class}">{status_text}</div>',
            unsafe_allow_html=True
        )

    # Metrics Row
    st.markdown("### 📈 Key Performance Indicators")

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.metric(
            label="Predicted RUL",
            value=f"{current_rul:.0f}",
            delta=None,
            help="Remaining cycles before maintenance required"
        )

    with metrics_col2:
        st.metric(
            label="Actual RUL",
            value=f"{true_rul:.0f}",
            delta=None,
            help="Ground truth from test data"
        )

    with metrics_col3:
        accuracy = (1 - error / max(true_rul, 1)) * 100
        st.metric(
            label="Prediction Accuracy",
            value=f"{max(0, accuracy):.1f}%",
            delta=None
        )

    with metrics_col4:
        st.metric(
            label="Operational Cycles",
            value=f"{current_cycle}",
            delta=None,
            help="Total cycles completed"
        )

    # Degradation Plot
    st.markdown("### 📉 Engine Health Over Time")
    degradation_fig = plot_degradation_curve(engine_data, true_rul)
    st.plotly_chart(degradation_fig, use_container_width=True)

    # Model Performance (Overall)
    st.markdown("### 🎯 Model Performance Metrics")

    # Calculate overall metrics
    all_predictions = []
    all_actuals = []

    for unit_id in available_engines:
        unit_data = test_data[test_data['unit_id'] == unit_id]
        pred_rul = unit_data['predicted_RUL'].iloc[-1]
        actual_rul = true_rul_dict.get(unit_id, 0)

        all_predictions.append(pred_rul)
        all_actuals.append(actual_rul)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    overall_rmse = np.sqrt(np.mean((all_predictions - all_actuals) ** 2))
    overall_r2 = 1 - (np.sum((all_actuals - all_predictions) ** 2) /
                      np.sum((all_actuals - np.mean(all_actuals)) ** 2))

    perf_col1, perf_col2, perf_col3 = st.columns(3)

    with perf_col1:
        st.metric("RMSE", f"{overall_rmse:.2f} cycles", help="Root Mean Squared Error")

    with perf_col2:
        st.metric("R² Score", f"{overall_r2:.4f}", help="Coefficient of Determination")

    with perf_col3:
        st.metric("Engines Analyzed", len(available_engines))

    # Footer
    st.markdown("""
        <div class="footer">
            <strong>Powered by NASA C-MAPS Dataset</strong><br>
            Commercial Modular Aero-Propulsion System Simulation (FD001 Subset)<br>
            Developed for Predictive Maintenance in Aviation MRO Operations
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
