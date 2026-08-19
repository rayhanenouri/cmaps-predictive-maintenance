"""Engine health monitoring dashboard for NASA C-MAPS RUL predictions."""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')
from data_loader import load_data

st.set_page_config(
    page_title="Engine Health Monitor",
    page_icon="✈",
    layout="wide"
)

st.markdown("""
    <style>
    /* Remove all Streamlit default styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Color system */
    :root {
        --bg-primary: #050810;
        --bg-surface: #0c1220;
        --bg-sidebar: #080d18;
        --border-color: #1e2d45;
        --text-primary: #e8eef4;
        --text-secondary: #6b7f94;
        --accent-blue: #1e90ff;
        --success: #00875a;
        --warning: #ff8c00;
        --danger: #d32f2f;
        --grid-color: #0f1a2e;
    }

    /* Main app background */
    .stApp {
        background-color: var(--bg-primary) !important;
    }

    .main {
        background-color: var(--bg-primary) !important;
        padding: 0 !important;
    }

    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* Typography */
    * {
        font-family: system-ui, -apple-system, sans-serif !important;
    }

    /* Header - full width black bar */
    .enterprise-header {
        background-color: #000000;
        padding: 16px 32px;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
    }

    .header-title {
        color: var(--text-primary);
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 2px;
        margin: 0;
    }

    .system-status {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--success);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: var(--success);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-color) !important;
        padding: 24px 16px !important;
    }

    .sidebar-title {
        color: var(--text-secondary);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-color);
    }

    .sidebar-section {
        margin: 24px 0;
        padding-bottom: 24px;
        border-bottom: 1px solid var(--border-color);
    }

    .sidebar-metric-label {
        color: var(--text-secondary);
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 4px;
    }

    .sidebar-metric-value {
        color: var(--text-primary);
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 16px;
    }

    /* KPI Status Boxes */
    .kpi-box {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 2px;
        padding: 20px 24px;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .kpi-label {
        color: var(--text-secondary);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .kpi-value {
        color: var(--text-primary);
        font-size: 32px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .kpi-value.success { color: var(--success); }
    .kpi-value.warning { color: var(--warning); }
    .kpi-value.danger { color: var(--danger); }

    /* Section headers */
    .section-title {
        color: var(--text-secondary);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 32px 0 16px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-color);
    }

    /* Chart containers */
    .chart-container {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 2px;
        padding: 20px;
        margin-bottom: 24px;
    }

    .chart-title {
        color: var(--text-secondary);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 16px;
        padding-left: 10px;
    }

    /* Fleet table styling */
    .fleet-table {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-color);
        border-radius: 2px;
        padding: 20px;
        margin-bottom: 24px;
    }

    /* Override Streamlit table styling */
    [data-testid="stDataFrame"] {
        background-color: var(--bg-surface) !important;
    }

    [data-testid="stDataFrame"] table {
        background-color: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        font-size: 12px !important;
    }

    [data-testid="stDataFrame"] thead tr th {
        background-color: #0a0f1c !important;
        color: var(--text-secondary) !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        border-bottom: 1px solid var(--border-color) !important;
        padding: 12px 16px !important;
    }

    [data-testid="stDataFrame"] tbody tr td {
        border-bottom: 1px solid #0f1521 !important;
        padding: 10px 16px !important;
        color: var(--text-primary) !important;
    }

    /* Footer */
    .enterprise-footer {
        text-align: center;
        padding: 24px;
        color: #2d3f54;
        font-size: 10px;
        letter-spacing: 1px;
        border-top: 1px solid var(--border-color);
        margin-top: 40px;
    }

    /* Override Streamlit selectbox */
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--bg-surface) !important;
        border-color: var(--border-color) !important;
    }

    /* Content padding */
    .content-wrapper {
        padding: 0 32px 32px 32px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_scaler():
    """load persisted model and scaler from training run"""
    try:
        models_dir = Path('models')
        with open(models_dir / 'xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model files not found. Please run training first.")
        st.stop()

@st.cache_data
def load_test_dataset():
    """run feature pipeline and generate per-engine predictions"""
    try:
        _, test_df, test_rul_df = load_data('data/')
        model, scaler = load_model_and_scaler()

        from feature_engineering import prepare_features
        X_train, y_train, X_test, test_rul, _, feature_names = prepare_features('data/')

        all_predictions = model.predict(X_test)
        all_predictions = np.maximum(all_predictions, 0)
        test_df['predicted_RUL'] = all_predictions

        # Get predictions per engine
        engine_predictions = {}
        engine_actuals = {}
        engine_cycles = {}
        current_idx = 0

        for unit_id in sorted(test_df['unit_id'].unique()):
            unit_mask = test_df['unit_id'] == unit_id
            unit_cycle_count = unit_mask.sum()

            last_pred = all_predictions[current_idx + unit_cycle_count - 1]
            engine_predictions[unit_id] = last_pred

            engine_idx = list(sorted(test_df['unit_id'].unique())).index(unit_id)
            engine_actuals[unit_id] = test_rul[engine_idx]
            engine_cycles[unit_id] = unit_cycle_count

            current_idx += unit_cycle_count

        return test_df, engine_predictions, engine_actuals, engine_cycles

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def get_status_class(rul):
    """map predicted RUL to operational status tier"""
    if rul > 50:
        return "HEALTHY", "success"
    elif 20 <= rul <= 50:
        return "MONITOR", "warning"
    else:
        return "CRITICAL", "danger"

def plot_sensor_trends(engine_data):
    """plot normalized sensor readings over flight cycles"""
    fig = go.Figure()

    sensors = ['sensor_2', 'sensor_3', 'sensor_4']
    colors = ['#1e90ff', '#4db8ff', '#7dd3ff']

    for sensor, color in zip(sensors, colors):
        if sensor in engine_data.columns:
            # smooth sensor noise before plotting
            values = engine_data[sensor].rolling(window=5, min_periods=1).mean().values
            normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)

            fig.add_trace(go.Scatter(
                x=engine_data['cycle'],
                y=normalized,
                mode='lines',
                name=sensor.upper().replace('_', ' '),
                line=dict(color=color, width=2),
                hovertemplate='Cycle: %{x}<br>Value: %{y:.3f}<extra></extra>'
            ))

    fig.update_layout(
        plot_bgcolor='#0c1220',
        paper_bgcolor='#0c1220',
        font=dict(color='#6b7f94', size=10, family='system-ui'),
        height=350,
        margin=dict(l=40, r=20, t=10, b=40),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9, color='#6b7f94')
        ),
        xaxis=dict(
            gridcolor='#0f1a2e',
            gridwidth=1,
            showline=False,
            zeroline=False,
            tickfont=dict(size=9, color='#6b7f94'),
            title=dict(text="Flight Cycles", font=dict(size=10, color='#6b7f94'))
        ),
        yaxis=dict(
            gridcolor='#0f1a2e',
            gridwidth=1,
            showline=False,
            zeroline=False,
            tickfont=dict(size=9, color='#6b7f94'),
            title=dict(text="Normalized", font=dict(size=10, color='#6b7f94'))
        )
    )

    return fig

def plot_degradation_trajectory(engine_data, predicted_rul):
    """estimated RUL trajectory from current cycle to end of life"""
    fig = go.Figure()

    # Calculate RUL at each cycle (reverse order from current)
    cycles = engine_data['cycle'].values
    max_cycle = cycles.max()

    # Create trajectory (predicted RUL decreases linearly for visualization)
    rul_trajectory = []
    for cycle in cycles:
        remaining_cycles = max_cycle - cycle
        trajectory_rul = predicted_rul + remaining_cycles
        rul_trajectory.append(trajectory_rul)

    # Color segments based on thresholds
    colors = []
    for rul in rul_trajectory:
        if rul > 50:
            colors.append('#00875a')  # Green
        elif 20 <= rul <= 50:
            colors.append('#ff8c00')  # Amber
        else:
            colors.append('#d32f2f')  # Red

    fig.add_trace(go.Scatter(
        x=cycles,
        y=rul_trajectory,
        mode='lines',
        line=dict(color='#1e90ff', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(30, 144, 255, 0.1)',
        hovertemplate='Cycle: %{x}<br>RUL: %{y:.0f}<extra></extra>'
    ))

    # Threshold lines
    fig.add_hline(y=50, line_dash="dot", line_color="#00875a", line_width=1,
                  annotation_text="HEALTHY THRESHOLD", annotation_position="right",
                  annotation=dict(font=dict(size=8, color='#6b7f94')))
    fig.add_hline(y=20, line_dash="dot", line_color="#d32f2f", line_width=1,
                  annotation_text="CRITICAL THRESHOLD", annotation_position="right",
                  annotation=dict(font=dict(size=8, color='#6b7f94')))

    fig.update_layout(
        plot_bgcolor='#0c1220',
        paper_bgcolor='#0c1220',
        font=dict(color='#6b7f94', size=10, family='system-ui'),
        height=350,
        margin=dict(l=40, r=20, t=10, b=40),
        showlegend=False,
        xaxis=dict(
            gridcolor='#0f1a2e',
            gridwidth=1,
            showline=False,
            zeroline=False,
            tickfont=dict(size=9, color='#6b7f94'),
            title=dict(text="Flight Cycles", font=dict(size=10, color='#6b7f94'))
        ),
        yaxis=dict(
            gridcolor='#0f1a2e',
            gridwidth=1,
            showline=False,
            zeroline=False,
            tickfont=dict(size=9, color='#6b7f94'),
            title=dict(text="Remaining Useful Life (cycles)", font=dict(size=10, color='#6b7f94'))
        )
    )

    return fig

def main():
    """entry point"""

    # Header
    st.markdown("""
        <div class="enterprise-header">
            <div class="header-title">ENGINE HEALTH MONITOR</div>
            <div class="system-status">
                <div class="status-dot"></div>
                SYSTEM ONLINE
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading system..."):
        test_df, engine_predictions, engine_actuals, engine_cycles = load_test_dataset()

    # Sidebar
    st.sidebar.markdown('<div class="sidebar-title">CONTROL PANEL</div>', unsafe_allow_html=True)

    available_engines = sorted(engine_predictions.keys())
    selected_engine = st.sidebar.selectbox(
        "Select Engine",
        available_engines,
        format_func=lambda x: f"Engine {x:03d}",
        label_visibility="collapsed"
    )

    st.sidebar.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)

    # Sidebar metrics
    st.sidebar.markdown('<div class="sidebar-metric-label">RMSE</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-metric-value">18.90 cycles</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-metric-label">R2 SCORE</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-metric-value">0.78</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-metric-label">ACCURACY (+/-20)</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-metric-value">77%</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-section"></div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sidebar-metric-label">DATASET</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div style="color: #6b7f94; font-size: 11px; line-height: 1.6;">NASA C-MAPS FD001<br>100 engines<br>21 sensors</div>', unsafe_allow_html=True)

    # Main content wrapper
    st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

    # Get selected engine data
    engine_data = test_df[test_df['unit_id'] == selected_engine].copy()
    predicted_rul = engine_predictions[selected_engine]
    actual_rul = engine_actuals[selected_engine]
    total_cycles = engine_cycles[selected_engine]
    error = abs(predicted_rul - actual_rul)

    status_text, status_class = get_status_class(predicted_rul)

    # ROW 1: KPI Status Boxes
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">ENGINE STATUS</div>
                <div class="kpi-value {status_class}">{status_text}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">PREDICTED RUL</div>
                <div class="kpi-value">{predicted_rul:.0f} CYC</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">ACTUAL RUL</div>
                <div class="kpi-value">{actual_rul:.0f} CYC</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">PREDICTION ERROR</div>
                <div class="kpi-value">{error:.1f} CYC</div>
            </div>
        """, unsafe_allow_html=True)

    # ROW 2: Charts
    st.markdown('<div style="margin-top: 32px;"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">SENSOR TREND ANALYSIS</div>', unsafe_allow_html=True)
        sensor_fig = plot_sensor_trends(engine_data)
        st.plotly_chart(sensor_fig, use_container_width=True, config={'displayModeBar': False, 'displaylogo': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">DEGRADATION TRAJECTORY</div>', unsafe_allow_html=True)
        trajectory_fig = plot_degradation_trajectory(engine_data, predicted_rul)
        st.plotly_chart(trajectory_fig, use_container_width=True, config={'displayModeBar': False, 'displaylogo': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ROW 3: Fleet Status Table
    st.markdown('<div style="margin-top: 32px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="fleet-table">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">FLEET STATUS OVERVIEW</div>', unsafe_allow_html=True)

    # Build fleet dataframe
    fleet_data = []
    for engine_id in sorted(engine_predictions.keys()):
        pred_rul = engine_predictions[engine_id]
        cycles_run = engine_cycles[engine_id]
        status, _ = get_status_class(pred_rul)

        fleet_data.append({
            'Engine ID': f'{engine_id:03d}',
            'Predicted RUL': f'{pred_rul:.0f}',
            'Status': status,
            'Cycles Run': cycles_run
        })

    fleet_df = pd.DataFrame(fleet_data)

    # Display table with custom styling
    st.dataframe(
        fleet_df,
        use_container_width=True,
        height=400,
        hide_index=True
    )

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # Close content wrapper

    # Footer
    st.markdown("""
        <div class="enterprise-footer">
            ENGINE HEALTH MONITOR v1.0 | NASA C-MAPS FD001 | XGBOOST REGRESSION
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
