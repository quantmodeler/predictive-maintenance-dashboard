import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import time
from sklearn.ensemble import RandomForestRegressor
import os
import runpy

# Check if model exists, if not, train it
if not os.path.exists('rul_model.pkl'):
    st.info("🔄 Training model for first time use... This may take a minute.")
    with st.spinner('Training in progress...'):
        try:
            runpy.run_path('train_model.py')
            st.success("✅ Model trained successfully!")
        except Exception as _e:
            st.error(f"❌ Model training failed: {_e}")
            st.stop()

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

# -------------------------------
# Load the trained model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load('rul_model.pkl')

model = load_model()

# -------------------------------
# Load test data for simulation
# -------------------------------
@st.cache_data
def load_test_data():
    # Use regex '\s+' to handle multiple spaces
    test = pd.read_csv('data/test_FD001.txt', sep='\s+', header=None)
    # Keep only the first 26 columns (in case of extra empty columns)
    if test.shape[1] > 26:
        test = test.iloc[:, :26]
    # Assign proper column names
    columns = ['engine', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1,22)]
    test.columns = columns
    return test

test_data = load_test_data()

# -------------------------------
# Get a list of unique engine IDs from test set
# -------------------------------
engines = test_data['engine'].unique()
engine_ids = sorted(engines)

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Settings")
# Engine selection
selected_engine = st.sidebar.selectbox("Select Engine ID", engine_ids, index=0)
# Alert thresholds
st.sidebar.subheader("Alert Thresholds")
temp_thresh = st.sidebar.slider("Temperature (sensor 2)", 500, 700, 620, help="Threshold for high temperature alarm")
vib_thresh = st.sidebar.slider("Vibration (sensor 3)", 20, 50, 35, help="Threshold for high vibration alarm")
press_thresh = st.sidebar.slider("Pressure (sensor 7)", 10, 40, 25, help="Threshold for high pressure alarm")
# Simulation speed
update_interval = st.sidebar.slider("Update interval (seconds)", 1, 5, 2)

# -------------------------------
# Prepare data for the selected engine
# -------------------------------
engine_data = test_data[test_data['engine'] == selected_engine].copy()
# Compute true RUL for the test set (if available) - for this simulation we'll use a simplified approach:
# We'll pretend that the last cycle of the engine is the end of life, and RUL decreases as cycles increase.
max_cycle_engine = engine_data['cycle'].max()
engine_data['RUL_true'] = max_cycle_engine - engine_data['cycle']

# We'll iterate through the cycles for this engine
total_cycles = len(engine_data)

# -------------------------------
# Initialize session state to keep track of cycle
# -------------------------------
if 'cycle_index' not in st.session_state:
    st.session_state.cycle_index = 0
    st.session_state.engine = selected_engine

# Reset if engine selection changes
if st.session_state.engine != selected_engine:
    st.session_state.cycle_index = 0
    st.session_state.engine = selected_engine

# -------------------------------
# Main dashboard area
# -------------------------------
st.title("🔧 Predictive Maintenance Simulator")
st.markdown(f"**Engine ID:** {selected_engine}  |  **Cycle:** {st.session_state.cycle_index + 1} / {total_cycles}")

# Placeholders for dynamic content
col1, col2, col3 = st.columns(3)
gauge_temp = col1.empty()
gauge_vib = col2.empty()
gauge_press = col3.empty()

pred_col, conf_col = st.columns(2)
pred_placeholder = pred_col.empty()
conf_placeholder = conf_col.empty()

alert_placeholder = st.empty()
failure_placeholder = st.empty()

# Button to manually advance (if user wants to step through)
if st.sidebar.button("Next Cycle"):
    st.session_state.cycle_index = (st.session_state.cycle_index + 1) % total_cycles


# Get current cycle data
current_row = engine_data.iloc[st.session_state.cycle_index]
sensors = current_row[[f'sensor{i}' for i in range(1,22)]].values.reshape(1, -1)

# Predict RUL
pred_rul = model.predict(sensors)[0]

# Simulate confidence interval (e.g., ±10% of predicted RUL)
ci_lower = max(0, pred_rul * 0.9)
ci_upper = pred_rul * 1.1

# Extract key sensor readings for display (choose appropriate indices from the 21 sensors)
# For demonstration, we map:
# - Temperature: sensor 2  (index 1 in zero-based)
# - Vibration:   sensor 3  (index 2)
# - Pressure:    sensor 7  (index 6)
temp = current_row['sensor2']
vib = current_row['sensor3']
press = current_row['sensor7']

# Check alerts
alerts = []
if temp > temp_thresh:
    alerts.append("🔥 High Temperature")
if vib > vib_thresh:
    alerts.append("⚠️ High Vibration")
if press > press_thresh:
    alerts.append("🌀 High Pressure")

# Display gauges using Plotly
def create_gauge(value, title, min_val, max_val, threshold):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': threshold},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, threshold], 'color': "lightgreen"},
                {'range': [threshold, max_val], 'color': "lightcoral"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold}}))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    return fig

with gauge_temp:
    st.plotly_chart(create_gauge(temp, "Temperature (°F)", 500, 700, temp_thresh), use_container_width=True)
with gauge_vib:
    st.plotly_chart(create_gauge(vib, "Vibration (g)", 20, 50, vib_thresh), use_container_width=True)
with gauge_press:
    st.plotly_chart(create_gauge(press, "Pressure (psi)", 10, 40, press_thresh), use_container_width=True)

# Display RUL prediction with confidence
pred_placeholder.metric("Predicted RUL", f"{pred_rul:.0f} cycles")
conf_placeholder.write(f"**Confidence Interval:** {ci_lower:.0f} – {ci_upper:.0f} cycles")

# Show alerts
if alerts:
    alert_placeholder.error(" | ".join(alerts))
else:
    alert_placeholder.success("All systems normal")

# Failure mode classification (simplified rule-based)
if pred_rul < 20:
    failure_placeholder.warning("🛑 **Critical**: Approaching failure – possible causes: High-pressure turbine (HPT) degradation")
elif pred_rul < 40:
    failure_placeholder.info("⚠️ **Degraded**: Moderate degradation – check fan and compressor")
elif pred_rul < 60:
    failure_placeholder.info("🔧 **Early wear**: Some degradation observed – schedule maintenance")
else:
    failure_placeholder.success("✅ **Healthy**: No significant degradation")

# Optional: Show raw sensor values in an expander
with st.expander("View all sensor readings"):
    st.dataframe(current_row[[f'sensor{i}' for i in range(1,22)]].to_frame().T)

# Add a simple trend chart of RUL over recent cycles
st.subheader("RUL Trend")
history_length = min(20, st.session_state.cycle_index + 1)
start_idx = max(0, st.session_state.cycle_index - history_length + 1)
history_data = engine_data.iloc[start_idx:st.session_state.cycle_index+1].copy()
# For each historical cycle, we need the predicted RUL (re-predict using model)
# To avoid recomputation, we could store predictions, but for simplicity we'll recompute here.
history_data['pred_RUL'] = history_data[[f'sensor{i}' for i in range(1,22)]].apply(
    lambda row: model.predict([row.values])[0], axis=1
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=history_data['cycle'], y=history_data['pred_RUL'],
                         mode='lines+markers', name='Predicted RUL'))
fig.add_trace(go.Scatter(x=history_data['cycle'], y=history_data['RUL_true'],
                         mode='lines', name='True RUL (simulated)', line=dict(dash='dash')))
fig.update_layout(title='RUL over recent cycles', xaxis_title='Cycle', yaxis_title='RUL')
st.plotly_chart(fig, use_container_width=True)
# Auto-refresh logic
if st.sidebar.checkbox("Auto refresh", value=True):
    time.sleep(update_interval)
    st.session_state.cycle_index = (st.session_state.cycle_index + 1) % total_cycles
    st.rerun()