import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import time
from sklearn.ensemble import RandomForestRegressor
import os

# Import failure mode classifier
from failure_modes import classify_failure_mode, get_failure_mode_icon

# Debug: Check if import worked
st.sidebar.write("✅ Failure modes module loaded")

# Check if model exists, if not, train it directly
if not os.path.exists('rul_model.pkl'):
    st.info("🔄 Training model for first time use... This may take a minute.")
    with st.spinner('Training in progress...'):
        try:
            # Load training data
            train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None)
            if train.shape[1] > 26:
                train = train.iloc[:, :26]
            
            # Assign column names
            columns = ['engine', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1,22)]
            train.columns = columns
            
            # Convert to numeric
            for col in train.columns:
                train[col] = pd.to_numeric(train[col], errors='coerce').fillna(0)
            
            # Compute RUL
            max_cycles = train.groupby('engine')['cycle'].max().reset_index()
            max_cycles.columns = ['engine', 'max_cycle']
            train = train.merge(max_cycles, on='engine')
            train['RUL'] = train['max_cycle'] - train['cycle']
            
            # Train model
            feature_columns = [f'sensor{i}' for i in range(1,22)]
            X = train[feature_columns]
            y = train['RUL']
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            # Save model
            joblib.dump(model, 'rul_model.pkl')
            st.success("✅ Model trained successfully!")
            
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")

# Load quantile models
@st.cache_resource
def load_models():
    return joblib.load('quantile_models.pkl')

models = load_models()

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
    test = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None)
    # Keep only the first 26 columns
    if test.shape[1] > 26:
        test = test.iloc[:, :26]
    
    # Assign column names
    columns = ['engine', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1,22)]
    test.columns = columns
    
    # Convert EVERY column to numeric, forcing errors to NaN then filling with 0
    for col in test.columns:
        test[col] = pd.to_numeric(test[col], errors='coerce').fillna(0)
    
    # Ensure engine and cycle are integers
    test['engine'] = test['engine'].astype(int)
    test['cycle'] = test['cycle'].astype(int)
    
    return test

test_data = load_test_data()

# Debug: Check data types (will appear in sidebar)
st.sidebar.write("Data types loaded:")
st.sidebar.write(test_data[['sensor1', 'sensor2', 'sensor3']].dtypes)

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
# Compute true RUL for the test set
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

# Button to manually advance
if st.sidebar.button("Next Cycle"):
    st.session_state.cycle_index = (st.session_state.cycle_index + 1) % total_cycles

# Get current cycle data
current_row = engine_data.iloc[st.session_state.cycle_index]
sensors = current_row[[f'sensor{i}' for i in range(1,22)]].values.reshape(1, -1)

# Convert sensors to DataFrame for prediction and ensure numeric type
sensors_df = pd.DataFrame([sensors[0]], columns=[f'sensor{i}' for i in range(1,22)])
sensors_df = sensors_df.astype(np.float64)

# Predict RUL with confidence intervals using quantile models
pred_lower = models['q10'].predict(sensors_df)[0]
pred_median = models['q50'].predict(sensors_df)[0]
pred_upper = models['q90'].predict(sensors_df)[0]

# Ensure non-negative predictions
pred_lower = max(0, pred_lower)
pred_median = max(0, pred_median)
pred_upper = max(0, pred_upper)

# Extract key sensor readings for display
temp = current_row['sensor2']
vib = current_row['sensor3']
press = current_row['sensor7']

# Use median for the main display, store all for confidence interval
pred_rul = pred_median
ci_lower = pred_lower
ci_upper = pred_upper

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
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        delta={'reference': threshold},
        gauge={
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

# Debug: Check if failure mode function is being called
st.sidebar.write("Calling failure mode classifier...")

# Sophisticated failure mode classification
failure_modes = classify_failure_mode(current_row, pred_rul, alerts, engine_history=None)

# Debug: Show what was returned
st.sidebar.write(f"Returned {len(failure_modes) if failure_modes else 0} failure modes")
if failure_modes:
    st.sidebar.write("First mode:", failure_modes[0])

# Force a test message to appear in main area
st.info("🔧 Test: If you see this, the app is running. Failure modes should appear below this line.")

# Display failure modes
if failure_modes:
    # First, clear the placeholder
    failure_placeholder.empty()
    
    # Create a nice display
    with failure_placeholder.container():
        for mode in failure_modes:
            if "IMMEDIATE" in mode:
                st.error(f"🚨 **{mode}**")
            elif "Schedule" in mode or "Plan" in mode:
                st.warning(f"📅 **{mode}**")
            elif "High Confidence" in mode:
                icon = mode.split()[0] if mode.split()[0] in ["🔥", "⚙️", "🌀", "⚠️", "💨", "📉"] else "🔧"
                st.error(f"{icon} **{mode}**")
            elif "Medium Confidence" in mode:
                icon = mode.split()[0] if mode.split()[0] in ["🔥", "⚙️", "🌀", "⚠️", "💨", "📉"] else "🔧"
                st.warning(f"{icon} **{mode}**")
            elif "Low Confidence" in mode:
                icon = mode.split()[0] if mode.split()[0] in ["🔥", "⚙️", "🌀", "⚠️", "💨", "📉"] else "🔧"
                st.info(f"{icon} **{mode}**")
            elif "Action items" in mode:
                st.info(f"💡 {mode}")
            else:
                st.success(f"✅ {mode}")
else:
    failure_placeholder.success("✅ **All systems operating normally**")

# Optional: Show raw sensor values in an expander
with st.expander("View all sensor readings"):
    st.dataframe(current_row[[f'sensor{i}' for i in range(1,22)]].to_frame().T)

# Add a simple trend chart of RUL over recent cycles
st.subheader("RUL Trend")
history_length = min(20, st.session_state.cycle_index + 1)
start_idx = max(0, st.session_state.cycle_index - history_length + 1)
history_data = engine_data.iloc[start_idx:st.session_state.cycle_index+1].copy()
# For each historical cycle, we need the predicted RUL
history_data['pred_RUL'] = history_data[[f'sensor{i}' for i in range(1,22)]].apply(
    lambda row: model.predict([row.values.astype(np.float64)])[0], axis=1
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=history_data['cycle'], y=history_data['pred_RUL'],
                         mode='lines+markers', name='Predicted RUL'))
fig.add_trace(go.Scatter(x=history_data['cycle'], y=history_data['RUL_true'],
                         mode='lines', name='True RUL (simulated)', line=dict(dash='dash')))
fig.update_layout(title='RUL over recent cycles', xaxis_title='Cycle', yaxis_title='RUL')
st.plotly_chart(fig, use_container_width=True)

# ============================================
# MULTI-ENGINE COMPARISON SECTION
# ============================================
st.markdown("---")
st.header("🔍 Multi-Engine Comparison")

# Select multiple engines
comparison_engines = st.multiselect(
    "Select engines to compare",
    options=engine_ids,
    default=[engine_ids[0]] if len(engine_ids) >= 1 else engine_ids,
    key="comparison_selector"
)

if comparison_engines and len(comparison_engines) > 0:
    # Create comparison data
    comparison_data = []
    
    with st.spinner("Loading comparison data..."):
        for engine_id in comparison_engines:
            # Get data for this engine
            engine_subset = test_data[test_data['engine'] == engine_id].copy()
            
            # Get last 30 cycles or all if less
            recent = engine_subset.tail(min(30, len(engine_subset)))
            
            if len(recent) > 0:
                # Predict RUL for each cycle
                predictions = []
                for idx, row in recent.iterrows():
                    sensor_row = row[[f'sensor{i}' for i in range(1,22)]].values.reshape(1, -1)
                    sensor_df = pd.DataFrame(sensor_row, columns=[f'sensor{i}' for i in range(1,22)])
                    sensor_df = sensor_df.astype(np.float64)
                    pred = models['q50'].predict(sensor_df)[0]
                    predictions.append(max(0, pred))
                
                recent['pred_RUL'] = predictions
                recent['engine_label'] = f"Engine {engine_id}"
                comparison_data.append(recent)
    
    if comparison_data:
        comparison_df = pd.concat(comparison_data)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 RUL Trends", "📊 Current Status", "📋 Detailed Data"])
        
        with tab1:
            # RUL Trend Comparison Chart
            fig_comp = go.Figure()
            
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
            for i, engine_id in enumerate(comparison_engines):
                engine_df = comparison_df[comparison_df['engine'] == engine_id]
                color = colors[i % len(colors)]
                
                # Main RUL line
                fig_comp.add_trace(go.Scatter(
                    x=engine_df['cycle'],
                    y=engine_df['pred_RUL'],
                    mode='lines+markers',
                    name=f'Engine {engine_id}',
                    line=dict(color=color, width=2),
                    marker=dict(size=6)
                ))
                
                # Add confidence band
                if 'q10' in models and 'q90' in models:
                    lower_bounds = []
                    upper_bounds = []
                    for idx, row in engine_df.iterrows():
                        sensor_row = row[[f'sensor{i}' for i in range(1,22)]].values.reshape(1, -1)
                        sensor_df = pd.DataFrame(sensor_row, columns=[f'sensor{i}' for i in range(1,22)])
                        sensor_df = sensor_df.astype(np.float64)
                        lower = max(0, models['q10'].predict(sensor_df)[0])
                        upper = max(0, models['q90'].predict(sensor_df)[0])
                        lower_bounds.append(lower)
                        upper_bounds.append(upper)
                    
                    fig_comp.add_trace(go.Scatter(
                        x=pd.concat([engine_df['cycle'], engine_df['cycle'][::-1]]),
                        y=pd.concat([pd.Series(upper_bounds), pd.Series(lower_bounds)[::-1]]),
                        fill='toself',
                        fillcolor=f'rgba({i*40},{255-i*40},100,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'Engine {engine_id} 90% CI',
                        showlegend=True
                    ))
            
            fig_comp.update_layout(
                title='RUL Comparison Across Engines',
                xaxis_title='Cycle Number',
                yaxis_title='Predicted Remaining Useful Life (cycles)',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Summary statistics
            st.subheader("📊 Summary Statistics")
            summary = comparison_df.groupby('engine').agg({
                'pred_RUL': ['last', 'min', 'mean', 'max'],
                'cycle': 'max'
            }).round(0)
            summary.columns = ['Current RUL', 'Min RUL', 'Avg RUL', 'Max RUL', 'Total Cycles']
            summary = summary.reset_index()
            summary['engine'] = summary['engine'].astype(int)
            summary = summary.set_index('engine')
            
            # Add color coding based on health status
            def color_rul(val):
                if val < 30:
                    return 'background-color: #ffcccc'
                elif val < 60:
                    return 'background-color: #ffffcc'
                else:
                    return 'background-color: #ccffcc'
            
            styled_summary = summary.style.map(color_rul, subset=['Current RUL'])
            st.dataframe(styled_summary, use_container_width=True)
        
        with tab2:
            # Current Status Dashboard
            st.subheader("📍 Current Engine Status")
            
            cols = st.columns(min(3, len(comparison_engines)))
            for idx, engine_id in enumerate(comparison_engines):
                with cols[idx % 3]:
                    engine_current = comparison_df[comparison_df['engine'] == engine_id].iloc[-1]
                    
                    # Create metric cards
                    st.metric(
                        label=f"Engine {engine_id}",
                        value=f"{engine_current['pred_RUL']:.0f} cycles",
                        delta=None
                    )
                    
                    # Sensor gauges in small
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Temp", f"{engine_current['sensor2']:.0f}°F")
                    col2.metric("Vib", f"{engine_current['sensor3']:.0f}g")
                    col3.metric("Press", f"{engine_current['sensor7']:.0f}psi")
                    
                    # Health indicator with failure mode
                    if engine_current['pred_RUL'] < 30:
                        st.error("🔴 CRITICAL")
                        # Show quick failure mode for this engine
                        quick_modes = classify_failure_mode(engine_current, engine_current['pred_RUL'], [], None)
                        if quick_modes:
                            st.caption(f"⚠️ {quick_modes[0]}")
                    elif engine_current['pred_RUL'] < 60:
                        st.warning("🟡 WARNING")
                        quick_modes = classify_failure_mode(engine_current, engine_current['pred_RUL'], [], None)
                        if quick_modes:
                            st.caption(f"📋 {quick_modes[0]}")
                    else:
                        st.success("🟢 HEALTHY")
                    
                    st.divider()
        
        with tab3:
            # Detailed data table
            st.subheader("📋 Detailed Sensor Data")
            
            display_cols = ['engine', 'cycle', 'pred_RUL', 'sensor2', 'sensor3', 'sensor7']
            display_df = comparison_df[display_cols].copy()
            display_df.columns = ['Engine', 'Cycle', 'RUL', 'Temp', 'Vib', 'Press']
            display_df = display_df.sort_values(['Engine', 'Cycle'], ascending=[True, False])
            
            st.dataframe(display_df, use_container_width=True)
            
            # Download button
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Comparison Data (CSV)",
                data=csv,
                file_name=f"engine_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No data available for selected engines")
else:
    st.info("👆 Select at least one engine from the dropdown above to start comparison")

# Auto-refresh logic (keep this at the very end)
if st.sidebar.checkbox("Auto refresh", value=True):
    time.sleep(update_interval)
    st.session_state.cycle_index = (st.session_state.cycle_index + 1) % total_cycles
    st.rerun()