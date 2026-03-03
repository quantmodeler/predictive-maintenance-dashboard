"""
Sophisticated failure mode classification based on sensor patterns
"""

def classify_failure_mode(sensors, pred_rul, alerts, engine_history=None):
    """
    Classify failure mode based on sensor readings and patterns
    
    Parameters:
    - sensors: dict-like object with sensor readings
    - pred_rul: predicted remaining useful life
    - alerts: list of active alerts
    - engine_history: optional DataFrame with historical data
    
    Returns:
    - List of failure modes with confidence levels
    """
    
    # Extract key sensors (using the actual sensor mappings from NASA dataset)
    sensor_data = {
        'T24': sensors.get('sensor2', 0),      # Total temperature at LPC outlet
        'T30': sensors.get('sensor3', 0),      # Total temperature at HPC outlet
        'T50': sensors.get('sensor4', 0),      # Total temperature at LPT outlet
        'P30': sensors.get('sensor7', 0),      # Total pressure at HPC outlet
        'Ps30': sensors.get('sensor6', 0),     # Static pressure at HPC outlet
        'phi': sensors.get('sensor1', 0),      # Fuel flow
        'NRf': sensors.get('sensor8', 0),      # Physical fan speed
        'NRc': sensors.get('sensor9', 0),      # Physical core speed
        'BPR': sensors.get('sensor10', 0),     # Bypass ratio
        'W31': sensors.get('sensor15', 0),     # HPT coolant bleed
        'W32': sensors.get('sensor16', 0),     # LPT coolant bleed
    }
    
    failure_modes = []
    confidences = []
    
    # -------------------------------------------------
    # 1. HIGH PRESSURE TURBINE (HPT) DEGRADATION
    # -------------------------------------------------
    # Indicators: High T50, high W31, low P30, high fuel flow
    hpt_score = 0
    if sensor_data['T50'] > 1550:
        hpt_score += 30
    if sensor_data['W31'] > 25:
        hpt_score += 25
    if sensor_data['P30'] < 350:
        hpt_score += 20
    if sensor_data['phi'] > 450:
        hpt_score += 15
    if pred_rul < 50:
        hpt_score += 10
    
    if hpt_score >= 50:
        failure_modes.append(("🔥 HPT Degradation", hpt_score))
    
    # -------------------------------------------------
    # 2. LOW PRESSURE TURBINE (LPT) DEGRADATION
    # -------------------------------------------------
    # Indicators: High T50, low BPR, high W32
    lpt_score = 0
    if sensor_data['T50'] > 1570:
        lpt_score += 35
    if sensor_data['BPR'] < 7.5:
        lpt_score += 30
    if sensor_data['W32'] > 35:
        lpt_score += 25
    if pred_rul < 40:
        lpt_score += 10
    
    if lpt_score >= 50:
        failure_modes.append(("⚙️ LPT Degradation", lpt_score))
    
    # -------------------------------------------------
    # 3. COMPRESSOR FOULING / EFFICIENCY LOSS
    # -------------------------------------------------
    # Indicators: High T30, low P30, high NRc
    compressor_score = 0
    if sensor_data['T30'] > 1550:
        compressor_score += 30
    if sensor_data['P30'] < 330:
        compressor_score += 35
    if sensor_data['NRc'] > 9500:
        compressor_score += 25
    if 'pressure' in str(alerts).lower():
        compressor_score += 10
    
    if compressor_score >= 50:
        failure_modes.append(("🌀 Compressor Fouling", compressor_score))
    
    # -------------------------------------------------
    # 4. BEARING WEAR / MECHANICAL VIBRATION
    # -------------------------------------------------
    # Indicators: High vibration, high NRf, high NRc
    bearing_score = 0
    vib_sensors = [sensors.get(f'sensor{i}', 0) for i in [3, 11, 12, 13]]
    avg_vib = sum(vib_sensors) / len(vib_sensors)
    
    if avg_vib > 40:
        bearing_score += 40
    if sensor_data['NRf'] > 2380:
        bearing_score += 25
    if sensor_data['NRc'] > 9600:
        bearing_score += 25
    if 'vibration' in str(alerts).lower():
        bearing_score += 10
    
    if bearing_score >= 50:
        failure_modes.append(("⚠️ Bearing Wear", bearing_score))
    
    # -------------------------------------------------
    # 5. COMBUSTION CHAMBER ISSUES
    # -------------------------------------------------
    # Indicators: High T30, high T50, high phi
    combustion_score = 0
    if sensor_data['T30'] > 1580:
        combustion_score += 30
    if sensor_data['T50'] > 1590:
        combustion_score += 30
    if sensor_data['phi'] > 470:
        combustion_score += 30
    if pred_rul < 30:
        combustion_score += 10
    
    if combustion_score >= 50:
        failure_modes.append(("🔥 Combustion Issues", combustion_score))
    
    # -------------------------------------------------
    # 6. FAN BLADE DAMAGE
    # -------------------------------------------------
    # Indicators: High NRf, high vibration, low BPR
    fan_score = 0
    if sensor_data['NRf'] > 2390:
        fan_score += 40
    if avg_vib > 45:
        fan_score += 35
    if sensor_data['BPR'] < 7.0:
        fan_score += 25
    
    if fan_score >= 50:
        failure_modes.append(("🌀 Fan Blade Damage", fan_score))
    
    # -------------------------------------------------
    # 7. SEAL LEAKAGE
    # -------------------------------------------------
    # Indicators: Low pressures, high temperatures
    seal_score = 0
    if sensor_data['P30'] < 310:
        seal_score += 40
    if sensor_data['Ps30'] < 45:
        seal_score += 35
    if sensor_data['T50'] > 1580:
        seal_score += 25
    
    if seal_score >= 50:
        failure_modes.append(("💨 Seal Leakage", seal_score))
    
    # -------------------------------------------------
    # Sort by confidence and format results
    # -------------------------------------------------
    failure_modes.sort(key=lambda x: x[1], reverse=True)
    
    # Add general degradation if no specific mode found
    if not failure_modes and pred_rul < 80:
        failure_modes.append(("📉 General Degradation", 60))
    
    # Format with confidence levels
    formatted_modes = []
    for mode, confidence in failure_modes[:3]:  # Show top 3
        if confidence >= 75:
            formatted_modes.append(f"{mode} (High Confidence)")
        elif confidence >= 50:
            formatted_modes.append(f"{mode} (Medium Confidence)")
        else:
            formatted_modes.append(f"{mode} (Low Confidence)")
    
    # Add recommended actions
    if alerts:
        formatted_modes.append("💡 Action items: " + ", ".join(alerts))
    
    # Add RUL-based recommendation
    if pred_rul < 20:
        formatted_modes.append("🚨 IMMEDIATE MAINTENANCE REQUIRED")
    elif pred_rul < 40:
        formatted_modes.append("📅 Schedule maintenance within next 10 cycles")
    elif pred_rul < 60:
        formatted_modes.append("🔧 Plan for upcoming maintenance")
    
    return formatted_modes


def get_failure_mode_icon(mode):
    """Return appropriate icon for each failure mode"""
    if "HPT" in mode or "Combustion" in mode:
        return "🔥"
    elif "LPT" in mode:
        return "⚙️"
    elif "Compressor" in mode or "Fan" in mode:
        return "🌀"
    elif "Bearing" in mode:
        return "⚠️"
    elif "Seal" in mode:
        return "💨"
    elif "General" in mode:
        return "📉"
    else:
        return "🔧"