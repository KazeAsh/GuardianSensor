import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json

# Page config
st.set_page_config(
    page_title="GuardianSeat Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚗 GuardianSeat Safety Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/car--v1.png", width=100)
    st.title("Controls")
    
    # Vehicle selection
    vehicle_id = st.selectbox(
        "Select Vehicle",
        ["Toyota-001", "Mazda-002", "Subaru-003", "Honda-004"]
    )
    
    # Time range
    time_range = st.select_slider(
        "Time Range",
        options=["1h", "6h", "12h", "24h", "7d"]
    )
    
    # Alert filters
    show_resolved = st.checkbox("Show Resolved Alerts", value=False)
    
    # Manual test buttons
    st.markdown("---")
    st.subheader("Test Scenarios")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚨 Emergency Test"):
            # Simulate emergency
            test_data = {
                "temperature_c": 42.5,
                "weight_left_kg": 18.0,
                "weight_right_kg": 0.0,
                "motion_detected": False,
                "door_state": "closed",
                "engine_state": "off",
                "timestamp": datetime.now().isoformat()
            }
            response = requests.post("http://localhost:8000/sensors", json=test_data)
            st.success("Emergency scenario triggered!")
    
    with col2:
        if st.button("✅ Safe Test"):
            # Simulate safe scenario
            test_data = {
                "temperature_c": 22.0,
                "weight_left_kg": 0.0,
                "weight_right_kg": 75.0,
                "motion_detected": True,
                "door_state": "open",
                "engine_state": "on",
                "timestamp": datetime.now().isoformat()
            }
            response = requests.post("http://localhost:8000/sensors", json=test_data)
            st.success("Safe scenario triggered!")

# Main dashboard layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Alerts", "📈 Analytics", "⚙️ System"])

with tab1:
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Temperature", "38.5°C", "+5.2°C", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Score", "0.72", "High", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Active Alerts", "3", "+2", delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Vehicle Status", "⚠️ Parked", "Child Detected")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts row
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Trend")
        
        # Generate sample temperature data
        hours = list(range(24))
        temps = [25 + np.sin(h/3) * 10 + np.random.randn() * 2 for h in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=temps, mode='lines+markers', name='Temperature',
                                line=dict(color='red', width=3)))
        
        # Add danger zone
        fig.add_hrect(y0=40, y1=50, line_width=0, fillcolor="red", opacity=0.2,
                     annotation_text="Danger Zone", annotation_position="top left")
        fig.add_hrect(y0=26, y1=40, line_width=0, fillcolor="orange", opacity=0.2,
                     annotation_text="Warning Zone", annotation_position="top left")
        
        fig.update_layout(
            xaxis_title="Hours",
            yaxis_title="Temperature (°C)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Components")
        
        # Risk breakdown
        risk_data = pd.DataFrame({
            'Factor': ['Temperature', 'Occupancy', 'Time Elapsed', 'Vehicle State'],
            'Risk Score': [0.8, 0.9, 0.6, 0.5]
        })
        
        fig = px.bar(risk_data, x='Factor', y='Risk Score', color='Risk Score',
                    color_continuous_scale=['green', 'yellow', 'red'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Vehicle status
    st.markdown("---")
    st.subheader("Vehicle Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.info("🚪 Doors: **Closed**")
    with col2:
        st.warning("🔥 Engine: **Off**")
    with col3:
        st.error("👶 Occupancy: **Child Detected**")
    with col4:
        st.success("🎥 Camera: **Active**")

with tab2:
    st.header("🚨 Alert Management")
    
    # Sample alerts data
    alerts_data = [
        {
            "id": 1,
            "level": "emergency",
            "timestamp": "2024-01-15 14:30:00",
            "message": "Child detected in vehicle with critical temperature (42°C)",
            "vehicle": "TOYOTA-001",
            "status": "active",
            "location": "Parking Lot A"
        },
        {
            "id": 2,
            "level": "critical", 
            "timestamp": "2024-01-15 13:45:00",
            "message": "Vehicle parked for 45 minutes with child inside",
            "vehicle": "TOYOTA-002",
            "status": "resolved",
            "location": "Mall Parking"
        },
        {
            "id": 3,
            "level": "warning",
            "timestamp": "2024-01-15 12:15:00",
            "message": "Temperature rising above safe levels",
            "vehicle": "TOYOTA-003",
            "status": "active",
            "location": "Residential Street"
        }
    ]
    
    # Display alerts
    for alert in alerts_data:
        if alert["status"] == "active" or show_resolved:
            alert_class = "alert-critical" if alert["level"] == "emergency" else "alert-warning"
            
            st.markdown(f'<div class="{alert_class}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.markdown(f"**{alert['level'].upper()}**")
            with col2:
                st.markdown(f"**{alert['message']}**")
                st.caption(f"{alert['timestamp']} | {alert['vehicle']} | {alert['location']}")
            with col3:
                if alert["status"] == "active":
                    if st.button("Resolve", key=f"resolve_{alert['id']}"):
                        st.success(f"Alert {alert['id']} resolved!")
                else:
                    st.success("✅ Resolved")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

with tab3:
    st.header("📈 Data Analytics")
    
    # Sensor data visualization
    st.subheader("Sensor Data Analysis")
    
    # Generate sample sensor data
    time_points = pd.date_range(start='2024-01-15 08:00', periods=100, freq='T')
    sensor_df = pd.DataFrame({
        'timestamp': time_points,
        'temperature': 25 + np.cumsum(np.random.randn(100) * 0.1) + np.sin(np.arange(100)/10) * 5,
        'weight_left': np.random.choice([0, 15, 18], 100, p=[0.3, 0.5, 0.2]),
        'weight_right': np.random.choice([0, 70, 75], 100, p=[0.6, 0.3, 0.1]),
        'motion': np.random.choice([True, False], 100, p=[0.7, 0.3])
    })
    
    # Multi-axis chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=sensor_df['timestamp'],
        y=sensor_df['temperature'],
        name="Temperature",
        yaxis="y1",
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=sensor_df['timestamp'],
        y=sensor_df['weight_left'],
        name="Left Seat Weight",
        yaxis="y2",
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=sensor_df['timestamp'],
        y=sensor_df['weight_right'],
        name="Right Seat Weight",
        yaxis="y2",
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title="Sensor Data Timeline",
        yaxis=dict(title="Temperature (°C)", side="left"),
        yaxis2=dict(title="Weight (kg)", side="right", overlaying="y"),
        xaxis_title="Time",
        height=400,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(sensor_df.describe())
    
    with col2:
        # Correlation heatmap
        corr_matrix = sensor_df[['temperature', 'weight_left', 'weight_right']].corr()
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu')
        fig.update_layout(title="Sensor Correlations")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("⚙️ System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Alert Thresholds")
        
        temp_threshold = st.slider(
            "Temperature Alert Threshold (°C)",
            min_value=20, max_value=50, value=26, step=1
        )
        
        time_threshold = st.slider(
            "Time Alert Threshold (minutes)",
            min_value=5, max_value=120, value=30, step=5
        )
        
        risk_threshold = st.slider(
            "Risk Score Emergency Threshold",
            min_value=0.1, max_value=1.0, value=0.7, step=0.1
        )
        
        if st.button("Save Thresholds", type="primary"):
            st.success("Thresholds saved successfully!")
    
    with col2:
        st.subheader("Notification Settings")
        
        email_notifications = st.checkbox("Email Notifications", value=True)
        sms_notifications = st.checkbox("SMS Notifications", value=True)
        push_notifications = st.checkbox("Push Notifications", value=True)
        
        emergency_contact = st.text_input("Emergency Contact Number", "+81 90-1234-5678")
        backup_contact = st.text_input("Backup Contact Number", "+81 90-9876-5432")
        
        if st.button("Save Notifications"):
            st.success("Notification settings saved!")
    
    st.markdown("---")
    st.subheader("System Information")
    
    sys_info = {
        "API Status": "✅ Running",
        "Database": "✅ Connected",
        "ML Model": "✅ Loaded",
        "Camera Feed": "✅ Active",
        "Sensor Network": "✅ Connected",
        "Last Update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    for key, value in sys_info.items():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.text(key)
        with col2:
            st.text(value)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>GuardianSeat Safety System | Built with safety in mind for all</p>
    <p>⚠️ This is a demonstration system.</p>
</div>
""", unsafe_allow_html=True)