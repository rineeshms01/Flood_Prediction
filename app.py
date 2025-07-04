import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import random
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .flood-warning {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        color: #c62828;
    }
    .no-flood {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler with error handling
@st.cache_resource
def load_models():
    try:
        model = load_model("flood_prediction_model.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler, True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, False

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state.predictions = []

# Main title
st.markdown('<h1 class="main-header">🌊 Rainfall & Flood Prediction System</h1>', unsafe_allow_html=True)

# Load models
model, scaler, models_loaded = load_models()

if not models_loaded:
    st.error("⚠️ Unable to load prediction models. Please ensure 'flood_prediction_model.h5' and 'scaler.pkl' files are available.")
    st.stop()

# Sidebar for additional controls
st.sidebar.header("🔧 Configuration")
simulation_mode = st.sidebar.selectbox(
    "Rainfall Simulation Mode",
    ["Moderate", "Heavy", "Light", "Custom Range"]
)

if simulation_mode == "Custom Range":
    min_rain = st.sidebar.slider("Minimum Rainfall (mm)", 0, 100, 10)
    max_rain = st.sidebar.slider("Maximum Rainfall (mm)", 100, 300, 150)
else:
    rain_ranges = {
        "Light": (5, 50),
        "Moderate": (10, 150),
        "Heavy": (50, 250)
    }
    min_rain, max_rain = rain_ranges[simulation_mode]

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Step 1: Place input
    st.markdown('<div class="section-header">📍 Location Details</div>', unsafe_allow_html=True)
    place = st.text_input("Enter Place Name", placeholder="e.g., Mumbai, Kerala, Chennai")
    
    if place:
        st.success(f"✅ Location set: {place}")

    # Step 2: Weekly rainfall estimation
    st.markdown('<div class="section-header">🌧️ Weekly Rainfall Prediction</div>', unsafe_allow_html=True)
    
    col_manual, col_auto = st.columns(2)
    
    with col_manual:
        manual_week = st.checkbox("📝 Enter This Week's Rainfall Manually")
        if manual_week:
            week_rainfall = st.number_input(
                "Enter this week's rainfall (mm)", 
                min_value=0.0, 
                max_value=500.0,
                step=0.1,
                help="Enter the observed or expected rainfall for this week"
            )
        else:
            week_rainfall = round(random.uniform(min_rain, max_rain), 2)
    
    with col_auto:
        if not manual_week:
            st.metric(
                label="Predicted Weekly Rainfall",
                value=f"{week_rainfall} mm",
                help="Simulated rainfall based on historical patterns"
            )
        
        # Rainfall category
        if week_rainfall < 25:
            category = "Light"
            color = "🟢"
        elif week_rainfall < 75:
            category = "Moderate"
            color = "🟡"
        else:
            category = "Heavy"
            color = "🔴"
        
        st.info(f"{color} **Category:** {category} Rainfall")

with col2:
    # Quick stats
    st.markdown('<div class="section-header">📊 Quick Stats</div>', unsafe_allow_html=True)
    
    # Generate some comparative stats
    avg_weekly = 45  # Assumed average
    if week_rainfall > avg_weekly * 1.5:
        comparison = f"⚠️ {((week_rainfall/avg_weekly-1)*100):.0f}% above average"
    elif week_rainfall < avg_weekly * 0.5:
        comparison = f"📉 {((1-week_rainfall/avg_weekly)*100):.0f}% below average"
    else:
        comparison = "📊 Within normal range"
    
    st.metric(
        label="Compared to Average",
        value=f"{week_rainfall} mm",
        delta=f"{week_rainfall-avg_weekly:.1f} mm"
    )
    st.write(comparison)

# Step 3: Monthly rainfall projection
st.markdown('<div class="section-header">📅 Monthly Rainfall Projection</div>', unsafe_allow_html=True)

# Generate monthly data with seasonal patterns
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# Seasonal multipliers (simulating monsoon patterns)
seasonal_multipliers = [0.3, 0.4, 0.6, 0.8, 1.2, 2.0, 
                       2.5, 2.2, 1.8, 1.0, 0.5, 0.3]

monthly_rainfall = []
for i, multiplier in enumerate(seasonal_multipliers):
    base_monthly = week_rainfall * 4 * multiplier
    variation = random.uniform(0.8, 1.2)
    monthly_rainfall.append(round(base_monthly * variation, 2))

rainfall_dict = {month: rain for month, rain in zip(months, monthly_rainfall)}
rainfall_df = pd.DataFrame([rainfall_dict])

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 Data Table", "🎯 Prediction"])

with tab1:
    # Create interactive plotly charts
    fig_line = px.line(
        x=months, 
        y=monthly_rainfall,
        title="Monthly Rainfall Trend",
        labels={'x': 'Month', 'y': 'Rainfall (mm)'},
        markers=True
    )
    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title="Rainfall (mm)",
        showlegend=False
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Bar chart
    fig_bar = px.bar(
        x=months, 
        y=monthly_rainfall,
        title="Monthly Rainfall Distribution",
        labels={'x': 'Month', 'y': 'Rainfall (mm)'},
        color=monthly_rainfall,
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("📋 Monthly Rainfall Data")
    
    # Enhanced table with formatting
    table_df = pd.DataFrame({
        'Month': months,
        'Rainfall (mm)': monthly_rainfall,
        'Category': ['Heavy' if r > 150 else 'Moderate' if r > 75 else 'Light' for r in monthly_rainfall]
    })
    
    st.dataframe(
        table_df,
        use_container_width=True,
        column_config={
            "Rainfall (mm)": st.column_config.NumberColumn(
                "Rainfall (mm)",
                help="Predicted monthly rainfall",
                format="%.2f mm"
            ),
            "Category": st.column_config.TextColumn(
                "Category",
                help="Rainfall intensity category"
            )
        }
    )
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Annual", f"{sum(monthly_rainfall):.0f} mm")
    with col2:
        st.metric("Average Monthly", f"{np.mean(monthly_rainfall):.0f} mm")
    with col3:
        st.metric("Peak Month", f"{max(monthly_rainfall):.0f} mm")
    with col4:
        st.metric("Peak Period", months[monthly_rainfall.index(max(monthly_rainfall))])

with tab3:
    # Step 4: Flood Prediction
    st.markdown('<div class="section-header">🎯 Flood Risk Assessment</div>', unsafe_allow_html=True)
    
    predict_button = st.button("🌊 Predict Flood Risk", type="primary", use_container_width=True)
    
    if predict_button:
        if not place.strip():
            st.warning("⚠️ Please enter a place name to proceed with prediction.")
        else:
            with st.spinner("Analyzing flood risk..."):
                # Prepare model input
                X_input = np.array(monthly_rainfall).reshape(1, -1)
                X_scaled = scaler.transform(X_input)
                
                # Flood prediction
                prediction_prob = model.predict(X_scaled)[0][0]
                prediction_result = "YES - Flood Expected!" if prediction_prob >= 0.5 else "NO - No Flood Expected"
                
                # Display results in styled boxes
                if prediction_prob >= 0.5:
                    st.markdown(f"""
                    <div class="prediction-box flood-warning">
                        <h3>⚠️ FLOOD WARNING</h3>
                        <p><strong>Location:</strong> {place}</p>
                        <p><strong>Weekly Rainfall:</strong> {week_rainfall} mm</p>
                        <p><strong>Risk Level:</strong> HIGH</p>
                        <p><strong>Confidence:</strong> {prediction_prob:.2%}</p>
                        <p><strong>Recommendation:</strong> Take necessary precautions and monitor weather updates closely.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box no-flood">
                        <h3>✅ LOW FLOOD RISK</h3>
                        <p><strong>Location:</strong> {place}</p>
                        <p><strong>Weekly Rainfall:</strong> {week_rainfall} mm</p>
                        <p><strong>Risk Level:</strong> LOW</p>
                        <p><strong>Confidence:</strong> {(1-prediction_prob):.2%}</p>
                        <p><strong>Status:</strong> Current conditions indicate low flood risk.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk breakdown
                st.subheader("📊 Risk Analysis Breakdown")
                
                risk_factors = pd.DataFrame({
                    'Factor': ['Total Annual Rainfall', 'Peak Month Rainfall', 'Monsoon Intensity', 'Seasonal Distribution'],
                    'Value': [f"{sum(monthly_rainfall):.0f} mm", f"{max(monthly_rainfall):.0f} mm", 
                             f"{np.std(monthly_rainfall):.0f} mm std", f"{len([r for r in monthly_rainfall if r > 100])} heavy months"],
                    'Risk Level': ['High' if sum(monthly_rainfall) > 1500 else 'Medium' if sum(monthly_rainfall) > 1000 else 'Low',
                                  'High' if max(monthly_rainfall) > 200 else 'Medium' if max(monthly_rainfall) > 150 else 'Low',
                                  'High' if np.std(monthly_rainfall) > 80 else 'Medium' if np.std(monthly_rainfall) > 50 else 'Low',
                                  'High' if len([r for r in monthly_rainfall if r > 100]) > 6 else 'Medium' if len([r for r in monthly_rainfall if r > 100]) > 3 else 'Low']
                })
                
                st.dataframe(risk_factors, use_container_width=True)
                
                # Store prediction history
                st.session_state.predictions.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Place": place,
                    "Week Rainfall (mm)": week_rainfall,
                    "Flood Prediction": prediction_result,
                    "Confidence": round(prediction_prob, 2),
                    "Total Annual (mm)": round(sum(monthly_rainfall), 2),
                    **rainfall_dict
                })
                
                st.success("✅ Prediction completed and saved to history!")

# Prediction History
if st.session_state.predictions:
    st.markdown('<div class="section-header">📚 Prediction History</div>', unsafe_allow_html=True)
    
    if st.checkbox("📋 Show Detailed Prediction History"):
        history_df = pd.DataFrame(st.session_state.predictions)
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(st.session_state.predictions))
        with col2:
            flood_predictions = len([p for p in st.session_state.predictions if "YES" in p["Flood Prediction"]])
            st.metric("Flood Warnings", flood_predictions)
        with col3:
            if len(st.session_state.predictions) > 0:
                avg_confidence = np.mean([p["Confidence"] for p in st.session_state.predictions])
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        # Display history table
        st.dataframe(
            history_df,
            use_container_width=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn(
                    "Prediction Time",
                    format="DD/MM/YYYY HH:mm"
                ),
                "Confidence": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f"
                )
            }
        )
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.predictions = []
            st.rerun()

# Footer with additional information
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h4>ℹ️ Important Notes:</h4>
    <ul>
        <li>This is a predictive model based on historical data and current rainfall patterns</li>
        <li>Actual flood conditions may vary based on local geography, drainage systems, and other factors</li>
        <li>For emergency situations, please consult local meteorological departments and disaster management authorities</li>
        <li>The model uses simulated data for demonstration purposes</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Additional features sidebar
st.sidebar.markdown("---")
st.sidebar.header("📱 Additional Features")

if st.sidebar.button("📄 Download Predictions"):
    if st.session_state.predictions:
        csv = pd.DataFrame(st.session_state.predictions).to_csv(index=False)
        st.sidebar.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name=f"flood_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.warning("No predictions to download")

# Model info
st.sidebar.markdown("---")
st.sidebar.info("""
**Model Information:**
- Algorithm: Neural Network
- Features: 12 monthly rainfall values
- Output: Flood probability (0-1)
- Threshold: 0.5 for flood prediction
""")