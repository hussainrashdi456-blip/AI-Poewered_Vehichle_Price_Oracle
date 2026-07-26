# ============================================
# AI Used Car Price Predictor - Streamlit App
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a0b2e 0%, #2d1b4e 50%, #1a0b2e 100%);
    }
    h1, h2, h3 {
        color: #c084fc !important;
        font-weight: 700 !important;
    }
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #a855f7, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #a78bfa;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(45, 27, 78, 0.6);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        border: 1px solid #7c3aed;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #c084fc;
    }
    .metric-label {
        color: #a78bfa;
        font-size: 0.9rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #2d1b4e, #1a0b2e);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid #c084fc;
        box-shadow: 0 10px 30px rgba(192, 132, 252, 0.2);
        margin: 20px 0;
    }
    .price-display {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #c084fc, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0b2e, #2d1b4e);
        border-right: 1px solid #7c3aed;
    }
    .stButton button {
        background: linear-gradient(135deg, #a855f7, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(168, 85, 247, 0.4);
    }
    hr {
        border-color: #7c3aed;
        margin: 20px 0;
    }
    .detail-card {
        background: rgba(45, 27, 78, 0.6);
        border-radius: 12px;
        padding: 15px;
        border-left: 4px solid #c084fc;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL LOADING FUNCTIONS
# ============================================
@st.cache_resource
def load_model():
    """Load trained XGBoost model"""
    possible_paths = [
        "best_tuned_used_car_price_model.pkl",
        "../model/best_tuned_used_car_price_model.pkl",
        "model/best_tuned_used_car_price_model.pkl",
        "best_used_car_price_model.pkl",
        "../model/best_used_car_price_model.pkl",
        "model/best_used_car_price_model.pkl",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                return model
            except Exception as e:
                continue
    
    st.error("❌ Model not found! Please check model files.")
    st.info(f"Current directory: {os.getcwd()}")
    st.info(f"Files: {os.listdir('.')}")
    return None

@st.cache_resource
def load_features():
    """Load feature names for encoding"""
    possible_paths = [
        "model_features.pkl",
        "../model/model_features.pkl",
        "model/model_features.pkl",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except:
                continue
    return None

@st.cache_data
def load_data():
    """Load dataset for dropdown options"""
    possible_paths = [
        "data/dummy_data.csv",
        "../data/dummy_data.csv",
        "dummy_data.csv",
        "data/vehicle_price_dataset.csv",
        "../data/vehicle_price_dataset.csv",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                
                categorical_cols = ['make', 'model', 'transmission', 'fuel_type', 
                                    'drivetrain', 'body_type', 'exterior_color', 
                                    'interior_color', 'accident_history', 
                                    'seller_type', 'condition', 'trim']
                
                for col in categorical_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.lower().str.strip()
                
                return df
            except:
                continue
    
    # Fallback dummy data
    st.warning("⚠️ Dataset not found. Using fallback options.")
    return pd.DataFrame({
        'make': ['toyota', 'honda', 'suzuki', 'bmw', 'mercedes'],
        'model': ['corolla', 'civic', 'alto', '3series', 'cclass'],
        'transmission': ['automatic', 'manual'],
        'fuel_type': ['petrol', 'diesel'],
        'condition': ['excellent', 'good', 'fair']
    })

# ============================================
# LOAD ALL RESOURCES
# ============================================
with st.spinner("🔄 Loading AI Model..."):
    model = load_model()
    model_features = load_features()
    df = load_data()

if model is None:
    st.stop()

# ============================================
# GET OPTIONS
# ============================================
makes = sorted(df['make'].dropna().unique())
transmissions = sorted(df['transmission'].dropna().unique())
fuel_types = sorted(df['fuel_type'].dropna().unique())
drivetrains = sorted(df['drivetrain'].dropna().unique())
body_types = sorted(df['body_type'].dropna().unique())
exterior_colors = sorted(df['exterior_color'].dropna().unique())
interior_colors = sorted(df['interior_color'].dropna().unique())
accident_histories = sorted(df['accident_history'].dropna().unique())
seller_types = sorted(df['seller_type'].dropna().unique())
conditions = sorted(df['condition'].dropna().unique())
trims = sorted(df['trim'].dropna().unique())

# ============================================
# TITLE
# ============================================
st.markdown('<p class="main-title">🚗 AI Used Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Vehicle Valuation at Your Fingertips</p>', unsafe_allow_html=True)

# ============================================
# METRICS
# ============================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">98.58%</div>
        <div class="metric-label">🎯 Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">0.3s</div>
        <div class="metric-label">⚡ Prediction Speed</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">$1,146</div>
        <div class="metric-label">📊 Avg Error (MAE)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">926k</div>
        <div class="metric-label">📚 Trained Records</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# MAIN LAYOUT
# ============================================
left_col, right_col = st.columns([1.2, 0.8])

# ============================================
# LEFT COLUMN - INPUTS
# ============================================
with left_col:
    st.markdown("### 📝 Vehicle Information")
    
    # Row 1: Make, Model, Trim
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        selected_make = st.selectbox("🏭 Make", makes, key="make")
        available_models = sorted(df[df['make'] == selected_make]['model'].dropna().unique())
        selected_model = st.selectbox("📌 Model", available_models, key="model")
    
    with col_b:
        selected_trim = st.selectbox("🎚️ Trim", trims, key="trim")
    
    with col_c:
        selected_year = st.slider("📅 Year", 2000, 2025, 2018, key="year")
    
    # Row 2: Mileage, Engine HP, Owner
    col_d, col_e, col_f = st.columns(3)
    with col_d:
        selected_mileage = st.number_input("📊 Mileage (miles)", min_value=0, max_value=300000, value=50000, step=1000, key="mileage")
    
    with col_e:
        selected_engine_hp = st.number_input("🔧 Engine HP", min_value=50, max_value=800, value=200, step=10, key="hp")
    
    with col_f:
        selected_owner_count = st.number_input("👥 Owner Count", min_value=1, max_value=10, value=1, step=1, key="owner")
    
    # Row 3: Transmission, Fuel, Drivetrain
    col_g, col_h, col_i = st.columns(3)
    with col_g:
        selected_transmission = st.selectbox("⚙️ Transmission", transmissions, key="transmission")
    
    with col_h:
        selected_fuel_type = st.selectbox("⛽ Fuel Type", fuel_types, key="fuel")
    
    with col_i:
        selected_drivetrain = st.selectbox("🔘 Drivetrain", drivetrains, key="drivetrain")
    
    # Row 4: Body, Exterior, Interior
    col_j, col_k, col_l = st.columns(3)
    with col_j:
        selected_body_type = st.selectbox("🚙 Body Type", body_types, key="body")
    
    with col_k:
        selected_exterior_color = st.selectbox("🎨 Exterior Color", exterior_colors, key="exterior")
    
    with col_l:
        selected_interior_color = st.selectbox("🪑 Interior Color", interior_colors, key="interior")
    
    # Row 5: Accident, Seller, Condition
    col_m, col_n, col_o = st.columns(3)
    with col_m:
        selected_accident_history = st.selectbox("⚠️ Accident History", accident_histories, key="accident")
    
    with col_n:
        selected_seller_type = st.selectbox("🏪 Seller Type", seller_types, key="seller")
    
    with col_o:
        selected_condition = st.selectbox("⭐ Condition", conditions, key="condition")
    
    # Row 6: Brand Popularity
    selected_brand_popularity = st.slider("🏆 Brand Popularity", 1, 100, 50, key="brand")

# ============================================
# RIGHT COLUMN - PREDICTION
# ============================================
with right_col:
    st.markdown("### 🔮 Price Prediction")
    
    # Feature engineering
    current_year = 2025
    vehicle_age = current_year - selected_year
    if vehicle_age == 0:
        vehicle_age = 1
    
    mileage_per_year = selected_mileage / vehicle_age
    hp_per_mileage = selected_engine_hp / (selected_mileage + 1)
    
    # Create input
    input_data = {
        'year': selected_year,
        'mileage': selected_mileage,
        'engine_hp': selected_engine_hp,
        'owner_count': selected_owner_count,
        'vehicle_age': vehicle_age,
        'mileage_per_year': mileage_per_year,
        'brand_popularity': selected_brand_popularity / 100,
        'hp_per_mileage': hp_per_mileage,
        'make': selected_make,
        'model': selected_model,
        'transmission': selected_transmission,
        'fuel_type': selected_fuel_type,
        'drivetrain': selected_drivetrain,
        'body_type': selected_body_type,
        'exterior_color': selected_exterior_color,
        'interior_color': selected_interior_color,
        'accident_history': selected_accident_history,
        'seller_type': selected_seller_type,
        'condition': selected_condition,
        'trim': selected_trim
    }
    
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    
    if model_features:
        for col in model_features:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_features]
    
    # Predict
    if st.button("🚀 PREDICT PRICE", use_container_width=True):
        start_time = time.time()
        prediction = model.predict(input_encoded)[0]
        end_time = time.time()
        pred_time = (end_time - start_time) * 1000
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="price-display">${prediction:,.2f}</div>
            <p style="text-align: center; color: #a78bfa; margin-top: 10px;">Estimated Fair Market Value</p>
            <p style="text-align: center; color: #7c3aed; font-size: 0.8rem;">
                ⚡ Predicted in {pred_time:.0f} milliseconds
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state['prediction'] = prediction
        st.session_state['car_details'] = input_data
        st.session_state['prediction_time'] = pred_time

# ============================================
# VEHICLE DETAILS
# ============================================
if 'prediction' in st.session_state:
    st.markdown("---")
    st.markdown("### 📋 Vehicle Details")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"""
        <div class="detail-card">
            <h4 style="color: #c084fc;">🚗 Basic Information</h4>
            <p><strong>Make:</strong> {st.session_state['car_details']['make'].title()}</p>
            <p><strong>Model:</strong> {st.session_state['car_details']['model'].title()}</p>
            <p><strong>Trim:</strong> {st.session_state['car_details']['trim'].title()}</p>
            <p><strong>Year:</strong> {st.session_state['car_details']['year']}</p>
            <p><strong>Vehicle Age:</strong> {st.session_state['car_details']['vehicle_age']} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown(f"""
        <div class="detail-card">
            <h4 style="color: #a855f7;">⚙️ Technical Specifications</h4>
            <p><strong>Mileage:</strong> {st.session_state['car_details']['mileage']:,} miles</p>
            <p><strong>Engine HP:</strong> {st.session_state['car_details']['engine_hp']} HP</p>
            <p><strong>Transmission:</strong> {st.session_state['car_details']['transmission'].title()}</p>
            <p><strong>Fuel Type:</strong> {st.session_state['car_details']['fuel_type'].title()}</p>
            <p><strong>Drivetrain:</strong> {st.session_state['car_details']['drivetrain'].upper()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Download report
    st.markdown("---")
    col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
    with col_d2:
        import json
        download_data = {
            "predicted_price": float(st.session_state['prediction']),
            "prediction_time_ms": st.session_state['prediction_time'],
            "vehicle_details": st.session_state['car_details'],
            "predicted_at": str(datetime.now())
        }
        json_str = json.dumps(download_data, indent=2)
        
        st.download_button(
            label="📥 Download Report (JSON)",
            data=json_str,
            file_name=f"car_price_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a78bfa; padding: 20px;">
    <p>🚀 AI-Powered Used Car Price Predictor | Final Year Project 2026</p>
    <p style="font-size: 0.8rem;">Powered by XGBoost | 98.58% Accuracy | 0.3 Seconds Prediction</p>
    <p style="font-size: 0.8rem;">Shah Abdul Latif University, Khairpur</p>
</div>
""", unsafe_allow_html=True)