"""
HealthPredict – Disease Risk Prediction using Lifestyle Analytics
Next-Generation SaaS Healthcare Analytics & Machine Learning Web Application
"""

import os
import sys

# Ensure project root is always at the top of sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.prediction import HealthPredictor
from src.explainability import HealthExplainer
from utils.helpers import (
    calculate_bmi,
    compute_lifestyle_quality_score,
    create_gauge_chart,
    create_biometric_radar_chart,
    simulate_lifestyle_intervention,
    generate_personalized_insights,
    validate_user_inputs,
    get_custom_css
)

# Page Configuration
st.set_page_config(
    page_title="HealthPredict | Disease Risk & Lifestyle Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Modern Healthcare SaaS Styling
st.markdown(get_custom_css(), unsafe_allow_html=True)


@st.cache_resource
def load_prediction_engine():
    """Initializes and caches the predictor and explainer."""
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    predictor = HealthPredictor(models_dir=models_dir)
    explainer = HealthExplainer(predictor)
    
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    metrics_data = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
            
    return predictor, explainer, metrics_data


predictor, explainer, metrics_data = load_prediction_engine()

# Initialize Session State with default profile
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'age': 46,
        'gender': 'Male',
        'height_cm': 175.0,
        'weight_kg': 82.0,
        'bmi': 26.8,
        'is_smoker': False,
        'cigs_per_day': 0,
        'physical_activity': 'Moderate (3-4 days/week)',
        'sleep_hours': 7.0,
        'diet_quality': 'Balanced',
        'stress_level': 'Moderate',
        'alcohol_intake': 'Occasional / Light',
        'blood_pressure_sys': 128.0,
        'blood_pressure_dia': 84.0,
        'glucose': 105.0,
        'cholesterol': 215.0,
        'resting_heart_rate': 72.0,
        'max_heart_rate': 152.0,
        'skin_thickness': 24.0,
        'insulin': 95.0,
        'diabetes_pedigree': 0.45,
        'pregnancies': 0,
        'chest_pain_type': 0,
        'exercise_angina': 0,
        'st_depression': 0.6,
        'st_slope': 1,
        'num_major_vessels': 0,
        'thalassemia': 2,
        'bp_meds': False,
        'has_diabetes': False
    }

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = predictor.predict_all(st.session_state.user_data)


# Sidebar Navigation & Sensitivity Controls
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 10px 0;'>
        <h2 style='color: #0284C7; margin: 0; font-weight: 800; font-size: 1.6rem;'>🩺 HealthPredict</h2>
        <span style='font-size: 0.78rem; color: #64748B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;'>Lifestyle & Disease Risk AI</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "📋 Health Assessment",
        "📊 Risk Dashboard & Simulator",
        "🔍 Explainable AI (SHAP)",
        "📈 Data Insights & Models",
        "ℹ️ About & Medical Disclaimer"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("#### ⚙️ Screening Sensitivity")
risk_low_thresh = st.sidebar.slider("Low-Risk Cutoff (%)", 10, 40, 30, 5)
risk_high_thresh = st.sidebar.slider("High-Risk Cutoff (%)", 50, 80, 60, 5)

st.sidebar.markdown(
    """
    <div style='background:#F8FAFC; border:1px solid #E2E8F0; padding:12px; border-radius:10px; font-size:0.78rem; color:#475569; margin-top:15px;'>
        <div style='display:flex; align-items:center; margin-bottom:4px;'>
            <span class='pulse-dot'></span>
            <b style='color:#0F172A;'>Live Local Inference Engine</b>
        </div>
        100% in-memory processing. Zero cloud telemetry.
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 1. HOME PAGE
# ==============================================================================
if page == "🏠 Home":
    st.markdown(
        """
        <div class='hero-gradient'>
            <span style='background:rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;'>
                AI-Powered Preventative Screening
            </span>
            <h1 style='color:white; font-size:2.5rem; font-weight:800; margin:12px 0 8px 0; letter-spacing:-0.02em;'>
                HealthPredict: Disease Risk Screening via Lifestyle Analytics
            </h1>
            <p style='font-size:1.1rem; color:rgba(255,255,255,0.9); margin-bottom:1.5rem; max-width:850px; line-height:1.6;'>
                Understand your lifestyle habits, estimate multi-disease subclinical risk probabilities, and uncover personalized evidence-based wellness guidance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_h1, col_h2, col_h3 = st.columns(3)
    
    with col_h1:
        st.markdown(
            """
            <div class='metric-card' style='border-top: 4px solid #7C3AED;'>
                <div style='font-size: 1.8rem; margin-bottom: 8px;'>🩸</div>
                <h4 style='color: #0F172A; margin: 0;'>Type 2 Diabetes Risk</h4>
                <p style='color: #64748B; font-size: 0.88rem; margin: 8px 0;'>Evaluates glycemic load, metabolic insulin sensitivity, and visceral adiposity indicators.</p>
                <span class='badge-pill' style='background: #EDE9FE; color: #7C3AED;'>XGBoost • 82.0% ROC-AUC</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_h2:
        st.markdown(
            """
            <div class='metric-card' style='border-top: 4px solid #EF4444;'>
                <div style='font-size: 1.8rem; margin-bottom: 8px;'>🫀</div>
                <h4 style='color: #0F172A; margin: 0;'>Coronary Heart Disease</h4>
                <p style='color: #64748B; font-size: 0.88rem; margin: 8px 0;'>Analyzes exercise tolerance, lipid profiles, ischemic ST depressions, and vascular resistance.</p>
                <span class='badge-pill' style='background: #FEE2E2; color: #DC2626;'>Logistic Reg • 90.9% Recall</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_h3:
        st.markdown(
            """
            <div class='metric-card' style='border-top: 4px solid #0284C7;'>
                <div style='font-size: 1.8rem; margin-bottom: 8px;'>🩺</div>
                <h4 style='color: #0F172A; margin: 0;'>Hypertension Risk</h4>
                <p style='color: #64748B; font-size: 0.88rem; margin: 8px 0;'>Evaluates hemodynamic pressure waveforms (Mean Arterial & Pulse Pressure) and smoking burden.</p>
                <span class='badge-pill' style='background: #E0F2FE; color: #0284C7;'>Random Forest • 95.4% ROC-AUC</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    col_btn1, col_btn2 = st.columns([2, 3])
    with col_btn1:
        if st.button("🚀 Launch Health Assessment", type="primary", use_container_width=True):
            st.session_state.active_page = "📋 Health Assessment"
            st.rerun()

    st.markdown(
        """
        <div class='disclaimer-box'>
            <b>⚠️ MANDATORY MEDICAL NOTICE:</b><br>
            HealthPredict is developed strictly for educational and screening purposes. It is <b>NOT</b> a diagnostic device and should never replace qualified clinical diagnosis or medical consultation.
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# 2. HEALTH ASSESSMENT PAGE (WITH LIVE TELEMETRY)
# ==============================================================================
elif page == "📋 Health Assessment":
    st.markdown("<h2 class='main-header'>📋 Interactive Health Assessment</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Adjust parameters below. The <b>Live Telemetry HUD</b> on the right reflects instant real-time risk predictions.</p>", unsafe_allow_html=True)
    
    # Preset Demo Profiles
    st.markdown("##### ⚡ Quick Profile Presets")
    c_p1, c_p2, c_p3 = st.columns(3)
    
    if c_p1.button("🟢 Young Active Profile (Low Risk)", use_container_width=True):
        st.session_state.user_data.update({
            'age': 28, 'gender': 'Female', 'height_cm': 168.0, 'weight_kg': 59.0,
            'is_smoker': False, 'cigs_per_day': 0, 'physical_activity': 'Active (5+ days/week)',
            'sleep_hours': 8.0, 'diet_quality': 'Healthy / Mediterranean', 'stress_level': 'Low',
            'alcohol_intake': 'None / Rarely', 'blood_pressure_sys': 110.0, 'blood_pressure_dia': 72.0,
            'glucose': 84.0, 'cholesterol': 175.0, 'resting_heart_rate': 60.0, 'max_heart_rate': 178.0,
            'skin_thickness': 18.0, 'insulin': 55.0, 'diabetes_pedigree': 0.25, 'pregnancies': 0,
            'chest_pain_type': 0, 'exercise_angina': 0, 'st_depression': 0.0, 'st_slope': 0,
            'num_major_vessels': 0, 'thalassemia': 1, 'bp_meds': False, 'has_diabetes': False
        })
        st.session_state.prediction_results = predictor.predict_all(st.session_state.user_data)
        st.rerun()

    if c_p2.button("🟠 Middle-Aged Profile (Moderate Risk)", use_container_width=True):
        st.session_state.user_data.update({
            'age': 49, 'gender': 'Male', 'height_cm': 176.0, 'weight_kg': 85.0,
            'is_smoker': False, 'cigs_per_day': 0, 'physical_activity': 'Moderate (3-4 days/week)',
            'sleep_hours': 6.5, 'diet_quality': 'Average', 'stress_level': 'Moderate',
            'alcohol_intake': 'Occasional / Light', 'blood_pressure_sys': 135.0, 'blood_pressure_dia': 86.0,
            'glucose': 114.0, 'cholesterol': 225.0, 'resting_heart_rate': 74.0, 'max_heart_rate': 146.0,
            'skin_thickness': 24.0, 'insulin': 110.0, 'diabetes_pedigree': 0.45, 'pregnancies': 0,
            'chest_pain_type': 1, 'exercise_angina': 0, 'st_depression': 0.8, 'st_slope': 1,
            'num_major_vessels': 0, 'thalassemia': 2, 'bp_meds': False, 'has_diabetes': False
        })
        st.session_state.prediction_results = predictor.predict_all(st.session_state.user_data)
        st.rerun()

    if c_p3.button("🔴 High-Risk Profile (Cardiometabolic)", use_container_width=True):
        st.session_state.user_data.update({
            'age': 59, 'gender': 'Male', 'height_cm': 172.0, 'weight_kg': 96.0,
            'is_smoker': True, 'cigs_per_day': 20, 'physical_activity': 'Sedentary (Little or no exercise)',
            'sleep_hours': 5.0, 'diet_quality': 'High Fast Food / Processed', 'stress_level': 'Severe / Chronic',
            'alcohol_intake': 'Frequent / Heavy', 'blood_pressure_sys': 158.0, 'blood_pressure_dia': 98.0,
            'glucose': 165.0, 'cholesterol': 268.0, 'resting_heart_rate': 86.0, 'max_heart_rate': 122.0,
            'skin_thickness': 34.0, 'insulin': 185.0, 'diabetes_pedigree': 0.85, 'pregnancies': 0,
            'chest_pain_type': 2, 'exercise_angina': 1, 'st_depression': 2.5, 'st_slope': 2,
            'num_major_vessels': 2, 'thalassemia': 3, 'bp_meds': True, 'has_diabetes': True
        })
        st.session_state.prediction_results = predictor.predict_all(st.session_state.user_data)
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_form, col_telemetry = st.columns([3, 2])
    
    with col_form:
        st.markdown("### 1️⃣ Demographics & Anthropometrics")
        cd1, cd2 = st.columns(2)
        with cd1:
            age = st.slider("Age (Years)", 18, 95, int(st.session_state.user_data['age']))
            gender = st.radio("Biological Sex", ["Male", "Female"], index=0 if st.session_state.user_data['gender'] == "Male" else 1, horizontal=True)
        with cd2:
            height_cm = st.slider("Height (cm)", 120.0, 220.0, float(st.session_state.user_data['height_cm']), 0.5)
            weight_kg = st.slider("Weight (kg)", 35.0, 180.0, float(st.session_state.user_data['weight_kg']), 0.5)
            
        bmi_info = calculate_bmi(height_cm, weight_kg)

        st.markdown("### 2️⃣ Daily Lifestyle Habits")
        cl1, cl2 = st.columns(2)
        with cl1:
            act_opts = ["Sedentary (Little or no exercise)", "Light (1-2 days/week)", "Moderate (3-4 days/week)", "Active (5+ days/week)"]
            cur_act = st.session_state.user_data['physical_activity']
            physical_activity = st.selectbox("Physical Activity", act_opts, index=act_opts.index(cur_act) if cur_act in act_opts else 2)
            
            is_smoker = st.checkbox("Current Tobacco Smoker", value=st.session_state.user_data['is_smoker'])
            cigs_per_day = st.slider("Cigarettes / Day", 0, 60, int(st.session_state.user_data['cigs_per_day']) if is_smoker else 0, disabled=not is_smoker)
            
            sleep_hours = st.slider("Sleep Duration (Hours)", 4.0, 12.0, float(st.session_state.user_data['sleep_hours']), 0.5)
            
        with cl2:
            diet_opts = ["Healthy / Mediterranean", "Balanced", "Average", "High Fast Food / Processed"]
            cur_diet = st.session_state.user_data['diet_quality']
            diet_quality = st.selectbox("Diet Quality", diet_opts, index=diet_opts.index(cur_diet) if cur_diet in diet_opts else 1)
            
            stress_opts = ["Low", "Moderate", "High", "Severe / Chronic"]
            cur_stress = st.session_state.user_data['stress_level']
            stress_level = st.selectbox("Stress Level", stress_opts, index=stress_opts.index(cur_stress) if cur_stress in stress_opts else 1)
            
            alc_opts = ["None / Rarely", "Occasional / Light", "Moderate (1-2 drinks/day)", "Frequent / Heavy"]
            cur_alc = st.session_state.user_data['alcohol_intake']
            alcohol_intake = st.selectbox("Alcohol Intake", alc_opts, index=alc_opts.index(cur_alc) if cur_alc in alc_opts else 1)

        st.markdown("### 3️⃣ Key Biometric Measurements")
        cb1, cb2 = st.columns(2)
        with cb1:
            bp_sys = st.slider("Systolic BP (mmHg)", 85.0, 220.0, float(st.session_state.user_data['blood_pressure_sys']), 1.0)
            bp_dia = st.slider("Diastolic BP (mmHg)", 50.0, 140.0, float(st.session_state.user_data['blood_pressure_dia']), 1.0)
            glucose = st.slider("Fasting Glucose (mg/dL)", 60.0, 300.0, float(st.session_state.user_data['glucose']), 1.0)
        with cb2:
            cholesterol = st.slider("Total Cholesterol (mg/dL)", 120.0, 420.0, float(st.session_state.user_data['cholesterol']), 1.0)
            resting_hr = st.slider("Resting Heart Rate (bpm)", 45.0, 140.0, float(st.session_state.user_data['resting_heart_rate']), 1.0)
            max_hr = st.slider("Max Achieved HR (bpm)", 80.0, 210.0, float(st.session_state.user_data['max_heart_rate']), 1.0)

        with st.expander("🔬 Additional Clinical Diagnostic Settings (Optional)"):
            ca1, ca2 = st.columns(2)
            with ca1:
                chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina (0)", 1: "Atypical Angina (1)", 2: "Non-anginal Pain (2)", 3: "Asymptomatic (3)"}[x], index=st.session_state.user_data['chest_pain_type'])
                exercise_angina = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", index=st.session_state.user_data['exercise_angina'])
                st_depression = st.slider("ST Depression (oldpeak)", 0.0, 5.0, float(st.session_state.user_data['st_depression']), 0.1)
                st_slope = st.selectbox("ST Slope", [0, 1, 2], format_func=lambda x: {0: "Upsloping (0)", 1: "Flat (1)", 2: "Downsloping (2)"}[x], index=st.session_state.user_data['st_slope'])
            with ca2:
                num_major_vessels = st.selectbox("Major Vessels (ca)", [0, 1, 2, 3], index=st.session_state.user_data['num_major_vessels'])
                thalassemia = st.selectbox("Thalassemia (thal)", [1, 2, 3], format_func=lambda x: {1: "Normal (1)", 2: "Fixed Defect (2)", 3: "Reversible Defect (3)"}[x], index=st.session_state.user_data['thalassemia'] - 1)
                skin_thickness = st.slider("Skin Fold (mm)", 5.0, 60.0, float(st.session_state.user_data['skin_thickness']), 1.0)
                insulin = st.slider("Serum Insulin (mu U/ml)", 10.0, 400.0, float(st.session_state.user_data['insulin']), 5.0)
                diabetes_pedigree = st.slider("Diabetes Pedigree", 0.05, 2.0, float(st.session_state.user_data['diabetes_pedigree']), 0.01)
                bp_meds = st.checkbox("On BP Meds", value=st.session_state.user_data['bp_meds'])
                has_diabetes = st.checkbox("History of Diabetes", value=st.session_state.user_data['has_diabetes'])
                pregnancies = st.number_input("Pregnancies", 0, 15, int(st.session_state.user_data['pregnancies']) if gender == "Female" else 0, disabled=gender == "Male")

        # Update current data dictionary in session
        current_data = {
            'age': age, 'gender': gender, 'height_cm': height_cm, 'weight_kg': weight_kg,
            'bmi': bmi_info['bmi'], 'is_smoker': is_smoker, 'cigs_per_day': cigs_per_day,
            'physical_activity': physical_activity, 'sleep_hours': sleep_hours,
            'diet_quality': diet_quality, 'stress_level': stress_level, 'alcohol_intake': alcohol_intake,
            'blood_pressure_sys': bp_sys, 'blood_pressure_dia': bp_dia, 'glucose': glucose,
            'cholesterol': cholesterol, 'resting_heart_rate': resting_hr, 'max_heart_rate': max_hr,
            'skin_thickness': skin_thickness, 'insulin': insulin, 'diabetes_pedigree': diabetes_pedigree,
            'pregnancies': pregnancies, 'chest_pain_type': chest_pain_type, 'exercise_angina': exercise_angina,
            'st_depression': st_depression, 'st_slope': st_slope, 'num_major_vessels': num_major_vessels,
            'thalassemia': thalassemia, 'bp_meds': bp_meds, 'has_diabetes': has_diabetes
        }
        st.session_state.user_data = current_data
        
        # Calculate instant live prediction
        live_preds = predictor.predict_all(current_data)
        st.session_state.prediction_results = live_preds

    # RIGHT COLUMN: LIVE INSTANT TELEMETRY HUD
    with col_telemetry:
        st.markdown(
            """
            <div class='telemetry-card'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
                    <span style='font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#38BDF8;'>
                        ⚡ Live Telemetry HUD
                    </span>
                    <span style='font-size:0.75rem; color:#94A3B8;'>Real-Time Active</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        
        # Live Badges
        bp_status = "Normal BP" if bp_sys < 120 and bp_dia < 80 else ("Elevated" if bp_sys < 130 and bp_dia < 80 else "Stage 1/2 HTN")
        bp_badge_color = "#10B981" if "Normal" in bp_status else ("#F59E0B" if "Elevated" in bp_status else "#EF4444")
        
        glyc_status = "Optimal Glycemic" if glucose < 100 else ("Pre-Diabetic Range" if glucose < 126 else "Elevated Glucose")
        glyc_badge_color = "#10B981" if "Optimal" in glyc_status else ("#F59E0B" if "Pre" in glyc_status else "#EF4444")
        
        st.markdown(
            f"""
            <div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;'>
                <span class='badge-pill' style='background:{bmi_info['badge_bg']}; color:{bmi_info['color']}; border:1px solid {bmi_info['color']};'>
                    BMI: {bmi_info['bmi']} ({bmi_info['category']})
                </span>
                <span class='badge-pill' style='background:rgba(255,255,255,0.1); color:{bp_badge_color}; border:1px solid {bp_badge_color};'>
                    BP: {int(bp_sys)}/{int(bp_dia)} ({bp_status})
                </span>
                <span class='badge-pill' style='background:rgba(255,255,255,0.1); color:{glyc_badge_color}; border:1px solid {glyc_badge_color};'>
                    {glucose} mg/dL ({glyc_status})
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Live Disease Risk Gauges
        st.plotly_chart(create_gauge_chart("🩸 Diabetes Risk", live_preds['diabetes']['probability_pct'], live_preds['diabetes']['color']), use_container_width=True)
        st.plotly_chart(create_gauge_chart("🫀 Heart Disease Risk", live_preds['heart']['probability_pct'], live_preds['heart']['color']), use_container_width=True)
        st.plotly_chart(create_gauge_chart("🩺 Hypertension Risk", live_preds['hypertension']['probability_pct'], live_preds['hypertension']['color']), use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.info("💡 Tip: Navigate to **📊 Risk Dashboard & Simulator** to run interactive live 'What-If' simulations!")

# ==============================================================================
# 3. RISK DASHBOARD & WHAT-IF SIMULATOR PAGE
# ==============================================================================
elif page == "📊 Risk Dashboard & Simulator":
    st.markdown("<h2 class='main-header'>📊 Risk Intelligence & Live Simulator</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Explore comprehensive risk estimates or simulate live health improvements with the What-If Simulator.</p>", unsafe_allow_html=True)
    
    if st.session_state.prediction_results is None:
        st.session_state.prediction_results = predictor.predict_all(st.session_state.user_data)

    preds = st.session_state.prediction_results
    user_data = st.session_state.user_data
    life_score = compute_lifestyle_quality_score(user_data)
    
    tab_dash, tab_sim = st.tabs(["📊 Comprehensive Risk Dashboard", "🔮 Live 'What-If' Health Simulator"])
    
    with tab_dash:
        col_ov1, col_ov2 = st.columns(2)
        
        with col_ov1:
            ls_color = life_score['color']
            ls_bg = life_score['badge_bg']
            ls_score = life_score['score']
            ls_rating = life_score['rating']
            st.markdown(
                f"<div class='metric-card' style='text-align:center; border-top:4px solid {ls_color};'>"
                f"<h5 style='color:#64748B; margin:0; text-transform:uppercase; font-size:0.82rem;'>Lifestyle Quality Index</h5>"
                f"<h1 style='color:{ls_color}; margin:6px 0; font-size:3rem; font-weight:800;'>{ls_score} <span style='font-size:1.2rem; color:#94A3B8; font-weight:400;'>/ 100</span></h1>"
                f"<span class='badge-pill' style='background:{ls_bg}; color:{ls_color};'>{ls_rating}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        with col_ov2:
            ov_score = preds['overall_risk_score']
            ov_color = '#10B981' if ov_score < 30 else ('#F59E0B' if ov_score <= 60 else '#EF4444')
            ov_bg = '#ECFDF5' if ov_score < 30 else ('#FFFBEB' if ov_score <= 60 else '#FEF2F2')
            st.markdown(
                f"<div class='metric-card' style='text-align:center; border-top:4px solid {ov_color};'>"
                f"<h5 style='color:#64748B; margin:0; text-transform:uppercase; font-size:0.82rem;'>Mean Composite Disease Risk</h5>"
                f"<h1 style='color:{ov_color}; margin:6px 0; font-size:3rem; font-weight:800;'>{ov_score}%</h1>"
                f"<span class='badge-pill' style='background:{ov_bg}; color:{ov_color};'>Across Screened Conditions</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("### 🩺 Multi-Condition Probabilistic Screening")
        cg1, cg2, cg3 = st.columns(3)
        with cg1:
            st.plotly_chart(create_gauge_chart("🩸 Type 2 Diabetes", preds['diabetes']['probability_pct'], preds['diabetes']['color']), use_container_width=True)
            st.markdown(f"<p style='font-size:0.85rem; color:#64748B; text-align:center;'>{preds['diabetes']['description']}</p>", unsafe_allow_html=True)
        with cg2:
            st.plotly_chart(create_gauge_chart("🫀 Coronary Heart Disease", preds['heart']['probability_pct'], preds['heart']['color']), use_container_width=True)
            st.markdown(f"<p style='font-size:0.85rem; color:#64748B; text-align:center;'>{preds['heart']['description']}</p>", unsafe_allow_html=True)
        with cg3:
            st.plotly_chart(create_gauge_chart("🩺 Hypertension", preds['hypertension']['probability_pct'], preds['hypertension']['color']), use_container_width=True)
            st.markdown(f"<p style='font-size:0.85rem; color:#64748B; text-align:center;'>{preds['hypertension']['description']}</p>", unsafe_allow_html=True)

        st.markdown("### 📈 Visual Biometric & Risk Profile")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_bar = go.Figure(go.Bar(
                x=['Diabetes', 'Heart Disease', 'Hypertension'],
                y=[preds['diabetes']['probability_pct'], preds['heart']['probability_pct'], preds['hypertension']['probability_pct']],
                marker_color=[preds['diabetes']['color'], preds['heart']['color'], preds['hypertension']['color']],
                text=[f"{preds['diabetes']['probability_pct']}%", f"{preds['heart']['probability_pct']}%", f"{preds['hypertension']['probability_pct']}%"],
                textposition='auto'
            ))
            fig_bar.update_layout(
                title="Disease Risk Screening Summary (%)",
                yaxis=dict(title="Probability (%)", range=[0, 100]),
                template="plotly_white", height=320, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_c2:
            fig_radar = create_biometric_radar_chart(user_data, life_score)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### 💡 Tailored Wellness Recommendations")
        insights = generate_personalized_insights(user_data, preds)
        for ins in insights:
            ins_color = ins['color']
            ins_bg = ins['badge_bg']
            ins_domain = ins['domain']
            ins_title = ins['title']
            ins_text = ins['text']
            st.markdown(
                f"<div class='insight-card' style='border-left:4px solid {ins_color};'>"
                f"<span class='badge-pill' style='background:{ins_bg}; color:{ins_color}; margin-bottom:6px;'>{ins_domain}</span>"
                f"<h5 style='margin:4px 0 6px 0; color:#0F172A;'>{ins_title}</h5>"
                f"<p style='margin:0; color:#334155; font-size:0.92rem; line-height:1.5;'>{ins_text}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    with tab_sim:
        st.markdown("### 🔮 Interactive 'What-If' Health Intervention Simulator")
        st.markdown("Simulate how specific lifestyle and biometric improvements would immediately reduce your predicted disease risk probabilities in real time.")
        
        sim_col_ctrl, sim_col_res = st.columns([2, 3])
        
        with sim_col_ctrl:
            st.markdown("#### 🎯 Configure Interventions")
            sim_weight_loss = st.slider("Weight Reduction (kg)", 0.0, 15.0, 4.0, 0.5)
            sim_sbp_drop = st.slider("Blood Pressure Drop (mmHg SBP)", 0.0, 30.0, 10.0, 1.0)
            sim_gluc_drop = st.slider("Fasting Glucose Reduction (mg/dL)", 0.0, 40.0, 15.0, 1.0)
            sim_chol_drop = st.slider("Total Cholesterol Drop (mg/dL)", 0.0, 60.0, 20.0, 5.0)
            sim_quit_smoke = st.checkbox("Stop Smoking Entirely", value=user_data.get('is_smoker', False))
            sim_boost_act = st.checkbox("Elevate Physical Activity to Active (5+ days/week)", value=True)
            
            intervention_payload = {
                'weight_loss_kg': sim_weight_loss,
                'sbp_reduction': sim_sbp_drop,
                'glucose_reduction': sim_gluc_drop,
                'chol_reduction': sim_chol_drop,
                'quit_smoking': sim_quit_smoke,
                'boost_activity': sim_boost_act
            }
            sim_results = simulate_lifestyle_intervention(user_data, predictor, intervention_payload)

        with sim_col_res:
            st.markdown("#### 📊 Projected Real-Time Impact")
            overall_drop = sim_results['overall_drop']
            base_score = sim_results['base_preds']['overall_risk_score']
            sim_score = sim_results['sim_preds']['overall_risk_score']
            st.markdown(
                f"<div class='sim-card'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<h4 style='color:#065F46; margin:0;'>Total Projected Risk Reduction</h4>"
                f"<span style='font-size:1.8rem; font-weight:800; color:#059669;'>-{overall_drop}%</span>"
                f"</div>"
                f"<p style='color:#047857; font-size:0.88rem; margin:6px 0 0 0;'>"
                f"Overall vulnerability decreases from <b>{base_score}%</b> down to <b>{sim_score}%</b>!"
                f"</p></div>",
                unsafe_allow_html=True
            )
            
            fig_sim = go.Figure()
            diseases_labels = ['Diabetes Risk', 'Heart Disease Risk', 'Hypertension Risk']
            base_vals = [
                sim_results['base_preds']['diabetes']['probability_pct'],
                sim_results['base_preds']['heart']['probability_pct'],
                sim_results['base_preds']['hypertension']['probability_pct']
            ]
            sim_vals = [
                sim_results['sim_preds']['diabetes']['probability_pct'],
                sim_results['sim_preds']['heart']['probability_pct'],
                sim_results['sim_preds']['hypertension']['probability_pct']
            ]
            fig_sim.add_trace(go.Bar(name='Current Baseline Risk', x=diseases_labels, y=base_vals, marker_color='#94A3B8'))
            fig_sim.add_trace(go.Bar(name='Projected Post-Intervention Risk', x=diseases_labels, y=sim_vals, marker_color='#10B981'))
            fig_sim.update_layout(
                barmode='group',
                title="Before vs. After Intervention Risk Comparison (%)",
                yaxis=dict(title="Probability (%)", range=[0, 100]),
                template="plotly_white", height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_sim, use_container_width=True)
            
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.metric("Diabetes Risk", f"{sim_results['sim_preds']['diabetes']['probability_pct']}%", f"-{sim_results['diabetes_drop']}%")
            with col_d2:
                st.metric("Heart Disease", f"{sim_results['sim_preds']['heart']['probability_pct']}%", f"-{sim_results['heart_drop']}%")
            with col_d3:
                st.metric("Hypertension", f"{sim_results['sim_preds']['hypertension']['probability_pct']}%", f"-{sim_results['hypertension_drop']}%")


# ==============================================================================
# 4. EXPLAINABLE AI (SHAP) PAGE
# ==============================================================================
elif page == "🔍 Explainable AI (SHAP)":
    st.markdown("<h2 class='main-header'>🔍 Explainable AI (SHAP Interpretability)</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Inspect exact mathematical feature contributions behind individual predictions using Shapley values.</p>", unsafe_allow_html=True)
    
    if st.session_state.prediction_results is None:
        st.session_state.prediction_results = predictor.predict_all(st.session_state.user_data)
        
    preds = st.session_state.prediction_results
    
    selected_disease = st.selectbox(
        "Select Screening Model to Explain",
        ["diabetes", "heart", "hypertension"],
        format_func=lambda x: {
            "diabetes": "🩸 Diabetes Risk Model (XGBoost)",
            "heart": "🫀 Heart Disease Risk Model (Logistic Regression)",
            "hypertension": "🩺 Hypertension Risk Model (Random Forest)"
        }[x]
    )
    
    with st.spinner("Computing local SHAP feature attributions..."):
        input_df = preds[selected_disease]['input_engineered']
        exp_dict = explainer.explain_prediction(selected_disease, input_df)
        
    col_exp1, col_exp2 = st.columns([3, 2])
    
    with col_exp1:
        st.markdown("#### 📊 Local Feature Contributions (SHAP Values)")
        fig_shap = explainer.create_shap_waterfall_chart(exp_dict)
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_exp2:
        st.markdown("#### 📝 Clinical Interpretation")
        narrative_html = exp_dict['narrative']
        st.markdown(
            f"<div class='metric-card'>{narrative_html}</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("#### 🔍 Primary Risk Accelerators")
        for feat_item in exp_dict['top_increasing'][:3]:
            feat_clean = feat_item['feature'].replace('_', ' ').title()
            shap_val = feat_item['shap_value']
            st.markdown(f"🔴 **{feat_clean}:** Contribution `+{shap_val:.3f}`")

        st.markdown("#### 🛡️ Primary Protective Attributes")
        for feat_item in exp_dict['top_decreasing'][:2]:
            feat_clean = feat_item['feature'].replace('_', ' ').title()
            shap_val = feat_item['shap_value']
            st.markdown(f"🟢 **{feat_clean}:** Contribution `{shap_val:.3f}`")


# ==============================================================================
# 5. DATA INSIGHTS & MODEL BENCHMARKS PAGE
# ==============================================================================
elif page == "📈 Data Insights & Models":
    st.markdown("<h2 class='main-header'>📈 Data Science Insights & Benchmarks</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Exploratory Data Analysis and verified multi-algorithm benchmark metrics.</p>", unsafe_allow_html=True)
    
    tab_eda, tab_models = st.tabs(["📊 Exploratory Data Analysis (EDA)", "🏆 Model Benchmarks & Comparison"])
    
    with tab_eda:
        eda_disease = st.selectbox(
            "Select Benchmark Dataset",
            ["Diabetes (PIMA Dataset)", "Heart Disease (Cleveland UCI)", "Hypertension (Framingham Study)"]
        )
        
        fig_dir = os.path.join(PROJECT_ROOT, 'outputs', 'figures')
        
        if "Diabetes" in eda_disease:
            st.markdown("**PIMA Indians Diabetes Dataset (NIDDK / UCI):** 768 patient records · 9 features · Target: `Outcome` (34.9% positive).")
            col_img1, col_img2 = st.columns(2)
            p1 = os.path.join(fig_dir, 'diabetes_correlation_matrix.png')
            p2 = os.path.join(fig_dir, 'diabetes_target_distribution.png')
            if os.path.exists(p1):
                col_img1.image(p1, caption="Diabetes Feature Correlation Matrix", use_container_width=True)
            if os.path.exists(p2):
                col_img2.image(p2, caption="Target Class Prevalence", use_container_width=True)
            p3 = os.path.join(fig_dir, 'diabetes_model_comparison.png')
            if os.path.exists(p3):
                st.image(p3, caption="Diabetes Model Comparison", use_container_width=True)

        elif "Heart" in eda_disease:
            st.markdown("**UCI Cleveland Heart Disease Dataset:** 303 clinical records · 14 features · Target: `target` (54.5% positive).")
            col_img1, col_img2 = st.columns(2)
            p1 = os.path.join(fig_dir, 'heart_correlation_matrix.png')
            p2 = os.path.join(fig_dir, 'heart_age_distribution.png')
            if os.path.exists(p1):
                col_img1.image(p1, caption="Heart Disease Feature Correlation Matrix", use_container_width=True)
            if os.path.exists(p2):
                col_img2.image(p2, caption="Heart Disease Distribution by Age", use_container_width=True)
            p3 = os.path.join(fig_dir, 'heart_model_comparison.png')
            if os.path.exists(p3):
                st.image(p3, caption="Heart Disease Model Comparison", use_container_width=True)

        else:
            st.markdown("**Framingham Cardiovascular Study:** 4,240 longitudinal cohort participants · 16 features · Target: `prevalentHyp` (31.1% prevalence).")
            col_img1, col_img2 = st.columns(2)
            p1 = os.path.join(fig_dir, 'hypertension_correlation_matrix.png')
            p2 = os.path.join(fig_dir, 'hypertension_sysbp_boxplot.png')
            if os.path.exists(p1):
                col_img1.image(p1, caption="Hypertension Correlation Matrix", use_container_width=True)
            if os.path.exists(p2):
                col_img2.image(p2, caption="Systolic BP vs Hypertension Diagnosis", use_container_width=True)
            p3 = os.path.join(fig_dir, 'hypertension_model_comparison.png')
            if os.path.exists(p3):
                st.image(p3, caption="Hypertension Model Comparison", use_container_width=True)

    with tab_models:
        benchmarks = metrics_data.get('all_benchmarks', [])
        if benchmarks:
            df_b = pd.DataFrame(benchmarks)
            
            sel_d = st.selectbox("Filter by Disease Module", ["All Diseases", "diabetes", "heart", "hypertension"])
            df_b_view = df_b[df_b['disease'] == sel_d] if sel_d != "All Diseases" else df_b
                
            display_cols = ['disease', 'model_name', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc', 'composite_screening_score']
            available_cols = [c for c in display_cols if c in df_b_view.columns]
            df_display = df_b_view[available_cols].copy()
            rename_map = {
                'disease': 'Disease', 'model_name': 'Algorithm',
                'accuracy': 'Accuracy', 'precision': 'Precision',
                'recall': 'Recall', 'specificity': 'Specificity',
                'f1': 'F1-Score', 'roc_auc': 'ROC-AUC',
                'composite_screening_score': 'Selection Score'
            }
            df_display.rename(columns=rename_map, inplace=True)
            
            for col in ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC', 'Selection Score']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")
                
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            st.markdown("#### 🧩 Interactive Confusion Matrix Visualizer")
            cm_col1, cm_col2 = st.columns(2)
            with cm_col1:
                cm_disease = st.selectbox("Disease", ["diabetes", "heart", "hypertension"], key="cm_dis")
            with cm_col2:
                model_options = [b['model_name'] for b in benchmarks if b['disease'] == cm_disease]
                cm_model = st.selectbox("Algorithm", model_options, key="cm_mod")
                
            selected_bench = next((b for b in benchmarks if b['disease'] == cm_disease and b['model_name'] == cm_model), None)
            if selected_bench and 'confusion_matrix' in selected_bench:
                cm_data = selected_bench['confusion_matrix']
                fig_cm = px.imshow(
                    cm_data,
                    labels=dict(x="Predicted Condition", y="Actual Condition", color="Count"),
                    x=['Negative (0)', 'Positive (1)'],
                    y=['Negative (0)', 'Positive (1)'],
                    text_auto=True,
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.warning("No benchmark data found. Please ensure model training has completed.")


# ==============================================================================
# 6. ABOUT & MEDICAL DISCLAIMER PAGE
# ==============================================================================
elif page == "ℹ️ About & Medical Disclaimer":
    st.markdown("<h2 class='main-header'>ℹ️ About HealthPredict</h2>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>System Architecture, Data Provenance, and Ethical Artificial Intelligence.</p>", unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns([3, 2])
    
    with col_a1:
        st.markdown("""
### 🏗️ Technical Architecture

| Component | Details |
|:---|:---|
| **Datasets** | PIMA Indians Diabetes (NIDDK), Cleveland Heart Disease (UCI), Framingham Heart Study |
| **Preprocessing** | ColumnTransformer with Median Imputation + StandardScaler (no data leakage) |
| **Algorithms** | Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost (5 per disease) |
| **Selection Criterion** | 0.40×Recall + 0.30×ROC-AUC + 0.15×F1 + 0.15×Accuracy |
| **Explainability** | SHAP (Shapley Additive Explanations) – local instance-level attribution |
| **Interface** | Streamlit with Plotly interactive charts |

**Feature Engineering Highlights:**
- *Glucose-BMI Interaction Index* (Diabetes)  
- *Heart Rate Reserve Ratio* = thalach / (220 - age) (Heart Disease)  
- *Mean Arterial Pressure* = DBP + (SBP - DBP)/3 (Hypertension)  
- *Pulse Pressure* = SBP - DBP (Hypertension)
        """)

    with col_a2:
        st.markdown(
            "<div class='metric-card'>"
            "<h4 style='color:#0284C7; margin-bottom:8px;'>🔒 100% Client-Side Privacy</h4>"
            "<p style='font-size:0.88rem; color:#475569;'>"
            "This application executes completely in local memory. No user-entered health parameters "
            "are ever sent to remote cloud APIs, third-party analytics, or persistent databases."
            "</p></div>",
            unsafe_allow_html=True
        )
        
        st.markdown(
            "<div class='metric-card' style='margin-top:0.5rem;'>"
            "<h4 style='color:#059669; margin-bottom:8px;'>📊 Benchmark Datasets</h4>"
            "<p style='font-size:0.85rem; color:#475569; line-height:1.6;'>"
            "<b>Diabetes:</b> PIMA Indians (NIDDK) · 768 rows<br>"
            "<b>Heart Disease:</b> Cleveland (UCI) · 303 rows<br>"
            "<b>Hypertension:</b> Framingham Study · 4,240 rows"
            "</p></div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div class='disclaimer-box'>"
        "<h4 style='margin:0 0 6px 0; color:#0369A1;'>⚖️ Mandatory Medical & Legal Disclaimer</h4>"
        "HealthPredict is developed strictly for educational, research, and non-diagnostic risk-screening demonstration. "
        "Predictions reflect probabilistic mathematical associations and must NOT be used for medical self-diagnosis, "
        "prescriptions, or clinical decision-making. If you have concerns about your blood pressure, cardiovascular health, "
        "or blood sugar levels, please consult a licensed medical doctor or qualified healthcare professional."
        "</div>",
        unsafe_allow_html=True
    )
