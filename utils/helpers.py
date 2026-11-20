"""
Helper Utilities Module for HealthPredict.
Enhanced with interactive Plotly gauge generators, What-If simulation engine,
biometric radar profiling, and modern SaaS healthcare styling.
"""

import numpy as np
import plotly.graph_objects as go


def calculate_bmi(height_cm: float, weight_kg: float) -> dict:
    """
    Calculates Body Mass Index (BMI) and returns WHO classification.
    """
    if height_cm <= 0 or weight_kg <= 0:
        return {'bmi': 0.0, 'category': 'Invalid', 'color': '#757575', 'badge_bg': '#F1F5F9'}
        
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    bmi = round(bmi, 1)
    
    if bmi < 18.5:
        category = 'Underweight'
        color = '#0288D1'  # Cyan/Blue
        badge_bg = '#E1F5FE'
    elif 18.5 <= bmi < 25.0:
        category = 'Normal Weight'
        color = '#10B981'  # Emerald Green
        badge_bg = '#ECFDF5'
    elif 25.0 <= bmi < 30.0:
        category = 'Overweight'
        color = '#F59E0B'  # Amber Orange
        badge_bg = '#FFFBEB'
    elif 30.0 <= bmi < 35.0:
        category = 'Obesity (Class I)'
        color = '#EF4444'  # Crimson Red
        badge_bg = '#FEF2F2'
    else:
        category = 'Severe Obesity (Class II/III)'
        color = '#991B1B'  # Deep Wine Red
        badge_bg = '#FEE2E2'
        
    return {
        'bmi': bmi,
        'category': category,
        'color': color,
        'badge_bg': badge_bg
    }


def compute_lifestyle_quality_score(user_data: dict) -> dict:
    """
    Computes a composite Lifestyle Quality Index (0-100).
    """
    score = 100
    deductions = []
    
    # 1. Physical Activity (Max deduction: 20)
    activity = user_data.get('physical_activity', 'Moderate (3-4 days/week)')
    if 'Sedentary' in activity:
        score -= 20
        deductions.append(('Physical Activity', -20, 'Sedentary lifestyle significantly elevates cardiovascular & metabolic vulnerability.'))
    elif 'Light' in activity:
        score -= 10
        deductions.append(('Physical Activity', -10, 'Light activity provides partial benefits; increasing aerobic frequency is recommended.'))

    # 2. Smoking (Max deduction: 25)
    is_smoker = user_data.get('is_smoker', False)
    cigs = user_data.get('cigs_per_day', 0)
    if is_smoker:
        smoke_penalty = min(25, 15 + int(cigs * 0.5))
        score -= smoke_penalty
        deductions.append(('Smoking', -smoke_penalty, f'Current smoking ({cigs} cigs/day) induces oxidative stress and vascular stiffening.'))

    # 3. Sleep (Max deduction: 15)
    sleep_hrs = user_data.get('sleep_hours', 7.5)
    if sleep_hrs < 6.0:
        score -= 15
        deductions.append(('Sleep Duration', -15, 'Short sleep (<6h) impairs glucose tolerance, leptin regulation, and elevates cortisol.'))
    elif sleep_hrs > 9.5:
        score -= 8
        deductions.append(('Sleep Duration', -8, 'Excessive sleep duration may correlate with systemic fatigue or disrupted sleep architecture.'))

    # 4. Diet Quality (Max deduction: 20)
    diet = user_data.get('diet_quality', 'Balanced')
    if 'Poor' in diet or 'High Fast Food' in diet:
        score -= 20
        deductions.append(('Diet Quality', -20, 'High consumption of ultra-processed foods, refined carbs, and saturated fats.'))
    elif 'Average' in diet:
        score -= 8
        deductions.append(('Diet Quality', -8, 'Diet would benefit from increased whole grains, dietary fiber, and plant antioxidants.'))

    # 5. Stress Level (Max deduction: 10)
    stress = user_data.get('stress_level', 'Moderate')
    if 'Severe' in stress or 'Chronic' in stress:
        score -= 10
        deductions.append(('Stress Level', -10, 'Chronic sympathetic activation sustains elevated resting blood pressure.'))
    elif 'High' in stress:
        score -= 6
        deductions.append(('Stress Level', -6, 'High stress levels trigger subclinical inflammatory cascades.'))

    # 6. Alcohol (Max deduction: 10)
    alcohol = user_data.get('alcohol_intake', 'Occasional / Light')
    if 'Heavy' in alcohol or 'Frequent' in alcohol:
        score -= 10
        deductions.append(('Alcohol Intake', -10, 'Heavy alcohol intake contributes to hypertension and hepatic stress.'))

    score = max(10, min(100, score))
    
    if score >= 80:
        rating = 'Optimal Lifestyle'
        color = '#10B981'
        badge_bg = '#ECFDF5'
    elif score >= 60:
        rating = 'Good – Minor Adjustments'
        color = '#0284C7'
        badge_bg = '#E0F2FE'
    elif score >= 40:
        rating = 'Moderate – Multiple Risk Flags'
        color = '#F59E0B'
        badge_bg = '#FFFBEB'
    else:
        rating = 'High Risk Lifestyle Pattern'
        color = '#EF4444'
        badge_bg = '#FEF2F2'

    return {
        'score': score,
        'rating': rating,
        'color': color,
        'badge_bg': badge_bg,
        'deductions': deductions
    }


def create_gauge_chart(title: str, value_pct: float, risk_color: str, low_thresh: float = 30.0, high_thresh: float = 60.0):
    """
    Renders an animated, sleek Plotly speedometer gauge chart.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={'suffix': "%", 'font': {'size': 28, 'color': '#1E293B', 'family': 'Inter, sans-serif'}},
        title={'text': f"<b>{title}</b>", 'font': {'size': 15, 'color': '#475569'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#CBD5E1', 'tickvals': [0, 30, 60, 100]},
            'bar': {'color': risk_color, 'thickness': 0.28},
            'bgcolor': "#FFFFFF",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, low_thresh], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [low_thresh, high_thresh], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [high_thresh, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#1E293B", 'width': 3},
                'thickness': 0.8,
                'value': value_pct
            }
        }
    ))
    
    fig.update_layout(
        height=190,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    return fig


def create_biometric_radar_chart(user_data: dict, life_score: dict):
    """
    Generates a biometric vulnerability radar profile comparing current profile vs optimal benchmark.
    """
    categories = ['Glycemic Strain', 'Blood Pressure', 'Lipid Strain', 'Body Mass (BMI)', 'Lifestyle Strain']
    
    val_glycemic = min(100, max(10, ((user_data.get('glucose', 90) - 70) / 130) * 100))
    val_bp = min(100, max(10, ((user_data.get('blood_pressure_sys', 120) - 90) / 90) * 100))
    val_chol = min(100, max(10, ((user_data.get('cholesterol', 200) - 140) / 160) * 100))
    val_bmi = min(100, max(10, ((user_data.get('bmi', 24) - 18.5) / 16.5) * 100))
    val_lifestyle = 100 - life_score['score']
    
    current_values = [val_glycemic, val_bp, val_chol, val_bmi, val_lifestyle]
    optimal_values = [20, 25, 25, 25, 15]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=optimal_values + [optimal_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.20)',
        line=dict(color='#10B981', width=1.5, dash='dash'),
        name='Optimal Clinical Target'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=current_values + [current_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(2, 132, 199, 0.35)',
        line=dict(color='#0284C7', width=2.5),
        name='Your Current Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9, color='#94A3B8')),
            angularaxis=dict(tickfont=dict(size=11, color='#334155', family='Inter, sans-serif'))
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        template="plotly_white",
        height=320,
        margin=dict(l=35, r=35, t=25, b=45)
    )
    return fig


def simulate_lifestyle_intervention(base_data: dict, predictor, intervention: dict) -> dict:
    """
    Live 'What-If' Simulation Engine:
    Modifies baseline parameters based on hypothetical user lifestyle interventions
    and recalculates disease probabilities to show actionable projected risk reduction.
    """
    sim_data = base_data.copy()
    
    # 1. Weight Loss Delta
    weight_loss_kg = intervention.get('weight_loss_kg', 0.0)
    if weight_loss_kg > 0:
        sim_data['weight_kg'] = max(40.0, sim_data['weight_kg'] - weight_loss_kg)
        height_m = sim_data['height_cm'] / 100.0
        sim_data['bmi'] = round(sim_data['weight_kg'] / (height_m ** 2), 1)
        
    # 2. Blood Pressure Reduction
    sbp_reduction = intervention.get('sbp_reduction', 0.0)
    if sbp_reduction > 0:
        sim_data['blood_pressure_sys'] = max(90.0, sim_data['blood_pressure_sys'] - sbp_reduction)
        sim_data['blood_pressure_dia'] = max(60.0, sim_data['blood_pressure_dia'] - (sbp_reduction * 0.6))
        
    # 3. Fasting Glucose Optimization
    glucose_reduction = intervention.get('glucose_reduction', 0.0)
    if glucose_reduction > 0:
        sim_data['glucose'] = max(75.0, sim_data['glucose'] - glucose_reduction)
        
    # 4. Cholesterol Optimization
    chol_reduction = intervention.get('chol_reduction', 0.0)
    if chol_reduction > 0:
        sim_data['cholesterol'] = max(140.0, sim_data['cholesterol'] - chol_reduction)

    # 5. Quit Smoking
    if intervention.get('quit_smoking', False):
        sim_data['is_smoker'] = False
        sim_data['cigs_per_day'] = 0
        
    # 6. Physical Activity Boost
    if intervention.get('boost_activity', False):
        sim_data['physical_activity'] = 'Active (5+ days/week)'
        sim_data['max_heart_rate'] = min(200.0, sim_data.get('max_heart_rate', 140.0) + 12.0)
        sim_data['resting_heart_rate'] = max(55.0, sim_data.get('resting_heart_rate', 75.0) - 6.0)

    # Predict simulated outcomes
    base_preds = predictor.predict_all(base_data)
    sim_preds = predictor.predict_all(sim_data)
    
    # Calculate risk drops
    d_drop = round(base_preds['diabetes']['probability_pct'] - sim_preds['diabetes']['probability_pct'], 1)
    h_drop = round(base_preds['heart']['probability_pct'] - sim_preds['heart']['probability_pct'], 1)
    hyp_drop = round(base_preds['hypertension']['probability_pct'] - sim_preds['hypertension']['probability_pct'], 1)
    overall_drop = round(base_preds['overall_risk_score'] - sim_preds['overall_risk_score'], 1)
    
    return {
        'sim_data': sim_data,
        'base_preds': base_preds,
        'sim_preds': sim_preds,
        'diabetes_drop': d_drop,
        'heart_drop': h_drop,
        'hypertension_drop': hyp_drop,
        'overall_drop': overall_drop
    }


def generate_personalized_insights(user_data: dict, prediction_results: dict) -> list:
    """
    Generates tailored, general wellness recommendations.
    """
    insights = []
    
    # 1. Physical Activity
    activity = user_data.get('physical_activity', 'Moderate (3-4 days/week)')
    if 'Sedentary' in activity or 'Light' in activity:
        insights.append({
            'domain': '🏃 Physical Activity & Movement',
            'title': 'Gradual Increase in Aerobic Conditioning',
            'icon': '🏃',
            'color': '#0284C7',
            'badge_bg': '#E0F2FE',
            'text': 'Aim for at least 150 minutes of moderate-intensity aerobic physical activity per week (such as brisk walking, swimming, or cycling), combined with 2 days of muscle-strengthening exercises, as medically appropriate.'
        })

    # 2. BMI & Weight Management
    bmi_info = calculate_bmi(user_data.get('height_cm', 170), user_data.get('weight_kg', 70))
    if bmi_info['bmi'] >= 25.0:
        insights.append({
            'domain': '⚖️ Weight & Body Composition',
            'title': 'Targeted Metabolic Weight Optimization',
            'icon': '⚖️',
            'color': '#D97706',
            'badge_bg': '#FEF3C7',
            'text': f"Your calculated BMI is {bmi_info['bmi']} kg/m² ({bmi_info['category']}). Clinical evidence indicates that a modest 5–7% reduction in body weight reduces visceral adipose tissue, relieving insulin resistance and decreasing vascular strain."
        })

    # 3. Smoking Cessation
    if user_data.get('is_smoker', False):
        insights.append({
            'domain': '🚭 Tobacco Cessation',
            'title': 'Vascular Recovery via Smoking Cessation',
            'icon': '🚭',
            'color': '#DC2626',
            'badge_bg': '#FEE2E2',
            'text': 'Cigarette smoke induces endothelial dysfunction and accelerates coronary atheroma formation. Within 12 months of cessation, excess coronary heart disease risk is halved compared to continuing smokers.'
        })

    # 4. Blood Pressure & Sodium
    sys_bp = user_data.get('blood_pressure_sys', 120)
    dia_bp = user_data.get('blood_pressure_dia', 80)
    if sys_bp >= 130 or dia_bp >= 85:
        insights.append({
            'domain': '🫀 Hemodynamic & Blood Pressure Support',
            'title': 'DASH Nutritional Strategy & Sodium Balance',
            'icon': '🫀',
            'color': '#E11D48',
            'badge_bg': '#FFE4E6',
            'text': f'With a blood pressure reading of {sys_bp}/{dia_bp} mmHg, adopting dietary patterns rich in potassium, magnesium, and dietary fiber (DASH diet) while capping sodium under 2,000 mg/day fosters arterial compliance.'
        })

    # 5. Fasting Blood Glucose
    glucose = user_data.get('glucose', 95)
    if glucose >= 100:
        insights.append({
            'domain': '🩸 Glycemic Balance',
            'title': 'Low-Glycemic Nutrition & Soluble Fiber',
            'icon': '🩸',
            'color': '#7C3AED',
            'badge_bg': '#EDE9FE',
            'text': f'Fasting glucose of {glucose} mg/dL falls in the elevated pre-diabetic screening zone (normal < 100 mg/dL). Prioritize complex legumes, oats, and leafy greens while minimizing liquid sugars and refined flour.'
        })

    # 6. Sleep Schedule
    sleep_hrs = user_data.get('sleep_hours', 7.5)
    if sleep_hrs < 7.0:
        insights.append({
            'domain': '🌙 Circadian Rhythm & Sleep',
            'title': 'Restorative Sleep Scheduling',
            'icon': '🌙',
            'color': '#4F46E5',
            'badge_bg': '#EEF2FF',
            'text': f'Getting {sleep_hrs} hours of sleep nightly is below the 7–9 hour target. Consistent sleep timing stabilizes nocturnal blood pressure dipping and prevents sympathetic overdrive.'
        })

    if not insights:
        insights.append({
            'domain': '🌟 Preventative Wellness',
            'title': 'Sustain Current Positive Habits',
            'icon': '🌟',
            'color': '#059669',
            'badge_bg': '#D1FAE5',
            'text': 'Your current inputs indicate strong adherence to healthy lifestyle parameters. Continue regular preventative health check-ups, balanced nutrient density, and daily movement.'
        })

    return insights


def validate_user_inputs(user_data: dict) -> list:
    """
    Validates input parameters for physiological plausibility.
    """
    warnings = []
    
    age = user_data.get('age', 35)
    if age < 18 or age > 110:
        warnings.append(f"Age ({age}) must be between 18 and 110.")
        
    height = user_data.get('height_cm', 170)
    if height < 90 or height > 240:
        warnings.append(f"Height ({height} cm) is outside standard adult ranges (90-240 cm).")
        
    weight = user_data.get('weight_kg', 70)
    if weight < 25 or weight > 250:
        warnings.append(f"Weight ({weight} kg) is outside standard adult ranges (25-250 kg).")
        
    sys_bp = user_data.get('blood_pressure_sys', 120)
    dia_bp = user_data.get('blood_pressure_dia', 80)
    if sys_bp <= dia_bp:
        warnings.append(f"Systolic BP ({sys_bp} mmHg) must be strictly higher than Diastolic BP ({dia_bp} mmHg).")
    if sys_bp < 70 or sys_bp > 260:
        warnings.append(f"Systolic BP ({sys_bp} mmHg) is outside standard measurable limits (70-260 mmHg).")
    if dia_bp < 40 or dia_bp > 160:
        warnings.append(f"Diastolic BP ({dia_bp} mmHg) is outside standard measurable limits (40-160 mmHg).")

    sleep = user_data.get('sleep_hours', 7)
    if sleep < 0 or sleep > 24:
        warnings.append(f"Sleep hours ({sleep}) must be between 0 and 24.")

    glucose = user_data.get('glucose', 90)
    if glucose < 40 or glucose > 500:
        warnings.append(f"Glucose level ({glucose} mg/dL) is outside standard limits (40-500 mg/dL).")
        
    return warnings


def get_custom_css():
    """
    Returns modern, polished CSS for a SaaS-level healthcare analytics aesthetic.
    """
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main-header {
            color: #0F172A;
            font-weight: 800;
            font-size: 2.2rem;
            letter-spacing: -0.02em;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #64748B;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }
        
        /* Modern Glass Cards */
        .metric-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 1.4rem;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05), 0 2px 6px -1px rgba(15, 23, 42, 0.03);
            border: 1px solid #E2E8F0;
            margin-bottom: 1.2rem;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 28px -4px rgba(15, 23, 42, 0.1), 0 4px 10px -2px rgba(15, 23, 42, 0.04);
            border-color: #CBD5E1;
        }
        
        /* Interactive Telemetry Box */
        .telemetry-card {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border-radius: 16px;
            padding: 1.5rem;
            color: #FFFFFF;
            box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 1rem;
        }
        
        .pulse-dot {
            height: 10px;
            width: 10px;
            background-color: #10B981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse 1.6s infinite;
            margin-right: 6px;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        
        /* Badges */
        .badge-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        
        .disclaimer-box {
            background: #F8FAFC;
            border-left: 4px solid #0284C7;
            padding: 1.1rem 1.4rem;
            border-radius: 10px;
            font-size: 0.88rem;
            color: #334155;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid #E2E8F0;
            border-left-width: 4px;
        }
        
        .insight-card {
            background: #FFFFFF;
            padding: 1.1rem 1.3rem;
            border-radius: 12px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            margin-bottom: 0.9rem;
            transition: transform 0.2s ease;
        }
        .insight-card:hover {
            transform: translateX(4px);
        }
        
        .hero-gradient {
            background: linear-gradient(135deg, #0284C7 0%, #0369A1 50%, #0F172A 100%);
            border-radius: 18px;
            padding: 2.2rem 2rem;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 12px 30px -6px rgba(2, 132, 199, 0.35);
        }
        
        /* Simulator Card */
        .sim-card {
            background: #F0FDF4;
            border: 1px solid #BBF7D0;
            border-radius: 14px;
            padding: 1.3rem;
            margin-bottom: 1rem;
        }
    </style>
    """
