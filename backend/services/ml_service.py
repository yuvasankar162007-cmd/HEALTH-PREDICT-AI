"""
Machine Learning Service for HealthPredict.
Coordinates multi-pipeline inference, feature engineering, sensitivity calibration, and preset generation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.feature_engineering import (
    engineer_diabetes_features,
    engineer_heart_features,
    engineer_hypertension_features
)
from utils.helpers import (
    calculate_bmi,
    compute_lifestyle_quality_score,
    generate_personalized_insights,
    validate_user_inputs
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


class MLService:
    """Unified Machine Learning prediction service."""
    
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MLService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, models_dir: str = MODELS_DIR):
        if self._initialized:
            return
            
        self.models_dir = models_dir
        self.pipelines: Dict[str, Any] = {}
        self.model_metrics: Dict[str, Any] = {}
        self.feature_metadata: Dict[str, Any] = {}
        
        self.load_artifacts()
        self._initialized = True

    def load_artifacts(self):
        """Loads serialized model pipelines and benchmark metadata."""
        for disease in ['diabetes', 'heart', 'hypertension']:
            path = os.path.join(self.models_dir, f"{disease}_pipeline.pkl")
            if os.path.exists(path):
                try:
                    self.pipelines[disease] = joblib.load(path)
                except Exception as e:
                    print(f"Error loading {disease} pipeline: {e}")
                    
        metrics_path = os.path.join(self.models_dir, 'model_metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    self.model_metrics = json.load(f)
            except Exception as e:
                print(f"Error loading model metrics: {e}")
                
        meta_path = os.path.join(self.models_dir, 'feature_metadata.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.feature_metadata = json.load(f)
            except Exception as e:
                print(f"Error loading feature metadata: {e}")

    def get_risk_category(self, probability: float, low_thresh: float = 0.30, high_thresh: float = 0.60) -> Dict[str, str]:
        """Categorizes disease probability into educational screening bands."""
        if probability < low_thresh:
            return {
                'level': 'LOW RISK',
                'color': '#10B981',  # Emerald Green
                'badge_bg': '#ECFDF5',
                'description': 'Estimated risk is within the low range compared to benchmark populations.'
            }
        elif probability <= high_thresh:
            return {
                'level': 'MODERATE RISK',
                'color': '#F59E0B',  # Amber/Orange
                'badge_bg': '#FFFBEB',
                'description': 'Estimated risk is elevated. Proactive lifestyle modifications are advised.'
            }
        else:
            return {
                'level': 'HIGH RISK',
                'color': '#EF4444',  # Crimson Red
                'badge_bg': '#FEF2F2',
                'description': 'Estimated risk is high. Consultation with a qualified healthcare professional is strongly advised.'
            }

    def predict_diabetes(self, user_dict: Dict[str, Any], low_thresh: float = 0.30, high_thresh: float = 0.60) -> Dict[str, Any]:
        """Runs Type 2 Diabetes screening prediction."""
        if 'diabetes' not in self.pipelines:
            raise RuntimeError("Diabetes model pipeline is not loaded.")
            
        df_raw = pd.DataFrame([{
            'Pregnancies': user_dict.get('pregnancies', 0),
            'Glucose': user_dict.get('glucose', 100.0),
            'BloodPressure': user_dict.get('blood_pressure_dia', 75.0),
            'SkinThickness': user_dict.get('skin_thickness', 20.0),
            'Insulin': user_dict.get('insulin', 80.0),
            'BMI': user_dict.get('bmi', 24.5),
            'DiabetesPedigreeFunction': user_dict.get('diabetes_pedigree', 0.47),
            'Age': user_dict.get('age', 35)
        }])
        
        df_engineered = engineer_diabetes_features(df_raw)
        pipeline = self.pipelines['diabetes']
        prob = float(pipeline.predict_proba(df_engineered)[0, 1])
        cat = self.get_risk_category(prob, low_thresh, high_thresh)
        
        return {
            'disease': 'Type 2 Diabetes Risk',
            'probability': prob,
            'probability_pct': round(prob * 100, 1),
            'category': cat['level'],
            'color': cat['color'],
            'badge_bg': cat['badge_bg'],
            'description': cat['description'],
            'input_engineered': df_engineered
        }

    def predict_heart_disease(self, user_dict: Dict[str, Any], low_thresh: float = 0.30, high_thresh: float = 0.60) -> Dict[str, Any]:
        """Runs Coronary Heart Disease screening prediction."""
        if 'heart' not in self.pipelines:
            raise RuntimeError("Heart Disease model pipeline is not loaded.")
            
        df_raw = pd.DataFrame([{
            'age': user_dict.get('age', 50),
            'sex': 1 if str(user_dict.get('gender', 'Male')).lower() == 'male' else 0,
            'cp': user_dict.get('chest_pain_type', 0),
            'trestbps': user_dict.get('blood_pressure_sys', 120.0),
            'chol': user_dict.get('cholesterol', 210.0),
            'fbs': 1 if user_dict.get('glucose', 100.0) > 120 else 0,
            'restecg': user_dict.get('rest_ecg', 0),
            'thalach': user_dict.get('max_heart_rate', 150.0),
            'exang': user_dict.get('exercise_angina', 0),
            'oldpeak': user_dict.get('st_depression', 0.5),
            'slope': user_dict.get('st_slope', 1),
            'ca': user_dict.get('num_major_vessels', 0),
            'thal': user_dict.get('thalassemia', 2)
        }])
        
        df_engineered = engineer_heart_features(df_raw)
        pipeline = self.pipelines['heart']
        prob = float(pipeline.predict_proba(df_engineered)[0, 1])
        cat = self.get_risk_category(prob, low_thresh, high_thresh)
        
        return {
            'disease': 'Coronary Heart Disease',
            'probability': prob,
            'probability_pct': round(prob * 100, 1),
            'category': cat['level'],
            'color': cat['color'],
            'badge_bg': cat['badge_bg'],
            'description': cat['description'],
            'input_engineered': df_engineered
        }

    def predict_hypertension(self, user_dict: Dict[str, Any], low_thresh: float = 0.30, high_thresh: float = 0.60) -> Dict[str, Any]:
        """Runs Hypertension screening prediction."""
        if 'hypertension' not in self.pipelines:
            raise RuntimeError("Hypertension model pipeline is not loaded.")
            
        df_raw = pd.DataFrame([{
            'male': 1 if str(user_dict.get('gender', 'Male')).lower() == 'male' else 0,
            'age': user_dict.get('age', 45),
            'currentSmoker': 1 if user_dict.get('is_smoker', False) else 0,
            'cigsPerDay': user_dict.get('cigs_per_day', 0),
            'BPMeds': 1 if user_dict.get('bp_meds', False) else 0,
            'totChol': user_dict.get('cholesterol', 200.0),
            'sysBP': user_dict.get('blood_pressure_sys', 120.0),
            'diaBP': user_dict.get('blood_pressure_dia', 80.0),
            'BMI': user_dict.get('bmi', 24.5),
            'heartRate': user_dict.get('resting_heart_rate', 72.0),
            'glucose': user_dict.get('glucose', 85.0),
            'diabetes': 1 if user_dict.get('has_diabetes', False) else 0
        }])
        
        df_engineered = engineer_hypertension_features(df_raw)
        pipeline = self.pipelines['hypertension']
        prob = float(pipeline.predict_proba(df_engineered)[0, 1])
        cat = self.get_risk_category(prob, low_thresh, high_thresh)
        
        return {
            'disease': 'Hypertension Risk',
            'probability': prob,
            'probability_pct': round(prob * 100, 1),
            'category': cat['level'],
            'color': cat['color'],
            'badge_bg': cat['badge_bg'],
            'description': cat['description'],
            'input_engineered': df_engineered
        }

    def predict_all(
        self,
        user_dict: Dict[str, Any],
        low_thresh: float = 0.30,
        high_thresh: float = 0.60
    ) -> Dict[str, Any]:
        """Runs unified multi-disease predictions with lifestyle, BMI, and wellness insights."""
        # Calculate BMI if missing
        height = user_dict.get('height_cm', 170.0)
        weight = user_dict.get('weight_kg', 70.0)
        bmi_info = calculate_bmi(height, weight)
        if not user_dict.get('bmi'):
            user_dict['bmi'] = bmi_info['bmi']
            
        res_diabetes = self.predict_diabetes(user_dict, low_thresh, high_thresh)
        res_heart = self.predict_heart_disease(user_dict, low_thresh, high_thresh)
        res_hypertension = self.predict_hypertension(user_dict, low_thresh, high_thresh)
        
        mean_prob = float(np.mean([
            res_diabetes['probability'],
            res_heart['probability'],
            res_hypertension['probability']
        ]))
        overall_score = round(mean_prob * 100, 1)
        overall_cat = self.get_risk_category(mean_prob, low_thresh, high_thresh)
        
        lifestyle_info = compute_lifestyle_quality_score(user_dict)
        warnings = validate_user_inputs(user_dict)
        recommendations = generate_personalized_insights(user_dict, {
            'diabetes': res_diabetes,
            'heart': res_heart,
            'hypertension': res_hypertension,
            'overall_risk_score': overall_score
        })
        
        return {
            'diabetes': {
                'disease': res_diabetes['disease'],
                'probability': res_diabetes['probability'],
                'probability_pct': res_diabetes['probability_pct'],
                'category': res_diabetes['category'],
                'color': res_diabetes['color'],
                'badge_bg': res_diabetes['badge_bg'],
                'description': res_diabetes['description']
            },
            'heart': {
                'disease': res_heart['disease'],
                'probability': res_heart['probability'],
                'probability_pct': res_heart['probability_pct'],
                'category': res_heart['category'],
                'color': res_heart['color'],
                'badge_bg': res_heart['badge_bg'],
                'description': res_heart['description']
            },
            'hypertension': {
                'disease': res_hypertension['disease'],
                'probability': res_hypertension['probability'],
                'probability_pct': res_hypertension['probability_pct'],
                'category': res_hypertension['category'],
                'color': res_hypertension['color'],
                'badge_bg': res_hypertension['badge_bg'],
                'description': res_hypertension['description']
            },
            'mean_disease_prob': mean_prob,
            'overall_risk_score': overall_score,
            'overall_risk_category': overall_cat['level'],
            'overall_color': overall_cat['color'],
            'bmi_info': bmi_info,
            'lifestyle_score': lifestyle_info,
            'recommendations': recommendations,
            'warnings': warnings
        }

    @staticmethod
    def get_preset_personas() -> Dict[str, Dict[str, Any]]:
        """Returns standard biomedical test personas for rapid demo & testing."""
        return {
            "healthy_young_adult": {
                "name": "Healthy Young Adult",
                "tagline": "Optimal cardiovascular & metabolic markers",
                "data": {
                    "patient_name": "Alex Chen (Persona)",
                    "age": 27,
                    "gender": "Female",
                    "height_cm": 168.0,
                    "weight_kg": 60.0,
                    "bmi": 21.3,
                    "physical_activity": "Active (5+ days/week)",
                    "sleep_hours": 8.0,
                    "diet_quality": "Balanced & Whole Foods",
                    "stress_level": "Low",
                    "alcohol_intake": "Occasional / Light",
                    "is_smoker": False,
                    "cigs_per_day": 0,
                    "blood_pressure_sys": 115.0,
                    "blood_pressure_dia": 74.0,
                    "resting_heart_rate": 62.0,
                    "max_heart_rate": 182.0,
                    "glucose": 84.0,
                    "cholesterol": 172.0,
                    "skin_thickness": 18.0,
                    "insulin": 45.0,
                    "diabetes_pedigree": 0.22,
                    "pregnancies": 0,
                    "chest_pain_type": 0,
                    "exercise_angina": 0,
                    "st_depression": 0.0,
                    "st_slope": 0,
                    "num_major_vessels": 0,
                    "thalassemia": 1,
                    "bp_meds": False,
                    "has_diabetes": False,
                    "notes": "Baseline healthy control profile."
                }
            },
            "sedentary_prediabetic": {
                "name": "Sedentary Pre-Diabetic",
                "tagline": "Elevated glucose, high BMI, low physical activity",
                "data": {
                    "patient_name": "Jordan Smith (Persona)",
                    "age": 48,
                    "gender": "Male",
                    "height_cm": 175.0,
                    "weight_kg": 94.0,
                    "bmi": 30.7,
                    "physical_activity": "Sedentary (<1 day/week)",
                    "sleep_hours": 5.5,
                    "diet_quality": "High Fast Food / Refined Carbs",
                    "stress_level": "High",
                    "alcohol_intake": "Moderate",
                    "is_smoker": False,
                    "cigs_per_day": 0,
                    "blood_pressure_sys": 134.0,
                    "blood_pressure_dia": 88.0,
                    "resting_heart_rate": 78.0,
                    "max_heart_rate": 145.0,
                    "glucose": 138.0,
                    "cholesterol": 228.0,
                    "skin_thickness": 32.0,
                    "insulin": 165.0,
                    "diabetes_pedigree": 0.72,
                    "pregnancies": 0,
                    "chest_pain_type": 1,
                    "exercise_angina": 0,
                    "st_depression": 0.8,
                    "st_slope": 1,
                    "num_major_vessels": 0,
                    "thalassemia": 2,
                    "bp_meds": False,
                    "has_diabetes": False,
                    "notes": "Pre-diabetic metabolic syndrome profile."
                }
            },
            "hypertensive_smoker": {
                "name": "Hypertensive Chronic Smoker",
                "tagline": "Stage 2 hypertension, heavy smoking, arterial stiffness",
                "data": {
                    "patient_name": "Robert Taylor (Persona)",
                    "age": 58,
                    "gender": "Male",
                    "height_cm": 172.0,
                    "weight_kg": 85.0,
                    "bmi": 28.7,
                    "physical_activity": "Light (1-2 days/week)",
                    "sleep_hours": 6.0,
                    "diet_quality": "Average / High Sodium",
                    "stress_level": "High",
                    "alcohol_intake": "Frequent / Heavy",
                    "is_smoker": True,
                    "cigs_per_day": 20,
                    "blood_pressure_sys": 158.0,
                    "blood_pressure_dia": 96.0,
                    "resting_heart_rate": 84.0,
                    "max_heart_rate": 138.0,
                    "glucose": 112.0,
                    "cholesterol": 255.0,
                    "skin_thickness": 26.0,
                    "insulin": 110.0,
                    "diabetes_pedigree": 0.48,
                    "pregnancies": 0,
                    "chest_pain_type": 2,
                    "exercise_angina": 1,
                    "st_depression": 1.4,
                    "st_slope": 1,
                    "num_major_vessels": 1,
                    "thalassemia": 2,
                    "bp_meds": True,
                    "has_diabetes": False,
                    "notes": "Vascular resistance and arterial stiffness screening."
                }
            },
            "high_risk_cardiac": {
                "name": "High-Risk Cardiac Patient",
                "tagline": "Exertional ischemia, ST depression, fluoroscopy vessels",
                "data": {
                    "patient_name": "Marcus Vance (Persona)",
                    "age": 64,
                    "gender": "Male",
                    "height_cm": 170.0,
                    "weight_kg": 88.0,
                    "bmi": 30.4,
                    "physical_activity": "Sedentary (<1 day/week)",
                    "sleep_hours": 5.0,
                    "diet_quality": "Poor / High Saturated Fat",
                    "stress_level": "Severe / Chronic",
                    "alcohol_intake": "Frequent / Heavy",
                    "is_smoker": True,
                    "cigs_per_day": 15,
                    "blood_pressure_sys": 164.0,
                    "blood_pressure_dia": 102.0,
                    "resting_heart_rate": 88.0,
                    "max_heart_rate": 126.0,
                    "glucose": 155.0,
                    "cholesterol": 290.0,
                    "skin_thickness": 28.0,
                    "insulin": 180.0,
                    "diabetes_pedigree": 0.85,
                    "pregnancies": 0,
                    "chest_pain_type": 3,
                    "exercise_angina": 1,
                    "st_depression": 2.6,
                    "st_slope": 2,
                    "num_major_vessels": 2,
                    "thalassemia": 3,
                    "bp_meds": True,
                    "has_diabetes": True,
                    "notes": "Complex multi-morbidity coronary and metabolic profile."
                }
            }
        }


ml_service = MLService()

