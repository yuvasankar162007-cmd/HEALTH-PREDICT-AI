"""
Prediction and Risk Assessment Module for HealthPredict.
Loads serialized model pipelines, processes runtime user inputs,
and calculates calibrated disease risk estimates with risk categorization.
"""

import os
import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import (
    engineer_diabetes_features,
    engineer_heart_features,
    engineer_hypertension_features
)


class HealthPredictor:
    """
    Unified prediction engine for Diabetes, Heart Disease, and Hypertension risk screening.
    """
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
            
        self.models_dir = models_dir
        self.pipelines = {}
        self.load_models()
        
    def load_models(self):
        """Loads all serialized model pipelines."""
        for disease in ['diabetes', 'heart', 'hypertension']:
            path = os.path.join(self.models_dir, f"{disease}_pipeline.pkl")
            if os.path.exists(path):
                self.pipelines[disease] = joblib.load(path)
            else:
                print(f"Warning: Model pipeline for {disease} not found at {path}")
                
    def get_risk_category(self, probability: float, low_thresh: float = 0.30, high_thresh: float = 0.60):
        """
        Categorizes probability into educational screening bands.
        """
        if probability < low_thresh:
            return {
                'level': 'LOW RISK',
                'color': '#2E7D32',  # Green
                'badge_bg': '#E8F5E9',
                'description': 'Estimated risk is within the low range compared to benchmark populations.'
            }
        elif probability <= high_thresh:
            return {
                'level': 'MODERATE RISK',
                'color': '#EF6C00',  # Amber/Orange
                'badge_bg': '#FFF3E0',
                'description': 'Estimated risk is elevated. Proactive lifestyle modifications are recommended.'
            }
        else:
            return {
                'level': 'HIGH RISK',
                'color': '#C62828',  # Red
                'badge_bg': '#FFEBEE',
                'description': 'Estimated risk is high. Consultation with a qualified healthcare professional is strongly advised.'
            }

    def predict_diabetes(self, user_data: dict) -> dict:
        """
        Predicts diabetes risk probability.
        """
        if 'diabetes' not in self.pipelines:
            raise RuntimeError("Diabetes model pipeline is not loaded.")
            
        df_raw = pd.DataFrame([{
            'Pregnancies': user_data.get('pregnancies', 0),
            'Glucose': user_data.get('glucose', 100.0),
            'BloodPressure': user_data.get('blood_pressure_dia', 75.0),
            'SkinThickness': user_data.get('skin_thickness', 20.0),
            'Insulin': user_data.get('insulin', 80.0),
            'BMI': user_data.get('bmi', 24.5),
            'DiabetesPedigreeFunction': user_data.get('diabetes_pedigree', 0.47),
            'Age': user_data.get('age', 35)
        }])
        
        df_engineered = engineer_diabetes_features(df_raw)
        pipeline = self.pipelines['diabetes']
        
        prob = pipeline.predict_proba(df_engineered)[0, 1]
        category_info = self.get_risk_category(prob)
        
        return {
            'disease': 'Diabetes Risk',
            'probability': float(prob),
            'probability_pct': round(prob * 100, 1),
            'category': category_info['level'],
            'color': category_info['color'],
            'badge_bg': category_info['badge_bg'],
            'description': category_info['description'],
            'input_engineered': df_engineered
        }

    def predict_heart_disease(self, user_data: dict) -> dict:
        """
        Predicts heart disease risk probability.
        """
        if 'heart' not in self.pipelines:
            raise RuntimeError("Heart Disease model pipeline is not loaded.")
            
        df_raw = pd.DataFrame([{
            'age': user_data.get('age', 50),
            'sex': 1 if user_data.get('gender', 'Male') == 'Male' else 0,
            'cp': user_data.get('chest_pain_type', 0),
            'trestbps': user_data.get('blood_pressure_sys', 120.0),
            'chol': user_data.get('cholesterol', 210.0),
            'fbs': 1 if user_data.get('glucose', 100.0) > 120 else 0,
            'restecg': user_data.get('rest_ecg', 0),
            'thalach': user_data.get('max_heart_rate', 150.0),
            'exang': user_data.get('exercise_angina', 0),
            'oldpeak': user_data.get('st_depression', 0.5),
            'slope': user_data.get('st_slope', 1),
            'ca': user_data.get('num_major_vessels', 0),
            'thal': user_data.get('thalassemia', 2)
        }])
        
        df_engineered = engineer_heart_features(df_raw)
        pipeline = self.pipelines['heart']
        
        prob = pipeline.predict_proba(df_engineered)[0, 1]
        category_info = self.get_risk_category(prob)
        
        return {
            'disease': 'Heart Disease Risk',
            'probability': float(prob),
            'probability_pct': round(prob * 100, 1),
            'category': category_info['level'],
            'color': category_info['color'],
            'badge_bg': category_info['badge_bg'],
            'description': category_info['description'],
            'input_engineered': df_engineered
        }

    def predict_hypertension(self, user_data: dict) -> dict:
        """
        Predicts hypertension risk probability.
        """
        if 'hypertension' not in self.pipelines:
            raise RuntimeError("Hypertension model pipeline is not loaded.")
            
        df_raw = pd.DataFrame([{
            'male': 1 if user_data.get('gender', 'Male') == 'Male' else 0,
            'age': user_data.get('age', 45),
            'currentSmoker': 1 if user_data.get('is_smoker', False) else 0,
            'cigsPerDay': user_data.get('cigs_per_day', 0),
            'BPMeds': 1 if user_data.get('bp_meds', False) else 0,
            'totChol': user_data.get('cholesterol', 200.0),
            'sysBP': user_data.get('blood_pressure_sys', 120.0),
            'diaBP': user_data.get('blood_pressure_dia', 80.0),
            'BMI': user_data.get('bmi', 24.5),
            'heartRate': user_data.get('resting_heart_rate', 72.0),
            'glucose': user_data.get('glucose', 85.0),
            'diabetes': 1 if user_data.get('has_diabetes', False) else 0
        }])
        
        df_engineered = engineer_hypertension_features(df_raw)
        pipeline = self.pipelines['hypertension']
        
        prob = pipeline.predict_proba(df_engineered)[0, 1]
        category_info = self.get_risk_category(prob)
        
        return {
            'disease': 'Hypertension Risk',
            'probability': float(prob),
            'probability_pct': round(prob * 100, 1),
            'category': category_info['level'],
            'color': category_info['color'],
            'badge_bg': category_info['badge_bg'],
            'description': category_info['description'],
            'input_engineered': df_engineered
        }

    def predict_all(self, user_data: dict) -> dict:
        """
        Runs predictions across all disease screening models.
        """
        res_diabetes = self.predict_diabetes(user_data)
        res_heart = self.predict_heart_disease(user_data)
        res_hypertension = self.predict_hypertension(user_data)
        
        # Calculate composite lifestyle & biometric risk score (0-100)
        mean_disease_risk = np.mean([
            res_diabetes['probability'],
            res_heart['probability'],
            res_hypertension['probability']
        ])
        
        return {
            'diabetes': res_diabetes,
            'heart': res_heart,
            'hypertension': res_hypertension,
            'mean_disease_prob': float(mean_disease_risk),
            'overall_risk_score': round(mean_disease_risk * 100, 1)
        }
