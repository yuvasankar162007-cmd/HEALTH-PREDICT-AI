"""
Batch Screening Service for HealthPredict.
Processes multi-patient CSV files and produces enriched risk assessment datasets.
"""

import io
import pandas as pd
from typing import Dict, Any, List, Tuple
from backend.services.ml_service import ml_service


class BatchService:
    """Service handling multi-record CSV batch screening."""

    @staticmethod
    def get_sample_csv_template() -> str:
        """Generates a downloadable sample CSV template for batch screening."""
        sample_df = pd.DataFrame([
            {
                'patient_name': 'Patient Alpha',
                'age': 35,
                'gender': 'Male',
                'height_cm': 178.0,
                'weight_kg': 76.0,
                'is_smoker': 0,
                'cigs_per_day': 0,
                'blood_pressure_sys': 120.0,
                'blood_pressure_dia': 78.0,
                'resting_heart_rate': 70.0,
                'max_heart_rate': 160.0,
                'glucose': 92.0,
                'cholesterol': 195.0,
                'skin_thickness': 20.0,
                'insulin': 70.0,
                'diabetes_pedigree': 0.35,
                'pregnancies': 0,
                'chest_pain_type': 0,
                'exercise_angina': 0,
                'st_depression': 0.0,
                'st_slope': 1,
                'num_major_vessels': 0,
                'thalassemia': 2,
                'bp_meds': 0,
                'has_diabetes': 0
            },
            {
                'patient_name': 'Patient Beta',
                'age': 54,
                'gender': 'Female',
                'height_cm': 162.0,
                'weight_kg': 84.0,
                'is_smoker': 1,
                'cigs_per_day': 15,
                'blood_pressure_sys': 148.0,
                'blood_pressure_dia': 92.0,
                'resting_heart_rate': 82.0,
                'max_heart_rate': 135.0,
                'glucose': 142.0,
                'cholesterol': 240.0,
                'skin_thickness': 30.0,
                'insulin': 150.0,
                'diabetes_pedigree': 0.65,
                'pregnancies': 2,
                'chest_pain_type': 2,
                'exercise_angina': 1,
                'st_depression': 1.8,
                'st_slope': 2,
                'num_major_vessels': 1,
                'thalassemia': 2,
                'bp_meds': 1,
                'has_diabetes': 0
            },
            {
                'patient_name': 'Patient Gamma',
                'age': 62,
                'gender': 'Male',
                'height_cm': 172.0,
                'weight_kg': 90.0,
                'is_smoker': 1,
                'cigs_per_day': 25,
                'blood_pressure_sys': 162.0,
                'blood_pressure_dia': 98.0,
                'resting_heart_rate': 86.0,
                'max_heart_rate': 122.0,
                'glucose': 160.0,
                'cholesterol': 275.0,
                'skin_thickness': 32.0,
                'insulin': 190.0,
                'diabetes_pedigree': 0.88,
                'pregnancies': 0,
                'chest_pain_type': 3,
                'exercise_angina': 1,
                'st_depression': 2.4,
                'st_slope': 2,
                'num_major_vessels': 2,
                'thalassemia': 3,
                'bp_meds': 1,
                'has_diabetes': 1
            }
        ])
        return sample_df.to_csv(index=False)

    @staticmethod
    def process_batch_csv(
        file_bytes: bytes,
        low_thresh: float = 0.30,
        high_thresh: float = 0.60
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        Parses CSV, runs screening predictions row-by-row, and returns structured results + enriched CSV string.
        """
        df = pd.read_csv(io.BytesIO(file_bytes))
        
        results_records = []
        enriched_rows = []

        total_rows = len(df)
        high_risk_count = 0
        mod_risk_count = 0
        low_risk_count = 0

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            user_data = {
                'patient_name': str(row_dict.get('patient_name', f'Patient #{idx+1}')),
                'age': int(row_dict.get('age', 45)),
                'gender': str(row_dict.get('gender', 'Male')),
                'height_cm': float(row_dict.get('height_cm', 170.0)),
                'weight_kg': float(row_dict.get('weight_kg', 75.0)),
                'is_smoker': bool(int(row_dict.get('is_smoker', 0))),
                'cigs_per_day': int(row_dict.get('cigs_per_day', 0)),
                'blood_pressure_sys': float(row_dict.get('blood_pressure_sys', 120.0)),
                'blood_pressure_dia': float(row_dict.get('blood_pressure_dia', 80.0)),
                'resting_heart_rate': float(row_dict.get('resting_heart_rate', 72.0)),
                'max_heart_rate': float(row_dict.get('max_heart_rate', 150.0)),
                'glucose': float(row_dict.get('glucose', 90.0)),
                'cholesterol': float(row_dict.get('cholesterol', 200.0)),
                'skin_thickness': float(row_dict.get('skin_thickness', 20.0)),
                'insulin': float(row_dict.get('insulin', 80.0)),
                'diabetes_pedigree': float(row_dict.get('diabetes_pedigree', 0.45)),
                'pregnancies': int(row_dict.get('pregnancies', 0)),
                'chest_pain_type': int(row_dict.get('chest_pain_type', 0)),
                'exercise_angina': int(row_dict.get('exercise_angina', 0)),
                'st_depression': float(row_dict.get('st_depression', 0.5)),
                'st_slope': int(row_dict.get('st_slope', 1)),
                'num_major_vessels': int(row_dict.get('num_major_vessels', 0)),
                'thalassemia': int(row_dict.get('thalassemia', 2)),
                'bp_meds': bool(int(row_dict.get('bp_meds', 0))),
                'has_diabetes': bool(int(row_dict.get('has_diabetes', 0)))
            }

            pred = ml_service.predict_all(user_data, low_thresh, high_thresh)

            if pred['overall_risk_category'] == 'HIGH RISK':
                high_risk_count += 1
            elif pred['overall_risk_category'] == 'MODERATE RISK':
                mod_risk_count += 1
            else:
                low_risk_count += 1

            record = {
                'row_index': idx + 1,
                'patient_name': user_data['patient_name'],
                'age': user_data['age'],
                'gender': user_data['gender'],
                'bmi': pred['bmi_info']['bmi'],
                'bp': f"{int(user_data['blood_pressure_sys'])}/{int(user_data['blood_pressure_dia'])}",
                'glucose': user_data['glucose'],
                'cholesterol': user_data['cholesterol'],
                'diabetes_pct': pred['diabetes']['probability_pct'],
                'diabetes_cat': pred['diabetes']['category'],
                'heart_pct': pred['heart']['probability_pct'],
                'heart_cat': pred['heart']['category'],
                'hypertension_pct': pred['hypertension']['probability_pct'],
                'hypertension_cat': pred['hypertension']['category'],
                'overall_score': pred['overall_risk_score'],
                'overall_category': pred['overall_risk_category'],
                'overall_color': pred['overall_color']
            }
            results_records.append(record)

            # Enriched row
            enriched_row = dict(row_dict)
            enriched_row.update({
                'Calculated_BMI': pred['bmi_info']['bmi'],
                'Diabetes_Risk_Prob': pred['diabetes']['probability'],
                'Diabetes_Risk_Category': pred['diabetes']['category'],
                'Heart_Disease_Risk_Prob': pred['heart']['probability'],
                'Heart_Disease_Risk_Category': pred['heart']['category'],
                'Hypertension_Risk_Prob': pred['hypertension']['probability'],
                'Hypertension_Risk_Category': pred['hypertension']['category'],
                'Overall_Risk_Score': pred['overall_risk_score'],
                'Overall_Risk_Category': pred['overall_risk_category']
            })
            enriched_rows.append(enriched_row)

        enriched_df = pd.DataFrame(enriched_rows)
        csv_buffer = io.StringIO()
        enriched_df.to_csv(csv_buffer, index=False)
        enriched_csv_str = csv_buffer.getvalue()

        summary = {
            'total_patients': total_rows,
            'high_risk_count': high_risk_count,
            'moderate_risk_count': mod_risk_count,
            'low_risk_count': low_risk_count,
            'avg_risk_score': round(float(enriched_df['Overall_Risk_Score'].mean()), 1) if not enriched_df.empty else 0.0
        }

        return results_records, enriched_csv_str, summary


batch_service = BatchService()

