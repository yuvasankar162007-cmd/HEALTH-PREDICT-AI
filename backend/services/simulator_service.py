"""
What-If Counterfactual Lifestyle Intervention Simulation Service.
Evaluates the quantitative impact of hypothetical healthy modifications on disease risks.
"""

from typing import Dict, Any
from backend.services.ml_service import ml_service


class SimulatorService:
    """Service executing counterfactual lifestyle simulations."""

    @staticmethod
    def simulate_intervention(
        base_dict: Dict[str, Any],
        weight_loss_kg: float = 0.0,
        sbp_reduction: float = 0.0,
        glucose_reduction: float = 0.0,
        chol_reduction: float = 0.0,
        quit_smoking: bool = False,
        boost_activity: bool = False,
        low_thresh: float = 0.30,
        high_thresh: float = 0.60
    ) -> Dict[str, Any]:
        """
        Executes counterfactual simulation and calculates risk drops.
        """
        sim_data = dict(base_dict)

        # 1. Weight Loss & BMI Adjustment
        if weight_loss_kg > 0:
            current_weight = float(sim_data.get('weight_kg', 75.0))
            new_weight = max(35.0, current_weight - weight_loss_kg)
            sim_data['weight_kg'] = new_weight
            height_m = float(sim_data.get('height_cm', 170.0)) / 100.0
            sim_data['bmi'] = round(new_weight / (height_m ** 2), 1)

        # 2. Blood Pressure Reduction
        if sbp_reduction > 0:
            current_sbp = float(sim_data.get('blood_pressure_sys', 120.0))
            current_dbp = float(sim_data.get('blood_pressure_dia', 80.0))
            sim_data['blood_pressure_sys'] = max(90.0, current_sbp - sbp_reduction)
            sim_data['blood_pressure_dia'] = max(55.0, current_dbp - (sbp_reduction * 0.6))

        # 3. Fasting Glucose Optimization
        if glucose_reduction > 0:
            current_glu = float(sim_data.get('glucose', 90.0))
            sim_data['glucose'] = max(70.0, current_glu - glucose_reduction)

        # 4. Cholesterol Optimization
        if chol_reduction > 0:
            current_chol = float(sim_data.get('cholesterol', 200.0))
            sim_data['cholesterol'] = max(130.0, current_chol - chol_reduction)

        # 5. Smoking Cessation
        if quit_smoking:
            sim_data['is_smoker'] = False
            sim_data['cigs_per_day'] = 0

        # 6. Physical Activity Boost
        if boost_activity:
            sim_data['physical_activity'] = 'Active (5+ days/week)'
            sim_data['max_heart_rate'] = min(205.0, float(sim_data.get('max_heart_rate', 140.0)) + 12.0)
            sim_data['resting_heart_rate'] = max(52.0, float(sim_data.get('resting_heart_rate', 75.0)) - 6.0)

        # Base and Simulated Predictions
        base_preds = ml_service.predict_all(base_dict, low_thresh, high_thresh)
        sim_preds = ml_service.predict_all(sim_data, low_thresh, high_thresh)

        # Calculate Delta Drops
        d_drop = round(base_preds['diabetes']['probability_pct'] - sim_preds['diabetes']['probability_pct'], 1)
        h_drop = round(base_preds['heart']['probability_pct'] - sim_preds['heart']['probability_pct'], 1)
        hyp_drop = round(base_preds['hypertension']['probability_pct'] - sim_preds['hypertension']['probability_pct'], 1)
        overall_drop = round(base_preds['overall_risk_score'] - sim_preds['overall_risk_score'], 1)

        return {
            'base_preds': base_preds,
            'sim_preds': sim_preds,
            'diabetes_drop': d_drop,
            'heart_drop': h_drop,
            'hypertension_drop': hyp_drop,
            'overall_drop': overall_drop,
            'simulated_bmi': sim_data.get('bmi', 24.0),
            'simulated_sbp': sim_data.get('blood_pressure_sys', 120.0),
            'simulated_glucose': sim_data.get('glucose', 90.0)
        }


simulator_service = SimulatorService()

