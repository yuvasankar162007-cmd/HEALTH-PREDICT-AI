"""
Explainable AI (XAI) Service for HealthPredict.
Utilizes SHAP to extract local instance feature attributions, base values,
and clinical plain-English narrative explanations.
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List

from backend.services.ml_service import ml_service
from src.explainability import HealthExplainer


class XAIService:
    """Service wrapping SHAP explainability calculations for REST endpoints."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(XAIService, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.explainer = HealthExplainer(ml_service)
        self._initialized = True

    def explain_disease(self, disease: str, user_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a given disease and input dictionary.
        Returns chart-ready feature values and accessible clinical narratives.
        """
        disease_lower = disease.lower().strip()
        if 'diabetes' in disease_lower:
            disease_key = 'diabetes'
            pred_res = ml_service.predict_diabetes(user_dict)
        elif 'heart' in disease_lower:
            disease_key = 'heart'
            pred_res = ml_service.predict_heart_disease(user_dict)
        elif 'hyper' in disease_lower:
            disease_key = 'hypertension'
            pred_res = ml_service.predict_hypertension(user_dict)
        else:
            raise ValueError(f"Unsupported disease module: {disease}")

        input_engineered = pred_res['input_engineered']
        raw_explanation = self.explainer.explain_prediction(disease_key, input_engineered)
        df_impact = raw_explanation['df_impact']

        # Format clean lists for frontend charts
        features = list(df_impact['feature'])
        shap_vals = [float(x) for x in df_impact['shap_value']]
        abs_impacts = [float(x) for x in df_impact['abs_impact']]

        # Clean top factors
        top_increasing = [
            {
                'feature': str(item['feature']).replace('_', ' ').title(),
                'shap_value': round(float(item['shap_value']), 4),
                'impact': 'Increases Risk'
            }
            for item in raw_explanation['top_increasing']
        ]

        top_decreasing = [
            {
                'feature': str(item['feature']).replace('_', ' ').title(),
                'shap_value': round(float(item['shap_value']), 4),
                'impact': 'Lowers Risk (Protective)'
            }
            for item in raw_explanation['top_decreasing']
        ]

        return {
            'disease': disease_key,
            'base_value': float(raw_explanation['base_value']),
            'features': [f.replace('_', ' ').title() for f in features],
            'shap_values': shap_vals,
            'abs_impacts': abs_impacts,
            'top_increasing': top_increasing,
            'top_decreasing': top_decreasing,
            'narrative': raw_explanation['narrative']
        }


xai_service = XAIService()

