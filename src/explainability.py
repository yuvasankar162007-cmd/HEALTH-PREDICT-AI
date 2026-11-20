"""
Explainable AI (XAI) Module for HealthPredict.
Utilizes SHAP (SHapley Additive exPlanations) to explain local instance predictions
and extract global model feature importances, formatted with interactive Plotly visualizations
and accessible clinical interpretations.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go


class HealthExplainer:
    """
    Computes local and global feature attributions for HealthPredict models.
    """
    def __init__(self, predictor):
        self.predictor = predictor
        self.explainers = {}
        self.transformed_feature_names = {}
        self._init_explainers()
        
    def _init_explainers(self):
        """Initializes SHAP explainers for loaded pipelines."""
        for disease, pipeline in self.predictor.pipelines.items():
            try:
                preprocessor = pipeline.named_steps['preprocessor']
                classifier = pipeline.named_steps['classifier']
                
                # Check explainer type
                if hasattr(classifier, 'feature_importances_'):
                    explainer = shap.TreeExplainer(classifier)
                else:
                    # Linear or general explainer
                    explainer = shap.Explainer(classifier.predict_proba if hasattr(classifier, 'predict_proba') else classifier)
                    
                self.explainers[disease] = explainer
            except Exception as e:
                print(f"Notice: Initialized fallback explainer for {disease}: {e}")

    def get_feature_names(self, preprocessor) -> list:
        """Extracts transformed feature names from Scikit-Learn preprocessor."""
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            return [f"Feature_{i}" for i in range(50)]

    def explain_prediction(self, disease: str, input_df: pd.DataFrame) -> dict:
        """
        Computes SHAP values and generates explanations for a single user prediction.
        """
        if disease not in self.predictor.pipelines:
            raise ValueError(f"Unknown disease: {disease}")
            
        pipeline = self.predictor.pipelines[disease]
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        
        # Transform input features through preprocessor
        X_trans = preprocessor.transform(input_df)
        if hasattr(X_trans, 'toarray'):
            X_trans = X_trans.toarray()
            
        raw_feat_names = self.get_feature_names(preprocessor)
        clean_feat_names = [f.replace('num__', '').replace('cat__', '') for f in raw_feat_names]
        
        # Compute SHAP values
        shap_values = None
        base_value = 0.5
        
        try:
            if disease in self.explainers:
                explainer = self.explainers[disease]
                sv = explainer(X_trans)
                
                # Handle binary classification multi-class output (select positive class)
                if len(sv.shape) == 3 and sv.shape[2] == 2:
                    shap_values = sv.values[0, :, 1]
                    base_value = float(sv.base_values[0, 1]) if hasattr(sv.base_values, '__len__') else float(sv.base_values)
                elif len(sv.shape) == 2:
                    shap_values = sv.values[0, :]
                    base_value = float(sv.base_values[0]) if hasattr(sv.base_values, '__len__') else float(sv.base_values)
                else:
                    shap_values = np.array(sv.values).flatten()[:len(clean_feat_names)]
        except Exception as e:
            print(f"SHAP explainer fallback for {disease}: {e}")
            
        # Fallback to feature importance / weight calculation if SHAP calculation encounters structure mismatch
        if shap_values is None or len(shap_values) != len(clean_feat_names):
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                shap_values = importances * (X_trans[0] - np.mean(X_trans[0]))
            elif hasattr(classifier, 'coef_'):
                coefs = classifier.coef_[0]
                shap_values = coefs * X_trans[0]
            else:
                shap_values = np.zeros(len(clean_feat_names))
                
        df_impact = pd.DataFrame({
            'feature': clean_feat_names,
            'shap_value': shap_values,
            'abs_impact': np.abs(shap_values)
        }).sort_values(by='abs_impact', ascending=False).reset_index(drop=True)
        
        # Top contributing factors
        top_risk_increasing = df_impact[df_impact['shap_value'] > 0].head(4).to_dict('records')
        top_risk_decreasing = df_impact[df_impact['shap_value'] < 0].head(4).to_dict('records')
        
        # Friendly readable text explanation
        narrative = self._generate_explanation_narrative(disease, top_risk_increasing, top_risk_decreasing)
        
        return {
            'disease': disease,
            'df_impact': df_impact,
            'top_increasing': top_risk_increasing,
            'top_decreasing': top_risk_decreasing,
            'base_value': base_value,
            'narrative': narrative
        }

    def _generate_explanation_narrative(self, disease: str, top_inc: list, top_dec: list) -> str:
        """
        Creates an educational, clear summary in plain English.
        """
        inc_names = [item['feature'].replace('_', ' ') for item in top_inc[:3]]
        dec_names = [item['feature'].replace('_', ' ') for item in top_dec[:2]]
        
        text = f"**Key Drivers for {disease.capitalize()} Risk Prediction:**\n\n"
        if inc_names:
            text += f"• **Primary Factors Increasing Risk:** The model identified `{', '.join(inc_names)}` as the most influential parameters driving the estimated risk upward for your profile.\n"
        if dec_names:
            text += f"• **Primary Protective Factors:** In contrast, `{', '.join(dec_names)}` helped moderate and reduce your risk score relative to baseline averages.\n\n"
            
        text += "> *Note on Explainability:* These contributions represent statistical associations learned by the machine learning model from historical medical benchmark data. Feature importance does not prove direct biological causation."
        return text

    def create_shap_waterfall_chart(self, explanation_dict: dict):
        """
        Generates a Plotly horizontal bar chart of top feature contributions.
        """
        df_top = explanation_dict['df_impact'].head(8).iloc[::-1]  # Reverse for top-down display
        
        colors = ['#EF5350' if val > 0 else '#66BB6A' for val in df_top['shap_value']]
        
        fig = go.Figure(go.Bar(
            x=df_top['shap_value'],
            y=[f.replace('_', ' ').title() for f in df_top['feature']],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{val:+.3f}" for val in df_top['shap_value']],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"Feature Impact on {explanation_dict['disease'].capitalize()} Prediction (SHAP Values)",
            xaxis_title="Impact on Model Output (Red = Pushes Risk Higher, Green = Lowers Risk)",
            yaxis_title="Biomedical & Lifestyle Feature",
            margin=dict(l=20, r=20, t=50, b=30),
            height=360,
            template="plotly_white",
            xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='#9E9E9E')
        )
        return fig
