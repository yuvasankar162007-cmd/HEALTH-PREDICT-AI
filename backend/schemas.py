"""
Pydantic Data Validation Schemas for HealthPredict REST API.
Enforces strict biomedical validation and rich Swagger documentation.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, model_validator


class SensitivityThresholds(BaseModel):
    """Screening sensitivity thresholds for risk classification."""
    low_threshold: float = Field(0.30, ge=0.05, le=0.50, description="Cutoff below which risk is classified as LOW")
    high_threshold: float = Field(0.60, ge=0.51, le=0.90, description="Cutoff above which risk is classified as HIGH")


class HealthInput(BaseModel):
    """Complete biometric and lifestyle input data for multi-disease risk assessment."""
    patient_name: Optional[str] = Field("Anonymous User", description="Optional patient name or identifier")
    age: int = Field(45, ge=18, le=110, description="Age in years")
    gender: str = Field("Male", description="Gender ('Male' or 'Female')")
    height_cm: float = Field(175.0, ge=90.0, le=240.0, description="Height in centimeters")
    weight_kg: float = Field(80.0, ge=30.0, le=250.0, description="Weight in kilograms")
    bmi: Optional[float] = Field(None, description="Body Mass Index (auto-calculated from height & weight if omitted)")
    
    # Lifestyle Habits
    physical_activity: str = Field("Moderate (3-4 days/week)", description="Physical activity frequency")
    sleep_hours: float = Field(7.0, ge=0.0, le=24.0, description="Average sleep per night in hours")
    diet_quality: str = Field("Balanced", description="Nutritional pattern assessment")
    stress_level: str = Field("Moderate", description="Perceived daily stress level")
    alcohol_intake: str = Field("Occasional / Light", description="Alcohol consumption frequency")
    is_smoker: bool = Field(False, description="Current cigarette smoker")
    cigs_per_day: int = Field(0, ge=0, le=100, description="Cigarettes per day if smoker")
    
    # Hemodynamics & Vitals
    blood_pressure_sys: float = Field(125.0, ge=70.0, le=260.0, description="Systolic Blood Pressure in mmHg")
    blood_pressure_dia: float = Field(82.0, ge=40.0, le=160.0, description="Diastolic Blood Pressure in mmHg")
    resting_heart_rate: float = Field(72.0, ge=40.0, le=180.0, description="Resting heart rate in bpm")
    max_heart_rate: float = Field(150.0, ge=60.0, le=220.0, description="Maximum achieved heart rate in bpm")
    
    # Laboratory Biomarkers & Clinical Parameters
    glucose: float = Field(98.0, ge=40.0, le=500.0, description="Fasting Blood Glucose in mg/dL")
    cholesterol: float = Field(200.0, ge=80.0, le=500.0, description="Total Serum Cholesterol in mg/dL")
    skin_thickness: float = Field(22.0, ge=0.0, le=99.0, description="Triceps skin fold thickness in mm")
    insulin: float = Field(85.0, ge=0.0, le=900.0, description="2-Hour serum insulin in mu U/ml")
    diabetes_pedigree: float = Field(0.45, ge=0.05, le=2.5, description="Diabetes pedigree function score")
    pregnancies: int = Field(0, ge=0, le=20, description="Number of times pregnant")
    
    # Cardiology Diagnostics (Cleveland UCI)
    chest_pain_type: int = Field(0, ge=0, le=3, description="Chest pain type (0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic)")
    exercise_angina: int = Field(0, ge=0, le=1, description="Exercise induced angina (1=Yes, 0=No)")
    st_depression: float = Field(0.5, ge=0.0, le=10.0, description="ST depression induced by exercise (oldpeak)")
    st_slope: int = Field(1, ge=0, le=2, description="Slope of peak exercise ST segment (0=Upsloping, 1=Flat, 2=Downsloping)")
    num_major_vessels: int = Field(0, ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thalassemia: int = Field(2, ge=0, le=3, description="Thalassemia scan (1=Normal, 2=Fixed defect, 3=Reversible defect)")
    
    # Medical History
    bp_meds: bool = Field(False, description="Currently taking anti-hypertensive medication")
    has_diabetes: bool = Field(False, description="Prior clinical diagnosis of diabetes")
    notes: Optional[str] = Field("", description="Optional clinical notes or annotations")

    @model_validator(mode='after')
    def validate_physiological_coherence(self):
        # Auto-compute BMI if missing or zero
        if not self.bmi or self.bmi <= 0:
            h_m = self.height_cm / 100.0
            self.bmi = round(self.weight_kg / (h_m ** 2), 1)
            
        # Ensure SBP > DBP
        if self.blood_pressure_sys <= self.blood_pressure_dia:
            # Gentle self-correction for edge cases
            self.blood_pressure_sys = self.blood_pressure_dia + 20.0
            
        return self


class SingleDiseasePrediction(BaseModel):
    """Screening outcome for a single disease module."""
    disease: str
    probability: float
    probability_pct: float
    category: str
    color: str
    badge_bg: str
    description: str


class BMICategoryInfo(BaseModel):
    """Calculated BMI information and WHO classification."""
    bmi: float
    category: str
    color: str
    badge_bg: str


class LifestyleScoreInfo(BaseModel):
    """Lifestyle Quality Index rating and itemized deductions."""
    score: int
    rating: str
    color: str
    badge_bg: str
    deductions: List[Any]


class PersonalizedInsight(BaseModel):
    """Evidence-based wellness and preventive suggestion."""
    domain: str
    title: str
    icon: str
    color: str
    badge_bg: str
    text: str


class ComprehensivePredictionResponse(BaseModel):
    """Unified multi-disease screening result payload."""
    diabetes: SingleDiseasePrediction
    heart: SingleDiseasePrediction
    hypertension: SingleDiseasePrediction
    mean_disease_prob: float
    overall_risk_score: float
    overall_risk_category: str
    overall_color: str
    bmi_info: BMICategoryInfo
    lifestyle_score: LifestyleScoreInfo
    recommendations: List[PersonalizedInsight]
    warnings: List[str]
    saved_record_id: Optional[int] = None


class WhatIfInterventionRequest(BaseModel):
    """Simulation input parameters for hypothetical lifestyle modifications."""
    base_data: HealthInput
    weight_loss_kg: float = Field(0.0, ge=0.0, le=40.0)
    sbp_reduction: float = Field(0.0, ge=0.0, le=50.0)
    glucose_reduction: float = Field(0.0, ge=0.0, le=100.0)
    chol_reduction: float = Field(0.0, ge=0.0, le=100.0)
    quit_smoking: bool = False
    boost_activity: bool = False


class WhatIfInterventionResponse(BaseModel):
    """Projected outcomes from hypothetical counterfactual intervention."""
    base_preds: ComprehensivePredictionResponse
    sim_preds: ComprehensivePredictionResponse
    diabetes_drop: float
    heart_drop: float
    hypertension_drop: float
    overall_drop: float
    simulated_bmi: float
    simulated_sbp: float
    simulated_glucose: float


class ShapFeatureImpact(BaseModel):
    """Individual feature Shapley attribution."""
    feature: str
    shap_value: float
    abs_impact: float


class ShapExplanationResponse(BaseModel):
    """Explainable AI response containing SHAP attributions and clinical narrative."""
    disease: str
    base_value: float
    features: List[str]
    shap_values: List[float]
    top_increasing: List[Dict[str, Any]]
    top_decreasing: List[Dict[str, Any]]
    narrative: str


class AssessmentRecordResponse(BaseModel):
    """Saved historical patient assessment database record."""
    id: int
    patient_name: str
    created_at: str
    age: int
    gender: str
    bmi: float
    blood_pressure_sys: float
    blood_pressure_dia: float
    glucose: float
    cholesterol: float
    lifestyle_score: float
    lifestyle_rating: str
    diabetes_prob: float
    diabetes_cat: str
    heart_prob: float
    heart_cat: str
    hypertension_prob: float
    hypertension_cat: str
    overall_risk_score: float
    notes: Optional[str] = ""
    input_data: Optional[Dict[str, Any]] = None


class StatsSummaryResponse(BaseModel):
    """Aggregated clinical telemetry and usage statistics."""
    total_assessments: int
    avg_overall_risk: float
    high_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    avg_lifestyle_score: float
    recent_timeline: List[Dict[str, Any]]

