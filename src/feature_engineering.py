"""
Feature Engineering Module for HealthPredict.
Applies domain-justified medical and physiological transformations
to raw biomedical datasets for Diabetes, Heart Disease, and Hypertension.
"""

import numpy as np
import pandas as pd


def engineer_diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer biomedical features for Diabetes prediction.
    Features:
    - Glucose_BMI_Interaction: Interaction of glucose and BMI (multiplicative metabolic strain)
    - Insulin_Glucose_Ratio: Index of insulin secretion vs circulating glucose
    - Age_Over_45: Binary marker for high-risk age group >= 45
    - High_Glucose_Flag: Impaired glucose tolerance threshold (>= 140 mg/dL)
    """
    df_out = df.copy()
    
    # 1. Glucose and BMI interaction (scaled)
    df_out['Glucose_BMI_Interaction'] = (df_out['Glucose'] * df_out['BMI']) / 100.0
    
    # 2. Insulin to Glucose Ratio (handling zero/low glucose safely)
    glucose_safe = np.where(df_out['Glucose'] == 0, np.nan, df_out['Glucose'])
    df_out['Insulin_Glucose_Ratio'] = df_out['Insulin'] / (glucose_safe + 1e-5)
    
    # 3. Age risk category (binary marker for high-risk age group >= 45)
    df_out['Age_Over_45'] = (df_out['Age'] >= 45).astype(int)
    
    # 4. High Glucose Indicator (>= 140 mg/dL)
    df_out['High_Glucose_Flag'] = (df_out['Glucose'] >= 140).astype(int)
    
    return df_out


def engineer_heart_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer physiological features for Heart Disease prediction.
    Features:
    - HR_Reserve_Ratio: Max achieved heart rate as % of theoretical max (220 - age)
    - Chol_BP_Index: Combined vascular resistance index (cholesterol and resting systolic BP)
    - Ischemia_Index: Interaction between exercise-induced ST depression (oldpeak) and angina
    - Male_Over_50: Interaction between male gender and age >= 50
    """
    df_out = df.copy()
    
    # 1. Heart Rate Reserve Ratio (achieved / predicted max HR)
    predicted_max_hr = 220 - df_out['age']
    predicted_max_hr = np.where(predicted_max_hr <= 0, 150, predicted_max_hr)
    df_out['HR_Reserve_Ratio'] = df_out['thalach'] / predicted_max_hr
    
    # 2. Cholesterol to BP index
    df_out['Chol_BP_Index'] = (df_out['chol'] / 200.0) * (df_out['trestbps'] / 120.0)
    
    # 3. Ischemia Severity Score
    df_out['Ischemia_Index'] = df_out['oldpeak'] * (df_out['exang'] + 1.0)
    
    # 4. Age and Gender interaction
    df_out['Male_Over_50'] = ((df_out['sex'] == 1) & (df_out['age'] >= 50)).astype(int)
    
    return df_out


def engineer_hypertension_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer hemodynamic and lifestyle features for Hypertension prediction.
    Features:
    - Pulse_Pressure: Systolic BP - Diastolic BP (measure of arterial stiffness)
    - Mean_Arterial_Pressure (MAP): Diastolic BP + 1/3(Systolic BP - Diastolic BP)
    - Smoking_Burden: Cigs per day * (age / 50.0)
    - High_BMI_Flag: Overweight/Obese marker (BMI >= 25.0)
    - High_Glucose_Flag: Elevated fasting blood glucose (>= 100 mg/dL)
    """
    df_out = df.copy()
    
    # 1. Pulse Pressure (mmHg)
    df_out['Pulse_Pressure'] = df_out['sysBP'] - df_out['diaBP']
    
    # 2. Mean Arterial Pressure (MAP in mmHg)
    df_out['Mean_Arterial_Pressure'] = df_out['diaBP'] + (df_out['Pulse_Pressure'] / 3.0)
    
    # 3. Smoking Burden proxy
    cigs = df_out['cigsPerDay'].fillna(0)
    df_out['Smoking_Burden'] = cigs * (df_out['age'] / 50.0)
    
    # 4. High BMI Flag (>= 25 kg/m^2)
    df_out['High_BMI_Flag'] = (df_out['BMI'] >= 25.0).astype(int)
    
    # 5. Elevated Glucose Flag (>= 100 mg/dL)
    glucose = df_out['glucose'].fillna(df_out['glucose'].median() if 'glucose' in df_out and not df_out['glucose'].empty else 80.0)
    df_out['High_Glucose_Flag'] = (glucose >= 100.0).astype(int)
    
    return df_out
