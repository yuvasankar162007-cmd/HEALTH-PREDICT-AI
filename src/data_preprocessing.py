"""
Data Preprocessing Module for HealthPredict.
Handles data loading, cleaning, missing-value imputation, categorical encoding,
feature scaling, and stratified train-test splitting.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    engineer_diabetes_features,
    engineer_heart_features,
    engineer_hypertension_features
)


def load_and_clean_diabetes_data(file_path: str):
    """
    Loads and cleans the PIMA Diabetes dataset.
    Replaces physiologically impossible zero values with NaN for proper imputation.
    """
    df = pd.read_csv(file_path)
    
    # Columns where 0 represents missing/unrecorded physiological values
    zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_as_missing:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            
    # Apply feature engineering
    df = engineer_diabetes_features(df)
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome'].astype(int)
    
    numeric_features = [c for c in X.columns if c not in ['Age_Over_45', 'High_Glucose_Flag']]
    categorical_features = ['Age_Over_45', 'High_Glucose_Flag']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    return X, y, preprocessor, df


def load_and_clean_heart_data(file_path: str):
    """
    Loads and cleans the UCI Cleveland Heart Disease dataset.
    """
    df = pd.read_csv(file_path)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Apply feature engineering
    df = engineer_heart_features(df)
    
    X = df.drop(columns=['target'])
    y = df['target'].astype(int)
    
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'HR_Reserve_Ratio', 'Chol_BP_Index', 'Ischemia_Index']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'Male_Over_50']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    return X, y, preprocessor, df


def load_and_clean_hypertension_data(file_path: str):
    """
    Loads and cleans the Framingham dataset for Hypertension risk estimation.
    """
    df = pd.read_csv(file_path)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Drop non-predictive or target-leakage columns
    drop_cols = ['prevalentHyp', 'TenYearCHD']
    if 'education' in df.columns:
        drop_cols.append('education')
        
    # Apply feature engineering
    df = engineer_hypertension_features(df)
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['prevalentHyp'].astype(int)
    
    numeric_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'Pulse_Pressure', 'Mean_Arterial_Pressure', 'Smoking_Burden']
    categorical_features = ['male', 'currentSmoker', 'BPMeds', 'diabetes', 'High_BMI_Flag', 'High_Glucose_Flag']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    return X, y, preprocessor, df


def process_and_split_all(data_raw_dir: str, data_processed_dir: str):
    """
    Processes all 3 datasets, performs stratified train-test splits,
    and saves the processed splits.
    """
    os.makedirs(data_processed_dir, exist_ok=True)
    datasets = {}
    
    # 1. Diabetes
    diabetes_raw = os.path.join(data_raw_dir, 'diabetes.csv')
    if os.path.exists(diabetes_raw):
        X_d, y_d, prep_d, full_d = load_and_clean_diabetes_data(diabetes_raw)
        X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
            X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
        )
        datasets['diabetes'] = {
            'X_train': X_train_d, 'X_test': X_test_d,
            'y_train': y_train_d, 'y_test': y_test_d,
            'preprocessor': prep_d, 'raw_df': full_d
        }
        full_d.to_csv(os.path.join(data_processed_dir, 'diabetes_processed.csv'), index=False)
        print("Processed Diabetes dataset: train shape", X_train_d.shape, "test shape", X_test_d.shape)
        
    # 2. Heart Disease
    heart_raw = os.path.join(data_raw_dir, 'heart.csv')
    if os.path.exists(heart_raw):
        X_h, y_h, prep_h, full_h = load_and_clean_heart_data(heart_raw)
        X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
            X_h, y_h, test_size=0.2, random_state=42, stratify=y_h
        )
        datasets['heart'] = {
            'X_train': X_train_h, 'X_test': X_test_h,
            'y_train': y_train_h, 'y_test': y_test_h,
            'preprocessor': prep_h, 'raw_df': full_h
        }
        full_h.to_csv(os.path.join(data_processed_dir, 'heart_processed.csv'), index=False)
        print("Processed Heart dataset: train shape", X_train_h.shape, "test shape", X_test_h.shape)

    # 3. Hypertension
    hyp_raw = os.path.join(data_raw_dir, 'hypertension.csv')
    if os.path.exists(hyp_raw):
        X_hyp, y_hyp, prep_hyp, full_hyp = load_and_clean_hypertension_data(hyp_raw)
        X_train_hyp, X_test_hyp, y_train_hyp, y_test_hyp = train_test_split(
            X_hyp, y_hyp, test_size=0.2, random_state=42, stratify=y_hyp
        )
        datasets['hypertension'] = {
            'X_train': X_train_hyp, 'X_test': X_test_hyp,
            'y_train': y_train_hyp, 'y_test': y_test_hyp,
            'preprocessor': prep_hyp, 'raw_df': full_hyp
        }
        full_hyp.to_csv(os.path.join(data_processed_dir, 'hypertension_processed.csv'), index=False)
        print("Processed Hypertension dataset: train shape", X_train_hyp.shape, "test shape", X_test_hyp.shape)
        
    return datasets


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    process_and_split_all(raw_dir, proc_dir)
