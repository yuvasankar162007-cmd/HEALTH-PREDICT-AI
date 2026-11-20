"""
Model Training and Selection Module for HealthPredict.
Trains Logistic Regression, Decision Tree, Random Forest, SVM, and XGBoost.
Evaluates all models, selects the best performing pipeline prioritizing Recall & ROC-AUC,
and serializes artifacts and metadata.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.data_preprocessing import process_and_split_all
from src.evaluate_models import (
    evaluate_model,
    save_eda_visualizations,
    save_evaluation_visualizations
)


def get_models_dict():
    """
    Returns candidate classification models with tuned regularization.
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=6, min_samples_split=4, random_state=42),
        'Support Vector Machine': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, eval_metric='logloss', random_state=42)
    }


def compute_screening_selection_score(metrics: dict) -> float:
    """
    Computes a composite score tailored for healthcare risk screening.
    Prioritizes Recall (Sensitivity) to minimize false negatives,
    balanced with ROC-AUC discrimination power and F1-score.
    """
    recall = metrics.get('recall', 0.0)
    roc_auc = metrics.get('roc_auc', 0.0)
    f1 = metrics.get('f1', 0.0)
    acc = metrics.get('accuracy', 0.0)
    
    # 40% Recall, 30% ROC-AUC, 15% F1, 15% Accuracy
    return (0.40 * recall) + (0.30 * roc_auc) + (0.15 * f1) + (0.15 * acc)


def train_and_evaluate_all():
    """
    Orchestrates data loading, preprocessing, model training, evaluation,
    model selection, and persistence.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    figures_dir = os.path.join(base_dir, 'outputs', 'figures')
    reports_dir = os.path.join(base_dir, 'outputs', 'reports')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    print(">>> 1. Processing and splitting datasets...")
    datasets = process_and_split_all(raw_dir, proc_dir)
    
    print(">>> 2. Generating and saving EDA figures...")
    save_eda_visualizations(datasets, figures_dir)
    
    all_results = []
    best_models_info = {}
    
    for disease_name, data in datasets.items():
        print(f"\n==================== TRAINING MODELS: {disease_name.upper()} ====================")
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        preprocessor = data['preprocessor']
        
        models_to_train = get_models_dict()
        best_pipeline = None
        best_score = -1.0
        best_model_name = ""
        best_metrics = None
        
        for model_name, clf in models_to_train.items():
            print(f"  --> Training {model_name}...")
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])
            pipe.fit(X_train, y_train)
            
            metrics = evaluate_model(pipe, X_test, y_test, model_name, disease_name)
            metrics['composite_screening_score'] = compute_screening_selection_score(metrics)
            all_results.append(metrics)
            
            print(f"      Accuracy: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | "
                  f"ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f} | Score: {metrics['composite_screening_score']:.4f}")
            
            if metrics['composite_screening_score'] > best_score:
                best_score = metrics['composite_screening_score']
                best_pipeline = pipe
                best_model_name = model_name
                best_metrics = metrics
                
        # Save the best model pipeline for this disease
        model_filename = f"{disease_name}_pipeline.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(best_pipeline, model_path)
        print(f"  [+] BEST MODEL SELECTED for {disease_name}: {best_model_name} (Composite Score: {best_score:.4f})")
        print(f"      Saved to {model_path}")
        
        best_models_info[disease_name] = {
            'best_model_name': best_model_name,
            'model_file': model_filename,
            'selection_score': best_score,
            'metrics': best_metrics,
            'feature_columns': list(X_train.columns)
        }
        
    print("\n>>> 3. Generating evaluation comparison charts...")
    save_evaluation_visualizations(all_results, figures_dir)
    
    # Save comparison table CSV and JSON
    df_metrics = pd.DataFrame(all_results)
    csv_report_path = os.path.join(reports_dir, 'model_comparison_report.csv')
    df_metrics.to_csv(csv_report_path, index=False)
    print(f"Saved full model comparison table to {csv_report_path}")
    
    json_metrics_path = os.path.join(models_dir, 'model_metrics.json')
    with open(json_metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'all_benchmarks': all_results,
            'best_models': best_models_info
        }, f, indent=2)
    print(f"Saved metrics benchmark JSON to {json_metrics_path}")
    
    # Save feature metadata schema
    save_feature_metadata(models_dir)
    print("\n>>> ALL MODELS TRAINED, EVALUATED, AND SAVED SUCCESSFULLY! <<<")


def save_feature_metadata(models_dir: str):
    """
    Saves comprehensive input schema and domain bounds for real-time validation and UI mapping.
    """
    metadata = {
        'diabetes': {
            'raw_features': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
            'ranges': {
                'Pregnancies': {'min': 0, 'max': 20, 'default': 0, 'step': 1, 'type': 'int'},
                'Glucose': {'min': 40, 'max': 300, 'default': 100, 'step': 1, 'type': 'float', 'unit': 'mg/dL'},
                'BloodPressure': {'min': 40, 'max': 220, 'default': 75, 'step': 1, 'type': 'float', 'unit': 'mmHg'},
                'SkinThickness': {'min': 5, 'max': 99, 'default': 20, 'step': 1, 'type': 'float', 'unit': 'mm'},
                'Insulin': {'min': 0, 'max': 900, 'default': 80, 'step': 1, 'type': 'float', 'unit': 'mu U/ml'},
                'BMI': {'min': 10.0, 'max': 65.0, 'default': 24.5, 'step': 0.1, 'type': 'float', 'unit': 'kg/m²'},
                'DiabetesPedigreeFunction': {'min': 0.05, 'max': 2.5, 'default': 0.47, 'step': 0.01, 'type': 'float'},
                'Age': {'min': 18, 'max': 100, 'default': 35, 'step': 1, 'type': 'int', 'unit': 'years'}
            }
        },
        'heart': {
            'raw_features': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
            'ranges': {
                'age': {'min': 18, 'max': 100, 'default': 50, 'step': 1, 'type': 'int', 'unit': 'years'},
                'sex': {'options': {'Female': 0, 'Male': 1}, 'default': 1, 'type': 'categorical'},
                'cp': {'options': {'Typical Angina (0)': 0, 'Atypical Angina (1)': 1, 'Non-anginal Pain (2)': 2, 'Asymptomatic (3)': 3}, 'default': 0, 'type': 'categorical'},
                'trestbps': {'min': 80, 'max': 220, 'default': 120, 'step': 1, 'type': 'int', 'unit': 'mmHg'},
                'chol': {'min': 100, 'max': 600, 'default': 210, 'step': 1, 'type': 'int', 'unit': 'mg/dL'},
                'fbs': {'options': {'Fasting Blood Sugar <= 120 mg/dL (0)': 0, 'Fasting Blood Sugar > 120 mg/dL (1)': 1}, 'default': 0, 'type': 'categorical'},
                'restecg': {'options': {'Normal (0)': 0, 'ST-T Wave Abnormality (1)': 1, 'Left Ventricular Hypertrophy (2)': 2}, 'default': 0, 'type': 'categorical'},
                'thalach': {'min': 60, 'max': 220, 'default': 150, 'step': 1, 'type': 'int', 'unit': 'bpm'},
                'exang': {'options': {'No (0)': 0, 'Yes (1)': 1}, 'default': 0, 'type': 'categorical'},
                'oldpeak': {'min': 0.0, 'max': 7.0, 'default': 1.0, 'step': 0.1, 'type': 'float', 'unit': 'mm'},
                'slope': {'options': {'Upsloping (0)': 0, 'Flat (1)': 1, 'Downsloping (2)': 2}, 'default': 1, 'type': 'categorical'},
                'ca': {'min': 0, 'max': 4, 'default': 0, 'step': 1, 'type': 'int'},
                'thal': {'options': {'Normal (1)': 1, 'Fixed Defect (2)': 2, 'Reversible Defect (3)': 3}, 'default': 2, 'type': 'categorical'}
            }
        },
        'hypertension': {
            'raw_features': ['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose', 'diabetes'],
            'ranges': {
                'male': {'options': {'Female': 0, 'Male': 1}, 'default': 1, 'type': 'categorical'},
                'age': {'min': 18, 'max': 100, 'default': 45, 'step': 1, 'type': 'int', 'unit': 'years'},
                'currentSmoker': {'options': {'Non-Smoker (0)': 0, 'Current Smoker (1)': 1}, 'default': 0, 'type': 'categorical'},
                'cigsPerDay': {'min': 0, 'max': 70, 'default': 0, 'step': 1, 'type': 'int'},
                'BPMeds': {'options': {'Not on Blood Pressure Meds (0)': 0, 'On Blood Pressure Meds (1)': 1}, 'default': 0, 'type': 'categorical'},
                'totChol': {'min': 100, 'max': 600, 'default': 200, 'step': 1, 'type': 'int', 'unit': 'mg/dL'},
                'sysBP': {'min': 80, 'max': 240, 'default': 120, 'step': 1, 'type': 'float', 'unit': 'mmHg'},
                'diaBP': {'min': 40, 'max': 150, 'default': 80, 'step': 1, 'type': 'float', 'unit': 'mmHg'},
                'BMI': {'min': 10.0, 'max': 65.0, 'default': 24.5, 'step': 0.1, 'type': 'float', 'unit': 'kg/m²'},
                'heartRate': {'min': 40, 'max': 180, 'default': 72, 'step': 1, 'type': 'float', 'unit': 'bpm'},
                'glucose': {'min': 40, 'max': 400, 'default': 85, 'step': 1, 'type': 'float', 'unit': 'mg/dL'},
                'diabetes': {'options': {'Non-Diabetic (0)': 0, 'Diabetic (1)': 1}, 'default': 0, 'type': 'categorical'}
            }
        }
    }
    
    metadata_path = os.path.join(models_dir, 'feature_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved feature metadata schema to {metadata_path}")


if __name__ == '__main__':
    train_and_evaluate_all()
