"""
Model Evaluation and Metric Calculation Module for HealthPredict.
Calculates Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score,
ROC-AUC, Confusion Matrices, and generates visualization artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def evaluate_model(model, X_test, y_test, model_name: str, disease_name: str):
    """
    Evaluates a trained classifier pipeline on test data.
    """
    y_pred = model.predict(X_test)
    
    # Calculate probabilities if supported
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        decision = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-decision))
    else:
        y_prob = y_pred.astype(float)
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    return {
        'disease': disease_name,
        'model_name': model_name,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'specificity': float(specificity),
        'f1': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    }


def save_eda_visualizations(datasets: dict, output_dir: str):
    """
    Generates and saves publication-quality EDA figures for the project.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    
    # 1. Diabetes correlation heatmap & target distribution
    if 'diabetes' in datasets:
        df_d = datasets['diabetes']['raw_df']
        plt.figure(figsize=(10, 8))
        corr_d = df_d.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_d, annot=True, fmt=".2f", cmap='Blues', cbar=True, square=True)
        plt.title('Diabetes Dataset Feature Correlation Matrix', fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diabetes_correlation_matrix.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Outcome', data=df_d, palette=['#4CAF50', '#F44336'])
        plt.title('Diabetes Prevalence Distribution (0=Negative, 1=Positive)', fontsize=13)
        plt.xlabel('Diabetes Status')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'diabetes_target_distribution.png'), dpi=300)
        plt.close()

    # 2. Heart Disease heatmap & features
    if 'heart' in datasets:
        df_h = datasets['heart']['raw_df']
        plt.figure(figsize=(12, 9))
        corr_h = df_h.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_h, annot=False, cmap='RdBu_r', cbar=True)
        plt.title('Heart Disease Dataset Correlation Heatmap', fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heart_correlation_matrix.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df_h, x='age', hue='target', multiple='stack', palette=['#2196F3', '#E91E63'])
        plt.title('Heart Disease Distribution Across Age Groups', fontsize=13)
        plt.xlabel('Age (Years)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heart_age_distribution.png'), dpi=300)
        plt.close()

    # 3. Hypertension heatmap & blood pressure
    if 'hypertension' in datasets:
        df_hyp = datasets['hypertension']['raw_df']
        plt.figure(figsize=(11, 8))
        corr_hyp = df_hyp.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_hyp, annot=False, cmap='magma', cbar=True)
        plt.title('Hypertension & Hemodynamic Feature Correlation Matrix', fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hypertension_correlation_matrix.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='prevalentHyp', y='sysBP', data=df_hyp, palette=['#009688', '#FF5722'])
        plt.title('Systolic Blood Pressure (sysBP) vs Hypertension Diagnosis', fontsize=13)
        plt.xlabel('Hypertension (0 = Normotensive, 1 = Hypertensive)')
        plt.ylabel('Systolic Blood Pressure (mmHg)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hypertension_sysbp_boxplot.png'), dpi=300)
        plt.close()


def save_evaluation_visualizations(results_list: list, output_dir: str):
    """
    Saves ROC Curves and Confusion Matrices for each disease and model.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_results = pd.DataFrame(results_list)
    
    # Save model comparison bar chart per disease
    diseases = df_results['disease'].unique()
    for d in diseases:
        df_d = df_results[df_results['disease'] == d]
        
        plt.figure(figsize=(10, 6))
        df_melt = df_d.melt(
            id_vars=['model_name'],
            value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            var_name='Metric',
            value_name='Score'
        )
        sns.barplot(x='model_name', y='Score', hue='Metric', data=df_melt, palette='Set2')
        plt.title(f'Model Performance Benchmark – {d.capitalize()} Risk', fontsize=14, pad=12)
        plt.xlabel('Algorithm')
        plt.ylabel('Score (0.0 to 1.0)')
        plt.ylim(0, 1.05)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{d}_model_comparison.png'), dpi=300)
        plt.close()
