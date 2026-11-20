"""
HealthPredict – Exploratory Data Analysis & Model Training Walkthrough
Interactive reproducible notebook/script for data science inspection and presentation.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import process_and_split_all
from src.train_models import train_and_evaluate_all


def main():
    print("====================================================================")
    print("      HEALTHPREDICT: DATA SCIENCE & ML EXPLORATION WORKFLOW         ")
    print("====================================================================")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    
    print("\n[Step 1] Loading and preprocessing datasets...")
    datasets = process_and_split_all(raw_dir, proc_dir)
    
    for name, d in datasets.items():
        print(f"\n--- Dataset: {name.upper()} ---")
        raw_df = d['raw_df']
        print(f"Total shape: {raw_df.shape}")
        print("Feature summary statistics:")
        print(raw_df.describe().T[['mean', 'std', 'min', '50%', 'max']].head(6))
        
    print("\n[Step 2] Executing model training and benchmark comparisons...")
    train_and_evaluate_all()
    
    print("\n[Step 3] Verification complete! Check 'outputs/figures/' and 'models/model_metrics.json'.")


if __name__ == '__main__':
    main()
