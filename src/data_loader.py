"""
Data loading and preprocessing utilities for the Student Dropout Risk Prediction System.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH, TARGET_COL, CLASS_TO_IDX, IDX_TO_CLASS

def load_raw_data(filepath: str = None) -> pd.DataFrame:
    """
    Load dataset from specified path or fallback default paths.
    """
    if filepath and os.path.exists(filepath):
        df = pd.read_csv(filepath)
    elif os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    elif os.path.exists("dataset.csv"):
        df = pd.read_csv("dataset.csv")
    else:
        raise FileNotFoundError(f"Could not locate dataset. Checked '{filepath}', '{DATA_PATH}', and 'dataset.csv'")
    
    # Assign unique Student ID if not present
    if "Student_ID" not in df.columns:
        df.insert(0, "Student_ID", [f"STU-{1000 + i}" for i in range(len(df))])
        
    return df

def get_feature_and_target_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Extract feature column names (excluding ID and Target) and target column name.
    """
    feature_cols = [c for c in df.columns if c not in ["Student_ID", TARGET_COL]]
    return feature_cols, TARGET_COL

def prepare_train_test_split(
    df: pd.DataFrame, 
    test_size: float = 0.20, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepare stratified train-test split on original 3-class target.
    SMOTE or scaling should only be applied to training data inside pipeline/folds.
    """
    feature_cols, target_col = get_feature_and_target_columns(df)
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, feature_cols
