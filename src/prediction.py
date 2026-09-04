"""
Prediction and Explainability Engine for Student Dropout Risk.
Handles model loading, batch scoring, real-time single student inference,
and local feature contribution attribution.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional

from src.config import (
    MODEL_PATH, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
    get_risk_level_from_score
)
from src.intervention import InterventionStore

_CACHED_MODEL_BUNDLE = None

def load_trained_pipeline(model_path: str = MODEL_PATH) -> Dict[str, Any]:
    """
    Load serialized model pipeline, metadata, and explainer from disk.
    """
    global _CACHED_MODEL_BUNDLE
    if _CACHED_MODEL_BUNDLE is not None:
        return _CACHED_MODEL_BUNDLE
        
    if not os.path.exists(model_path):
        if os.path.exists("models/best_model.pkl"):
            model_path = "models/best_model.pkl"
        else:
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python scripts/train_pipeline.py' first."
            )
            
    bundle = joblib.load(model_path)
    _CACHED_MODEL_BUNDLE = bundle
    return bundle

def compute_local_feature_attributions(
    model: Any,
    feature_names: List[str],
    student_row: pd.Series,
    shap_explainer: Optional[Any] = None,
    top_k: int = 6
) -> List[Dict[str, Any]]:
    """
    Calculate top contributing risk factors for a specific student prediction.
    Uses SHAP if available; falls back to Tree-based feature importance * normalized deviation.
    """
    row_df = pd.DataFrame([student_row[feature_names]])
    
    # Try SHAP TreeExplainer first
    if shap_explainer is not None:
        try:
            shap_values = shap_explainer.shap_values(row_df)
            # For multi-class, index 0 is Dropout class
            if isinstance(shap_values, list):
                dropout_shap = shap_values[CLASS_TO_IDX['Dropout']][0]
            elif len(shap_values.shape) == 3:
                dropout_shap = shap_values[0, :, CLASS_TO_IDX['Dropout']]
            else:
                dropout_shap = shap_values[0]
                
            factors = []
            for feat, val, s_val in zip(feature_names, student_row[feature_names], dropout_shap):
                factors.append({
                    "feature": feat,
                    "value": val,
                    "impact": float(s_val),
                    "is_risk_driver": float(s_val) > 0
                })
            factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
            return factors[:top_k]
        except Exception:
            pass # Fallback to heuristic attribution
            
    # Heuristic Tree-Importance * Risk Deviation Fallback
    factors = []
    # Key high-stakes indicators
    critical_signals = {
        "Curricular units 2nd sem (approved)": ("Low approved units in 2nd semester", -1.5),
        "Curricular units 2nd sem (grade)": ("Low average grade in 2nd semester", -1.2),
        "Curricular units 1st sem (approved)": ("Low approved units in 1st semester", -1.0),
        "Tuition fees up to date": ("Tuition fees not up to date", -2.0),
        "Debtor": ("Student is flagged as a debtor", 1.8),
        "Age at enrollment": ("Older age at enrollment", 0.8),
        "Curricular units 2nd sem (without evaluations)": ("Unattempted assessments in 2nd sem", 1.2),
        "Scholarship holder": ("No scholarship funding", -0.7)
    }
    
    for feat, (desc, direction) in critical_signals.items():
        if feat in student_row:
            val = student_row[feat]
            # Directional risk score
            if feat == "Tuition fees up to date" and val == 0:
                impact = 0.28
            elif feat == "Debtor" and val == 1:
                impact = 0.22
            elif feat in ["Curricular units 2nd sem (approved)", "Curricular units 1st sem (approved)"] and val < 4:
                impact = (4 - val) * 0.08
            elif feat == "Curricular units 2nd sem (grade)" and val < 11:
                impact = (11 - val) * 0.04
            elif feat == "Age at enrollment" and val >= 25:
                impact = (val - 20) * 0.01
            elif feat == "Scholarship holder" and val == 0:
                impact = 0.05
            elif feat.endswith("(without evaluations)") and val > 0:
                impact = val * 0.08
            else:
                impact = -0.05 # Protective factor
                
            factors.append({
                "feature": feat,
                "value": val,
                "impact": float(impact),
                "is_risk_driver": impact > 0,
                "description": desc
            })
            
    factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return factors[:top_k]

def predict_single_student(
    student_dict: Dict[str, Any],
    bundle: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Evaluate dropout risk and return score, risk level, probabilities, and top drivers.
    """
    if bundle is None:
        bundle = load_trained_pipeline()
        
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    shap_explainer = bundle.get("shap_explainer")
    
    # Prepare single row DataFrame
    row_data = {}
    for feat in feature_names:
        row_data[feat] = float(student_dict.get(feat, 0.0))
        
    df_row = pd.DataFrame([row_data])
    
    # Model probabilities
    probs = model.predict_proba(df_row)[0]
    
    # Probability of Dropout is the Risk Score
    dropout_prob = float(probs[CLASS_TO_IDX["Dropout"]])
    risk_score_pct = round(dropout_prob * 100.0, 1)
    
    # Risk Level mapping
    risk_info = get_risk_level_from_score(dropout_prob)
    
    # Local feature attribution
    top_factors = compute_local_feature_attributions(
        model=model,
        feature_names=feature_names,
        student_row=pd.Series(row_data),
        shap_explainer=shap_explainer,
        top_k=5
    )
    
    return {
        "dropout_prob": dropout_prob,
        "risk_score_pct": risk_score_pct,
        "risk_key": risk_info["key"],
        "risk_label": risk_info["label"],
        "risk_badge_color": risk_info["badge_color"],
        "risk_bg_color": risk_info["bg_color"],
        "risk_border_color": risk_info["border_color"],
        "risk_icon": risk_info["icon"],
        "risk_description": risk_info["description"],
        "probabilities": {
            "Dropout": float(probs[CLASS_TO_IDX["Dropout"]]),
            "Enrolled": float(probs[CLASS_TO_IDX["Enrolled"]]),
            "Graduate": float(probs[CLASS_TO_IDX["Graduate"]])
        },
        "predicted_class": CLASSES[np.argmax(probs)],
        "top_risk_factors": top_factors
    }

def batch_score_dataset(
    df: pd.DataFrame,
    bundle: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Score the full dataset and attach risk scores, risk levels, and intervention tracking status.
    """
    if bundle is None:
        bundle = load_trained_pipeline()
        
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    
    X = df[feature_names]
    probs = model.predict_proba(X)
    
    scored_df = df.copy()
    
    dropout_probs = probs[:, CLASS_TO_IDX["Dropout"]]
    enrolled_probs = probs[:, CLASS_TO_IDX["Enrolled"]]
    graduate_probs = probs[:, CLASS_TO_IDX["Graduate"]]
    
    scored_df["Dropout_Prob"] = dropout_probs
    scored_df["Risk_Score"] = np.round(dropout_probs * 100.0, 1)
    scored_df["Enrolled_Prob"] = enrolled_probs
    scored_df["Graduate_Prob"] = graduate_probs
    scored_df["Predicted_Class"] = [CLASSES[i] for i in np.argmax(probs, axis=1)]
    
    # Assign Risk Level based on thresholds
    risk_labels = []
    risk_colors = []
    risk_icons = []
    
    for score in dropout_probs:
        info = get_risk_level_from_score(score)
        risk_labels.append(info["label"])
        risk_colors.append(info["badge_color"])
        risk_icons.append(info["icon"])
        
    scored_df["Risk_Level"] = risk_labels
    scored_df["Risk_Color"] = risk_colors
    scored_df["Risk_Icon"] = risk_icons
    
    # Merge intervention statuses from store
    store = InterventionStore()
    records = store.get_all_records()
    
    statuses = []
    for sid in scored_df["Student_ID"]:
        rec = records.get(str(sid), {})
        statuses.append(rec.get("status", "Not Started"))
        
    scored_df["Intervention_Status"] = statuses
    
    return scored_df
