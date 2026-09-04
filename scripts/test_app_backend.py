"""
Backend verification script to validate prediction engine, intervention engine,
status persistence, and batch scoring.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATA_PATH, MODEL_PATH, get_risk_level_from_score
from src.data_loader import load_raw_data
from src.prediction import load_trained_pipeline, batch_score_dataset, predict_single_student
from src.intervention import InterventionEngine, InterventionStore

def test_full_pipeline():
    print("1. Testing Model Loading...")
    bundle = load_trained_pipeline(MODEL_PATH)
    assert bundle is not None, "Failed to load bundle"
    print(f"   Model Name: {bundle['model_name']}")
    print(f"   Test Accuracy: {bundle['metrics']['accuracy']:.4f}")
    print(f"   Dropout Recall: {bundle['metrics']['dropout_recall']:.4f}")
    
    print("\n2. Testing Batch Scoring on dataset.csv...")
    df = load_raw_data(DATA_PATH)
    scored_df = batch_score_dataset(df, bundle)
    assert len(scored_df) == len(df), "Scored DF length mismatch"
    assert "Risk_Score" in scored_df.columns
    assert "Risk_Level" in scored_df.columns
    print(f"   Batch Scored {len(scored_df):,} students successfully.")
    print("   Risk Level Distribution:")
    print(scored_df["Risk_Level"].value_counts())
    
    print("\n3. Testing Single Student Prediction & Explainability...")
    sample_student = df.iloc[0].to_dict()
    diag = predict_single_student(sample_student, bundle)
    print(f"   Predicted Dropout Risk: {diag['risk_score_pct']}% ({diag['risk_label']})")
    print(f"   Top Risk Drivers: {[f['feature'] for f in diag['top_risk_factors'][:3]]}")
    
    print("\n4. Testing Intervention Engine...")
    interventions = InterventionEngine.generate_recommendations(
        student_data=sample_student,
        risk_level=diag['risk_label'],
        risk_score_pct=diag['risk_score_pct'],
        top_risk_factors=diag['top_risk_factors']
    )
    print(f"   Generated {len(interventions)} actionable interventions:")
    for item in interventions:
        print(f"     - [{item['priority']} Priority] {item['title']} (Owner: {item['owner']})")
        
    print("\n5. Testing Intervention Persistence Store...")
    store = InterventionStore()
    store.update_student_record("STU-1000", "In Progress", "Meeting scheduled with faculty advisor.")
    rec = store.get_student_record("STU-1000")
    assert rec["status"] == "In Progress"
    print(f"   Stored record verified: {rec}")
    
    print("\nALL BACKEND TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_full_pipeline()
