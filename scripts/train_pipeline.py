"""
Complete Machine Learning Training, EDA, Comparison, Tuning, and Serialization Pipeline.
Evaluates Logistic Regression, Decision Tree, Random Forest, and XGBoost/GradientBoosting.
Prioritizes Dropout Recall and Macro F1 score to minimize false negatives for student dropout risk.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import (
    DATA_PATH, MODEL_PATH, TARGET_COL, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
)
from src.data_loader import load_raw_data, prepare_train_test_split

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

def run_eda(df: pd.DataFrame):
    """
    Generate comprehensive Exploratory Data Analysis charts and save them as static figures.
    """
    print("=" * 60)
    print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    # 1. Target Class Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts = df[TARGET_COL].value_counts()
    colors = ['#10B981' if c == 'Graduate' else '#EF4444' if c == 'Dropout' else '#F59E0B' for c in target_counts.index]
    bars = ax.bar(target_counts.index, target_counts.values, color=colors, edgecolor='black', alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{height:,} ({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title("Student Outcome Class Distribution", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Number of Students", fontsize=12)
    ax.set_ylim(0, max(target_counts.values) * 1.15)
    plt.tight_layout()
    fig.savefig("reports/figures/01_target_distribution.png", dpi=300)
    plt.close()
    print("Saved: reports/figures/01_target_distribution.png")

    # 2. Key Academic & Financial Features Correlation with Target (Encoded)
    df_encoded = df.copy()
    # Map target: Dropout=1, Enrolled=0.5, Graduate=0 for simple directional correlation
    risk_numeric = {'Dropout': 1.0, 'Enrolled': 0.5, 'Graduate': 0.0}
    df_encoded['Dropout_Risk_Indicator'] = df_encoded[TARGET_COL].map(risk_numeric)
    
    numeric_cols = [c for c in df_encoded.select_dtypes(include=[np.number]).columns if c not in ['Student_ID']]
    corr_series = df_encoded[numeric_cols].corr()['Dropout_Risk_Indicator'].drop('Dropout_Risk_Indicator').sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    top_corr = pd.concat([corr_series.head(8), corr_series.tail(8)])
    corr_colors = ['#10B981' if val < 0 else '#EF4444' for val in top_corr.values]
    bars = ax.barh(top_corr.index, top_corr.values, color=corr_colors, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_title("Top Correlated Features with Dropout Risk", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Pearson Correlation with Dropout Risk Indicator", fontsize=12)
    plt.tight_layout()
    fig.savefig("reports/figures/02_top_correlations.png", dpi=300)
    plt.close()
    print("Saved: reports/figures/02_top_correlations.png")

    # 3. Key Feature Distributions by Target Class
    key_features = [
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 1st sem (approved)',
        'Age at enrollment',
        'Tuition fees up to date',
        'Scholarship holder'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    palette = {'Graduate': '#10B981', 'Dropout': '#EF4444', 'Enrolled': '#F59E0B'}
    
    for i, feature in enumerate(key_features):
        if feature in df.columns:
            if df[feature].nunique() <= 3:
                # Proportional bar chart for binary/categorical
                prop_df = df.groupby([feature, TARGET_COL]).size().unstack(fill_value=0)
                prop_df = prop_df.div(prop_df.sum(axis=1), axis=0) * 100
                prop_df[['Dropout', 'Enrolled', 'Graduate']].plot(
                    kind='bar', stacked=True, ax=axes[i], 
                    color=['#EF4444', '#F59E0B', '#10B981'], edgecolor='black', alpha=0.85
                )
                axes[i].set_ylabel("Percentage (%)")
                axes[i].set_title(f"{feature} vs Outcome", fontweight='bold')
                axes[i].legend(title="Outcome", loc='upper right', fontsize=8)
            else:
                for target_class in ['Graduate', 'Enrolled', 'Dropout']:
                    subset = df[df[TARGET_COL] == target_class][feature]
                    sns.kdeplot(subset, ax=axes[i], label=target_class, color=palette[target_class], linewidth=2.2, fill=True, alpha=0.2)
                axes[i].set_title(f"{feature} Distribution", fontweight='bold')
                axes[i].set_ylabel("Density")
                axes[i].legend(title="Outcome", loc='upper right')
    
    plt.suptitle("Key Predictive Signals Partitioned by Student Outcome", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig("reports/figures/03_feature_distributions.png", dpi=300)
    plt.close()
    print("Saved: reports/figures/03_feature_distributions.png")

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """
    Train and compare multiple models with stratified evaluation.
    Computes precision, recall, F1, ROC-AUC, and focuses heavily on Dropout Recall.
    """
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING, BENCHMARKING & COMPARISON")
    print("=" * 60)
    
    # Define baseline and candidate models
    models = {
        "Logistic Regression (Scaled)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight='balanced_subsample',
            min_samples_split=5, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
        )
    }
    
    if HAS_XGBOOST:
        # For XGBoost multi-class, target needs 0, 1, 2 encoding
        y_train_num = y_train.map(CLASS_TO_IDX)
        y_test_num = y_test.map(CLASS_TO_IDX)
        models["XGBoost Classifier"] = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.06, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss',
            random_state=42, n_jobs=-1
        )
    
    results = []
    trained_models = {}
    
    y_test_bin = label_binarize(y_test, classes=CLASSES)
    
    for name, model in models.items():
        print(f"\n--- Training & Evaluating: {name} ---")
        if "XGBoost" in name:
            model.fit(X_train, y_train_num)
            preds_num = model.predict(X_test)
            probs = model.predict_proba(X_test)
            preds = [IDX_TO_CLASS[p] for p in preds_num]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
            
        trained_models[name] = model
        
        # Calculate comprehensive metrics
        acc = accuracy_score(y_test, preds)
        macro_prec = precision_score(y_test, preds, average='macro')
        macro_rec = recall_score(y_test, preds, average='macro')
        macro_f1 = f1_score(y_test, preds, average='macro')
        
        # Per-class metrics
        class_recalls = recall_score(y_test, preds, average=None, labels=CLASSES)
        class_f1s = f1_score(y_test, preds, average=None, labels=CLASSES)
        
        dropout_recall = class_recalls[CLASS_TO_IDX['Dropout']]
        dropout_f1 = class_f1s[CLASS_TO_IDX['Dropout']]
        
        roc_auc_ovr = roc_auc_score(y_test_bin, probs, multi_class='ovr', average='macro')
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro F1": macro_f1,
            "Macro Recall": macro_rec,
            "Macro Precision": macro_prec,
            "Dropout Recall": dropout_recall,
            "Dropout F1": dropout_f1,
            "ROC-AUC (OvR)": roc_auc_ovr,
            "Predictions": preds,
            "Probabilities": probs
        })
        
        print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Dropout Recall: {dropout_recall:.4f} | ROC-AUC: {roc_auc_ovr:.4f}")
    
    metrics_df = pd.DataFrame(results).drop(columns=['Predictions', 'Probabilities'])
    print("\n--- MODEL BENCHMARK COMPARISON TABLE ---")
    print(metrics_df.to_string(index=False))
    
    return results, trained_models

def tune_best_model(X_train, y_train, feature_names):
    """
    Perform Hyperparameter Tuning using Stratified 5-Fold Cross Validation on top ensemble candidate.
    Optimizes for Macro F1 and Dropout Recall.
    """
    print("\n" + "=" * 60, flush=True)
    print("STEP 3: HYPERPARAMETER TUNING (RANDOM FOREST / XGBOOST)", flush=True)
    print("=" * 60, flush=True)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if HAS_XGBOOST:
        print("Tuning XGBoost Classifier with 5-Fold CV...", flush=True)
        y_train_num = y_train.map(CLASS_TO_IDX)
        param_grid = {
            'n_estimators': [150, 250],
            'max_depth': [3, 4],
            'learning_rate': [0.05, 0.08],
            'subsample': [0.85],
            'colsample_bytree': [0.85]
        }
        xgb_base = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42, n_jobs=1)
        grid_search = GridSearchCV(
            xgb_base, param_grid, cv=cv, scoring='f1_macro', n_jobs=4, verbose=0
        )
        grid_search.fit(X_train, y_train_num)
        best_estimator = grid_search.best_estimator_
        print(f"Best XGBoost Params: {grid_search.best_params_}", flush=True)
        print(f"Best CV Macro F1: {grid_search.best_score_:.4f}", flush=True)
    else:
        print("Tuning Random Forest Classifier with 5-Fold CV...", flush=True)
        param_grid = {
            'n_estimators': [150, 250],
            'max_depth': [8, 12],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced_subsample']
        }
        rf_base = RandomForestClassifier(random_state=42, n_jobs=1)
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=cv, scoring='f1_macro', n_jobs=4, verbose=0
        )
        grid_search.fit(X_train, y_train)
        best_estimator = grid_search.best_estimator_
        print(f"Best RF Params: {grid_search.best_params_}", flush=True)
        print(f"Best CV Macro F1: {grid_search.best_score_:.4f}", flush=True)
        
    return best_estimator, grid_search.best_params_, grid_search.best_score_

def generate_evaluation_artifacts(best_model, X_test, y_test, results, feature_names):
    """
    Generate ROC curves, Confusion Matrix, and Global Feature Importance charts.
    """
    print("\n" + "=" * 60)
    print("STEP 4: GENERATING EVALUATION CHARTS & ARTIFACTS")
    print("=" * 60)
    
    # 1. Model Comparison Chart
    comp_df = pd.DataFrame(results).drop(columns=['Predictions', 'Probabilities'])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(comp_df))
    width = 0.22
    
    ax.bar(x - width, comp_df['Accuracy'], width, label='Accuracy', color='#3B82F6', alpha=0.85, edgecolor='black')
    ax.bar(x, comp_df['Macro F1'], width, label='Macro F1', color='#10B981', alpha=0.85, edgecolor='black')
    ax.bar(x + width, comp_df['Dropout Recall'], width, label='Dropout Recall (Critical)', color='#EF4444', alpha=0.85, edgecolor='black')
    
    ax.set_ylabel('Score (0.0 - 1.0)', fontsize=12)
    ax.set_title('Comprehensive Model Benchmark Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['Model'], rotation=15, ha='right', fontsize=10)
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    fig.savefig("reports/figures/04_model_comparison.png", dpi=300)
    plt.close()
    print("Saved: reports/figures/04_model_comparison.png")

    # Evaluate best model on test set
    if HAS_XGBOOST and isinstance(best_model, xgb.XGBClassifier):
        y_test_num = y_test.map(CLASS_TO_IDX)
        y_pred_num = best_model.predict(X_test)
        y_probs = best_model.predict_proba(X_test)
        y_pred = [IDX_TO_CLASS[p] for p in y_pred_num]
    else:
        y_pred = best_model.predict(X_test)
        y_probs = best_model.predict_proba(X_test)
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=True,
                linewidths=1, linecolor='black')
    ax.set_title("Normalized Confusion Matrix (Final Model)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    fig.savefig("reports/figures/05_confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved: reports/figures/05_confusion_matrix.png")

    # 3. Multi-class ROC Curves
    y_test_bin = label_binarize(y_test, classes=CLASSES)
    fig, ax = plt.subplots(figsize=(8, 6))
    class_colors = {'Dropout': '#EF4444', 'Enrolled': '#F59E0B', 'Graduate': '#10B981'}
    
    for i, cls in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=class_colors[cls], lw=2.5,
                label=f'{cls} (AUC = {roc_auc:.3f})')
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('One-vs-Rest ROC Curves for Student Outcome Classes', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    fig.savefig("reports/figures/06_roc_curves.png", dpi=300)
    plt.close()
    print("Saved: reports/figures/06_roc_curves.png")

    # 4. Global Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['clf'], 'feature_importances_'):
        importances = best_model.named_steps['clf'].feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
        
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    top_15 = feat_imp_df.head(15).iloc[::-1]
    ax.barh(top_15['Feature'], top_15['Importance'], color='#3B82F6', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Relative Feature Importance (MDI / Gain)', fontsize=12)
    ax.set_title('Top 15 Predictive Features for Student Dropout Risk', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig("reports/figures/07_global_feature_importance.png", dpi=300)
    plt.close()
    print("Saved: reports/figures/07_global_feature_importance.png")
    
    return feat_imp_df, cm, y_probs

def main():
    print("Loading data from dataset.csv...")
    df = load_raw_data(DATA_PATH)
    print(f"Dataset Loaded Successfully. Total Records: {len(df):,}, Columns: {len(df.columns)}")
    
    # Run EDA
    run_eda(df)
    
    # Train / Test split
    X_train, X_test, y_train, y_test, feature_names = prepare_train_test_split(df, test_size=0.20, random_state=42)
    print(f"Training Samples: {len(X_train):,} | Test Samples: {len(X_test):,}")
    
    # Train & benchmark models
    results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names)
    
    # Hyperparameter tuning
    best_tuned_model, best_params, cv_score = tune_best_model(X_train, y_train, feature_names)
    
    # Generate charts and feature importance
    feat_imp_df, cm, y_probs = generate_evaluation_artifacts(best_tuned_model, X_test, y_test, results, feature_names)
    
    # Fit SHAP Explainer for in-app explainability
    shap_explainer = None
    if HAS_SHAP:
        print("\nFitting SHAP TreeExplainer for local prediction attributions...")
        try:
            shap_explainer = shap.TreeExplainer(best_tuned_model)
            print("SHAP TreeExplainer successfully initialized!")
        except Exception as e:
            print(f"Note: SHAP explainer fallback due to: {e}")
            shap_explainer = None
            
    # Calculate final evaluation metrics on test set
    if HAS_XGBOOST and isinstance(best_tuned_model, xgb.XGBClassifier):
        y_test_num = y_test.map(CLASS_TO_IDX)
        test_preds_num = best_tuned_model.predict(X_test)
        test_probs = best_tuned_model.predict_proba(X_test)
        test_preds = [IDX_TO_CLASS[p] for p in test_preds_num]
    else:
        test_preds = best_tuned_model.predict(X_test)
        test_probs = best_tuned_model.predict_proba(X_test)

    final_accuracy = accuracy_score(y_test, test_preds)
    final_macro_f1 = f1_score(y_test, test_preds, average='macro')
    final_dropout_recall = recall_score(y_test, test_preds, average=None, labels=CLASSES)[CLASS_TO_IDX['Dropout']]
    y_test_bin = label_binarize(y_test, classes=CLASSES)
    final_roc_auc = roc_auc_score(y_test_bin, test_probs, multi_class='ovr', average='macro')

    # Serialize trained pipeline bundle to disk
    bundle = {
        "model": best_tuned_model,
        "model_name": "Tuned XGBoost Classifier" if HAS_XGBOOST else "Tuned Random Forest Classifier",
        "feature_names": feature_names,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "feature_importance_df": feat_imp_df,
        "shap_explainer": shap_explainer,
        "metrics": {
            "accuracy": final_accuracy,
            "macro_f1": final_macro_f1,
            "dropout_recall": final_dropout_recall,
            "roc_auc_ovr": final_roc_auc,
            "comparison_results": pd.DataFrame(results).drop(columns=['Predictions', 'Probabilities']).to_dict(orient='records')
        },
        "best_params": best_params
    }
    
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSUCCESS! Model pipeline bundle persisted to: {MODEL_PATH}")
    print(f"Final Model Metrics on Holdout Test Set:")
    print(f"  - Accuracy: {final_accuracy*100:.2f}%")
    print(f"  - Macro F1: {final_macro_f1*100:.2f}%")
    print(f"  - Dropout Recall: {final_dropout_recall*100:.2f}% (High-Risk Capture Rate)")
    print(f"  - ROC-AUC (OvR): {final_roc_auc:.4f}")

if __name__ == "__main__":
    main()
