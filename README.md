# Student Dropout Risk Prediction & Intervention System

A Machine Learning-based system designed to **identify students who may be at risk of dropping out**, understand the major factors behind their risk, and suggest suitable interventions.

---

## Key Features

1. **Dropout Risk Prediction**

   * Predicts whether a student is likely to **Graduate, Remain Enrolled, or Dropout**.
   * Generates a **Dropout Risk Score (0–100%)**.
   * Risk levels:

     * 🟢 **Low Risk:** Below 40%
     * 🟡 **Medium Risk:** 40–70%
     * 🔴 **High Risk:** Above 70%

2. **Machine Learning Model Comparison**

   * Compared multiple models including:

     * Logistic Regression
     * Decision Tree
     * Random Forest
     * Gradient Boosting
     * XGBoost
   * **Tuned XGBoost** was selected as the final model.
   * Accuracy: **77.74%**
   * Dropout Recall: **75.00%**
   * ROC-AUC: **0.8955**

3. **Risk Factor Analysis**

   * Identifies the major factors contributing to a student's dropout risk.
   * Uses **feature importance and SHAP** for explainability.

4. **Intervention System**

   * Suggests suitable actions based on identified risk factors, such as:

     * Academic tutoring
     * Financial counseling
     * Academic advising
     * Assessment support
     * Student outreach

5. **Interactive Dashboard**

   * Built using **Streamlit**.
   * Faculty can:

     * View overall student risk
     * Search and filter students
     * View individual risk reports
     * Check risk factors
     * Track intervention progress

---

## Model Performance

| Model               |   Accuracy | Dropout Recall |    ROC-AUC |
| ------------------- | ---------: | -------------: | ---------: |
| Logistic Regression |     73.90% |         67.96% |     0.8775 |
| Decision Tree       |     68.36% |         64.79% |     0.8346 |
| Random Forest       |     76.38% |         72.18% |     0.8874 |
| Gradient Boosting   |     76.27% |         72.54% |     0.8928 |
| **Tuned XGBoost**   | **77.74%** |     **75.00%** | **0.8955** |

### Final Model

**Tuned XGBoost** was selected because it achieved the **highest accuracy, dropout recall, and ROC-AUC** among the tested models.

---
