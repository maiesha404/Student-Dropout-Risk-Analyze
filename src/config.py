"""
Configuration Module for Student Dropout Risk Prediction & Intervention System.
Central source of truth for risk thresholds, class mappings, UI styling, and metadata.
"""

from typing import Dict, Any

# Dataset Constants
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/best_model.pkl"
INTERVENTIONS_STORE_PATH = "data/interventions_store.json"

TARGET_COL = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Risk Level Thresholds (Probability of Dropout: 0.0 to 1.0)
# Low Risk: [0.0, 0.40), Medium Risk: [0.40, 0.70), High Risk: [0.70, 1.00]
RISK_THRESHOLDS = {
    "LOW_MAX": 0.40,
    "MEDIUM_MAX": 0.70,
}

# Risk Level Definitions & Badges
RISK_LEVELS = {
    "LOW": {
        "label": "Low Risk",
        "badge_color": "#10B981",  # Emerald Green
        "bg_color": "rgba(16, 185, 129, 0.15)",
        "border_color": "rgba(16, 185, 129, 0.4)",
        "icon": "🟢",
        "description": "Student is on a stable trajectory with low likelihood of dropout based on historical patterns."
    },
    "MEDIUM": {
        "label": "Medium Risk",
        "badge_color": "#F59E0B",  # Amber / Warm Yellow
        "bg_color": "rgba(245, 158, 11, 0.15)",
        "border_color": "rgba(245, 158, 11, 0.4)",
        "icon": "🟡",
        "description": "Student exhibits early warning indicators (academic friction or enrollment hesitation) requiring monitoring."
    },
    "HIGH": {
        "label": "High Risk",
        "badge_color": "#EF4444",  # Crimson Red
        "bg_color": "rgba(239, 68, 68, 0.15)",
        "border_color": "rgba(239, 68, 68, 0.4)",
        "icon": "🔴",
        "description": "Student is facing severe academic, financial, or engagement hurdles requiring urgent intervention."
    }
}

# Intervention Statuses
INTERVENTION_STATUSES = [
    "Not Started",
    "Under Review",
    "In Progress",
    "Follow-up Scheduled",
    "Completed"
]

STATUS_COLORS = {
    "Not Started": "#94A3B8",
    "Under Review": "#38BDF8",
    "In Progress": "#F59E0B",
    "Follow-up Scheduled": "#A855F7",
    "Completed": "#10B981"
}

# Feature Categories for Grouping and UI Display
FEATURE_CATEGORIES = {
    "Academic Performance (2nd Semester)": [
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (without evaluations)"
    ],
    "Academic Performance (1st Semester)": [
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (credited)",
        "Curricular units 1st sem (without evaluations)"
    ],
    "Financial & Administrative": [
        "Tuition fees up to date",
        "Debtor",
        "Scholarship holder"
    ],
    "Demographics & Background": [
        "Age at enrollment",
        "Gender",
        "Marital status",
        "Displaced",
        "Educational special needs",
        "International",
        "Nacionality"
    ],
    "Admission & Program": [
        "Course",
        "Application mode",
        "Application order",
        "Daytime/evening attendance",
        "Previous qualification"
    ],
    "Family Background": [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation"
    ],
    "Macroeconomic Context": [
        "Unemployment rate",
        "Inflation rate",
        "GDP"
    ]
}

# Human Friendly Course Mapping (from Portuguese Higher Education standard code list)
COURSE_MAPPING = {
    33: "Biofuel Production Tech",
    171: "Animation & Multimedia Design",
    8014: "Social Service (Evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising & Marketing",
    9773: "Journalism & Communication",
    9853: "Basic Education",
    9991: "Management (Evening)"
}

def get_risk_level_from_score(score: float) -> Dict[str, Any]:
    """
    Given a dropout probability (0.0 to 1.0 or 0 to 100), return the corresponding risk metadata.
    """
    if score > 1.0:
        score = score / 100.0
    
    if score < RISK_THRESHOLDS["LOW_MAX"]:
        return {
            "key": "LOW",
            **RISK_LEVELS["LOW"],
            "score_pct": score * 100.0
        }
    elif score < RISK_THRESHOLDS["MEDIUM_MAX"]:
        return {
            "key": "MEDIUM",
            **RISK_LEVELS["MEDIUM"],
            "score_pct": score * 100.0
        }
    else:
        return {
            "key": "HIGH",
            **RISK_LEVELS["HIGH"],
            "score_pct": score * 100.0
        }
