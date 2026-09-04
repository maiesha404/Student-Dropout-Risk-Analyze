"""
Intervention Engine for Student Dropout Risk Mitigation.
Decoupled rule-based recommendation system matching student diagnostic signals
and risk drivers with targeted institutional support actions.
Includes lightweight local persistence for intervention statuses.
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.config import INTERVENTIONS_STORE_PATH, INTERVENTION_STATUSES

class InterventionEngine:
    """
    Decoupled Rule-Based Intervention Engine.
    Maps diagnostic signals from student features and risk levels to actionable support plans.
    """
    
    @staticmethod
    def generate_recommendations(
        student_data: Dict[str, Any],
        risk_level: str,
        risk_score_pct: float,
        top_risk_factors: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable recommendations based on student's feature indicators and risk profile.
        """
        interventions = []
        
        # 1. Academic Performance Rules (2nd Semester)
        u2_approved = float(student_data.get("Curricular units 2nd sem (approved)", 5.0))
        u2_enrolled = float(student_data.get("Curricular units 2nd sem (enrolled)", 5.0))
        u2_grade = float(student_data.get("Curricular units 2nd sem (grade)", 12.0))
        u2_no_eval = float(student_data.get("Curricular units 2nd sem (without evaluations)", 0.0))
        
        if u2_approved < 3.0 or u2_grade < 10.0:
            interventions.append({
                "id": "ACAD-01",
                "category": "Academic Support",
                "title": "Intensive Peer Tutoring & Subject Review",
                "priority": "High" if risk_level == "High Risk" else "Medium",
                "driver": f"Low 2nd Semester Approved Units ({int(u2_approved)}) and Average Grade ({u2_grade:.1f}/20)",
                "action": "Assign dedicated departmental peer tutor for difficult subjects and schedule bi-weekly progress checkpoints.",
                "owner": "Department Academic Advisory Committee"
            })
        elif u2_approved < 5.0:
            interventions.append({
                "id": "ACAD-02",
                "category": "Academic Support",
                "title": "Study Skills & Exam Preparation Workshop",
                "priority": "Medium",
                "driver": f"Partial Semester Completion ({int(u2_approved)}/{int(u2_enrolled)} units approved)",
                "action": "Enroll in time management and exam preparation clinic ahead of next assessment cycle.",
                "owner": "Faculty Learning Center"
            })
            
        # 2. Academic Performance Rules (1st Semester)
        u1_approved = float(student_data.get("Curricular units 1st sem (approved)", 5.0))
        u1_grade = float(student_data.get("Curricular units 1st sem (grade)", 12.0))
        u1_no_eval = float(student_data.get("Curricular units 1st sem (without evaluations)", 0.0))
        
        if u1_approved < 3.0 and u2_approved < 4.0:
            interventions.append({
                "id": "ACAD-03",
                "category": "Academic Planning",
                "title": "Curricular Load Restructuring & Course Pacing",
                "priority": "High",
                "driver": "Chronic multi-semester credit deficit across 1st and 2nd semesters",
                "action": "Meet with Academic Dean to adjust credit registration load and balance prerequisite course pacing.",
                "owner": "Academic Guidance Counselor"
            })

        # 3. Financial Obstacle Rules
        tuition_up_to_date = int(student_data.get("Tuition fees up to date", 1))
        debtor = int(student_data.get("Debtor", 0))
        scholarship = int(student_data.get("Scholarship holder", 0))
        
        if tuition_up_to_date == 0:
            interventions.append({
                "id": "FIN-01",
                "category": "Financial Aid",
                "title": "Emergency Tuition Restructuring & Hardship Grant Review",
                "priority": "High",
                "driver": "Tuition fees currently in arrears / overdue",
                "action": "Fast-track consultation with Student Financial Services for deferred installment plans or emergency hardship relief.",
                "owner": "Student Financial Services"
            })
            
        if debtor == 1:
            interventions.append({
                "id": "FIN-02",
                "category": "Financial Aid",
                "title": "Institutional Debt Counseling & Work-Study Options",
                "priority": "High" if tuition_up_to_date == 0 else "Medium",
                "driver": "Active debtor status flag recorded in registry",
                "action": "Provide confidential debt management session and evaluate on-campus work-study opportunities.",
                "owner": "Campus Welfare & Bursar Office"
            })
            
        if scholarship == 0 and (risk_level == "High Risk" or risk_level == "Medium Risk") and tuition_up_to_date == 0:
            interventions.append({
                "id": "FIN-03",
                "category": "Financial Aid",
                "title": "Need-Based Scholarship Re-evaluation",
                "priority": "Medium",
                "driver": "Non-scholarship student experiencing economic pressure",
                "action": "Review eligibility criteria for regional and institutional need-based stipends.",
                "owner": "Scholarships Office"
            })

        # 4. Engagement & Attendance Rules
        if (u1_no_eval > 0 or u2_no_eval > 0):
            interventions.append({
                "id": "ENG-01",
                "category": "Student Engagement",
                "title": "Proactive Assessment Attendance Check-in",
                "priority": "Medium",
                "driver": f"Uncompleted evaluations flagged ({int(u1_no_eval + u2_no_eval)} total unit assessments missed)",
                "action": "Course coordinator to contact student regarding personal or scheduling obstacles preventing exam attendance.",
                "owner": "Student Success Mentor"
            })
            
        attendance_mode = int(student_data.get("Daytime/evening attendance", 1))
        if attendance_mode == 0: # Evening
            interventions.append({
                "id": "ENG-02",
                "category": "Student Engagement",
                "title": "Evening Student Flexible Support & Digital Office Hours",
                "priority": "Low",
                "driver": "Evening course attendance schedule",
                "action": "Ensure access to recorded lectures, asynchronous tutoring, and weekend academic consultation.",
                "owner": "Evening Program Coordinator"
            })

        # 5. Demographics & Special Needs Rules
        displaced = int(student_data.get("Displaced", 0))
        if displaced == 1 and risk_level != "Low Risk":
            interventions.append({
                "id": "WEL-01",
                "category": "Student Welfare",
                "title": "Relocation & Campus Community Integration Check-in",
                "priority": "Low",
                "driver": "Displaced student residing away from hometown",
                "action": "Connect with campus student housing and community peer network to reduce isolation.",
                "owner": "Campus Community Life"
            })

        special_needs = int(student_data.get("Educational special needs", 0))
        if special_needs == 1:
            interventions.append({
                "id": "WEL-02",
                "category": "Accessibility Support",
                "title": "Special Needs Learning Accommodation Review",
                "priority": "High" if risk_level != "Low Risk" else "Medium",
                "driver": "Special educational needs accommodations on file",
                "action": "Conduct accessibility audit of current coursework and exam accommodation provisions.",
                "owner": "Disability & Inclusion Center"
            })

        age = float(student_data.get("Age at enrollment", 20.0))
        if age >= 25.0 and risk_level != "Low Risk":
            interventions.append({
                "id": "WEL-03",
                "category": "Advising Support",
                "title": "Mature & Non-Traditional Student Mentorship",
                "priority": "Low",
                "driver": f"Mature student (Age {int(age)} at enrollment)",
                "action": "Invite to mature student peer network and provide career/workplace balance guidance.",
                "owner": "Adult Education Services"
            })
            
        # If no specific risk rule fired but student is at medium/high risk
        if not interventions and risk_level != "Low Risk":
            interventions.append({
                "id": "GEN-01",
                "category": "General Advising",
                "title": "Comprehensive 360° Academic Advisor Check-in",
                "priority": "Medium",
                "driver": "Elevated multivariate risk score requiring holistic review",
                "action": "Schedule a 30-minute 1-on-1 holistic advising session to explore academic and personal wellbeing.",
                "owner": "Assigned Faculty Advisor"
            })
        elif not interventions and risk_level == "Low Risk":
            interventions.append({
                "id": "GEN-00",
                "category": "General Advising",
                "title": "Routine Term Milestone Review",
                "priority": "Low",
                "driver": "Healthy academic trajectory",
                "action": "Standard end-of-semester degree progression verification.",
                "owner": "Academic Department"
            })

        return interventions

class InterventionStore:
    """
    Lightweight JSON-backed store for tracking intervention workflow statuses per student.
    Persists across sessions.
    """
    def __init__(self, filepath: str = INTERVENTIONS_STORE_PATH):
        self.filepath = filepath
        self._ensure_file()
        
    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({}, f)
                
    def get_student_record(self, student_id: str) -> Dict[str, Any]:
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            return data.get(str(student_id), {
                "status": "Not Started",
                "notes": "",
                "last_updated": None,
                "assigned_to": "Unassigned"
            })
        except Exception:
            return {
                "status": "Not Started",
                "notes": "",
                "last_updated": None,
                "assigned_to": "Unassigned"
            }
            
    def update_student_record(
        self,
        student_id: str,
        status: str,
        notes: str = "",
        assigned_to: str = "Faculty Advisor"
    ) -> bool:
        if status not in INTERVENTION_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {INTERVENTION_STATUSES}")
            
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
            
        data[str(student_id)] = {
            "status": status,
            "notes": notes,
            "assigned_to": assigned_to,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)
            
        return True

    def get_all_records(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.filepath, "r") as f:
                return json.load(f)
        except Exception:
            return {}
