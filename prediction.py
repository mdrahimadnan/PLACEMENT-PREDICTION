import numpy as np
import pandas as pd
from models import get_saved_model, train_and_evaluate_model
from load_data import load_data

def predict_student_placement(data: dict) -> dict:
    """Runs placement probability prediction using trained Scikit-learn model pipeline."""
    pipeline_obj = get_saved_model()
    
    # If model is not trained yet, train default Random Forest on dataset automatically
    if pipeline_obj is None:
        df = load_data()
        train_and_evaluate_model(df, model_name="Random Forest")
        pipeline_obj = get_saved_model()

    if pipeline_obj is None:
        raise RuntimeError("Failed to load or train machine learning model pipeline.")

    model = pipeline_obj["model"]
    scaler = pipeline_obj["scaler"]
    feature_names = pipeline_obj["feature_names"]
    model_name = pipeline_obj.get("model_name", "Random Forest")

    # Extract inputs
    student_name = data.get("student_name", "Student")
    gender = data.get("gender", "Male")
    city = data.get("city", "Mumbai")
    college_tier = data.get("college_tier", "Tier2")
    stream = data.get("stream", "Computer Science")
    specialisation = data.get("specialisation", "Software Engineering")
    hostel = data.get("hostel", "No")
    backlogs = data.get("backlogs", "No")
    
    cgpa = float(data.get("cgpa", 7.5))
    attendance = float(data.get("attendance", 80.0))
    internships = int(data.get("internships", 1))
    projects = int(data.get("projects", 2))
    workshops = float(data.get("workshops", 1))
    certifications = int(data.get("certifications", 1))
    publications = int(data.get("publications", 0))
    aptitude_score = float(data.get("aptitude_score", 70.0))
    soft_skills = float(data.get("soft_skills", 3.5))
    coding_score = float(data.get("coding_score", 75.0))
    mock_interview = float(data.get("mock_interview", 70.0))
    extra_curricular = int(data.get("extra_curricular", 1))

    # Helper for CGPA Tier
    if cgpa >= 8.5:
        cgpa_tier = "High"
    elif cgpa >= 7.0:
        cgpa_tier = "Medium"
    else:
        cgpa_tier = "Low"

    # Construct single record dataframe matching feature space
    input_dict = {
        "Gender": gender,
        "City": city,
        "CollegeTier": college_tier,
        "Stream": stream,
        "Specialisation": specialisation,
        "Hostel": hostel,
        "HistoryOfBacklogs": backlogs,
        "SGPA_Sem1": cgpa, "SGPA_Sem2": cgpa, "SGPA_Sem3": cgpa, "SGPA_Sem4": cgpa,
        "SGPA_Sem5": cgpa, "SGPA_Sem6": cgpa, "SGPA_Sem7": cgpa, "SGPA_Sem8": cgpa,
        "CGPA": cgpa,
        "AttendancePercent": attendance,
        "Internships": internships,
        "Projects": projects,
        "Workshops": workshops,
        "Certifications": certifications,
        "Publications": publications,
        "AptitudeTestScore": aptitude_score,
        "SoftSkillsRating": soft_skills,
        "CodingTestScore": coding_score,
        "MockInterviewScore": mock_interview,
        "ExtraCurricular": extra_curricular,
        "CGPA_Tier": cgpa_tier
    }

    raw_row_df = pd.DataFrame([input_dict])

    # Categorical One-Hot Encoding matching training columns
    cat_cols = list(raw_row_df.select_dtypes(exclude='number').columns)
    row_encoded = pd.get_dummies(raw_row_df, columns=cat_cols)

    # Reindex to match trained feature_names (filling missing dummy columns with 0)
    aligned_df = row_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale numeric features
    scaled_row = scaler.transform(aligned_df)

    # Run ML Model Inference
    if hasattr(model, "predict_proba"):
        proba_val = float(model.predict_proba(scaled_row)[0][1])
    else:
        pred_val = int(model.predict(scaled_row)[0])
        proba_val = 1.0 if pred_val == 1 else 0.0

    prob_percentage = round(proba_val * 100, 1)
    is_placed = proba_val >= 0.50
    status = "LIKELY PLACED 🎉" if is_placed else "LIKELY NOT PLACED ⚠️"

    # Estimate Salary Package
    if is_placed:
        base_salary = 4.0 + (cgpa / 10.0) * 6.0 + (coding_score / 100.0) * 5.0 + (internships * 0.8)
        if college_tier == "Tier1":
            base_salary += 3.5
        elif college_tier == "Tier2":
            base_salary += 1.5
        est_package = round(min(max(base_salary, 3.5), 45.0), 2)
    else:
        est_package = 0.0

    # Personalized Recommendations
    recommendations = []
    if coding_score < 70:
        recommendations.append("Increase Coding Test Score to 75+ to boost tech interview selection rates.")
    if cgpa < 7.5:
        recommendations.append("Improve CGPA above 7.5 to clear campus recruitment eligibility cutoffs.")
    if internships < 1:
        recommendations.append("Complete at least 1 industry internship to build practical experience.")
    if backlogs.strip().lower() in ["yes", "1", "true"]:
        recommendations.append("Clear active backlogs prior to campus placement drives.")
    if aptitude_score < 65:
        recommendations.append("Practice aptitude & problem-solving tests to clear initial screening rounds.")
    if mock_interview < 65:
        recommendations.append("Participate in mock HR and technical interviews to refine soft skills.")

    if not recommendations:
        recommendations.append("Outstanding candidate profile! Strong potential for top tier recruitment drives.")

    return {
        "status": "success",
        "student_name": student_name,
        "placement_status": status,
        "is_placed": is_placed,
        "probability": prob_percentage,
        "estimated_package": est_package,
        "model_used": model_name,
        "recommendations": recommendations,
        "stream": stream,
        "tier": college_tier
    }
