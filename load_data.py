import os
import pandas as pd
import numpy as np
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "placement_predict_50k Dataset.csv")

_cached_df = None

def load_data(path: str = DATA_PATH, force_reload: bool = False) -> pd.DataFrame:
    """Loads the student placement dataset using a relative project path with in-memory caching."""
    global _cached_df
    if _cached_df is not None and not force_reload:
        return _cached_df

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    _cached_df = df
    return df

def get_duplicate_count(df: pd.DataFrame = None) -> int:
    """Returns the count of duplicate rows in the dataset."""
    if df is None:
        df = load_data()
    return int(df.duplicated().sum())

def filter_dataframe(df: pd.DataFrame, filters: dict = None) -> pd.DataFrame:
    """Filters the dataframe based on user selected dropdown filters."""
    if df is None:
        df = load_data()
    
    if not filters:
        return df

    filtered_df = df.copy()
    
    if filters.get("gender") and filters.get("gender") != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == filters["gender"]]
        
    if filters.get("city") and filters.get("city") != "All":
        filtered_df = filtered_df[filtered_df["City"] == filters["city"]]
        
    if filters.get("stream") and filters.get("stream") != "All":
        filtered_df = filtered_df[filtered_df["Stream"] == filters["stream"]]
        
    if filters.get("tier") and filters.get("tier") != "All":
        filtered_df = filtered_df[filtered_df["CollegeTier"] == filters["tier"]]

    if filters.get("specialisation") and filters.get("specialisation") != "All":
        filtered_df = filtered_df[filtered_df["Specialisation"] == filters["specialisation"]]

    if filters.get("placement_status") and filters.get("placement_status") != "All":
        try:
            status_val = int(filters["placement_status"])
            filtered_df = filtered_df[filtered_df["PlacementStatus"] == status_val]
        except (ValueError, TypeError):
            pass

    return filtered_df

def get_data_summary(df: pd.DataFrame = None) -> dict:
    """Calculates comprehensive dataset statistics dynamically from actual data."""
    if df is None:
        df = load_data()

    total_students = int(len(df))
    if total_students == 0:
        return {
            "n_rows": 0, "n_cols": 0, "placed_count": 0, "unplaced_count": 0,
            "placement_rate": 0.0, "avg_cgpa": 0.0, "avg_attendance": 0.0,
            "avg_salary": 0.0, "total_anomalies": 0, "anomaly_rate": 0.0,
            "columns": [], "numeric_cols": [], "categorical_cols": [],
            "dtypes": {}, "missing_counts": {}, "missing_percentages": {},
            "duplicate_count": 0, "preview": []
        }

    placed_mask = df["PlacementStatus"] == 1 if "PlacementStatus" in df.columns else pd.Series([False]*len(df))
    placed_count = int(placed_mask.sum())
    unplaced_count = int((~placed_mask).sum())
    placement_rate = round((placed_count / total_students) * 100, 1)

    avg_cgpa = round(float(df["CGPA"].dropna().mean()), 2) if "CGPA" in df.columns else 0.0
    avg_attendance = round(float(df["AttendancePercent"].dropna().mean()), 2) if "AttendancePercent" in df.columns else 0.0

    placed_salaries = df[df["PlacementStatus"] == 1]["Salary Package"] if "PlacementStatus" in df.columns else df["Salary Package"]
    placed_salaries_valid = placed_salaries[placed_salaries > 0].dropna()
    avg_salary = round(float(placed_salaries_valid.mean()), 2) if len(placed_salaries_valid) > 0 else 0.0

    total_anomalies = int((df["IsAnomaly"] == 1).sum()) if "IsAnomaly" in df.columns else 0
    anomaly_rate = round((total_anomalies / total_students) * 100, 1)

    numeric_cols = list(df.select_dtypes(include='number').columns)
    categorical_cols = list(df.select_dtypes(exclude='number').columns)

    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_percentages = {
        col: round(float((count / total_students) * 100), 2)
        for col, count in missing_counts.items() if count > 0
    }

    unique_options = {
        "gender": sorted([str(x) for x in df["Gender"].dropna().unique()]) if "Gender" in df.columns else [],
        "city": sorted([str(x) for x in df["City"].dropna().unique()]) if "City" in df.columns else [],
        "tier": sorted([str(x) for x in df["CollegeTier"].dropna().unique()]) if "CollegeTier" in df.columns else [],
        "stream": sorted([str(x) for x in df["Stream"].dropna().unique()]) if "Stream" in df.columns else [],
        "specialisation": sorted([str(x) for x in df["Specialisation"].dropna().unique()]) if "Specialisation" in df.columns else [],
    }

    summary = {
        "n_rows": total_students,
        "n_cols": int(df.shape[1]),
        "placed_count": placed_count,
        "unplaced_count": unplaced_count,
        "placement_rate": placement_rate,
        "avg_cgpa": avg_cgpa,
        "avg_attendance": avg_attendance,
        "avg_salary": avg_salary,
        "total_anomalies": total_anomalies,
        "anomaly_rate": anomaly_rate,
        "columns": list(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "duplicate_count": int(df.duplicated().sum()),
        "unique_options": unique_options,
        "preview": df.head(25).fillna("-").to_dict(orient="records")
    }

    return summary

if __name__ == "__main__":
    df = load_data()
    print("Dataset Loaded Successfully. Shape:", df.shape)
    print("Summary Stats:", get_data_summary(df))