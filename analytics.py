import numpy as np
import pandas as pd
from load_data import load_data, filter_dataframe

def get_placement_analytics(df: pd.DataFrame) -> dict:
    """Computes placement distribution stats across demographics and academic factors."""
    total_students = len(df)
    placed_df = df[df["PlacementStatus"] == 1]
    placed_count = len(placed_df)
    overall_rate = round((placed_count / total_students) * 100, 2) if total_students > 0 else 0.0

    def calc_group_rates(col_name: str):
        if col_name not in df.columns:
            return []
        grouped = df.groupby(col_name)["PlacementStatus"].agg(
            total="count",
            placed="sum"
        ).reset_index()
        grouped["rate"] = (grouped["placed"] / grouped["total"] * 100).round(2)
        grouped["col_val"] = grouped[col_name].astype(str)
        grouped.sort_values(by="rate", ascending=False, inplace=True)
        return grouped[["col_val", "total", "placed", "rate"]].to_dict(orient="records")

    by_gender = calc_group_rates("Gender")
    by_city = calc_group_rates("City")
    by_stream = calc_group_rates("Stream")
    by_tier = calc_group_rates("CollegeTier")
    by_specialisation = calc_group_rates("Specialisation")
    by_cgpa_tier = calc_group_rates("CGPA_Tier")
    by_backlogs = calc_group_rates("HistoryOfBacklogs")
    by_hostel = calc_group_rates("Hostel")

    # Placement Factor Association Analysis (Point Biserial Correlation with PlacementStatus)
    numeric_cols = list(df.select_dtypes(include='number').columns)
    associations = []
    
    if "PlacementStatus" in numeric_cols:
        for col in numeric_cols:
            if col in ["PlacementStatus", "StudentID", "Salary Package", "IsAnomaly"]:
                continue
            series = df[[col, "PlacementStatus"]].dropna()
            if len(series) > 0:
                corr = float(series[col].corr(series["PlacementStatus"]))
                if not np.isnan(corr):
                    associations.append({
                        "feature": col,
                        "correlation": round(corr, 3),
                        "abs_corr": abs(round(corr, 3))
                    })
        associations.sort(key=lambda x: x["abs_corr"], reverse=True)

    return {
        "overall_placement_rate": overall_rate,
        "total_students": total_students,
        "placed_count": placed_count,
        "unplaced_count": total_students - placed_count,
        "by_gender": by_gender,
        "by_city": by_city[:10],
        "by_stream": by_stream,
        "by_tier": by_tier,
        "by_specialisation": by_specialisation[:10],
        "by_cgpa_tier": by_cgpa_tier,
        "by_backlogs": by_backlogs,
        "by_hostel": by_hostel,
        "associations": associations
    }

def get_salary_analytics(df: pd.DataFrame) -> dict:
    """Computes salary statistics and distributions exclusively for placed students."""
    placed_df = df[(df["PlacementStatus"] == 1) & (df["Salary Package"] > 0)].copy()
    
    if len(placed_df) == 0:
        return {"has_data": False}

    salaries = placed_df["Salary Package"]

    stats_summary = {
        "avg_salary": round(float(salaries.mean()), 2),
        "median_salary": round(float(salaries.median()), 2),
        "min_salary": round(float(salaries.min()), 2),
        "max_salary": round(float(salaries.max()), 2),
        "std_salary": round(float(salaries.std()), 2),
        "total_placed_with_salary": len(placed_df)
    }

    # Salary by Stream
    salary_by_stream = placed_df.groupby("Stream")["Salary Package"].agg(
        avg="mean", median="median", min="min", max="max"
    ).round(2).reset_index().to_dict(orient="records")

    # Salary by College Tier
    salary_by_tier = placed_df.groupby("CollegeTier")["Salary Package"].agg(
        avg="mean", median="median", min="min", max="max"
    ).round(2).reset_index().to_dict(orient="records")

    # Subsample scatter data points for fast Plotly rendering
    if len(placed_df) > 2000:
        scatter_sample = placed_df.sample(2000, random_state=42)
    else:
        scatter_sample = placed_df

    return {
        "has_data": True,
        "stats": stats_summary,
        "salary_values": scatter_sample["Salary Package"].tolist(),
        "by_stream": salary_by_stream,
        "by_tier": salary_by_tier,
        "salary_vs_cgpa": {
            "cgpa": scatter_sample["CGPA"].tolist(),
            "salary": scatter_sample["Salary Package"].tolist()
        },
        "salary_vs_coding": {
            "coding": scatter_sample["CodingTestScore"].dropna().tolist(),
            "salary": scatter_sample.loc[scatter_sample["CodingTestScore"].notna(), "Salary Package"].tolist()
        },
        "salary_vs_interview": {
            "interview": scatter_sample["MockInterviewScore"].dropna().tolist(),
            "salary": scatter_sample.loc[scatter_sample["MockInterviewScore"].notna(), "Salary Package"].tolist()
        }
    }

def get_anomaly_analytics(df: pd.DataFrame) -> dict:
    """Calculates dataset anomaly statistics and returns anomalous student records."""
    if "IsAnomaly" not in df.columns:
        return {"has_data": False}

    total_records = len(df)
    anomalous_df = df[df["IsAnomaly"] == 1].copy()
    normal_df = df[df["IsAnomaly"] == 0].copy()
    
    anomalous_cnt = len(anomalous_df)
    normal_cnt = len(normal_df)
    anomaly_pct = round((anomalous_cnt / total_records) * 100, 2) if total_records > 0 else 0.0

    # Anomalies by Stream
    by_stream = df.groupby("Stream")["IsAnomaly"].agg(
        total="count", anomalies="sum"
    ).reset_index()
    by_stream["rate"] = (by_stream["anomalies"] / by_stream["total"] * 100).round(2)
    by_stream_list = by_stream.to_dict(orient="records")

    # Anomalies by College Tier
    by_tier = df.groupby("CollegeTier")["IsAnomaly"].agg(
        total="count", anomalies="sum"
    ).reset_index()
    by_tier["rate"] = (by_tier["anomalies"] / by_tier["total"] * 100).round(2)
    by_tier_list = by_tier.to_dict(orient="records")

    # Anomalies by Placement Status
    by_status = df.groupby("PlacementStatus")["IsAnomaly"].agg(
        total="count", anomalies="sum"
    ).reset_index()
    by_status["status_label"] = by_status["PlacementStatus"].map({1: "Placed", 0: "Unplaced"})
    by_status_list = by_status.to_dict(orient="records")

    # Top Anomaly Table Preview
    anomaly_preview_cols = [
        "StudentID", "Gender", "City", "CollegeTier", "Stream", "CGPA",
        "AttendancePercent", "CodingTestScore", "MockInterviewScore",
        "PlacementStatus", "Salary Package", "IsAnomaly"
    ]
    avail_cols = [c for c in anomaly_preview_cols if c in anomalous_df.columns]
    
    anomaly_table = anomalous_df[avail_cols].head(50).fillna("-").to_dict(orient="records")

    return {
        "has_data": True,
        "total_records": total_records,
        "normal_records": normal_cnt,
        "anomalous_records": anomalous_cnt,
        "anomaly_percentage": anomaly_pct,
        "by_stream": by_stream_list,
        "by_tier": by_tier_list,
        "by_status": by_status_list,
        "anomaly_table": anomaly_table
    }

def get_paginated_students(
    df: pd.DataFrame,
    filters: dict = None,
    search: str = None,
    page: int = 1,
    per_page: int = 25
) -> dict:
    """Returns filtered, searched, and paginated student records."""
    filtered_df = filter_dataframe(df, filters)

    # Search filter across text & numeric columns
    if search:
        s = str(search).strip().lower()
        mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        search_cols = ["StudentID", "Gender", "City", "Stream", "CollegeTier", "Specialisation"]
        for col in search_cols:
            if col in filtered_df.columns:
                mask = mask | filtered_df[col].astype(str).str.lower().str.contains(s)
        filtered_df = filtered_df[mask]

    total_records = len(filtered_df)
    total_pages = max(1, int(np.ceil(total_records / per_page)))
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    records = filtered_df.iloc[start_idx:end_idx].fillna("-").to_dict(orient="records")

    return {
        "total_records": total_records,
        "total_pages": total_pages,
        "current_page": page,
        "per_page": per_page,
        "records": records
    }
