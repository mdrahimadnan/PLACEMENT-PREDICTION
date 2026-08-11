import numpy as np
import pandas as pd

def get_overview_info(df: pd.DataFrame) -> dict:
    """Generates dataset overview details including column metadata."""
    total_rows = len(df)
    cols_meta = []
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        missing_cnt = int(df[col].isna().sum())
        missing_pct = round((missing_cnt / total_rows) * 100, 2)
        unique_cnt = int(df[col].nunique())
        sample_vals = ", ".join([str(x) for x in df[col].dropna().unique()[:3]])
        
        cols_meta.append({
            "column": col,
            "data_type": dtype_str,
            "missing_count": missing_cnt,
            "missing_percentage": missing_pct,
            "unique_count": unique_cnt,
            "sample_values": sample_vals
        })

    target_dist = {}
    if "PlacementStatus" in df.columns:
        vc = df["PlacementStatus"].value_counts().to_dict()
        target_dist = {str(k): int(v) for k, v in vc.items()}

    return {
        "n_rows": total_rows,
        "n_cols": len(df.columns),
        "numeric_count": len(df.select_dtypes(include='number').columns),
        "categorical_count": len(df.select_dtypes(exclude='number').columns),
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "unique_students": int(df["StudentID"].nunique()) if "StudentID" in df.columns else total_rows,
        "target_distribution": target_dist,
        "columns_metadata": cols_meta
    }

def get_missing_analysis(df: pd.DataFrame) -> dict:
    """Computes missing value stats and horizontal bar chart dataset."""
    total_rows = len(df)
    missing_data = []
    
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        if cnt > 0:
            pct = round((cnt / total_rows) * 100, 2)
            missing_data.append({
                "feature": col,
                "missing_count": cnt,
                "missing_percentage": pct
            })
            
    missing_data.sort(key=lambda x: x["missing_count"], reverse=True)
    
    return {
        "total_missing_cells": int(df.isna().sum().sum()),
        "columns_with_missing": len(missing_data),
        "missing_summary": missing_data
    }

def get_univariate_analysis(df: pd.DataFrame, feature: str, viz_type: str = "histogram") -> dict:
    """Computes distribution statistics and plot coordinates for a continuous feature."""
    if feature not in df.columns or not np.issubdtype(df[feature].dtype, np.number):
        return {"error": f"Invalid numerical feature: {feature}"}

    series = df[feature].dropna()
    if len(series) == 0:
        return {"error": f"No non-null data for feature: {feature}"}

    # Descriptive Statistics
    q1 = float(np.percentile(series, 25))
    q3 = float(np.percentile(series, 75))
    iqr = float(q3 - q1)
    
    mode_val = float(series.mode()[0]) if not series.mode().empty else float(series.median())
    
    stats_summary = {
        "feature": feature,
        "count": int(len(series)),
        "mean": round(float(series.mean()), 3),
        "median": round(float(series.median()), 3),
        "mode": round(mode_val, 3),
        "std": round(float(series.std()), 3),
        "variance": round(float(series.var()), 3),
        "min": round(float(series.min()), 3),
        "max": round(float(series.max()), 3),
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "skewness": round(float(series.skew()), 3),
        "kurtosis": round(float(series.kurtosis()), 3)
    }

    # Calculate KDE or Histogram Bins for Plotly
    values = series.tolist()
    
    # Subsample if values count > 5000 for fast chart rendering
    if len(values) > 5000:
        sampled_values = list(np.random.choice(values, 5000, replace=False))
    else:
        sampled_values = values

    return {
        "stats": stats_summary,
        "viz_type": viz_type,
        "values": sampled_values
    }

def get_categorical_analysis(df: pd.DataFrame, feature: str) -> dict:
    """Computes frequency distributions for categorical features."""
    if feature not in df.columns:
        return {"error": f"Column {feature} not found in dataset."}

    vc = df[feature].value_counts(dropna=False)
    total = len(df)
    
    categories = []
    counts = []
    percentages = []

    for cat, count in vc.items():
        cat_str = str(cat) if pd.notna(cat) else "Missing"
        pct = round((count / total) * 100, 2)
        categories.append(cat_str)
        counts.append(int(count))
        percentages.append(pct)

    return {
        "feature": feature,
        "categories": categories,
        "counts": counts,
        "percentages": percentages,
        "total": total
    }

def get_bivariate_analysis(df: pd.DataFrame, x_feature: str, y_feature: str, group_by: str = None, viz_type: str = "scatter") -> dict:
    """Generates dataset for bivariate relationship visualizer."""
    cols_to_select = [x_feature, y_feature]
    if group_by and group_by != "None" and group_by in df.columns:
        cols_to_select.append(group_by)
        
    sub_df = df[cols_to_select].dropna()
    
    # Subsample if > 3000 rows for smooth Plotly performance
    if len(sub_df) > 3000:
        sample_df = sub_df.sample(3000, random_state=42)
    else:
        sample_df = sub_df

    data = {
        "x": sample_df[x_feature].tolist(),
        "y": sample_df[y_feature].tolist(),
        "x_name": x_feature,
        "y_name": y_feature,
        "viz_type": viz_type
    }

    if group_by and group_by != "None" and group_by in df.columns:
        data["group_by"] = group_by
        data["group_values"] = sample_df[group_by].astype(str).tolist()

    return data

def get_correlation_analysis(df: pd.DataFrame) -> dict:
    """Computes numerical feature correlation matrix and strongest association pairs."""
    numeric_df = df.select_dtypes(include='number').drop(columns=['StudentID'], errors='ignore')
    
    corr_matrix = numeric_df.corr().round(3)
    
    cols = list(corr_matrix.columns)
    z_matrix = corr_matrix.values.tolist()

    # Extract top positive & negative correlations (excluding self correlations)
    corr_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            col1 = cols[i]
            col2 = cols[j]
            val = float(corr_matrix.iloc[i, j])
            if not np.isnan(val):
                corr_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(val, 3),
                    "abs_corr": abs(round(val, 3))
                })

    corr_pairs.sort(key=lambda x: x["abs_corr"], reverse=True)

    top_positive = [p for p in corr_pairs if p["correlation"] > 0][:5]
    top_negative = [p for p in corr_pairs if p["correlation"] < 0][:5]

    return {
        "columns": cols,
        "z_matrix": z_matrix,
        "top_positive": top_positive,
        "top_negative": top_negative
    }

def get_outlier_analysis(df: pd.DataFrame, feature: str) -> dict:
    """Calculates IQR outlier bounds and counts for numerical features."""
    if feature not in df.columns or not np.issubdtype(df[feature].dtype, np.number):
        return {"error": f"Invalid numerical feature: {feature}"}

    series = df[feature].dropna()
    q1 = float(np.percentile(series, 25))
    q3 = float(np.percentile(series, 75))
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_count = int(len(outliers))
    outlier_pct = round((outlier_count / len(series)) * 100, 2)

    # Subsample values for boxplot rendering
    vals = series.tolist()
    if len(vals) > 3000:
        sample_vals = list(np.random.choice(vals, 3000, replace=False))
    else:
        sample_vals = vals

    return {
        "feature": feature,
        "total_count": len(series),
        "q1": round(q1, 3),
        "q3": round(q3, 3),
        "iqr": round(iqr, 3),
        "lower_bound": round(lower_bound, 3),
        "upper_bound": round(upper_bound, 3),
        "outlier_count": outlier_count,
        "outlier_percentage": outlier_pct,
        "sample_values": sample_vals
    }

def get_auto_insights(df: pd.DataFrame) -> list:
    """Generates dynamically calculated analytical insights from actual data."""
    insights = []
    
    if "PlacementStatus" in df.columns and "CGPA" in df.columns:
        placed_cgpa = df[df["PlacementStatus"] == 1]["CGPA"].mean()
        unplaced_cgpa = df[df["PlacementStatus"] == 0]["CGPA"].mean()
        diff = round(placed_cgpa - unplaced_cgpa, 2)
        insights.append({
            "title": "Academic Performance & Placement",
            "text": f"Placed students have an average CGPA of {placed_cgpa:.2f}, compared to {unplaced_cgpa:.2f} for unplaced students (a difference of {diff} grade points)."
        })

    if "PlacementStatus" in df.columns and "Stream" in df.columns:
        stream_rates = df.groupby("Stream")["PlacementStatus"].mean() * 100
        best_stream = stream_rates.idxmax()
        best_rate = stream_rates.max()
        lowest_stream = stream_rates.idxmin()
        lowest_rate = stream_rates.min()
        insights.append({
            "title": "Stream Placement Comparison",
            "text": f"The highest placement rate was observed in {best_stream} ({best_rate:.1f}%), while {lowest_stream} recorded a placement rate of {lowest_rate:.1f}%."
        })

    if "PlacementStatus" in df.columns and "CodingTestScore" in df.columns:
        high_coding = df[df["CodingTestScore"] >= 75]["PlacementStatus"].mean() * 100
        low_coding = df[df["CodingTestScore"] < 50]["PlacementStatus"].mean() * 100
        insights.append({
            "title": "Coding Benchmark Threshold",
            "text": f"Students scoring 75+ on coding tests show a {high_coding:.1f}% placement rate, compared to {low_coding:.1f}% for students scoring below 50."
        })

    if "PlacementStatus" in df.columns and "Internships" in df.columns:
        intern_rates = df.groupby("Internships")["PlacementStatus"].mean() * 100
        zero_intern = intern_rates.get(0, 0.0)
        multi_intern = df[df["Internships"] >= 2]["PlacementStatus"].mean() * 100
        insights.append({
            "title": "Internship Experience",
            "text": f"Students with 2 or more internships achieved a {multi_intern:.1f}% placement rate, whereas those with zero internships had a {zero_intern:.1f}% placement rate."
        })

    return insights
