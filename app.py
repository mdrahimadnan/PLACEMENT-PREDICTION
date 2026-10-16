import os
import io
from flask import Flask, render_template, request, jsonify, Response
from load_data import (
    load_data, get_data_summary, get_duplicate_count, filter_dataframe
)
from eda import (
    get_overview_info, get_missing_analysis, get_univariate_analysis,
    get_categorical_analysis, get_bivariate_analysis, get_correlation_analysis,
    get_outlier_analysis, get_auto_insights
)
from preprocessing import inspect_dataset_pre_processing, preprocess_pipeline
from models import train_and_evaluate_model, get_model_comparison, get_saved_model
from prediction import predict_student_placement
from analytics import (
    get_placement_analytics, get_salary_analytics, get_anomaly_analytics, get_paginated_students
)

app = Flask(__name__, template_folder="Templates", static_folder="Static")

# Global error handler helper
def json_error(message: str, status_code: int = 400):
    return jsonify({"status": "error", "message": message}), status_code

# Helper to get current dataset (supports uploaded or default)
def get_current_df():
    try:
        return load_data()
    except Exception as e:
        return None

# ================= FLASK PAGE ROUTES =================

@app.route("/")
@app.route("/dashboard")
def index():
    df = get_current_df()
    if df is None:
        return render_template("index.html", error="Placement dataset CSV not found.", active_tab="tab-overview")
    
    summary = get_data_summary(df)
    insights = get_auto_insights(df)
    return render_template("index.html", summary=summary, insights=insights, active_tab="tab-overview")

@app.route("/data-loading")
def page_data_loading():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-loading")

@app.route("/eda")
def page_eda():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-eda")

@app.route("/preprocessing")
def page_preprocessing():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-preprocessing")

@app.route("/model-training")
def page_model_training():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-training")

@app.route("/evaluation")
def page_evaluation():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-evaluation")

@app.route("/placement-analysis")
def page_placement_analysis():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-placement")

@app.route("/salary-analysis")
def page_salary_analysis():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-salary")

@app.route("/anomaly-analysis")
def page_anomaly_analysis():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-anomaly")

@app.route("/students")
def page_students():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-students")

@app.route("/prediction")
def page_prediction():
    df = get_current_df()
    summary = get_data_summary(df) if df is not None else None
    return render_template("index.html", summary=summary, active_tab="tab-predictor")


# ================= FLASK JSON API ENDPOINTS =================

@app.route("/api/dashboard-stats", methods=["GET"])
def api_dashboard_stats():
    df = get_current_df()
    if df is None:
        return json_error("Dataset not available.")

    filters = {
        "gender": request.args.get("gender", "All"),
        "city": request.args.get("city", "All"),
        "stream": request.args.get("stream", "All"),
        "tier": request.args.get("tier", "All"),
        "specialisation": request.args.get("specialisation", "All"),
        "placement_status": request.args.get("placement_status", "All")
    }

    filtered_df = filter_dataframe(df, filters)
    stats = get_data_summary(filtered_df)
    
    # Also include quick chart arrays for Dashboard visuals
    stream_counts = filtered_df["Stream"].value_counts().to_dict() if "Stream" in filtered_df.columns else {}
    placed_by_stream = filtered_df.groupby("Stream")["PlacementStatus"].mean().round(4).to_dict() if "PlacementStatus" in filtered_df.columns else {}
    placed_by_tier = filtered_df.groupby("CollegeTier")["PlacementStatus"].mean().round(4).to_dict() if "CollegeTier" in filtered_df.columns else {}
    placed_by_gender = filtered_df.groupby("Gender")["PlacementStatus"].mean().round(4).to_dict() if "Gender" in filtered_df.columns else {}

    cgpa_sample = filtered_df["CGPA"].dropna().sample(min(2000, len(filtered_df)), random_state=42).tolist() if "CGPA" in filtered_df.columns else []
    
    placed_salaries = filtered_df[(filtered_df["PlacementStatus"] == 1) & (filtered_df["Salary Package"] > 0)]["Salary Package"] if "PlacementStatus" in filtered_df.columns else filtered_df["Salary Package"]
    salary_sample = placed_salaries.dropna().sample(min(2000, len(placed_salaries)), random_state=42).tolist() if len(placed_salaries) > 0 else []

    return jsonify({
        "status": "success",
        "summary": stats,
        "charts": {
            "placement_distribution": {"placed": stats["placed_count"], "unplaced": stats["unplaced_count"]},
            "stream_counts": stream_counts,
            "placement_by_stream": {k: round(v*100, 1) for k, v in placed_by_stream.items()},
            "placement_by_tier": {k: round(v*100, 1) for k, v in placed_by_tier.items()},
            "placement_by_gender": {k: round(v*100, 1) for k, v in placed_by_gender.items()},
            "cgpa_sample": cgpa_sample,
            "salary_sample": salary_sample
        }
    })

@app.route("/api/eda/overview", methods=["GET"])
def api_eda_overview():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    return jsonify({"status": "success", "overview": get_overview_info(df)})

@app.route("/api/eda/missing", methods=["GET"])
def api_eda_missing():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    return jsonify({"status": "success", "missing": get_missing_analysis(df)})

@app.route("/api/eda/univariate", methods=["GET"])
def api_eda_univariate():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    feature = request.args.get("feature", "CGPA")
    viz_type = request.args.get("viz_type", "histogram")
    res = get_univariate_analysis(df, feature, viz_type)
    if "error" in res: return json_error(res["error"])
    return jsonify({"status": "success", "data": res})

@app.route("/api/eda/categorical", methods=["GET"])
def api_eda_categorical():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    feature = request.args.get("feature", "Stream")
    res = get_categorical_analysis(df, feature)
    if "error" in res: return json_error(res["error"])
    return jsonify({"status": "success", "data": res})

@app.route("/api/eda/bivariate", methods=["GET"])
def api_eda_bivariate():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    x_feature = request.args.get("x_feature", "CGPA")
    y_feature = request.args.get("y_feature", "CodingTestScore")
    group_by = request.args.get("group_by", "PlacementStatus")
    viz_type = request.args.get("viz_type", "scatter")
    res = get_bivariate_analysis(df, x_feature, y_feature, group_by, viz_type)
    return jsonify({"status": "success", "data": res})

@app.route("/api/eda/correlation", methods=["GET"])
def api_eda_correlation():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    return jsonify({"status": "success", "correlation": get_correlation_analysis(df)})

@app.route("/api/eda/outliers", methods=["GET"])
def api_eda_outliers():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    feature = request.args.get("feature", "CGPA")
    res = get_outlier_analysis(df, feature)
    if "error" in res: return json_error(res["error"])
    return jsonify({"status": "success", "data": res})

@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    
    imputation_strategy = request.json.get("imputation_strategy", "mean") if request.is_json else request.form.get("imputation_strategy", "mean")
    test_size = float(request.json.get("test_size", 0.2)) if request.is_json else float(request.form.get("test_size", 0.2))

    pre_info = inspect_dataset_pre_processing(df)
    pipeline_res = preprocess_pipeline(df, imputation_strategy=imputation_strategy, test_size=test_size)

    return jsonify({
        "status": "success",
        "before": pre_info,
        "after": {
            "imputation_strategy": imputation_strategy,
            "encoded_features_count": pipeline_res["n_features_encoded"],
            "train_samples": pipeline_res["train_size"],
            "test_samples": pipeline_res["test_size"],
            "feature_cols": pipeline_res["feature_names_encoded"][:10]
        }
    })

@app.route("/api/train-model", methods=["POST"])
def api_train_model():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")

    data = request.get_json(force=True) if request.is_json else request.form
    model_name = data.get("model_name", "Random Forest")
    imputation_strategy = data.get("imputation_strategy", "mean")
    test_size = float(data.get("test_size", 0.2))
    max_depth = int(data.get("max_depth", 10))
    n_estimators = int(data.get("n_estimators", 100))

    metrics = train_and_evaluate_model(
        df,
        model_name=model_name,
        imputation_strategy=imputation_strategy,
        test_size=test_size,
        max_depth=max_depth,
        n_estimators=n_estimators
    )

    comparison = get_model_comparison()

    return jsonify({
        "status": "success",
        "metrics": metrics,
        "comparison": comparison
    })

@app.route("/api/model-comparison", methods=["GET"])
def api_model_comparison():
    return jsonify({"status": "success", "comparison": get_model_comparison()})

@app.route("/api/placement-analytics", methods=["GET"])
def api_placement_analytics():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    return jsonify({"status": "success", "analytics": get_placement_analytics(df)})

@app.route("/api/salary-analytics", methods=["GET"])
def api_salary_analytics():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    return jsonify({"status": "success", "analytics": get_salary_analytics(df)})

@app.route("/api/anomaly-analytics", methods=["GET"])
def api_anomaly_analytics():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")
    return jsonify({"status": "success", "analytics": get_anomaly_analytics(df)})

@app.route("/api/students", methods=["GET"])
def api_students():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")

    filters = {
        "gender": request.args.get("gender", "All"),
        "city": request.args.get("city", "All"),
        "stream": request.args.get("stream", "All"),
        "tier": request.args.get("tier", "All"),
        "specialisation": request.args.get("specialisation", "All"),
        "placement_status": request.args.get("placement_status", "All")
    }

    search = request.args.get("search", "").strip()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 25))

    res = get_paginated_students(df, filters=filters, search=search, page=page, per_page=per_page)
    return jsonify({"status": "success", "data": res})

@app.route("/api/export-students-csv", methods=["GET"])
def api_export_students_csv():
    df = get_current_df()
    if df is None: return json_error("Dataset unavailable.")

    filters = {
        "gender": request.args.get("gender", "All"),
        "city": request.args.get("city", "All"),
        "stream": request.args.get("stream", "All"),
        "tier": request.args.get("tier", "All"),
        "specialisation": request.args.get("specialisation", "All"),
        "placement_status": request.args.get("placement_status", "All")
    }

    filtered_df = filter_dataframe(df, filters)
    
    csv_buffer = io.StringIO()
    filtered_df.to_csv(csv_buffer, index=False)

    return Response(
        csv_buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=filtered_student_records.csv"}
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict() if request.form else (request.get_json(force=True) if request.is_json else {})
        res = predict_student_placement(data)
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)