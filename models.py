import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from preprocessing import preprocess_pipeline, LEAKAGE_COLUMNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "model_pipeline.joblib")

# Global in-memory cache for trained models evaluation metrics
trained_models_store = {}

def train_and_evaluate_model(
    df: pd.DataFrame,
    model_name: str = "Random Forest",
    imputation_strategy: str = "mean",
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: int = 10,
    n_estimators: int = 100
) -> dict:
    """Trains selected machine learning model and returns complete evaluation metrics."""
    global trained_models_store

    # Preprocess dataset
    prep = preprocess_pipeline(
        df,
        imputation_strategy=imputation_strategy,
        test_size=test_size,
        random_state=random_state
    )

    X_train, X_test = prep["X_train"], prep["X_test"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    feature_names = prep["feature_names_encoded"]

    # Instantiate model
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    else:  # Default: Random Forest
        model_name = "Random Forest"
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
        )

    # Train model
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc_val = float(roc_auc_score(y_test, y_proba))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred).tolist()

    # ROC Curve coordinates (subsampled for fast chart rendering)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    if len(fpr) > 100:
        indices = np.linspace(0, len(fpr) - 1, 100, dtype=int)
        fpr_list = list(fpr[indices].round(4))
        tpr_list = list(tpr[indices].round(4))
    else:
        fpr_list = list(fpr.round(4))
        tpr_list = list(tpr.round(4))

    # Feature Importance / Coefficients
    importances = []
    if hasattr(clf, "feature_importances_"):
        raw_imp = clf.feature_importances_
        for name, score in zip(feature_names, raw_imp):
            importances.append({"feature": name, "importance": round(float(score), 4)})
        importances.sort(key=lambda x: x["importance"], reverse=True)
    elif hasattr(clf, "coef_"):
        raw_coef = clf.coef_[0]
        for name, score in zip(feature_names, raw_coef):
            importances.append({"feature": name, "importance": round(float(score), 4)})
        importances.sort(key=lambda x: abs(x["importance"]), reverse=True)

    result = {
        "model_name": model_name,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "auc": round(auc_val, 4),
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr_list, "tpr": tpr_list},
        "feature_importances": importances[:15],
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    # Store in memory cache
    trained_models_store[model_name] = result

    # Save trained model and pipeline artifact for Prediction API
    save_payload = {
        "model": clf,
        "scaler": prep["scaler"],
        "feature_names": feature_names,
        "imputation_strategy": imputation_strategy,
        "model_name": model_name,
        "metrics": result
    }
    joblib.dump(save_payload, MODEL_SAVE_PATH)

    return result

def get_model_comparison() -> dict:
    """Returns side-by-side comparison metrics for all trained models."""
    global trained_models_store
    
    models_list = list(trained_models_store.values())
    if not models_list:
        return {"has_models": False, "models": []}

    # Highlight best model based on accuracy / f1
    sorted_models = sorted(models_list, key=lambda x: x["f1_score"], reverse=True)
    best_model_name = sorted_models[0]["model_name"]

    return {
        "has_models": True,
        "best_model": best_model_name,
        "models": models_list
    }

def get_saved_model():
    """Loads the persisted model pipeline from disk if available."""
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            return joblib.load(MODEL_SAVE_PATH)
        except Exception:
            return None
    return None
