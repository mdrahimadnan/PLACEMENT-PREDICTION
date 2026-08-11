import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

# Explicit feature definition to prevent data leakage
LEAKAGE_COLUMNS = ["PlacementStatus", "Salary Package", "StudentID", "IsAnomaly"]

CATEGORICAL_FEATURES = [
    "Gender", "City", "CollegeTier", "Stream", "Specialisation",
    "Hostel", "HistoryOfBacklogs", "CGPA_Tier"
]

def inspect_dataset_pre_processing(df: pd.DataFrame) -> dict:
    """Calculates dataset health metrics before preprocessing."""
    missing_summary = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
    
    return {
        "shape": list(df.shape),
        "total_rows": len(df),
        "total_cols": df.shape[1],
        "missing_summary": missing_summary,
        "missing_total_count": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

def preprocess_pipeline(df: pd.DataFrame, imputation_strategy: str = "mean", test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Applies missing value imputation, one-hot encoding, feature scaling,
    and returns a leakage-free train/test dataset split.
    """
    df_clean = df.copy()

    # Drop target & leakage columns to form feature matrix X
    feature_cols = [c for c in df_clean.columns if c not in LEAKAGE_COLUMNS]
    
    X = df_clean[feature_cols].copy()
    y = df_clean["PlacementStatus"].values if "PlacementStatus" in df_clean.columns else np.zeros(len(df_clean))

    # Identify numeric and categorical columns within features
    numeric_cols = list(X.select_dtypes(include='number').columns)
    cat_cols = list(X.select_dtypes(exclude='number').columns)

    # 1. Handle Missing Values
    if imputation_strategy == "drop":
        valid_indices = X.dropna().index
        X = X.loc[valid_indices]
        y = y[valid_indices]
    else:
        # Numeric Imputation
        num_strategy = "median" if imputation_strategy == "median" else "mean"
        num_imputer = SimpleImputer(strategy=num_strategy)
        if len(numeric_cols) > 0 and X[numeric_cols].isna().sum().sum() > 0:
            X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

        # Categorical Imputation
        cat_imputer = SimpleImputer(strategy="most_frequent")
        if len(cat_cols) > 0 and X[cat_cols].isna().sum().sum() > 0:
            X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # 2. Categorical Encoding (One-Hot Encoding with pandas get_dummies for clean column names)
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    encoded_feature_names = list(X_encoded.columns)

    # 3. Train-Test Split BEFORE Scaling to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4. Feature Scaling (Fit ONLY on training set, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "imputation_strategy": imputation_strategy,
        "feature_cols_raw": feature_cols,
        "feature_names_encoded": encoded_feature_names,
        "n_features_encoded": len(encoded_feature_names),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "train_size": len(X_train_scaled),
        "test_size": len(X_test_scaled),
        "total_samples": len(X_encoded)
    }
