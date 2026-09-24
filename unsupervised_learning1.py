# ============================================================
# UNSUPERVISED LEARNING - CLUSTERING ANALYSIS
# K-Means + PCA + Cluster Profiling + Heatmap + DBSCAN
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.decomposition import PCA


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)

try:
    from IPython.display import display
except ImportError:
    display = print


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Google Colab
try:
    from google.colab import files

    print("Please upload your CSV file:")
    uploaded = files.upload()

    csv_files = [
        name for name in uploaded.keys()
        if name.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("No CSV file was uploaded.")

    DATA_PATH = csv_files[0]

except ImportError:

    # Jupyter / Local Python
    current_dir = os.getcwd()

    csv_candidates = [
        os.path.join(current_dir, f)
        for f in os.listdir(current_dir)
        if f.lower().endswith(".csv")
        and not f.startswith("unsupervised_clustered")
    ]

    if not csv_candidates:
        raise FileNotFoundError(
            "No CSV file found. Place the dataset in the working directory."
        )

    DATA_PATH = csv_candidates[0]


# Read dataset
df = pd.read_csv(DATA_PATH)

print("\n============================================")
print("DATASET INFORMATION")
print("============================================")

print("Dataset path:", DATA_PATH)
print("Shape:", df.shape)

print("\nFirst 5 rows:")
display(df.head())

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nDataset Info:")
df.info()


# ============================================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numeric_cols_all = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\n============================================")
print("COLUMN INFORMATION")
print("============================================")

print("\nNumerical columns:")
print(numeric_cols_all)

print("\nCategorical columns:")
print(categorical_cols)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n============================================")
print("MISSING VALUES")
print("============================================")

missing_values = (
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .to_frame("missing_count")
)

display(missing_values.head(20))


# ============================================================
# 5. COLUMNS EXCLUDED FROM UNSUPERVISED LEARNING
# ============================================================

EXCLUDE_FROM_UNSUPERVISED = [
    "StudentID",
    "PlacementStatus",
    "Salary Package",
    "IsAnomaly"
]

available_excluded = [
    c for c in EXCLUDE_FROM_UNSUPERVISED
    if c in df.columns
]

missing_excluded = [
    c for c in EXCLUDE_FROM_UNSUPERVISED
    if c not in df.columns
]

print("\n============================================")
print("EXCLUDED COLUMNS")
print("============================================")

print("Excluded from clustering:")
print(available_excluded)

if missing_excluded:
    print("\nColumns not found:")
    print(missing_excluded)


# ============================================================
# 6. CORRELATION WITH PLACEMENT STATUS
# ============================================================

if "PlacementStatus" in df.columns:

    print("\n============================================")
    print("CORRELATION WITH PLACEMENT STATUS")
    print("============================================")

    # Only calculate correlation if PlacementStatus is numeric
    if pd.api.types.is_numeric_dtype(df["PlacementStatus"]):

        corr = (
            df[numeric_cols_all]
            .corr(numeric_only=True)["PlacementStatus"]
            .sort_values(ascending=False)
        )

        display(
            corr.to_frame("corr_with_PlacementStatus")
        )

    else:
        print(
            "PlacementStatus is categorical, so numeric correlation "
            "was skipped."
        )


# ============================================================
# 7. SELECT FEATURES FOR CLUSTERING
# ============================================================

FEATURES = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular"
]

# Keep only columns actually present in dataset
FEATURES = [
    c for c in FEATURES
    if c in df.columns
]

print("\n============================================")
print("CLUSTERING FEATURES")
print("============================================")

print("Number of modeling features:", len(FEATURES))
print(FEATURES)


if len(FEATURES) < 2:
    raise ValueError(
        "At least two valid numerical features are required for clustering."
    )


# ============================================================
# 8. CREATE FEATURE MATRIX
# ============================================================

X_raw = df[FEATURES].copy()

print("\n============================================")
print("FEATURE STATISTICS")
print("============================================")

display(X_raw.describe().T)


# ============================================================
# 9. PREPROCESSING
#    Missing values -> Median
#    Scaling -> StandardScaler
# ============================================================

print("\n============================================")
print("PREPROCESSING")
print("============================================")

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

# Impute missing values
X_imputed = imputer.fit_transform(X_raw)

# Scale features
X_scaled = scaler.fit_transform(X_imputed)

print("Original shape:", X_raw.shape)
print("Scaled shape:", X_scaled.shape)

print("Missing values after imputation:",
      np.isnan(X_scaled).sum())


# ============================================================
# 10. K-MEANS FOR K = 2 TO 7
#     Compare Random Initialization vs K-Means++
# ============================================================

print("\n============================================")
print("K-MEANS ANALYSIS")
print("============================================")

k_values = range(2, 8)

results_list = []

for k in k_values:

    # ----------------------------------------
    # Random Initialization
    # ----------------------------------------

    km_rand = KMeans(
        n_clusters=k,
        init="random",
        n_init=10,
        random_state=RANDOM_STATE
    )

    labels_rand = km_rand.fit_predict(X_scaled)

    sil_rand = silhouette_score(
        X_scaled,
        labels_rand,
        sample_size=min(10000, len(X_scaled)),
        random_state=RANDOM_STATE
    )


    # ----------------------------------------
    # K-Means++ Initialization
    # ----------------------------------------

    km_plus = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE
    )

    labels_plus = km_plus.fit_predict(X_scaled)

    sil_plus = silhouette_score(
        X_scaled,
        labels_plus,
        sample_size=min(10000, len(X_scaled)),
        random_state=RANDOM_STATE
    )


    # ----------------------------------------
    # Store results
    # ----------------------------------------

    results_list.append({
        "K": k,
        "Random_Inertia": km_rand.inertia_,
        "KMeans++_Inertia": km_plus.inertia_,
        "Random_Silhouette": sil_rand,
        "KMeans++_Silhouette": sil_plus
    })


# Convert results to DataFrame
kmeans_results = pd.DataFrame(results_list)

print("\nK-Means Results:")
display(kmeans_results.round(4))


# ============================================================
# 11. ELBOW METHOD
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    kmeans_results["K"],
    kmeans_results["Random_Inertia"],
    marker="o",
    label="Random Initialization"
)

plt.plot(
    kmeans_results["K"],
    kmeans_results["KMeans++_Inertia"],
    marker="o",
    label="K-Means++"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method: Inertia vs K")
plt.legend()

plt.tight_layout()

plt.savefig(
    "elbow_inertia_vs_k.png",
    dpi=150
)

plt.show()


# ============================================================
# 12. SILHOUETTE SCORE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    kmeans_results["K"],
    kmeans_results["Random_Silhouette"],
    marker="o",
    label="Random Initialization"
)

plt.plot(
    kmeans_results["K"],
    kmeans_results["KMeans++_Silhouette"],
    marker="o",
    label="K-Means++"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")

plt.axhline(
    0,
    linewidth=0.8
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "silhouette_score_vs_k.png",
    dpi=150
)

plt.show()


# ============================================================
# 13. SELECT BEST K
# ============================================================

best_k_silhouette = int(
    kmeans_results.loc[
        kmeans_results["KMeans++_Silhouette"].idxmax(),
        "K"
    ]
)

K_BEST = best_k_silhouette

print("\n============================================")
print("BEST K")
print("============================================")

print(
    "K with highest K-Means++ silhouette:",
    K_BEST
)


# ============================================================
# 14. FINAL K-MEANS - RANDOM INITIALIZATION
# ============================================================

km_random_final = KMeans(
    n_clusters=K_BEST,
    init="random",
    n_init=10,
    random_state=RANDOM_STATE
)

labels_random_final = km_random_final.fit_predict(
    X_scaled
)


# ============================================================
# 15. FINAL K-MEANS - K-MEANS++
# ============================================================

km_plus_final = KMeans(
    n_clusters=K_BEST,
    init="k-means++",
    n_init=10,
    random_state=RANDOM_STATE
)

labels_plus_final = km_plus_final.fit_predict(
    X_scaled
)


# ============================================================
# 16. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "K-Means Random",
        "K-Means++"
    ],

    "K": [
        K_BEST,
        K_BEST
    ],

    "Inertia": [
        km_random_final.inertia_,
        km_plus_final.inertia_
    ],

    "Silhouette": [

        silhouette_score(
            X_scaled,
            labels_random_final,
            sample_size=min(10000, len(X_scaled)),
            random_state=RANDOM_STATE
        ),

        silhouette_score(
            X_scaled,
            labels_plus_final,
            sample_size=min(10000, len(X_scaled)),
            random_state=RANDOM_STATE
        )
    ],

    "Davies_Bouldin": [

        davies_bouldin_score(
            X_scaled,
            labels_random_final
        ),

        davies_bouldin_score(
            X_scaled,
            labels_plus_final
        )
    ],

    "Calinski_Harabasz": [

        calinski_harabasz_score(
            X_scaled,
            labels_random_final
        ),

        calinski_harabasz_score(
            X_scaled,
            labels_plus_final
        )
    ]
})


print("\n============================================")
print("MODEL COMPARISON")
print("============================================")

display(comparison.round(4))


# ============================================================
# 17. PCA FOR 2-D VISUALIZATION
# ============================================================

print("\n============================================")
print("PCA VISUALIZATION")
print("============================================")

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca2 = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

print(
    "PC1 explained variance:",
    round(explained_variance[0] * 100, 2),
    "%"
)

print(
    "PC2 explained variance:",
    round(explained_variance[1] * 100, 2),
    "%"
)

print(
    "Total explained variance:",
    round(explained_variance.sum() * 100, 2),
    "%"
)


# ============================================================
# 18. PCA CLUSTER PLOT
# ============================================================

plt.figure(figsize=(9, 6))

scatter = plt.scatter(
    X_pca2[:, 0],
    X_pca2[:, 1],
    c=labels_plus_final,
    s=20,
    alpha=0.6,
    cmap="tab10"
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    f"K-Means++ Clusters (K={K_BEST}) in PCA 2-D Space"
)

plt.colorbar(
    scatter,
    label="Cluster"
)

plt.tight_layout()

plt.show()


# ============================================================
# 19. CLUSTER PROFILING
# ============================================================

print("\n============================================")
print("CLUSTER PROFILES")
print("============================================")

profile_df = df[FEATURES].copy()

profile_df["Cluster"] = labels_plus_final

cluster_profile = (
    profile_df
    .groupby("Cluster")[FEATURES]
    .mean()
    .round(2)
)

cluster_sizes = (
    pd.Series(labels_plus_final)
    .value_counts()
    .sort_index()
    .rename("Student_Count")
)

cluster_summary = pd.concat(
    [
        cluster_sizes,
        cluster_profile
    ],
    axis=1
)

display(cluster_summary)


# ============================================================
# 20. HEATMAP OF CLUSTER PROFILES
# ============================================================

print("\n============================================")
print("CLUSTER PROFILE HEATMAP")
print("============================================")

cluster_profile_scaled = cluster_profile.copy()

# Standardize each feature across clusters
feature_std = cluster_profile_scaled.std(
    ddof=0
).replace(0, 1)

heatmap_data = (
    cluster_profile_scaled -
    cluster_profile_scaled.mean()
) / feature_std


plt.figure(figsize=(14, 7))

sns.heatmap(
    heatmap_data.T,
    center=0,
    cmap="coolwarm"
)

plt.xlabel("Cluster")
plt.ylabel("Feature")

plt.title(
    "Cluster Profiles - Relative Strength by Feature"
)

plt.tight_layout()

plt.show()


# ============================================================
# 21. DBSCAN
# ============================================================

print("\n============================================")
print("DBSCAN ANALYSIS")
print("============================================")


# DBSCAN can become expensive for large datasets.
# Therefore, use a reproducible sample.

DBSCAN_SAMPLE_SIZE = min(
    10000,
    len(X_scaled)
)

dbscan_rng = np.random.default_rng(
    RANDOM_STATE
)

dbscan_idx = dbscan_rng.choice(
    len(X_scaled),
    size=DBSCAN_SAMPLE_SIZE,
    replace=False
)

# IMPORTANT:
# X_scaled is already a NumPy array,
# so DO NOT use .iloc here.

X_db_sample = X_scaled[dbscan_idx]


print(
    "DBSCAN parameter-search sample shape:",
    X_db_sample.shape
)


# ============================================================
# 22. DBSCAN PARAMETERS
# ============================================================

EPS_VALUES = [
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0
]

MIN_SAMPLES_VALUES = [
    5,
    10,
    20
]


# ============================================================
# 23. DBSCAN PARAMETER SEARCH
# ============================================================

dbscan_results = []

for eps in EPS_VALUES:

    for min_samples in MIN_SAMPLES_VALUES:

        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )

        labels = db.fit_predict(
            X_db_sample
        )


        # Number of clusters
        n_clusters = (
            len(set(labels))
            - (1 if -1 in labels else 0)
        )


        # Percentage of noise
        noise_pct = (
            (labels == -1).mean()
            * 100
        )


        # Ignore noise for clustering metrics
        non_noise = labels != -1


        # Calculate metrics only when
        # at least 2 clusters exist
        if (
            n_clusters >= 2
            and non_noise.sum() > 1
        ):

            sil = silhouette_score(
                X_db_sample[non_noise],
                labels[non_noise]
            )

            dbi = davies_bouldin_score(
                X_db_sample[non_noise],
                labels[non_noise]
            )

        else:

            sil = np.nan
            dbi = np.nan


        # Store result
        dbscan_results.append({

            "eps": eps,

            "min_samples": min_samples,

            "clusters": n_clusters,

            "noise_pct": noise_pct,

            "silhouette_non_noise": sil,

            "davies_bouldin_non_noise": dbi
        })


# ============================================================
# 24. DISPLAY DBSCAN RESULTS
# ============================================================

dbscan_results = pd.DataFrame(
    dbscan_results
)

print("\n============================================")
print("DBSCAN RESULTS")
print("============================================")

display(
    dbscan_results
    .sort_values(
        [
            "silhouette_non_noise",
            "noise_pct"
        ],
        ascending=[
            False,
            True
        ]
    )
    .round(4)
)


# ============================================================
# 25. FINISHED
# ============================================================

print("\n============================================")
print("ANALYSIS COMPLETED")
print("============================================")

print("Dataset shape:", df.shape)

print(
    "Number of features used:",
    len(FEATURES)
)

print(
    "Best K:",
    K_BEST
)

print(
    "PCA explained variance:",
    round(
        explained_variance.sum() * 100,
        2
    ),
    "%"
)

print("\nGenerated files:")
print("- elbow_inertia_vs_k.png")
print("- silhouette_score_vs_k.png")

print("\nAll clustering analysis completed successfully.")