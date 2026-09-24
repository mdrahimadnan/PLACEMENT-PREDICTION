"""
=============================================================================
UNSUPERVISED LEARNING - PLACEMENT DATASET ANALYSIS
=============================================================================
This script demonstrates the complete Unsupervised Learning pipeline:
  1. Data Loading & Exploratory Preprocessing
  2. Feature Scaling & Missing Value Imputation
  3. Dimensionality Reduction with PCA (2D/3D + Explained Variance)
  4. Optimal Clusters Search:
     - Elbow Method (Inertia)
     - Silhouette Score
     - Davies-Bouldin Score
     - Calinski-Harabasz Score
  5. K-Means Clustering & Cluster Profiling
  6. Hierarchical (Agglomerative) Clustering & Dendrogram
  7. Density-Based Clustering (DBSCAN) with K-Distance Estimation
  8. Cluster Insights vs Placement Outcomes
=============================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reliable script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 160)


def get_base_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return r"C:\studies\ML\PLACEMENT_PREDICTION"


def main():
    print("=" * 80)
    print(" [*] STARTING UNSUPERVISED LEARNING PIPELINE")
    print("=" * 80)

    base_dir = get_base_dir()

    # -------------------------------------------------------------------------
    # 1. LOAD DATASET (Explicitly resolves in C:\studies\ML\PLACEMENT_PREDICTION)
    # -------------------------------------------------------------------------
    candidates = [
        os.path.join(base_dir, "placement_predict_50k Dataset.csv"),
        r"C:\studies\ML\PLACEMENT_PREDICTION\placement_predict_50k Dataset.csv",
        "placement_predict_50k Dataset.csv",
        os.path.join("..", "placement_predict_50k Dataset.csv"),
    ]

    data_path = None
    for cand in candidates:
        if os.path.exists(cand):
            data_path = os.path.abspath(cand)
            break

    if not data_path:
        raise FileNotFoundError(
            f"Dataset not found! Looked in:\n" + "\n".join(f" - {c}" for c in candidates)
        )

    print(f"\n[1] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)
    print(f"    Dataset loaded successfully! Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # -------------------------------------------------------------------------
    # 2. SELECT FEATURES FOR CLUSTERING
    # -------------------------------------------------------------------------
    clustering_features = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "Projects",
        "Workshops",
        "Certifications",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    # Ensure all selected features exist in df
    clustering_features = [col for col in clustering_features if col in df.columns]
    print(f"\n[2] Features selected for unsupervised segmentation ({len(clustering_features)}):")
    for feat in clustering_features:
        print(f"    - {feat}")

    X_raw = df[clustering_features].copy()

    # -------------------------------------------------------------------------
    # 3. PREPROCESSING PIPELINE (IMPUTE + SCALE)
    # -------------------------------------------------------------------------
    print("\n[3] Preprocessing: Imputing missing values and applying StandardScaler...")
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_scaled = preprocessor.fit_transform(X_raw)
    print("    Standardization complete (mean ~ 0, variance ~ 1).")

    # Sample a manageable subset for computationally expensive algorithms (Dendrogram / DBSCAN / Silhouette)
    sample_size = min(5000, len(df))
    print(f"\n[Note] For high-efficiency evaluation, using a random sample of {sample_size:,} records")
    sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
    X_sample = X_scaled[sample_indices]

    # -------------------------------------------------------------------------
    # 4. DIMENSIONALITY REDUCTION WITH PCA
    # -------------------------------------------------------------------------
    print("\n[4] Performing Principal Component Analysis (PCA)...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    print(f"    Explained Variance: PC1 = {var_exp[0]:.2%}, PC2 = {var_exp[1]:.2%} (Total: {var_exp.sum():.2%})")

    # -------------------------------------------------------------------------
    # 5. FINDING OPTIMAL K FOR K-MEANS
    # -------------------------------------------------------------------------
    print("\n[5] Evaluating optimal clusters (k = 2 to 6)...")
    k_range = range(2, 7)
    inertias = []
    silhouettes = []
    davies_bouldins = []
    calinskis = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

        # Calculate metrics on sample for fast execution
        labels_sample = km.predict(X_sample)
        sil = silhouette_score(X_sample, labels_sample)
        db = davies_bouldin_score(X_sample, labels_sample)
        ch = calinski_harabasz_score(X_sample, labels_sample)

        silhouettes.append(sil)
        davies_bouldins.append(db)
        calinskis.append(ch)

        print(f"    k={k} | Inertia={km.inertia_:.1f} | Silhouette={sil:.4f} | Davies-Bouldin={db:.4f} | Calinski-Harabasz={ch:.1f}")

    # Plot metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Elbow (Inertia)
    axes[0].plot(list(k_range), inertias, marker="o", color="#2b5c8f", lw=2)
    axes[0].set_title("Elbow Method (Inertia)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)")

    # Silhouette Score
    axes[1].plot(list(k_range), silhouettes, marker="s", color="#2ca02c", lw=2)
    axes[1].set_title("Silhouette Score (Higher is Better)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Coefficient")

    # Davies-Bouldin Score
    axes[2].plot(list(k_range), davies_bouldins, marker="^", color="#d62728", lw=2)
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Davies-Bouldin Index")

    plt.tight_layout()
    eval_plot_path = os.path.join(base_dir, "unsupervised_kmeans_evaluation.png")
    plt.savefig(eval_plot_path, dpi=150)
    plt.close()
    print(f"    Saved evaluation plot to '{eval_plot_path}'")

    # Choose optimal K (default k=3 or best silhouette)
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n    [+] Highest Silhouette Score achieved at k = {best_k}")

    # -------------------------------------------------------------------------
    # 6. FIT FINAL K-MEANS MODEL & PROFILING
    # -------------------------------------------------------------------------
    optimal_k = 3  # High/Mid/Low student aptitude tiers
    print(f"\n[6] Fitting Final K-Means with k = {optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
    df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

    # Cluster Profiles (Mean of key metrics)
    print("\n--- Cluster Summary Profile (Original Feature Scale) ---")
    profile = df.groupby("KMeans_Cluster")[clustering_features].mean().round(2)
    print(profile)

    # Check placement rate per cluster if ground truth available
    if "PlacementStatus" in df.columns:
        print("\n--- Cluster vs Placement Rate ---")
        placement_rate = df.groupby("KMeans_Cluster")["PlacementStatus"].agg(
            Count="count",
            Placed_Percentage=lambda x: round((x.mean() * 100), 2)
        )
        print(placement_rate)

    # -------------------------------------------------------------------------
    # 7. PCA VISUALIZATION OF K-MEANS CLUSTERS
    # -------------------------------------------------------------------------
    print("\n[7] Visualizing Clusters via PCA 2D Projection...")
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        X_pca[sample_indices, 0],
        X_pca[sample_indices, 1],
        c=df.iloc[sample_indices]["KMeans_Cluster"],
        cmap="viridis",
        alpha=0.6,
        s=30
    )
    plt.colorbar(scatter, label="KMeans Cluster")
    plt.title(f"K-Means Clustering (k={optimal_k}) Projected on PCA Components", fontsize=14, fontweight="bold")
    plt.xlabel(f"Principal Component 1 ({var_exp[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({var_exp[1]:.1%} variance)")
    pca_plot_path = os.path.join(base_dir, "unsupervised_pca_clusters.png")
    plt.savefig(pca_plot_path, dpi=150)
    plt.close()
    print(f"    Saved PCA visualization to '{pca_plot_path}'")

    # -------------------------------------------------------------------------
    # 8. HIERARCHICAL / AGGLOMERATIVE CLUSTERING & DENDROGRAM
    # -------------------------------------------------------------------------
    print("\n[8] Hierarchical Clustering Analysis...")
    dendro_sample_size = 150  # Sample subset for readable dendrogram
    dendro_indices = np.random.choice(len(df), size=dendro_sample_size, replace=False)
    X_dendro = X_scaled[dendro_indices]

    # Compute Linkage matrix using Ward's method
    Z = linkage(X_dendro, method="ward", metric="euclidean")

    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=10,
        show_contracted=True
    )
    plt.title("Hierarchical Clustering Dendrogram (Ward Linkage, Sample n=150)", fontsize=14, fontweight="bold")
    plt.xlabel("Cluster Size / Sample Index")
    plt.ylabel("Euclidean Distance (Ward)")
    dendro_plot_path = os.path.join(base_dir, "unsupervised_dendrogram.png")
    plt.savefig(dendro_plot_path, dpi=150)
    plt.close()
    print(f"    Saved Dendrogram plot to '{dendro_plot_path}'")

    # Fit Agglomerative Clustering on sample
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    agg_labels = agg_clustering.fit_predict(X_sample[:1000])
    agg_sil = silhouette_score(X_sample[:1000], agg_labels)
    print(f"    Agglomerative Clustering Silhouette Score (k={optimal_k}): {agg_sil:.4f}")

    # -------------------------------------------------------------------------
    # 9. DENSITY-BASED CLUSTERING (DBSCAN)
    # -------------------------------------------------------------------------
    print("\n[9] Density-Based Spatial Clustering (DBSCAN)...")
    # Determine eps using NearestNeighbors k-distance graph
    k_neighbors = 2 * len(clustering_features)
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X_sample)
    distances, _ = nn.kneighbors(X_sample)
    distances = np.sort(distances[:, -1])

    # Run DBSCAN
    eps_val = 2.5
    min_samples_val = k_neighbors
    dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    db_labels = dbscan.fit_predict(X_sample)

    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = list(db_labels).count(-1)
    print(f"    DBSCAN (eps={eps_val}, min_samples={min_samples_val}):")
    print(f"    - Estimated Clusters: {n_clusters_db}")
    print(f"    - Noise / Outlier Points: {n_noise} ({n_noise / len(X_sample):.1%})")

    if n_clusters_db > 1:
        valid_mask = db_labels != -1
        if valid_mask.sum() > 0 and len(set(db_labels[valid_mask])) > 1:
            db_sil = silhouette_score(X_sample[valid_mask], db_labels[valid_mask])
            print(f"    - DBSCAN Silhouette Score (excluding noise): {db_sil:.4f}")

    # -------------------------------------------------------------------------
    # 10. SAVE RESULTS
    # -------------------------------------------------------------------------
    output_csv = os.path.join(base_dir, "unsupervised_clustered_students.csv")
    output_cols = ["StudentID"] if "StudentID" in df.columns else []
    output_cols += clustering_features + ["KMeans_Cluster"]
    if "PlacementStatus" in df.columns:
        output_cols.append("PlacementStatus")
    if "Salary Package" in df.columns:
        output_cols.append("Salary Package")

    df[output_cols].to_csv(output_csv, index=False)
    print(f"\n[10] Exported clustered dataset to '{output_csv}'")

    print("\n" + "=" * 80)
    print(" [OK] UNSUPERVISED LEARNING PIPELINE COMPLETED SUCCESSFULLY!")
    print(" Generated Artifacts:")
    print(f"   1. {eval_plot_path}")
    print(f"   2. {pca_plot_path}")
    print(f"   3. {dendro_plot_path}")
    print(f"   4. {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
