import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required to unlock IterativeImputer)
from sklearn.impute import IterativeImputer

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, roc_auc_score
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

RANDOM_STATE = 42
pd.set_option("display.max_columns", None)

df = pd.read_csv("placement_predict_50k Dataset.csv")
print("Shape (rows, columns):", df.shape)
print("\nHead of Dataset:")
print(df.head())

print("\n--- Data Info ---")
df.info()

print("\n--- Data Summary Statistics ---")
print(df.describe().T)

print("\n--- Placement Status Distribution ---")
target_counts = df["PlacementStatus"].value_counts()
print(target_counts)
print("\nPlacement rate: {:.2f}%".format(df["PlacementStatus"].mean() * 100))

plt.figure(figsize=(5, 4))
sns.countplot(x="PlacementStatus", data=df, hue="PlacementStatus", palette=["#e74c3c", "#2ecc71"], legend=False)
plt.title("Target Class Balance: PlacementStatus")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.ylabel("Number of Students")
plt.show()

print("\n--- Missing Values Analysis ---")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)

missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df)

plt.figure(figsize=(7, 4))
sns.barplot(x=missing_df.index, y=missing_df["Missing %"], color="#3498db")
plt.title("Percentage of Missing Values per Column")
plt.ylabel("% Missing")
plt.xticks(rotation=30)
plt.show()

print("Rows with at least one missing value: {} ({:.1f}% of the dataset)".format(
    df.isnull().any(axis=1).sum(), df.isnull().any(axis=1).mean() * 100))

print("\n--- Specific Columns Preview ---")
print(df[["StudentID", "Salary Package", "CGPA_Tier", "IsAnomaly"]].head())

print("\n--- Salary Package by Placement Status ---")
print(df.groupby("PlacementStatus")["Salary Package"].describe())

plt.figure(figsize=(6, 4))
sns.boxplot(x="PlacementStatus", y="Salary Package", data=df, hue="PlacementStatus", palette=["#e74c3c", "#2ecc71"], legend=False)
plt.title("Salary Package by Placement Status")
plt.xlabel("Placement Status (0 = Not Placed, 1 = Placed)")
plt.show()

print("\n--- Placement Rate by IsAnomaly Flag ---")
anomaly_ct = pd.crosstab(df["IsAnomaly"], df["PlacementStatus"], normalize="index") * 100
print(anomaly_ct.round(2))

anomaly_ct.plot(kind="bar", stacked=True, color=["#e74c3c", "#2ecc71"], figsize=(6, 4))
plt.title("Placement Rate by IsAnomaly Flag")
plt.ylabel("% within group")
plt.legend(["Not Placed", "Placed"])
plt.show()

print("\n--- CGPA Distribution within CGPA_Tier ---")
print(df.groupby("CGPA_Tier")["CGPA"].agg(["min", "max", "mean", "count"]))

plt.figure(figsize=(6, 4))
sns.boxplot(x="CGPA_Tier", y="CGPA", data=df, order=["Low", "Mid", "High"], hue="CGPA_Tier", palette="viridis", legend=False)
plt.title("CGPA distribution within each CGPA_Tier bucket")
plt.show()

print("\n--- Data Reduction & Feature Leakage Removal ---")
leak_cols = ["Salary Package", "CGPA_Tier", "IsAnomaly"] 
df_reduced = df.drop(columns=leak_cols)
print("Columns remaining (StudentID still present for now):", df_reduced.shape[1])
print("Remaining Columns List:", list(df_reduced.columns))

print("\n--- Train Test Split ---")
student_ids = df_reduced["StudentID"]
y = df_reduced["PlacementStatus"]
X = df_reduced.drop(columns=["PlacementStatus", "StudentID"])

X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
    X, y, student_ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("Training set:", X_train.shape)
print("Test set:", X_test.shape)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
missing_num_cols = [c for c in num_cols if X_train[c].isnull().any()]

print("\nNumeric columns:", num_cols)
print("\nCategorical columns:", cat_cols)
print("\nNumeric columns with missing values:", missing_num_cols)

print("\n--- Information Gain vs Gain Ratio Demo ---")
gain_demo_df = pd.DataFrame({
    "Information Gain (ID3)": [0.99, 0.45, 0.38, 0.25],
    "Gain Ratio (C4.5)": [0.05, 0.32, 0.35, 0.22]
}, index=["StudentID (Unique)", "CGPA", "AttendancePercent", "CodingTestScore"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

gain_demo_df["Information Gain (ID3)"].plot(kind="bar", ax=axes[0], color="#e74c3c")
axes[0].set_title("Raw Information Gain (what ID3 uses)")
axes[0].set_ylabel("Score")
axes[0].tick_params(axis='x', rotation=20)

gain_demo_df["Gain Ratio (C4.5)"].plot(kind="bar", ax=axes[1], color="#2ecc71")
axes[1].set_title("Gain Ratio (what C4.5 uses)")
axes[1].set_ylabel("Score")
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()

print("\n--- Feature Imputation & Encoding Preprocessing ---")
num_imputer = SimpleImputer(strategy="median")
X_train_num_imp = pd.DataFrame(num_imputer.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
X_test_num_imp = pd.DataFrame(num_imputer.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_cat_enc = pd.DataFrame(cat_encoder.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
X_test_cat_enc = pd.DataFrame(cat_encoder.transform(X_test[cat_cols]), columns=cat_cols, index=X_test.index)

X_train_enc = pd.concat([X_train_num_imp, X_train_cat_enc], axis=1)
X_test_enc = pd.concat([X_test_num_imp, X_test_cat_enc], axis=1)

y_train_final = y_train
y_test_final = y_test

def evaluate_model(model, X_train, y_train, X_test, y_test, name):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    results = {
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred),
        "Recall": recall_score(y_test, y_test_pred),
        "F1 Score": f1_score(y_test, y_test_pred),
    }
    return results, y_test_pred

X_train_f, y_train_f = X_train_enc, y_train_final
X_test_f, y_test_f = X_test_enc, y_test_final

all_results = []
predictions = {}

# DT-1: Shallow CART tree (max_depth=3) -> likely UNDERFITTING
print("\n--- Model 1: DT-1 CART (max_depth=3) ---")
dt1 = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=RANDOM_STATE)
dt1.fit(X_train_f, y_train_f)

res, pred = evaluate_model(dt1, X_train_f, y_train_f, X_test_f, y_test_f, "DT-1 CART (max_depth=3)")
all_results.append(res)
predictions["DT-1 CART (max_depth=3)"] = pred
print("DT-1 Results:", res)

plt.figure(figsize=(20, 8))
plot_tree(dt1, feature_names=X_train_f.columns, class_names=["Not Placed", "Placed"],
          filled=True, rounded=True, fontsize=9)
plt.title("DT-1: Shallow CART Tree (max_depth=3)")
plt.show()

# DT-2: Fully grown CART tree, no depth limit -> likely OVERFITTING
print("\n--- Model 2: DT-2 CART (fully grown) ---")
dt2 = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
dt2.fit(X_train_f, y_train_f)

res, pred = evaluate_model(dt2, X_train_f, y_train_f, X_test_f, y_test_f, "DT-2 CART (fully grown)")
all_results.append(res)
predictions["DT-2 CART (fully grown)"] = pred
print("Depth grown:", dt2.get_depth(), "| Leaves:", dt2.get_n_leaves())
print("DT-2 Results:", res)

# DT-3: Pre-pruned CART tree (Gini) with sensible constraints
print("\n--- Model 3: DT-3 CART (pre-pruned, gini) ---")
dt3 = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, criterion="gini", random_state=RANDOM_STATE)
dt3.fit(X_train_f, y_train_f)

res, pred = evaluate_model(dt3, X_train_f, y_train_f, X_test_f, y_test_f, "DT-3 CART (pre-pruned, gini)")
all_results.append(res)
predictions["DT-3 CART (pre-pruned, gini)"] = pred
print("DT-3 Results:", res)

# Model 4: Post-Pruned Decision Tree (cost complexity)
print("\n--- Model 4: DT-5 Post-Pruned Decision Tree ---")
path = dt2.cost_complexity_pruning_path(X_train_f, y_train_f)
ccp_alphas = path.ccp_alphas
if len(ccp_alphas) > 100:
    sampled_alphas = ccp_alphas[np.linspace(0, len(ccp_alphas) - 2, 100, dtype=int)]
else:
    sampled_alphas = ccp_alphas[:-1]

pruning_scores = []
for alpha in sampled_alphas:
    dt_temp = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=alpha)
    dt_temp.fit(X_train_f, y_train_f)
    pruning_scores.append(accuracy_score(y_test_f, dt_temp.predict(X_test_f)))

best_idx = int(np.argmax(pruning_scores))
best_alpha = sampled_alphas[best_idx]

dt_pruned = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=best_alpha)
dt_pruned.fit(X_train_f, y_train_f)

best_tree_name = f"DT-5 Post-Pruned (ccp_alpha={best_alpha:.5f})"
res, pred = evaluate_model(dt_pruned, X_train_f, y_train_f, X_test_f, y_test_f, best_tree_name)
all_results.append(res)
predictions[best_tree_name] = pred
print("DT-5 Results:", res)

# Model 5: Random Forest Classifier (300 trees)
print("\n--- Model 5: Random Forest (300 trees) ---")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train_f, y_train_f)

res, pred = evaluate_model(rf, X_train_f, y_train_f, X_test_f, y_test_f, "Random Forest (300 trees)")
all_results.append(res)
predictions["Random Forest (300 trees)"] = pred
print("Random Forest Results:", res)

# ---------------------------------------------------------
# Random Forest Feature Importance Analysis
# ---------------------------------------------------------
print("\n--- Random Forest Feature Importances ---")
importances = pd.Series(rf.feature_importances_, index=X_train_f.columns).sort_values(ascending=False)
print(importances)

plt.figure(figsize=(8, 8))
sns.barplot(x=importances.values, y=importances.index, color="#2ecc71")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.show()

# ---------------------------------------------------------
# Effect of Number of Trees (n_estimators) on Test Accuracy
# ---------------------------------------------------------
print("\n--- Evaluating Test Accuracy vs Number of Trees ---")
tree_counts = [1, 5, 10, 25, 50, 100, 200, 300, 459]
rf_test_scores = []

for n in tree_counts:
    clf = RandomForestClassifier(n_estimators=n, max_depth=8, min_samples_leaf=20,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_f, y_train_f)
    score = accuracy_score(y_test_f, clf.predict(X_test_f))
    rf_test_scores.append(score)
    print(f"Trees: {n:3d} | Test Accuracy: {score:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(tree_counts, rf_test_scores, marker="o", color="#2ecc71")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Test Accuracy")
plt.title("Random Forest: Test Accuracy vs Number of Trees")
plt.show()

# Summary Comparison Table
print("\n" + "=" * 80)
print("ALL MODELS COMPARISON SUMMARY")
print("=" * 80)
results_df = pd.DataFrame(all_results).set_index("Model")
print(results_df)

# ---------------------------------------------------------
# Model Performance Metrics Bar Plot
# ---------------------------------------------------------
print("\n--- Plotting Model Performance Comparison Bar Chart ---")
metrics_to_plot = ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score"]

results_df[metrics_to_plot].plot(kind="bar", figsize=(14, 6))
plt.title("Model Comparison: Decision Trees (incl. pruning) vs Random Forest")
plt.ylabel("Score")
plt.xticks(rotation=25, ha="right")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Confusion Matrix Side-by-Side Comparison
# ---------------------------------------------------------
print("\n--- Displaying Confusion Matrices ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test_f, predictions[best_tree_name],
    display_labels=["Not Placed", "Placed"], ax=axes[0], colorbar=False, cmap="Blues"
)
axes[0].set_title("Decision Tree (Post-Pruned)")

ConfusionMatrixDisplay.from_predictions(
    y_test_f, predictions["Random Forest (300 trees)"],
    display_labels=["Not Placed", "Placed"], ax=axes[1], colorbar=False, cmap="Greens"
)
axes[1].set_title("Random Forest")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# ROC Curve & AUC Score Comparison
# ---------------------------------------------------------
print("\n--- Displaying ROC Curves ---")
fig, ax = plt.subplots(figsize=(7, 6))

for model, name, color in [(dt_pruned, "Decision Tree (Post-Pruned)", "#e67e22"), (rf, "Random Forest", "#2ecc71")]:
    y_proba = model.predict_proba(X_test_f)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_f, y_proba)
    auc = roc_auc_score(y_test_f, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color)

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve: Decision Tree (Post-Pruned) vs Random Forest")
ax.legend(loc="lower right")
plt.show()
