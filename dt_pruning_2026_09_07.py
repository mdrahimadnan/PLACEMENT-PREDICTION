"""
Decision Tree Post-Pruning (Cost Complexity Pruning) Demo
File: dt_pruning_2026_09_07.py
Date: 2026-09-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

RANDOM_STATE = 42
pd.set_option("display.max_columns", None)

# 1. Load Dataset
df = pd.read_csv("placement_predict_50k Dataset.csv")
print("Dataset Shape:", df.shape)

# 2. Data Reduction & Leakage Removal
leak_cols = ["Salary Package", "CGPA_Tier", "IsAnomaly"]
df_reduced = df.drop(columns=leak_cols)

student_ids = df_reduced["StudentID"]
y = df_reduced["PlacementStatus"]
X = df_reduced.drop(columns=["PlacementStatus", "StudentID"])

# 3. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 4. Preprocessing: Imputation & Encoding
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()

num_imputer = SimpleImputer(strategy="median")
X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_cat = pd.DataFrame(cat_encoder.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
X_test_cat = pd.DataFrame(cat_encoder.transform(X_test[cat_cols]), columns=cat_cols, index=X_test.index)

X_train_f = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_f = pd.concat([X_test_num, X_test_cat], axis=1)

y_train_f = y_train
y_test_f = y_test

def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    y_train_pred = model.predict(X_tr)
    y_test_pred = model.predict(X_te)
    results = {
        "Model": name,
        "Train Accuracy": accuracy_score(y_tr, y_train_pred),
        "Test Accuracy": accuracy_score(y_te, y_test_pred),
        "Precision": precision_score(y_te, y_test_pred),
        "Recall": recall_score(y_te, y_test_pred),
        "F1 Score": f1_score(y_te, y_test_pred),
    }
    return results, y_test_pred

all_results = []
predictions = {}

# 5. Fit Fully Grown Decision Tree (DT-2)
print("\n--- Fitting Fully Grown Decision Tree (DT-2) ---")
dt2 = DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)
dt2.fit(X_train_f, y_train_f)
res2, pred2 = evaluate_model(dt2, X_train_f, y_train_f, X_test_f, y_test_f, "DT-2 CART (fully grown)")
print("Fully Grown Nodes:", dt2.tree_.node_count, "| Leaves:", dt2.get_n_leaves(), "| Depth:", dt2.get_depth())
print("Fully Grown Results:", res2)

# 6. Cost Complexity Pruning Path Calculation
print("\n--- Calculating Cost Complexity Pruning Path ---")
path = dt2.cost_complexity_pruning_path(X_train_f, y_train_f)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Sample alphas to speed up evaluation across grid
if len(ccp_alphas) > 100:
    sampled_indices = np.linspace(0, len(ccp_alphas) - 2, 100, dtype=int)
    sampled_alphas = ccp_alphas[sampled_indices]
else:
    sampled_alphas = ccp_alphas[:-1]

pruning_train_scores = []
pruning_test_scores = []

for alpha in sampled_alphas:
    dt_temp = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=alpha)
    dt_temp.fit(X_train_f, y_train_f)
    pruning_train_scores.append(accuracy_score(y_train_f, dt_temp.predict(X_train_f)))
    pruning_test_scores.append(accuracy_score(y_test_f, dt_temp.predict(X_test_f)))

# Plot Accuracy vs Alpha for training and test sets
plt.figure(figsize=(10, 5))
plt.plot(sampled_alphas, pruning_train_scores, marker='o', label="Train Accuracy", drawstyle="steps-post", color="#3498db")
plt.plot(sampled_alphas, pruning_test_scores, marker='o', label="Test Accuracy", drawstyle="steps-post", color="#e74c3c")
plt.xlabel("effective alpha (ccp_alpha)")
plt.ylabel("accuracy")
plt.title("Accuracy vs Alpha for Training and Testing Sets")
plt.legend()
plt.savefig("ccp_alpha_pruning_curve.png", bbox_inches="tight")
plt.show()

# 7. Select Best Alpha and Train DT-5 Post-Pruned
best_idx = int(np.argmax(pruning_test_scores))
best_alpha = sampled_alphas[best_idx]

dt_pruned = DecisionTreeClassifier(random_state=RANDOM_STATE, ccp_alpha=best_alpha)
dt_pruned.fit(X_train_f, y_train_f)

res, pred = evaluate_model(dt_pruned, X_train_f, y_train_f, X_test_f, y_test_f,
                            f"DT-5 Post-Pruned (ccp_alpha={best_alpha:.5f})")
all_results.append(res)
predictions[f"DT-5 Post-Pruned (ccp_alpha={best_alpha:.5f})"] = pred

print("\n" + "=" * 60)
print("POST-PRUNED DECISION TREE (DT-5) SUMMARY")
print("=" * 60)
print(f"Chosen alpha        : {best_alpha:.6f}")
print(f"Fully-grown nodes   : {dt2.tree_.node_count}")
print(f"Pruned tree nodes   : {dt_pruned.tree_.node_count}")
print(f"Pruned tree depth   : {dt_pruned.get_depth()}")
print(f"Pruned tree leaves  : {dt_pruned.get_n_leaves()}")
print("\nEvaluation Performance Metrics:")
for k, v in res.items():
    if isinstance(v, float):
        print(f"  {k:15s}: {v:.4f}")
    else:
        print(f"  {k:15s}: {v}")

# 8. Save and Display Visual Tree Plot
plt.figure(figsize=(22, 10))
plot_tree(
    dt_pruned, 
    feature_names=X_train_f.columns, 
    class_names=["Not Placed", "Placed"],
    filled=True, 
    rounded=True, 
    fontsize=8,
    max_depth=4
)
plt.title(f"DT-5: Post-Pruned Decision Tree (ccp_alpha={best_alpha:.5f}, depth={dt_pruned.get_depth()})")
plt.savefig("decision_tree_post_pruned.png", bbox_inches="tight")
plt.show()
print("\nPost-pruned tree plot saved as 'decision_tree_post_pruned.png'!")
