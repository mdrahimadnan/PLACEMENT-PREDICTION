"""
=============================================================
  Placement Prediction -- Logistic Regression Demo
  Dataset : placement_predict_50k Dataset.csv
  Model   : Logistic Regression (single feature: CGPA)
=============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------
# STEP 1 -- Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("placement_predict_50k Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# STEP 2 -- Select Input Feature & Target
# ---------------------------------------------------------

# Input feature (single feature model)
X = df[["CGPA"]]

# Target variable
# NOTE: PlacementStatus is already stored as integers (0 / 1)
#   0 = Not Placed
#   1 = Placed
y = df["PlacementStatus"]

print("\nInput X (first 5 rows):")
print(X.head())

print("\nTarget y (first 5 rows):")
print(y.head())

# ---------------------------------------------------------
# STEP 3 -- Explore the Target Variable
# ---------------------------------------------------------
print("\nValue counts:")
print(y.value_counts())

print("\nUnique values:", y.unique())

# ---------------------------------------------------------
# STEP 4 -- No Mapping Needed (already 0 / 1)
# ---------------------------------------------------------
# If your dataset had STRING labels like "Placed" / "Not Placed",
# you would convert them like this:
#
#   y = y.map({"Not Placed": 0, "Placed": 1})
#
# But in THIS dataset PlacementStatus is ALREADY numeric (0/1),
# so we use it directly -- no mapping needed.

print("\nClass distribution:")
print(y.value_counts())
print("  0 = Not Placed  |  1 = Placed")

# ---------------------------------------------------------
# STEP 5 -- Scatter Plot: CGPA vs Placement Status
# ---------------------------------------------------------
sample   = df.sample(2000, random_state=42)
X_sample = sample["CGPA"]
y_sample = sample["PlacementStatus"]

plt.figure(figsize=(8, 5))
plt.scatter(X_sample, y_sample, alpha=0.3, color="#8b5cf6")
plt.xlabel("CGPA")
plt.ylabel("Placement Status")
plt.title("CGPA vs Placement Status")
plt.yticks([0, 1], ["Not Placed", "Placed"])
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("cgpa_vs_placement_scatter.png", dpi=130, bbox_inches="tight")
plt.show()
print("Scatter plot saved as: cgpa_vs_placement_scatter.png")

# ---------------------------------------------------------
# STEP 6 -- Train / Test Split  (80% / 20%)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining samples:", len(X_train))
print("Testing  samples:", len(X_test))

# ---------------------------------------------------------
# STEP 7 -- Train Logistic Regression Model
# ---------------------------------------------------------
model = LogisticRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)

print("\nModel trained!")
print("  Coefficient (slope) :", round(model.coef_[0][0], 6))
print("  Intercept            :", round(model.intercept_[0], 6))
print(
    "\n  Equation: logit(P) = {:.4f} x CGPA + ({:.4f})".format(
        model.coef_[0][0], model.intercept_[0]
    )
)

# ---------------------------------------------------------
# STEP 8 -- Predict Probabilities
# ---------------------------------------------------------
probabilities = model.predict_proba(X_test.values.reshape(-1, 1))

print("\npredict_proba() -- First 10 rows:")
print("  [P(Not Placed=0)   P(Placed=1)]")
for i, row in enumerate(probabilities[:10]):
    print("  Row {:>2}: [P=0: {:.4f}   P=1: {:.4f}]".format(i + 1, row[0], row[1]))

# ---------------------------------------------------------
# BONUS -- Full Model Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test.values.reshape(-1, 1))

print("\n" + "=" * 55)
print("Model Performance on Test Set")
print("  Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Placed", "Placed"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(
    "\n  [TN={:<5}  FP={:<5}]\n  [FN={:<5}  TP={:<5}]".format(
        cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    )
)
