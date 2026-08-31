# ============================================================
# PLACEMENT DATASET - SCALING + ENCODING COMPLETE CODE
# ============================================================

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# IMPORTS
# -----------------------------
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    TargetEncoder
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone


# ============================================================
# SETTINGS
# ============================================================

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
np.set_printoptions(suppress=True, precision=4)

DATA = "placement_predict_50k Dataset.csv"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATA)

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# NUMERICAL COLUMNS
# ============================================================

NUM = [
    "CGPA",
    "AttendancePercent",
    "AptitudeTestScore",
    "CodingTestScore",
    "Salary Package"
]

print("\n" + "=" * 70)
print("NUMERICAL COLUMN STATISTICS")
print("=" * 70)

print(
    df[NUM]
    .agg(["min", "max", "mean", "std"])
    .T
    .round(3)
)


# ============================================================
# 1. EUCLIDEAN DISTANCE
# ============================================================

print("\n" + "=" * 70)
print("EUCLIDEAN DISTANCE - STUDENT 1 vs STUDENT 2")
print("=" * 70)

a = df.loc[0, ["CGPA", "AttendancePercent"]].values.astype(float)
b = df.loc[1, ["CGPA", "AttendancePercent"]].values.astype(float)

contrib = (a - b) ** 2

print("A =", a)
print("B =", b)

print("(CGPA diff)^2   =", round(contrib[0], 4))
print("(Attend diff)^2 =", round(contrib[1], 4))

euclidean = np.sqrt(contrib.sum())

print("euclidean       =", round(euclidean, 4))

if contrib.sum() != 0:
    print(
        "attendance share of the distance =",
        f"{100 * contrib[1] / contrib.sum():.2f}%"
    )


# ============================================================
# 2. STANDARDIZATION - MANUAL CALCULATION
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZATION - MANUAL CALCULATION")
print("=" * 70)

cg = df["CGPA"]

mu = cg.mean()
sd_pop = cg.std(ddof=0)
sd_samp = cg.std(ddof=1)

print(f"n     = {len(cg)}")
print(f"mu    = {mu:.6f}")
print(f"sigma = {sd_pop:.6f}   (ddof=0, sklearn)")
print(f"sigma = {sd_samp:.6f}   (ddof=1, pandas)")


# First five students
hand = df.loc[:4, ["StudentID", "CGPA"]].copy()

hand["z = (x-mu)/sigma"] = (
    (hand["CGPA"] - mu) / sd_pop
).round(4)

print("\nFirst 5 manual calculations:")
print(hand.to_string(index=False))


# ============================================================
# 3. STANDARD SCALER
# ============================================================

print("\n" + "=" * 70)
print("STANDARD SCALER")
print("=" * 70)

scaler = StandardScaler()

scaler.fit(df[["CGPA"]])

z = scaler.transform(df[["CGPA"]])

print("mean_ :", scaler.mean_)
print("scale_:", scaler.scale_)
print("var_  :", scaler.var_)

print("first 5:", z[:5].ravel())

print(f"after -> mean = {z.mean():.10f}")
print(f"after -> std  = {z.std():.10f}")

print(
    f"range -> min = {z.min():.4f}, "
    f"max = {z.max():.4f}"
)


# ============================================================
# 4. STANDARDIZE ALL FIVE NUMERICAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STANDARDIZATION OF ALL NUMERICAL COLUMNS")
print("=" * 70)

X = df[NUM].fillna(df[NUM].median())

standard_scaler = StandardScaler()

Z = standard_scaler.fit_transform(X)

print(
    pd.DataFrame(
        Z[:5],
        columns=NUM
    ).round(4).to_string(index=False)
)

print("\nMeans after:")
print(np.round(Z.mean(0), 12))

print("\nStandard deviations after:")
print(np.round(Z.std(0), 6))


# ============================================================
# 5. MIN-MAX SCALING MANUAL
# ============================================================

print("\n" + "=" * 70)
print("MIN-MAX SCALING - MANUAL CALCULATION")
print("=" * 70)

xmin = cg.min()
xmax = cg.max()

print(f"min = {xmin}")
print(f"max = {xmax}")
print(f"range = {xmax - xmin}")

hand = df.loc[:4, ["StudentID", "CGPA"]].copy()

hand["x_scaled"] = (
    (hand["CGPA"] - xmin) /
    (xmax - xmin)
).round(4)

print("\nFirst 5:")
print(hand.to_string(index=False))


# ============================================================
# 6. MIN-MAX SCALER
# ============================================================

print("\n" + "=" * 70)
print("MIN-MAX SCALER")
print("=" * 70)

mm = MinMaxScaler().fit(df[["CGPA"]])

m = mm.transform(df[["CGPA"]])

print("data_min_ :", mm.data_min_)
print("data_max_ :", mm.data_max_)
print("scale_    :", mm.scale_)
print("min_      :", mm.min_)

print("first 5   :", m[:5].ravel())

print(f"min  = {m.min():.4f}")
print(f"max  = {m.max():.4f}")
print(f"mean = {m.mean():.4f}")


# ============================================================
# 7. MIN-MAX (-1,1)
# ============================================================

m2 = MinMaxScaler(
    feature_range=(-1, 1)
).fit_transform(df[["CGPA"]])

print("\nfeature_range=(-1,1)")
print("first 5:", m2[:5].ravel())


# ============================================================
# 8. OUTLIER EFFECT ON SCALERS
# ============================================================

print("\n" + "=" * 70)
print("OUTLIER EFFECT ON SCALERS")
print("=" * 70)

s = df["Salary Package"].copy()

# Artificial typo/outlier
s2 = s.copy()
s2.iloc[7] = 2600.0

X2 = s2.values.reshape(-1, 1)

ok = np.ones(len(s), bool)
ok[7] = False

rows = []

for name, Sc in [
    ("MinMaxScaler", MinMaxScaler),
    ("StandardScaler", StandardScaler),
    ("RobustScaler", RobustScaler)
]:

    clean = Sc().fit_transform(
        s.values.reshape(-1, 1)
    ).ravel()

    dirty = Sc().fit_transform(
        X2
    ).ravel()

    rows.append([
        name,
        round(
            clean[ok].max() -
            clean[ok].min(), 4
        ),
        round(
            dirty[ok].max() -
            dirty[ok].min(), 4
        )
    ])

t = pd.DataFrame(
    rows,
    columns=[
        "scaler",
        "clean width",
        "width after the typo"
    ]
)

t["% of range kept"] = (
    100 *
    t["width after the typo"] /
    t["clean width"]
).round(1)

print(t.to_string(index=False))


# ============================================================
# 9. ROBUST SCALING MANUAL
# ============================================================

print("\n" + "=" * 70)
print("ROBUST SCALING - MANUAL CALCULATION")
print("=" * 70)

X_coding = df[
    ["CodingTestScore"]
].fillna(
    df["CodingTestScore"].median()
)

q1, q2, q3 = np.percentile(
    X_coding.values,
    [25, 50, 75]
)

print(f"Q1     = {q1}")
print(f"median = {q2}")
print(f"Q3     = {q3}")
print(f"IQR    = {q3 - q1:.4f}")


hand = df.loc[
    :4,
    ["StudentID", "CodingTestScore"]
].copy()

hand["robust"] = (
    (hand["CodingTestScore"] - q2) /
    (q3 - q1)
).round(4)

print("\nFirst 5:")
print(hand.to_string(index=False))


# ============================================================
# 10. ROBUST SCALER
# ============================================================

print("\n" + "=" * 70)
print("ROBUST SCALER")
print("=" * 70)

rb = RobustScaler().fit(X_coding)

r = rb.transform(X_coding)

print("center_ (median):", rb.center_)
print("scale_  (IQR)   :", rb.scale_)

print("first 5         :", r[:5].ravel())

print(
    f"median after = {np.median(r):.6f}"
)

print(
    f"IQR after = "
    f"{np.percentile(r, 75) - np.percentile(r, 25):.6f}"
)


# ============================================================
# 11. HEAD-TO-HEAD SCALER COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("SCALER COMPARISON - CODING TEST SCORE")
print("=" * 70)

cmp = pd.DataFrame({
    "raw":
        X_coding.values.ravel(),

    "standard":
        StandardScaler()
        .fit_transform(X_coding)
        .ravel(),

    "minmax":
        MinMaxScaler()
        .fit_transform(X_coding)
        .ravel(),

    "robust":
        RobustScaler()
        .fit_transform(X_coding)
        .ravel(),

    "maxabs":
        MaxAbsScaler()
        .fit_transform(X_coding)
        .ravel()
})

print(
    cmp.describe()
    .T
    .round(4)
)


# ============================================================
# 12. MAX ABS SCALER
# ============================================================

print("\n" + "=" * 70)
print("MAX ABS SCALER - SALARY PACKAGE")
print("=" * 70)

ma = MaxAbsScaler().fit(
    df[["Salary Package"]]
)

print("max_abs_:", ma.max_abs_)

print(
    "raw    :",
    df["Salary Package"].head().values
)

print(
    "scaled :",
    ma.transform(
        df[["Salary Package"]]
    )[:5].ravel()
)


# ============================================================
#                 ENCODING MODELS
# ============================================================

print("\n\n")
print("#" * 70)
print("ENCODING MODELS")
print("#" * 70)


# ============================================================
# CATEGORICAL COLUMNS
# ============================================================

CAT = [
    "Gender",
    "City",
    "CollegeTier",
    "Stream",
    "Specialisation",
    "Hostel",
    "Degree",
    "WorkExperience"
]

# Use only columns that actually exist
CAT = [
    col for col in CAT
    if col in df.columns
]

print("\nCategorical columns found:")
print(CAT)


# ============================================================
# 13. LABEL ENCODING
# ============================================================

print("\n" + "=" * 70)
print("LABEL ENCODING")
print("=" * 70)

le = LabelEncoder().fit(
    df["City"]
)

print("classes_ (alphabetical):")

for i, cat in enumerate(le.classes_):
    print(
        f"  {cat:<15} -> {i}"
    )


S = df.loc[
    :7,
    ["StudentID", "Gender", "City"]
].copy()

S["Gender_label"] = (
    LabelEncoder()
    .fit_transform(df["Gender"])
    [:8]
)

S["City_label"] = (
    le.transform(S["City"])
)

print("\nFirst 8 encoded rows:")
print(S.to_string(index=False))


# ============================================================
# 14. WHY LABEL ENCODING CAN BE DANGEROUS
# ============================================================

print("\n" + "=" * 70)
print("LABEL ENCODING - THE TRAP")
print("=" * 70)

print(
    "(Ahmedabad + Pune) / 2 =",
    (0 + 9) / 2,
    "= 'Hyderabad-ish' -> meaningless"
)

print(
    "Pune - Delhi =",
    9 - 3,
    "'units of city' -> meaningless"
)

print(
    "\nA linear model or distance-based model "
    "may incorrectly use these numerical relationships."
)


# ============================================================
# 15. ORDINAL ENCODING
# ============================================================

print("\n" + "=" * 70)
print("ORDINAL ENCODING")
print("=" * 70)

if "CGPA_Tier" in df.columns:

    oe = OrdinalEncoder(
        categories=[
            ["Low", "Mid", "High"]
        ]
    ).fit(
        df[["CGPA_Tier"]]
    )

else:
    oe = None


if "CollegeTier" in df.columns:

    oe2 = OrdinalEncoder(
        categories=[
            ["Tier3", "Tier2", "Tier1"]
        ]
    ).fit(
        df[["CollegeTier"]]
    )

else:
    oe2 = None


cols_needed = [
    "StudentID",
    "CGPA_Tier",
    "CollegeTier"
]

cols_existing = [
    c for c in cols_needed
    if c in df.columns
]

S = df.loc[
    :7,
    cols_existing
].copy()


if oe is not None:
    S["CGPA_Tier_ord"] = (
        oe.transform(
            df.loc[:7, ["CGPA_Tier"]]
        )
        .astype(int)
    )


if oe2 is not None:
    S["CollegeTier_ord"] = (
        oe2.transform(
            df.loc[:7, ["CollegeTier"]]
        )
        .astype(int)
    )


print(S.to_string(index=False))


# ============================================================
# 16. CHECK ORDINAL ORDER AGAINST TARGET
# ============================================================

print("\n" + "=" * 70)
print("CHECK ORDINAL ORDER AGAINST PLACEMENT TARGET")
print("=" * 70)

if (
    "CGPA_Tier" in df.columns
    and "PlacementStatus" in df.columns
):

    print("\nCGPA Tier:")
    print(
        df.groupby(
            "CGPA_Tier",
            observed=True
        )["PlacementStatus"]
        .agg(["count", "mean"])
        .reindex(
            ["Low", "Mid", "High"]
        )
        .round(4)
    )


if (
    "CollegeTier" in df.columns
    and "PlacementStatus" in df.columns
):

    print("\nCollege Tier:")
    print(
        df.groupby(
            "CollegeTier",
            observed=True
        )["PlacementStatus"]
        .agg(["count", "mean"])
        .reindex(
            ["Tier3", "Tier2", "Tier1"]
        )
        .round(4)
    )


# ============================================================
# 17. ONE-HOT ENCODING
# ============================================================

print("\n" + "=" * 70)
print("ONE-HOT ENCODING - CITY")
print("=" * 70)

# Compatibility with older/newer sklearn versions
try:
    ohe = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    ).fit(
        df[["City"]]
    )
except TypeError:
    ohe = OneHotEncoder(
        sparse=False,
        handle_unknown="ignore"
    ).fit(
        df[["City"]]
    )


oh = ohe.transform(
    df.loc[:7, ["City"]]
)

feature_names = [
    c.replace("City_", "")
    for c in ohe.get_feature_names_out()
]

out = pd.DataFrame(
    oh.astype(int),
    columns=feature_names
)

out.insert(
    0,
    "City",
    df.loc[:7, "City"].values
)

print(
    out.to_string(index=False)
)

print(
    "\nrow sums (always 1):",
    oh.sum(1)
)


# ============================================================
# 18. DUMMY VARIABLE TRAP
# ============================================================

print("\n" + "=" * 70)
print("DUMMY VARIABLE TRAP")
print("=" * 70)

try:
    ohd = OneHotEncoder(
        sparse_output=False,
        drop="first"
    ).fit(
        df[["City"]]
    )
except TypeError:
    ohd = OneHotEncoder(
        sparse=False,
        drop="first"
    ).fit(
        df[["City"]]
    )

print(
    "drop='first' ->",
    ohd.transform(
        df[["City"]]
    ).shape[1],
    "columns"
)

print(
    "reference category =",
    ohe.categories_[0][0],
    "(now all-zeros)"
)


# ============================================================
# 19. ONE-HOT ENCODE ALL CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("ONE-HOT ENCODING ALL CATEGORICAL COLUMNS")
print("=" * 70)

try:
    full_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    full = full_encoder.fit_transform(
        df[CAT]
    )

except TypeError:

    full_encoder = OneHotEncoder(
        sparse=False,
        handle_unknown="ignore"
    )

    full = full_encoder.fit_transform(
        df[CAT]
    )


print(
    "\nTOTAL one-hot columns from categorical columns:",
    full.shape[1]
)

print(
    f"dense float64 memory : "
    f"{full.nbytes / 1e6:.2f} MB"
)


# Sparse version
try:
    sparse_encoder = OneHotEncoder(
        handle_unknown="ignore"
    )

    sparse_full = sparse_encoder.fit_transform(
        df[CAT]
    )

    print(
        f"sparse CSR data      : "
        f"{sparse_full.data.nbytes / 1e6:.2f} MB"
    )

except Exception as e:
    print("Sparse calculation skipped:", e)


# ============================================================
# 20. BINARY ENCODING
# ============================================================

print("\n" + "=" * 70)
print("BINARY ENCODING")
print("=" * 70)


def binary_encode(series, name):

    cats = sorted(
        series.dropna().unique()
    )

    # Map categories to integers
    mapping = {
        c: i + 1
        for i, c in enumerate(cats)
    }

    # Number of binary bits required
    nbits = int(
        np.ceil(
            np.log2(
                len(cats) + 1
            )
        )
    )

    codes = (
        series
        .map(mapping)
        .fillna(0)
        .to_numpy(dtype=int)
    )

    cols = {}

    for b in range(nbits):

        cols[
            f"{name}_{nbits - b}"
        ] = (
            codes >>
            (nbits - 1 - b)
        ) & 1

    return (
        pd.DataFrame(cols),
        mapping,
        nbits
    )


bdf, bmap, nbits = binary_encode(
    df["City"],
    "City"
)

print(
    f"{len(bmap)} cities -> "
    f"ceil(log2({len(bmap)}+1)) = "
    f"{nbits} columns"
)

print(
    f"One-hot needed = {len(bmap)} columns"
)


# ============================================================
# 21. BINARY ENCODING TABLE
# ============================================================

tbl = pd.DataFrame({
    "City": list(bmap.keys()),
    "ordinal": list(bmap.values())
})

tbl["binary"] = tbl[
    "ordinal"
].apply(
    lambda v:
        format(
            v,
            f"0{nbits}b"
        )
)


for b in range(nbits):

    tbl[
        f"City_{nbits-b}"
    ] = (
        tbl["binary"]
        .str[b]
        .astype(int)
    )


print("\nBinary encoding table:")
print(
    tbl.to_string(index=False)
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n")
print("#" * 70)
print("FINAL SUMMARY")
print("#" * 70)

print("\nDataset shape:", df.shape)

print("\nNumerical columns:")
print(NUM)

print("\nCategorical columns:")
print(CAT)

print("\nScaling techniques completed:")
print("1. StandardScaler")
print("2. MinMaxScaler")
print("3. RobustScaler")
print("4. MaxAbsScaler")

print("\nEncoding techniques completed:")
print("1. Label Encoding")
print("2. Ordinal Encoding")
print("3. One-Hot Encoding")
print("4. Binary Encoding")

print("\nEverything completed successfully.")
print("#" * 70)