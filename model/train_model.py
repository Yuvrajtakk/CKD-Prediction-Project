
import os
import sys
import numpy as np
import pandas as pd
import joblib  # ✅ Use joblib, NOT pickle — safer across sklearn versions

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "chronickidneydisease.csv")
MODEL_PATH = os.path.join(BASE_DIR, "pipeline.pkl")

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df.replace("?", np.nan, inplace=True)

if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# ── 2. Rename columns (handles datasets with short UCI names) ──────────────────
# The CSV uses abbreviated column names. Map to full names for clarity.
COLUMN_MAP = {
    "rbc": "rbc",
    "pc": "pc",
    "bgr": "bgr",
    "bu": "bu",
    "pe": "pe",
    "ane": "ane",
    "dm": "dm",
    "cad": "cad",
    "classification": "classification",
}

# ── 3. Fix known dirty values in categorical columns ─────────────────────────
# The UCI CKD dataset has tab/space-prefixed values in dm and cad
for col in ["dm", "cad", "classification"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# classification column: 'ckd' → 1, 'notckd'/'not ckd' → 0
df["classification"] = df["classification"].apply(
    lambda x: 1 if "notckd" not in x and x.startswith("ckd") else 0
)

# ── 4. Feature selection ───────────────────────────────────────────────────────
# These 8 clinically meaningful features are used (same as original project)
SELECTED_FEATURES = ["rbc", "pc", "bgr", "bu", "pe", "ane", "dm", "cad"]

# Verify all columns exist
missing = [c for c in SELECTED_FEATURES if c not in df.columns]
if missing:
    print(f"❌ Missing columns in dataset: {missing}")
    print(f"   Available columns: {list(df.columns)}")
    sys.exit(1)

X = df[SELECTED_FEATURES].copy()
y = df["classification"].copy()

print(f"   Dataset shape: {df.shape}")
print(f"   Class distribution:\n{y.value_counts()}\n")

# ── 5. Identify column types AFTER loading ────────────────────────────────────
# bgr and bu are numeric; rbc, pc, pe, ane, dm, cad are categorical
NUM_COLS = ["bgr", "bu"]
CAT_COLS = ["rbc", "pc", "pe", "ane", "dm", "cad"]

# ── 6. Build preprocessing pipelines ─────────────────────────────────────────
#
# ✅ KEY FIX: Use OrdinalEncoder instead of OneHotEncoder.
#    OrdinalEncoder avoids the _RemainderColsList internals that caused the
#    version-mismatch crash. It's also simpler (no sparse matrices) and works
#    perfectly for binary categorical features like yes/no, normal/abnormal.
#
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,   # unknown categories → -1 at inference time
        ),
    ),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, NUM_COLS),
        ("cat", cat_pipeline, CAT_COLS),
    ],
    remainder="drop",   # ✅ Explicit 'drop' avoids _RemainderColsList entirely
)

# ── 7. Full pipeline ──────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",  # handles class imbalance gracefully
        ),
    ),
])

# ── 8. Train / evaluate ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🚀 Training pipeline...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {acc * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=["No CKD", "CKD"]))

# ── 9. Save with joblib ───────────────────────────────────────────────────────
# joblib is the sklearn-recommended serializer — it handles numpy arrays
# and sklearn internals far more robustly than pickle across versions.
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Pipeline saved → {MODEL_PATH}")
print("   Now commit this pipeline.pkl and push to GitHub.")
