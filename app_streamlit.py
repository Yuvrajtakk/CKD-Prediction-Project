"""
app_streamlit.py — CKD Prediction Streamlit App
Trains the model at startup if pipeline.pkl is missing or version-incompatible.
This guarantees the pkl is ALWAYS trained on the same sklearn version that loads it.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CKD Prediction", page_icon="🩺", layout="centered")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pipeline.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "dataset", "chronickidneydisease.csv")

FEATURE_COLUMNS = ["rbc", "pc", "bgr", "bu", "pe", "ane", "dm", "cad"]
NUM_COLS        = ["bgr", "bu"]
CAT_COLS        = ["rbc", "pc", "pe", "ane", "dm", "cad"]

# ── Train helper ──────────────────────────────────────────────────────────────
def train_and_save():
    df = pd.read_csv(DATA_PATH)
    df.replace("?", np.nan, inplace=True)
    if "id" in df.columns:
        df.drop("id", axis=1, inplace=True)

    for col in ["dm", "cad", "classification"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    df["classification"] = df["classification"].apply(
        lambda x: 1 if "notckd" not in x and x.startswith("ckd") else 0
    )

    X = df[FEATURE_COLUMNS].copy()
    y = df["classification"].copy()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_COLS),
            ("cat", cat_pipeline, CAT_COLS),
        ],
        remainder="drop",
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")),
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass  # read-only filesystem on cloud — that's fine, model is in memory
    return model

# ── Load or train model ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Setting up model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass  # version mismatch — retrain fresh
    return train_and_save()

pipeline = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🩺 Chronic Kidney Disease Prediction")
st.markdown("Enter the patient's clinical parameters below.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Blood & Cell Indicators")
    rbc = st.selectbox("Red Blood Cells (rbc)", ["normal", "abnormal"])
    pc  = st.selectbox("Pus Cell (pc)",          ["normal", "abnormal"])
    bgr = st.number_input("Blood Glucose Random (bgr) — mgs/dl", min_value=0.0, max_value=500.0, value=120.0)
    bu  = st.number_input("Blood Urea (bu) — mgs/dl",            min_value=0.0, max_value=400.0, value=40.0)

with col2:
    st.subheader("Comorbidities & Symptoms")
    pe  = st.selectbox("Pedal Edema (pe)",              ["no", "yes"])
    ane = st.selectbox("Anemia (ane)",                  ["no", "yes"])
    dm  = st.selectbox("Diabetes Mellitus (dm)",        ["no", "yes"])
    cad = st.selectbox("Coronary Artery Disease (cad)", ["no", "yes"])

st.divider()

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_data = pd.DataFrame(
        [[rbc, pc, bgr, bu, pe, ane, dm, cad]],
        columns=FEATURE_COLUMNS,
    )
    try:
        prediction = pipeline.predict(input_data)[0]
        proba      = pipeline.predict_proba(input_data)[0]
        ckd_prob   = proba[1] * 100

        st.divider()
        if prediction == 1:
            st.error("⚠️ **CKD Detected** — Please consult a nephrologist.")
        else:
            st.success("✅ **No CKD Detected** — No signs of Chronic Kidney Disease.")

        st.markdown(f"**Confidence:** CKD `{ckd_prob:.1f}%` | No CKD `{100 - ckd_prob:.1f}%`")
        st.progress(int(ckd_prob))

    except Exception as e:
        st.error(f"Prediction error: {e}")

st.divider()
st.caption("⚠️ For educational purposes only. Not a substitute for medical advice.")
