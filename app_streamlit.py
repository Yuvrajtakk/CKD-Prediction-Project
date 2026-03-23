
import os
import joblib  # ✅ Must match how the model was saved (joblib, not pickle)
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🩺",
    layout="centered",
)

# ── Model Loading ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pipeline.pkl")

# These MUST match the exact order/names used in train_model.py
FEATURE_COLUMNS = ["rbc", "pc", "bgr", "bu", "pe", "ane", "dm", "cad"]


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the sklearn pipeline. Cached so it only runs once per session."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file not found at `{MODEL_PATH}`. "
            "Please run `python model/train_model.py` and push `pipeline.pkl` to the repo."
        )
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()


pipeline = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🩺 Chronic Kidney Disease Prediction")
st.markdown(
    "Enter the patient's clinical parameters below. "
    "This tool uses a Random Forest classifier trained on the UCI CKD dataset."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Blood & Cell Indicators")
    rbc = st.selectbox(
        "Red Blood Cells (rbc)",
        options=["normal", "abnormal"],
        help="Presence of normal or abnormal red blood cells in urine",
    )
    pc = st.selectbox(
        "Pus Cell (pc)",
        options=["normal", "abnormal"],
        help="Presence of normal or abnormal pus cells in urine",
    )
    bgr = st.number_input(
        "Blood Glucose Random (bgr) — mgs/dl",
        min_value=0.0,
        max_value=500.0,
        value=120.0,
        step=1.0,
    )
    bu = st.number_input(
        "Blood Urea (bu) — mgs/dl",
        min_value=0.0,
        max_value=400.0,
        value=40.0,
        step=1.0,
    )

with col2:
    st.subheader("Comorbidities & Symptoms")
    pe = st.selectbox(
        "Pedal Edema (pe)",
        options=["no", "yes"],
        help="Swelling of feet/ankles due to fluid retention",
    )
    ane = st.selectbox(
        "Anemia (ane)",
        options=["no", "yes"],
        help="Low hemoglobin / red blood cell count",
    )
    dm = st.selectbox(
        "Diabetes Mellitus (dm)",
        options=["no", "yes"],
        help="Patient has diabetes",
    )
    cad = st.selectbox(
        "Coronary Artery Disease (cad)",
        options=["no", "yes"],
        help="Patient has coronary artery disease",
    )

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    # Build dataframe with EXACT column order from training
    input_data = pd.DataFrame(
        [[rbc, pc, bgr, bu, pe, ane, dm, cad]],
        columns=FEATURE_COLUMNS,
    )

    try:
        prediction = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0]

        st.divider()
        if prediction == 1:
            st.error(
                "⚠️ **CKD Detected** — This patient likely has Chronic Kidney Disease. "
                "Please consult a nephrologist immediately."
            )
        else:
            st.success(
                "✅ **No CKD Detected** — No signs of Chronic Kidney Disease found."
            )

        # Show confidence
        ckd_prob = proba[1] * 100
        no_ckd_prob = proba[0] * 100
        st.markdown(f"**Confidence:** CKD `{ckd_prob:.1f}%` | No CKD `{no_ckd_prob:.1f}%`")
        st.progress(int(ckd_prob))

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.info(
            "This usually means the input data format doesn't match training. "
            f"Expected columns: `{FEATURE_COLUMNS}`"
        )

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This tool is for educational purposes only. "
    "It is NOT a substitute for professional medical diagnosis. "
    "Always consult a healthcare provider for medical advice."  
)
