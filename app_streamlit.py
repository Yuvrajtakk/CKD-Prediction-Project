import streamlit as st
import pandas as pd
import pickle
import os

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "pipeline.pkl")

@st.cache_resource
def load_model():
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

pipeline = load_model()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="CKD Prediction", layout="centered")

st.title("🧠 Chronic Kidney Disease Prediction")

st.write("Fill patient details:")

# Inputs
rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
bgr = st.number_input("Blood Glucose Random", min_value=0.0)
bu = st.number_input("Blood Urea", min_value=0.0)
pe = st.selectbox("Pedal Edema", ["yes", "no"])
ane = st.selectbox("Anemia", ["yes", "no"])
dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    if pipeline is None:
        st.error("Model not loaded")
    else:
        try:
            data = pd.DataFrame(
                [[rbc, pc, bgr, bu, pe, ane, dm, cad]],
                columns=[
                    'rbc',
                    'pc',
                    'bgr',
                    'bu',
                    'pe',
                    'ane',
                    'dm',
                    'cad'
                ]
            )

            prediction = pipeline.predict(data)

            if prediction[0] == 0:
                st.success("✅ No CKD detected")
            else:
                st.error("⚠️ CKD detected. Consult a doctor.")

        except Exception as e:
            st.error(f"Prediction error: {e}")