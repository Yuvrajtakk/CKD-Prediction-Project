import streamlit as st
import pandas as pd
import pickle
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "pipeline.pkl")
pipeline = pickle.load(open(model_path, "rb"))

st.title("CKD Prediction App")

# Inputs
rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
bgr = st.number_input("Blood Glucose Random")
bu = st.number_input("Blood Urea")
pe = st.selectbox("Pedal Edema", ["yes", "no"])
ane = st.selectbox("Anemia", ["yes", "no"])
dm = st.selectbox("Diabetes Mellitus", ["yes", "no"])
cad = st.selectbox("Coronary Artery Disease", ["yes", "no"])

if st.button("Predict"):
    data = pd.DataFrame([[rbc, pc, bgr, bu, pe, ane, dm, cad]],
                        columns=['rbc','pc','bgr','bu','pe','ane','dm','cad'])

    prediction = pipeline.predict(data)

    if prediction[0] == 0:
        st.success("No CKD detected ✅")
    else:
        st.error("CKD detected ⚠️")