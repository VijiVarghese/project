"""
Maternity Readmission Risk Predictor — Streamlit Dashboard
Healthcare Data Analytics Capstone Project 1
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maternity Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load / Train Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load saved model or retrain if file not found."""
    if os.path.exists("rf_readmission_model.pkl"):
        with open("rf_readmission_model.pkl", "rb") as f:
            return pickle.load(f)

    # Retrain from CSV if model file is missing
    df = pd.read_csv("maternity_master.csv")
    df = df[(df["Age"] >= 18) & (df["Age"] <= 45) & (df["LOS"] >= 2)].copy()
    df["DeliveryType_enc"]  = (df["DeliveryType"]  == "Cesarean").astype(int)
    df["Location_enc"]      = (df["Location"]      == "Rural"   ).astype(int)
    df["Complications_enc"] = (df["Complications"] == "Yes"     ).astype(int)
    df["Readmitted_enc"]    = (df["Readmitted"]    == "Yes"     ).astype(int)

    features = ["Age", "DeliveryType_enc", "LaborDuration", "Complications_enc", "LOS", "Location_enc"]
    X = df[features]
    y = df["Readmitted_enc"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏥 Maternity Readmission Risk Predictor")
st.markdown(
    "Enter patient details below to estimate the **30-day hospital readmission risk**. "
    "This tool is intended for clinical decision support — always consult a physician."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Input Panel
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Patient Demographics")
    age           = st.slider("Patient Age (years)", min_value=18, max_value=45, value=28, step=1)
    location      = st.selectbox("Location", options=["Urban", "Rural"])

with col2:
    st.subheader("Delivery Information")
    delivery_type = st.selectbox("Delivery Type", options=["Vaginal", "Cesarean"])
    labor_hrs     = st.slider("Labor Duration (hours)", min_value=1.0, max_value=16.0, value=8.0, step=0.5)

with col3:
    st.subheader("Clinical Indicators")
    complications = st.selectbox("Complications Present?", options=["No", "Yes"])
    los_days      = st.slider("Length of Stay (days)", min_value=2.0, max_value=15.0, value=7.0, step=0.5)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
delivery_enc   = 1 if delivery_type == "Cesarean" else 0
location_enc   = 1 if location       == "Rural"   else 0
complication_enc = 1 if complications == "Yes"    else 0

patient_vector = np.array([[age, delivery_enc, labor_hrs, complication_enc, los_days, location_enc]])
risk_score     = model.predict_proba(patient_vector)[0][1]
risk_pct       = round(risk_score * 100, 1)
prediction     = "High Risk" if risk_score >= 0.40 else "Low Risk"

# Display result
res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    if prediction == "High Risk":
        st.error(f"### ⚠️ {prediction}")
        st.metric("Readmission Probability", f"{risk_pct}%", delta="Above threshold")
    else:
        st.success(f"### ✅ {prediction}")
        st.metric("Readmission Probability", f"{risk_pct}%", delta="Below threshold", delta_color="off")

with res_col2:
    st.markdown("#### What this means")
    if prediction == "High Risk":
        st.warning(
            "This patient has an elevated risk of returning to hospital within 30 days. "
            "Consider scheduling a follow-up call within 48 hours of discharge, "
            "reviewing medication adherence, and assessing home support availability."
        )
    else:
        st.info(
            "This patient shows a lower readmission risk profile. "
            "Standard post-discharge instructions and a routine follow-up call in 1 week are recommended."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Risk Breakdown (mini chart)
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Risk Score Breakdown")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(7, 1.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_xlabel("Readmission Probability (%)", fontsize=10)

# Background gradient zones
ax.barh(0.5, 40,  left=0,  height=0.6, color="#2ECC71", alpha=0.3)
ax.barh(0.5, 35,  left=40, height=0.6, color="#E67E22", alpha=0.3)
ax.barh(0.5, 25,  left=75, height=0.6, color="#E74C3C", alpha=0.3)

# Marker
ax.plot(risk_pct, 0.5, 'v', color='black', markersize=12, zorder=5)
ax.text(risk_pct, 0.85, f"{risk_pct}%", ha='center', fontsize=9, fontweight='bold')

# Legend patches
ax.legend(
    handles=[
        mpatches.Patch(color="#2ECC71", alpha=0.5, label="Low (0–40%)"),
        mpatches.Patch(color="#E67E22", alpha=0.5, label="Medium (40–75%)"),
        mpatches.Patch(color="#E74C3C", alpha=0.5, label="High (75–100%)"),
    ],
    loc="upper right", fontsize=8
)
ax.spines[['top','right','left']].set_visible(False)
st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer note
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This tool is for educational and decision-support purposes only. "
    "It does not replace clinical judgment. Model trained on 490 anonymised maternity records. "
    "Ethics audit indicates potential location-based bias — Rural predictions may be less accurate."
)
