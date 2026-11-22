"""Streamlit web app for the School Fee Risk Predictor (synthetic data demo).

Run:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent

st.set_page_config(page_title="School Fee Risk Predictor", page_icon="🏫")
st.title("🏫 School Fee Risk Predictor")
st.caption("Demo built on synthetic data. Not real student information.")

model_path = HERE / "model.joblib"
if not model_path.exists():
    st.error("Model file not found. Run `python train.py` first.")
    st.stop()

model = joblib.load(model_path)

st.subheader("Family details")
col1, col2 = st.columns(2)
with col1:
    grade_band = st.selectbox("Grade band", ["LKG-UKG", "1-5", "6-8", "9-11"])
    monthly_fee = st.number_input("Monthly fee (Rs)", 500, 3000, 1300, step=50)
    income_band = st.selectbox("Income band", ["Low", "Medium", "High"])
    siblings = st.slider("Siblings in school", 0, 3, 0)
with col2:
    distance = st.slider("Distance from school (km)", 0.0, 20.0, 4.0, step=0.5)
    late_last_year = st.slider("Late payments last year", 0, 12, 1)
    replies = st.selectbox("Parent replies to messages", ["Always", "Sometimes", "Rarely"])
    concession = st.checkbox("Has fee concession")

row = pd.DataFrame(
    [
        {
            "grade_band": grade_band,
            "monthly_fee": monthly_fee,
            "income_band": income_band,
            "siblings_in_school": siblings,
            "distance_km": distance,
            "late_payments_last_year": late_last_year,
            "parent_replies_to_messages": replies,
            "fee_concession": int(concession),
        }
    ]
)

if st.button("Predict risk", type="primary"):
    risk = float(model.predict_proba(row)[0, 1])
    st.metric("Chance of paying late", f"{risk:.0%}")
    st.progress(min(max(risk, 0.0), 1.0))
    if risk >= 0.7:
        st.error("High risk: send an early reminder and offer a payment plan.")
    elif risk >= 0.4:
        st.warning("Medium risk: send a friendly reminder before the due date.")
    else:
        st.success("Low risk: no action needed.")
